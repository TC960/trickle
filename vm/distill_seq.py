"""Sequential block-wise distillation -- fixes error compounding.

The teacher-forced version of this trains every block on inputs captured from
the CLEAN model. That is wrong at inference time: block i actually receives the
QUANTIZED block i-1's output, which it has never seen. Per-block error looks
tiny (0.4% RMS) but accumulates down the residual stream, and 60 layers of it
destroyed the model (perplexity 188k vs 5.19 baseline).

Sequential reconstruction fixes this by carrying two hidden-state streams:

    teacher_h  the clean model's activations
    student_h  what the quantized model actually produces

Block i is trained to map *student_h* (drifted) onto the teacher's correct
output. So each block learns to correct the drift accumulated so far, instead of
adding to it. This is the standard fix (BRECQ / sequential GPTQ lineage).

Rotary embeddings and attention masks depend on position, not on weights, so
they are captured once from the clean model and reused for both streams; only
the hidden states differ.
"""

import argparse
import copy
import gc
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from airllm_ternary.qat import (swap_to_qat, ternary_ste, ternary_stats,
                                uniform_ste)
from distill import CALIB, _SKIP_KWARGS, _move, find_blocks, block_out


def block_device(block):
    """Where this block's weights live; hidden states must be moved to match."""
    for p in block.parameters():
        return p.device
    return torch.device("cuda:0")


def fidelity(a, b):
    """Both direction AND magnitude.

    Cosine alone is scale-invariant, so a block can score 0.9999 while emitting
    systematically mis-scaled activations -- which is exactly how the
    teacher-forced run looked healthy per-block and still collapsed end to end.
    Relative L2 catches that; report both.
    """
    a, b = a.float(), b.float().to(a.device)
    return {
        "cosine": F.cosine_similarity(a.flatten(), b.flatten(), dim=0).item(),
        "rel_l2": ((a - b).norm() / a.norm().clamp_min(1e-9)).item(),
        "mag_ratio": (b.norm() / a.norm().clamp_min(1e-9)).item(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--bits", type=int, default=0,
                    help="0 = ternary; 2/3/4 = asymmetric uniform")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--n-calib", type=int, default=64,
                    help="calibration sequences; 8 was far too few")
    ap.add_argument("--max-blocks", type=int, default=None)
    ap.add_argument("--skip-first", type=int, default=0,
                    help="leave the first N blocks in bf16")
    ap.add_argument("--skip-last", type=int, default=0)
    ap.add_argument("--out", default="/ephemeral/work/out/distill_seq.jsonl")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--teacher", default=None,
                    help="deep_eval teacher stats (.pt). If given, reports flip "
                         "rate and KL, which is what the result is judged on; "
                         "perplexity alone cannot see a token-choice change.")
    ap.add_argument("--save-dir", default=None,
                    help="write the quantized model out so it can be sharded/streamed")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda"
    print(f"=== sequential distill {args.model} steps={args.steps} lr={args.lr} ===",
          flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # device_map="auto" spans both GPUs; a 62 GB model plus two activation
    # streams plus optimizer state does not fit comfortably on one 80 GB card.
    n_gpu = torch.cuda.device_count()
    # Reserve headroom per card for activations + optimizer state, and cap CPU at
    # 0 so accelerate never offloads a block to host memory (which would break
    # the manual per-block calls below with a device mismatch).
    max_mem = {i: "68GiB" for i in range(n_gpu)}
    max_mem["cpu"] = "0GiB"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="auto",
        max_memory=max_mem, low_cpu_mem_usage=True,
    ).eval()
    print(f"  device map spans {n_gpu} GPUs", flush=True)
    blocks, path = find_blocks(model)
    n_blocks = len(blocks)
    print(f"  {n_blocks} blocks at {path}", flush=True)

    def calib_batches():
        """Real text from wikitext train, chunked to full-length sequences.

        Held-out from the test split we evaluate on. Falls back to the small
        hand-written set only if the dataset is unavailable.
        """
        try:
            from datasets import load_dataset
            ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1",
                              split="train")
            text = "\n\n".join(ds["text"])
            ids = tok(text, return_tensors="pt").input_ids[0]
            n = args.n_calib
            L = args.seq_len
            # Spread the samples across the corpus rather than taking a
            # contiguous prefix, so one topic cannot dominate.
            stride = max(1, (ids.numel() - L) // n)
            out = []
            for k in range(n):
                start = k * stride
                chunk = ids[start:start + L]
                if chunk.numel() < L:
                    break
                out.append({"input_ids": chunk.unsqueeze(0).to(dev),
                            "attention_mask": torch.ones(1, L, dtype=torch.long,
                                                         device=dev)})
            print(f"  calibration: {len(out)} x {L} tokens "
                  f"({len(out)*L/1000:.0f}k total) from wikitext train",
                  flush=True)
            return out
        except Exception as exc:
            print(f"  (calib fallback, {type(exc).__name__}); using {len(CALIB)} samples",
                  flush=True)
            return [{k: v.to(dev) for k, v in
                     tok(t, return_tensors="pt", truncation=True,
                         max_length=args.seq_len, padding="max_length").items()}
                    for t in CALIB]

    batches = calib_batches()

    # Capture block-0 inputs and every block's kwargs from one clean pass.
    caps = [[] for _ in blocks]
    handles = []

    def mk(i):
        def hook(_m, a, kw):
            caps[i].append((_move(a, "cpu"),
                            {k: _move(v, "cpu") for k, v in kw.items()
                             if k not in _SKIP_KWARGS}))
        return hook

    for i, b in enumerate(blocks):
        handles.append(b.register_forward_pre_hook(mk(i), with_kwargs=True))
    with torch.no_grad():
        for batch in batches:
            model(**batch, use_cache=False)
    for h in handles:
        h.remove()
    print(f"  captured kwargs for {n_blocks} blocks", flush=True)

    # Two parallel streams. teacher_h stays clean; student_h drifts and gets
    # corrected by each block we train.
    teacher_h = [_move(caps[0][j][0][0], dev) for j in range(len(batches))]
    student_h = [t.clone() for t in teacher_h]

    lo = args.skip_first
    hi = (args.max_blocks or n_blocks) - args.skip_last
    results = []

    for i in range(n_blocks):
        block = blocks[i]
        bdev = block_device(block)
        kws = [_move(caps[i][j][1], bdev) for j in range(len(batches))]
        # Carry both streams onto this block's device before using them.
        teacher_h = [h.to(bdev) for h in teacher_h]
        student_h = [h.to(bdev) for h in student_h]

        # Teacher targets: clean input -> clean output.
        with torch.no_grad():
            targets = [block_out(block, (teacher_h[j],), kws[j]).detach()
                       for j in range(len(batches))]

        if not (lo <= i < hi):
            # Left in bf16: advance both streams through the untouched block.
            with torch.no_grad():
                for j in range(len(batches)):
                    student_h[j] = block_out(block, (student_h[j],), kws[j]).detach()
                    teacher_h[j] = targets[j]
            print(f"  block {i}: kept bf16", flush=True)
            continue

        # Fidelity of naive rounding, measured on the DRIFTED input -- the
        # honest "do nothing" baseline at this point in the network.
        with torch.no_grad():
            naive_blk = copy.deepcopy(block)
            try:
                from accelerate.hooks import remove_hook_from_module
                remove_hook_from_module(naive_blk, recurse=True)
            except Exception:
                pass
            naive_blk = naive_blk.to(bdev)
            for m in naive_blk.modules():
                if isinstance(m, nn.Linear):
                    q = (uniform_ste(m.weight.float(), args.bits, args.group_size)
                         if args.bits else
                         ternary_ste(m.weight.float(), args.group_size))
                    m.weight.copy_(q.to(m.weight.dtype))
            naive = fidelity(targets[0],
                             block_out(naive_blk, (student_h[0],), kws[0]))
            del naive_blk
            gc.collect(); torch.cuda.empty_cache()

        student = copy.deepcopy(block)
        try:
            from accelerate.hooks import remove_hook_from_module
            remove_hook_from_module(student, recurse=True)
        except Exception:
            pass
        qat = swap_to_qat(student, args.group_size, args.bits)
        student = student.to(bdev)
        student.train()
        params = [m.latent_weight for m in qat.values()]
        opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

        t0, first = time.time(), None
        # --steps 0 is a legitimate mode: pure round-to-nearest with no
        # reconstruction, which is how the 8-bit pipeline sanity check runs.
        loss = None
        for step in range(args.steps):
            j = step % len(batches)
            out = block_out(student, (student_h[j],), kws[j])
            energy = targets[j].float().pow(2).mean().clamp_min(1e-8)
            loss = F.mse_loss(out.float(), targets[j].float()) / energy
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step()
            if first is None:
                first = loss.item()

        with torch.no_grad():
            trained = fidelity(targets[0],
                               block_out(student, (student_h[0],), kws[0]))
            # Bake the trained ternary weights into the real model.
            for p_, m in qat.items():
                dst = block
                for part in p_.split("."):
                    dst = getattr(dst, part)
                q = (uniform_ste(m.latent_weight.detach(), args.bits, args.group_size)
                     if args.bits else
                     ternary_ste(m.latent_weight.detach(), args.group_size))
                dst.weight.copy_(q.to(dst.weight.dtype))
            # Advance both streams: student through the now-quantized block.
            for j in range(len(batches)):
                student_h[j] = block_out(block, (student_h[j],), kws[j]).detach()
                teacher_h[j] = targets[j]

        rec = {"block": i, "naive": naive, "trained": trained,
               "initial_loss": first,
               "final_loss": loss.item() if loss is not None else None,
               "seconds": round(time.time() - t0, 1),
               "codes": ternary_stats(next(iter(qat.values())))}
        results.append(rec)
        print(f"  block {i:>2}: naive rel_l2 {naive['rel_l2']:.4f} -> "
              f"trained {trained['rel_l2']:.4f}  "
              f"(cos {trained['cosine']:.5f}, mag {trained['mag_ratio']:.4f})  "
              f"{rec['seconds']}s", flush=True)

        del student, qat, opt, params, targets
        gc.collect(); torch.cuda.empty_cache()

    with open(args.out, "a") as h:
        for r in results:
            h.write(json.dumps({**r, "model": args.model,
                                "group_size": args.group_size,
                                "steps": args.steps}) + "\n")

    if args.save_dir:
        # The weights in `model` are already the dequantized quantized values
        # (written back per block), so saving here captures the artifact we just
        # measured -- no re-quantization, no drift between eval and export.
        print(f"\n  saving quantized model -> {args.save_dir}", flush=True)
        model.save_pretrained(args.save_dir, safe_serialization=True)
        tok.save_pretrained(args.save_dir)
        import subprocess
        sz = subprocess.run(["du", "-sh", args.save_dir], capture_output=True,
                            text=True).stdout.split()[0]
        print(f"  saved ({sz})", flush=True)

    print("\n=== evaluating end to end ===", flush=True)
    from perplexity import load_wikitext, perplexity
    ids, nb = load_wikitext(tok)

    # Behaviour first. A quantizer can hold perplexity almost exactly while
    # changing which token the model actually emits -- int8 measured +0.49%
    # perplexity against a 3.8% flip rate on this very model -- so a perplexity
    # match is not on its own evidence that the pipeline is faithful.
    behav = {}
    if args.teacher:
        from deep_eval import behaviour_delta, collect
        teacher = torch.load(args.teacher, map_location="cpu")
        student = collect(model, ids, 2048, 1024, device=dev, verbose=False)
        flip, kl = behaviour_delta(teacher, student)
        behav = {"flip_rate": flip, "kl_mean": kl}
        print(f"\n  FLIP RATE {flip*100:.3f}%   KL {kl:.6f}", flush=True)

    met = perplexity(model, ids, 2048, device=dev, n_bytes=nb)
    label = f"w{args.bits}g{args.group_size}" if args.bits else "ternary"
    tag = args.tag or (f"{args.model.split('/')[-1]}-{label}-seq"
                       f"{'-skip%d%d' % (args.skip_first, args.skip_last) if (args.skip_first or args.skip_last) else ''}")
    rec = {"tag": tag, "model": args.model, "quant": f"{label}-sequential", "bits": args.bits,
           "group_size": args.group_size, "steps": args.steps,
           "skip_first": args.skip_first, "skip_last": args.skip_last,
           **behav, **met}
    print(f"\n  {tag} PERPLEXITY {met['perplexity']:.4f}  BPB {met.get('bits_per_byte')}\n",
          flush=True)
    with open("/ephemeral/work/out/ppl.jsonl", "a") as h:
        h.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
