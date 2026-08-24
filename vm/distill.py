"""Block-wise quantization-aware distillation at 31B scale.

Full end-to-end QAT on a 31B model needs ~375 GB of optimizer state. This does
the affordable thing: treat each full-precision decoder block as a teacher for
its ternary student and train them to match, one block at a time, so peak memory
is bounded by a single block rather than the whole model.

Phase 1  run the teacher once over calibration text, capturing the exact inputs
         each block receives (including rotary embeddings and masks -- these are
         architecture-specific and reproducing them by hand is how bugs get in)
Phase 2  for each block, train a ternary copy to match the teacher's output,
         then free everything and advance

Teacher inputs are captured from the CLEAN model, so quantization error cannot
compound across depth and each block is an independent problem. Every block's
"before" (naive rounding) and "after" (trained) fidelity is recorded so the
training has to prove it did something.
"""

import argparse
import copy
import gc
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from airllm_ternary.qat import swap_to_qat, ternary_stats
from airllm_ternary.qat import ternary_ste

CALIB = [
    "The transformer architecture replaced recurrence with self-attention, letting every position attend to every other position in a single step.",
    "Quantization reduces the numerical precision of model weights. The central question is always which errors a network can absorb and which it cannot.",
    "She walked down to the harbour before dawn, when the boats were still dark shapes against the water and nothing had started moving yet.",
    "def binary_search(items, target):\n    low, high = 0, len(items) - 1\n    while low <= high:\n        mid = (low + high) // 2",
    "The mitochondrion generates most of the chemical energy needed to power a cell's biochemical reactions, stored as adenosine triphosphate.",
    "In 1687 Newton published the Principia, which set out the laws of motion and universal gravitation that dominated physics for two centuries.",
    "Economic growth in developing markets has historically correlated with infrastructure investment, though causality runs in both directions.",
    "The patient presented with intermittent chest pain radiating to the left arm, accompanied by shortness of breath on exertion.",
]

_SKIP_KWARGS = frozenset(
    {"past_key_value", "past_key_values", "use_cache", "cache_position", "layer_idx"}
)


def _move(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        moved = [_move(o, device) for o in obj]
        return tuple(moved) if isinstance(obj, tuple) else moved
    if isinstance(obj, dict):
        return {k: _move(v, device) for k, v in obj.items()}
    return obj


def find_blocks(model):
    """Locate the decoder block list, whatever the architecture calls it."""
    for path in ("model.language_model.layers", "model.layers",
                 "language_model.model.layers", "model.decoder.layers"):
        obj = model
        try:
            for part in path.split("."):
                obj = getattr(obj, part)
            if isinstance(obj, nn.ModuleList):
                return obj, path
        except AttributeError:
            continue
    raise RuntimeError("could not locate decoder blocks")


@torch.no_grad()
def capture(model, blocks, batches, store="cpu"):
    """Record the (args, kwargs) every block receives from the clean model."""
    captured = [[] for _ in blocks]
    handles = []

    def make_hook(index):
        def hook(_m, args, kwargs):
            clean = {k: v for k, v in kwargs.items() if k not in _SKIP_KWARGS}
            captured[index].append((_move(args, store), _move(clean, store)))
        return hook

    for i, block in enumerate(blocks):
        handles.append(block.register_forward_pre_hook(make_hook(i), with_kwargs=True))
    try:
        for batch in batches:
            model(**batch, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return captured


def block_out(block, args, kwargs):
    out = block(*args, **kwargs)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def naive_fidelity(block, captured, group_size, device):
    """Output cosine from plain absmean rounding, no training. The bar to beat."""
    student = copy.deepcopy(block).to(device).eval()
    for m in student.modules():
        if isinstance(m, nn.Linear):
            m.weight.copy_(ternary_ste(m.weight.float(), group_size).to(m.weight.dtype))
    cosines = []
    for args, kwargs in captured:
        a, k = _move(args, device), _move(kwargs, device)
        tgt = block_out(block, a, k)
        got = block_out(student, a, k)
        cosines.append(F.cosine_similarity(tgt.float().flatten(),
                                           got.float().flatten(), dim=0).item())
    del student
    return sum(cosines) / len(cosines)


def train_block(block, captured, *, group_size, steps, lr, device, log_every=50):
    """Train a ternary copy of one block to match its teacher's outputs."""
    block = block.to(device).eval()
    for p in block.parameters():
        p.requires_grad_(False)

    student = copy.deepcopy(block)
    qat = swap_to_qat(student, group_size)
    student = student.to(device).train()

    params = [m.latent_weight for m in qat.values()]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    # Teacher targets are fixed for this block; compute once.
    targets = []
    with torch.no_grad():
        for args, kwargs in captured:
            a, k = _move(args, device), _move(kwargs, device)
            targets.append(block_out(block, a, k).detach())

    first = None
    loss = torch.tensor(float('nan'))
    for step in range(steps):
        i = step % len(captured)
        a, k = _move(captured[i][0], device), _move(captured[i][1], device)
        tgt = targets[i]

        out = block_out(student, a, k)
        # Normalize by target energy: activation magnitude grows with depth, so a
        # fixed lr otherwise under-trains shallow blocks and oscillates on deep.
        energy = tgt.float().pow(2).mean().clamp_min(1e-8)
        loss = F.mse_loss(out.float(), tgt.float()) / energy

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if first is None:
            first = loss.item()
        if step % log_every == 0 or step == steps - 1:
            print(f"      step {step:>4} loss {loss.item():.6f} "
                  f"({loss.item()/first:.3f} of initial)", flush=True)

    with torch.no_grad():
        cosines = []
        for (args, kwargs), tgt in zip(captured, targets):
            a, k = _move(args, device), _move(kwargs, device)
            got = block_out(student, a, k)
            cosines.append(F.cosine_similarity(tgt.float().flatten(),
                                               got.float().flatten(), dim=0).item())
    # Export the trained latent weights as dequantized ternary values, keyed by
    # module path, so the caller can write them back into the real model.
    exported = {}
    with torch.no_grad():
        for path, mod in qat.items():
            exported[path] = ternary_ste(
                mod.latent_weight.detach(), group_size
            ).to(torch.bfloat16).cpu()

    stats = ternary_stats(next(iter(qat.values())))
    result = {
        "initial_loss": first,
        "final_loss": loss.item(),
        "loss_reduction": (first / max(loss.item(), 1e-12)) if first else None,
        "trained_cosine": sum(cosines) / len(cosines),
        "codes": stats,
    }
    del student, qat, opt, targets, params
    gc.collect(); torch.cuda.empty_cache()
    return result, exported


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--max-blocks", type=int, default=None)
    ap.add_argument("--out", default="distill.jsonl")
    ap.add_argument("--tag-suffix", default="")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda:0"
    print(f"=== distill {args.model} g={args.group_size} steps={args.steps} ===",
          flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map={"": 0}, low_cpu_mem_usage=True
    ).eval()
    blocks, path = find_blocks(model)
    print(f"  {len(blocks)} blocks at {path}", flush=True)

    batches = []
    for text in CALIB:
        enc = tok(text, return_tensors="pt", truncation=True,
                  max_length=args.seq_len, padding="max_length")
        batches.append({k: v.to(device) for k, v in enc.items()})

    print("  capturing teacher activations...", flush=True)
    t0 = time.time()
    captured = capture(model, blocks, batches)
    print(f"  captured in {time.time()-t0:.0f}s", flush=True)

    n = args.max_blocks or len(blocks)
    for i in range(n):
        print(f"\n  block {i}/{n-1}", flush=True)
        naive = naive_fidelity(blocks[i], captured[i], args.group_size, device)
        print(f"      PTQ baseline cosine {naive:.4f}", flush=True)
        started = time.time()
        res, exported = train_block(blocks[i], captured[i],
                                    group_size=args.group_size,
                                    steps=args.steps, lr=args.lr, device=device)

        # Write the distilled ternary weights straight into the live model, so
        # that after the final block the model IS the distilled ternary model
        # and can be evaluated end to end without reassembly.
        with torch.no_grad():
            for path, w in exported.items():
                mod = blocks[i]
                for part in path.split("."):
                    mod = getattr(mod, part)
                mod.weight.copy_(w.to(mod.weight.device, mod.weight.dtype))
        res.update(block=i, model=args.model, naive_cosine=naive,
                   gain=res["trained_cosine"] - naive,
                   seconds=round(time.time() - started, 1),
                   group_size=args.group_size, steps=args.steps)
        print(f"      trained cosine {res['trained_cosine']:.4f} "
              f"(+{res['gain']:.4f})  {res['seconds']}s", flush=True)
        with open(args.out, "a") as h:
            h.write(json.dumps(res) + "\n")
        captured[i] = None  # release this block's activations
        gc.collect()

    print("\n=== all blocks distilled; evaluating end to end ===", flush=True)

    # THE number: does 60 blocks of 0.9999 per-block fidelity survive
    # compounding? Per-block cosine cannot answer this.
    from perplexity import load_wikitext, perplexity
    ids, n_bytes = load_wikitext(tok)
    metrics = perplexity(model, ids, 2048, device=device, n_bytes=n_bytes)
    record = {"tag": f"{args.model.split(chr(47))[-1]}-ternary{args.tag_suffix or ('-distilled' if args.steps else '-naive')}",
              "model": args.model, "quant": "ternary-distilled",
              "group_size": args.group_size, "steps": args.steps,
              "blocks_distilled": n, **metrics}
    print(f"\n  {record['tag']} PERPLEXITY {metrics['perplexity']:.4f}\n", flush=True)
    with open("/ephemeral/work/out/ppl.jsonl", "a") as h:
        h.write(json.dumps(record) + "\n")

    print("=== distill complete ===", flush=True)


if __name__ == "__main__":
    main()
