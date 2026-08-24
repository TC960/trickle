"""QLoRA recovery: train adapters on a frozen 4-bit base to undo quantization damage.

The measured problem this targets, on Gemma 4 31B at 4-bit:

    flip rate 8.25%   KL 0.0896   GSM8K 86.0 -> 80.0

That is a FIDELITY loss -- the quantized model disagrees with its own bf16 self
on one token in twelve -- not a capability ceiling. So the loss function here is
KL against the bf16 teacher, not cross-entropy against ground truth. We are not
trying to make Gemma smarter; we are trying to make the 4-bit copy behave like
the 16-bit original, which is a strictly easier target and directly optimizes
the number we measured.

Cross-entropy on ground truth is available via --ce-weight for the case where
task recovery (GSM8K) matters more than teacher agreement, but it is off by
default: it optimizes something we did not measure and cannot attribute.

The teacher is precomputed top-K logprobs from deep_eval.py --save-teacher.
Storing full-vocab teacher logits for Gemma's 262144-token vocabulary costs
161 GB, which is why we keep top-K only -- a lesson already paid for once.
"""

import argparse
import json
import math
import time

import torch
import torch.nn.functional as F


def window_plan(n_tokens, window=2048, stride=1024):
    """Replay deep_eval.collect()'s windowing to map teacher rows to tokens.

    This is not the identity. collect() scores only the tokens each window
    newly reveals, so the flat teacher array is a concatenation of variable
    -length spans: window 0 contributes 2047 rows covering tokens 1..2047,
    window 1 contributes 1023 rows covering tokens 2049..3071, and token 2048
    is never scored at all. Assuming teacher_row == token_index -- which the
    first version of this script did -- silently trains the adapters against
    the wrong target distribution and yields a confident, wrong result.

    Returns (plan, total_rows) where each entry gives the chunk bounds, the
    offset `lo` into that chunk's logits, the row count, and the teacher offset.
    """
    plan, prev_end, off = [], 0, 0
    for begin in range(0, n_tokens, stride):
        end = min(begin + window, n_tokens)
        new = end - prev_end
        if new <= 0:
            continue
        length = end - begin
        lo = max(0, length - new)
        lp_rows = (length - 1 - lo) if lo < length - 1 else (length - lo)
        n = min(lp_rows, length - (lo + 1))
        if n > 0:
            plan.append({"begin": begin, "end": end, "lo": lo, "n": n, "off": off})
            off += n
        prev_end = end
        if end == n_tokens:
            break
    return plan, off


def kd_loss(student_logits, t_lp, t_idx, temperature=1.0):
    """KL(teacher || student) over the teacher's top-K support.

    Both distributions are renormalized over the K indices the teacher kept, so
    this is a proper KL on that support rather than a truncated sum that
    silently discards mass. Matches the convention in deep_eval.compare, so the
    training objective and the reported metric measure the same thing.
    """
    # Teacher: renormalize the kept top-K mass to sum to 1.
    t_logp = (t_lp / temperature).log_softmax(dim=-1)

    # Student: full log_softmax first (correct normalizer), then gather.
    s_logp_full = (student_logits / temperature).log_softmax(dim=-1)
    s_at = s_logp_full.gather(-1, t_idx)
    s_logp = s_at.log_softmax(dim=-1)

    return F.kl_div(s_logp, t_logp, log_target=True, reduction="batchmean")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="4-bit base: a saved dir, or an HF id to load with bnb")
    ap.add_argument("--teacher", required=True,
                    help="EVAL teacher (wikitext test) -- what flip rate is "
                         "reported against")
    ap.add_argument("--train-teacher", default=None,
                    help="TRAIN teacher (wikitext train). Required for an "
                         "honest number: the first run of this script trained "
                         "and evaluated on the same wikitext test split, so "
                         "its 3.95% flip rate was measured on training data "
                         "and is an upper bound, not a generalization "
                         "estimate. Falls back to --teacher only if you "
                         "explicitly want to reproduce that contaminated run.")
    ap.add_argument("--load-4bit", action="store_true",
                    help="quantize on load with bitsandbytes nf4")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--alpha", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.0)
    # A regex, not a suffix list. Gemma 4 is multimodal: the 27-layer vision
    # tower wraps its projections in Gemma4ClippableLinear, which peft cannot
    # adapt, and a bare "q_proj" suffix matches those too and hard-fails. The
    # language model's projections are plain nn.Linear, so scope to them --
    # which is also what we want, since the damage we measured is on text.
    ap.add_argument("--targets",
                    default=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$")
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--ce-weight", type=float, default=0.0,
                    help="add ground-truth CE; off by default, see module docstring")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--save-dir", default="/ephemeral/work/models/gemma4-w4-lora")
    ap.add_argument("--out", default="/ephemeral/work/out/qlora.jsonl")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from deep_eval import behaviour_delta, collect
    from perplexity import load_wikitext

    tok = AutoTokenizer.from_pretrained(args.model)

    load_kw = dict(dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
    if args.load_4bit:
        from transformers import BitsAndBytesConfig
        load_kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kw)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    peft_cfg = LoraConfig(
        r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
        target_modules=args.targets, bias="none",
        task_type="CAUSAL_LM")
    model = get_peft_model(model, peft_cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  LoRA r={args.rank}  targets={args.targets}")
    adapted = [n for n, _ in model.named_modules() if n.endswith("lora_A.default")]
    print(f"  adapted {len(adapted)} modules "
          f"(expect ~410 = 60 layers x 7 proj, minus 10 missing v_proj)", flush=True)
    print(f"  trainable {trainable/1e6:.1f}M / {total/1e9:.2f}B "
          f"({100*trainable/total:.3f}%)  = {trainable*2/1e6:.0f} MB bf16 adapter",
          flush=True)

    # Two teachers, two splits. Training signal comes from `train`; the flip
    # rate that gets reported comes from `test`, which the adapter never saw.
    eval_teacher = torch.load(args.teacher, map_location="cpu")
    if args.train_teacher:
        train_teacher = torch.load(args.train_teacher, map_location="cpu")
        train_split = "train"
    else:
        print("  !! WARNING: no --train-teacher; training and evaluating on "
              "the SAME split. The reported flip rate will be contaminated.",
              flush=True)
        train_teacher = eval_teacher
        train_split = "test"

    ids = load_wikitext(tok, split=train_split)[0]
    eval_ids = load_wikitext(tok, split="test")[0]
    dev = next(model.parameters()).device
    t_lp_all = train_teacher["topk_lp"].float()
    t_idx_all = train_teacher["topk_idx"].long()
    t_tgt_all = train_teacher["target"]
    n_pos = t_lp_all.shape[0]
    print(f"  train split: {train_split} ({ids.size(1)} tok)   "
          f"eval split: test ({eval_ids.size(1)} tok)", flush=True)

    # The teacher was collected at window=2048, stride=1024; the plan must use
    # the same geometry or the rows do not line up.
    plan, total = window_plan(ids.size(1), 2048, 1024)
    print(f"  teacher: {n_pos} positions x top-{t_lp_all.shape[1]}", flush=True)
    print(f"  plan   : {len(plan)} windows, {total} rows", flush=True)
    if total != n_pos:
        raise RuntimeError(f"window plan gives {total} rows but teacher has "
                           f"{n_pos}; geometry does not match")

    # Alignment assertion: the teacher stored the target token at every scored
    # position, so it must equal the token the plan says sits there. This is
    # the check that would have caught the off-by-window bug immediately.
    for w in plan[:5] + plan[-2:]:
        want = ids[0, w["begin"] + w["lo"] + 1: w["begin"] + w["lo"] + 1 + w["n"]]
        got = t_tgt_all[w["off"]: w["off"] + w["n"]]
        if not torch.equal(want, got):
            raise RuntimeError(
                f"teacher/token misalignment at window begin={w['begin']}: "
                f"expected {want[:8].tolist()} got {got[:8].tolist()}")
    print("  alignment check passed on 7 sampled windows", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.1)

    model.train()
    t0 = time.time()
    running = 0.0
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        for _ in range(args.accum):
            w = plan[torch.randint(len(plan), (1,)).item()]
            chunk = ids[:, w["begin"]:w["end"]].to(dev)
            lo, n, off = w["lo"], w["n"], w["off"]

            logits = model(chunk).logits[0]
            s_rows = logits[lo:lo + n]
            t_lp = t_lp_all[off:off + n].to(dev)
            t_idx = t_idx_all[off:off + n].to(dev)

            loss = kd_loss(s_rows, t_lp, t_idx)
            if args.ce_weight > 0:
                tgt = chunk[0, lo + 1:lo + 1 + n]
                loss = loss + args.ce_weight * F.cross_entropy(s_rows.float(), tgt)

            (loss / args.accum).backward()
            running += loss.item() / args.accum

        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); sched.step()

        if (step + 1) % 10 == 0:
            print(f"  step {step+1:>4}/{args.steps}  kd_loss {running/10:.5f}  "
                  f"lr {sched.get_last_lr()[0]:.2e}  {time.time()-t0:.0f}s",
                  flush=True)
            running = 0.0

    print("\n  saving adapters ->", args.save_dir, flush=True)
    model.save_pretrained(args.save_dir)

    # The point of the exercise: did the flip rate actually come down?
    print("\n=== behavioural eval after recovery ===", flush=True)
    model.eval()
    with torch.inference_mode():
        student = collect(model, eval_ids, 2048, 1024, device=dev, verbose=False)
    flip, kl = behaviour_delta(eval_teacher, student)
    print(f"  flip rate {flip*100:.3f}%   KL {kl:.6f}", flush=True)
    print(f"  (pre-LoRA 4-bit baseline was 8.25% / 0.089567)", flush=True)

    rec = {"tag": args.tag or f"qlora-r{args.rank}", "model": args.model,
           "arm": "treatment", "pair_id": "qlora-recovery",
           "rank": args.rank, "alpha": args.alpha, "steps": args.steps,
           "lr": args.lr, "ce_weight": args.ce_weight,
           "adapter_mb": round(trainable * 2 / 1e6, 1),
           "train_split": train_split, "eval_split": "test",
           "contaminated": train_split == "test",
           "flip_rate": flip, "kl_mean": kl}
    with open(args.out, "a") as h:
        h.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
