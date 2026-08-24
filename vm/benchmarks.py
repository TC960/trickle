"""Downstream task benchmarks via lm-evaluation-harness.

Perplexity measures next-token prediction. It does NOT measure capability -- a
model can predict Wikipedia well and still have lost its ability to reason. These
four tasks catch damage perplexity hides:

    mmlu           knowledge across 57 subjects (multiple choice)
    hellaswag      commonsense reasoning (multiple choice)
    arc_challenge  hard science questions (multiple choice)
    gsm8k          grade-school math, generative -- the most sensitive of the
                   four, because one wrong token derails an entire chain of
                   arithmetic, so quantization damage shows up here first

    python benchmarks.py --model X --quant nf4 --limit 300
"""

import argparse
import json
import time

import torch

TASKS = ["mmlu", "hellaswag", "arc_challenge", "gsm8k"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", default="none", choices=["none", "int8", "nf4"])
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--limit", type=int, default=None,
                    help="samples per task; None = full (slow)")
    ap.add_argument("--batch-size", default="8")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default="/ephemeral/work/out/bench.jsonl")
    args = ap.parse_args()

    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tag = args.tag or f"{args.model.split('/')[-1]}-{args.quant}"
    tasks = args.tasks.split(",")
    print(f"=== bench {tag} :: {tasks} :: limit={args.limit} ===", flush=True)

    kwargs = dict(dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
    if args.quant in ("int8", "nf4"):
        from transformers import BitsAndBytesConfig
        kwargs.pop("dtype")
        kwargs["quantization_config"] = (
            BitsAndBytesConfig(load_in_8bit=True) if args.quant == "int8"
            else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.bfloat16,
                                    bnb_4bit_use_double_quant=True))

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs).eval()
    tok = AutoTokenizer.from_pretrained(args.model)
    print(f"  loaded in {time.time()-t0:.0f}s", flush=True)

    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=args.batch_size)
    started = time.time()
    res = lm_eval.simple_evaluate(model=lm, tasks=tasks, limit=args.limit,
                                  bootstrap_iters=0)

    # Keep only the headline metric per task; lm-eval returns a lot of variants.
    scores = {}
    for task, m in res["results"].items():
        for key in ("acc_norm,none", "acc,none", "exact_match,strict-match",
                    "exact_match,flexible-extract"):
            if key in m:
                scores[task] = round(float(m[key]) * 100, 2)
                break
    record = {"tag": tag, "model": args.model, "quant": args.quant,
              "limit": args.limit, "scores": scores,
              "minutes": round((time.time() - started) / 60, 1)}
    print("\n  " + json.dumps(scores, indent=2), flush=True)
    with open(args.out, "a") as h:
        h.write(json.dumps(record) + "\n")
    print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
