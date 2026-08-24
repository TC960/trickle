"""Correctness harness: streamed output vs. unstreamed reference.

For a natively-ternary checkpoint the engine performs no arithmetic on the
weights -- it only moves packed bytes around. So the bar is exact token equality,
not a perplexity tolerance. Anything less means we have a bug.

Deliberately strict: if this fails, the fix is to find the bug, not to loosen the
comparison.
"""

import argparse
import json
import resource
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from airllm_ternary.loader import ResidencyManager
from airllm_ternary.model import load_streaming_model
from airllm_ternary.shard import load_manifest

PROMPTS = [
    "The capital of France is",
    "In a single sentence, explain why the sky appears blue:",
    "def fibonacci(n):",
    "The three laws of thermodynamics state that",
]


def peak_rss_gb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e9 if sys.platform == "darwin" else raw / 1e6


def reference_outputs(model_id, tokenizer, prompts, device, max_new_tokens):
    """Generate with stock transformers -- the ground truth."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16
    ).to(device).eval()

    results = []
    for prompt in prompts:
        ids = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            out = model.generate(
                **ids, max_new_tokens=max_new_tokens, do_sample=False
            )
        results.append(out[0].tolist())

    del model
    if device == "mps":
        torch.mps.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="microsoft/bitnet-b1.58-2B-4T")
    parser.add_argument("--shards", required=True)
    parser.add_argument("--budget-gb", type=float, default=0.75)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-new-tokens", type=int, default=24)
    parser.add_argument("--no-prefetch", action="store_true")
    parser.add_argument("--out", default="verify_streaming.json")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    manifest = load_manifest(args.shards)
    total_gb = manifest["summary"]["total_bytes"] / 1e9
    largest_mb = manifest["summary"]["largest_layer_bytes"] / 1e6

    print(f"shards {total_gb:.2f} GB, largest layer {largest_mb:.0f} MB")
    print(f"budget {args.budget_gb:.2f} GB "
          f"({'streaming' if args.budget_gb < total_gb else 'resident'})\n")

    print("=== reference (stock transformers, unstreamed) ===")
    started = time.time()
    reference = reference_outputs(
        args.model, tokenizer, PROMPTS, args.device, args.max_new_tokens
    )
    print(f"generated {len(reference)} completions in {time.time()-started:.1f}s")
    for prompt, ids in zip(PROMPTS, reference):
        print(f"  {tokenizer.decode(ids, skip_special_tokens=True)!r}")

    print("\n=== streamed (our engine) ===")
    model, manager = load_streaming_model(
        args.model, args.shards,
        budget_gb=args.budget_gb, device=args.device,
        prefetch=not args.no_prefetch,
    )

    streamed = []
    started = time.time()
    total_new = 0
    for prompt in PROMPTS:
        ids = tokenizer(prompt, return_tensors="pt").to(args.device)
        with torch.inference_mode():
            out = model.generate(
                **ids, max_new_tokens=args.max_new_tokens, do_sample=False
            )
        streamed.append(out[0].tolist())
        total_new += out.shape[1] - ids["input_ids"].shape[1]
    elapsed = time.time() - started

    for prompt, ids in zip(PROMPTS, streamed):
        print(f"  {tokenizer.decode(ids, skip_special_tokens=True)!r}")

    print("\n" + "=" * 70)
    mismatches = []
    for index, (want, got) in enumerate(zip(reference, streamed)):
        if want != got:
            first = next(
                (i for i, (a, b) in enumerate(zip(want, got)) if a != b), None
            )
            mismatches.append({"prompt_index": index, "first_diff_token": first})

    if mismatches:
        print(f"FAIL: {len(mismatches)}/{len(PROMPTS)} completions differ")
        for entry in mismatches:
            index = entry["prompt_index"]
            print(f"\n  prompt {index}: {PROMPTS[index]!r}")
            print(f"    first divergence at token {entry['first_diff_token']}")
            print(f"    reference: {tokenizer.decode(reference[index])!r}")
            print(f"    streamed:  {tokenizer.decode(streamed[index])!r}")
    else:
        print(f"PASS: all {len(PROMPTS)} completions are token-identical")

    report = manager.report()
    print("\n--- engine ---")
    print(f"  {total_new} tokens in {elapsed:.1f}s = {total_new/elapsed:.2f} tok/s")
    print(f"  peak RSS:       {peak_rss_gb():.2f} GB")
    print(f"  budget:         {report['budget_mb']:.0f} MB")
    print(f"  resident now:   {report['resident_mb']:.0f} MB")
    print(f"  bytes read:     {report['gb_read']:.2f} GB")
    print(f"  cache hit rate: {report['hit_rate']:.3f}")
    print(f"  evictions:      {report['evictions']}")
    print(f"  prefetch hits:  {report['prefetch_hits']}")

    # Record only the reproducibility-relevant settings. Absolute paths are
    # machine-specific and would leak local directory structure into a file
    # that is meant to be publishable evidence.
    safe_config = {
        key: value for key, value in vars(args).items()
        if key not in ("shards", "out")
    }

    with open(args.out, "w") as handle:
        json.dump({
            "passed": not mismatches,
            "mismatches": mismatches,
            "engine": report,
            "tok_s": round(total_new / elapsed, 3),
            # NOTE: this process also loaded the reference model for comparison,
            # so peak RSS here is NOT the streaming engine's footprint. Measure
            # that in a dedicated process (see chat.py's /stats).
            "peak_rss_gb_including_reference": round(peak_rss_gb(), 3),
            "config": safe_config,
        }, handle, indent=2)

    manager.close()
    sys.exit(1 if mismatches else 0)


if __name__ == "__main__":
    main()
