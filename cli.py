"""Command line entry point.

  python cli.py build  <model_dir> <shard_dir>   quantize + shard a checkpoint
  python cli.py run    <model_dir> <shard_dir>   generate under a memory budget
  python cli.py bench  <model_dir> <shard_dir>   sweep budgets, report the curve
"""

import argparse
import json
import time
from pathlib import Path

import torch

from airllm_ternary import PrecisionPolicy, build_shards, load_manifest
from airllm_ternary.model import load_streaming_model


def _peak_rss_gb() -> float:
    import resource

    # macOS reports ru_maxrss in bytes; Linux in kilobytes.
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import sys

    return raw / 1e9 if sys.platform == "darwin" else raw / 1e6


def cmd_build(args):
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(args.model_dir)
    text_config = getattr(config, "text_config", config)
    num_layers = getattr(text_config, "num_hidden_layers", 60)

    policy = PrecisionPolicy(
        num_layers=num_layers,
        group_size=args.group_size,
        pack_mode=args.pack_mode,
        skip_first_layers=args.skip_first,
        skip_last_layers=args.skip_last,
        quantize_vision=args.quantize_vision,
    )

    print(f"quantizing {args.model_dir}")
    print(f"  {num_layers} layers, group={args.group_size}, mode={args.pack_mode}")
    print(f"  bf16 fallback: first {args.skip_first}, last {args.skip_last}")
    print(f"  vision tower: {'ternary' if args.quantize_vision else 'bf16'}\n")

    manifest = build_shards(
        args.model_dir, args.shard_dir, policy,
        measure_error=not args.no_error_check, verbose=args.verbose,
    )

    errors = [
        (name, entry["error"]["cosine"])
        for name, entry in manifest["tensors"].items()
        if "error" in entry
    ]
    if errors:
        errors.sort(key=lambda pair: pair[1])
        mean_cosine = sum(c for _, c in errors) / len(errors)
        print(f"\n  mean cosine similarity: {mean_cosine:.4f}")
        print("  worst-reconstructed tensors (candidates for bf16 fallback):")
        for name, cosine in errors[:8]:
            print(f"    {cosine:.4f}  {name}")


def cmd_run(args):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    manifest = load_manifest(args.shard_dir)
    total_gb = manifest["summary"]["total_bytes"] / 1e9

    print(f"shards: {total_gb:.2f} GB    budget: {args.budget_gb:.2f} GB")
    mode = "fully resident" if args.budget_gb >= total_gb else "streaming"
    print(f"mode: {mode}\n")

    started = time.time()
    model, manager = load_streaming_model(
        args.model_dir, args.shard_dir,
        budget_gb=args.budget_gb, device=args.device,
        prefetch=not args.no_prefetch, cache_dequant=args.cache_dequant,
    )
    print(f"model assembled in {time.time() - started:.1f}s")

    inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
    generate_started = time.time()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=args.temperature > 0,
            temperature=args.temperature if args.temperature > 0 else None,
        )
    elapsed = time.time() - generate_started

    generated = output.shape[1] - inputs["input_ids"].shape[1]
    print("\n" + "=" * 70)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print("=" * 70)
    print(f"\n{generated} tokens in {elapsed:.1f}s = {generated/elapsed:.2f} tok/s")
    print(f"peak RSS: {_peak_rss_gb():.2f} GB")
    print(f"residency: {json.dumps(manager.report(), indent=2)}")
    manager.close()


def cmd_bench(args):
    """Sweep the residency budget to expose the memory/throughput trade-off."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    manifest = load_manifest(args.shard_dir)
    total_gb = manifest["summary"]["total_bytes"] / 1e9
    largest_gb = manifest["summary"]["largest_layer_bytes"] / 1e9

    budgets = args.budgets or [
        round(largest_gb * 2, 2), round(total_gb / 4, 2),
        round(total_gb / 2, 2), round(total_gb * 1.1, 2),
    ]
    print(f"shard total {total_gb:.2f} GB, largest layer {largest_gb*1000:.0f} MB")
    print(f"sweeping budgets: {budgets}\n")

    rows = []
    for budget in budgets:
        model, manager = load_streaming_model(
            args.model_dir, args.shard_dir,
            budget_gb=budget, device=args.device,
            prefetch=not args.no_prefetch, cache_dequant=args.cache_dequant,
        )
        inputs = tokenizer(args.prompt, return_tensors="pt").to(args.device)
        started = time.time()
        with torch.inference_mode():
            output = model.generate(
                **inputs, max_new_tokens=args.max_new_tokens, do_sample=False
            )
        elapsed = time.time() - started
        generated = output.shape[1] - inputs["input_ids"].shape[1]
        report = manager.report()
        rows.append({
            "budget_gb": budget,
            "tok_s": round(generated / elapsed, 2),
            "gb_read": report["gb_read"],
            "hit_rate": round(report["hit_rate"], 3),
            "evictions": report["evictions"],
            "peak_rss_gb": round(_peak_rss_gb(), 2),
        })
        print(f"  {budget:>6.2f} GB -> {rows[-1]['tok_s']:>6.2f} tok/s  "
              f"read {report['gb_read']:>6.2f} GB  hit {report['hit_rate']:.2f}")
        manager.close()
        del model

    Path(args.out).write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="quantize and shard a checkpoint")
    build.add_argument("model_dir")
    build.add_argument("shard_dir")
    build.add_argument("--group-size", type=int, default=128)
    build.add_argument("--pack-mode", choices=["2bit", "trit5"], default="2bit")
    build.add_argument("--skip-first", type=int, default=1)
    build.add_argument("--skip-last", type=int, default=1)
    build.add_argument("--quantize-vision", action="store_true")
    build.add_argument("--no-error-check", action="store_true")
    build.add_argument("--verbose", action="store_true")
    build.set_defaults(func=cmd_build)

    common = dict(device="mps", prompt="The key insight about quantization is")
    run = sub.add_parser("run", help="generate text under a memory budget")
    run.add_argument("model_dir")
    run.add_argument("shard_dir")
    run.add_argument("--budget-gb", type=float, default=4.0)
    run.add_argument("--device", default=common["device"])
    run.add_argument("--prompt", default=common["prompt"])
    run.add_argument("--max-new-tokens", type=int, default=64)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--no-prefetch", action="store_true")
    run.add_argument("--cache-dequant", action="store_true")
    run.set_defaults(func=cmd_run)

    bench = sub.add_parser("bench", help="sweep residency budgets")
    bench.add_argument("model_dir")
    bench.add_argument("shard_dir")
    bench.add_argument("--budgets", type=float, nargs="*")
    bench.add_argument("--device", default=common["device"])
    bench.add_argument("--prompt", default=common["prompt"])
    bench.add_argument("--max-new-tokens", type=int, default=32)
    bench.add_argument("--no-prefetch", action="store_true")
    bench.add_argument("--cache-dequant", action="store_true")
    bench.add_argument("--out", default="bench.json")
    bench.set_defaults(func=cmd_bench)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
