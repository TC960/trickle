"""Streaming throughput and correctness -- the measurement this project never made.

Two things, in order of importance:

1. CORRECTNESS. Run the same prompt with a budget large enough to hold every
   layer resident, then with a budget that forces eviction on every layer. The
   logits must be identical. If they are not, the streaming engine is not
   faithfully reproducing the model and no throughput number matters.

2. THROUGHPUT. Reported so it TRANSFERS to other hardware. A raw tok/s figure
   measured here is close to meaningless for the Jetson target: after the first
   pass the OS page cache holds the shards, so "reads" come from RAM, and this
   box's NVMe is far faster than an Orin Nano's anyway.

   Instead we measure the two components separately --
       compute seconds per token   (cache-warm, so ~pure compute)
       bytes read per token        (from the residency manager's counters)
   -- and then model end-to-end time for a range of storage bandwidths:
       t_token = compute + bytes / bandwidth
   That is a number you can carry to any device by plugging in its disk speed.
"""

import argparse
import json
import resource
import sys
import time

import torch

from airllm_ternary.model import load_streaming_model
from airllm_ternary.shard import load_manifest


def peak_rss_gb() -> float:
    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return raw / 1e9 if sys.platform == "darwin" else raw / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--model", required=True, help="HF id, for the tokenizer")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--budgets-gb", default="0.5,1,2,4,64")
    ap.add_argument("--new-tokens", type=int, default=32)
    ap.add_argument("--prompt", default="The key insight about quantization is")
    ap.add_argument("--out", default="/ephemeral/work/out/stream_bench.jsonl")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    manifest = load_manifest(args.shards)
    summary = manifest["summary"]
    total_gb = summary["total_bytes"] / 1e9
    largest_mb = summary["largest_layer_bytes"] / 1e6
    print(f"  shards      {total_gb:.2f} GB, largest layer {largest_mb:.0f} MB")
    print(f"  bits/weight {summary.get('bits_per_weight')}\n", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(args.prompt, return_tensors="pt").input_ids.to(args.device)

    budgets = [float(b) for b in args.budgets_gb.split(",")]
    reference_logits = None
    results = []

    for budget in sorted(budgets, reverse=True):   # widest first = the reference
        model, manager = load_streaming_model(
            args.model, args.shards, budget_gb=budget, device=args.device,
            prefetch=True,
        )
        model.eval()

        # Warm the page cache and any lazy init, so the timed pass measures
        # compute rather than first-touch overhead.
        with torch.inference_mode():
            model(ids)

        before = manager.report()
        torch.cuda.synchronize() if args.device == "cuda" else None
        started = time.time()
        with torch.inference_mode():
            out = model.generate(ids, max_new_tokens=args.new_tokens,
                                 do_sample=False)
        torch.cuda.synchronize() if args.device == "cuda" else None
        elapsed = time.time() - started
        after = manager.report()

        generated = out.shape[1] - ids.shape[1]
        gb_read = after["gb_read"] - before["gb_read"]
        rec = {
            "budget_gb": budget,
            "resident_mb": round(after["resident_mb"], 1),
            "tokens": generated,
            "seconds": round(elapsed, 3),
            "tok_per_s_cachewarm": round(generated / elapsed, 3),
            "compute_s_per_token": round(elapsed / max(generated, 1), 4),
            "gb_read_per_token": round(gb_read / max(generated, 1), 4),
            "cache_hit_rate": after["hit_rate"],
            "evictions": after["evictions"],
            "peak_rss_gb": round(peak_rss_gb(), 2),
        }

        # Correctness: greedy logits must not depend on the memory budget.
        with torch.inference_mode():
            logits = model(ids).logits.float().cpu()
        if reference_logits is None:
            reference_logits = logits
            rec["max_abs_logit_delta"] = 0.0
            rec["is_reference"] = True
        else:
            rec["max_abs_logit_delta"] = float(
                (logits - reference_logits).abs().max())
            rec["is_reference"] = False

        results.append(rec)
        print(f"  budget {budget:>5.1f} GB | resident {rec['resident_mb']:>7.1f} MB"
              f" | {rec['tok_per_s_cachewarm']:>7.3f} tok/s"
              f" | {rec['gb_read_per_token']:>6.3f} GB/tok"
              f" | hit {rec['cache_hit_rate']:.3f}"
              f" | max|dlogit| {rec['max_abs_logit_delta']:.2e}", flush=True)

        manager.close()
        del model, manager
        if args.device == "cuda":
            torch.cuda.empty_cache()

    # --- the transferable part ------------------------------------------------
    tight = min(results, key=lambda r: r["budget_gb"])
    print(f"\n  === projected end-to-end at budget {tight['budget_gb']} GB ===")
    print(f"  compute {tight['compute_s_per_token']:.4f} s/tok, "
          f"reads {tight['gb_read_per_token']:.3f} GB/tok\n")
    print(f"  {'storage':<28}{'s / token':>12}{'tok / s':>10}")
    projections = {}
    for label, bw in [("Orin Nano eMMC ~0.3 GB/s", 0.3),
                      ("SD card / USB3 ~0.5 GB/s", 0.5),
                      ("Orin NVMe Gen3 ~1.5 GB/s", 1.5),
                      ("Orin NVMe Gen4 ~3.0 GB/s", 3.0),
                      ("this box (cache-warm)", None)]:
        if bw is None:
            t = tight["compute_s_per_token"]
        else:
            t = tight["compute_s_per_token"] + tight["gb_read_per_token"] / bw
        projections[label] = round(t, 3)
        print(f"  {label:<28}{t:>12.3f}{1/t:>10.3f}")

    exact = all(r["max_abs_logit_delta"] == 0.0 for r in results)
    print(f"\n  streaming bit-exact across all budgets: "
          f"{'YES' if exact else 'NO -- INVESTIGATE'}")

    payload = {"tag": args.tag or "stream-bench", "shards": args.shards,
               "model": args.model, "total_gb": round(total_gb, 3),
               "largest_layer_mb": round(largest_mb, 1),
               "bits_per_weight": summary.get("bits_per_weight"),
               "bit_exact": exact, "budgets": results,
               "projected_s_per_token": projections}
    with open(args.out, "a") as h:
        h.write(json.dumps(payload) + "\n")
    print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
