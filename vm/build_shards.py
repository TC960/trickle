"""Shard an HF checkpoint into per-layer streaming shards.

Generalizes build_bitnet.py, which only handled natively-ternary checkpoints.
`--bits 4` writes the asymmetric uniform format (codes + scales + zeros) that
matches what distill_seq.py trains against; `--bits 0` keeps the ternary path.

Reads the source lazily, one tensor at a time, so a 62 GB checkpoint never
lands in RAM whole.
"""

import argparse
import json
import time
from pathlib import Path

from airllm_ternary.policy import PrecisionPolicy
from airllm_ternary.shard import build_shards


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or local dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=4,
                    help="0 = ternary, 4 or 8 = asymmetric uniform")
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--num-layers", type=int, default=60)
    ap.add_argument("--skip-first", type=int, default=0)
    ap.add_argument("--skip-last", type=int, default=0,
                    help="layers kept in bf16. The sensitivity profile says the "
                         "last layers are by far the most fragile -- layer 59 "
                         "alone flips 26.7%% of tokens when ternarized.")
    ap.add_argument("--quantize-vision", action="store_true")
    ap.add_argument("--no-error", action="store_true",
                    help="skip per-tensor reconstruction error (faster)")
    args = ap.parse_args()

    src = Path(args.model)
    if not src.exists():
        from huggingface_hub import snapshot_download
        src = Path(snapshot_download(
            args.model,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]))
        print(f"  source: {src}", flush=True)

    policy = PrecisionPolicy(
        num_layers=args.num_layers,
        group_size=args.group_size,
        bits=args.bits,
        skip_first_layers=args.skip_first,
        skip_last_layers=args.skip_last,
        quantize_vision=args.quantize_vision,
    )

    started = time.time()
    manifest = build_shards(src, args.out, policy, measure_error=not args.no_error,
                            verbose=False)
    summary = manifest["summary"]

    errors = [t["error"]["rel_l2"] for t in manifest["tensors"].values()
              if "error" in t and isinstance(t["error"], dict)
              and "rel_l2" in t["error"]]
    print(f"\n  bits            {args.bits}  (g{args.group_size})")
    print(f"  bits/weight     {summary.get('bits_per_weight')}")
    print(f"  quantized       {summary['quantized_params']/1e9:.2f}B params")
    print(f"  kept bf16       {summary['high_precision_params']/1e9:.2f}B params")
    print(f"  on disk         {summary['total_bytes']/1e9:.2f} GB")
    print(f"  largest shard   {summary['largest_layer_bytes']/1e6:.0f} MB "
          f"<- this bounds streaming peak memory")
    if errors:
        errors.sort()
        print(f"  rel_l2 error    median {errors[len(errors)//2]:.4f}  "
              f"max {errors[-1]:.4f}")
    print(f"  built in        {time.time()-started:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
