"""One-command setup: download a ternary checkpoint and shard it for streaming.

    python -m airllm_ternary.build_bitnet

Downloads the model if absent, then splits it into one shard per decoder layer.
Nothing is re-quantized -- these checkpoints ship ternary, so this is a lossless
repackaging that lets the engine read exactly one layer's bytes at a time.
"""

import argparse
import os
from pathlib import Path

from .shard_native import build_native_shards

DEFAULT_MODEL = "microsoft/bitnet-b1.58-2B-4T"


def default_shard_dir(model_id: str) -> Path:
    cache = os.environ.get("AIRLLM_CACHE", str(Path.home() / ".cache/airllm-ternary"))
    return Path(cache) / f"shards-{model_id.split('/')[-1]}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--shards", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    from huggingface_hub import snapshot_download

    print(f"resolving {args.model} ...")
    model_dir = snapshot_download(
        args.model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"]
    )
    print(f"  {model_dir}")

    shard_dir = Path(args.shards) if args.shards else default_shard_dir(args.model)
    print(f"\nsharding into {shard_dir} ...")
    manifest = build_native_shards(model_dir, shard_dir, verbose=not args.quiet)

    summary = manifest["summary"]
    pinned = manifest["shards"]["globals"]["nbytes"] / 1e6
    print(f"""
done.
  {summary['num_layer_shards']} layer shards, largest {summary['largest_layer_bytes']/1e6:.0f} MB
  pinned globals (embeddings + norms): {pinned:.0f} MB
  total on disk: {summary['total_bytes']/1e9:.2f} GB

  python chat.py --shards {shard_dir}
""")


if __name__ == "__main__":
    main()
