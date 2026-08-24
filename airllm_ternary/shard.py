"""Shard builder: convert an HF checkpoint into per-layer ternary shards.

Reads the original safetensors lazily (one tensor at a time, never the whole
62 GB), applies the precision policy, and writes one file per decoder layer:

    shards/
      globals.safetensors        embeddings, final norm, vision projector
      language.000.safetensors   decoder layer 0
      ...
      language.059.safetensors   decoder layer 59
      vision.000.safetensors     vision encoder layer 0
      ...
      manifest.json              shapes, scales layout, per-shard byte counts

Quantized tensors are stored as three entries -- `<name>.packed` (uint8),
`<name>.scales` (bf16), and shape metadata in the manifest -- so the loader can
reconstruct without re-reading the original checkpoint.
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .policy import PrecisionPolicy, shard_key
from .ternary import check_group_size, pack, quantization_error, quantize_ternary
from .uniform import (
    bits_per_weight,
    dequantize_uniform,
    pack_uniform,
    quantize_uniform,
)


def _iter_source_tensors(model_dir: Path):
    """Yield (name, safetensors_handle) for every tensor, opening each file once."""
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        weight_map = json.loads(index_path.read_text())["weight_map"]
        by_file = defaultdict(list)
        for name, filename in weight_map.items():
            by_file[filename].append(name)
    else:
        by_file = {"model.safetensors": None}

    for filename, names in by_file.items():
        with safe_open(model_dir / filename, framework="pt", device="cpu") as handle:
            for name in names if names is not None else handle.keys():
                yield name, handle


def build_shards(
    model_dir,
    output_dir,
    policy: PrecisionPolicy,
    *,
    measure_error: bool = True,
    verbose: bool = True,
):
    """Quantize and shard a checkpoint. Returns the manifest dict."""
    if not policy.bits:
        check_group_size(policy.group_size, policy.pack_mode)

    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shards = defaultdict(dict)
    manifest = {
        "group_size": policy.group_size,
        "pack_mode": policy.pack_mode,
        "bits": policy.bits,
        "num_layers": policy.num_layers,
        "tensors": {},
        "shards": {},
    }

    started = time.time()
    quantized_params = 0
    kept_params = 0

    for name, handle in _iter_source_tensors(model_dir):
        tensor = handle.get_tensor(name)
        key = shard_key(name)
        entry = {"shard": key, "shape": list(tensor.shape), "dtype": "bfloat16"}

        if policy.is_quantizable(name, tensor.shape):
            if policy.bits:
                codes, scales, zeros = quantize_uniform(
                    tensor, policy.bits, policy.group_size)
                shards[key][f"{name}.packed"] = pack_uniform(codes, policy.bits)
                shards[key][f"{name}.scales"] = scales
                shards[key][f"{name}.zeros"] = zeros
                entry["format"] = "uniform"
                entry["bits"] = policy.bits
            else:
                codes, scales = quantize_ternary(tensor, policy.group_size)
                shards[key][f"{name}.packed"] = pack(codes, policy.pack_mode)
                shards[key][f"{name}.scales"] = scales
                entry["format"] = "ternary"

            entry["n_groups"] = scales.shape[0]
            quantized_params += tensor.numel()

            if measure_error:
                if policy.bits:
                    reconstructed = dequantize_uniform(
                        shards[key][f"{name}.packed"], scales, zeros,
                        bits=policy.bits,
                        group_size=policy.group_size,
                        out_features=tensor.shape[0],
                        in_features=tensor.shape[1],
                    )
                else:
                    from .ternary import dequantize

                    reconstructed = dequantize(
                        shards[key][f"{name}.packed"],
                        scales,
                        group_size=policy.group_size,
                        out_features=tensor.shape[0],
                        in_features=tensor.shape[1],
                        mode=policy.pack_mode,
                    )
                entry["error"] = quantization_error(tensor, reconstructed)
        else:
            shards[key][name] = tensor.to(torch.bfloat16)
            entry["format"] = "dense"
            kept_params += tensor.numel()

        manifest["tensors"][name] = entry
        if verbose:
            print(f"  {name:<70} {policy.describe(name, tensor.shape)}")

    # Write one file per shard. Each is self-contained so the loader can mmap
    # exactly the layer it needs without touching any other layer's bytes.
    for key, tensors in sorted(shards.items()):
        path = output_dir / f"{key}.safetensors"
        save_file(tensors, str(path))
        manifest["shards"][key] = {
            "file": path.name,
            "nbytes": path.stat().st_size,
            "num_tensors": len(tensors),
        }

    manifest["summary"] = {
        "quantized_params": quantized_params,
        "high_precision_params": kept_params,
        "total_bytes": sum(s["nbytes"] for s in manifest["shards"].values()),
        "largest_layer_bytes": max(
            s["nbytes"] for k, s in manifest["shards"].items() if k != "globals"
        ),
        "build_seconds": round(time.time() - started, 1),
        "bits_per_weight": (
            round(bits_per_weight(policy.group_size, policy.bits), 4)
            if policy.bits else None
        ),
    }

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if verbose:
        summary = manifest["summary"]
        total_gb = summary["total_bytes"] / 1e9
        peak_mb = summary["largest_layer_bytes"] / 1e6
        print(
            f"\n  quantized {quantized_params/1e9:.2f}B params, "
            f"kept {kept_params/1e9:.2f}B in bf16"
        )
        print(f"  on disk: {total_gb:.2f} GB    largest layer: {peak_mb:.0f} MB")
        print(f"  built in {summary['build_seconds']}s -> {output_dir}")

    return manifest


def load_manifest(shard_dir) -> dict:
    return json.loads((Path(shard_dir) / "manifest.json").read_text())
