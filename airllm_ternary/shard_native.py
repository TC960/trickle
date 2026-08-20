"""Re-shard an already-ternary checkpoint into per-layer streaming shards.

For natively-ternary models (BitNet, Falcon-E) there is nothing to quantize --
the weights arrived as packed ternary. This step is pure repackaging: split one
big safetensors file into one file per decoder layer, copying the packed bytes
through untouched.

That makes the correctness bar unusually sharp. Because no arithmetic happens,
streamed output must be **bit-identical** to the unstreamed reference, not merely
close. Any difference at all is a bug in the engine.
"""

import json
import time
from collections import defaultdict
from pathlib import Path

import torch
import transformers
from accelerate import init_empty_weights
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig

from .bitnet_format import packed_row_dim
from .policy import shard_key

SCALE_SUFFIX = "_scale"


def linear_shapes(model_dir) -> dict:
    """Authoritative {tensor_name: (out_features, in_features)} from the config.

    Packed shapes are ambiguous -- ceil(n/4) maps four values of n to the same
    row count -- so we take the true dimensions from a meta-device instantiation
    rather than inferring them from the file.
    """
    config = AutoConfig.from_pretrained(model_dir)
    model_class = getattr(transformers, config.architectures[0])
    with init_empty_weights():
        model = model_class(config)

    shapes = {}
    for path, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            shapes[f"{path}.weight"] = (module.out_features, module.in_features)
    return shapes


def _iter_tensors(model_dir: Path):
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


def build_native_shards(model_dir, output_dir, *, verbose: bool = True) -> dict:
    """Split a packed-ternary checkpoint into per-layer shards, losslessly."""
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shapes = linear_shapes(model_dir)
    shards = defaultdict(dict)
    manifest = {
        "format": "bitnet",
        "source": str(model_dir),
        "tensors": {},
        "shards": {},
    }

    started = time.time()
    packed_params = 0
    dense_params = 0

    for name, handle in _iter_tensors(model_dir):
        # Scales ride along with their weight; they are not standalone entries.
        if name.endswith(SCALE_SUFFIX):
            shards[shard_key(name)][name] = handle.get_tensor(name)
            continue

        tensor = handle.get_tensor(name)
        key = shard_key(name)
        shards[key][name] = tensor

        entry = {"shard": key, "shape": list(tensor.shape)}
        if tensor.dtype == torch.uint8 and name in shapes:
            out_features, in_features = shapes[name]
            expected_rows = packed_row_dim(out_features)
            if tensor.shape[0] != expected_rows:
                raise ValueError(
                    f"{name}: packed rows {tensor.shape[0]} != expected "
                    f"{expected_rows} for out_features={out_features}"
                )
            entry.update(
                format="bitnet",
                out_features=out_features,
                in_features=in_features,
                scale_name=f"{name}{SCALE_SUFFIX}",
            )
            packed_params += out_features * in_features
        else:
            entry["format"] = "dense"
            entry["dtype"] = str(tensor.dtype).replace("torch.", "")
            dense_params += tensor.numel()

        manifest["tensors"][name] = entry
        if verbose and entry["format"] == "bitnet":
            print(f"  {name:<56} packed {tuple(tensor.shape)} "
                  f"-> {entry['out_features']}x{entry['in_features']}")

    for key, tensors in sorted(shards.items()):
        path = output_dir / f"{key}.safetensors"
        save_file(tensors, str(path))
        manifest["shards"][key] = {
            "file": path.name,
            "nbytes": path.stat().st_size,
            "num_tensors": len(tensors),
        }

    layer_shards = {k: v for k, v in manifest["shards"].items() if k != "globals"}
    manifest["summary"] = {
        "packed_params": packed_params,
        "dense_params": dense_params,
        "total_bytes": sum(s["nbytes"] for s in manifest["shards"].values()),
        "largest_layer_bytes": max(s["nbytes"] for s in layer_shards.values()),
        "num_layer_shards": len(layer_shards),
        "build_seconds": round(time.time() - started, 1),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    if verbose:
        summary = manifest["summary"]
        print(f"\n  {summary['packed_params']/1e9:.2f}B packed params, "
              f"{summary['dense_params']/1e9:.2f}B dense")
        print(f"  {summary['num_layer_shards']} layer shards, "
              f"largest {summary['largest_layer_bytes']/1e6:.0f} MB")
        print(f"  total {summary['total_bytes']/1e9:.2f} GB in "
              f"{summary['build_seconds']}s -> {output_dir}")

    return manifest
