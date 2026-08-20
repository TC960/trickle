"""Assembly: build a runnable Gemma 4 whose projections stream from ternary shards.

The model is never constructed with real weights. It is instantiated on the
`meta` device (zero allocation), every nn.Linear is swapped for a streamable
TernaryLinear / HighPrecisionLinear, and the remaining dense tensors -- norms,
embeddings, layer scalars, the patch embedder -- are materialized once and kept.

That split is deliberate. The projections are ~99% of the parameters and are the
only thing worth streaming; the dense remainder is a few megabytes outside the
embedding table, so paging it in and out would cost I/O and save nothing.

Forward pre-hooks on each decoder layer ask the residency manager for that
layer's shard and hint the next one, which is what turns a normal transformers
forward pass into a layer-by-layer streaming one. transformers still owns
attention, RoPE, masking and the KV cache, so Gemma 4's asymmetric head dims
and per-layer scalars stay correct without us reimplementing them.
"""

import re
from pathlib import Path

import torch
import torch.nn as nn
import transformers
from accelerate import init_empty_weights
from transformers import AutoConfig

from .linear import BitNetLinear, HighPrecisionLinear, TernaryLinear
from .loader import ResidencyManager, ShardStore
from .policy import shard_key
from .shard import load_manifest

# Module paths that ARE a decoder/encoder layer, e.g. `model.language_model.layers.7`
_LAYER_RE = re.compile(r"\blayers\.(\d+)$")


def _module_by_path(root: nn.Module, path: str) -> nn.Module:
    module = root
    for part in path.split("."):
        module = getattr(module, part)
    return module


def _swap_linears(model: nn.Module, manifest: dict):
    """Replace every nn.Linear with a streamable equivalent.

    Returns {shard_key: [(module, weight_tensor_name), ...]} so the residency
    manager knows which modules a given shard feeds.
    """
    # Absent for natively-ternary manifests, which carry no quantizer settings.
    group_size = manifest.get("group_size", 128)
    pack_mode = manifest.get("pack_mode", "2bit")
    bindings = {}

    for path, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        weight_name = f"{path}.weight"
        entry = manifest["tensors"].get(weight_name)
        if entry is None:
            continue  # tensor absent from the checkpoint (e.g. tied head)

        has_bias = module.bias is not None
        if entry["format"] == "ternary":
            replacement = TernaryLinear(
                module.in_features,
                module.out_features,
                group_size=group_size,
                pack_mode=pack_mode,
                bias=has_bias,
            )
        elif entry["format"] == "bitnet":
            replacement = BitNetLinear(module.in_features, module.out_features)
        else:
            replacement = HighPrecisionLinear(
                module.in_features, module.out_features, bias=has_bias
            )

        parent_path, _, attr = path.rpartition(".")
        parent = _module_by_path(model, parent_path) if parent_path else model
        setattr(parent, attr, replacement)

        bindings.setdefault(entry["shard"], []).append((replacement, weight_name))

    return bindings


def _materialize_dense(model: nn.Module, store: ShardStore, manifest: dict, device, dtype):
    """Load every non-Linear tensor and attach it permanently.

    These are the tensors the policy kept dense and that no streamable module
    owns: embeddings, RMSNorm gains, Gemma 4's per-layer scalars, the vision
    patch embedder and its normalization constants.
    """
    linear_weights = set()
    for path, module in model.named_modules():
        if isinstance(module, (TernaryLinear, BitNetLinear, HighPrecisionLinear)):
            linear_weights.add(f"{path}.weight")
            linear_weights.add(f"{path}.bias")

    wanted = {
        name: entry
        for name, entry in manifest["tensors"].items()
        if entry["format"] == "dense" and name not in linear_weights
    }

    # Group by shard so each shard file is opened exactly once.
    by_shard = {}
    for name, entry in wanted.items():
        by_shard.setdefault(entry["shard"], []).append(name)

    loaded = 0
    for key, names in by_shard.items():
        handle = store._handle(key)
        for name in names:
            tensor = handle.get_tensor(name).to(device=device, dtype=dtype)
            parent_path, _, attr = name.rpartition(".")
            parent = _module_by_path(model, parent_path)
            # Meta-device params must be swapped out wholesale, not copied into.
            if attr in parent._parameters:
                parent._parameters[attr] = nn.Parameter(tensor, requires_grad=False)
            else:
                parent.register_buffer(attr, tensor, persistent=False)
            loaded += tensor.numel()

    return loaded


def _make_binder(modules, manifest, device, dtype, cache_dequant):
    """Build the callback that installs one shard's tensors into its modules."""

    def bind(tensors):
        if tensors is None:  # eviction
            for module, _ in modules:
                module.evict()
            return

        for module, weight_name in modules:
            entry = manifest["tensors"][weight_name]
            bias = tensors.get(weight_name.replace(".weight", ".bias"))
            if bias is not None:
                bias = bias.to(device=device, dtype=dtype)

            if entry["format"] == "ternary":
                module.load(
                    tensors[f"{weight_name}.packed"].to(device),
                    tensors[f"{weight_name}.scales"].to(device=device, dtype=dtype),
                    bias,
                    cache_dequant=cache_dequant,
                )
            elif entry["format"] == "bitnet":
                # Packed bytes stay uint8 on the device; only the scale is cast.
                module.load(
                    tensors[weight_name].to(device),
                    tensors[entry["scale_name"]].to(device=device, dtype=dtype),
                    cache_dequant=cache_dequant,
                )
            else:
                module.load(
                    tensors[weight_name].to(device=device, dtype=dtype), bias
                )

    return bind


def _attach_streaming_hooks(model: nn.Module, manager: ResidencyManager):
    """Make each decoder layer fetch its own shard before it runs."""
    layer_paths = []
    for path, module in model.named_modules():
        if _LAYER_RE.search(path):
            layer_paths.append((path, module))

    # Establish run order so each layer can hint its successor.
    ordered = sorted(
        layer_paths,
        key=lambda pair: (
            "vision" in pair[0],
            int(_LAYER_RE.search(pair[0]).group(1)),
        ),
    )
    keys = [shard_key(f"{path}.x") for path, _ in ordered]

    for index, (path, module) in enumerate(ordered):
        current = keys[index]
        following = keys[index + 1] if index + 1 < len(keys) else None

        def pre_hook(_mod, _args, _key=current, _next=following):
            manager.ensure(_key)
            if _next is not None:
                manager.hint(_next)  # overlap next load with this layer's compute

        module.register_forward_pre_hook(pre_hook)

    return ordered


def load_streaming_model(
    model_dir,
    shard_dir,
    *,
    budget_gb: float = 4.0,
    device: str = "mps",
    dtype: torch.dtype = torch.bfloat16,
    prefetch: bool = True,
    cache_dequant: bool = False,
):
    """Build a Gemma 4 that runs from ternary shards inside `budget_gb`.

    Set `budget_gb` above the shard total to keep everything resident after the
    first pass; set it near one layer's size for minimum-footprint streaming.
    """
    manifest = load_manifest(shard_dir)
    config = AutoConfig.from_pretrained(model_dir)

    # Resolve the concrete class from the checkpoint rather than going through an
    # Auto* mapping: Gemma4ForConditionalGeneration is multimodal and is not
    # registered under AutoModelForCausalLM.
    model_class = getattr(transformers, config.architectures[0])
    with init_empty_weights():
        model = model_class(config)
    model.eval()

    # Direct instantiation skips generation_config.json, which `from_pretrained`
    # would have loaded. Without it there are no stop tokens, so generation runs
    # to max_new_tokens and the model cheerfully role-plays both sides of a
    # conversation past its own end-of-turn marker.
    try:
        model.generation_config = transformers.GenerationConfig.from_pretrained(
            model_dir
        )
    except OSError:
        pass  # no generation_config.json published; keep the default

    bindings = _swap_linears(model, manifest)

    store = ShardStore(shard_dir, manifest, device="cpu")
    # The globals shard holds the tied embedding table; it is needed at both the
    # start and end of every token, so pinning it avoids reloading 2.8 GB twice.
    manager = ResidencyManager(
        store,
        budget_bytes=int(budget_gb * 1e9),
        prefetch=prefetch,
        pinned=("globals",),
    )

    _materialize_dense(model, store, manifest, device, dtype)

    # Re-establish weight tying. Replacing `embed_tokens.weight` with a fresh
    # Parameter silently breaks the tie to `lm_head`, leaving the output
    # projection on the meta device -- which yields garbage logits rather than an
    # error. Models with `tie_word_embeddings` have no lm_head tensor in the
    # checkpoint, so nothing else would ever populate it.
    if getattr(config, "tie_word_embeddings", False):
        model.tie_weights()
        head = model.get_output_embeddings()
        if head is not None and head.weight.is_meta:
            raise RuntimeError(
                "lm_head is still on the meta device after tie_weights(); "
                "the output projection would produce garbage"
            )

    for key, modules in bindings.items():
        manager.register_binder(
            key, _make_binder(modules, manifest, device, dtype, cache_dequant)
        )

    # Non-layer Linears (the vision projector) have no decoder hook to trigger
    # them, so bind their shard once up front and let it stay pinned.
    if "globals" in bindings:
        manager.ensure("globals")

    _attach_streaming_hooks(model, manager)

    model._residency = manager
    model._manifest = manifest
    return model, manager
