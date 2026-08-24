"""Precision policy: which tensors get ternarized and which stay bf16.

Post-training ternarization is destructive, so a handful of tensors are always
excluded. The reasoning per exclusion:

  embed_tokens   262144 x 5376 = 1.41B params, and `tie_word_embeddings` means
                 the same matrix is also the output head. Ternarizing it damages
                 both the input representation and every logit. Big win in bytes,
                 unacceptable cost in quality -- kept bf16.
  norms          RMSNorm gains, q_norm/k_norm. Kilobytes each; nothing to gain.
  layer_scalar   Gemma 4's per-layer scalar. One value. Ternarizing it is absurd.
  vision tower   ~430M params across 27 layers at hidden 1152. Small share of the
                 model, and vision encoders are far more sensitive to weight noise
                 than decoder MLPs. Kept bf16 by default; flip with `quantize_vision`.
  first/last N   The layers nearest the embedding and the logits carry the most
                 outlier-heavy activations. Configurable bf16 fallback.
"""

import re
from dataclasses import dataclass, field

# Tensors matching any of these never get quantized, regardless of config.
ALWAYS_HIGH_PRECISION = (
    r"embed_tokens",
    r"embed_vision",
    r"\.norm\.weight$",
    r"layernorm",
    r"_norm\.weight$",
    r"layer_scalar",
    r"patch_embedder",
    r"position_embedding_table",
    r"std_bias",
    r"std_scale",
    r"\.bias$",
)

# Only these projections inside a decoder layer are worth ternarizing; together
# they are ~99% of the layer's parameters.
QUANTIZABLE_PROJECTIONS = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)

# Matches the layer index in any of the naming schemes we encounter:
# `model.layers.N.`, `model.language_model.layers.N.`,
# `model.vision_tower.encoder.layers.N.`
_LAYER_RE = re.compile(r"\blayers\.(\d+)\.")


def _layer_index(name: str):
    """Layer index for a tensor, or None if it lives outside the layer stack."""
    match = _LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def _is_vision(name: str) -> bool:
    return "vision" in name


@dataclass
class PrecisionPolicy:
    """Decides the storage format for each tensor in the checkpoint."""

    num_layers: int = 60
    group_size: int = 128
    pack_mode: str = "2bit"

    # 0 = ternary (the pack_mode above applies). >= 2 = asymmetric uniform at
    # that bit width, which needs a zero point per group and so uses its own
    # storage format -- see uniform.py.
    bits: int = 0

    # Decoder layers at the very start / end kept in bf16.
    skip_first_layers: int = 1
    skip_last_layers: int = 1

    quantize_vision: bool = False

    # Explicit escape hatch: substrings forced to bf16 by the sensitivity sweep.
    force_high_precision: tuple = field(default_factory=tuple)

    def _always_high(self, name: str) -> bool:
        return any(re.search(p, name) for p in ALWAYS_HIGH_PRECISION)

    def is_quantizable(self, name: str, shape) -> bool:
        """True if this tensor should be stored as packed ternary codes."""
        if len(shape) != 2:
            return False  # scalars, norms, position tables
        if self._always_high(name):
            return False
        if any(s in name for s in self.force_high_precision):
            return False
        if not any(p in name for p in QUANTIZABLE_PROJECTIONS):
            return False

        index = _layer_index(name)
        if index is None:
            return False
        if _is_vision(name):
            return self.quantize_vision

        in_head = index < self.skip_first_layers
        in_tail = index >= self.num_layers - self.skip_last_layers
        return not (in_head or in_tail)

    def describe(self, name: str, shape) -> str:
        """Human-readable reason for the decision, for `--explain` output."""
        if self.is_quantizable(name, shape):
            if self.bits:
                return f"uniform w{self.bits} g{self.group_size}"
            return f"ternary g{self.group_size}/{self.pack_mode}"
        if len(shape) != 2:
            return "bf16 (not a matrix)"
        if self._always_high(name):
            return "bf16 (always high precision)"
        if any(s in name for s in self.force_high_precision):
            return "bf16 (forced by policy)"
        if _is_vision(name):
            return "bf16 (vision tower)"
        index = _layer_index(name)
        if index is not None:
            if index < self.skip_first_layers:
                return f"bf16 (first {self.skip_first_layers} layers)"
            if index >= self.num_layers - self.skip_last_layers:
                return f"bf16 (last {self.skip_last_layers} layers)"
        return "bf16 (not a target projection)"


def shard_key(name: str) -> str:
    """Map a tensor name to the shard that will hold it.

    One shard per decoder layer is what makes layer-by-layer streaming possible:
    the loader can bring in exactly the bytes needed for the layer about to run.
    """
    index = _layer_index(name)
    if index is None:
        # Embeddings, final norm, vision projector: loaded once and pinned.
        return "globals"
    prefix = "vision" if _is_vision(name) else "language"
    return f"{prefix}.{index:03d}"
