"""AirLLM-style layer streaming + ternary quantization for Gemma 4.

Two ideas combined:

  ternary quantization  shrinks each decoder layer ~7.5x versus bf16, so the
                        bytes that must move per layer drop from ~958 MB to
                        ~127 MB
  layer streaming       runs the model one layer at a time inside a fixed byte
                        budget, so peak memory is set by the budget rather than
                        by the 31B parameter count

The second is what AirLLM does; the first is what makes it fast enough to be
worth doing, because AirLLM's bottleneck is per-layer I/O and ternary attacks
exactly that.
"""

from .linear import HighPrecisionLinear, TernaryLinear
from .loader import ResidencyManager, ShardStore
from .model import load_streaming_model
from .policy import PrecisionPolicy, shard_key
from .shard import build_shards, load_manifest
from .ternary import dequantize, pack, quantize_ternary, unpack

__version__ = "0.1.0"

__all__ = [
    "TernaryLinear",
    "HighPrecisionLinear",
    "ResidencyManager",
    "ShardStore",
    "load_streaming_model",
    "PrecisionPolicy",
    "shard_key",
    "build_shards",
    "load_manifest",
    "quantize_ternary",
    "dequantize",
    "pack",
    "unpack",
]
