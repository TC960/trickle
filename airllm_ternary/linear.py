"""TernaryLinear: an nn.Linear replacement backed by packed ternary codes.

There is no ternary matmul kernel on MPS, so this dequantizes to bf16 and calls
the normal F.linear. The win is bytes on disk and bytes resident, not FLOPs --
compute is identical to bf16 once the weight is materialized.

Weight buffers start empty. The residency manager fills them when the owning
layer is about to run and clears them afterwards, which is what allows a 31B
model to execute inside a budget far smaller than its own size.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ternary import dequantize
from .uniform import dequantize_uniform


class TernaryLinear(nn.Module):
    """Linear layer whose weight lives as 2-bit codes plus per-group scales."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        group_size: int = 128,
        pack_mode: str = "2bit",
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.pack_mode = pack_mode
        self.compute_dtype = dtype

        # Populated on load, dropped on evict. Registered as buffers so that
        # .to(device) and state_dict traversal behave normally when they exist.
        self.register_buffer("packed", None, persistent=False)
        self.register_buffer("scales", None, persistent=False)
        self.register_buffer("bias", None, persistent=False)

        # Optional materialized bf16 weight. Set only when the residency manager
        # has spare budget: trades memory for skipping dequant on every token.
        self._weight_cache = None
        self._has_bias = bias

    @property
    def is_loaded(self) -> bool:
        return self.packed is not None

    def load(self, packed, scales, bias=None, *, cache_dequant: bool = False):
        """Attach weight data. Called by the residency manager."""
        self.packed = packed
        self.scales = scales
        self.bias = bias
        self._weight_cache = self.dequantized() if cache_dequant else None

    def evict(self):
        """Release all weight data, returning this layer to zero footprint."""
        self.packed = None
        self.scales = None
        self.bias = None
        self._weight_cache = None

    def dequantized(self) -> torch.Tensor:
        """Reconstruct the bf16 weight matrix from packed codes."""
        return dequantize(
            self.packed,
            self.scales,
            group_size=self.group_size,
            out_features=self.out_features,
            in_features=self.in_features,
            mode=self.pack_mode,
            dtype=self.compute_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._weight_cache is not None:
            weight = self._weight_cache
        elif self.packed is not None:
            weight = self.dequantized()
        else:
            raise RuntimeError(
                f"TernaryLinear[{self.in_features}->{self.out_features}] used while "
                "evicted; the residency manager should have loaded it first"
            )
        return F.linear(x, weight.to(x.dtype), self.bias)

    def nbytes(self) -> int:
        """Resident byte count, for budget accounting."""
        total = 0
        for tensor in (self.packed, self.scales, self.bias, self._weight_cache):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
        return total

    def extra_repr(self) -> str:
        state = "loaded" if self.is_loaded else "evicted"
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"g={self.group_size}, mode={self.pack_mode}, {state}"
        )


class UniformLinear(nn.Module):
    """Linear whose weight lives as packed n-bit codes, per-group scales AND
    per-group zero points.

    Separate from TernaryLinear rather than a flag on it, because the zero point
    is a third tensor that has to be streamed, evicted and byte-counted
    alongside the other two. Folding that into TernaryLinear would put a
    perpetually-None buffer on the ternary path.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bits: int = 4,
        group_size: int = 128,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.compute_dtype = dtype

        self.register_buffer("packed", None, persistent=False)
        self.register_buffer("scales", None, persistent=False)
        self.register_buffer("zeros", None, persistent=False)
        self.register_buffer("bias", None, persistent=False)

        self._weight_cache = None
        self._has_bias = bias

    @property
    def is_loaded(self) -> bool:
        return self.packed is not None

    def load(self, packed, scales, zeros, bias=None, *, cache_dequant: bool = False):
        self.packed = packed
        self.scales = scales
        self.zeros = zeros
        self.bias = bias
        self._weight_cache = self.dequantized() if cache_dequant else None

    def evict(self):
        self.packed = None
        self.scales = None
        self.zeros = None
        self.bias = None
        self._weight_cache = None

    def dequantized(self) -> torch.Tensor:
        return dequantize_uniform(
            self.packed,
            self.scales,
            self.zeros,
            bits=self.bits,
            group_size=self.group_size,
            out_features=self.out_features,
            in_features=self.in_features,
            dtype=self.compute_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._weight_cache is not None:
            weight = self._weight_cache
        elif self.packed is not None:
            weight = self.dequantized()
        else:
            raise RuntimeError(
                f"UniformLinear[{self.in_features}->{self.out_features}] used "
                "while evicted; the residency manager should have loaded it"
            )
        return F.linear(x, weight.to(x.dtype), self.bias)

    def nbytes(self) -> int:
        total = 0
        for t in (self.packed, self.scales, self.zeros, self.bias,
                  self._weight_cache):
            if t is not None:
                total += t.numel() * t.element_size()
        return total

    def extra_repr(self) -> str:
        state = "loaded" if self.is_loaded else "evicted"
        return (f"in={self.in_features}, out={self.out_features}, "
                f"w{self.bits} g={self.group_size}, {state}")


class BitNetLinear(nn.Module):
    """Linear backed by the HuggingFace 'bitnet' packed layout.

    Used for natively-ternary checkpoints, where the packed bytes come straight
    from the published model. Unlike TernaryLinear these weights were produced
    by the model's own training, so nothing here changes their values -- this is
    a reader, not a quantizer.
    """

    def __init__(self, in_features: int, out_features: int,
                 *, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.compute_dtype = dtype

        self.register_buffer("packed", None, persistent=False)
        self.register_buffer("scale", None, persistent=False)
        self._weight_cache = None

    @property
    def is_loaded(self) -> bool:
        return self.packed is not None

    def load(self, packed, scale, bias=None, *, cache_dequant: bool = False):
        self.packed = packed
        self.scale = scale
        self._weight_cache = self.dequantized() if cache_dequant else None

    def evict(self):
        self.packed = None
        self.scale = None
        self._weight_cache = None

    def dequantized(self) -> torch.Tensor:
        """Unpacked ternary codes, deliberately UNSCALED.

        transformers keeps `weight` as raw {-1,0,+1} and applies `weight_scale`
        to the layer's output instead. Float arithmetic is not associative, so we
        replicate that order rather than folding the scale into the weight.
        """
        from .bitnet_format import unpack_bitnet

        return unpack_bitnet(self.packed, self.out_features).to(self.compute_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        from .bitnet_format import act_quant

        if self._weight_cache is not None:
            weight = self._weight_cache
        elif self.packed is not None:
            weight = self.dequantized()
        else:
            raise RuntimeError(
                f"BitNetLinear[{self.in_features}->{self.out_features}] used "
                "while evicted; the residency manager should have loaded it"
            )

        # Order matters for bit-exactness: quantize activations, matmul against
        # unscaled ternary codes, then scale the output.
        x = act_quant(x)
        output = F.linear(x, weight.to(x.dtype))
        return output * self.scale.to(output.dtype)

    def nbytes(self) -> int:
        total = 0
        for tensor in (self.packed, self.scale, self._weight_cache):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
        return total

    def extra_repr(self) -> str:
        state = "loaded" if self.is_loaded else "evicted"
        return f"in={self.in_features}, out={self.out_features}, bitnet, {state}"


class HighPrecisionLinear(nn.Module):
    """Streamable bf16 linear, for tensors the policy excluded from ternary.

    Shares the load/evict contract with TernaryLinear so the residency manager
    can treat both uniformly.
    """

    def __init__(self, in_features: int, out_features: int, *, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", None, persistent=False)
        self.register_buffer("bias", None, persistent=False)
        self._has_bias = bias

    @property
    def is_loaded(self) -> bool:
        return self.weight is not None

    def load(self, weight, bias=None, *, cache_dequant: bool = False):
        self.weight = weight
        self.bias = bias

    def evict(self):
        self.weight = None
        self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight is None:
            raise RuntimeError(
                f"HighPrecisionLinear[{self.in_features}->{self.out_features}] "
                "used while evicted"
            )
        return F.linear(x, self.weight.to(x.dtype), self.bias)

    def nbytes(self) -> int:
        total = 0
        for tensor in (self.weight, self.bias):
            if tensor is not None:
                total += tensor.numel() * tensor.element_size()
        return total
