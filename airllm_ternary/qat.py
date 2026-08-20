"""Quantization-aware training primitives: straight-through estimation.

The problem with rounding weights to {-1, 0, +1} after training is that nothing
ever got a chance to compensate. The round is not differentiable, so the usual
fix is a straight-through estimator: the forward pass uses the quantized weight,
the backward pass pretends the quantizer was the identity function and sends the
gradient to an underlying full-precision "latent" weight.

Training then optimizes latent weights whose *quantized* form works well, rather
than optimizing weights and hoping they survive quantization.

This module provides the mechanism. `reconstruct.py` drives it one block at a
time, which is what keeps a 31B model trainable inside a consumer memory budget.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .ternary import pack


def ternary_ste(weight: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    """Quantize to ternary in the forward pass, pass gradients straight through.

    The identity `w + (w_q - w).detach()` evaluates to `w_q` numerically while
    having derivative 1 with respect to `w`. That is the whole straight-through
    trick, and it avoids needing a custom autograd.Function.
    """
    original_shape = weight.shape
    in_features = original_shape[-1]

    pad = (-in_features) % group_size
    if pad:
        weight = F.pad(weight, (0, pad))

    groups = weight.reshape(-1, group_size)
    scales = groups.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
    quantized = (groups / scales).round().clamp(-1, 1) * scales
    quantized = quantized.reshape(*original_shape[:-1], -1)

    if pad:
        quantized = quantized[..., :in_features]
        weight = weight[..., :in_features]

    return weight + (quantized - weight).detach()


class QATLinear(nn.Module):
    """Linear layer that trains a latent weight through a ternary quantizer.

    Holds a full-precision latent weight that is the thing actually optimized.
    Every forward pass quantizes it, so the loss always reflects quantized
    behaviour and the optimizer learns to place latent values where rounding
    lands favourably.
    """

    def __init__(self, in_features: int, out_features: int, *, group_size: int = 128,
                 bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        # Trained in fp32: ternary rounding is extremely sensitive to small
        # latent updates, and bf16's 8 mantissa bits lose most of them.
        self.latent_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    @classmethod
    def from_linear(cls, linear: nn.Module, group_size: int = 128) -> "QATLinear":
        """Seed a QAT layer from a trained bf16 linear."""
        module = cls(
            linear.in_features, linear.out_features,
            group_size=group_size, bias=linear.bias is not None,
        )
        with torch.no_grad():
            module.latent_weight.copy_(linear.weight.float())
            if linear.bias is not None:
                module.bias.copy_(linear.bias.float())
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = ternary_ste(self.latent_weight, self.group_size)
        return F.linear(x, weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

    @torch.no_grad()
    def export(self, pack_mode: str = "2bit"):
        """Freeze the trained latent weight into packed ternary codes + scales."""
        weight = self.latent_weight.detach()
        in_features = weight.shape[1]

        pad = (-in_features) % self.group_size
        if pad:
            weight = F.pad(weight, (0, pad))

        groups = weight.reshape(-1, self.group_size)
        scales = groups.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
        codes = (groups / scales).round().clamp(-1, 1).to(torch.int8)
        return pack(codes, pack_mode), scales.to(torch.bfloat16)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, g={self.group_size}"


@torch.no_grad()
def ternary_stats(module: QATLinear) -> dict:
    """Distribution of the learned codes, useful for spotting collapse.

    A layer that has drifted to nearly all zeros, or lost its zeros entirely,
    is a sign the learning rate or scale initialization is wrong.
    """
    weight = module.latent_weight
    groups = weight.reshape(-1, module.group_size)
    scales = groups.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
    codes = (groups / scales).round().clamp(-1, 1)

    total = codes.numel()
    return {
        "frac_neg": (codes == -1).sum().item() / total,
        "frac_zero": (codes == 0).sum().item() / total,
        "frac_pos": (codes == 1).sum().item() / total,
    }


def swap_to_qat(block: nn.Module, group_size: int = 128) -> dict:
    """Replace every nn.Linear inside a block with a QATLinear.

    Returns {module_path: QATLinear} so the caller can export trained weights
    back out under their original tensor names.
    """
    replaced = {}
    for path, module in list(block.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        qat_module = QATLinear.from_linear(module, group_size)

        parent_path, _, attr = path.rpartition(".")
        parent = block
        if parent_path:
            for part in parent_path.split("."):
                parent = getattr(parent, part)
        setattr(parent, attr, qat_module)
        replaced[path] = qat_module

    return replaced
