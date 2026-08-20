"""Ternary weight quantization: BitNet b1.58 absmean rule, grouped and bit-packed.

Weights collapse to {-1, 0, +1} with one bf16 scale per group of `group_size`
values taken along the input dimension. Two packing modes are supported:

  "2bit"  - 4 values per byte via bit shifts. 2.00 bits/weight. Fast dequant.
  "trit5" - 5 base-3 trits per byte (3^5 = 243 < 256). 1.60 bits/weight, which
            is the honest "1.58-bit" figure; ~20% smaller but slower to unpack
            because it needs integer div/mod instead of shifts.

Group size must divide evenly into the packing stride: a multiple of 4 for
"2bit", a multiple of 5 for "trit5".
"""

import torch
import torch.nn.functional as F

PACK_STRIDE = {"2bit": 4, "trit5": 5}

# Positional weights for base-3 packing: trit i contributes value * 3**i.
_POW3 = (1, 3, 9, 27, 81)


def check_group_size(group_size: int, mode: str) -> None:
    """Raise if `group_size` is incompatible with the packing mode's stride."""
    if mode not in PACK_STRIDE:
        raise ValueError(f"unknown pack mode {mode!r}, expected one of {tuple(PACK_STRIDE)}")
    stride = PACK_STRIDE[mode]
    if group_size % stride:
        raise ValueError(
            f"group_size={group_size} must be a multiple of {stride} for mode {mode!r} "
            f"(try {group_size // stride * stride} or {(group_size // stride + 1) * stride})"
        )


def quantize_ternary(weight: torch.Tensor, group_size: int = 128):
    """Quantize a [out_features, in_features] weight to grouped ternary codes.

    Uses the BitNet b1.58 absmean rule: divide each group by the mean of its
    absolute values, round to nearest, clamp to {-1, 0, +1}. The scale is a
    plain mean rather than a max, which is what keeps zeros plentiful and the
    reconstruction unbiased.

    Returns (codes int8 [n_groups, group_size], scales bf16 [n_groups, 1]).
    The input dimension is right-padded with zeros to a multiple of group_size;
    padded lanes quantize to 0 and are sliced off again at dequant time.
    """
    if weight.ndim != 2:
        raise ValueError(f"expected a 2D weight, got shape {tuple(weight.shape)}")

    in_features = weight.shape[1]
    pad = (-in_features) % group_size
    if pad:
        weight = F.pad(weight, (0, pad))

    groups = weight.float().reshape(-1, group_size)
    # clamp_min guards against an all-zero group producing a divide-by-zero.
    scales = groups.abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
    codes = (groups / scales).round_().clamp_(-1, 1).to(torch.int8)
    return codes, scales.to(torch.bfloat16)


def pack(codes: torch.Tensor, mode: str = "2bit") -> torch.Tensor:
    """Pack int8 codes in {-1, 0, +1} into a dense uint8 buffer."""
    stride = PACK_STRIDE[mode]
    # Shift to unsigned {0, 1, 2} so the values fit in 2 bits / one base-3 digit.
    shifted = (codes + 1).reshape(*codes.shape[:-1], -1, stride)

    if mode == "2bit":
        u = shifted.to(torch.uint8)
        packed = u[..., 0] | (u[..., 1] << 2) | (u[..., 2] << 4) | (u[..., 3] << 6)
        return packed

    # trit5: accumulate in int16 to avoid overflow, max value is 2*121 = 242.
    u = shifted.to(torch.int16)
    packed = sum(u[..., i] * _POW3[i] for i in range(stride))
    return packed.to(torch.uint8)


def unpack(packed: torch.Tensor, mode: str = "2bit") -> torch.Tensor:
    """Inverse of `pack`. Returns int8 codes in {-1, 0, +1}."""
    if mode == "2bit":
        shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=packed.device)
        digits = (packed.unsqueeze(-1) >> shifts) & 3
    else:
        x = packed.to(torch.int16).unsqueeze(-1)
        divisors = torch.tensor(_POW3, dtype=torch.int16, device=packed.device)
        digits = torch.div(x, divisors, rounding_mode="floor") % 3

    codes = digits.reshape(*packed.shape[:-1], -1).to(torch.int8)
    return codes - 1


def dequantize(
    packed: torch.Tensor,
    scales: torch.Tensor,
    *,
    group_size: int,
    out_features: int,
    in_features: int,
    mode: str = "2bit",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reconstruct a [out_features, in_features] weight from packed ternary codes."""
    codes = unpack(packed, mode).reshape(-1, group_size)
    weight = codes.to(dtype) * scales.to(dtype)
    # Undo the group reshape, then drop the right-padding added at quant time.
    return weight.reshape(out_features, -1)[:, :in_features].contiguous()


def packed_nbytes(numel: int, group_size: int, mode: str) -> int:
    """Bytes needed to store `numel` weights: packed codes plus bf16 scales."""
    stride = PACK_STRIDE[mode]
    return numel // stride + (numel // group_size) * 2


def quantization_error(original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
    """Relative error metrics for a single tensor, used by the sensitivity sweep."""
    original = original.float()
    reconstructed = reconstructed.float()
    residual = original - reconstructed
    denom = original.norm().clamp_min(1e-12)

    # Cosine similarity is the metric that tracks downstream quality best;
    # relative Frobenius error is easier to reason about across tensor sizes.
    return {
        "rel_fro": (residual.norm() / denom).item(),
        "cosine": F.cosine_similarity(
            original.flatten(), reconstructed.flatten(), dim=0
        ).item(),
        "sparsity": (reconstructed == 0).float().mean().item(),
    }
