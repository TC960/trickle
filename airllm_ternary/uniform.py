"""Asymmetric uniform n-bit quantization, in the shard/stream format.

`ternary.py` handles the {-1, 0, +1} case, which needs one scale per group.
Uniform quantization needs a scale *and* a zero point, so it cannot reuse that
format -- hence a separate module rather than a mode flag.

The arithmetic here is deliberately identical to `qat.uniform_ste`, which is
what `distill_seq.py` optimizes against. If the two ever diverge, a model would
be trained under one quantizer and served under another, and the served
behaviour would silently stop matching the measured behaviour. There is a test
asserting they agree exactly.

Storage cost per weight, at group_size=128:

    4 bits (codes) + 16 bits (bf16 scale) / 128 + 8 bits (uint8 zero) / 128
      = 4.1875 bits/weight

The zero point fits in a uint8 for any bit width <= 8, since it is an integer in
[0, 2**bits - 1].
"""

import torch
import torch.nn.functional as F

# How many quantized values fit in one uint8 byte.
VALUES_PER_BYTE = {4: 2, 8: 1}
SUPPORTED_BITS = tuple(sorted(VALUES_PER_BYTE))


def check_bits(bits: int) -> None:
    if bits not in VALUES_PER_BYTE:
        raise ValueError(
            f"unsupported bit width {bits}; expected one of {SUPPORTED_BITS}. "
            "Ternary (1.58-bit) lives in ternary.py, not here."
        )


def quantize_uniform(weight: torch.Tensor, bits: int = 4, group_size: int = 128):
    """Quantize [out, in] to grouped asymmetric uniform codes.

    Per group: map [min, max] onto the integers [0, 2**bits - 1]. The zero point
    is where the real value 0.0 lands on that integer grid, so an asymmetric
    weight distribution is represented without wasting levels -- which is the
    whole reason 2-bit uniform beats ternary at equal width.

    Returns (codes uint8 [n_groups, group_size], scales bf16 [n_groups, 1],
    zeros uint8 [n_groups, 1]). The input dim is right-padded to a multiple of
    group_size; padding is sliced off at dequant time.
    """
    check_bits(bits)
    if weight.ndim != 2:
        raise ValueError(f"expected a 2D weight, got shape {tuple(weight.shape)}")

    in_features = weight.shape[1]
    pad = (-in_features) % group_size
    if pad:
        weight = F.pad(weight, (0, pad))

    groups = weight.float().reshape(-1, group_size)
    qmax = 2 ** bits - 1
    lo = groups.min(dim=1, keepdim=True).values
    hi = groups.max(dim=1, keepdim=True).values
    # clamp_min guards a constant group, where hi == lo would divide by zero.
    scales = ((hi - lo) / qmax).clamp_min(1e-8)
    # Round the scale to its STORAGE precision before deriving anything from
    # it. Quantizing against an fp32 scale and then storing bf16 means dequant
    # uses a different scale than quantization did, which shifts every code by
    # a fraction of a step -- a systematic error, not a rounding wobble.
    scales = scales.to(torch.bfloat16).to(torch.float32)
    zeros = torch.round(-lo / scales)

    codes = torch.clamp(torch.round(groups / scales) + zeros, 0, qmax)
    return codes.to(torch.uint8), scales.to(torch.bfloat16), zeros.to(torch.uint8)


def pack_uniform(codes: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Pack uint8 codes in [0, 2**bits - 1] into a dense uint8 buffer."""
    check_bits(bits)
    per_byte = VALUES_PER_BYTE[bits]
    if per_byte == 1:
        return codes.contiguous()

    grouped = codes.reshape(*codes.shape[:-1], -1, per_byte)
    packed = grouped[..., 0].to(torch.uint8)
    for i in range(1, per_byte):
        packed = packed | (grouped[..., i].to(torch.uint8) << (bits * i))
    return packed


def unpack_uniform(packed: torch.Tensor, bits: int = 4) -> torch.Tensor:
    """Inverse of `pack_uniform`."""
    check_bits(bits)
    per_byte = VALUES_PER_BYTE[bits]
    if per_byte == 1:
        return packed

    mask = (1 << bits) - 1
    shifts = torch.tensor([bits * i for i in range(per_byte)],
                          dtype=torch.uint8, device=packed.device)
    digits = (packed.unsqueeze(-1) >> shifts) & mask
    return digits.reshape(*packed.shape[:-1], -1)


def dequantize_uniform(
    packed: torch.Tensor,
    scales: torch.Tensor,
    zeros: torch.Tensor,
    *,
    bits: int,
    group_size: int,
    out_features: int,
    in_features: int,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reconstruct a [out_features, in_features] weight from packed codes."""
    codes = unpack_uniform(packed, bits).reshape(-1, group_size)
    # Subtract the zero point in float: codes and zeros are unsigned, so doing
    # this in integer arithmetic would wrap around for codes below the zero.
    centred = codes.to(torch.float32) - zeros.to(torch.float32)
    weight = (centred * scales.to(torch.float32)).to(dtype)
    return weight.reshape(out_features, -1)[:, :in_features].contiguous()


def packed_nbytes_uniform(numel: int, group_size: int, bits: int) -> int:
    """Bytes for `numel` weights: packed codes + bf16 scales + uint8 zeros."""
    check_bits(bits)
    n_groups = -(-numel // group_size)
    return numel // VALUES_PER_BYTE[bits] + n_groups * 2 + n_groups


def bits_per_weight(group_size: int, bits: int) -> float:
    """Effective storage cost including scale and zero-point overhead."""
    check_bits(bits)
    return bits + 16 / group_size + 8 / group_size
