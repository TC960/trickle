"""Round-trip and agreement tests for uniform n-bit quantization.

The critical one is `test_matches_qat_uniform_ste`. `distill_seq.py` trains
against `qat.uniform_ste`; the streaming engine serves via
`uniform.dequantize_uniform`. If those two disagree even slightly, the model
is trained under one quantizer and served under another, and every measured
number stops describing the thing that actually runs.
"""

import torch

from airllm_ternary.qat import uniform_ste
from airllm_ternary.uniform import (
    SUPPORTED_BITS,
    bits_per_weight,
    dequantize_uniform,
    pack_uniform,
    quantize_uniform,
    unpack_uniform,
)


def test_pack_unpack_roundtrip_is_exact():
    for bits in SUPPORTED_BITS:
        qmax = 2 ** bits - 1
        codes = torch.randint(0, qmax + 1, (64, 128), dtype=torch.uint8)
        packed = pack_uniform(codes, bits)
        assert torch.equal(unpack_uniform(packed, bits), codes), f"{bits}-bit"


def test_packed_size_is_what_we_claim():
    codes = torch.zeros(64, 128, dtype=torch.uint8)
    assert pack_uniform(codes, 4).numel() == 64 * 64      # 2 values per byte
    assert pack_uniform(codes, 8).numel() == 64 * 128     # 1 value per byte


def test_quantize_dequantize_recovers_shape_and_range():
    torch.manual_seed(0)
    weight = torch.randn(96, 300, dtype=torch.bfloat16)
    for bits in SUPPORTED_BITS:
        codes, scales, zeros = quantize_uniform(weight, bits, group_size=128)
        packed = pack_uniform(codes, bits)
        out = dequantize_uniform(
            packed, scales, zeros, bits=bits, group_size=128,
            out_features=96, in_features=300,
        )
        assert out.shape == weight.shape
        err = (out.float() - weight.float()).norm() / weight.float().norm()
        # 4-bit on gaussian data lands well under 10% relative error.
        assert err < 0.12, f"{bits}-bit relative error {err:.4f}"


def test_more_bits_is_never_worse():
    torch.manual_seed(0)
    weight = torch.randn(64, 256, dtype=torch.bfloat16)
    errs = {}
    for bits in SUPPORTED_BITS:
        codes, scales, zeros = quantize_uniform(weight, bits, group_size=128)
        out = dequantize_uniform(
            pack_uniform(codes, bits), scales, zeros, bits=bits, group_size=128,
            out_features=64, in_features=256,
        )
        errs[bits] = (out.float() - weight.float()).norm().item()
    assert errs[8] < errs[4], errs


def test_matches_qat_uniform_ste():
    """The served weight must equal the weight training optimized against."""
    torch.manual_seed(0)
    for shape in [(64, 256), (96, 300)]:      # padded and unpadded
        weight = torch.randn(*shape)
        for bits in SUPPORTED_BITS:
            trained = uniform_ste(weight, bits=bits, group_size=128)

            codes, scales, zeros = quantize_uniform(weight, bits, group_size=128)
            served = dequantize_uniform(
                pack_uniform(codes, bits), scales, zeros,
                bits=bits, group_size=128,
                out_features=shape[0], in_features=shape[1],
                dtype=torch.float32,
            )
            gap = (trained - served).abs().max().item()
            assert gap < 1e-5, (
                f"shape={shape} bits={bits}: trained and served weights differ "
                f"by {gap:.2e} -- the streaming engine would not reproduce the "
                f"quantizer distill_seq optimized against"
            )


def test_bits_per_weight_accounting():
    # 4 bits + 16-bit scale/128 + 8-bit zero/128
    assert abs(bits_per_weight(128, 4) - 4.1875) < 1e-9
    assert abs(bits_per_weight(64, 4) - 4.375) < 1e-9
