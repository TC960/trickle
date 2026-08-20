"""Correctness tests for the ternary quantize/pack/unpack round trip."""

import torch

from airllm_ternary.ternary import (
    check_group_size,
    dequantize,
    pack,
    packed_nbytes,
    quantization_error,
    quantize_ternary,
    unpack,
)


def test_pack_unpack_roundtrip_is_lossless():
    """Packing is pure storage: unpack(pack(x)) must equal x exactly."""
    for mode, group in (("2bit", 128), ("trit5", 120)):
        codes = torch.randint(-1, 2, (64, group), dtype=torch.int8)
        restored = unpack(pack(codes, mode), mode)
        assert torch.equal(codes, restored), f"{mode} round trip lost data"


def test_all_code_values_survive_packing():
    """Every one of the 3 symbols must survive at every position in a byte."""
    for mode, stride in (("2bit", 4), ("trit5", 5)):
        # Exhaustive over one full byte's worth of positions.
        import itertools

        combos = list(itertools.product([-1, 0, 1], repeat=stride))
        codes = torch.tensor(combos, dtype=torch.int8)
        restored = unpack(pack(codes, mode), mode)
        assert torch.equal(codes, restored), f"{mode} lost a symbol"


def test_quantize_produces_only_ternary_values():
    weight = torch.randn(256, 512)
    codes, scales = quantize_ternary(weight, group_size=128)
    assert set(codes.unique().tolist()) <= {-1, 0, 1}
    assert scales.shape == (codes.shape[0], 1)
    assert (scales.float() > 0).all()


def test_dequantize_recovers_shape_and_scale():
    """Reconstruction must match the original shape and stay correlated."""
    torch.manual_seed(0)
    weight = torch.randn(128, 384) * 0.02  # realistic transformer weight scale
    codes, scales = quantize_ternary(weight, group_size=128)
    packed = pack(codes, "2bit")
    restored = dequantize(
        packed, scales,
        group_size=128, out_features=128, in_features=384,
        mode="2bit", dtype=torch.float32,
    )
    assert restored.shape == weight.shape

    metrics = quantization_error(weight, restored)
    # Ternary on Gaussian weights lands around 0.9 cosine; well above chance.
    assert metrics["cosine"] > 0.85, metrics
    assert 0.2 < metrics["sparsity"] < 0.5, metrics


def test_padding_handles_non_multiple_input_dims():
    """in_features not divisible by group_size must still round trip cleanly."""
    weight = torch.randn(32, 300)  # 300 is not a multiple of 128
    codes, scales = quantize_ternary(weight, group_size=128)
    restored = dequantize(
        pack(codes, "2bit"), scales,
        group_size=128, out_features=32, in_features=300,
        mode="2bit", dtype=torch.float32,
    )
    assert restored.shape == (32, 300)
    assert quantization_error(weight, restored)["cosine"] > 0.85


def test_trit5_is_denser_than_2bit():
    """Base-3 packing must actually deliver its ~20% size advantage."""
    numel = 1_000_000
    two_bit = packed_nbytes(numel, 128, "2bit")
    trit = packed_nbytes(numel, 120, "trit5")
    assert trit < two_bit
    bits_per_weight = trit * 8 / numel
    assert 1.55 < bits_per_weight < 1.75, bits_per_weight


def test_group_size_validation_rejects_bad_strides():
    check_group_size(128, "2bit")
    check_group_size(120, "trit5")
    for bad, mode in ((127, "2bit"), (128, "trit5")):
        try:
            check_group_size(bad, mode)
        except ValueError:
            continue
        raise AssertionError(f"expected {bad}/{mode} to be rejected")


def test_smaller_groups_reduce_error():
    """The core justification for group-wise scales over a single per-tensor one."""
    torch.manual_seed(0)
    weight = torch.randn(256, 1024)
    # Make one column band much larger, the outlier pattern that wrecks
    # per-tensor scaling in real checkpoints.
    weight[:, :64] *= 25.0

    errors = {}
    for group in (1024, 128, 32):
        codes, scales = quantize_ternary(weight, group_size=group)
        restored = dequantize(
            pack(codes, "2bit"), scales,
            group_size=group, out_features=256, in_features=1024,
            mode="2bit", dtype=torch.float32,
        )
        errors[group] = quantization_error(weight, restored)["rel_fro"]

    assert errors[32] < errors[128] < errors[1024], errors
