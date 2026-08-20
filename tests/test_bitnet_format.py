"""Verify our bitnet unpacker matches transformers' own implementation.

This is the highest-value test in the suite. The HF layout packs along dim 0 in
a strided pattern, and a contiguous reader yields weights in the wrong order --
producing a model that runs cleanly and emits fluent nonsense. Checking against
the reference implementation is the only way to be sure.
"""

import pytest
import torch

from airllm_ternary.bitnet_format import (
    dequantize_bitnet,
    pack_bitnet,
    packed_row_dim,
    unpack_bitnet,
)

# transformers is the authority on this format; skip rather than guess if absent.
transformers_bitnet = pytest.importorskip("transformers.integrations.bitnet")


def test_matches_transformers_unpacker():
    """Our unpack must agree with transformers' unpack_weights, exactly."""
    torch.manual_seed(0)
    for out_features, in_features in ((16, 8), (2560, 2560), (640, 6912), (160, 512)):
        packed = torch.randint(
            0, 256, (packed_row_dim(out_features), in_features), dtype=torch.uint8
        )
        theirs = transformers_bitnet.unpack_weights(packed, dtype=torch.float32)
        ours = unpack_bitnet(packed, out_features).float()

        assert theirs.shape == ours.shape, (theirs.shape, ours.shape)
        assert torch.equal(theirs, ours), (
            f"mismatch at {out_features}x{in_features}: "
            f"{(theirs != ours).sum().item()} differing elements"
        )


def test_pack_unpack_roundtrip():
    torch.manual_seed(1)
    codes = torch.randint(-1, 2, (256, 64), dtype=torch.int8)
    assert torch.equal(unpack_bitnet(pack_bitnet(codes), 256), codes)


def test_pack_matches_transformers_packer():
    """Our packer must produce bytes transformers can read back."""
    torch.manual_seed(2)
    codes = torch.randint(-1, 2, (64, 32), dtype=torch.int8)
    packed = pack_bitnet(codes)
    recovered = transformers_bitnet.unpack_weights(packed, dtype=torch.float32)
    assert torch.equal(recovered, codes.float())


def test_only_ternary_values_emerge():
    packed = torch.randint(0, 256, (32, 16), dtype=torch.uint8)
    codes = unpack_bitnet(packed, 128)
    # Code 3 is unreachable from a valid packer but decodes to +2 if present;
    # random bytes exercise it, so the valid range here is -1..2.
    assert codes.min() >= -1 and codes.max() <= 2


def test_dequantize_applies_scale():
    codes = torch.tensor([[-1, 0], [1, -1], [0, 1], [1, 0]], dtype=torch.int8)
    packed = pack_bitnet(codes)
    scale = torch.tensor([0.05], dtype=torch.bfloat16)
    weight = dequantize_bitnet(packed, scale, 4, dtype=torch.float32)
    expected = codes.float() * 0.05
    assert torch.allclose(weight, expected, atol=1e-3), (weight, expected)


def test_real_checkpoint_tensor_roundtrips():
    """Round-trip an actual tensor from the downloaded BitNet checkpoint."""
    import glob
    import os

    pattern = os.path.expanduser(
        "~/.cache/airllm-ternary/hf-cache/hub/"
        "models--microsoft--bitnet-b1.58-2B-4T/snapshots/*/model.safetensors"
    )
    matches = glob.glob(pattern)
    if not matches:
        pytest.skip("BitNet checkpoint not downloaded")

    from safetensors import safe_open

    with safe_open(matches[0], framework="pt", device="cpu") as handle:
        packed = handle.get_tensor("model.layers.0.self_attn.q_proj.weight")

    # q_proj is [640, 2560] packed -> [2560, 2560] unpacked.
    assert packed.dtype == torch.uint8 and tuple(packed.shape) == (640, 2560)
    codes = unpack_bitnet(packed, 2560)
    assert tuple(codes.shape) == (2560, 2560)

    theirs = transformers_bitnet.unpack_weights(packed, dtype=torch.float32)
    assert torch.equal(theirs, codes.float())

    # A real trained ternary layer should use all three symbols in quantity.
    fractions = [(codes == v).float().mean().item() for v in (-1, 0, 1)]
    assert all(f > 0.05 for f in fractions), fractions
    assert torch.equal(pack_bitnet(codes), packed), "repacking changed the bytes"
