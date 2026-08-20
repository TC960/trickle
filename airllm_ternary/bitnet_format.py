"""Reader for the HuggingFace "bitnet" packed weight layout.

This is the format `microsoft/bitnet-b1.58-2B-4T` and the Falcon-E models ship
in. We read it rather than invent our own, so shards stay interchangeable with
the upstream checkpoints.

Layout, verified against `transformers/integrations/bitnet.py` and against the
actual checkpoint tensors:

  - 4 ternary values per uint8, so exactly 2.00 bits per weight
  - codes are {0, 1, 2} meaning {-1, 0, +1}; decode is `code - 1`
  - packing runs along **dim 0** (output features) and is **strided, not
    contiguous**: with `row_dim = ceil(rows / 4)`, original row `r` lives in
    packed row `r % row_dim` at bit offset `2 * (r // row_dim)`, LSB first
  - one bf16 scale per tensor, in a sibling `<name>.weight_scale`

The strided part is the trap. A contiguous reader produces plausible-looking
weights in the wrong order and a model that emits fluent nonsense, with no error
anywhere. The unit tests check against transformers' own unpacker for exactly
this reason.
"""

import torch

# Values per packed byte, and the offset that turns codes into {-1, 0, +1}.
VALUES_PER_BYTE = 4
CODE_OFFSET = 1


def packed_row_dim(out_features: int) -> int:
    """Number of packed rows used to store `out_features` original rows."""
    return (out_features + VALUES_PER_BYTE - 1) // VALUES_PER_BYTE


def unpack_bitnet(packed: torch.Tensor, out_features: int) -> torch.Tensor:
    """Unpack a uint8 tensor into int8 ternary codes in {-1, 0, +1}.

    `packed` has shape [ceil(out_features/4), ...]; the result has shape
    [out_features, ...].
    """
    row_dim = packed.shape[0]
    rest = packed.shape[1:]

    # Extract the four 2-bit fields, each giving one slab of original rows.
    slabs = []
    for index in range(VALUES_PER_BYTE):
        codes = (packed >> (2 * index)) & 0b11
        slabs.append(codes.to(torch.int8) - CODE_OFFSET)

    unpacked = torch.cat(slabs, dim=0)
    # The final slab is short when out_features is not a multiple of 4.
    return unpacked[:out_features].reshape(out_features, *rest)


def pack_bitnet(codes: torch.Tensor) -> torch.Tensor:
    """Inverse of `unpack_bitnet`. Takes int8 in {-1, 0, +1}, returns uint8."""
    out_features = codes.shape[0]
    row_dim = packed_row_dim(out_features)

    shifted = (codes + CODE_OFFSET).to(torch.uint8)
    if out_features % VALUES_PER_BYTE:
        pad_rows = row_dim * VALUES_PER_BYTE - out_features
        padding = torch.zeros(
            (pad_rows, *codes.shape[1:]), dtype=torch.uint8, device=codes.device
        )
        shifted = torch.cat([shifted, padding], dim=0)

    packed = torch.zeros(
        (row_dim, *codes.shape[1:]), dtype=torch.uint8, device=codes.device
    )
    for index in range(VALUES_PER_BYTE):
        slab = shifted[index * row_dim : (index + 1) * row_dim]
        packed |= slab << (2 * index)
    return packed


def dequantize_bitnet(
    packed: torch.Tensor,
    scale: torch.Tensor,
    out_features: int,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Reconstruct the full-precision weight matrix.

    Note the scale is per-tensor here, not per-group. That is the format's main
    quality limitation and the reason Q2_0 (one scale per 64 weights) does
    better -- but it is what these checkpoints ship, so we honour it.
    """
    codes = unpack_bitnet(packed, out_features)
    return codes.to(dtype) * scale.to(dtype)


def act_quant(activation: torch.Tensor) -> torch.Tensor:
    """Per-token symmetric int8 activation quantization.

    BitNet is W1.58**A8**: the weights are ternary AND the activations are
    quantized to 8 bits before every matmul. Skipping this does not merely lose
    a little precision -- it changes the numerics enough to produce fluent
    nonsense, because the model was trained expecting quantized activations.

    Mirrors `transformers.integrations.bitnet.ActQuant.forward` exactly,
    including the float() upcast and the op order, so results are bit-identical.
    """
    dtype = activation.dtype
    activation = activation.float()
    scale = 127 / activation.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    activation = (activation * scale).round().clamp(-128, 127) / scale
    return activation.to(dtype)


def bitnet_nbytes(out_features: int, in_features: int) -> int:
    """Packed byte count for one weight matrix, including its scale."""
    return packed_row_dim(out_features) * in_features + 2


def infer_out_features(packed_shape, weight_shape=None) -> int:
    """Recover the original out_features from a packed tensor's shape.

    Ambiguous by itself -- ceil(n/4) maps four different n to the same row_dim --
    so the model config is the authority. This returns the largest candidate,
    which is correct whenever out_features is a multiple of 4 (true for every
    tensor in the shipped checkpoints).
    """
    if weight_shape is not None:
        return weight_shape[0]
    return packed_shape[0] * VALUES_PER_BYTE
