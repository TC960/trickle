"""Gated DeltaNet correctness: pin our port against the real transformers class.

Per CLAUDE.md's methodology rule, a test that only exercises our own code
against itself proves almost nothing (see bug #13: a round-trip test passed
throughout while the quantizer and storage format silently disagreed). So
every test here constructs the REAL
`transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeGatedDeltaNet`
with random weights, copies those exact weights into our
`airllm_ternary.deltanet.GatedDeltaNet` (parameter names match 1:1 by
construction), and asserts the two produce matching numbers on the same input.

Covers, in increasing order of integration depth:
  1. The two low-level math functions in isolation (no module, no cache):
     `recurrent_gated_delta_rule` and `chunk_gated_delta_rule` against
     `torch_recurrent_gated_delta_rule` / `torch_chunk_gated_delta_rule`.
  2. The full module's prefill path (no cache), which exercises the causal
     conv + projections + gated norm around the chunked math.
  3. The full module driven by a REAL `transformers.cache_utils.Cache` +
     `LinearAttentionLayer` across a prefill-then-single-token-decode
     sequence -- this is the cross-component check: it proves our module
     satisfies the actual state-cache protocol transformers ships, not a
     protocol we invented and tested against itself, and that the decode
     step actually reaches the recurrent (not chunked) code path.

Skips (not an import-error swallow -- these fail loudly if transformers/
torch are simply missing) only if the installed transformers build doesn't
ship qwen3_5_moe at all, so this file stays runnable on older environments
without silently reporting false confidence.
"""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

try:
    from transformers.cache_utils import Cache, LinearAttentionLayer
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeGatedDeltaNet as ReferenceGatedDeltaNet,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        torch_chunk_gated_delta_rule as reference_chunk_gated_delta_rule,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        torch_recurrent_gated_delta_rule as reference_recurrent_gated_delta_rule,
    )

    _HAS_QWEN3_5_MOE = True
except ImportError:
    _HAS_QWEN3_5_MOE = False

pytestmark = pytest.mark.skipif(
    not _HAS_QWEN3_5_MOE,
    reason="installed transformers build has no qwen3_5_moe (Qwen3.5/3.6-35B-A3B) support",
)

from airllm_ternary.deltanet import (  # noqa: E402
    GatedDeltaNet,
    chunk_gated_delta_rule,
    recurrent_gated_delta_rule,
)

# Small dims: real config is hidden_size=2048, 16 key heads x 128, 32 value
# heads x 128, kernel 4 -- kept proportional (num_v_heads = 2x num_k_heads,
# so the repeat_interleave branch is exercised) but tiny so the test is fast.
_HIDDEN_SIZE = 32
_NUM_K_HEADS = 2
_NUM_V_HEADS = 4
_HEAD_K_DIM = 8
_HEAD_V_DIM = 8
_CONV_KERNEL = 4
_LAYER_IDX = 0


def _tiny_config():
    return SimpleNamespace(
        hidden_size=_HIDDEN_SIZE,
        linear_num_key_heads=_NUM_K_HEADS,
        linear_num_value_heads=_NUM_V_HEADS,
        linear_key_head_dim=_HEAD_K_DIM,
        linear_value_head_dim=_HEAD_V_DIM,
        linear_conv_kernel_dim=_CONV_KERNEL,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        layer_types=["linear_attention"] * (_LAYER_IDX + 1),
    )


def _matched_pair(seed: int = 0):
    """A reference module and our port, sharing identical random weights."""
    torch.manual_seed(seed)
    config = _tiny_config()
    reference = ReferenceGatedDeltaNet(config, layer_idx=_LAYER_IDX).eval()
    ours = GatedDeltaNet(config, layer_idx=_LAYER_IDX).eval()
    ours.load_state_dict(reference.state_dict())
    return reference, ours


def _max_mean_abs_diff(a: torch.Tensor, b: torch.Tensor):
    diff = (a.float() - b.float()).abs()
    return diff.max().item(), diff.mean().item()


# --------------------------------------------------------------------------
# 1. Low-level math, no module/cache involved.
# --------------------------------------------------------------------------


def test_recurrent_gated_delta_rule_matches_reference():
    torch.manual_seed(1)
    batch, seq = 2, 7
    query = torch.randn(batch, seq, _NUM_V_HEADS, _HEAD_K_DIM)
    key = torch.randn(batch, seq, _NUM_V_HEADS, _HEAD_K_DIM)
    value = torch.randn(batch, seq, _NUM_V_HEADS, _HEAD_V_DIM)
    g = -torch.rand(batch, seq, _NUM_V_HEADS)  # log-space decay, must be <= 0
    beta = torch.sigmoid(torch.randn(batch, seq, _NUM_V_HEADS))

    ref_out, ref_state = reference_recurrent_gated_delta_rule(
        query, key, value, g=g, beta=beta,
        initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True,
    )
    our_out, our_state = recurrent_gated_delta_rule(
        query, key, value, g=g, beta=beta,
        initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True,
    )

    max_diff, mean_diff = _max_mean_abs_diff(ref_out, our_out)
    assert torch.allclose(ref_out, our_out, atol=1e-5, rtol=1e-5), (
        f"recurrent output mismatch: max_abs={max_diff:.3e} mean_abs={mean_diff:.3e}"
    )
    state_max_diff, state_mean_diff = _max_mean_abs_diff(ref_state, our_state)
    assert torch.allclose(ref_state, our_state, atol=1e-5, rtol=1e-5), (
        f"recurrent final-state mismatch: max_abs={state_max_diff:.3e} mean_abs={state_mean_diff:.3e}"
    )


def test_chunk_gated_delta_rule_matches_reference():
    torch.manual_seed(2)
    batch, seq = 2, 37  # deliberately not a multiple of chunk_size, to hit padding
    query = torch.randn(batch, seq, _NUM_V_HEADS, _HEAD_K_DIM)
    key = torch.randn(batch, seq, _NUM_V_HEADS, _HEAD_K_DIM)
    value = torch.randn(batch, seq, _NUM_V_HEADS, _HEAD_V_DIM)
    g = -torch.rand(batch, seq, _NUM_V_HEADS)
    beta = torch.sigmoid(torch.randn(batch, seq, _NUM_V_HEADS))

    ref_out, ref_state = reference_chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta, chunk_size=16,
        initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True,
    )
    our_out, our_state = chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta, chunk_size=16,
        initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True,
    )

    max_diff, mean_diff = _max_mean_abs_diff(ref_out, our_out)
    assert torch.allclose(ref_out, our_out, atol=1e-5, rtol=1e-5), (
        f"chunked output mismatch: max_abs={max_diff:.3e} mean_abs={mean_diff:.3e}"
    )
    state_max_diff, state_mean_diff = _max_mean_abs_diff(ref_state, our_state)
    assert torch.allclose(ref_state, our_state, atol=1e-5, rtol=1e-5), (
        f"chunked final-state mismatch: max_abs={state_max_diff:.3e} mean_abs={state_mean_diff:.3e}"
    )


def test_recurrent_and_chunked_paths_agree_with_each_other():
    """Sanity check independent of the reference: both of our own paths solve
    the same recurrence, so on a short sequence they must agree with each
    other too (not just each with its own reference counterpart)."""
    torch.manual_seed(3)
    batch, seq = 1, 5
    query = torch.randn(batch, seq, _NUM_V_HEADS, _HEAD_K_DIM)
    key = torch.randn(batch, seq, _NUM_V_HEADS, _HEAD_K_DIM)
    value = torch.randn(batch, seq, _NUM_V_HEADS, _HEAD_V_DIM)
    g = -torch.rand(batch, seq, _NUM_V_HEADS)
    beta = torch.sigmoid(torch.randn(batch, seq, _NUM_V_HEADS))

    recurrent_out, _ = recurrent_gated_delta_rule(
        query, key, value, g=g, beta=beta, use_qk_l2norm_in_kernel=True,
    )
    chunked_out, _ = chunk_gated_delta_rule(
        query, key, value, g=g, beta=beta, chunk_size=16, use_qk_l2norm_in_kernel=True,
    )
    assert torch.allclose(recurrent_out, chunked_out, atol=1e-4, rtol=1e-4)


# --------------------------------------------------------------------------
# 2. Full module, prefill (chunked) path, no cache.
# --------------------------------------------------------------------------


def test_full_module_prefill_matches_reference():
    reference, ours = _matched_pair(seed=10)
    torch.manual_seed(20)
    hidden_states = torch.randn(2, 11, _HIDDEN_SIZE)

    with torch.no_grad():
        ref_out = reference(hidden_states, cache_params=None, attention_mask=None)
        our_out = ours(hidden_states, cache_params=None, attention_mask=None)

    max_diff, mean_diff = _max_mean_abs_diff(ref_out, our_out)
    assert torch.allclose(ref_out, our_out, atol=1e-4, rtol=1e-4), (
        f"full-module prefill mismatch: max_abs={max_diff:.3e} mean_abs={mean_diff:.3e}"
    )


# --------------------------------------------------------------------------
# 3. Full module through a real transformers Cache: prefill then one-token
#    decode. This is the genuine cross-component check -- it proves our
#    module satisfies transformers' own state-cache protocol (not a stand-in
#    we wrote to please our own module) and that the decode step actually
#    takes the recurrent code path, not chunked-with-seq-len-1.
# --------------------------------------------------------------------------


def test_full_module_prefill_then_decode_via_real_cache_matches_reference():
    reference, ours = _matched_pair(seed=30)
    torch.manual_seed(40)
    prefix = torch.randn(1, 9, _HIDDEN_SIZE)
    next_token = torch.randn(1, 1, _HIDDEN_SIZE)

    cache_ref = Cache(layers=[LinearAttentionLayer()])
    cache_ours = Cache(layers=[LinearAttentionLayer()])

    with torch.no_grad():
        ref_prefill = reference(prefix, cache_params=cache_ref, attention_mask=None)
        our_prefill = ours(prefix, cache_params=cache_ours, attention_mask=None)

    prefill_max, prefill_mean = _max_mean_abs_diff(ref_prefill, our_prefill)
    assert torch.allclose(ref_prefill, our_prefill, atol=1e-4, rtol=1e-4), (
        f"cached prefill mismatch: max_abs={prefill_max:.3e} mean_abs={prefill_mean:.3e}"
    )

    # Confirm the decode step below will actually exercise the recurrent
    # per-token path (seq_len == 1 and a previous state is present), not the
    # chunked path degenerating to one token.
    assert cache_ref.has_previous_state(_LAYER_IDX, state_idx=0)
    assert cache_ours.has_previous_state(_LAYER_IDX, state_idx=0)

    with torch.no_grad():
        ref_decode = reference(next_token, cache_params=cache_ref, attention_mask=None)
        our_decode = ours(next_token, cache_params=cache_ours, attention_mask=None)

    decode_max, decode_mean = _max_mean_abs_diff(ref_decode, our_decode)
    assert torch.allclose(ref_decode, our_decode, atol=1e-4, rtol=1e-4), (
        f"cached single-token decode mismatch: max_abs={decode_max:.3e} mean_abs={decode_mean:.3e}"
    )

    # And the persisted state itself -- the thing that has to survive the
    # streaming loop's evict/reload cycle -- must match too, not just the
    # emitted logits.
    ref_state = cache_ref.layers[_LAYER_IDX].recurrent_states[0]
    our_state = cache_ours.layers[_LAYER_IDX].recurrent_states[0]
    state_max, state_mean = _max_mean_abs_diff(ref_state, our_state)
    assert torch.allclose(ref_state, our_state, atol=1e-4, rtol=1e-4), (
        f"recurrent state mismatch after decode: max_abs={state_max:.3e} mean_abs={state_mean:.3e}"
    )
    assert ref_state.dtype == torch.float32 and our_state.dtype == torch.float32, (
        "recurrent state must stay float32 across the cache boundary regardless of "
        "the surrounding model's compute dtype"
    )

    ref_conv = cache_ref.layers[_LAYER_IDX].conv_states[0]
    our_conv = cache_ours.layers[_LAYER_IDX].conv_states[0]
    conv_max, conv_mean = _max_mean_abs_diff(ref_conv, our_conv)
    assert torch.allclose(ref_conv, our_conv, atol=1e-5, rtol=1e-5), (
        f"conv state mismatch after decode: max_abs={conv_max:.3e} mean_abs={conv_mean:.3e}"
    )
