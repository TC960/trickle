"""Does the EXISTING streaming engine already handle a hybrid DeltaNet+MoE model?

model.py never reimplements attention (or DeltaNet): it meta-instantiates the
real `transformers` model class wholesale, replaces every `nn.Linear` it finds
with a streamable equivalent, and hooks each `...layers.N` module to
fetch/evict its shard. None of that logic is specific to "standard" attention
-- it is generic over whatever `nn.Module` tree the real HF class happens to
build. This file checks how far that genericity actually carries for
`Qwen3_5MoeForCausalLM` (Qwen3.5/3.6-35B-A3B's text model: alternating
Gated-DeltaNet "linear_attention" layers and ordinary "full_attention" layers,
both MoE-routed), using the REAL reference `Qwen3_5MoeGatedDeltaNet` -- not
this project's port in `deltanet.py` -- so this is a test of
`loader.py`/`model.py`'s wiring, not of the DeltaNet math (see
`test_deltanet.py` for that).

Two findings came out of actually running this, one good and one a genuine gap:

GOOD: DeltaNet's own tensors (`in_proj_*`, `conv1d`, `dt_bias`, `A_log`, `norm`,
`out_proj`) don't match any name in `policy.QUANTIZABLE_PROJECTIONS`, so the
EXISTING policy already stores them dense/bf16, and the EXISTING `_swap_linears`
/ `_attach_streaming_hooks` already stream them layer-by-layer -- no new
tensor-name handling was needed for the DeltaNet layer itself. Likewise, one
orthogonal snag surfaced and was worked around, unrelated to this project's
code: transformers' MoE block picks a `grouped_mm`-based forward by default,
whose CPU meta-shape-check requires bf16 inputs and has no working CPU kernel
at toy dimensions. Fixed with transformers' own public
`model.set_experts_implementation("eager")` call.

GAP (found here, not previously known): this installed transformers version's
`Qwen3_5MoeExperts` stores all experts' weights as two fused 3D
`nn.Parameter`s (`gate_up_proj`, `down_proj`, shape `[num_experts, ...]`), NOT
as per-expert `nn.Linear` submodules. But `save_pretrained` serializes them to
disk as per-expert 2D tensors (`experts.0.gate_proj.weight`,
`experts.1.up_proj.weight`, ...) -- and `policy.py`'s name-matching marks
those split names "ternary" because they contain "gate_proj"/"up_proj"/
"down_proj". Neither `_swap_linears` (they aren't `nn.Linear`) nor
`_materialize_dense` (they're marked "ternary", not "dense") ever binds them
back into the live model's actual `gate_up_proj`/`down_proj` parameters, which
are built via `init_empty_weights()` and never get touched again. Routed-expert
weights silently stay on the meta device for the entire run. The forward pass
does not crash (see `test_routed_moe_expert_weights_are_not_actually_loaded`
for why, verified directly) -- which is exactly the "plausible output from a
broken instrument is worse than no output" trap CLAUDE.md warns about: nothing
here would tell you the routed-expert contribution is wrong unless you check
`.is_meta` yourself, as this test does.

This second finding is a concrete, previously-unstated blocker for a real
end-to-end run beyond the abstract "MoE shard granularity" concern already in
CLAUDE.md's pivot notes -- `policy.py`/`shard.py` need to learn this
checkpoint's actual per-expert-vs-fused parameter layout before MoE weights can
be streamed (or even correctly loaded at all) for this architecture. Fixing
that is out of scope here (the task was DeltaNet); this file exists to make
the gap precise and reproducible rather than leave it as a guess.
"""

import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

try:
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    _HAS_QWEN3_5_MOE = True
except ImportError:
    _HAS_QWEN3_5_MOE = False

pytestmark = pytest.mark.skipif(
    not _HAS_QWEN3_5_MOE,
    reason="installed transformers build has no qwen3_5_moe (Qwen3.5/3.6-35B-A3B) support",
)

from airllm_ternary import PrecisionPolicy, build_shards  # noqa: E402
from airllm_ternary.linear import TernaryLinear  # noqa: E402
from airllm_ternary.model import load_streaming_model  # noqa: E402

NUM_LAYERS = 4  # layer_types -> linear, linear, linear, full (one full_attention_interval cycle)

# What DOES get correctly streamed today: the 4 self_attn projections on the
# one full_attention layer, plus each layer's shared_expert (a plain
# Qwen3_5MoeMLP with ordinary nn.Linear gate/up/down projections). The
# per-token-ROUTED experts (see module docstring) do not.
_EXPECTED_STREAMED_TERNARY = 1 * 4 + NUM_LAYERS * 3


def _tiny_hybrid_config():
    return Qwen3_5MoeTextConfig(
        vocab_size=256,
        hidden_size=32,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        hidden_act="silu",
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
        linear_conv_kernel_dim=4,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts_per_tok=2,
        num_experts=4,
        full_attention_interval=4,
    )


@pytest.fixture(scope="module")
def tiny_hybrid_checkpoint():
    config = _tiny_hybrid_config()
    assert config.layer_types == [
        "linear_attention", "linear_attention", "linear_attention", "full_attention",
    ], config.layer_types  # sanity: this is the architecture we mean to be testing

    torch.manual_seed(0)
    model = Qwen3_5MoeForCausalLM(config).to(torch.bfloat16).eval()

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / "model"
        model.save_pretrained(model_dir, safe_serialization=True)
        yield model_dir


def _build_shards(model_dir, out_dir):
    policy = PrecisionPolicy(
        num_layers=NUM_LAYERS, skip_first_layers=0, skip_last_layers=0, group_size=8,
    )
    return build_shards(model_dir, out_dir, policy, measure_error=False, verbose=False)


def test_build_shards_classifies_deltanet_tensors_as_dense(tiny_hybrid_checkpoint):
    """DeltaNet's own tensors don't match any QUANTIZABLE_PROJECTIONS name, so
    the existing policy already stores them dense/bf16 -- no new tensor-name
    handling was needed for this part."""
    with tempfile.TemporaryDirectory() as tmp:
        manifest = _build_shards(tiny_hybrid_checkpoint, tmp)

        deltanet_tensors = [name for name in manifest["tensors"] if ".linear_attn." in name]
        assert deltanet_tensors, "no linear_attn tensors found -- fixture didn't build a hybrid model"
        for name in deltanet_tensors:
            assert manifest["tensors"][name]["format"] == "dense", (
                f"{name} was unexpectedly quantized; DeltaNet's own weights should "
                "stay dense until policy.py is deliberately extended to ternarize them"
            )

        # Every DeltaNet layer's tensors land in that layer's own shard, same
        # as any other decoder layer -- shard_key() only looks at the `layers.N`
        # index, not the layer's internal type.
        for i in range(3):  # layers 0-2 are linear_attention per full_attention_interval=4
            assert manifest["tensors"][f"model.layers.{i}.linear_attn.out_proj.weight"]["shard"] == f"language.{i:03d}"


def test_routed_moe_expert_tensor_names_collide_with_the_ternarizable_whitelist(tiny_hybrid_checkpoint):
    """The GAP, part 1: `save_pretrained` splits each layer's fused
    `experts.gate_up_proj`/`experts.down_proj` into per-expert 2D tensors named
    `experts.{i}.gate_proj.weight` / `.up_proj.weight` / `.down_proj.weight` on
    disk, which DO match `QUANTIZABLE_PROJECTIONS`, so `build_shards` marks all
    of them "ternary". That's real data sitting in the shard files -- but (see
    the next test) it's data nothing ever reads back."""
    with tempfile.TemporaryDirectory() as tmp:
        manifest = _build_shards(tiny_hybrid_checkpoint, tmp)
        expert_tensors = [
            name for name, entry in manifest["tensors"].items()
            if ".mlp.experts." in name and entry["format"] == "ternary"
        ]
        assert len(expert_tensors) == NUM_LAYERS * 4 * 3  # 4 experts x {gate,up,down}_proj per layer


def test_routed_moe_expert_weights_are_not_actually_loaded(tiny_hybrid_checkpoint):
    """The GAP, part 2 -- the actual bug: the live model's `Qwen3_5MoeExperts`
    module (built fresh via `init_empty_weights()` inside `load_streaming_model`)
    holds the FUSED 3D parameters `gate_up_proj` / `down_proj`, not per-expert
    `nn.Linear`s. `_swap_linears` only swaps `isinstance(module, nn.Linear)`
    (these aren't), and `_materialize_dense` only binds tensors whose manifest
    `format == "dense"` (these are "ternary"). So the split per-expert entries
    from the previous test are never bound to anything, and these two
    parameters are left on the meta device for the entire run -- silently: no
    exception, no NaN in the final logits (the shared_expert path still
    produces real numbers), just a routed-expert contribution that was never
    actually computed from real weights.
    """
    with tempfile.TemporaryDirectory() as tmp:
        manifest = _build_shards(tiny_hybrid_checkpoint, tmp)
        model, manager = load_streaming_model(
            tiny_hybrid_checkpoint, tmp, budget_gb=10.0, device="cpu", prefetch=False,
        )
        model.set_experts_implementation("eager")

        for layer in model.model.layers:
            experts = layer.mlp.experts
            assert experts.gate_up_proj.is_meta, (
                "gate_up_proj is no longer meta -- if policy.py/shard.py were fixed to "
                "reassemble the fused parameter, this test (and its docstring) is stale "
                "and should be deleted, not adjusted"
            )
            assert experts.down_proj.is_meta

        # The run still "succeeds" in the sense of not raising -- which is the
        # point: a plausible-looking result from a broken path is worse than a
        # crash, per CLAUDE.md's standing methodology notes.
        input_ids = torch.randint(0, 256, (1, 6))
        with torch.inference_mode():
            output = model(input_ids)
        assert not torch.isnan(output.logits).any()
        manager.close()


def test_streaming_generation_runs_through_hybrid_layers(tiny_hybrid_checkpoint):
    """DeltaNet layers, the full-attention layer, and each layer's
    shared_expert all stream correctly (load -> compute -> evict) through the
    UNMODIFIED existing engine. Routed MoE experts do not participate
    meaningfully yet (previous test) -- counted here only to pin the number so
    a future fix changes this assertion, not silently un-notices it."""
    with tempfile.TemporaryDirectory() as tmp:
        manifest = _build_shards(tiny_hybrid_checkpoint, tmp)
        largest = manifest["summary"]["largest_layer_bytes"]
        globals_bytes = manifest["shards"]["globals"]["nbytes"]
        budget_gb = (globals_bytes + largest * 2) / 1e9

        model, manager = load_streaming_model(
            tiny_hybrid_checkpoint, tmp, budget_gb=budget_gb, device="cpu", prefetch=False,
        )
        model.set_experts_implementation("eager")

        ternary_modules = [m for m in model.modules() if isinstance(m, TernaryLinear)]
        assert len(ternary_modules) == _EXPECTED_STREAMED_TERNARY, (
            len(ternary_modules), _EXPECTED_STREAMED_TERNARY,
        )

        type_names = {type(m).__name__ for m in model.modules()}
        assert "Qwen3_5MoeGatedDeltaNet" in type_names
        assert "Qwen3_5MoeAttention" in type_names

        input_ids = torch.randint(0, 256, (1, 6))
        with torch.inference_mode():
            output = model.generate(input_ids, max_new_tokens=4, do_sample=False, use_cache=True)
        assert output.shape[1] == 10

        report = manager.report()
        assert report["evictions"] > 0, "budget was too generous to force real eviction traffic"
        manager.close()


def test_streaming_output_matches_full_residency(tiny_hybrid_checkpoint):
    """Same invariant as test_pipeline.py's dense-model version: the residency
    budget must change memory traffic, never the numbers -- true here too,
    though (given the previous two tests) this is currently a statement about
    the DeltaNet/attention/shared-expert path, since the routed-expert
    contribution is meta/uninitialized identically in both runs."""
    with tempfile.TemporaryDirectory() as tmp:
        manifest = _build_shards(tiny_hybrid_checkpoint, tmp)
        total = manifest["summary"]["total_bytes"]
        globals_bytes = manifest["shards"]["globals"]["nbytes"]
        largest = manifest["summary"]["largest_layer_bytes"]

        input_ids = torch.randint(0, 256, (1, 6))
        outputs = []
        for budget in ((globals_bytes + largest * 2) / 1e9, total * 4 / 1e9):
            model, manager = load_streaming_model(
                tiny_hybrid_checkpoint, tmp, budget_gb=budget, device="cpu", prefetch=False,
            )
            model.set_experts_implementation("eager")
            with torch.inference_mode():
                outputs.append(
                    model.generate(input_ids, max_new_tokens=5, do_sample=False, use_cache=True)
                )
            manager.close()

        assert torch.equal(outputs[0], outputs[1]), (
            "streaming and resident modes diverged on a DeltaNet+MoE hybrid model; "
            "residency must not affect math here either"
        )


# Follow-up work this file deliberately does not attempt (see CLAUDE.md Part 6
# and the module docstring above for the newly-found specifics):
#   - Fix policy.py/shard.py to recognize this transformers version's fused
#     `experts.gate_up_proj` / `experts.down_proj` parameter layout (per-expert
#     split on disk, fused in the live module) so routed-expert weights are
#     actually loaded -- this blocks correct output, not just efficient I/O.
#   - Per-expert shard granularity: even once loading is fixed, shard_key() has
#     no notion of "which experts fired"; a real 256-expert layer would still
#     ship every expert's weights in one shard file, which is most of the
#     memory win MoE was supposed to buy back.
#   - The linear/full-attention KV+recurrent-state Cache is transformers' own
#     generic `Cache`/`LinearAttentionLayer` (see deltanet.py's module
#     docstring) -- correct here because `generate()` builds it automatically,
#     but never exercised against a real multi-GB checkpoint or a real Jetson
#     device.
#   - GGUF-to-safetensors conversion for a real downloaded checkpoint; this
#     test only ever round-trips a checkpoint this process itself wrote.
