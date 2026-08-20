"""End-to-end pipeline test on a synthetic checkpoint.

Builds a tiny randomly-initialized model, shards it to ternary, then runs
generation through the streaming loader. This exercises the real code path --
quantize, shard, meta-instantiate, swap linears, hook layers, evict, generate --
without downloading anything.
"""

import tempfile
from pathlib import Path

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from airllm_ternary import PrecisionPolicy, build_shards, load_manifest
from airllm_ternary.linear import TernaryLinear
from airllm_ternary.model import load_streaming_model

NUM_LAYERS = 6


@pytest.fixture(scope="module")
def tiny_checkpoint():
    """A small real checkpoint on disk, saved as safetensors."""
    config = LlamaConfig(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    torch.manual_seed(0)
    model = LlamaForCausalLM(config).to(torch.bfloat16).eval()

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / "model"
        model.save_pretrained(model_dir, safe_serialization=True)
        yield model_dir, model


def _build(model_dir, out_dir, **policy_kwargs):
    policy = PrecisionPolicy(
        num_layers=NUM_LAYERS,
        skip_first_layers=0,
        skip_last_layers=0,
        **policy_kwargs,
    )
    return build_shards(model_dir, out_dir, policy, measure_error=True, verbose=False)


def test_shards_are_written_per_layer(tiny_checkpoint):
    model_dir, _ = tiny_checkpoint
    with tempfile.TemporaryDirectory() as tmp:
        manifest = _build(model_dir, tmp, group_size=64)

        # One shard per decoder layer, plus the pinned globals shard.
        assert "globals" in manifest["shards"]
        for i in range(NUM_LAYERS):
            assert f"language.{i:03d}" in manifest["shards"], f"missing layer {i}"

        quantized = [
            n for n, e in manifest["tensors"].items() if e["format"] == "ternary"
        ]
        # 7 projections per layer: q, k, v, o, gate, up, down.
        assert len(quantized) == NUM_LAYERS * 7, len(quantized)


def test_embeddings_and_norms_stay_dense(tiny_checkpoint):
    """The policy must never ternarize the embedding table or any norm."""
    model_dir, _ = tiny_checkpoint
    with tempfile.TemporaryDirectory() as tmp:
        manifest = _build(model_dir, tmp, group_size=64)
        for name, entry in manifest["tensors"].items():
            if "embed_tokens" in name or "norm" in name:
                assert entry["format"] == "dense", f"{name} was quantized"


def test_streaming_generation_runs_and_evicts(tiny_checkpoint):
    """The core claim: generation works with a budget far below model size."""
    model_dir, _ = tiny_checkpoint
    with tempfile.TemporaryDirectory() as tmp:
        manifest = _build(model_dir, tmp, group_size=64)
        largest = manifest["summary"]["largest_layer_bytes"]

        # Budget deliberately tiny: room for the pinned globals plus ~2 layers,
        # forcing real eviction traffic through the manager.
        globals_bytes = manifest["shards"]["globals"]["nbytes"]
        budget_gb = (globals_bytes + largest * 2) / 1e9

        model, manager = load_streaming_model(
            model_dir, tmp, budget_gb=budget_gb, device="cpu", prefetch=False,
        )

        # Verify the swap actually happened.
        ternary_modules = [
            m for m in model.modules() if isinstance(m, TernaryLinear)
        ]
        assert len(ternary_modules) == NUM_LAYERS * 7

        input_ids = torch.randint(0, 512, (1, 8))
        with torch.inference_mode():
            output = model.generate(
                input_ids, max_new_tokens=5, do_sample=False, use_cache=True
            )

        assert output.shape[1] == 13, output.shape

        report = manager.report()
        assert report["evictions"] > 0, "budget was too generous to force eviction"
        assert report["misses"] >= NUM_LAYERS
        assert manager.resident_bytes <= manager.budget_bytes
        manager.close()


def test_full_residency_avoids_reloading(tiny_checkpoint):
    """With a budget above model size, layers load once and stay put."""
    model_dir, _ = tiny_checkpoint
    with tempfile.TemporaryDirectory() as tmp:
        manifest = _build(model_dir, tmp, group_size=64)
        total = manifest["summary"]["total_bytes"]

        model, manager = load_streaming_model(
            model_dir, tmp, budget_gb=total * 4 / 1e9, device="cpu", prefetch=False,
        )
        input_ids = torch.randint(0, 512, (1, 8))
        with torch.inference_mode():
            model.generate(input_ids, max_new_tokens=5, do_sample=False)

        report = manager.report()
        assert report["evictions"] == 0, report
        # Each layer read exactly once across all 5 generated tokens.
        assert report["misses"] == NUM_LAYERS + 1, report  # +1 for globals
        manager.close()


def test_streaming_output_matches_full_residency(tiny_checkpoint):
    """Residency budget must change memory use, never the numerics."""
    model_dir, _ = tiny_checkpoint
    with tempfile.TemporaryDirectory() as tmp:
        manifest = _build(model_dir, tmp, group_size=64)
        total = manifest["summary"]["total_bytes"]
        globals_bytes = manifest["shards"]["globals"]["nbytes"]
        largest = manifest["summary"]["largest_layer_bytes"]

        input_ids = torch.randint(0, 512, (1, 8))
        outputs = []
        for budget in ((globals_bytes + largest * 2) / 1e9, total * 4 / 1e9):
            model, manager = load_streaming_model(
                model_dir, tmp, budget_gb=budget, device="cpu", prefetch=False,
            )
            with torch.inference_mode():
                outputs.append(
                    model.generate(input_ids, max_new_tokens=6, do_sample=False)
                )
            manager.close()

        assert torch.equal(outputs[0], outputs[1]), (
            "streaming and resident modes diverged; residency must not affect math"
        )


def test_prefetch_thread_serves_hits(tiny_checkpoint):
    """Prefetch should satisfy some loads off the critical path."""
    model_dir, _ = tiny_checkpoint
    with tempfile.TemporaryDirectory() as tmp:
        manifest = _build(model_dir, tmp, group_size=64)
        globals_bytes = manifest["shards"]["globals"]["nbytes"]
        largest = manifest["summary"]["largest_layer_bytes"]

        model, manager = load_streaming_model(
            model_dir, tmp,
            budget_gb=(globals_bytes + largest * 2) / 1e9,
            device="cpu", prefetch=True,
        )
        input_ids = torch.randint(0, 512, (1, 8))
        with torch.inference_mode():
            model.generate(input_ids, max_new_tokens=8, do_sample=False)

        assert manager.report()["prefetch_hits"] > 0, manager.report()
        manager.close()
