# airLLM + ternary

**Bounded-memory inference for 1.58-bit ternary LLMs.** Runs a transformer one
layer at a time from disk, keeping peak resident weights under a configurable
byte budget — with output proven bit-identical to conventional loading.

Target hardware is an 8 GB NVIDIA Jetson Orin Nano. Development and all current
measurements are on Apple Silicon (M5 Max, MPS).

---

## The idea

Large language models are memory-bound. A 31B-parameter model needs ~62 GB just
to hold its weights, which rules out edge devices. Two techniques address this,
and they compound:

**Ternary quantization** stores each weight as `-1`, `0`, or `+1` plus a shared
scale — roughly 8× smaller than bf16, at ~1.6–2.0 bits per weight.

**Layer streaming** loads one transformer layer off disk, runs it, discards it,
and loads the next. Peak memory is bounded by a single layer instead of the whole
network.

They pair well because **streaming is bottlenecked entirely on bytes-read-per-layer**,
and ternary is exactly what shrinks that:

| | bf16 | ternary |
|---|---|---|
| BitNet-2B layer | ~139 MB | **~17 MB** |
| Gemma-4-31B layer (projected) | ~958 MB | **~127 MB** |

---

## Results

Measured on `microsoft/bitnet-b1.58-2B-4T` (2.4B params, 30 layers), Apple M5 Max.

### Correctness

Because a natively-ternary checkpoint needs no arithmetic on its weights, the
engine only moves packed bytes. So the bar is **exact token equality**, not a
perplexity tolerance:

```
PASS: all 4 completions are token-identical to the unstreamed reference
      max|Δ| = 0 across all 30 decoder layers, embeddings, and logits
```

### Memory / throughput trade-off

| budget | tok/s | peak RSS | bytes read | evictions | correct |
|---|---|---|---|---|---|
| 0.05 GB | 4.36 | 0.69 GB | 12.53 GB | 718 | ✅ |
| 0.10 GB | 4.28 | 0.71 GB | 12.53 GB | 715 | ✅ |
| 0.20 GB | 4.27 | 0.70 GB | 12.53 GB | 709 | ✅ |
| 1.20 GB | 7.68 | 0.70 GB | 0.52 GB | 0 | ✅ |

Only a **1.8× throughput penalty for 24× more I/O** — because ternary layers are
small enough that a background prefetch thread hides most of the read latency.

### Quantization-aware distillation

Naive post-training rounding to ternary is destructive. Block-wise distillation
recovers it. Measured on `SmolLM2-135M`, 6 blocks, 300 steps each:

| block | naive PTQ | distilled | gain | MSE drop |
|---|---|---|---|---|
| 0 | 0.7358 | **0.9482** | +0.2124 | 5.1× |
| 1 | 0.7483 | **0.9697** | +0.2215 | 7.8× |
| 2 | 0.8097 | 0.8442 | +0.0344 | 1.2× |
| 3 | 0.9911 | 0.9975 | +0.0065 | 3.7× |
| 4 | 0.9767 | 0.9960 | +0.0193 | 4.6× |
| 5 | 0.9638 | 0.9931 | +0.0294 | 4.4× |

**Mean output cosine 0.871 → 0.958.** The early blocks, which PTQ mangles worst,
recover the most. Block 2 resists training — likely massive-activation
pathology — and is a candidate for bf16 fallback.

---

## Quick start

```bash
source activate.sh                        # creates venv with uv, sets paths
python -m airllm_ternary.build_bitnet     # download + shard (one-time)
python chat.py                            # chat, 0.75 GB budget
```

```bash
python chat.py --budget-gb 0.05           # minimum footprint, heavy streaming
pytest tests/ -q                          # 20 tests
```

In-session commands: `/stats` (engine counters), `/reset`, `/raw`, `/quit`.

Verify correctness against the reference yourself:

```bash
python experiments/verify_streaming.py --shards "$AIRLLM_SHARDS"
```

---

## How it works

**The model is never built with real weights.** It's instantiated on PyTorch's
`meta` device (zero allocation), then every `nn.Linear` is swapped for a
streamable packed-ternary equivalent, and forward pre-hooks on each decoder layer
fetch that layer's shard before it runs.

This delegates attention, RoPE, masking, and KV-cache management to
`transformers`. Architecture-specific correctness is inherited rather than
reimplemented — which matters for models with asymmetric head dimensions or
per-layer scalars, where a hand-rolled forward pass is very easy to get subtly
wrong.

**The budget is a dial, not a switch.** One code path spans the range:

| budget | behaviour |
|---|---|
| ≥ shard total | all layers resident after first touch; an ordinary quantized model |
| a few layers | true streaming, ~1 layer of I/O per layer of compute |
| 1 layer | minimum footprint, maximum I/O |

**Only the projections stream.** `q/k/v/o/gate/up/down` are ~99% of each layer's
parameters. Norms, per-layer scalars, and the embedding table are materialized
once and pinned — paging a few kilobytes of RMSNorm gains in and out would cost
I/O and save nothing.

---

## Two bugs the correctness harness caught

Both produced **fluent, plausible, completely wrong text** while passing every
smoke test. This is the failure mode that makes quantized inference dangerous to
ship, and the reason the bar here is token equality rather than "looks fine."

**1. Missing activation quantization.** BitNet is W1.58**A8** — ternary weights
*and* per-token int8 activations. Implementing only the weight half changes the
numerics enough to wreck the model, because it was trained expecting quantized
activations.

**2. Severed weight tying.** With `tie_word_embeddings: True`, `lm_head` shares
the embedding matrix. Replacing `embed_tokens.weight` with a fresh `nn.Parameter`
leaves `lm_head` pointing at the original meta tensor — an unmaterialized output
projection, with no error raised. There's now a guard that fails loudly.

A third, subtler one: instantiating a model from config skips
`generation_config.json`, which `from_pretrained` would have loaded. Without it
there are no stop tokens, so the model runs to `max_new_tokens` and cheerfully
role-plays both sides of the conversation.

---

## Packed format notes

All of these store the same `-1/0/+1`. The bits-per-weight difference is
**packing efficiency and scale granularity**, not value precision — and scale
granularity is what drives quality, in the opposite direction from bit count:

| format | bpw | weights per scale | CUDA kernels |
|---|---|---|---|
| TQ1_0 (llama.cpp) | 1.69 | 256 | ❌ none |
| TQ2_0 (llama.cpp) | 2.06 | 256 | ❌ none |
| **Q2_0** (llama.cpp) | 2.25 | **64** | ✅ MMQ + MMVQ |
| HF `bitnet` | 2.00 | per-tensor | via `AutoBitLinear` |

**Q2_0 spends ~9% more bytes to get 4× more scales** — higher fidelity *and* the
only ternary format with merged CUDA support. For a CUDA target like the Jetson,
TQ2_0 would silently fall back to CPU.

This repo reads the HF `bitnet` layout (4 values per `uint8`, strided along dim
0, per-tensor scale), verified byte-exact against
`transformers.integrations.bitnet.unpack_weights`. It also implements its own
2-bit and base-3 trit packing (1.60 bpw, exploiting 3⁵ = 243 < 256) for
quantizing non-ternary models.

---

## Repo layout

```
airllm_ternary/
  loader.py           LRU residency manager, byte budget, prefetch thread
  model.py            meta-device assembly, linear swapping, streaming hooks
  linear.py           TernaryLinear / BitNetLinear / HighPrecisionLinear
  bitnet_format.py    HF packed-ternary reader + A8 activation quantization
  ternary.py          BitNet b1.58 absmean quantization, 2-bit & trit packing
  policy.py           which tensors stay high-precision, and why
  shard.py            quantize + shard a full-precision checkpoint
  shard_native.py     losslessly re-shard an already-ternary checkpoint
  qat.py              straight-through estimator, QATLinear
  reconstruct.py      block-wise quantization-aware distillation
  build_bitnet.py     one-command download + shard

experiments/
  verify_streaming.py  streamed vs reference, exact token equality
  qat_validate.py      distillation vs naive PTQ, head to head
  debug_divergence.py  layer-by-layer tensor diffing

tests/                 20 tests
results/               measured outputs
```

`results/qat_v1_*` used a raw MSE objective, which oscillated on deeper blocks
because activation magnitude grows with depth. `results/qat_v2_*` uses a
scale-invariant loss and is the reported result.

---

## Limitations

**Ternary weights are dequantized to bf16 before matmul.** There is no sub-8-bit
arithmetic kernel here. The win is memory and I/O, not FLOPs — compute is
identical to bf16 once a weight is materialized. Real ternary compute speedup
needs custom Metal/CUDA kernels.

**The embedding table is the actual bottleneck.** On BitNet-2B it is 657 MB —
*larger than all 30 ternary layers combined* (522 MB) — and it's unquantized and
pinned. So peak memory floors at ~0.70 GB no matter how tight the budget. Layer
streaming took this model from 1.17 GB to 0.70 GB: a 40% saving, not the 8× the
per-layer numbers suggest.

**Cache hit rate is 0 at tight budgets, and that's inherent.** A cyclic scan over
N layers with room for fewer than N misses on every access regardless of eviction
policy. Prefetch is what recovers throughput, not caching.

**No natively-ternary model above ~4B parameters exists publicly.** The largest is
`SpectraSuite/TriLM_3.9B`. Everything bigger (Falcon3-10B-1.58bit,
Llama3-8B-1.58) is a *converted* full-precision model. Layer streaming only
really pays off above that ceiling — which is the argument for the distillation
track.

**Not yet validated:** end-to-end perplexity (the 0.958 is *per-block* output
cosine on a 135M model; errors compound with depth), anything on actual Jetson
hardware, and distillation at scale.

---

## Roadmap

1. **Quantize/stream the embedding table** — now the binding constraint
2. **Jetson Orin Nano port** — Q2_0 format, real CUDA target
3. **H100 distillation** — build a ternary model large enough that streaming pays

---

## References

- BitNet b1.58 — [arXiv:2504.12285](https://arxiv.org/abs/2504.12285) · [microsoft/bitnet-b1.58-2B-4T](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
- AirLLM — layer-by-layer inference
- [oLLM](https://github.com/Mega4alik/ollm) — layer streaming, fp16/bf16 only
- [fucina](https://github.com/matteo-grella/fucina) — ternary + bounded streaming, MoE-expert granularity, TQ2_0
- llama.cpp ternary quant types `TQ1_0` / `TQ2_0` / `Q2_0`
