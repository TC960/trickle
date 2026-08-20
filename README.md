# airLLM + ternary — Gemma 4 31B

Layer-by-layer streaming inference (AirLLM-style) with BitNet b1.58 ternary
weights, for `google/gemma-4-31B`.

## Why the two belong together

AirLLM's bottleneck is **bytes moved per layer**. It runs one layer at a time so
peak memory is bounded by a single layer instead of the whole model, but it pays
for that by reading every layer from disk on every forward pass.

Ternary quantization attacks exactly that cost:

| | bf16 | ternary (2-bit packed, g=128) |
|---|---|---|
| Sliding layer (~479M params) | 958 MB | **~127 MB** |
| Full-attention layer (~545M) | 1.09 GB | ~145 MB |
| All 60 decoder layers | ~59 GB | **~7.8 GB** |
| + tied embeddings (kept bf16) | 2.8 GB | 2.8 GB |

**~7.5× less I/O per layer.** That is the whole thesis.

## What Gemma 4 31B actually is

Worth knowing, because several of these drove design decisions:

- **60 decoder layers** — 50 sliding (window 1024) + 10 full, strict 5:1 pattern
- **Asymmetric attention** — sliding layers use `head_dim=256` / 16 KV heads;
  full-attention layers use `head_dim=512` / only **4** KV heads
- `attention_k_eq_v: true`, QK-norm, and a per-layer `layer_scalar` tensor
- hidden 5376, MLP 21504, vocab 262144, **tied** embeddings, 256K context
- A 27-layer / 1152-hidden vision tower (~430M params)
- 62.5 GB on disk at bf16, ungated on HF

Those 4 global KV heads matter: they make the KV cache unusually cheap. At 32K
context the whole cache is only ~3.5 GB, because the 50 sliding layers are
capped by their 1024 window and only the 10 full layers grow with sequence
length.

## Design decisions

**We do not reimplement Gemma 4's forward pass.** The model is instantiated on
the `meta` device (zero allocation), every `nn.Linear` is swapped for a
streamable `TernaryLinear`, and forward pre-hooks on each decoder layer fetch
that layer's shard. transformers keeps ownership of attention, RoPE, masking and
the KV cache — so the asymmetric head dims, `layer_scalar` and QK-norm stay
correct without us reproducing them.

**The budget is a dial, not a switch.** One code path covers the whole range:

| budget | behaviour |
|---|---|
| ≥ shard total | every layer resident after first touch; an ordinary quantized model |
| a few layers | true streaming, ~1 layer of I/O per layer of compute |
| 1 layer | minimum footprint, maximum I/O |

A background thread prefetches layer N+1 while layer N computes.

**Only the projections stream.** `q/k/v/o/gate/up/down` are ~99% of each layer's
parameters. Norms, `layer_scalar` and the embedding table are materialized once
and pinned — paging a few kilobytes of RMSNorm gains in and out would cost I/O
and save nothing.

**What never gets ternarized**, and why:

- `embed_tokens` — 262144 × 5376 = 1.41B params, and `tie_word_embeddings` means
  the same matrix is also the output head. Quantizing it damages both the input
  representation and every logit.
- norms / `layer_scalar` — kilobytes each, nothing to gain
- the vision tower — small share of params, and vision encoders are far more
  sensitive to weight noise than decoder MLPs (`--quantize-vision` to override)
- first and last decoder layer — most outlier-heavy activations (tunable)

## Packing modes

- `2bit` — 4 values/byte via shifts. 2.00 bits/weight, fast dequant. Default.
- `trit5` — 5 base-3 trits/byte (3⁵ = 243 < 256). **1.60 bits/weight**, the
  honest "1.58-bit" figure, ~20% smaller but slower to unpack.

## Usage

```bash
source activate.sh     # creates/activates the venv and sets cache paths

python cli.py build  <model_dir> shards/ --verbose
python cli.py run    <model_dir> shards/ --budget-gb 4 --prompt "..."
python cli.py bench  <model_dir> shards/          # sweep the budget curve
```

`build` reports per-tensor reconstruction cosine similarity and lists the
worst-reconstructed tensors, which are the candidates for
`force_high_precision` in the policy.

## Status

Working and tested:

- ternary quantize / pack / unpack, both modes, round-trip lossless (8 tests)
- full pipeline: quantize → shard → meta-instantiate → swap → stream → generate
  (6 tests, on a synthetic checkpoint)
- **streaming and fully-resident modes produce bit-identical output** — residency
  changes memory, never numerics
- eviction respects the budget; prefetch serves real hits

Not yet validated:

- **Quality on the real 31B.** Post-training ternarization without QAT is
  destructive, and no perplexity number has been measured yet. This is the main
  open risk.
- Throughput on the real model.

## Caveats worth knowing

**There is no ternary matmul kernel on MPS.** Weights are dequantized to bf16 and
run through normal `F.linear`. The win is bytes on disk and bytes resident, *not*
FLOPs — compute is identical to bf16 once a weight is materialized. Real ternary
compute speedup would need custom Metal kernels.

**AirLLM may be more than this machine needs.** Ternary weights (~10.6 GB) plus
KV at 32K (~3.5 GB) is ~14 GB against 36 GB of unified memory — the model fits
resident. Streaming earns its keep when targeting smaller devices or when
minimizing footprint on principle. Set `--budget-gb` high and you get the fast
path for free.

**Google ships a QAT 4-bit** (`google/gemma-4-31B-it-qat-w4a16-ct`). It had
training-time budget we don't, so it's the honest quality ceiling to measure
ternary against.
