# Résumé / portfolio copy

Everything below describes work that actually exists in this repo and has passing
tests. Nothing is aspirational. See the honesty notes at the bottom before an
interview.

---

## THE PARAGRAPH (use this one)

Conversational, covers the whole project including where it's headed.

> I'm building a system that runs large language models on hardware that has no
> business fitting them — the target is an 8GB NVIDIA Jetson Orin Nano. It
> combines two techniques that work far better together than apart. The first is
> **ternary quantization**: instead of storing each weight as a 16-bit number,
> you store it as just −1, 0, or +1 plus a shared scaling factor, which gets you
> roughly 8× smaller at about 1.6 bits per weight. The second is **layer-by-layer
> streaming**: rather than loading the whole model into memory, you read a single
> transformer layer off the SSD, run it, discard it, and load the next — so peak
> memory is bounded by one layer instead of the entire network. They pair well
> because streaming is bottlenecked entirely on bytes-read-per-layer, and ternary
> is exactly what shrinks that: in my case from ~139MB to ~17MB per layer.
>
> So far I've built the streaming engine in PyTorch and have it running
> Microsoft's BitNet b1.58 — a 2.4-billion-parameter natively-ternary model — at
> about **700MB peak memory and 7.5 tokens/second** on an M-series Mac, staying
> correct all the way down to a **50MB** weight budget, which forces 718 layer
> evictions in a single 24-token generation. The part I'm most pleased with is
> the correctness harness: since these checkpoints arrive already ternary, my
> engine performs no arithmetic on the weights at all, which means streamed
> output should be **bit-identical** to the same model loaded conventionally —
> and it is, token-for-token, with **zero numerical deviation across all 30
> decoder layers**. That strict bar paid for itself immediately by catching two
> bugs that produced fluent, plausible, completely wrong text while passing every
> other test I had: a missing activation-quantization step (BitNet quantizes
> activations to 8 bits as well as weights, which I'd overlooked), and a severed
> weight-tying reference that silently left the output projection
> uninitialized.
>
> The other half is quality, and it's where the harder work sits. Naively
> rounding a normal full-precision model's weights down to ternary destroys it —
> I measured early transformer blocks collapsing to 0.73 output similarity
> against their originals. Proper quantization-aware training would fix that but
> needs roughly **375GB of optimizer state** for a 31B model, which isn't
> happening on consumer hardware. So I built a **block-wise quantization-aware
> distillation** pipeline instead: each full-precision layer acts as teacher to
> its ternary student, gradients flow through a straight-through estimator into
> latent full-precision weights, and only one block is resident at a time —
> cutting the requirement to about **8GB** and recovering those damaged layers
> from 0.73 to **0.95+**. Next is porting the engine to the Jetson and running
> the distillation at scale on an H100 to produce a large ternary model actually
> worth streaming, since the biggest one publicly available today is only 3.9B
> parameters.

### Shorter cut (~6 lines)

> I'm building a system to run large language models on an 8GB NVIDIA Jetson Orin
> Nano by combining ternary quantization — storing each weight as just −1, 0 or
> +1, about 8× smaller — with layer-by-layer streaming, where you read one
> transformer layer off the SSD, run it, and discard it, so peak memory is bounded
> by a single layer rather than the whole model. The two compound, because
> streaming is bottlenecked on bytes-per-layer and ternary is precisely what
> shrinks it. I have the engine running a 2.4B-parameter ternary model at ~700MB
> and 7.5 tok/s, proven **bit-identical** to the conventionally-loaded reference
> across all 30 layers — a bar strict enough that it caught two bugs producing
> fluent-but-wrong output. I also built a block-wise quantization-aware
> distillation pipeline that trains one layer at a time, cutting the memory needed
> from ~375GB to ~8GB and lifting quantized-layer fidelity from 0.73 to 0.95+.
> Next: the Jetson port, and H100 distillation to build a ternary model large
> enough that streaming genuinely pays off.

---

## One-liner

> Built a bounded-memory inference engine that runs 1.58-bit ternary LLMs
> layer-by-layer from disk with bit-exact fidelity, plus a memory-bounded
> quantization-aware distillation pipeline that makes ternary models usable.

---

## Résumé bullets (dense version)

**Ternary LLM Streaming Inference Engine** — *Python, PyTorch, HuggingFace Transformers*

- Engineered a bounded-footprint inference engine executing a 2.4B-parameter
  **1.58-bit ternary** LLM one transformer layer at a time from disk, constraining
  peak resident weights to a configurable byte budget via an **LRU residency
  manager with asynchronous prefetch** that overlaps layer I/O with compute;
  sustained correct generation at a 50 MB budget across 718 evictions.
- Established **bit-exact correctness** against the unstreamed reference
  implementation — token-identical generations with maximum absolute deviation of
  0 across all 30 decoder layers, embeddings, and output logits — closing the
  silent-corruption failure mode where a quantized engine emits fluent but
  semantically wrong text while passing every smoke test.
- Implemented **BitNet b1.58** quantization end to end: absmean weight
  quantization to {−1, 0, +1} with group-wise scaling, dual bit-packing backends
  (2-bit shift-packed, and **base-3 trit packing at 1.60 bits/weight** exploiting
  3⁵ = 243 < 256), and a byte-exact reader for HuggingFace's strided packed
  `bitnet` layout validated tensor-by-tensor against the reference unpacker.
- Designed a **memory-bounded quantization-aware distillation** pipeline —
  straight-through estimator over latent fp32 weights, teacher-forced block
  inputs, and a scale-invariant reconstruction objective — reducing optimizer
  state from ~375 GB (naive end-to-end QAT on a 31B model) to ~8 GB by training a
  single transformer block at a time; improved per-block output fidelity from
  **0.871 → 0.958 cosine** against the full-precision teacher.
- Diagnosed and eliminated two silent-corruption defects via layer-wise
  activation diffing: absent **W1.58A8 per-token int8 activation quantization**,
  and a severed `tie_word_embeddings` reference leaving the output projection
  unmaterialized on the `meta` device.
- Conducted a first-principles survey of the low-bit quantization landscape
  (TQ1_0/TQ2_0/Q2_0/I2_S packed formats, per-backend kernel coverage), surfacing
  that **the target format had zero CUDA support** and selecting an alternative
  with 4× finer scale granularity — higher fidelity *and* GPU-viable.

---

## Compressed version (3 bullets, for a tight résumé)

- Built a bounded-memory streaming inference engine for **1.58-bit ternary LLMs**,
  executing a 2.4B-parameter model layer-by-layer from disk within a configurable
  byte budget using LRU residency and asynchronous prefetch; validated
  **token-identical, zero-deviation** output against the unstreamed reference.
- Implemented BitNet b1.58 ternary quantization with group-wise scaling and dual
  bit-packing backends (2-bit and **base-3 trit packing at 1.60 bits/weight**),
  plus a byte-exact reader for HuggingFace's packed format.
- Designed a **memory-bounded quantization-aware distillation** pipeline using
  straight-through estimation and per-block teacher forcing, cutting optimizer
  state ~45× and lifting quantized-layer fidelity from 0.871 to 0.958 cosine
  versus the full-precision teacher.

---

## Project description (portfolio / LinkedIn)

Modern language models are memory-bound: a 31B-parameter model needs ~62 GB just
to hold its weights, which rules out edge hardware. Two techniques address this
independently — **ternary quantization** compresses each weight to {−1, 0, +1}
(~8× smaller), and **layer streaming** executes one layer at a time so peak
memory is bounded by a single layer rather than the full model. They had not been
combined in the PyTorch ecosystem, and they are complementary: streaming is
bottlenecked on bytes-per-layer, which is exactly what ternary attacks.

This project implements both and composes them. The inference engine instantiates
a model on PyTorch's `meta` device (zero allocation), substitutes every linear
layer with a packed-ternary equivalent, and drives residency through forward
pre-hooks against an LRU manager with a background prefetch thread — deliberately
delegating attention, RoPE, masking, and KV-cache management to `transformers` so
that architecture-specific correctness is inherited rather than reimplemented.

Correctness is enforced at the strictest available bar: because a
natively-ternary checkpoint requires no arithmetic on the weights, streamed
output must be **token-identical** to the unstreamed reference, not merely close.
This caught two defects that produced fluent, plausible, wrong text — the exact
failure mode that makes quantized inference dangerous to ship.

The second half addresses quality. Naive post-training rounding to ternary is
destructive, measurably so: early transformer blocks degrade to 0.73 output
cosine against their full-precision counterparts. Full quantization-aware
training would resolve this but requires ~375 GB of optimizer state for a 31B
model. The pipeline here instead performs **block-wise quantization-aware
distillation** — treating each full-precision block as a teacher for its ternary
student, with gradients flowing through a straight-through estimator into latent
weights — bounding peak memory to a single block and recovering the damaged layers
to 0.95+.

---

## Skills demonstrated

`Quantization (PTQ/QAT/STE)` · `Knowledge distillation` · `PyTorch internals
(meta device, forward hooks, autograd, custom nn.Module)` · `Bit-level data
packing` · `Memory hierarchy & I/O-bound systems design` · `LRU caching &
asynchronous prefetch` · `Numerical debugging (layer-wise activation diffing)` ·
`Transformer architecture` · `HuggingFace Transformers / safetensors` ·
`Apple MPS & CUDA backends` · `Test-driven numerical validation`

---

## Honesty notes — read before interviewing on this

Things you should **not** claim, because they aren't done:

- **Not yet run on the Jetson.** All measurements are Apple Silicon (MPS). ARM
  findings are from upstream bug reports, not our own runs.
- **No end-to-end perplexity number.** The 0.958 figure is *per-block* output
  cosine on a 135M model. Errors compound with depth; 30 layers at 0.95 is not
  0.95 overall. Say "per-block reconstruction fidelity," never "the model is 95%
  as good."
- **No H100 distillation run yet.** The pipeline exists and is validated at small
  scale; it has not trained a large model.
- **Ternary weights are dequantized to bf16 before matmul.** There is no
  sub-8-bit arithmetic kernel here — the win is memory and I/O, not FLOPs. If
  asked "does it run faster?", the honest answer is: it runs *at all*, in less
  memory, at ~1.8× the latency of the fully-resident path.
- **The engine has not yet delivered a large memory win in practice.** On this
  model the unquantized embedding table (657 MB) exceeds all 30 ternary layers
  combined (522 MB), so peak memory floors at ~0.70 GB regardless of budget.
  That's a real and interesting finding — present it as one. It's the kind of
  measurement that shows you actually ran the thing.

The strongest genuine claim: **you built a quantized inference engine and proved
it bit-exact against a reference, then used that harness to find two bugs that
would otherwise have shipped silently.** That is more impressive to an engineer
than any throughput number, because it demonstrates you know how quantized
systems fail.
