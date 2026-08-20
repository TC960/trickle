# What we're actually building — plain language

Written 2026-08-18. This replaces the jargon I dumped on you. Read top to bottom;
each section is short.

---

## The 30-second version

We want to run a large language model on a **small device** — an 8 GB Jetson
Orin Nano. Two tricks, combined:

1. **Make the weights tiny.** Store each weight as just `-1`, `0`, or `+1`
   instead of a 16-bit number. That's ~8× smaller.
2. **Only hold one layer in memory at a time.** Load layer 1 from the SSD, run
   it, throw it away, load layer 2. Peak memory = one layer, not the whole model.

Trick 1 makes trick 2 fast, because trick 2's bottleneck is *how many bytes you
read per layer*. That's the whole idea.

---

## Glossary — the terms I threw at you

### Ternary quantization
A normal model stores each weight as a 16-bit number (65,536 possible values).
**Ternary** stores it as one of three values: `-1`, `0`, `+1`. You also keep one
normal number per group of weights, called a **scale**, which says "multiply all
these -1/0/+1s by 0.037." So the weight isn't literally -1, it's -1 × scale.

Why it works at all: neural networks are enormously redundant. What matters is
mostly the *sign* and rough magnitude of each weight, not its precise value.

Why it's called "1.58-bit": three possible values needs log₂(3) = 1.58 bits of
information. In practice you round up to 2 bits per weight for speed.

### Layer streaming (what AirLLM does)
A transformer is a stack of near-identical layers — Gemma 4 31B has 60 of them.
Normally you load all 60 into memory. Layer streaming loads **one at a time**:

```
load layer 0 → run it → free it → load layer 1 → run it → free it → ...
```

Peak memory becomes one layer instead of sixty. The cost is that you re-read
every layer from disk for **every token you generate**. That's why the byte count
per layer matters so much, and why ternary helps.

### PTQ vs QAT — the important distinction
- **PTQ** (post-training quantization): take a finished model, round its weights
  to -1/0/+1. Fast — minutes. Also **destructive**, because nothing in the model
  ever got a chance to adapt.
- **QAT** (quantization-aware training): *train* the model with the rounding
  happening in the loop, so it learns weights that survive rounding. Much better
  quality. Costs real compute.

I built PTQ first. Your instinct that it was bad was correct — I measured it and
it was mangling the early layers badly (see the numbers section below).

### oLLM
An open-source project (github.com/Mega4alik/ollm) that does layer streaming —
same trick as AirLLM. It's impressive: it runs an 80B model on an 8 GB GPU.

**But it streams full-precision weights only.** Its README says outright: *"No
quantization is used—only fp16/bf16 precision."* So it moves ~10× more bytes per
layer than it needs to, and runs at about 0.5 tokens/sec. It's the "streaming
without ternary" corner of the map.

### fucina — "the Matteo thing"
A project (github.com/matteo-grella/fucina) that **does combine ternary +
streaming**. I flagged it because it contradicts the claim in your brief that
nobody has shipped that combination. Details in its own section below.

### The packed formats: I2_S, TQ1_0, TQ2_0, Q2_0
These are all just **file layouts** — different conventions for how you cram
-1/0/+1 values into bytes. Think of them as ZIP vs RAR: same content, different
container. Which one you pick matters only because *the fast code that reads them
is format-specific.*

| Format | Who uses it | Bits per weight | Scale granularity |
|---|---|---|---|
| **I2_S** | bitnet.cpp | 2.0 | 1 per tensor |
| **TQ1_0** | llama.cpp | 1.69 | 1 per 256 weights |
| **TQ2_0** | llama.cpp, fucina | 2.06 | 1 per 256 weights |
| **Q2_0** | llama.cpp (new) | 2.25 | **1 per 64 weights** |
| HF "bitnet" | BitNet, Falcon-E | 2.0 | 1 per tensor |

### "Dense layer-granular ternary streaming in PyTorch/HF is unclaimed"
Sorry — that was four pieces of jargon in a row. Unpacked:

- **dense** = a normal model, as opposed to a "mixture of experts" model (MoE),
  which has many specialist sub-networks and only uses a few per token
- **layer-granular** = we stream whole layers; fucina streams individual experts
  inside a layer
- **PyTorch/HF** = the Python ecosystem (PyTorch + HuggingFace), as opposed to
  the C++/Zig ecosystem where llama.cpp and fucina live

So: *"nobody has built ternary + layer-streaming for normal models in Python."*
Which is a fair description of the gap, and it's the gap we sit in.

### "The doc in point 4"
A single markdown file in a pull request to the bitnet.cpp project (PR #507)
that documents their file format. **It has the sign mapping backwards** — it says
`00 → 0, 01 → +1, 10 → -1`, but the actual code does `value = code - 1`.

Why I mentioned it: if we'd written our reader from that doc, every weight would
have had the wrong sign and the model would have output garbage — with no error
message. It's a landmine, not a feature. Ignore it; trust the source code.

---

## Your distillation idea is the actual plan

You said:

> "if we're using a parental teacher model wherein the parent has a higher
> quantization and a student model has a smaller quantization and then
> quantization aware training, basically distilling a bigger, more capable model
> into a smaller quantized model to retain its qualities"

**Yes. That's exactly right, and it's the state of the art.** It has a name —
*quantization-aware distillation* — and it's how every good ternary model above
2B was made. Falcon3-10B-1.58bit and Llama3-8B-1.58 were both built this way:
take a strong full-precision model, convert to ternary, keep training with the
original as teacher.

It's also, partly, what I already built. `reconstruct.py` does this **one layer
at a time**: the full-precision layer is the teacher, the ternary copy is the
student, and the student trains to match the teacher's output. I did it per-layer
because that was the only version that fit on your Mac.

With H100 credits, we can do the stronger version — see below.

---

## Why "no native ternary model above 4B" was a problem

**Two ways to get a ternary model:**

| | How | Quality |
|---|---|---|
| **Native** | trained in ternary from scratch | best |
| **Converted** | take a normal model, make it ternary | depends on effort |

The largest **native** ternary model anyone has published is 3.9B parameters.
Nothing bigger exists — Microsoft's biggest is 2.4B, and their 2026 releases went
sideways into embeddings and speech rather than scaling up.

Why that mattered: your brief said to **validate the engine against a native
ternary model** (sensible — it removes model quality as a variable), then
**scale up to the largest one available**. But "the largest available" tops out
at 3.9B, so the scaling phase had nowhere to go.

**Your H100 credits dissolve this.** If no native ternary model above 4B exists,
we make one. That converts a dead end into the interesting part of the project.

---

## Format decision: Q2_0, and it's better than you think

You asked whether Q2_0 would be worse quality than ternary since it costs more
bits. **It's the opposite — Q2_0 is higher quality *and* it's the one that works
on your Jetson.** Here's why.

The bits-per-weight difference between these formats is **not** about value
precision. All of them store the same `-1/0/+1`. The difference is:

1. **Packing efficiency** — how tightly the codes are crammed into bytes
2. **Scale granularity** — how many weights share one scale

Number 2 is what drives quality, and it goes the *opposite* way from bit count:

| Format | bpw | Weights per scale | Quality |
|---|---|---|---|
| TQ1_0 | 1.69 | 256 | lower |
| TQ2_0 | 2.06 | 256 | lower |
| **Q2_0** | **2.25** | **64** | **higher** |

Q2_0 spends slightly more bytes to get **4× more scales**. More scales = each
scale fits its local group of weights better = less error.

I'd already verified this principle by accident. One of my unit tests
(`test_smaller_groups_reduce_error`) checks that shrinking the group size reduces
quantization error, specifically on weights with outliers. It passes. Q2_0 is
that finding baked into a file format.

**And the decisive practical point:** Q2_0 has working **CUDA** kernels, merged
July 2026. TQ1_0/TQ2_0 have **none** — a pull request to add them has been open
19 months. Your Jetson is a CUDA device. With TQ2_0 you'd silently fall back to
running on the CPU.

So Q2_0 wins on quality *and* on actually working. Decision made.

The people who created Q2_0 said as much in their PR: *"Why not use TQ1_0 /
TQ2_0? They support group size 256, our models are group size 128, and also
cpu-only and harder to accelerate on Metal/CUDA."*

---

## What fucina is, and how we're different

**What it is:** a from-scratch inference engine written in Zig (not a fork of
anything), one author, ~6 weeks old, 17 stars. It streams **mixture-of-experts**
models from disk with a memory budget, and supports ternary weights. It publishes
real numbers: a 142 GB model on a 64 GB Mac at ~24 GB peak memory.

**Why I flagged it:** your brief's central claim is that ternary + bounded
streaming has never shipped together. As written, that's false — fucina ships it.

**How we're genuinely different:**

| | fucina | us |
|---|---|---|
| Streams | MoE **experts** | whole **layers** |
| Model type | mixture-of-experts | dense (normal) |
| Language | Zig | Python / PyTorch |
| Format | TQ2_0 (no CUDA) | Q2_0 (CUDA works) |
| Your Jetson | ✗ format has no CUDA | ✓ |

You spotted the important one yourself: **its format doesn't work on your
hardware.** TQ2_0 has no CUDA kernels, so fucina's approach can't run on a
Jetson GPU regardless of how good the engine is.

The differences are real, but I'd rewrite the project pitch to be defensible:

> *"No CUDA-capable, dense, layer-granular ternary streaming engine exists.
> The one shipped ternary-streaming implementation is expert-granular, in a
> format with no GPU support, outside the Python ecosystem."*

That's honest and still worth building. Claiming "nobody has done ternary +
streaming" is not, and someone would call it out.

---

## What I measured so far

PTQ (naive rounding) versus one-layer-at-a-time distillation, on a small real
model. "Cosine" = how closely the quantized layer's output matches the original;
1.0 is perfect.

| layer | PTQ | distilled | gain |
|---|---|---|---|
| 0 | 0.7358 | **0.9482** | +0.21 |
| 1 | 0.7483 | **0.9697** | +0.22 |
| 2 | 0.8097 | 0.8442 | +0.03 |
| 3 | 0.9911 | 0.9975 | +0.01 |
| 4 | 0.9767 | 0.9960 | +0.02 |
| 5 | 0.9638 | 0.9931 | +0.03 |

**Reading:** PTQ badly damages the early layers (0.73–0.75). Distillation
recovers them to 0.95+. This is the evidence that your "PTQ is bad, do QAT"
instinct was right.

**Layer 2 is stubborn** — it barely improves. Some layers resist quantization,
probably due to a known pathology called "massive activations." The plan is to
detect those and leave them at full precision.

**Important caveat:** these are per-layer numbers on a 135M model. Errors
compound across depth — 60 layers at 0.95 each is *not* 0.95 overall. The number
that actually settles it is end-to-end text quality, which I have not measured.

---

## Compute reality check for the H100

Three tiers, honestly costed. "Tokens" = how much text you train on.

| Approach | Tokens needed | H100 time | Quality |
|---|---|---|---|
| Block distillation *(built)* | ~500 sequences | minutes | good |
| Layer distillation + real data | ~100M | hours | better |
| Full end-to-end QAT distillation | 10–100B | **days to months** | best |

The published recipes sit in tier 3 — Llama3-8B-1.58 used 100B tokens. For
scale: 100B tokens on an 8B model is roughly **139 days on a single H100**. So
"full QAT" in the literature sense is not something a modest credit balance buys.

**The sweet spot is tier 2**, and it's under-explored. Logit distillation from a
strong teacher converges far faster than raw pretraining, because the teacher
gives a much richer signal per token than "predict the next word." A few hours of
H100 on a 1–3B student is a genuinely reasonable bet.

My recommendation: run tier 1 (free, already built), measure, then spend H100
time on tier 2 only where tier 1 leaves damage.

---

## Model choice

You said you don't care and to pick. My pick, with reasoning:

**For validating the engine: `microsoft/bitnet-b1.58-2B-4T`.**
- Best-trained ternary model in existence (4 trillion tokens), so if output looks
  wrong, it's *our bug*, not the model being weak — which is the entire point of
  a reference model
- Ships packed, and also ships a full-precision sibling for comparison
- Runs in stock HuggingFace, so an unstreamed reference is one line of code

**For the scale test: `SpectraSuite/TriLM_3.9B_Unpacked`** — the largest native
ternary model. Note it's published *unpacked* in fp16 (~8 GB), so we pack it
ourselves. Fine; that exercises our packer.

**For the distillation experiment: a student we create**, teacher TBD. This is
where the H100 credits go and where the actual novelty is.

I'm dropping Falcon-E-3B — it's good, but BitNet-2B-4T's training budget makes it
the better reference, and TriLM covers the size axis.

---

## What changes in the code

| Component | Status |
|---|---|
| Residency manager (streaming, budget, prefetch) | **keep** — model-agnostic |
| Per-layer sharding | **keep** |
| Distillation trainer (`reconstruct.py`) | **keep**, extend for H100 |
| My invented packing (`2bit`, `trit5`) | **replace** with Q2_0 |
| Dequantize-to-bf16 in `TernaryLinear` | **replace** — should stay packed |
| Gemma 4 31B as target | **park** — no ternary reference exists for it |

The streaming engine survives intact. The quantization half gets rebuilt on real
formats.

Worth noting: my `trit5` packing mode turned out to be the same scheme as
llama.cpp's TQ1_0 — base-3, 5 values per byte, since 3⁵ = 243 fits in 256. Right
idea, but there's no reason to keep a private version of a standard format.

---

## Open questions for you

1. **What's the actual end goal?** "Run a good model on the Jetson" and "publish
   something novel" point at different priorities. If it's the former, we may not
   need the H100 at all.
2. **How much H100 budget, concretely?** Tier 2 vs tier 3 is the difference
   between hours and months.
3. **Does Gemma 4 come back later?** It's parked, not deleted. Making a good
   ternary Gemma 4 is a legitimate goal — it's just a research project, not a
   weekend.
