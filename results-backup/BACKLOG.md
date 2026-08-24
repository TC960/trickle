# Backlog & methodology log

Running record of what we've learned, what's wrong with our current approach, and
what to try next. Added to rather than rewritten, so the reasoning stays visible.

---

## Part 1 — What we now know is wrong with our evaluation

These are settled, with sources. Don't relitigate them; design around them.

### Perplexity is insufficient on its own
Perplexity is the inverse geometric mean of token probabilities, so losses on
some tokens **cancel against gains on others**. Measured consequence
([Accuracy is Not All You Need](https://arxiv.org/abs/2509.09141), NeurIPS 2024,
Table 19): on Llama2-13b-chat, WikiText-2 perplexity stayed at **exactly 5.70**
while greedy-token agreement fell **61.3% → 21.5%**. A 2.4% perplexity spread
across quantization schemes spanned a **~250× KL range**.

**What we do instead:** flip rate (argmax disagreement vs the original model), KL
divergence, and NLL stratified by token frequency. Implemented in `deep_eval.py`.

### WikiText-2 is a bad dataset for *this* problem
It reads only **~8% of the embedding table** (19,138 of Qwen's 248,320 rows) and
**0.21% of non-ASCII rows**. So it structurally cannot evaluate embedding or
vocabulary compression — which is exactly what we were using it for. That is
almost certainly why compressing embeddings looked *free or beneficial*: the
benchmark never touches most of what we deleted.

Corroboration: [Kurtic et al.](https://arxiv.org/abs/2411.02355) (ACL 2025) ran
500,000+ compression evaluations on Llama-3.1 and used perplexity **zero times**.

### Comparing across models makes things worse, not better
Perplexity is **not comparable across tokenizers** — different vocabularies
segment text differently, so the per-token quantity isn't the same thing. Only
within-model deltas are valid. **Bits-per-byte** normalizes by raw UTF-8 bytes
and *is* comparable; use it whenever tokenization changes (vocab trimming, or
cross-model claims).

### Where this leaves the evaluation stack
1. **flip rate** — primary. What a user actually experiences.
2. **KL divergence** — distribution shift perplexity hides.
3. **Downstream tasks** — MMLU / HellaSwag / ARC-C / **GSM8K**. GSM8K matters
   most: generative arithmetic, so one corrupted token kills the answer.
4. **bits-per-byte** — for anything that changes tokenization.
5. perplexity — kept only for comparability with published work.

---

## Part 2 — The reframing worth taking seriously

**Domain specialization instead of general compression.**

Current framing: "shrink the model while preserving general capability." That is
the hardest possible version of the problem, and it's why ternary keeps failing.

Better framing: **"build a model that is small *because* it only does what I
need."** If the deployment never needs organic chemistry or Kazakh, capacity
spent on them is pure waste — and unlike quantization, deleting it costs nothing
on the tasks you care about.

This changes the evaluation too, and for the better: you evaluate on **your
target domain**, not on WikiText. That sidesteps most of Part 1's problems,
because a domain eval actually exercises the parts of the model you kept.

Three compression axes, all composable:

| axis | what it removes | evidence so far |
|---|---|---|
| **vocabulary trimming** | tokens never used in your domain | measured: **~80% of vocab unused** on English+code; 2.2 GB (Gemma) / 4.2 GB (Qwen) removable |
| **attention head pruning** | heads that don't fire on your domain | correct terminology; well-studied; not yet tried here |
| **depth / layer pruning** | whole transformer layers | ShortGPT, Shortened-LLaMA; not yet tried here |

"Attention head pruning" is the right term. Related: structured pruning,
domain-adaptive pruning, task-specific distillation.

**Why this is likely to work better than ternary:** quantization degrades
*everything a little*; pruning degrades *specific things a lot* while leaving the
rest intact. If you can name what you don't need, pruning is strictly the better
trade.

---

## Part 3 — Backlog

### A. A/B ladder on the best approach (highest priority)
Build up from the full model, one change at a time, each measured against the
previous rung with flip rate + domain eval:

1. bf16 full model — control
2. + best weight quantization (int8 is free at +0.49%; int4/nf4 costs +5% on Gemma)
3. + vocabulary trimming to the target domain
4. + attention head pruning
5. + layer/depth pruning
6. + KV cache quantization (int8/FP8, **not** sub-4-bit — see below)

Every rung keeps the previous one as its control. Never change two things at once.

### B. Domain-specific pruning — the new idea, needs design
Open questions to answer before building:
- How do you *identify* domain-irrelevant heads? (activation statistics on
  domain vs non-domain text; attention entropy; gradient/Fisher attribution)
- How much can you prune before the model breaks, and is the curve sharp or gradual?
- Does it compose with quantization or do the losses compound?
- What's the right domain eval? Probably a held-out slice of the target domain
  plus GSM8K as a canary for general reasoning damage.

### C. Broad quantization-method research
Agent dispatched. Specifically mapping the middle ground between PTQ and full
QAT: learnable-scales-only, LoRA-on-quantized-base (QA-LoRA / LoftQ / LQ-LoRA /
ApiQ / EfficientQAT), rotation methods (QuaRot / SpinQuant / QuIP# / OptR),
short-run "healing", and sensitivity-based mixed precision.

### D. Full QAT, if PTQ is conclusively dead
Budget exists (~36 GPU-hours). Precedent: Falcon3-10B-1.58bit and
Llama3-8B-1.58-100B-tokens are both **QAT conversions at 8–10B**, which is direct
evidence that the "ternary fails above 2–3B" folk claim is a **PTQ** ceiling, not
a fundamental one. The cheapest real QAT is already implemented:
`distill_e2e.py` — freeze the ternary codes, train only the ~230M per-group
scales against teacher logit KL.

### E. Things already known, don't rediscover
- `transformers` v5 **removed** the v4 quantized-cache classes, and its
  `QuantizedCache` **hard-raises** on both our models (sliding and linear
  attention unsupported). KV quantization must go through **vLLM** (TurboQuant,
  both models supported) or **llama.cpp** (`-ctk q8_0 -ctv q8_0 -fa`; quantized V
  *requires* flash attention).
- Pair ternary/2-bit weights with **8-bit KV, not 2-bit** — no evidence supports
  stacking both aggressively. Closest precedent (RCP, W2A4KV4, +2.84 ppl)
  required full QAT.
- **Nothing in the model is unstreamable.** Verified: `lm_head` computes exact
  logits in vocab chunks (8× less resident, 1.07× compute). KV streams alongside
  its layer. The real currency is **bytes-read-per-token**, not resident bytes.
- At ternary weights, the bf16 embedding table is **32.5% of the checkpoint** and
  ~2.6 GB of per-token streaming bandwidth — about 20 ternary layers' worth.
- Merge-closure is mandatory when trimming vocab: frequency pruning alone
  produced 19,269 unreachable tokens on Gemma and would have silently broken the
  tokenizer.

### F. Open niche worth a writeup
**Nobody has stratified perplexity by token rarity for compressed models.** Our
harness now does exactly this. If the finding is clean, it's publishable and it
directly resolves the "does compression eat the tail?" dispute — which is
currently contested ([Asymmetric Harms](https://arxiv.org/abs/2608.19670),
posted 2026-08-20, argues compression hurts *common* knowledge more).
