# Recovering quality from an already-quantized LLM

*Survey commissioned 2026-08-24 for `google/gemma-4-31B` at w4g128. Sources are
arXiv abstracts read directly unless marked otherwise.*

> **Editorial note — one conclusion in this report is withdrawn.**
> Section 3 originally argued that our 3-bit result is "at or better than the
> published band" by comparing our +13.9% relative perplexity penalty against
> 7–8B models at +17% to +34%. **That comparison does not hold**, for two
> reasons given below in section 3. The corrected conclusion is that we *cannot
> currently say* whether our 3-bit path is good or bad. The rest of the report
> stands.

---

## 1. The three interventions most likely to help

### 1.1 Distil from the FP teacher's logits — highest expected gain

**Quantization-Aware Healing** (arXiv:2608.20953, 21 Aug 2026, released as
Hypernova-60B). On a GPT-OSS 120B→60B→MXFP4 pipeline the 4-bit student
**matches or beats its bfloat16 source on 7 of 9 benchmarks**. Two findings
transfer:

- Against a matched QAT baseline it reached comparable peak **~7× faster and
  stayed stable under continued training**, without hand-tuned early stopping.
  Their QAT baseline "converged slowly and collapsed past its peak."
- QAH **does train the 4-bit student** — it is not a frozen-weight method. What
  it changes is the *target*: KL against the uncompressed teacher's
  distribution rather than cross-entropy against hard labels.

This is the family our KD-LoRA run belongs to, and it argues for eventually
widening from adapters to updating the quantized weights themselves.

### 1.2 Targeted error compensators on the layers that actually break

**SPEAR** (arXiv:2606.11244, Jun 2026) keeps the 4-bit model as-is and inserts
**lightweight error compensators modulated by per-token gates**, at layers
chosen by a CKA-guided entropy-aware diagnostic. Recovers **56–75% of the
W4→FP16 perplexity gap at <1% model memory**.

This maps onto our strongest existing measurement: per-layer ternarization
damage spans 0.034 to 0.975 KL, and **layer 59 alone flips 26.7% of tokens**.
Our damage is already known to be concentrated; this is the published method
for spending a tiny parameter budget exactly there.

Caveat from their abstract: the W4 gap they target is largest "particularly for
smaller models." At 31B the gap is smaller, so scale expectations down.

### 1.3 Bidirectional incoherence processing

**KronQ** (arXiv:2607.07964, Jul 2026) makes a specific criticism of
GPTQ-family objectives, including ours: GPTQ builds its objective from **input
activation statistics only**, implicitly assuming all output channels
contribute equally to layer reconstruction. Under a Kronecker-factored Hessian
the loss depends on **both activation and gradient covariances**, so KronQ
extends the rotation to the output dimension using gradient covariance.

We already get 36–41% block-reconstruction error reduction from real Hessians
over RTN. KronQ's claim is that we are using half the Hessian. This modifies
the quantizer we already run rather than replacing it.

## 2. What to skip

**Bolt-on low-rank correction on top of GPTQ.** The systematic cross-method
study (arXiv:2507.17417) finds that combining low-rank compensation with GPTQ
**"can occasionally outperform GPTQ alone"** — inconsistent, not reliable.
GlowQ (arXiv:2603.25385, listing only) gives the magnitude: 0.17 perplexity and
0.42 percentage points at 4-bit. The reason is structural: LQER/QERA-style
corrections are benchmarked against RTN, where large residual error remains.
Sequential Hessian-based reconstruction has already absorbed most of it.

**Rotation as a bolt-on.** Across 31 rotation papers surveyed (QuaRot,
SpinQuant, DuQuant and 2026 successors including DuQuant++ arXiv:2604.17789,
ConQuR arXiv:2605.10793, ReSpinQuant arXiv:2604.11080, TORQ arXiv:2605.19561),
**every one applies rotation before quantization**. Once weights are on a
scalar grid the information is gone. It does compose as a *pre-step* with
sequential block reconstruction — QuaRot and SpinQuant are both rotate-then-GPTQ
— so adopting it means re-running our quantizer, not rewriting it.

**MLP channel pruning.** Already settled by our own data: 6.8% of parameters
removed flips 10.07% of tokens.

**A note on every number in this report:** no paper surveyed reports flip rate
or KL against the teacher. Their recovery figures are accuracy- and
perplexity-based, so they are *not* directly comparable to our primary
criterion. Treat them as directional only.

## 3. Is our 3-bit cliff real? — CONCLUSION WITHDRAWN

The original analysis compared our relative penalty against published 7–8B
results:

| Model | W4A16 | W3A16 | 3-bit penalty vs 4-bit |
|---|---|---|---|
| LLaMA-3-8B | 5.62 | 7.55 | +34.3% |
| Qwen-2.5-7B | 6.38 | 7.46 | +16.9% |
| Mistral-7B | 5.73 | 6.88 | +20.1% |
| **Gemma-4-31B (ours)** | **5.3407** | — | **+13.9%** |

*(RDQ, arXiv:2607.10137 — listing-level only, unverified.)*

**This comparison is invalid, and it fails in the direction that flatters us.**

1. **Scale mismatch.** Larger models are generally more compressible — they
   carry more redundancy. The original report stated this and then used it as
   *support* for the conclusion, which inverts the logic. If a 31B model should
   compress better than an 8B one, then landing at +13.9% against their +17% to
   +34% is roughly what scale alone predicts. It is not evidence that our
   method is good, and it is entirely consistent with our 3-bit path
   underperforming what a 31B model should achieve.

2. **Relative perplexity penalty is model-dependent.** The ratio depends on the
   baseline, the tokenizer, the training corpus, and the architecture. This
   project's own standing rule is that perplexity is **not comparable across
   tokenizers** — bits-per-byte is. Comparing perplexity *ratios* across four
   different models with four different tokenizers inherits that problem rather
   than escaping it.

**Corrected conclusion: we do not know whether our 3-bit is good.** The survey
found **no paper reporting 3-bit weight-only results for a 30B-class dense
model with a full baseline**. That is a genuine gap in the literature, not an
omission in the search. Establishing whether our 3-bit underperforms requires
either a same-size published comparison or an internal one — running a
published method end-to-end on Gemma-4-31B ourselves.

**What does survive from this section:** our 2-bit failure mode is a known one.
KronQ reports that on LLaMA-3-70B 2-bit weight-only, **GPTQ and GPTAQ diverge
or produce degenerate quantizations, exceeding 2000 perplexity**, while KronQ
reaches **7.93**. Our +429% is the textbook GPTQ-class 2-bit failure, not
evidence that 2 bits is impossible. Headroom at ≤3 bits lies in **changing the
codebook** (vector/lattice/trellis: QuIP# 2402.04396, AQLM 2401.06118, QTIP
2406.11235) and **adding rotation**, not in tuning the scalar quantizer.

## 4. Method table

Verified = arXiv abstract page read directly. Listing = API search summary only.

| Method | arXiv | Date | Reported gain | Bits | Model size | Impl. | Status |
|---|---|---|---|---|---|---|---|
| **QAH / Hypernova** | 2608.20953 | Aug 2026 | Beats bf16 source on 7/9 benchmarks; 7× faster than QAT | MXFP4 | 120B→60B | Open weights | Verified |
| **SPEAR** | 2606.11244 | Jun 2026 | 56–75% of W4→FP16 ppl gap; <1% memory | W4 | n/s | Unknown | Verified |
| **Recover-LoRA** | 2606.04238 | Jun 2026 | 80–95% recovery on 9/12 bench; data-free | W2 gate/up + W4 | 4B–20B | Unknown | Verified |
| **KronQ** | 2607.07964 | Jul 2026 | 7.93 ppl vs GPTQ >2000 | 2-bit W-only | LLaMA-3-70B | Unknown | Verified |
| **QERA** | 2410.06040 | Oct 2024 | +2.97% vs ZeroQuant-V2; −0.28 ppl vs LQER | 4-bit / 2-bit | Llama-3.1-70B | Yes | Verified |
| Comprehensive PTQ eval | 2507.17417 | Jul 2025 | Low-rank+GPTQ only "occasionally" beats GPTQ | INT4/FP4 | n/s | — | Verified |
| GlowQ | 2603.25385 | Mar 2026 | 0.17 ppl, 0.42pp acc | 4-bit | n/s | Unknown | Listing |
| AdaMX | 2608.03867 | Aug 2026 | Removes 83%/82% of MXFP4 loss, data-free | MXFP4 | 3B–70B | Unknown | Listing |
| DuQuant++ | 2604.17789 | Apr 2026 | SOTA MXFP4 W4A4; halves rotation cost | W4A4 | LLaMA-3 | Unknown | Listing |
| ConQuR | 2605.10793 | May 2026 | Closed-form Procrustes; no storage overhead | W4A4/KV4 | 3B–70B | Unknown | Listing |
| Quant Experts | 2602.24059 | Feb 2026 | MoE of low-rank adapters | low-bit | 2B–70B | Unknown | Listing |
| RILQ | 2412.01129 | Dec 2024 | Rank-insensitive 2-bit compensation | 2-bit | LLaMA-2/3 | Unknown | Listing |
| LQER | 2402.02446 | Feb 2024 | "Nearly-lossless" W4A8 | W4A8 | — | Yes | Title only |
| CALDERA | 2405.18886 | May 2024 | Low-rank + low-precision decomposition | sub-4 | — | Yes | Title only |
| LoftQ | 2310.08659 | Oct 2023 | Superseded by QERA | 2–4 bit | — | Yes (PEFT) | Title only |

## 5. Recommended order of work

1. **Finish the uncontaminated KD-LoRA rerun**, targeting teacher logits rather
   than hard labels, per QAH.
2. **Add output-side incoherence processing** to the sequential quantizer
   (KronQ). Reuses the Hessian infrastructure we already have.
3. **Prototype SPEAR-style gated compensators on layer 59 and the top-KL layers
   only.** Our per-layer damage profile already identifies the targets.
4. Fit QERA's closed-form residual correction **only as a cheap ablation** —
   expect ~0.2 perplexity, and check it against flip rate before believing it.
5. If ≤3 bits becomes a requirement, that is a **codebook change** plus
   rotation, not tuning on the current path.
6. **Establish a same-scale 3-bit reference**, since section 3 shows we have
   none. Either find a 30B-class published result or run a published method on
   Gemma-4-31B ourselves.
