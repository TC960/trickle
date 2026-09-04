# Research threads — findings and re-dispatch briefs

Six research agents ran during the 2026-08-21/22 session. Their findings exist
**only in the chat transcript**, so the load-bearing conclusions are captured
here. Thread 6 was killed mid-flight; its brief is ready to re-run.

Nobody needs to read the chat. Read this.

---

## Thread 1 — llama.cpp ternary support ✅ COMPLETE

- **TQ1_0 (1.69 bpw) and TQ2_0 (2.06 bpw) are alive**, not deprecated, and
  actively gaining backends (Metal merged 2026-08-13, Vulkan 2026-08-12).
- **CUDA has ZERO TQ support.** A TQ2_0 model on a CUDA build silently falls
  back to CPU. The enabling PR (#11183) has been open **19 months**.
- **`Q2_0` is the CUDA-viable ternary format** — 2.25 bpw, group size **64**
  (vs TQ's 256), merged CUDA MMQ+MMVQ July 2026. More scales = better quality,
  so it is *higher* fidelity than TQ2_0 despite costing more bits.
- PR #25294 (MoE expert streaming, O_DIRECT + LRU) is **open, unmerged, stalled** —
  zero reviews in 6 weeks, merge conflicts. Its companion issue was auto-closed
  as stale.
- **The project's original premise was false as stated.**
  [fucina](https://github.com/matteo-grella/fucina) already ships ternary +
  bounded streaming together, in TQ2_0's own wire format. Scoped honestly: it is
  expert-granular MoE, in Zig, in a format with no CUDA support. Defensible
  rewording: *"no CUDA-capable, dense, layer-granular ternary streaming engine
  exists."*

## Thread 2 — natively-ternary checkpoints & bitnet.cpp ✅ COMPLETE

- **No natively-ternary model above ~4B exists.** Largest is
  `SpectraSuite/TriLM_3.9B` (fp16 unpacked only). Best-trained is
  `microsoft/bitnet-b1.58-2B-4T` (4T tokens). Most usable is
  `tiiuae/Falcon-E-3B-Instruct` (packed, 32k ctx).
- **Everything ≥7B is a QAT *conversion***: Falcon3-10B-1.58bit,
  Llama3-8B-1.58-100B-tokens. **This is the direct evidence that "ternary fails
  above 2–3B" is a PTQ ceiling, not a fundamental one.**
- **bitnet.cpp produces garbage on Jetson Orin specifically** (issue #585 names
  the device). GGUF packs I2_S at 128/32; the ARM kernel unpacks at 64/16.
  Open since Feb 2026, unmerged, zero releases ever tagged on a 40k-star repo.
- oLLM streams layers but is **fp16/bf16 only** — "No quantization is used."
- The one published I2_S doc (PR #507) has the **sign mapping backwards**.
  Trust the code, not the doc.

## Thread 3 — structural compression ⚠️ PARTIAL (died to laptop sleep)

- Confirmed `Qwen/Qwen3.8-27B` is real, released 2026-08-14.
- Everything else in this brief was superseded by Thread 5.

## Thread 4 — the embedding-compression anomaly ✅ COMPLETE, and it overturned a finding

- **[Accuracy is Not All You Need](https://arxiv.org/abs/2509.09141) (NeurIPS
  2024, Table 19): WikiText-2 perplexity stayed at exactly 5.70 while
  greedy-token agreement collapsed 61.3% → 21.5%.** A 2.4% perplexity spread
  spanned a ~250× KL range. **Perplexity alone cannot detect behavioural damage.**
- **Our "embedding compression improves quality" result is probably an
  artifact.** WikiText-2 reads only **7.71%** of Qwen's embedding rows and
  **0.21%** of non-ASCII rows. The benchmark never touches ~92% of what we
  deleted.
- The closest published comparable (CARVQ) reports int4 embedding-only at
  **+1.4%**; we measured −0.49%. Opposite sign.
- [Asymmetric Harms](https://arxiv.org/abs/2608.19670) (2026-08-20) argues
  compression damages **common** knowledge more than rare — contradicting the
  denoising story outright.
- Kurtic et al. (ACL 2025) ran 500,000+ compression evals and used perplexity
  **zero times**.
- **Open niche:** nobody has stratified perplexity by token rarity for
  compressed models. `deep_eval.py` now does exactly this.

## Thread 5 — vocab trimming & KV cache ✅ COMPLETE

- **`transformers` v5 removed the v4 quantized-cache classes**, and its
  `QuantizedCache` **hard-raises** on both our models (sliding + linear attention
  unsupported). KV quantization must go via **vLLM** (TurboQuant; both models
  supported) or **llama.cpp** (`-ctk q8_0 -ctv q8_0 -fa`; quantized V *requires*
  flash attention).
- TurboQuant is real (arXiv 2504.19874, in vLLM): `k8v4` = 2.6× at +1.17% ppl.
  **Sharp cliff below 4 bits.** vLLM's own source disputes its novelty vs
  DRIVE/EDEN, and two 2026 rebuttals exist.
- **Pair ternary weights with 8-bit KV, not 2-bit.** Closest precedent for
  stacking (RCP, W2A4KV4, +2.84 ppl) required **full QAT**.
- Gemma-4 KV = "0.78 GiB constant + 80 KiB/token" (50 sliding layers capped at
  window 1024; only 10 full layers grow). 1.41 GiB @8K, 3.28 GiB @32K.
- **Vocab trimming traps:** `resize_token_embeddings` only truncates the tail,
  and Gemma has **187,029 merge-order violations** — tail truncation breaks the
  tokenizer. Gemma's 256 `<0xNN>` byte tokens must ALL survive (`fuse_unk` makes
  a miss unrecoverable). Multimodal tokens sit at IDs 255999–258884. Free win:
  6,227 `<unusedNNN>` rows (~64 MB).
- Best method: **leaf-based pruning** (arXiv 2512.03989, EACL 2026) — the only
  one that provably never creates unreachable tokens. 62.5% removed from
  Llama-3.1-8B with no degradation.
  `git clone https://github.com/taidopurason/tokenizer-extension`
- **No published PPL/MMLU numbers exist for post-hoc vocab trimming of a 27B+
  decoder.** Our regime is unmeasured — we'd be generating the evidence.

---

## Thread 6 — PTQ→QAT middle ground ❌ KILLED MID-FLIGHT, re-run this

**Why it matters:** our best ternary result is perplexity 97.55 vs a 5.19
baseline. Full QAT needs ~375 GB of optimizer state. The whole question is what
lives between those two poles.

**Note for whoever re-dispatches:** the session's WebSearch budget was exhausted
(200/200). Instruct the agent to use the **arXiv API, GitHub API, HF API, PyPI
JSON and WebFetch** instead — those work.

**Re-dispatch prompt:**

> Map the middle ground between post-training quantization and full
> quantization-aware training, current as of August 2026, ranked by (quality at
> ~2 bits) / (compute required). I need methods I can RUN, with public code.
> WebSearch is exhausted — use the arXiv API, GitHub API, HF API and WebFetch.
>
> Context: google/gemma-4-31B, bf16 wikitext-2 perplexity 5.1876. Measured:
> int8 = 5.2132 (+0.49%); nf4 = 5.4484 (+5.03%); naive ternary PTQ = 155,493,424;
> teacher-forced block reconstruction = 222,296; sequential drift-aware block
> reconstruction = 97.55. Hardware: 3x A100 80GB across two nodes, ~36 GPU-hours.
>
> Cover: (1) methods training only a small parameter subset end-to-end —
> learnable scales/clipping, QA-LoRA, LoftQ, LQ-LoRA, ApiQ, EfficientQAT stage 2,
> bias/norm-only; give trainable-parameter fraction, 2-bit quality, repo.
> (2) training-free PTQ improvements — QuaRot, SpinQuant, QuIP#, OSCAR, OptR,
> GPTQ/OBQ successors, AWQ; what is the best published 2-bit number on a ~30B
> dense model? (3) "healing" runs — how few tokens recover most quality?
> (4) sensitivity-based mixed precision — how to CHOOSE which layers stay at
> higher bits (Hessian/Fisher/outlier metrics), and what does "2-bit except 10%
> of layers at 4-bit" achieve? (5) Is "ternary fails above 2-3B" a PTQ limit or
> fundamental? Find data. (6) anything new in 2026 that supersedes this taxonomy.
>
> Give install commands and hyperparameters. Flag paper-only vs working code.
> Prioritise what fits in 36 A100-hours.

---

## Further threads worth dispatching

**A. Domain-specific attention-head pruning.** How to identify heads irrelevant
to a target domain (activation statistics, attention entropy, Fisher
attribution); how far you can prune before collapse; whether it composes with
quantization. This is the most promising unexplored axis — see `BACKLOG.md` Part 2.

**B. Evaluation design for compressed models.** Given perplexity is discredited
here, what is the minimal eval suite that reliably detects compression damage?
Ground it in Kurtic et al. (ACL 2025) and ACBench (ICML 2025).

**C. Does the Jetson toolchain accept a hand-trimmed tokenizer?** If deployment
is llama.cpp/GGUF, test this BEFORE investing further in trimming — it could
invalidate the whole approach.
