# Claude Code Prompt — Post-Handoff Next Session

> Companion to `CLAUDE.md`. **Read `CLAUDE.md` first and in full.** It carries
> project state, 17 known bugs, standing methodology rules, and measured results.
> This document does not repeat them — it defines what to do next and what to
> watch out for.

---

## POSTURE

Be adversarial. This project's documented failure mode is **optimizing a metric
before checking which constraint binds** — perplexity before flip rate, bit-width
before throughput. Before executing any task below, state which constraint it
addresses and what would have to be true for that constraint to be the binding one.

Per `CLAUDE.md` Part 8: the user's tentative hypotheses have repeatedly beaten
confident assertions from the model. **Verify, don't evaluate from intuition.**
If you are about to assert something about model architecture, checkpoint
availability, or what does or doesn't exist — search first.

This session added a sharper version of that same lesson: **a mature,
widely-used eval library (`lm_eval`) produced two confidently-plausible wrong
numbers in a row** (43.6%, then 50.3% GSM8K for Qwen3.6-35B-A3B) before the
real number (94-95%) surfaced. Both wrong numbers came from the library's
own defaults/conventions colliding with this specific model's behavior, not
from anything exotic. Before trusting any eval number — especially a
surprising one, in either direction — pull raw generated samples and read
them. An aggregate score from well-tested code is not automatically ground
truth.

All standing rules in `CLAUDE.md` Part 1 apply without exception. Especially:
every treatment needs a control; single-seed differences under ~5% are not
results; do not lead with perplexity; nothing heavy runs locally.

---

## STATE SUMMARY (do not re-derive, this is settled)

- Dense 31B streaming is **memory-feasible** (3.16 GB resident, bit-exact at
  every budget) and **speed-infeasible** (15.333 GB read/token → 5–51 s/token at
  Jetson storage speeds).
- **The MoE pivot to Qwen3.6-35B-A3B is resolved and supported.** GSM8K
  94-95% (n=1000) beats Gemma 4 31B's 86.0 (n=200) — not just "doesn't lose
  badly," it wins. Memory-constrained decode throughput (plain OS page-cache
  eviction, no custom residency logic) is 1.52 tok/s at a Jetson-realistic
  4GB budget — a real 90% drop from unconstrained, but still categorically
  more usable than dense Gemma's 5-51 s/token. Full detail and the three
  eval-harness bugs it took to get a trustworthy capability number: `CLAUDE.md`
  Parts 3 and 6.
- **Qwen3.6-35B-A3B quantizes far better than Gemma did.** Q4_K_M (the actual
  4-bit deployment target) vs Q8_0 (bf16 proxy) on GSM8K: 94.6%/92.3% vs
  95.3%/94.0% — costs ~1-2 points, not Gemma's ~6-point dense-path hit.
- **All three target-use-case benchmarks (Priority 3) are done and look
  good**: HumanEval 61.0%, MBPP 62.6%, 3/3 on a hand-built browser-agent
  job-application task set, and ~83% average across 6 technical MMLU
  subjects (custom task — GPQA itself is gated on HF, not resolved).
- 4-bit is the practical floor for the *dense* Gemma path. 3-bit +17%, 2-bit
  +429%, ternary +1,780% ppl.
- KD-LoRA at r=32 is **dead** — the real control killed it. Do not revive without
  a new reason. (Note: this was PTQ-recovery, not the same thing as the
  distillation question raised this session — see Priority 3 below.)
- MLP channel pruning is ~25× worse per unit compression than quantization.
- LRU gives a **0% hit rate** on cyclic dense-layer traversal. Fixed partition
  would give a guaranteed N/60. For MoE *experts* specifically this is a
  different problem — see Priority 1 below, don't conflate the two.
- Most downstream arms are indistinguishable at `--limit 200`; this session's
  Qwen numbers ran at `--limit 1000` specifically to clear that bar.
- Infrastructure: `ternary-h200` is currently stopped and would not restart
  across two attempts this session (80+ min hangs, provider-side, not fixed
  from the CLI). All Qwen work ran on an unrelated box. Try starting it again
  before assuming it's dead; not urgent since nothing there is blocking
  anything below. Full detail: `CLAUDE.md` Part 7.

---

# PRIORITY 1 — MOE EXPERT RESIDENCY MANAGER (the engineering half of the pivot)

The pivot decision is made; the pivot isn't *built* yet. This session measured
throughput under a memory budget using **plain mmap with zero eviction
policy** — that number (1.52 tok/s at 4GB) is the floor, not the ceiling.

## 1.1 — Build a real residency manager for MoE experts

This is a genuinely different problem from the dense-layer fixed-partition
idea (Priority 2 below): dense-layer access is strictly cyclic with no reuse
within a pass (LRU's pathological case, fixed-partition is the clean fix).
Expert access is skewed-but-drifting: published measurements on the same
"A3B" family (Qwen3-30B-A3B) show real popularity skew concentrated by task
domain, but hot-expert-set intersection across consecutive micro-batches is
below 0.5 — meaning **an adaptive cache (LRU/SLRU), not a static fixed
partition, is the right structure for experts.** A closed llama.cpp feature
request (unimplemented) proposed exactly this — SLRU, GPU+RAM two-tier —
with a working PoC showing steady-state 12-14 tok/s / ~98-100% hit rate on a
different model, vs 0.5-1 tok/s with no cache. That's the ceiling this
project should be aiming at, not 1.52 tok/s.

Also check for and correctly exclude the **shared expert** (fires every
token, unevictable) from any hit-rate accounting, or the numbers will be
flattering and wrong.

## 1.2 — Verify the expert-skew hypothesis on the actual model, not just the analog

The Qwen3-30B-A3B skew data is a proxy, not a direct measurement on
Qwen3.6-35B-A3B. Cheapest path: instrument actual expert-selection IDs during
the capability eval's generations (the GSM8K run already produced 1000 real
generations through this exact model — router decisions during that run may
be recoverable without a fresh pass, check `llama-server`'s verbose/debug
output options first before assuming a rerun is required).

## 1.3 — Gated DeltaNet / linear-attention streaming support

`CLAUDE.md` Part 4 confirms Qwen3.6-35B-A3B is 10 cycles of (3 Gated DeltaNet
linear-attention blocks + 1 full attention block) — this project's custom
streaming code (`airllm_ternary/loader.py`, `model.py`) has never implemented
this. Two paths, very different cost:

- Extend the custom streaming engine to a novel attention kernel: multi-day+,
  and MoE also forces shard granularity from per-layer to per-expert, a
  bigger change to `shard.py` than the new attention op alone.
- Use what already exists: `unsloth/Qwen3.6-35B-A3B-GGUF` and `transformers`
  (via `trust_remote_code`) both already implement this architecture. This
  session's whole pipeline (llama.cpp + GGUF) sidestepped this problem
  entirely rather than solving it — worth deciding explicitly whether the
  project continues on llama.cpp for the MoE path (cheap, proven this
  session) or still wants the custom engine for some other reason (e.g.
  tighter Jetson-specific control that llama.cpp's general-purpose mmap
  path doesn't give you — Priority 1.1 above is exactly the place that
  distinction would start to matter).

---

# PRIORITY 2 — FIXED-PARTITION RESIDENCY FOR DENSE LAYERS (Gemma, unchanged)

Ranked #2 in `CLAUDE.md`, independent of Priority 1 above — different access
pattern, different fix. LRU's 0% hit rate on cyclic traversal is pathological,
not marginal. A fixed partition pinning N layers gives a guaranteed N/60.
Implement in `loader.py`'s `ResidencyManager`.

Report hit rate and GB/token as a function of N, and combine with the vocab
trimming result below — Part 3 notes the pinned embedding table is what sets the
3.16 GB resident floor, and trimming it to 51,269 tokens (5.11×, 1.41 GB → ~280 MB)
converts directly into ~4 more resident layers.

**Caveat that must be respected:** vocab trimming has never been evaluated for
quality, because wikitext reads only 7.7% of embedding rows and structurally
cannot judge it. An embedding-compression result on Qwen was already retracted
for exactly this reason. **Do not report a vocab-trimming quality number from any
wikitext-based eval.** Build a domain eval that exercises the vocabulary — code
tokens, tool-call JSON, technical English — or report the memory win with quality
explicitly marked unevaluated.

---

# PRIORITY 3 — TARGET-USE-CASE BENCHMARKS AND DISTILLATION

Raised directly by the user: GSM8K/MMLU are compression-damage canaries, not
measures of what this deployment is actually for. **All three real target
uses are now measured:**

1. **Technical Q&A** ("ask it things while learning") — **done.** GPQA (the
   standard "harder than MMLU" pick) is gated on HuggingFace, not resolved
   this session — worth revisiting with an approved HF account. Built a
   proxy instead: 6 technical MMLU subjects (college CS/physics/math,
   electrical engineering, computer security, machine learning), full test
   splits, via a custom task (`generate_until`, not `loglikelihood` — MMLU's
   stock loglikelihood form hits the same logprobs incompatibility as MMLU
   proper, and its own `mmlu_generative` group stops on a bare `"\n"` and
   reads only the first line, which breaks the same way gsm8k's stop
   condition did). Result: **~83% average, 73-97% range** across subjects.
   This eval had its own bug too — first attempt scored 0% flat, caused by a
   regex extraction filter that silently returned empty matches; see
   `CLAUDE.md` Part 5 bug 18. Fixed and reconfirmed against known-correct
   samples before trusting the full run.
2. **Coding** — **done.** HumanEval 61.0% (n=164), MBPP 62.6% (n=500) on
   Q8_0. See `CLAUDE.md` Part 3. SWE-bench/LiveCodeBench remain open if
   multi-file/agentic coding signal is wanted beyond single-function
   completion.
3. **Agentic browser use** — **done, positive signal.** Real academic
   benchmarks (WebArena, OSWorld, WorkArena) turned out to need self-hosted
   Docker sites / full desktop VMs / gated access — all ruled out as
   multi-day builds. Built a small hand-authored proxy instead: 3 local
   mock job-application pages (simple form, 2-step wizard, listing-with-
   distractors) driven by `browser-use` with this model as the LLM. **3/3
   passed**, including correctly picking the one correct listing among two
   distractors. This is a hand-built proxy, not a validated benchmark — a
   larger/harder task set would give a more trustworthy signal, but the
   floor is real: the model can drive a browser through multi-step
   form-filling tasks representative of the target use case.

**All three domains now look solid enough that distillation is worth
attempting on its own merits** — no domain is disqualifying the model.

## Distillation — raised as a question, not yet attempted

Goal: make Qwen3.6-35B-A3B (or a successor) smaller/faster via some
combination of the techniques already in this project's toolkit — QAT, PEFT/
LoRA-style adapters, pruning — potentially combined with training a smaller
student against a larger teacher's outputs.

**Closest prior evidence, and it's a cautionary one:** KD-LoRA (training a
LoRA adapter against teacher logits) was tried for Gemma's nf4 *recovery*
and the real control killed it — an 18% flip-rate improvement bought nothing
on GSM8K. Not the same goal (that was PTQ-recovery, not training a smaller
general-purpose student), but the lesson transfers directly: **a technique
that improves a proxy metric is not automatically a technique that improves
the downstream task, and any distillation attempt needs its own matched
control before its result is trusted.**

**Cheaper thing to check first:** Google's official Gemma 4 QAT release
(verified to exist this session — see Priority 4 below) is already-distilled/
QAT'd vendor work. If it clears the bar on the target-use-case benchmarks
above, that's a much cheaper win than building a custom distillation
pipeline (real training compute, a teacher, curated data, its own eval —
substantially bigger than anything else in this project so far, which has
only done PTQ/inference-time work, no training runs). Only reach for custom
distillation if the vendor checkpoint doesn't pan out on the benchmarks that
actually matter for this deployment.

---

# PRIORITY 4 — THE EXTERNAL BASELINE THIS PROJECT HAS NEVER HAD

**Verified this session: the release is real, not speculative.** Official
Gemma 4 QAT checkpoints exist — `google/gemma-4-31B-it-qat-q4_0-unquantized`,
plus GGUF (Q4_0), compressed-tensors (w4a16, for vLLM), and mobile (wNa8o8)
variants, across E2B/E4B/12B/26B-A4B/31B. The actual head-to-head run has
**not** been done yet. Tasks, unchanged from before:

1. **Find Google's eval methodology** — benchmarks, sample sizes, thinking
   mode on/off, FP16 baseline. If their eval set differs from ours, their
   claimed quality retention may not transfer to GSM8K-style generative
   tasks — where this project has repeatedly found damage concentrates, and
   where Priority 3's target-use-case benchmarks live too.
2. **Run the head-to-head on our harness**, `--limit 1000`, reporting in the
   mandated order (flip rate → KL → downstream → bpb → perplexity last). Arms:
   Google official QAT · our w4g128 · nf4 (bnb) control · bf16 reference.
   Register via `registry.py` with proper `arm`/`pair_id`.
3. **Answer explicitly:**
   - Does Google's QAT beat w4g128 on flip rate and KL?
   - Does it beat it on GSM8K, and on Priority 3's target-use-case benchmarks?
   - Should downstream work build on Google's QAT checkpoints instead of our
     own quantizer? Adopting a stronger base is a good outcome, not a failure.

---

# PRIORITY 5 — CLOSE THE OPEN CONTROLS

From `CLAUDE.md` Part 6 item 6, plus one anomaly worth resolving:

- **`vm/controls.sh` is staged and unrun.** The mixed-precision conclusion
  ("sensitivity-ranked worse than naive first-N") currently rests on an unmatched
  comparison — sensitivity used pure RTN while the depth sweep used `distill_seq`
  with 200 reconstruction steps. Run the control. The conclusion may not survive.
- **The flip-rate / GSM8K anomaly.** nf4 has a *worse* flip rate than w4g128
  (9.18% vs 8.25%) yet scores *higher* on GSM8K (84.0 vs 80.0). Part 3 flags this
  as unexplained. Re-run both at `--limit 1000`. If it persists at larger n, the
  project's assumption that flip rate predicts task performance needs revising —
  and since flip rate is the mandated headline metric, that would be significant.
- **MMLU/`lm_eval` logprobs incompatibility, found this session, unresolved.**
  `llama-server` returns logprobs in the newer OpenAI chat-completions format
  (`logprobs.content[].logprob`); `lm_eval`'s completions parser expects the
  legacy `token_logprobs` list. Every MMLU request errors and retries forever.
  Needed if MMLU (or any loglikelihood-style task, including GPQA/MMLU-Pro
  from Priority 3) is wanted alongside GSM8K for the llama.cpp path.

---

# WHAT NOT TO DO

- **Do not revive KD-LoRA r=32** for PTQ recovery without new evidence — the
  control killed it. A *distillation* attempt (Priority 3) is a different
  goal and not covered by this warning, but still needs its own control.
- **Do not pursue ternary or 2-bit** for the dense Gemma path. Dead at this
  scale (+1,780% / +429% ppl).
- **Do not pursue MLP channel pruning** as a primary lever — 25× worse per unit
  of compression than quantization.
- **Do not fall back to an 8B model** without explicitly costing it. Project data
  says an 8B fallback costs ~20–30 GSM8K points, and Gemma at 4-bit beats
  Qwen3.8-27B at bf16 by 14 GSM8K points while being 3× smaller. Qwen3.6-35B-A3B
  now clears the bar directly (Priority 1's summary), so this fallback question
  is largely moot unless the MoE engineering work in Priority 1 turns out
  intractable.
- **Do not report any vocab-trimming quality number from wikitext.** Structurally
  invalid; one such conclusion has already been retracted.
- **Do not trust an eval number — especially a surprising one — without pulling
  raw samples.** This session's whole GSM8K saga (43.6% → 50.3% → 94-95%) was
  three rounds of a plausible-looking wrong number. Read actual generated text
  before reporting anything.
- **Do not build a custom distillation pipeline before checking Google's QAT
  release against Priority 3's target-use-case benchmarks.** Cheaper win,
  check it first.
