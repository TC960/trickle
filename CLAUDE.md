# Project instructions and state

Handoff document. Read this first. It carries the project's goal, everything
measured, every bug found, the standing methodology rules, and the decision
currently on the table.

---

# PART 1 — STANDING RULES

## Evaluation: do not lead with perplexity

**Perplexity is not adequate for judging compressed models here.** Do not report
it as the headline number, and do not let it decide direction.

It is a geometric mean of token probabilities, so losses on some tokens cancel
gains on others. Measured consequence ([Accuracy is Not All You
Need](https://arxiv.org/abs/2509.09141), NeurIPS 2024, Table 19): WikiText-2
perplexity held at **exactly 5.70** while greedy-token agreement collapsed from
**61.3% to 21.5%**. A 2.4% perplexity spread across quantization schemes covered
a **~250× KL range**. Kurtic et al. (ACL 2025) ran 500,000+ compression
evaluations and used perplexity **zero times**.

Our own data, same model and same compression, measured four ways:

| metric | change under 4-bit |
|---|---|
| perplexity | +2.96% |
| MMLU | −0.9% |
| HellaSwag | −0.7% |
| **GSM8K** | **−7.0%** |

Report in this order:

1. **Flip rate** — fraction of positions where the compressed model's argmax
   differs from the bf16 teacher's. What a user experiences.
2. **KL divergence** vs the teacher — catches distribution shift perplexity cancels.
3. **Downstream tasks** — MMLU, HellaSwag, ARC-C, and especially **GSM8K**.
   GSM8K is generative, so one corrupted token kills the answer; damage shows up
   there first. Multiple-choice tasks are forgiving and can look fine on a
   broken model.
4. **Bits-per-byte** whenever tokenization changes. Perplexity is NOT comparable
   across tokenizers.
5. **Perplexity** — last, only for comparability with published work.

`deep_eval.py` implements 1–2 plus frequency-stratified NLL. `benchmarks.py`
does 3. `sensitivity.py` ranks layers by 1–2, not perplexity.

Also: WikiText-2 reads only **~8% of the embedding table**, so it structurally
cannot evaluate embedding or vocabulary compression.

## Other standing rules

- **Verify before asserting.** Search or read primary sources. This project's
  assumptions have been wrong repeatedly: lm_head streamability, KV
  streamability, "Qwen 3.8 doesn't exist", "ternary fails above 3B", "Muse
  Glimmer isn't real". The user's tentative hypotheses have a better hit rate
  than my confident assertions.
- **Every treatment needs a control**, recorded via `registry.py` with
  `arm`/`pair_id`. `report.py` refuses to display a treatment without one.
- **Single-seed differences under ~5% are not results.** Say so.
- **Local proxies mislead.** Cosine similarity is scale-invariant and hid a 16%
  per-block error. Weight-space error said ternary ≈ 2-bit; end-to-end they
  differ 3.6×. Prefer end-to-end measurement.
- **Sanity-check the pipeline itself.** 8-bit quantization should reproduce
  bf16 within ~0.1%. It does (+0.025%).
- **A test that exercises one component against itself proves far less than one
  that pins two components to each other.** Three bugs escaped round-trip tests
  and were caught only by cross-component assertions. See Part 5.
- **Do not run anything heavy locally.** The project directory lives inside
  OneDrive (`~/Library/CloudStorage/OneDrive-.../Desktop/`), and `.venv` has
  30,389 files in it. Every import round-trips through cloud sync; a trivial
  unit test never finishes. All work runs on the Brev box.
- **Never run `chmod`.** Never run commands needing interactive approval.

---

# PART 2 — WHAT THIS PROJECT IS

Run a large language model on an **8 GB NVIDIA Jetson Orin Nano** by combining:

1. **Quantization** — fewer bits per weight (16 → 4)
2. **AirLLM-style layer streaming** — load one layer, compute, evict, load next.
   Peak memory becomes the largest single layer, not the whole model.

Original target: `google/gemma-4-31B`, 61.6 GB in bf16.

**A pivot to MoE is now on the table and is probably correct. See Part 6.**

---

# PART 3 — RESULTS

## Gemma 4 31B, quantization

| config | flip rate | KL | GSM8K | MMLU | perplexity |
|---|---|---|---|---|---|
| bf16 | — | — | **86.0** | 82.99 | 5.1876 |
| w8g128 (ours) | 2.07% | 0.0140 | — | — | 5.1889 |
| int8 (bnb) | 3.84% | 0.0302 | — | — | 5.2132 |
| **w4g128 (ours)** | **8.25%** | **0.0896** | **80.0** | 82.27 | 5.3407 |
| nf4 (bnb) | 9.18% | 0.1213 | — | — | 5.4484 |
| nf4 (control) | 9.18% | 0.1213 | **84.0** | 82.65 | 5.4484 |
| nf4 + KD-LoRA r=32 | 7.54% | 0.0989 | **83.5** | 82.97 | — |
| w3g128 | — | — | — | — | 6.0835 |
| w2g128 | — | — | — | — | 27.43 |
| ternary (all 60 layers) | — | — | — | — | 97.55 |

**4-bit is the practical floor.** 3-bit costs +17%, 2-bit +429%, ternary
+1,780%. Ternary was the original ambition and is dead at this scale.

The bf16 downstream row is `--limit 200`; treat small differences as noise.

## Streaming — the deliverable, proven

Gemma 4 sharded to 4 bits: **19.30 GB on disk, 4.1875 bits/weight, largest
layer 280 MB**, built in 833s.

```
budget  64.0 GB | resident 18212.6 MB | 1.456 tok/s |  0.000 GB/tok | max|dlogit| 0.00e+00
budget   4.0 GB | resident  3911.7 MB | 0.472 tok/s | 15.333 GB/tok | max|dlogit| 0.00e+00
budget   2.0 GB | resident  3159.5 MB | 0.472 tok/s | 15.333 GB/tok | max|dlogit| 0.00e+00
budget   1.0 GB | resident  3159.5 MB | 0.473 tok/s | 15.333 GB/tok | max|dlogit| 0.00e+00
budget   0.5 GB | resident  3159.5 MB | 0.471 tok/s | 15.333 GB/tok | max|dlogit| 0.00e+00
```

- **Bit-exact at every budget.** `max|Δlogit| = 0.00e+00`. The engine reproduces
  the model exactly at 31B.
- **Resident floors at 3.16 GB** — fits an 8 GB device. That floor is pinned
  embeddings + vision tower + norms; `--budget-gb` does not govern it.
- **15.333 GB read per token**, matching the 15.2 GB arithmetic prediction.
- **LRU gives a 0% hit rate.** Layer traversal is cyclic (0→59, repeat), which
  is LRU's pathological case: it evicts exactly what is needed soonest. Below a
  4 GB budget nothing changes because there was nothing left to lose. **A fixed
  partition — pin N layers, stream the rest — would give a guaranteed N/60 hit
  rate.** Not yet implemented.

### Throughput caveat — important

The projection table `stream_bench.py` prints is **wrong in both directions**:

- **Too pessimistic:** `ShardStore` stages shards in CPU RAM and copies to GPU
  over PCIe every token (~0.6–1.0 s of the 2.12 s "compute"). A Jetson has
  unified memory and never pays this.
- **Too optimistic:** it holds compute at H200 speed, which Orin cannot match.

**What survives is hardware-independent: 15.333 GB read per token.** At Jetson
storage speeds (0.3–3 GB/s) that alone is 5–51 s/token before any compute. Too
slow for interactive use.

**Batching amortizes it** — 32 tokens per pass is 0.48 GB/token. But decode is
autoregressive, so you can only batch across concurrent requests, not within one
conversation. Streaming is viable for throughput, not for latency.

## Other axes

- **GPTQ with real Hessians beats round-to-nearest by 36–41%** at 2/3/4 bits and
  both group sizes. An earlier "GPTQ is 33% worse" result came from a
  rank-deficient test Hessian; `H = I` reproduces RTN exactly, proving the
  implementation correct.
- **MLP channel pruning is not competitive.** Removing 6.8% of parameters flips
  10.07% of tokens; 27% removed flips 30.27%. Roughly 25× worse per unit of
  compression than quantization. Caveat: no recovery finetuning was applied.
- **Mixed precision by per-layer sensitivity is worse than naive first-N.**
  Sensitivity-ranked selection scored 210.56 / 9,584 / 50,432 perplexity at
  N=30/40/45. **The matched control never ran** — the depth sweep it would be
  compared against used `distill_seq` with 200 reconstruction steps while
  `sensitivity.py --mode allocate` is pure round-to-nearest. `--layers` was
  added to make the control runnable; `vm/controls.sh` is staged, unrun.
- **Per-layer sensitivity map** (KL and flip rate, one layer ternarized at a
  time): most tolerant layer 4 (KL 0.034, 4.29% flips); most sensitive layer 59
  (KL 0.975, **26.67% flips**), then 53, 58, 52, 40, 49, 29. ~28× spread.
  Full-attention layers averaged ~3× the damage of sliding-window layers.
- **Vocabulary:** only **20.6%** of Gemma's 262,144 tokens fire on English and
  code. Trimming to 51,269 shrinks the table 5.11× (1.41 GB → ~280 MB) at +15%
  token inflation. That table is **pinned**, so this converts directly into ~4
  more resident layers. Never evaluated for quality — wikitext reads 7.7% of
  rows, so it cannot judge this.
- **KD-LoRA recovery — the control killed it.** Trained against bf16 teacher
  logits (KL, not hard labels). 490 MB adapter at r=32. It cut the flip rate
  9.18% → 7.54%, a real 18% improvement. **It bought nothing downstream:
  GSM8K went 84.0 → 83.5, slightly WORSE than the unadapted nf4 control.**

  This was reported mid-session as a success ("83.5 vs 80.0") by comparing
  against `w4g128` — a different quantizer with no adapter — because the real
  control had not yet run. Against the actual control the gain vanishes. On
  this evidence KD-LoRA is not worth shipping at r=32.

  Also note an earlier run showed 3.95% flips, but trained and evaluated on the
  same wikitext split; the contamination inflated the apparent gain ~3×.

## Statistical power — read before believing any downstream number

All downstream numbers above use `--limit 200`. On GSM8K at p≈0.84 that is a
standard error of **±2.6 points**. Consequences:

| comparison | gap | verdict |
|---|---|---|
| nf4 84.0 vs nf4+LoRA 83.5 | 0.5 | noise |
| nf4 84.0 vs w4g128 80.0 | 4.0 | ~1σ, noise |
| bf16 86.0 vs w4g128 80.0 | 6.0 | ~1.5σ, borderline |

**Most arms cannot be distinguished at n=200.** By this project's own rule —
differences under ~5% relative are not results — only bf16 → w4g128 (7%
relative) clears the bar, and barely.

There is also something unexplained: **nf4 has a worse flip rate than our
w4g128 (9.18% vs 8.25%) yet scores higher on GSM8K (84.0 vs 80.0)**. Either
that is noise, or flip rate and task performance are less tightly coupled than
this project has assumed. Both possibilities need larger n.

**Re-run the headline comparisons at `--limit 1000` before drawing conclusions
from them.** It halves the error bars and costs about an hour.

## Qwen3.8-27B (secondary model)

bf16 GSM8K 66.0, MMLU 85.14. int8 4.03% flips; nf4 8.06% flips, GSM8K 60.4.
More quantization-robust than Gemma (nf4 costs +0.90% perplexity vs +5.03%) but
20 GSM8K points weaker at full precision. **Gemma at 4-bit beats Qwen at bf16 by
14 GSM8K points while being 3× smaller.**

## Qwen3.6-35B-A3B (MoE pivot candidate) — Priority 1 resolved, pivot supported

Both open questions from Part 6 now have answers.

**Capability: GSM8K 95.3% strict-match / 94.0% flexible-extract, n=1000, 5-shot,
via Q8_0 GGUF through llama.cpp.** This *beats* Gemma 4 31B's 86.0 (n=200), and
does so by more than either model's stderr (Qwen ±0.7-0.8, Gemma ±2.6). The
decision criterion in the handoff prompt ("within roughly one standard error,
does not need to win") was cleared with room to spare — it won outright.

**Getting a trustworthy number took three real bugs, each of which quietly
produced a plausible-looking wrong result:**

1. `lm_eval`'s API-model path defaults `max_gen_toks=256`. This model produces
   spontaneous `<think>...</think>` chain-of-thought even under raw few-shot
   completion (no chat template, no system prompt asking for it) — 256 tokens
   cuts that off before the answer. First real number: **43.6%**.
2. Raising `max_gen_toks` to 2048 only partially fixed it → **50.3%**. Sample
   inspection found the actual cause: the stock `gsm8k` task's stop condition
   fires on the bare string `"Question:"`, and this model's `<think>` preamble
   routinely writes `"- Question: ..."` while describing the prompt format to
   itself — matching the stop condition almost immediately, before it reaches
   the real problem. 426 of 1000 responses were the *identical* 24-token
   truncated string, independent of what the actual question was. Confirmed
   directly: same prompt, stop condition removed, model reasons for 1041
   tokens and lands on the correct answer.
3. Fix: a custom task (`gsm8k_fixed.yaml`, stops only on `"\nQuestion:"`, a
   genuine new-fewshot-example boundary) plus a 3072-token budget. Truncation
   dropped from ~48% of responses to 2 of 1000. Final, trusted number: 94-95%.

MMLU is separately broken and unresolved: llama-server returns logprobs in the
newer OpenAI chat-completions format (`logprobs.content[].logprob`) rather
than the legacy `token_logprobs` list `lm_eval`'s completions parser expects,
so every loglikelihood request errors and retries forever. Not fixed — GSM8K
is the metric this project's own methodology already treats as decisive
(Part 1: generative tasks show damage first, multiple-choice is forgiving).

**Throughput under a Jetson-realistic memory budget** (Q4_K_M GGUF, `-ngl 0`
so weights live in host-mmap'd page cache the way they would on Jetson's
unified memory, no custom residency manager — plain OS page-cache eviction
under a `systemd-run` cgroup memory limit, cold cache dropped before each run):

| budget | decode tok/s | vs unconstrained | major page faults |
|---|---|---|---|
| unconstrained (22.3 GB resident) | 14.89 ± 0.15 | — | 1,601 |
| 8 GB | 4.54 ± 0.46 | −69% | 562,554 |
| 4 GB | 1.52 ± 0.03 | **−90%** | 2,029,163 |

Real degradation — plain mmap/page-cache eviction does not gracefully handle
this model's expert-access pattern. But even the worst case (1.52 tok/s ≈
0.66 s/token) is categorically more usable than dense Gemma streaming's
projected 5-51 s/token. This measurement needed a fix too: the first attempt
showed *zero* difference between budgets (all three showed ~21GB resident,
~0 major faults) because the unconstrained baseline ran first and warmed the
OS-wide page cache, letting the "constrained" runs free-ride on already-cached
pages. Fixed by dropping the page cache (`echo 3 > drop_caches`) before each
of the three runs.

The measuring box's raw cold-read disk bandwidth is 739 MB/s (`dd ... iflag=direct`)
— mid-range for Jetson's expected 0.3-3 GB/s, so this isn't an optimistic
proxy; a real Jetson on slower storage would likely see worse degradation
than measured here.

### Quantization sweep (the Qwen equivalent of the Gemma w8g128/w4g128 sweep)

Same fixed `gsm8k_fixed` methodology, n=1000, on the Q4_K_M GGUF (22.1 GB, the
actual Jetson deployment target) instead of Q8_0:

| quant | strict-match | flexible-extract |
|---|---|---|
| Q8_0 (bf16 proxy) | 95.3% | 94.0% |
| Q4_K_M (deployment target) | 94.6% | 92.3% |

4-bit costs ~0.7-1.7 points here — nowhere near Gemma's dense-path cost at
4-bit (86.0 → 80.0 w4g128, a 6-point drop). Qwen3.6-35B-A3B tolerates
quantization to the actual deployment target far better than Gemma did.

### Target-use-case benchmarks (Part 6.5's three domains, now measured)

All on Q8_0 via `llama.cpp` + `lm_eval`, same raw-completion methodology:

| domain | benchmark | result |
|---|---|---|
| coding | HumanEval, n=164 | pass@1 61.0% ± 3.8% |
| coding | MBPP, n=500 | pass@1 62.6% ± 2.2% |
| agentic browsing | 3 hand-built job-application mock tasks (`browser-use` + this model as the driving LLM) | **3/3 passed** |
| technical Q&A | 6 MMLU-technical subjects, full test splits, custom `generate_until` task (see below) | college_physics 97.1%, electrical_engineering 84.8%, machine_learning 83.9%, college_computer_science 84.0%, computer_security 76.0%, college_mathematics 73.0% — **~83% average** |

**Technical Q&A had to be built too, and for the same reason as everything
else this session: the obvious path was gated.** GPQA (the standard "harder
than MMLU" pick) is a gated HF dataset requiring an approved account — not
resolvable in-session, so parked, not solved. MMLU's own `loglikelihood`
form hits the same logprobs incompatibility as MMLU proper. `lm_eval` does
ship an `mmlu_generative` group avoiding logprobs entirely, but its stock
template stops generation on a bare `"\n"` and extracts only the first line
of the response — both assumptions this model breaks the same way it broke
gsm8k's stop condition, for the same underlying reason (spontaneous `<think>`
reasoning even under raw completion). Built a small custom fix instead: 6
representative "technical" MMLU subjects (college CS/physics/math, electrical
engineering, computer security, machine learning), asking the model to
conclude with `Final Answer: X` and stopping only on `</s>`/`<|im_end|>`.

**This one had its own bug, caught the same way as the others: read the
samples before trusting the score.** First attempt scored a flat 0% across
the board — looked like the model failed every question. Reading the raw
generations showed it answering correctly every time; the extraction regex
(`(?<=Final Answer: )(.*)(?=.)`, copied from `gpqa`'s own filter style) was
the bug — the trailing lookahead `(?=.)` requires a character *after* the
match, which fails and backtracks to an empty capture whenever the answer
letter is the literal last character of the response, which is exactly what
happens when generation stops right after "Final Answer: X". Fixed by
dropping the trailing lookahead entirely (`(?<=Final Answer: )[A-D]` needs
nothing after the match). Confirmed the fix against known-correct samples
before running the full 6-subject set.

Both `humaneval`/`mbpp` needed `HF_ALLOW_CODE_EVAL=1` set (separate safety
gate from `lm_eval`'s own `--confirm_run_unsafe_code`, both required since
these execute model-generated code — reasonable on an isolated rented VM,
this is the standard benchmark protocol). Sample inspection confirmed the
model answers HumanEval/MBPP directly, with no `<think>` preamble at all —
different behavior from GSM8K, and it means the earlier stop-string
truncation bug (bugs 14-15) never had a chance to recur here; confirmed clean
before trusting the number, not after.

**Agentic browsing setup, since there was no existing harness for this:**
`browser-use` requires Python ≥3.11 (box only had 3.10 — added via deadsnakes
PPA into a dedicated venv, left the main venv alone). `use_vision=False` is
required — this model's GGUF has no mmproj loaded, text-only. Real academic
web-agent benchmarks (WebArena, OSWorld, WorkArena) need self-hosted Docker
sites, full desktop VMs, or gated access — all multi-day builds, not
attempted. Real job sites were also rejected: automating actual applications
means spamming real employers and isn't reproducible if the site changes.
Instead: 3 local HTML mock job-application pages of increasing complexity
(single form; 2-step wizard with select/radio/number fields; a job-listing
page with two distractor roles to check the model doesn't misapply), served
over local HTTP (`browser-use` blocks `file://` navigation as a security
default — cost one failed run before catching this), graded by parsing the
agent's own final report for a "SUBMITTED: {json}" string and checking the
submitted fields match what was asked. All 3 passed, including correctly
identifying and applying to the one correct listing among distractors.
This is a hand-built proxy, not a validated academic benchmark — treat the
3/3 as a positive signal, not a rigorous score.

### Running the same technique sweep on Qwen that this project ran on Gemma

User request: apply everything tried on Gemma (ternary, streaming, vocab
pruning, expert deletion, QLoRA, QAT, PTQ) to Qwen3.6-35B-A3B too, and look
for further improvement. Status per technique:

- **PTQ: extended, and the result is startling.** Q8_0/Q4_K_M already done
  (Part 3 above). Lower-bit sweep via `gsm8k_fixed`, n=1000, mirroring
  Gemma's w3g128/w2g128/ternary floor-finding:

  | quant | bits (approx) | strict-match | flexible-extract |
  |---|---|---|---|
  | Q8_0 | 8 | 95.3% | 94.0% |
  | Q4_K_M | 4 | 94.6% | 92.3% |
  | Q3_K_M | 3 | 95.5% | 91.0% |
  | Q2_K_XL | 2 | 94.1% | 90.1% |
  | IQ1_M | ~1.6-1.8 (ternary-adjacent floor) | 93.0% | 90.1% |

  Across the entire range from 8-bit down to ~1.6-bit, GSM8K drops only
  ~2-4 points total. Contrast Gemma's dense path: 3-bit already cost +17%
  perplexity, 2-bit was +429%, ternary was completely broken (+1,780%).

  **This is not a clean architecture-vs-architecture comparison and should
  not be reported as one.** Gemma's low-bit numbers came from this project's
  own simple uniform quantizer (`airllm_ternary/uniform.py`). Qwen's low-bit
  numbers came from unsloth's `IQ`/`Q*_K` quants, which use per-tensor
  importance calibration (an `imatrix_unsloth.gguf_file` ships in that repo)
  and mixed-precision block allocation — a materially more sophisticated PTQ
  method than naive uniform quantization. The result mixes two variables
  (architecture *and* quantizer quality) and isolating "MoE is inherently
  more quantization-tolerant" from "this used a better quantizer" would need
  the same method applied to both models — not done. Report the numbers as
  measured; don't claim the clean architectural conclusion they'd suggest at
  a glance.
- **Ternary: not directly done, floor approximated instead** (see caveat
  above — IQ1_M is not literally our ternary code path). True ternary via
  our own quantizer needs our own inference engine to run the resulting
  non-standard format — the same underlying blocker as custom streaming
  below. IQ1_M (llama.cpp's most aggressive available i-quant, ~1.6-1.8 bit
  equivalent) is the closest available proxy for "how low can this model go,"
  included in the sweep above.
- **Streaming: already tested, via a different engine than expected.** The
  8GB/4GB memory-constrained throughput numbers (this file, above) *are* a
  streaming test — just via `llama.cpp`'s built-in mmap loader, not this
  project's own `airllm_ternary/loader.py`. That engine doesn't support this
  model's hybrid attention yet (see below).
- **Vocab/token pruning: measured.** Only **10.14%** of Qwen's 248,077-token
  vocabulary fires across a combined English-prose (wikitext-2) + code
  (HumanEval/MBPP) + technical-English (the 6 MMLU subjects from the
  technical-Q&A eval) corpus — 25,167 of 248,077 unique token IDs used. Even
  sparser than Gemma's 20.6% (Part 3, Qwen entry). Qwen's embeddings are
  untied (Part 4), so a vocab trim benefits both the input embedding *and*
  the separate lm_head — likely a bigger absolute resident-memory win than
  Gemma's tied-embedding case, at a comparable or better ratio. Not yet
  trimmed-and-evaluated; same caveat as Gemma applies — don't judge trimmed
  vocab quality on wikitext alone (Part 1).
- **Expert deletion: measured, decided not to build.** Instrumented
  `llama.cpp` directly (patched `examples/eval-callback` to filter+dump the
  `ffn_moe_topk` tensor — the actual per-token top-8 expert-selection
  indices, one line of `ggml_argsort_top_k` output per token per layer;
  needed a second patch too, since the debug printer truncates any tensor
  dimension over 6 elements by default, which would have silently sampled 6
  of ~7000 tokens). Two real findings:
  - Across a mixed GSM8K+code+MMLU prompt (~7,500 tokens), **96.5-100% of
    all 256 experts in every one of the 40 layers fired at least once**, and
    the single most popular expert per layer only accounted for 0.57-0.76%
    of selections (vs. 0.39% uniform baseline) — essentially no small "cold"
    set exists when usage is measured across a domain mix. Global,
    domain-agnostic expert deletion is not viable on this evidence.
  - Split by domain instead: **top-20-expert overlap between GSM8K/code/MMLU
    is only ~20-24%** — confirming real domain-specific routing skew exists
    (the mechanism the sibling doc `gemma4-jetson-compression-prompt.md`
    proposed for a *different* Gemma MoE variant). But **even restricted to
    one domain alone, ~124-128 of 256 experts (≈48-50%) are still needed to
    cover 90% of that domain's own selections** — the ceiling on
    domain-specialized pruning is real but modest (roughly 2x, not 5-10x).
    Combined with there being no existing tooling to actually remove experts
    from a GGUF/safetensors checkpoint and repack it, the engineering cost
    doesn't currently clear the bar set by this measurement. Recorded as a
    considered no, not a skipped question.
- **QLoRA / QAT: scoped, not attempted.** The model's `transformers` class
  (`Qwen3_5MoeForConditionalGeneration`) loads *natively* in the installed
  library (v5.16.1, no `trust_remote_code` needed — confirmed by direct
  import). But the working venv has no `torch`/`bitsandbytes`/`peft`/
  `accelerate` installed (only ever needed `transformers` for tokenizer
  utilities so far), and a real run needs a training data/loss pipeline and
  genuine wall-clock training time on top of that — a distinct, larger
  undertaking than anything else done this session, which has been
  inference-only throughout. Feasible; not started.
- **Custom-engine streaming + true ternary, scoped together (same blocker).**
  Pulled the actual reference implementation: this model's `transformers`
  class is native (not a custom per-repo file), confirmed via
  `config.json`'s `architectures: ["Qwen3_5MoeForConditionalGeneration"]`
  (internal codename "Qwen3.5"; also revealed a detail not previously
  recorded — `mtp_num_hidden_layers: 1`, i.e. this model also has a
  multi-token-prediction head). The Gated DeltaNet mechanism itself is only
  ~350 lines total. The **decode-time (recurrent) form is genuinely simple**
  — ~50 lines, a per-token loop maintaining one small fixed-size state
  tensor per layer (`[heads, k_dim, v_dim]`, does not grow with sequence
  length), structurally just like a KV-cache: update, persist across the
  streaming loop's evict/reload cycle, repeat. The **prefill (chunked/
  parallel) form is more involved** (~80 lines, cumulative-decay masking) but
  a v1 could reuse the simple recurrent loop for prefill too, at some
  prompt-processing speed cost, deferring the optimized path. The recurrent
  state must be kept in float32 regardless of the rest of the model's
  precision (confirmed from the reference code, matches `mamba_ssm_dtype:
  float32` in config). HF's own code has a fallback-to-optimized-kernel hook
  pointing at `fla` (Flash Linear Attention) — a maintained fast
  implementation already exists to adapt from later, not something to build
  from scratch. **Revised estimate: a bounded few-day task with a concrete
  reference to port from, not the open-ended multi-week unknown it looked
  like before this was actually checked.** Still not attempted — this
  remains the honest gap between "streaming was tested" (true, via
  llama.cpp) and "our own engine supports this model" (not yet true).

---

# PART 4 — ARCHITECTURE NOTES

## Gemma 4 31B
60 layers, hidden 5376, FFN 21504, vocab 262144, **tied embeddings**.
50 sliding-window (1024) + 10 full-attention layers at indices 5,11,...,59.
`attention_k_eq_v: true` on the 10 global layers — `v_proj` appears 50× against
60× for q/k/o. Multimodal: **27-layer vision tower**, whose projections are
wrapped in `Gemma4ClippableLinear` (peft cannot adapt these; scope LoRA targets
to `.*language_model.*` or it hard-fails).
Budget: MLP 20.81B (67.5%), attention 8.59B (27.9%), embeddings 1.41B (4.6%).

## Qwen3.6-35B-A3B (the pivot candidate)
Verified from config.json: 40 layers, hidden 2048, **256 experts, 8 active per
token**, `moe_intermediate_size` 512, plus one shared expert. Vocab 248320,
untied. Hybrid **linear (Mamba-style) attention** with full attention every 4
layers — our streaming code does not handle this. Also multimodal.

| | Gemma 4 (dense) | Qwen3.6-35B-A3B (MoE) |
|---|---|---|
| body params | 29.4B | 33.1B |
| **touched per token** | **29.4B** | **1.89B** |
| **read per token @4-bit** | **15.2 GB** | **0.97 GB** |

**15.6× less I/O per token.** At 4 bits one expert is 1.62 MB, so a 5 GB cache
holds ~3,000 of 10,240 experts, and expert popularity is skewed so an LRU cache
should do better than its share. `loader.py`'s residency manager is arguably
better suited to experts than to whole layers.

Caveat: 40% of the per-token read is attention, which MoE does not help.

## Muse Glimmer 30B (evaluated, rejected)
52 layers, hidden 6656, 32 query heads / **2 KV heads**, vocab 202048, untied,
50-layer vision tower. **Dense**, so 12.3 GB/token — a 20% improvement on a
number that needs to fall 15×. Untied embeddings cost 2.69B pinned params
against Gemma's 1.41B. Its one advantage: published quantized checkpoints
(RedHatAI FP8, community GGUF) exist, which would give the external baseline
this project has never had.

## Jetson Orin Nano 8 GB budget (approximate)
| | |
|---|---|
| total | 8.0 GB |
| JetPack / Ubuntu | ~2.0 GB |
| CUDA context | ~0.8 GB |
| KV cache + activations | ~0.5–1.0 GB |
| **left for weights** | **~4.2–4.7 GB** |

That is ~7–8B parameters resident at 4 bits, **not 14B**. Our streamed Gemma at
3.16 GB resident fits — a 31B model where only 8B would otherwise go.

---

# PART 5 — BUGS FOUND IN OUR OWN CODE

Thirteen. Note which produced *believable wrong numbers* rather than crashing.

1. Missing W1.58A8 activation quantization → fluent garbage
2. Severed `tie_word_embeddings` → `lm_head` left on meta device
3. Teacher-forced reconstruction → trained on inputs blocks never see
4. Cosine similarity as fidelity metric → scale-invariant, hid 16% real error
5. `reconstruction_cosine` aliasing → always returned exactly 1.0
6. Clone fix broke `was_tied` → `--untie` silently a no-op
7. QAT params allocated on CPU → device mismatch crash
8. Teacher logits stored full-vocab → 161 GB
9. Queue races → repeated OOMs misdiagnosed as memory-math errors
10. `mlp_prune.py` cloned all 60 blocks' MLP weights (~41 GB) → OOM
11. `perplexity()` passed `labels=`, upcasting 2048×262144 to fp32 → OOM
12. **Watchdog reported liveness it could not observe** — `${B1:-1}` made an
    unreachable box read identical to a busy one; it logged 11 hours of work
    that never happened, and I relayed that as fact. Also used
    `nvidia-smi --query-compute-apps=pid`, which returns nothing inside these
    VMs regardless of load. Backups ran only at start and end, so ~5h of
    results died with the box.
13. **Scale-precision mismatch.** `quantize_ternary` derived codes from an fp32
    scale and stored bf16; dequant used bf16. Every weight displaced by a
    fraction of a step. Present from the first commit. Fixed by rounding the
    scale to storage precision *first*, then deriving codes. `qat.py` STEs now
    round to bf16 too, so training optimizes what ships.

14. **`lm_eval`'s API-model `max_gen_toks` defaults to 256.** Silently truncated
    a reasoning model's chain-of-thought before it reached an answer. First
    Qwen3.6-35B-A3B GSM8K number: 43.6%, believable-looking and wrong.
15. **Stop-string collision with the model's own output.** The stock `gsm8k`
    task stops on bare `"Question:"`; this model's `<think>` preamble writes
    that literal word while describing the prompt format to itself, firing
    the stop condition within ~24 tokens for 426 of 1000 responses,
    independent of the actual question. Second number: 50.3%, also
    believable-looking and wrong. True number, once fixed: 94-95%. A 51-point
    swing from a stop-string, not a model or quantization difference.
16. **Page-cache confound in a memory-constrained benchmark.** First
    throughput-under-budget test showed literally zero difference between
    unconstrained, 8GB, and 4GB cgroup limits — because the unconstrained run
    went first and warmed the OS-wide page cache, letting the "constrained"
    runs free-ride on already-resident pages instead of being forced to read
    cold from disk under their own budget. Looked like "OS caching handles
    this gracefully"; was actually "the test never ran under the condition it
    claimed to."
17. **`pkill -f pipeline.sh` killed its own invoking shell.** The pattern
    matched the literal string "pipeline.sh" inside the SSH command's own
    argv, not just the target process — a version of bug 12's lesson
    (instruments observing themselves) in a new place.

18. **Regex extraction filter silently returned empty matches.** A custom
    technical-Q&A task (Part 3) asked the model to conclude with
    `Final Answer: X` and extracted it with a filter copied from `gpqa`'s own
    style: `(?<=Final Answer: )(.*)(?=.)`. The trailing lookahead `(?=.)`
    requires a character *after* the match; when the answer letter is the
    literal last character of the response (routine here, since generation
    just stops after it), the regex engine backtracks to an empty capture
    instead of failing loudly. Scored 0/10 on a sample where every single
    answer was actually correct. Fixed by dropping the trailing lookahead.

**1–6, 12–16, 18 produced believable wrong results.** 7–11 and 17 crashed loudly.

Bug 13 survived because **round-trip tests passed throughout** — pack/unpack was
always exact. The fault lived in the relationship between quantizer and storage
format, which no round-trip test can see. It surfaced only once a test asserted
*trained weight == served weight*. Two attempted fixes made it worse
(8e-03 → 4e-01) before the cause was visible.

Also retracted: an embedding-compression "improvement" on Qwen, because
wikitext reads 7.71% of embedding rows and cannot evaluate what was deleted.

---

# PART 6 — CURRENT DECISION

**The dense-model streaming approach is memory-feasible and speed-infeasible.**
3.16 GB resident for a 31B model is a real result. 15.333 GB/token is not
survivable on Jetson storage.

**The pivot to Qwen3.6-35B-A3B is now evidence-backed, not just plausible.**
Both gating questions from the previous version of this section are answered
(see Part 3 for full detail):

- **Capability: resolved, pivot supported.** GSM8K 94-95% (n=1000) vs Gemma's
  86.0 (n=200) — beats it, not just "doesn't lose badly." Took three eval-harness
  bugs to get a trustworthy number (Part 5, bugs 14-16); the first two attempts
  (43.6%, 50.3%) were both wrong and both looked plausible.
- **Throughput under memory constraint: real degradation, but still usable.**
  Decode drops 69% at an 8GB budget and 90% at 4GB (Jetson-realistic) via plain
  OS page-cache eviction — no custom residency manager. Absolute speed at 4GB
  (1.52 tok/s) is still categorically better than dense Gemma's projected
  5-51 s/token.

Turned out **not** to need clearing the Gemma cache or downloading 70GB bf16 —
quantized GGUF (Q8_0 36.9GB as bf16 proxy, Q4_K_M 22.1GB as the deployment
target) via `llama.cpp` did the job on a cheap rented box, independent of
`ternary-h200`. See Part 7 for what actually happened to that box.

**Benchmark choice caveat, flagged post-hoc:** GSM8K/MMLU were chosen as
compression-damage canaries (Part 1 — generative tasks show damage first),
not as measures of fitness for the actual target use cases (technical Q&A,
coding, agentic browser use). They're still the right instrument for "did
quantization/distillation break the model," but a *separate* eval axis for
"is this model actually good at what I want it for" is still needed — see
Part 6.5 below.

## Ranked next steps

1. ~~Benchmark Qwen3.6-35B-A3B bf16~~ — **done, see Part 3.** Pivot decided.
2. **Fixed-partition residency** replacing LRU, still open. Turns a 0% hit
   rate into a guaranteed N/60 for the *dense* Gemma case. Separately, for
   Qwen's MoE experts specifically, note the residency question is different
   in kind: published measurements on the same "A3B" family (Qwen3-30B-A3B)
   show real but *drifting* skew — hot-expert-set intersection across
   consecutive micro-batches is below 0.5 — which argues for an adaptive
   cache (LRU/SLRU) for experts, not a static fixed partition. Two different
   answers for two different access patterns; don't conflate them.
3. **Build the actual Jetson-side MoE residency manager** for Qwen3.6-35B-A3B.
   The 8GB/4GB throughput numbers above used plain mmap with no eviction
   policy at all — a real residency manager (Priority 2's adaptive-cache
   logic, applied to experts) should beat 1.52 tok/s at 4GB, not just match it.
4. **Vocab trimming**, now motivated: the pinned table is the resident floor,
   and 20.6% of tokens are used. Needs an eval that isn't wikitext.
5. **Fused 4-bit kernels** — this session used `llama.cpp`/GGUF directly for
   Qwen, which resolves this item for the MoE path. Still open for whether the
   dense-Gemma path adopts the same rather than hand-written CUDA.
6. **The missing controls**: nf4 GSM8K (for KD-LoRA), and `vm/controls.sh`
   (for mixed precision).
7. **GPUDirect Storage / pinned+async copies** to remove the CPU staging hop.
8. **Fix or route around the MMLU/`lm_eval` logprobs incompatibility** (bug
   list, Part 5) if MMLU is wanted alongside GSM8K for Qwen going forward.

Do not spend more on compression before 2 and 3. The repeated failure mode in
this project has been optimizing a metric before checking which constraint
binds — perplexity before flip rate, bit-width before throughput.

## PART 6.5 — TARGET-USE-CASE BENCHMARKS AND DISTILLATION

The three actual intended uses of this deployment are: **technical Q&A**
(ask-it-things-while-learning), **coding** (including generating presentation
material while learning something), and **agentic browser use** (a browser/
computer-use agent, specifically for job applications). GSM8K/MMLU are
damage-detection canaries, not measures of these three things.

**All three are now measured** (see Part 3 for results and methodology):
HumanEval 61.0% / MBPP 62.6%; 3/3 on a hand-built job-application mock-task
set via `browser-use`; and ~83% average (73-97% range) across 6 technical
MMLU subjects via a custom fixed task. GPQA (the more standard "harder than
MMLU" pick) is gated on HuggingFace and wasn't resolved this session — worth
revisiting with an approved HF account if a stronger technical-Q&A signal is
wanted than the MMLU-subject proxy used here.

If a stronger agentic-coding signal is wanted beyond HumanEval/MBPP's
single-function completion, SWE-bench or LiveCodeBench are the standard
step up — not attempted this session, no compatibility check done.

**Distillation, to make the model smaller/faster:** raised as a question, not
yet attempted for Qwen. Relevant prior evidence from this project: KD-LoRA
(training a LoRA adapter against teacher logits) was tried for Gemma's nf4
*recovery* and the real control killed it — 18% flip-rate improvement bought
nothing on GSM8K (Part 3). That was PTQ-recovery (nudging an already-quantized
model back toward its own bf16 behavior), not the same thing as distilling a
smaller general-purpose student from a larger teacher, but it's the closest
data point this project has, and the lesson generalizes: a technique that
improves a proxy metric (flip rate) is not automatically a technique that
improves the downstream task. Any distillation attempt needs its own control
run before its result is trusted, not compared against a different arm the
way the original KD-LoRA report mistakenly was.

Before building a custom distillation pipeline (real training compute, a
teacher, curated data, and its own eval — a materially bigger undertaking
than anything else in this doc, which has so far only done PTQ/inference-time
work, no training runs): **check Google's official Gemma 4 QAT release first**
(Priority 3 in `NEXT_SESSION.md`, verified to exist this session, not yet
benchmarked). If a vendor-trained, already-distilled/QAT'd checkpoint clears
the bar on the target-use-case benchmarks above, that's a much cheaper win
than training a custom student model. Only reach for custom distillation if
that doesn't pan out.

---

# PART 7 — INFRASTRUCTURE

- **Box:** `ternary-h200` on Brev, nebius `gpu-h200-sxm.1gpu-16vcpu-200gb`,
  141 GB VRAM, $5.40/hr, **stoppable**. `/ephemeral` survives stop/start —
  verified twice. **As of this session: STOPPED and will not restart** —
  two `brev start` attempts each hung 80+ minutes at STOPPED with the
  backend reporting healthy; looks like a provider-side wedge, not something
  fixable from the CLI. Left alone rather than risking `brev reset`, which
  only documents preserving `/home/brev/workspace/`, not `/ephemeral` where
  the Gemma shards actually live. The Qwen3.6-35B-A3B work this session ran
  entirely on a separate, unrelated box (below) — nothing on `ternary-h200`
  was touched or is at risk. Try `brev start ternary-h200` again before
  assuming it's permanently dead; if it's still wedged, the Gemma shards are
  rebuildable in 833s (Part 3) and not worth much recovery effort on their own.
- **We don't need H200 for MoE-side work.** The Qwen3.6-35B-A3B evaluation and
  the memory-constrained throughput test both fit comfortably on a 48GB-VRAM
  box — H200's extra headroom buys nothing here, and at $5.40/hr vs ~$1-2/hr
  for an L40S-class box it's pure waste for this workload. Reserve H200 for
  whatever specifically needs 141GB VRAM.
- **`brev start` on a previously-stopped instance is unreliable in this
  environment; `brev create` (fresh instance) is not.** Both `ternary-h200`
  and one L40S instance got stuck at STOPPED across multiple `start` attempts
  this session; every `brev create` succeeded within seconds. If a box won't
  start within a few minutes, stop retrying `start` and provision a fresh one
  instead — cheap and reliable, versus an open-ended wait on a wedged one.
- **`brev create` without an explicit `--type` can pick a bad SKU under
  transient capacity pressure.** `gpu-l40s-a.1gpu-8vcpu-32gb` failed twice in a
  row this session (real `BUILD=CREATE_FAILED`, not a hang) before switching
  providers (AWS `g6e.xlarge`, then `massedcompute:shadeform`) succeeded. If a
  fresh `create` fails, try a different provider/SKU before assuming the
  whole approach is broken.
- **Not every cheap instance is stoppable — check the FEATURES column in
  `brev search` before committing to a workflow that depends on pause/resume.**
  `massedcompute_L40S` via the `shadeform` broker ($1.06/hr, 625GB disk, 12
  vcpu — used for all of this session's Qwen work) has no `S` in FEATURES and
  is almost certainly create/delete-only. Fine for a single continuous
  unattended run (queue the whole pipeline, no idle-guard needed since nothing
  should sit idle); wrong choice if the workflow needs to pause and resume
  without losing state, which is the exact failure mode Part 1's "previous box"
  bullet already warns about.
- **`pkill -f <pattern>` run over `brev exec`/`ssh` can match its own
  invoking shell** if the pattern string appears in the SSH command's own
  argv (bug 17, Part 5). Prefer killing by PID, or put the kill command in a
  script *file* invoked by a short, non-matching filename instead of inlining
  it — this also sidesteps the broader issue that complex multi-operator
  inline strings through `brev exec` are unreliable (heredocs, `&&`/`&`
  chains, and nested quoting have all silently failed to reach the remote
  shell at some point this session); write a script file and `brev copy` it
  instead whenever a command is more than one or two simple pipe stages.
- **Always provision `--stoppable`** for anything meant to be paused and
  resumed. The Hyperstack 2×A100 box from an earlier session supported
  neither `reset`, `stop` nor `start` — only `delete` — so when it went
  UNHEALTHY the only way to stop billing was to destroy the disk.
- **Disk is only 120 GB** and `save_pretrained` writes bf16 (62.5 GB for
  Gemma), which is why KD-LoRA ran on nf4 rather than our better w4g128.
- `./vm/vmsh 'cmd'` and `./vm/vmcp local :remote` wrap ssh/scp.
  `BREV_HOST=<name>` selects the box.
- **Idle shutdown now runs ON THE BOX**, as a systemd timer, and this is the
  right place for it. Every laptop-side version died when the laptop slept, a
  session ended, or re-arming was forgotten — which cost ~10h of idle A100+H200
  billing once and several more hours later. `/etc/systemd/system/idle-guard.{service,timer}`,
  script at `/ephemeral/work/idle_guard.sh`, fires every 5 min, shuts down after
  3 consecutive idle checks (~15 min).

  Idle requires BOTH signals: GPU memory < 2 GB **and** no python process
  running our scripts. Either alone gives false positives — GPU memory reads
  near-zero while a job loads a 59 GB checkpoint from disk.

  Escape hatch: `touch /ephemeral/work/KEEPALIVE`. Log:
  `/ephemeral/work/logs/idle_guard.log`. **Reinstall this on any new box before
  starting long work.**
- `watchdog.sh` (laptop-side) still exists for live monitoring and continuous
  backup, but is no longer the thing preventing runaway spend.
- **Sync all code before running.** Two runs failed on stale checkouts
  (`--bits` missing, `uniform_ste` missing). md5-verify after copying.
- `git` and `gh` work; two accounts (`TC960` active). Repos:
  `TC960/trickle` (code, branch `research-run-aug23`, PR #1) and
  `TC960/to-learn` (plain-language explainers). **Commit as
  `84581798+TC960@users.noreply.github.com`**, not the BCG email.
- Web search has a per-session cap (200) that compaction does not reset;
  `CLAUDE_CODE_MAX_WEB_SEARCHES_PER_SESSION` raises it for *new* sessions.

## Key files

| file | purpose |
|---|---|
| `airllm_ternary/uniform.py` | asymmetric n-bit quantize/pack/dequantize |
| `airllm_ternary/shard.py` | checkpoint → per-layer shards, incremental writes |
| `airllm_ternary/loader.py` | `ShardStore` + `ResidencyManager` (LRU — replace) |
| `airllm_ternary/model.py` | meta-device load, linear swapping, streaming hooks |
| `vm/stream_bench.py` | bit-exactness + throughput, projected across bandwidths |
| `vm/deep_eval.py` | flip rate, KL, frequency-stratified NLL, `behaviour_delta` |
| `vm/sensitivity.py` | per-layer KL ranking; `--layers` for controls |
| `vm/qlora_recover.py` | KD-LoRA against teacher logits |
| `tests/test_uniform.py` | **the gate**: trained weight == served weight |
| `AUDIT.md` | evidence quality of every claim; bugs 1-13 (14-17 added this session, not yet ported into AUDIT.md) |
| `reports/` | post-PTQ recovery survey (one conclusion withdrawn) |

---

# PART 8 — NOTES TO WHOEVER PICKS THIS UP

Written at the end of the session that produced most of Parts 3–5. These are
patterns, not facts, and they cost real time to learn.

**Check which constraint binds before optimizing anything.** This project spent
two days on bit-widths before measuring throughput, and throughput turned out to
decide everything. The same mistake in miniature: perplexity was optimized
before flip rate, and flip rate before downstream tasks. Every time, the metric
being improved was not the one that mattered. When you find yourself refining a
number, ask what would have to be true for that number to be the bottleneck.

**A test that exercises one component against itself proves almost nothing.**
Bug 13 sat in the ternary path from the first commit and survived every
round-trip test, because pack/unpack really was exact — the fault lived in the
relationship between the quantizer and the storage format. It surfaced only when
a test asserted *the weight training optimizes equals the weight serving
produces*. Same shape as the cosine-similarity failure and the watchdog failure.
Prefer tests that pin two independent components to each other.

**Controls are not bureaucracy.** The KD-LoRA result was reported as a success
and was wrong — the comparison was against a different quantizer because the
real control hadn't run. When it did, the gain vanished. Nothing about the
enthusiasm was dishonest; the baseline was simply absent, and absent baselines
default to whatever flatters.

**The user's tentative hypotheses have beaten my confident assertions
repeatedly.** lm_head streamability, KV streamability, Qwen 3.8's existence,
Muse Glimmer's existence, MoE suiting streaming, and the observation that the
PCIe hop invalidated my throughput projection — all his, several offered with
"I might be talking out of my ass". Verify rather than evaluate from intuition.

**Instrument the instruments.** Three separate times, monitoring reported things
it could not observe: cosine similarity that could not see magnitude error, a
watchdog that could not distinguish a dead box from a busy one, a queue that
reported `exit 0` because a command substitution clobbered `$?`. Plausible
output from a broken instrument is worse than no output.

**A borrowed eval harness is not automatically trustworthy just because it's
somebody else's well-tested code.** `lm_eval` is a widely-used, mature
library; it still produced two confidently-plausible wrong numbers in a row
(43.6%, then 50.3%) for this model specifically, because its defaults and
stop-condition conventions were built around models that don't spontaneously
write chain-of-thought under raw completion. The fix each time required
reading actual generated text, not just trusting an aggregate score — the
same lesson as bug 13, applied to a tool instead of a hand-written quantizer.
Before trusting *any* eval number, especially a surprising one (in either
direction — too low or too high), pull a handful of raw samples and read them.

**What is actually left.** Streaming a dense 31B model is memory-feasible
(3.16 GB resident, bit-exact) and speed-infeasible (15.333 GB/token). The MoE
pivot addresses the binding constraint directly, and as of this session it's
no longer just a hypothesis — Qwen3.6-35B-A3B beats Gemma on the damage-canary
benchmark (94-95% vs 86.0 GSM8K) and remains usable (not fast) under a
Jetson-realistic memory budget (1.52 tok/s at 4GB via plain page-cache
eviction, no custom residency logic yet). The engineering work — a real
residency manager for MoE experts, and the Gated-DeltaNet/linear-attention
streaming support this codebase doesn't have yet — is what's left, not the
question of whether it's worth building. Separately, the user flagged that
GSM8K/MMLU don't measure the three things this deployment is actually for
(technical Q&A, coding, agentic browser use) — see Part 6.5. That's a real
gap, but it's a different question from "is the pivot justified," which is
now answered.
