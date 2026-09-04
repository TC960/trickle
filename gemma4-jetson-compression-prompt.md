# Claude Code Mega-Prompt: Gemma4 → Jetson Orin Nano Compression Program

> Paste everything below the line into your Claude Code session.

---

## ROLE & POSTURE

You are acting as a senior ML systems engineer specializing in model compression and edge inference. I need you to be **adversarial and critical**, not agreeable. Specifically:

- If a track I describe is unlikely to work, say so early and explain the mechanism of failure, don't just execute it.
- Challenge my assumptions explicitly. I have already burned significant time on a failed pipeline and I would rather be told "this is the wrong lever" than get polite execution of a bad plan.
- When you are uncertain, distinguish clearly between "I know this" and "this is an estimate/hypothesis that needs measurement."
- Do not fabricate benchmark numbers, paper results, or API details. If you need a number, measure it or mark it as unknown.
- Prefer measurement over reasoning whenever a measurement is cheap. Do not theorize about something you could profile in 20 minutes.

---

## HARDWARE CONSTRAINT (non-negotiable)

- **Target device:** NVIDIA Jetson Orin Nano, 8GB unified memory (shared between CPU and GPU).
- **Realistic weight budget:** ~5GB. The remainder must cover OS, runtime, CUDA context, KV cache, and activations.
- **Critical:** Orin Nano is the most memory-*bandwidth*-constrained device in the Jetson family. Autoregressive decoding here is bandwidth-bound, not compute-bound. **Fitting in memory does not imply usable speed.** Every track below must report projected and measured tokens/sec, not just memory footprint.
- Storage is eMMC/NVMe class, single-digit GB/s at best. Any design that requires moving large amounts of weight data per token is dead on arrival — prove otherwise with measured bandwidth before building on it.

---

## EXECUTION ENVIRONMENT — I DO NOT HAVE THE JETSON ON HAND RIGHT NOW

All work happens on **cloud GPU (NVIDIA Brev credits)**. The Orin Nano is the deployment target, not the development environment. Split all work accordingly:

**PHASE 1 — do all of this now, on cloud GPU:**
- Full eval harness construction
- Track A in its entirety (diagnosis, recalibration, controlled re-runs, before/after evals)
- Stock-E4B baseline measurement
- Track C steps 1–8 (calibration sets, routing profiling, concentration/overlap analysis, pruning, router retraining, KD, quantization, KV cache quantization)
- Track B step 1 (sparsity-vs-quality curve, predictor training)

**PHASE 2 — blocked until I have the device. Do NOT attempt to fake these numbers:**
- Real measured tokens/sec
- True memory budget under actual runtime + OS overhead
- Track B's storage→memory bandwidth gate (needs real eMMC/SD/NVMe measurement)
- Checkpoint swap latency for the multi-specialist design

**Emulate the memory constraint during Phase 1.** Cap the cloud GPU to 8GB visible memory — via a container VRAM cap, or `torch.cuda.set_per_process_memory_fraction`, or equivalent — so every candidate is forced to confront the real capacity limit during development rather than at deployment. Build this cap into the test harness so it applies automatically, not as a manual step I have to remember. This does **not** emulate bandwidth, only capacity — be explicit about that limitation in any report.

**Estimate the bandwidth ceiling analytically in the meantime.** Look up the published memory bandwidth spec for the specific Orin Nano variant (the original 8GB and the "Super" refresh differ meaningfully — do not guess, and tell me if you can't confirm which I have). Then compute `theoretical_max_tok_per_sec ≈ memory_bandwidth ÷ model_size_in_memory`, and apply a 50–70% real-world derating. Report this as a **projected ceiling with explicit uncertainty**, clearly labeled as an estimate, not a measurement. Use it to sanity-check whether each candidate's footprint is even in the right neighborhood.

---

## TARGET WORKLOAD (this narrowness is my main advantage — exploit it)

I do **not** need a general-purpose model. I need exactly three domains:

1. **Agentic tool-calling / browsing** — structured function calls, multi-turn tool loops, error recovery, JSON/XML call formats.
2. **Coding** — generation, debugging, repo-level context.
3. **Technical English / ML / deep-tech domain content** — reading and reasoning over technical material.

Everything else (multilingual, creative writing, general trivia, vision beyond what's needed) is **dead weight I am willing to destroy**. Design around this. A generalist model provider structurally cannot take this tradeoff; I can. This asymmetry is the core thesis of the whole program — if a decision doesn't exploit it, question whether it's the right decision.

---

## WHAT I HAVE ALREADY TRIED AND FAILED AT

I built an end-to-end pipeline on **Gemma4-31B (dense, ~30.7B params)** combining: QAT, LoRA/PEFT, PTQ (GPTQ), knowledge distillation, and mixed-precision experiments.

**Result:** Stuck at ~15.33GB. That is almost exactly 30.7B × 0.5 bytes = clean 4-bit weight-only quantization. So the pipeline works, but 4-bit is the practical floor for a dense 31B and it's still ~2x my budget.

**Also:** GPTQ + mixed precision attempts produced badly degraded quality. Perplexity blew up; downstream benchmarks were also poor (I did check beyond perplexity — I know perplexity is a weak proxy post-quantization).

I already have working infrastructure for: KD, PTQ, QAT, LoRA, FlashAttention, and paged KV cache management.

**Streaming per-token weights is ruled out for the dense 31B** — dense means every parameter is touched every forward pass, so streaming means moving 15GB/token. Do not re-propose this for the dense model.

---

# THE THREE TRACKS

Execute these as **separate, parallel investigations**. Do not let one block the others. Track A is a prerequisite for trusting anything else, so start there, but it should be fast.

---

## TRACK A — Forensic post-mortem of the failed quantization pipeline

**Goal:** Determine *why* the GPTQ/QAT/LoRA/KD stack degraded so badly, before I repeat the same mistake on a smaller model. This is cheap and high-value; do it first.

Investigate each of these hypotheses **against my actual code and configs** (ask me for files, don't assume):

1. **Order of operations.** Did I merge LoRA into full-precision weights and *then* quantize? That compounds error badly. The correct QLoRA-style pattern is: quantize the base model first, then train adapters on top of the frozen quantized base in higher precision. Verify which order my pipeline actually used.

2. **Outlier channel handling.** Dense transformers reliably have a small number of high-magnitude activation channels. Plain GPTQ without salience-aware protection chokes on these. Check whether my config did anything analogous to AWQ's activation-magnitude-based channel protection. If not, this is a prime suspect — re-run with AWQ or GPTQ-with-outlier-protection as a controlled comparison.

3. **Calibration data mismatch.** GPTQ's calibration set determines which activation ranges are preserved. If I calibrated on generic web text but deploy on tool-calling JSON + code + technical English, I optimized quantization for the wrong distribution. **This is likely a significant and easily-fixed error.** Rebuild the calibration set from my three target domains and re-run. Report the delta.

4. **Unprotected sensitive layers.** Check whether embeddings, final layer norms, and (critically) any router weights were quantized. These are far more sensitive than mid-network FFN weights. Recommend a mixed-precision layer allocation that keeps these in fp16/bf16.

5. **Evaluation methodology.** Confirm my eval harness isn't itself the problem — check prompt templates match the model's expected chat format, sampling params match the model card recommendations, and that quantized-model evals aren't being run with settings tuned for the fp16 model.

**Deliverable:** A written diagnosis ranking these by likelihood, with a controlled re-run of the most promising fix and a measured before/after on my domain benchmarks (not perplexity alone). If you conclude my pipeline was actually correct and 4-bit-dense-31B is simply infeasible, say that plainly.

---

## TRACK B — Contextual sparsity predictor + custom inference loop

**Goal:** Test whether dense-model activation sparsity can be converted into an actual memory/bandwidth win via predictive weight loading.

**Background I already understand — don't re-explain:** I measured ~20% activation density on GSM8K/MMLU/coding. I understand this is *post-hoc* sparsity (discovered after the matmul) and is therefore not directly exploitable — you can't know a neuron's output is zero without computing it. The only way to exploit it is a cheap predictor that runs *before* the FFN matmul and predicts which output neurons will be hot, so you can skip loading/computing the cold rows.

**Reference implementations to study:** Deja Vu, PowerInfer, and Apple's "LLM in a Flash" (the last is specifically about memory-constrained devices and is the closest analogue to my situation). Read what these actually do before writing code.

**Tasks:**

1. **Re-measure sparsity properly and honestly.** Gemma4 uses gated activations (SwiGLU/GeGLU family), which produce small-but-nonzero values, not hard ReLU zeros. My "20%" was almost certainly a magnitude threshold I chose. Re-measure across a sweep of thresholds and report the **quality-vs-sparsity curve**, not a single number. Specifically: at what threshold does downstream task accuracy start dropping, and what's the achievable sparsity at acceptable quality? Do this on my three target domains, not GSM8K.

2. **Feasibility gate — do this before building anything.** Measure actual sustained storage→unified-memory bandwidth on my Orin Nano. Then compute: at the measured achievable sparsity, how much data must move per token, and what tokens/sec does that imply? **If this is below ~5 tok/s, stop and report — the track is dead and I want to know immediately.** Published implementations typically achieve 2-4x effective reduction, not the naive 5x, once predictor misses and fallback costs are counted. Use the pessimistic number.

3. **If the gate passes:** train per-layer low-rank predictors that take the FFN input and predict hot output neurons. Report predictor accuracy (precision/recall on hot-neuron identification) per layer. Note that gated activations make this measurably harder than the ReLU models most papers target — if predictor accuracy is poor, that's a real finding, report it rather than tuning indefinitely.

4. **Custom inference loop.** Only if steps 1-3 look viable. Note that llama.cpp's mmap gives passive OS-page-cache benefit but does *not* do predictive skipping — I need explicit predict-then-load. Check whether PowerInfer's codebase can be adapted to Gemma4's architecture and whether anyone has run it on Jetson unified memory (I suspect not — verify rather than assume).

**Kill criteria (respect these, don't sink time past them):** predictor accuracy below ~85%, or projected tok/s below 5, or adaptation to Gemma4 architecture requiring more than a few days of work → write up the negative result and stop. A clean negative result is a valid deliverable here.

---

## TRACK C — Domain-specialized expert pruning (PRIMARY TRACK — highest expected value)

**Target model: Gemma4-26B-A4B (the MoE variant).**

**IMPORTANT — do not confuse the model variants:**
- **Gemma4-26B-A4B** = MoE with router + experts → **this is the pruning target**.
- **Gemma4-E4B** = dense edge model using Per-Layer Embeddings, **no experts, no router, nothing to prune** → E4B is only relevant as a *distillation student* in the fallback plan, never as an expert-pruning target.

**Core idea — read this carefully, it is the central architectural bet of the program:**

The pruned MoE is **not** required to fit entirely in 8GB. The design is a **hot/cold expert split**:

- **Hot experts** — the subset that covers the overwhelming majority of routing mass for a given domain — stay resident in the Orin Nano's 8GB.
- **Cold experts** live remotely (network-attached host, or local NVMe) and are streamed in only when the router selects them.

**This makes the problem a cache hit-rate problem, not a per-token bandwidth problem.** If routing profiling shows 90-95% of tokens route entirely within the resident hot set, the system only stalls on the tail. This is fundamentally different from the dense-model streaming case (where every parameter is touched every token and streaming is hopeless), and the distinction matters — do not conflate them.

**Prior art to study and differentiate from:** PowerInfer does hot/cold splitting at the *neuron* level against SSD. This design applies the same principle at the *expert* level, with a network tier, on edge hardware, with domain-specialized hot sets. Verify whether this specific combination has been done; if it has, learn from it, and if it hasn't, that is a point in favor of the work being novel rather than a warning sign.

**Critical measurements this design lives or dies on — get these early:**
- **Hot-set hit rate per domain** at various resident-set sizes. This is THE number. Plot hit rate vs. resident memory.
- **Cold-expert fetch latency** over the actual transport (network or NVMe). A stall is only acceptable if it's tens of ms, not seconds.
- **Effective tok/s** accounting for stall frequency × stall cost.
- Whether hot sets are **stable within a session** or churn (churn destroys the model).

**Additionally**, build 2-3 separately pruned+quantized checkpoints, one per domain, swapped at session or task-classification boundaries — never per token. Hot-set composition will differ per domain; that's expected and is the point.

**Pipeline:**

1. **Build a proper calibration/profiling set per domain.** This must be large and diverse *within* each domain — varied phrasings, edge cases, error-recovery paths, realistic tool-call formats (actual JSON/XML function calls as they'll appear in deployment, not prose descriptions of tools). Garbage calibration data here poisons everything downstream. Tell me if what I provide looks insufficient.

2. **Profile expert routing.** Run forward passes over each domain's calibration set. For every MoE layer, log which experts are selected and with what frequency. Produce per-layer, per-domain routing histograms.

3. **Analyze routing concentration — this determines whether the whole track is worth it.** Report, per domain: what fraction of experts covers 90% / 95% / 99% of routing mass? **If routing is roughly uniform across experts, this track yields little and you should tell me so immediately.** Also check: how much do the three domains' expert sets *overlap*? High overlap means one shared checkpoint might suffice; low overlap validates the multi-checkpoint plan. Report this explicitly — it changes the architecture of the solution.

4. **Prune.** Per layer, keep experts covering ~90-95% of routing mass for that domain; delete the rest entirely (whole expert weight matrices). Leave attention and shared/dense components untouched.

5. **Retrain the router — do NOT simply truncate it.** This is the step that's commonly skipped and then blamed on the technique. The router's softmax was trained over N experts; silently dropping columns leaves it distributing probability over a landscape it wasn't trained for. Run a short fine-tune (order hundreds to low-thousands of steps) so the router re-learns correct mass distribution over surviving experts.

6. **KD recovery pass. Teacher = Gemma4-31B dense. Student = my pruned 26B-A4B.**

   Use **logit / soft-target distillation**. The only hard requirement for this is a **shared tokenizer/vocabulary** between teacher and student — hidden dims, layer count, and dense-vs-MoE differences are irrelevant for logit KD. **Verify vocab compatibility between Gemma4-31B and Gemma4-26B-A4B before writing any code.** They're the same model family so this should hold, but confirm rather than assume; if it doesn't hold, fall back to the unpruned 26B-A4B as teacher and tell me.

   Use the 31B rather than the unpruned 26B-A4B because the 31B is meaningfully stronger on nearly every benchmark. This means the goal is not merely *recovering* pruning damage but potentially exceeding stock 26B-A4B quality on my three domains.

   Additionally: log the **unpruned 26B-A4B's** outputs on the eval set as a reference baseline, so I can distinguish "pruning damage recovered" from "pruning damage masked by a stronger teacher." It is a measuring stick, not the teacher.

   I already have KD infrastructure — retarget it rather than rebuilding.

7. **Quantize to 4-bit — with the router protected.** Keep router weights in fp16/bf16 even when everything else goes to 4-bit. Router logits feed a discrete top-k selection, so quantization noise can flip which experts are chosen, cascading into an entirely different and wrong computation path. This is not an optional refinement. Apply all Track A findings (calibration data from target domain, outlier protection, correct LoRA/quantization ordering, sensitive-layer protection) here.

8. **Also quantize the KV cache** (4 or 8-bit) — I have paged KV infrastructure already, this is cheap headroom.

**Sizing hypothesis to test (this is an estimate, verify it, don't trust it):** MoE expert/FFN layers typically dominate total params while attention stays shared. If profiling shows ~1/3 of experts suffice per domain, ~26B total could drop into the 10-14B range per specialist → ~5-7GB at 4-bit. That lands in budget. **Report the real number from actual profiling; if it doesn't reach the budget, say so rather than fudging the pruning ratio to hit a target at the cost of quality.**

**Also build:** a lightweight task classifier to route incoming requests to the correct specialist checkpoint, plus measured checkpoint swap latency (load time from storage) so I know the real cost of a domain switch.

---

---

# TRACK 0 — EVAL HARNESS AUDIT & DOMAIN BENCHMARK REBUILD

**This blocks everything else. No compression work proceeds until Track 0 is done.** I recently produced a comparison table whose numbers I cannot trust, and I do not currently know basic facts about my own eval setup. Every decision downstream depends on measurement I can believe.

**Use parallel subagents for this track.** Spawn separate subagents for the audit, the statistics work, and each domain's benchmark construction, then synthesize. These are independent workstreams and should not run serially.

## Subagent 0.1 — Audit the existing eval harness

Read my existing eval code and answer these concretely. Do not guess; if the code doesn't make something clear, say "cannot determine from code" and tell me what to check manually.

1. **Sample size per benchmark.** How many questions is each benchmark actually running? Full set or subset? (I suspect a subset of roughly 200 — scores moving in 0.5 increments imply n≈200.)
2. **Chat template.** Is the model's chat template being applied, or is the model receiving raw text where it expects instruct formatting?
3. **Shot count.** Zero-shot or few-shot? MMLU is conventionally reported 5-shot; zero-shot scores materially lower.
4. **Thinking mode.** Gemma4 has a configurable thinking/reasoning mode. Is it on? Do the reference numbers I'm comparing against assume it's on?
5. **Scoring method.** Likelihood comparison over answer choices, or parsing generated text? These give different numbers and different failure modes.
6. **Sampling params.** Do they match the model card's recommendations, and are they held identical across all arms being compared?

**Known red flag to investigate:** my bf16 baseline scored 82.99 on plain MMLU, but Gemma4-31B reportedly scores 85.2 on MMLU-**Pro**, which is the *harder* benchmark. Plain MMLU should come out higher, typically ~88-90. Something in the setup is not matching reference conditions. Find it.

## Subagent 0.2 — Statistical validity pass

1. Compute the standard error for each benchmark at the actual sample size in use.
2. Determine which differences in my existing results are distinguishable from noise and which are not. My working assumption is that at n≈200 and ~75% accuracy, SE is roughly ±3 points, meaning nearly every difference I recorded is inside the noise band and supports **no conclusion**.
3. Specify the minimum sample size needed to detect a difference I'd actually care about (say, 2 points) at reasonable confidence.
4. Design the eval protocol going forward: required n, number of seeds, whether to report confidence intervals or run significance tests. **Every result table produced from here on must carry error bars or explicit CIs.** No more bare point estimates.

## Subagents 0.3 / 0.4 / 0.5 — Build real domain benchmarks (one subagent each)

MMLU, HellaSwag, ARC-C, and GSM8K are **the wrong benchmarks for me** and should be demoted to secondary regression checks, not decision metrics. They measure general knowledge and commonsense multiple-choice; I deploy on agentic tool-calling, coding, and technical English. HellaSwag in particular is saturated and near-useless as a discriminator.

Build a real eval suite, one subagent per domain:

**0.3 — Agentic tool-calling.** This is my highest-risk domain and needs the most rigor.
- **Schema adherence must be a first-class metric**, not a footnote: valid JSON/XML rate, correct function name selection, correct argument names, correct argument types, required-args-present rate.
- Multi-turn trajectory success (did the whole task complete, not just individual calls).
- Error recovery behavior when a tool returns an error or unexpected output.
- **Rationale — take this seriously:** compression degrades strict format adherence *before* it degrades knowledge recall. A model can hold 82.97 MMLU while emitting malformed tool calls, and a knowledge benchmark would show nothing. In an agent loop a single malformed call kills the whole trajectory. This is the most likely way a compressed model looks fine on paper and is unusable in practice.
- Consider existing suites (BFCL / Berkeley Function-Calling Leaderboard, τ-bench style multi-turn) as starting points, adapted to my actual tool formats.

**0.4 — Coding.** Real pass rates on executable tests, in my target languages. HumanEval/MBPP are acceptable starting points but are saturated and contaminated — prefer harder, more recent sets, and include repo-context tasks if feasible since that's closer to my use.

**0.5 — Technical English / ML domain.** Comprehension and reasoning over real technical material in my domain, not general trivia.

## Subagent 0.6 — Flip-rate instrumentation

Flip rate is the % of individual questions where the compressed model's answer differs from the bf16 reference, in either direction. It is valuable because aggregate accuracy hides behavioral churn: 5% right→wrong plus 4% wrong→right nets to almost no accuracy change while the model behaves substantially differently.

- **Report flips directionally: right→wrong and wrong→right as separate numbers.** A single symmetric number obscures the thing I need to see.
- Extend flip rate to the new domain benchmarks, not just multiple-choice.
- For agentic tasks specifically, measure trajectory-level divergence, not just final-answer flips.
- **Treat high churn as a warning even when accuracy is flat.** In agentic use, a single wrong step kills a trajectory, so behavioral instability matters more than average-case accuracy.

## Track 0 deliverables

1. A written diagnosis of what was wrong with the previous eval setup.
2. A rebuilt harness: correct chat templating, adequate n, multi-seed, error bars, domain benchmarks primary and general benchmarks secondary.
3. **Re-run of my previous comparison (bf16 / nf4 control / nf4+KD-LoRA / w4g128) on the corrected harness**, with a memory-footprint column added — the original table had no footprint column, which makes it impossible to judge whether my custom w4g128 scheme buys anything over off-the-shelf NF4. Note that w4g128 lost to the NF4 control on every metric in the original table; if that holds up on a valid harness, w4g128 should be dropped.
4. A clear statement of which of my previous conclusions survive and which were noise.

---

## TRACK A.5 — HEAD-TO-HEAD AGAINST GOOGLE'S OFFICIAL QAT RELEASE

**Do this immediately after Track 0, before any further custom compression work.** It may invalidate or redirect large parts of the program, and it is cheap.

**Context:** Google released official QAT versions of Gemma 4 on approximately June 5, 2026, reportedly cutting memory ~72% while holding quality within a few points of FP16. I built my own QAT/KD/PTQ/LoRA pipeline without knowing this. I need to know whether I reinvented something Google already did better.

**Tasks:**

1. **Find and verify the official QAT release.** Locate Google's QAT checkpoints on Hugging Face / Kaggle for the Gemma 4 variants relevant to me. Confirm which variants have QAT versions, what format they ship in (note: some are LiteRT-LM mobile format rather than standard GGUF, which affects whether I can actually use them on Jetson), and their actual measured memory footprints. **Report what you can verify and flag what you cannot — do not assume the ~72% figure is accurate for my specific variant.**

2. **Establish what Google evaluated on.** Find their published methodology: which benchmarks, what sample sizes, whether thinking mode was on, what the FP16 baseline was. If their eval set differs materially from mine, note that their "within a few points" claim may not transfer to agentic tool-calling and coding — which is exactly my domain and exactly where quantization damage tends to concentrate.

3. **Run the head-to-head on MY Track 0 harness.** Same benchmarks, same sample sizes, same seeds, error bars on everything. Arms to compare:
   - Google official QAT
   - My nf4 control
   - My nf4 + KD-LoRA
   - My custom w4g128
   - bf16 reference
   
   Add a **memory footprint column** and, where possible, a projected tok/s column. The absence of a footprint column made my original comparison uninterpretable.

4. **Answer these questions explicitly:**
   - Does Google's QAT beat my pipeline on *general* benchmarks? (Likely yes — they had more compute.)
   - Does it beat my pipeline on *my three domains specifically*? (Not obvious — they optimized for general use, I can optimize narrowly. This is the whole thesis of the program.)
   - Is the gap explained by technique, or by compute/data scale? Be specific about which.
   - Should I abandon custom quantization and build on top of Google's QAT checkpoints instead?

5. **If Google's QAT wins outright:** say so plainly and recommend adopting it as the base for all downstream work (pruning, KD, domain specialization) rather than continuing custom quantization. Building domain specialization on top of a strong official QAT base is a perfectly good outcome and is not a failure.

**This comparison is a legitimate deliverable regardless of outcome.** A rigorous, well-evaluated "here is where my hand-rolled pipeline beat and lost to the vendor's official release, and here is why" is a real result. Write it up as such.

---

## BASELINE MEASUREMENT (do this in the first day — it is not a track, it is a floor)

**Do NOT build a 31B → E4B distillation project.** Generic distillation into E4B would likely just reproduce what Google already shipped — I don't have more compute or better general data than they did for the general case.

Instead, do the cheap version: **take stock Gemma4-E4B (dense, ~4B, ~2GB at 4-bit), quantize it, and run it through the eval harness on my three domains.** This should be roughly an hour of work.

Why this matters:
- If stock quantized E4B already scores acceptably on agentic tool-calling, coding, and technical English, **the entire Track C project may be unnecessary** and I need to know that before investing weeks.
- If it's badly insufficient (likely on the agentic side), this quantifies exactly how much lift Track C must deliver to justify itself.

Calibration on expectations: stock E4B roughly matches Gemma3-27B overall, but the gap to 31B on hard multi-step reasoning is large (E4B ~42.5% vs 31B ~89.2% on AIME 2026). Knowledge and perception tasks transfer far better than compositional reasoning and long-horizon agentic planning — so expect E4B to look worse on agentic/tool-calling than on technical reading comprehension.

Report this baseline number alongside every subsequent result so I can always see what the cheap option would have given me.

---

## CROSS-CUTTING REQUIREMENTS

- **Track 0 gates everything.** No compression work starts until the eval harness is audited and rebuilt. I have already been burned by numbers I couldn't trust; measuring badly is worse than not measuring, because it produces confident wrong decisions.
- **Perplexity is explicitly not an acceptable primary metric** — it correlates poorly with downstream quality after aggressive quantization. I already learned this the hard way.
- **Domain benchmarks are the decision metrics. MMLU/HellaSwag/ARC-C/GSM8K are secondary regression checks only.** Never make a go/no-go call on the general benchmarks.
- **Every result table carries error bars or explicit confidence intervals, and states its sample size.** Bare point estimates are not acceptable output.
- **Use subagents aggressively for parallelizable work** — harness audit, per-domain benchmark construction, per-layer routing profiling, and independent track investigations should all run in parallel rather than serially. Synthesize results centrally.
- **Profile on the actual Orin Nano early and often.** Do not build the entire pipeline and discover tok/s at the end. Get an early on-device number for whatever the current best candidate is.
- **Report memory as a full budget breakdown** — weights + KV cache at target context length + activations + runtime/CUDA overhead + OS. Not just weight footprint. A common failure is hitting the weight target exactly and then getting OOM'd by KV cache at long context.
- Every track reports **both** memory footprint and measured tokens/sec.
- Maintain a running decision log of what was tried, what the measured result was, and what was ruled out with the reason.

---

## FIRST RESPONSE — what I want from you before you write any code

1. Your critical assessment of this plan. What's wrong with it? What am I over-optimistic about? Which track would you kill?
2. Your ranking of the three tracks by expected value, with reasoning.
3. What files, configs, and data you need from me to start Track A.
4. Any assumption in the above that you think is factually wrong — including anything about the Gemma4 variants, the hardware, or the techniques. Correct me.

Do not start implementing until we've agreed on the plan.
