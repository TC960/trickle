# NEXT SESSION — start here

`CLAUDE.md` is the full handoff. This file is the short version: what is true
now, what to do next, and what will waste your time if you rediscover it.

**Nothing is running. No GPU instances exist. Nothing is billing.** Verify with
`brev ls` before assuming otherwise.

---

## 1. The one-paragraph state

The model choice is **settled**: `Qwen3.6-35B-A3B` at **Q4_K_M** (~22 GB). It
beats Gemma 4 31B by ~10 points at every quantization tier, it does not collapse
at 2-bit the way the dense model does, and — measured this session — **4-bit
costs nothing detectable on agentic coding versus 8-bit**. What is *not* settled
is whether it is good enough at the three things the deployment is actually for
(Part 0). One of those three now has a real number; the other two do not.

## 2. Numbers you can trust

| axis | measurement | value |
|---|---|---|
| **Agentic coding** | SWE-bench Verified, n=50, mini-swe-agent, step_limit 100 | **36% resolved (18/50)** |
| same, 8-bit control | Q8_0, identical subset | 32% (16/50), **paired McNemar p=0.73 → null** |
| Damage canary | GSM8K n=1000 (Q4_K_M) | 94.6% |
| Coding (weak proxy) | HumanEval / MBPP | 61.0% / 62.6% |
| Technical Q&A (proxy) | 6 MMLU subjects, custom task | ~83% |
| Agentic browsing | **nothing credible** | 3 hand-written mock pages, 3/3 |

**Quantization ladder, matched quantizer, GSM8K n=1000:**

| tier | Gemma 4 31B | Qwen3.6-35B-A3B |
|---|---|---|
| Q8_0 | 85.3% | 95.3% |
| Q4_K_M | 85.1% | 94.6% |
| Q3_K_M | 83.2% | 95.5% |
| Q2_K | 71.7% | 94.1% |

## 3. Do not redo these — they are answered

- **Gemma vs Qwen.** Settled. Qwen wins everywhere. Note the old "4-bit costs
  Gemma 6 GSM8K points" claim was **our own quantizer's fault**, not 4-bit's —
  with an imatrix quantizer Gemma drops 0.2 points. Any low-bit number from
  `airllm_ternary/uniform.py` describes the quantizer, not the architecture.
- **Expert pruning.** Measured and declined: 96.5–100% of all 256 experts fire,
  and ~50% are needed even within a single domain. Ceiling ~2×, not 5–10×.
  *Reopenable* if per-domain builds are acceptable (see Part 7.5).
- **MLP / channel pruning.** ~25× worse per unit of compression than quantization.
- **Vocabulary trimming.** Measured properly in bits-per-byte. Trimming to 18.3%
  of vocab shrinks embeddings 5.44×, and **wikitext reports it as free (−0.21%)
  while HumanEval reports +132%**. Token counting understates the damage ~18×.
  Verdict: not free on the target domains; burden of proof sits with the trim.
- **FP8 checkpoints cannot be trained.** `w8a8_block_dynamic_fp8_matmul_grouped`
  has no autograd formula. Forward works, backward raises.

## 4. What to do next, in order

1. **Agentic browsing — the big gap.** Your end goal is a job applier and there
   is *no* credible measurement of it. Full research is in this session:
   - **Use WebArena-Lite in `webrl` text mode.** Text-only (no vision — the
     model has no `mmproj`), open-model baselines at our scale, self-hosted.
   - **Skip the `map` site**: ~180 GB for 31 tasks; 134 of 165 tasks don't need
     it and none need Wikipedia.
   - **Exclude the LLM-judged tasks.** ~14% are graded by a judge that, in the
     default config, is *the same local endpoint* — the model grading itself.
     Report the deterministic subset with an honest denominator.
   - **Expect single digits.** GPT-4o scores 13.9% here, Llama-3.1-8B 4.8%,
     humans ~78%. Our 36% on SWE-bench does **not** transfer; this measures
     long-horizon action-format compliance, not code generation.
   - Working scripts are checked in at `vm/swebench/browse_bootstrap.sh`,
     `webarena_sites.sh`, `webarena_harness.sh`. Infra reached "all four sites
     live, model serving, harness installing, 3-task pilot launched" before this
     session ended. Resume from the pilot.
   - **Job applications specifically:** no published benchmark exists. Closest
     is WorkArena's "Forms" category (blocked: needs your own ServiceNow
     developer instance). Best free proxy: **67 of 165 WebArena-Lite tasks are
     graded by `program_html`**, i.e. on page state after a write — structurally
     identical to submitting an application.
2. **Rerun SWE-bench at step_limit 250.** 22 of 50 failures were the step cap,
   not the model. This is cheap now the harness works and should move 36% up.
3. **Fixed-partition residency + the MoE residency manager** (old Part 6 items 2
   and 3). Still the real engineering work for actually shipping on Jetson.
4. **Do not prioritise QLoRA.** Attempted twice, completed zero times (both
   instances reclaimed mid-training). More importantly the probe showed peft can
   reach only **0.047–0.094% of parameters** — attention only, because the 32.2B
   of routed experts are fused 3D tensors it cannot target. Low ceiling.

## 5. Things that will eat your day if you rediscover them

**Harness bugs (all produced confident, wrong numbers):**
- PyPI `sweagent` is a dead 0.0.1 stub depending on a nonexistent
  `togetherunidiff`. Use **mini-swe-agent** — SWE-agent's own README says it has
  superseded SWE-agent, and it is the official Bash-Only leaderboard scaffold.
- mini-swe-agent v2 **defaults to native tool calling**; its own
  `swebench_backticks.yaml` sets backtick prompts but not `model_class`.
  `--model-class litellm_textbased` is mandatory or you get a silent mismatch.
- **Qwen3.6 is a reasoning model.** llama.cpp puts thinking in
  `reasoning_content` and the answer in `content`; under a small token budget
  `content` comes back **empty** and any agent loop sees blank replies.
  Use `--reasoning-budget 0`.
- **mini-swe-agent auto-resumes**, skipping anything already in `preds.json`. A
  stale empty entry makes a rerun execute *zero* instances. `--redo-existing`
  or wipe the output dir.
- **swebench v5 dropped the legacy eval path** — it now fails with
  `KeyError: 'image'`. Use `swebench eval verified -p preds.jsonl --run-id X`,
  and note it wants JSONL while mini-swe-agent writes dict-keyed JSON.

**Box/infra traps:**
- **`llama.cpp` needs `-c` sized for `--parallel`.** The context is *divided*
  across slots. `-c 32768 --parallel 4` gives each agent 8k, which truncates
  every action mid-format. Use `-c 131072 --parallel 4` (32k each). A 0/50 score
  was caused by exactly this and looked like a model failure.
- These images ship **gcc 12.3 but g++ 11.4**, and `cc1plus` exists only for 11,
  so nvcc cannot build. `apt-get install g++-12` and pin
  `CMAKE_C_COMPILER`/`CMAKE_CXX_COMPILER`/`CMAKE_CUDA_HOST_COMPILER`.
- They also ship **an NVIDIA driver but no CUDA toolkit**. Install
  `cuda-toolkit-12-8`.
- **The in-VM idle guard does not stop Brev billing** — Brev restarts what it
  shuts down. Only `brev delete` is reliable, and verify it afterwards. Use a
  *hard wall-clock deadline* guard (in `vm/swebench/bootstrap.sh`) that also
  fires on boot, so a restart after the deadline powers straight back off.
- `brev copy` + `brev exec` chained in one command frequently exceeds a 10-min
  timeout. Copy and exec separately, and prefer one large script over several
  small ones — each round trip costs ~2 min of SSH setup.

**Two bugs of mine, both in the checks meant to catch bugs:**
- A guard requiring `llama-server` ≥100 KB **false-failed a good build** — it is
  an ~18 KB thin executable against `libllama-server-impl.so`. Test with `-x`.
- `"$bin" --version | head -3 || die` **cannot detect a missing binary**: in a
  pipeline `||` binds to `head`, which always succeeds.

## 6. The standing lesson, earned five more times this session

A mature, widely-used eval harness is not trustworthy until verified against
*your* model on *your* stack. Every wrong number this session was believable.
The gate meant to catch them was itself mis-specified — it demanded a non-empty
patch, but an instance that hits the step cap legitimately produces none, so it
condemned a **working** setup. What resolved it was reading the raw trajectory
and seeing the model emit textbook `THOUGHT:` + a bash block with
`returncode 0`.

**Read the raw output. Aggregate scores lie in both directions.**
