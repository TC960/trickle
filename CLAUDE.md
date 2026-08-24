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

**1–6, 12 and 13 produced believable wrong results.** 7–11 crashed loudly.

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

The pivot to **Qwen3.6-35B-A3B** is likely correct:
- 15.6× less I/O per token, the dominant term
- keeps 30B-class capability (an 8B fallback costs ~20–30 GSM8K points)
- user notes Gemma 4 trails Qwen 3.6 on many public benchmarks anyway
- Gemma's own 26B-A4B MoE is not competitive with Qwen 3.6 either

Blocking: **needs ~70 GB, box has 7.7 GB free.** Requires clearing the Gemma
cache. Back up shards first.

## Ranked next steps

1. **Benchmark Qwen3.6-35B-A3B bf16** — GSM8K/MMLU, ~1h. Decides the pivot.
2. **Fixed-partition residency** replacing LRU. Turns a 0% hit rate into a
   guaranteed N/60.
3. **Vocab trimming**, now motivated: the pinned table is the resident floor,
   and 20.6% of tokens are used. Needs an eval that isn't wikitext.
4. **Fused 4-bit kernels** — llama.cpp/GGUF, MLC-LLM or TensorRT-LLM rather
   than hand-written CUDA. Fixes dequantization, not I/O. Matters once the
   model is resident.
5. **The missing controls**: nf4 GSM8K (for KD-LoRA), and `vm/controls.sh`
   (for mixed precision).
6. **GPUDirect Storage / pinned+async copies** to remove the CPU staging hop.

Do not spend more on compression before 1 and 2. The repeated failure mode in
this project has been optimizing a metric before checking which constraint
binds — perplexity before flip rate, bit-width before throughput.

---

# PART 7 — INFRASTRUCTURE

- **Box:** `ternary-h200` on Brev, nebius `gpu-h200-sxm.1gpu-16vcpu-200gb`,
  141 GB VRAM, $5.40/hr, **stoppable**. `/ephemeral` survives stop/start —
  verified twice.
- **Always provision `--stoppable`.** The previous box (Hyperstack 2×A100)
  supported neither `reset`, `stop` nor `start` — only `delete` — so when it
  went UNHEALTHY the only way to stop billing was to destroy the disk.
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
| `AUDIT.md` | evidence quality of every claim; all 13 bugs |
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

**What is actually left.** Streaming a dense 31B model is memory-feasible
(3.16 GB resident, bit-exact) and speed-infeasible (15.333 GB/token). The MoE
pivot addresses the binding constraint directly and nothing else does. Do that
before any further compression work.
