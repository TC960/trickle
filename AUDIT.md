# Evidence audit — every claim, and how much it's actually worth

Written because the user had to catch that our headline bit-width curve was
perplexity-only. That should not have been his job. This is a systematic pass
over every claim made in this project, self-critical by design.

**Legend:** ✅ solid · ⚠️ partial · ❌ not established

---

## A. Claims that hold up

| claim | evidence | status |
|---|---|---|
| Streaming engine is bit-exact | 4/4 completions token-identical, `max\|Δ\|=0` across all 30 layers to logits (BitNet-2B) | ✅ |
| **Streaming is bit-exact at 31B** | `max\|Δlogit\| = 0.00e+00` streaming Gemma 4 at a 4 GB budget vs fully resident. First time the two halves of this project were connected | ✅ |
| BitNet is W1.58A8, not weight-only | found by debugging; fixing it changed garbage → coherent | ✅ |
| int8 nearly free on weights | +0.49% ppl AND 3.8% flip rate, both measured | ✅ |
| nf4 costs Gemma 5.6× more than Qwen | +5.03% vs +0.90%, same code, same eval | ✅ |
| ~80% of vocabulary never fires | direct token counting, two models | ✅ |
| Vocab trimming = only 3.6% of params | computed from the actual checkpoint | ✅ |
| MLP is 67.5% of params | computed from the actual checkpoint | ✅ |
| Weight-space error mispredicts end-to-end | ternary/2-bit errors nearly equal (0.530/0.535) yet ppl differs 3.6× | ✅ |
| Frequency-damage direction is method-dependent | nf4 hurts rare 28× more; SVD hurts common most | ✅ |

## B. Claims that are weaker than I presented them

| claim | the problem | status |
|---|---|---|
| **Bit-width curve (the headline)** | **perplexity ONLY.** No flip rate, no GSM8K. On the metric I spent the session arguing is inadequate | ⚠️ being fixed |
| "4-bit beats nf4" | 5.3433 vs 5.4484 — a **1.9% relative gap, single seed, no variance estimate.** Could be noise | ⚠️ |
| Ternary depth knee | perplexity only; and it ternarized the FIRST N layers, conflating position with tolerance | ⚠️ |
| Per-layer sensitivity ranking | profiled on a **truncated eval** (200k chars) for speed — may rank layers wrongly | ⚠️ |
| Qwen downstream numbers | `--limit 250` per task, so noisy; and only the bf16 baseline ever completed | ⚠️ |
| Embedding "untied" results | the `was_tied` pointer bug made `--untie` a **no-op** for every run after the clone fix | ⚠️ suspect |

## C. Things I claimed or implied that are NOT established

| gap | why it matters |
|---|---|
| **No inference speed measured for any Gemma config** | Streaming is a throughput trade. We have zero tok/s numbers for the thing we're building |
| **GPTQ never validated** | Implementation is correct at `H=I` but has never beaten round-to-nearest on real data |
| **Learnable-threshold experiment never completed** | OOM'd repeatedly; the PV-Tuning prediction is still untested |
| **Mixed precision never ran** | The most promising config, still queued |
| **Vocab trimming never evaluated for quality** | We have size numbers and a valid tokenizer; zero quality measurements |
| **No published baseline reproduced** | The GuidedQuant 2-bit Gemma eval OOM'd. We have no external reference point measured by us |
| **Qwen ternary = 8765 ppl, never discussed** | I ran it and never reported it. It reinforces the ternary conclusion and I skipped it |
| **Single seed everywhere** | No run repeated. No error bars anywhere. Small gaps may be noise |
| **No baseline BPB** | bf16 runs predate the BPB metric, so cross-tokenizer comparisons are incomplete |

## D. Bugs found in my own code during this project

Recorded because the rate matters: thirteen, and several produced
*plausible-looking wrong numbers* rather than crashes.

1. Missing W1.58A8 activation quantization → fluent garbage
2. Severed `tie_word_embeddings` → `lm_head` left on meta device
3. Teacher-forced reconstruction → trained on inputs blocks never see
4. Cosine similarity as fidelity metric → scale-invariant, hid 16% real error
5. `reconstruction_cosine` aliasing → always returned exactly 1.0
6. Clone fix broke `was_tied` → `--untie` silently became a no-op
7. QAT params allocated on CPU → device mismatch crash
8. Teacher logits stored full-vocab → 161 GB
9. Queue races → repeated OOMs misdiagnosed as memory-math errors
10. `mlp_prune.py` cloned all 60 blocks' MLP weights to allow restore — a
    second copy of 20.81B params (~41 GB) → OOM after 3h of useful work
11. `perplexity()` passed `labels=` to the model, so transformers ran
    cross-entropy over 2048×262144 upcast to fp32 (2.1 GB in one tensor) → OOM
    on any 80 GB card
12. **The watchdog reported liveness it could not observe** (see G)
13. **Scale-precision mismatch in the quantizers** (see H) -- present in the
    ternary path from the very first commit

**Of these, 1–6 and 12 produced believable but wrong results.** Only 7–11
crashed loudly. That ratio is the argument for the reference-comparison harness.

Fix 11 changes how every perplexity number in this project is computed, so it
was verified rather than assumed: on gpt2 the chunked path agrees with the
reference `labels=` path to **3.9e-08 relative**, and end-to-end perplexity
matches to 7 significant figures (29.912254 vs 29.912252). `perplexity()` now
runs that cross-check automatically on the first window of every call.

## G. The monitoring told me things it could not know (2026-08-22)

`autoshutdown.sh` was 40 lines and had three faults that compounded into the
loss of ~5 hours of unrecoverable results:

1. `B1=${B1:-1}` — ssh returning nothing (box unreachable) was assigned the
   same value as "one GPU process running". An instance that had silently died
   was indistinguishable from one hard at work.
2. Liveness came from `nvidia-smi --query-compute-apps=pid`, which returns
   nothing inside these VMs regardless of load. Box2 logged `procs=0` for hours
   while it was actively running benchmarks. The "both idle" exit condition
   could therefore never fire on either box.
3. Backups ran only at the start and the end of the window, so when a box died
   mid-run everything since the last sync went with it.

**The failure that matters is not the lost compute, it is that I read those log
lines the next morning and reported "the GPU was busy the whole time" as fact.**
It was an artifact of fault 1. Same class of error as the cosine-similarity
episode: an instrument that cannot observe what it claims to report, trusted
because it produced plausible output.

Replaced by `watchdog.sh`: three explicit states (`busy`/`idle`/`unreachable`,
never conflated), load measured by GPU memory rather than PIDs, and backup every
poll cycle.

Separately, `gorgeous-copper-mite` proved unrecoverable — SSH, ping, `brev
refresh`, `reset`, `stop`, `start`, and `brev exec` all failed while Brev
reported it RUNNING. That provider exposes only `delete`. Lesson recorded in
`watchdog.sh`: **provision with `--stoppable`**, or the only way to end a
billing state is to destroy the disk.

## H. Codes quantized against one scale, reconstructed with another

`quantize_ternary` computed codes from an **fp32** scale and then stored that
scale as **bf16**. Dequantization used the stored bf16 value. So every weight
was reconstructed against a scale ~0.4% different from the one its code was
derived from -- a systematic displacement of a fraction of a quantization step,
not rounding noise. The same fault appeared in the new `quantize_uniform`.

Fixed in both by rounding the scale to its storage precision **first**, then
deriving codes and zero points from it, so quantize -> store -> dequantize is
self-consistent. `qat.py`'s straight-through estimators now round to bf16 too,
so training optimizes the quantizer that actually ships.

**Consequence for prior results:** every ternary number was measured with the
mismatch present, making them pessimistic rather than flattering; `distill_seq`
4-bit numbers trained against fp32 scales, making them slightly optimistic
relative to anything streamed. Headline conclusions are unaffected -- 4-bit
viable, ternary dead -- but exact figures want re-measuring.

**Why it survived this long:** the round-trip tests passed throughout. Packing
and unpacking were always exact; the bug lived in the relationship between the
quantizer and the storage format, which no round-trip test can see. It surfaced
only when a test asserted that the weight *training* optimizes equals the weight
*serving* produces. Two attempted fixes made the error worse (8e-03 -> 4e-01)
before the real cause became visible.

The general lesson, and the third instance of it this project: a test that
exercises one component against itself proves far less than a test that pins two
components to each other.

## E. What would actually make this defensible

Ranked by how much each closes a real hole:

1. **Flip rate + GSM8K on the 3-bit and 4-bit artifacts** — running now
2. **Stream a quantized Gemma end to end and verify bit-exactness** — closes the project's original goal, still unproven at scale
3. **Repeat one config with a different seed** — turns "4-bit beats nf4" from anecdote into a result, or kills it
4. **Full-size benchmarks** (not `--limit 200`) on the final artifact only
5. **Re-run the untied embedding configs** now that `was_tied` is fixed
6. **Measure tok/s** for at least the final config

## F. The most likely thing still wrong

Given nine bugs found, assume more exist. The highest-risk unverified area is
the **sequential reconstruction pipeline** — it produced every headline number,
and its only correctness check is that perplexity looks reasonable. There is no
reference implementation to compare against, unlike the streaming engine, which
had one and was proven exact.

A cheap guard: run it at 8-bit, where output should be nearly identical to bf16.
If 8-bit sequential reconstruction does NOT land within ~0.1% of 5.1876, the
pipeline has a bug affecting every result above.
