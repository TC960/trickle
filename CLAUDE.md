# Project instructions

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

Our own data shows the same gap: int8 measures **+0.49% perplexity but changes
3.8% of token choices**; nf4 is +5.03% and **9.2% flips**.

**Report in this order:**

1. **Flip rate** — fraction of positions where the compressed model's argmax
   differs from the bf16 teacher's. This is what a user experiences.
2. **KL divergence** vs the teacher — catches distribution shift perplexity cancels.
3. **Downstream tasks** — MMLU, HellaSwag, ARC-C, and especially **GSM8K**.
   GSM8K is generative, so one corrupted token kills the answer; damage shows up
   there first. Multiple-choice tasks are far more forgiving and can look fine
   on a broken model.
4. **Bits-per-byte** whenever tokenization changes (vocab trimming, or any
   cross-model comparison). Perplexity is NOT comparable across tokenizers.
5. **Perplexity** — last, and only for comparability with published work.

`deep_eval.py` implements 1–2 plus frequency-stratified NLL. `benchmarks.py`
does 3. Use them. Do not ship a headline number that only has perplexity behind
it — that mistake was already made once here and had to be walked back.

Also: WikiText-2 reads only **~8% of the embedding table**, so it structurally
cannot evaluate embedding or vocabulary compression. Use a different eval for
those.

## Other standing rules for this project

- **Verify before asserting.** Search or read primary sources; this project's
  assumptions have been wrong repeatedly (lm_head streamability, KV
  streamability, "Qwen 3.8 doesn't exist", "ternary fails above 3B").
- **Every treatment needs a control**, recorded via `registry.py` with
  `arm`/`pair_id`. `report.py` refuses to display a treatment without one.
- **Single-seed differences under ~5% are not results.** Say so rather than
  presenting them as findings.
- **Local proxies mislead.** Cosine similarity is scale-invariant and hid a 16%
  per-block error. Weight-space error said ternary ≈ 2-bit; end-to-end they
  differ 3.6×. Prefer end-to-end measurement.
- **Sanity-check the pipeline itself.** Quantizing at 8-bit should reproduce
  bf16 within ~0.1%. If it doesn't, every result above it is suspect.
- See `AUDIT.md` for the current evidence quality of every claim.
