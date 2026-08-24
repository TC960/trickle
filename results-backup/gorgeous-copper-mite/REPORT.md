# Ternary compression study — results

_Generated 2026-08-22 03:15 UTC_

## Bottom line

**Qwen3.8-27B** (baseline perplexity 6.5241)
- int8 weights: 6.5485 (+0.38%) — effectively free
- nf4 weights: 6.5826 (+0.90%)

**gemma-4-31B** (baseline perplexity 5.1876)
- int8 weights: 5.2132 (+0.49%) — effectively free
- nf4 weights: 5.4484 (+5.03%)

**Best ternary result: smoke-seq = 5.2788 (+1.76%)**

## Weight quantization

| model | method | perplexity | delta | size |
|---|---|---|---|---|
| Qwen3.8-27B | none | 6.5241 | +0.00% | 53.8 GB |
| Qwen3.8-27B | int8 | 6.5485 | +0.38% | 29.4 GB |
| Qwen3.8-27B | nf4 | 6.5826 | +0.90% | 17.3 GB |
| Qwen3.8-27B | ternary-sequential | 8765.4258 | +1.34e+05% | - |
| gemma-4-31B | none | 5.1876 | +0.00% | 62.5 GB |
| gemma-4-31B | int8 | 5.2132 | +0.49% | 32.7 GB |
| gemma-4-31B | ternary-sequential | 5.2788 | +1.76% | - |
| gemma-4-31B | ternary-sequential | 5.2877 | +1.93% | - |
| gemma-4-31B | nf4 | 5.4484 | +5.03% | 17.8 GB |
| gemma-4-31B | ternary-sequential | 6.3106 | +21.65% | - |
| gemma-4-31B | ternary-sequential | 7.7091 | +48.61% | - |
| gemma-4-31B | ternary-sequential | 74.7624 | +1.34e+03% | - |
| gemma-4-31B | ternary-sequential | 97.5467 | +1.78e+03% | - |
| gemma-4-31B | ternary-distilled | 222295.7031 | +4.29e+06% | - |
| gemma-4-31B | ternary-distilled | 155493424.0000 | +3e+09% | - |

## How many layers can go ternary

| blocks ternary | perplexity | delta |
|---|---|---|
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |
| ? | nan | - |

## Embedding table compression

| model | method | perplexity | delta | table |
|---|---|---|---|---|
| Qwen3.8-27B | svd | 6.3725 | -2.32% | 1038.1 MB |
| Qwen3.8-27B | int4 | 6.4920 | -0.49% | 655.6 MB |
| Qwen3.8-27B | int4 | 6.4920 | -0.49% | 655.6 MB |
| Qwen3.8-27B | int4 | 6.5100 | -0.22% | 715.2 MB |
| Qwen3.8-27B | int8 | 6.5203 | -0.06% | 1271.9 MB |
| Qwen3.8-27B | int8 | 6.5203 | -0.06% | 1271.9 MB |
| Qwen3.8-27B | none | 6.5241 | +0.00% | 2542.8 MB |
| gemma-4-31B | none | 5.1876 | +0.00% | 2818.6 MB |
| gemma-4-31B | int8-untied | 5.1877 | +0.00% | 4228.4 MB |
| gemma-4-31B | int4-untied | 5.1885 | +0.02% | 3545.2 MB |
| gemma-4-31B | int8 | 5.1895 | +0.04% | 1409.8 MB |
| gemma-4-31B | int4 | 5.2248 | +0.72% | 792.7 MB |
| gemma-4-31B | int4 | 5.2775 | +1.73% | 726.7 MB |
| gemma-4-31B | svd | 502.5529 | +9.59e+03% | 1095.8 MB |

> **Caveat:** wikitext-2 reads only ~8% of the embedding rows, so these numbers cannot speak to compression of the ~92% it never touches. Treat apparent improvements with suspicion.

## Vocabulary trimming

**gemma-4-31B**: vocab 262144, table 2819 MB, only **54019 tokens (20.6%) actually used** on English+code

**Qwen3.8-27B**: vocab 248320, table 5086 MB, only **41947 tokens (16.9%) actually used** on English+code

| model | vocab | table after | saved | shrink | token inflation | valid |
|---|---|---|---|---|---|---|
| gemma-4-31B | 262144→51269 | 551 MB | 2267 MB | 5.11x | +15.0% | yes |
| gemma-4-31B | 262144→75175 | 808 MB | 2010 MB | 3.49x | +10.0% | yes |
| gemma-4-31B | 262144→75175 | 808 MB | 2010 MB | 3.49x | +10.0% | yes |

## Methodology notes and known limits

- **Perplexity alone is insufficient.** Added flip rate, KL and frequency-stratified NLL for this reason.
- **Bugs found and fixed during the study:** teacher-forced block reconstruction (trained on inputs the blocks never see at inference); cosine similarity as a fidelity metric (scale-invariant, so blind to magnitude drift — real per-block error was ~16%, not 0.01%); tied-embedding detection broken by a clone; QAT parameters allocated on CPU; missing W1.58A8 activation quantization; severed weight tying leaving lm_head on the meta device.
- **Cross-model perplexity is not comparable** (different tokenizers). Bits-per-byte is reported where available and is comparable.
- Calibration: 64x2048 tokens from wikitext train, held out from the test split used for evaluation.

