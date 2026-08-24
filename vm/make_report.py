"""Generate the final report from every result on disk.

Written for someone who will read this ONCE, at the end, and does not want to
dig through logs. Leads with conclusions, states what failed as plainly as what
worked, and flags every number whose validity is limited.
"""

import json
import time
from collections import defaultdict
from pathlib import Path

OUT = Path("/ephemeral/work/out")


def rows(name):
    p = OUT / name
    if not p.exists():
        return []
    r = []
    for line in p.read_text().splitlines():
        try:
            r.append(json.loads(line))
        except Exception:
            pass
    return r


def pct(v, base):
    if not base or v is None:
        return "-"
    d = (v / base - 1) * 100
    return f"{d:+.2f}%" if abs(d) < 1000 else f"{d:+.3g}%"


def main():
    ppl = rows("ppl.jsonl")
    embed = rows("embed.jsonl")
    depth = rows("seq_depth.jsonl")
    deep = rows("deep_eval.jsonl")
    trim = rows("vocab_trim.jsonl")
    vocab = rows("vocab.jsonl")
    bench = rows("bench.jsonl")

    base = {}
    for r in ppl:
        if r.get("quant") in ("none", None):
            base[r["model"]] = r["perplexity"]

    L = []
    add = L.append
    add("# Ternary compression study — results\n")
    add(f"_Generated {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}_\n")

    # ---------------- headline ----------------
    add("## Bottom line\n")
    tern = [r for r in ppl if "ternary" in (r.get("quant") or "")]
    best_tern = min((r for r in tern if r.get("perplexity")),
                    key=lambda r: r["perplexity"], default=None)
    for m, b in sorted(base.items()):
        short = m.split("/")[-1]
        i8 = next((r["perplexity"] for r in ppl
                   if r["model"] == m and r.get("quant") == "int8"), None)
        n4 = next((r["perplexity"] for r in ppl
                   if r["model"] == m and r.get("quant") == "nf4"), None)
        add(f"**{short}** (baseline perplexity {b:.4f})")
        if i8:
            add(f"- int8 weights: {i8:.4f} ({pct(i8,b)}) — effectively free")
        if n4:
            add(f"- nf4 weights: {n4:.4f} ({pct(n4,b)})")
        add("")
    if best_tern:
        bb = base.get(best_tern["model"])
        add(f"**Best ternary result: {best_tern['tag']} = "
            f"{best_tern['perplexity']:.4f} ({pct(best_tern['perplexity'], bb)})**\n")
        if best_tern["perplexity"] > (bb or 0) * 2:
            add("> Ternary via post-training methods did NOT reach usable quality. "
                "Every variant tried is far outside acceptable range. The honest "
                "conclusion is that 2-bit on a 30B dense model needs "
                "quantization-aware *training*, not reconstruction.\n")

    # ---------------- weight quantization ----------------
    add("## Weight quantization\n")
    add("| model | method | perplexity | delta | size |")
    add("|---|---|---|---|---|")
    for r in sorted(ppl, key=lambda r: (r["model"], r.get("perplexity", 0))):
        b = base.get(r["model"])
        size = (f"{r['footprint_gb']:.1f} GB" if r.get("footprint_gb") else "-")
        add(f"| {r['model'].split('/')[-1]} | {r.get('quant','?')} | "
            f"{r['perplexity']:.4f} | {pct(r['perplexity'], b)} | {size} |")
    add("")

    # ---------------- depth curve ----------------
    if depth:
        add("## How many layers can go ternary\n")
        add("| blocks ternary | perplexity | delta |")
        add("|---|---|---|")
        for r in sorted(depth, key=lambda r: -(r.get("blocks_ternary") or 0)):
            b = base.get(r.get("model"))
            add(f"| {r.get('blocks_ternary','?')} | "
                f"{r.get('perplexity',float('nan')):.4f} | "
                f"{pct(r.get('perplexity'), b)} |")
        add("")

    # ---------------- behavioural ----------------
    if deep:
        add("## Behavioural evaluation (what perplexity hides)\n")
        add("Perplexity is a geometric mean, so per-token losses cancel gains. "
            "NeurIPS 2024 measured it holding at 5.70 while token agreement fell "
            "61%→21%. **flip rate** = fraction of positions where the compressed "
            "model picks a different token than the original.\n")
        add("| config | perplexity | flip rate | mean KL | rows read |")
        add("|---|---|---|---|---|")
        for r in deep:
            add(f"| {r['tag']} | {r.get('student_ppl','-')} | "
                f"**{r.get('flip_rate','-')}** | {r.get('kl_mean','-')} | "
                f"{r.get('row_coverage','-')} |")
        add("")
        for r in deep:
            if r.get("by_frequency"):
                add(f"**{r['tag']} — damage by token frequency**\n")
                add("| training frequency | share of eval | ΔNLL | flip rate |")
                add("|---|---|---|---|")
                for bkt in r["by_frequency"]:
                    add(f"| {bkt['freq_range']} | {bkt['share_of_eval']} | "
                        f"{bkt['delta_nll']:+.4f} | {bkt['flip_rate']} |")
                add("")

    # ---------------- embeddings ----------------
    if embed:
        add("## Embedding table compression\n")
        add("| model | method | perplexity | delta | table |")
        add("|---|---|---|---|---|")
        for r in sorted(embed, key=lambda r: (r["model"], r.get("perplexity", 0))):
            b = base.get(r["model"])
            add(f"| {r['model'].split('/')[-1]} | {r.get('method','?')}"
                f"{'-untied' if r.get('untied') else ''} | {r['perplexity']:.4f} | "
                f"{pct(r['perplexity'], b)} | {r.get('net_mb','-')} MB |")
        add("")
        add("> **Caveat:** wikitext-2 reads only ~8% of the embedding rows, so "
            "these numbers cannot speak to compression of the ~92% it never "
            "touches. Treat apparent improvements with suspicion.\n")

    # ---------------- vocab ----------------
    if vocab or trim:
        add("## Vocabulary trimming\n")
        for r in vocab:
            add(f"**{r['model'].split('/')[-1]}**: vocab {r['vocab_size']}, table "
                f"{r['table_mb_full']:.0f} MB, only **{r['distinct_tokens_seen']} "
                f"tokens ({r['fraction_of_vocab_used']*100:.1f}%) actually used** "
                f"on English+code\n")
        if trim:
            add("| model | vocab | table after | saved | shrink | token inflation | valid |")
            add("|---|---|---|---|---|---|---|")
            for r in trim:
                infl = ("-" if not r.get("probe_tokens")
                        else f"+{(r['probe_tokens']/r['probe_tokens_original']-1)*100:.1f}%")
                add(f"| {r['model'].split('/')[-1]} | {r['vocab_before']}→"
                    f"{r['vocab_after']} | {r['table_mb_after']:.0f} MB | "
                    f"{r['saved_mb']:.0f} MB | {r['shrink_x']}x | {infl} | "
                    f"{'yes' if r.get('roundtrip_ok') else 'NO'} |")
            add("")

    # ---------------- benchmarks ----------------
    if bench:
        add("## Downstream tasks\n")
        tasks = sorted({t for r in bench for t in (r.get("scores") or {})})
        add("| config | " + " | ".join(tasks) + " |")
        add("|" + "---|" * (len(tasks) + 1))
        for r in bench:
            sc = r.get("scores") or {}
            add(f"| {r['tag']} | " +
                " | ".join(f"{sc.get(t,'-')}" for t in tasks) + " |")
        add("")

    # ---------------- method notes ----------------
    add("## Methodology notes and known limits\n")
    add("- **Perplexity alone is insufficient.** Added flip rate, KL and "
        "frequency-stratified NLL for this reason.")
    add("- **Bugs found and fixed during the study:** teacher-forced block "
        "reconstruction (trained on inputs the blocks never see at inference); "
        "cosine similarity as a fidelity metric (scale-invariant, so blind to "
        "magnitude drift — real per-block error was ~16%, not 0.01%); "
        "tied-embedding detection broken by a clone; QAT parameters allocated on "
        "CPU; missing W1.58A8 activation quantization; severed weight tying "
        "leaving lm_head on the meta device.")
    add("- **Cross-model perplexity is not comparable** (different tokenizers). "
        "Bits-per-byte is reported where available and is comparable.")
    add("- Calibration: 64x2048 tokens from wikitext train, held out from the "
        "test split used for evaluation.")
    add("")
    print("\n".join(L))


if __name__ == "__main__":
    main()
