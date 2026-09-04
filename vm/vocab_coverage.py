"""How much of a 262k vocabulary does real text actually use?

The embedding table is the memory bottleneck, and its size is set by vocab size.
Gemma-4-31B carries 262,144 tokens for multilingual + code coverage. If a
deployment only ever sees English and code, most of those rows never fire -- and
a row that never fires can be dropped with ZERO quality cost on that domain, by
definition. That makes this a measurement, not a quality tradeoff.

Reports the coverage curve: how many distinct tokens account for 99%, 99.9%,
99.99% of all occurrences, and the exact table size at each cut.
CPU only -- no GPU needed.
"""

import argparse
import json
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="google/gemma-4-31B,Qwen/Qwen3.8-27B")
    ap.add_argument("--out", default="/ephemeral/work/out/vocab.jsonl")
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoConfig, AutoTokenizer

    # Deliberately mixed: general prose plus code, i.e. a realistic English-plus-
    # code deployment. Multilingual text is excluded on purpose -- that is the
    # coverage we are asking about dropping.
    corpus = []
    wt = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    corpus.append("\n".join(wt["text"][:40000]))
    try:
        code = load_dataset("code_search_net", "python", split="train",
                            streaming=True, trust_remote_code=True)
        snippets = []
        for i, row in enumerate(code):
            if i >= 4000:
                break
            snippets.append(row.get("whole_func_string") or "")
        corpus.append("\n".join(snippets))
        print(f"  code: {len(snippets)} snippets")
    except Exception as exc:
        print(f"  (code corpus unavailable: {type(exc).__name__}; prose only)")

    text = "\n\n".join(corpus)
    print(f"  corpus: {len(text)/1e6:.1f} M chars", flush=True)

    for model_id in args.models.split(","):
        tok = AutoTokenizer.from_pretrained(model_id)
        cfg = AutoConfig.from_pretrained(model_id)
        tcfg = getattr(cfg, "text_config", cfg)
        vocab = getattr(tcfg, "vocab_size", tok.vocab_size)
        hidden = getattr(tcfg, "hidden_size", 0)
        tied = bool(getattr(tcfg, "tie_word_embeddings", False))

        counts = Counter()
        step = 400_000
        for i in range(0, len(text), step):
            counts.update(tok(text[i:i + step], add_special_tokens=False).input_ids)

        total = sum(counts.values())
        ordered = counts.most_common()
        used = len(ordered)

        # bf16 bytes; a tied model pays once, an untied model pays twice.
        copies = 1 if tied else 2
        full_mb = vocab * hidden * 2 * copies / 1e6

        rows = []
        cum = 0
        marks = {0.99: None, 0.999: None, 0.9999: None, 1.0: used}
        for idx, (_tid, c) in enumerate(ordered, 1):
            cum += c
            for frac in (0.99, 0.999, 0.9999):
                if marks[frac] is None and cum / total >= frac:
                    marks[frac] = idx
        for frac, keep in marks.items():
            kept_mb = keep * hidden * 2 * copies / 1e6
            rows.append({"coverage": frac, "tokens_kept": keep,
                         "table_mb": round(kept_mb, 1),
                         "saved_mb": round(full_mb - kept_mb, 1),
                         "shrink_x": round(full_mb / max(kept_mb, 1e-9), 2)})

        rec = {"model": model_id, "vocab_size": vocab, "hidden": hidden,
               "tied": tied, "table_mb_full": round(full_mb, 1),
               "distinct_tokens_seen": used,
               "fraction_of_vocab_used": round(used / vocab, 4),
               "total_token_occurrences": total, "curve": rows}
        print(f"\n=== {model_id} ===")
        print(f"  vocab {vocab}, table {full_mb:.0f} MB, tied={tied}")
        print(f"  distinct tokens actually used: {used} "
              f"({used/vocab*100:.1f}% of vocab)")
        for r in rows:
            print(f"  {r['coverage']*100:>7.2f}% coverage -> keep {r['tokens_kept']:>6} "
                  f"tokens = {r['table_mb']:>7.1f} MB "
                  f"(saves {r['saved_mb']:>7.1f} MB, {r['shrink_x']}x smaller)")
        with open(args.out, "a") as h:
            h.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
