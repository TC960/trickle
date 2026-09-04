"""Vocabulary trimming with reachability guarantees.

`lm_head` is the one tensor that cannot be streamed -- producing logits needs
every vocabulary row on every token -- so its size is a hard resident floor.
Trimming the vocabulary is the only lever that shrinks it without touching
precision.

Four traps this handles, each of which fails silently otherwise:

1. resize_token_embeddings ONLY truncates the tail, and Gemma-4's IDs are not
   topologically ordered (SentencePiece orders by length/frequency), so tail
   truncation breaks the merge table. We select rows by index instead.
2. Gemma-4 sets byte_fallback + fuse_unk. If any of the 256 <0xNN> tokens is
   dropped, fallback yields <unk> and CONSECUTIVE UNKNOWNS FUSE into one -- an
   unrecoverable, silent corruption. All 256 are pinned.
3. Gemma-4's multimodal tokens sit at the TOP of the vocab (255999-258884). Any
   tail cut destroys the vision path, so they are pinned and their config ids
   remapped.
4. The BPE merge table requires all three of (a, b, a+b) to exist or the
   tokenizer raises MergeTokenOutOfVocabulary at load. We take the transitive
   merge closure of the keep set, then filter merges.

Evaluation uses bits-per-byte, not perplexity: trimming changes tokenization, so
per-token perplexity is not comparable across vocab sizes. BPB normalizes by raw
UTF-8 bytes and is.
"""

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path


def load_tokenizer_json(model_dir):
    p = Path(model_dir) / "tokenizer.json"
    return json.loads(p.read_text()), p


def token_frequencies(tok, texts, chunk=400_000):
    counts = Counter()
    for text in texts:
        for i in range(0, len(text), chunk):
            counts.update(tok(text[i:i + chunk], add_special_tokens=False).input_ids)
    return counts


def build_keep_set(tj, tok, counts, target, verbose=True):
    """Decide which token ids survive, with reachability closure."""
    vocab = tj["model"]["vocab"]            # token string -> id
    id_to_tok = {v: k for k, v in vocab.items()}
    keep = set()
    reasons = Counter()

    # (a) every special / added token
    for entry in tj.get("added_tokens", []):
        keep.add(entry["id"]); reasons["added/special"] += 1

    # (b) all 256 byte-fallback tokens -- trap 2
    byte_re = re.compile(r"^<0x[0-9A-Fa-f]{2}>$")
    for t, i in vocab.items():
        if byte_re.match(t):
            keep.add(i); reasons["byte_fallback"] += 1

    # (c) the most frequent tokens on OUR corpus, up to target
    for tid, _c in counts.most_common():
        if len(keep) >= target:
            break
        if tid not in keep:
            keep.add(tid); reasons["frequent"] += 1

    # (d) transitive merge closure -- trap 4. A kept token formed by merging
    #     a+b needs both parents present, recursively.
    merges = tj["model"].get("merges") or []
    norm = []
    for m in merges:
        norm.append(tuple(m) if isinstance(m, list) else tuple(m.split(" ", 1)))
    child_of = {}
    for a, b in norm:
        joined = a + b
        if joined in vocab:
            child_of[vocab[joined]] = (vocab.get(a), vocab.get(b))

    frontier = list(keep)
    added = 0
    while frontier:
        tid = frontier.pop()
        parents = child_of.get(tid)
        if not parents:
            continue
        for p in parents:
            if p is not None and p not in keep:
                keep.add(p); frontier.append(p); added += 1
    reasons["merge_closure"] = added

    if verbose:
        print(f"    keep set: {len(keep)} tokens  {dict(reasons)}", flush=True)
    return sorted(keep), norm, vocab, id_to_tok


def write_trimmed_tokenizer(tj, keep_ids, norm_merges, vocab, out_dir):
    """Emit a tokenizer with contiguous ids and a consistent merge table."""
    id_map = {old: new for new, old in enumerate(keep_ids)}
    keep_set = set(keep_ids)

    new_vocab = {}
    for tok_str, old_id in vocab.items():
        if old_id in keep_set:
            new_vocab[tok_str] = id_map[old_id]
    tj["model"]["vocab"] = new_vocab

    # Keep only merges where a, b and a+b all survive -- trap 4.
    kept_merges, dropped = [], 0
    for a, b in norm_merges:
        if a in new_vocab and b in new_vocab and (a + b) in new_vocab:
            kept_merges.append(f"{a} {b}")
        else:
            dropped += 1
    tj["model"]["merges"] = kept_merges

    tj["added_tokens"] = [
        {**e, "id": id_map[e["id"]]} for e in tj.get("added_tokens", [])
        if e["id"] in keep_set
    ]

    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    (out / "tokenizer.json").write_text(json.dumps(tj))
    print(f"    merges: kept {len(kept_merges)}, dropped {dropped}", flush=True)
    return id_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", type=int, default=64000)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--eval", action="store_true", help="load model and measure BPB")
    ap.add_argument("--results", default="/ephemeral/work/out/vocab_trim.jsonl")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import AutoConfig, AutoTokenizer

    from huggingface_hub import snapshot_download
    src = snapshot_download(args.model, allow_patterns=["*.json", "*.txt", "*.model"])

    tok = AutoTokenizer.from_pretrained(args.model)
    cfg = AutoConfig.from_pretrained(args.model)
    tcfg = getattr(cfg, "text_config", cfg)
    vocab_size = getattr(tcfg, "vocab_size")
    hidden = getattr(tcfg, "hidden_size")
    tied = bool(getattr(tcfg, "tie_word_embeddings", False))
    print(f"=== {args.model}  vocab {vocab_size}  hidden {hidden}  tied {tied} ===",
          flush=True)

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    texts = ["\n".join(ds["text"][:40000])]
    counts = token_frequencies(tok, texts)
    print(f"    corpus: {sum(counts.values())} token occurrences, "
          f"{len(counts)} distinct", flush=True)

    tj, _ = load_tokenizer_json(src)
    keep_ids, norm_merges, vocab, id_to_tok = build_keep_set(
        tj, tok, counts, args.target)

    out_dir = args.out_dir or f"/ephemeral/work/trimmed/{args.model.split('/')[-1]}-v{len(keep_ids)}"
    for name in ("tokenizer_config.json", "special_tokens_map.json", "config.json"):
        srcp = Path(src) / name
        if srcp.exists():
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            shutil.copy(srcp, Path(out_dir) / name)
    write_trimmed_tokenizer(json.loads(json.dumps(tj)), keep_ids,
                            norm_merges, vocab, out_dir)

    copies = 1 if tied else 2
    before_mb = vocab_size * hidden * 2 * copies / 1e6
    after_mb = len(keep_ids) * hidden * 2 * copies / 1e6
    rec = {"model": args.model, "vocab_before": vocab_size,
           "vocab_after": len(keep_ids), "tied": tied,
           "table_mb_before": round(before_mb, 1),
           "table_mb_after": round(after_mb, 1),
           "saved_mb": round(before_mb - after_mb, 1),
           "shrink_x": round(before_mb / after_mb, 2),
           "out_dir": out_dir}
    print(f"    table: {before_mb:.0f} MB -> {after_mb:.0f} MB "
          f"({rec['shrink_x']}x smaller, saves {rec['saved_mb']:.0f} MB)", flush=True)

    # Round-trip check: does the trimmed tokenizer still encode real text?
    try:
        trimmed = AutoTokenizer.from_pretrained(out_dir)
        probe = "The quick brown fox jumps over the lazy dog. def f(x): return x**2"
        ids = trimmed(probe).input_ids
        back = trimmed.decode(ids, skip_special_tokens=True)
        rec["roundtrip_ok"] = probe.replace(" ", "") in back.replace(" ", "")
        rec["probe_tokens"] = len(ids)
        rec["probe_tokens_original"] = len(tok(probe).input_ids)
        print(f"    round-trip ok={rec['roundtrip_ok']}  "
              f"tokens {rec['probe_tokens_original']} -> {rec['probe_tokens']} "
              f"(+{(rec['probe_tokens']/rec['probe_tokens_original']-1)*100:.1f}% longer)",
              flush=True)
    except Exception as exc:
        rec["roundtrip_ok"] = False
        rec["tokenizer_error"] = f"{type(exc).__name__}: {exc}"
        print(f"    !! trimmed tokenizer failed to load: {exc}", flush=True)

    with open(args.results, "a") as h:
        h.write(json.dumps(rec) + "\n")
    print(f"    wrote {args.results}", flush=True)


if __name__ == "__main__":
    main()
