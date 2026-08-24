"""Behavioural evaluation: what perplexity cannot see.

WHY THIS EXISTS
"Accuracy is Not All You Need" (Dutta et al., NeurIPS 2024, Table 19) shows
WikiText-2 perplexity holding at exactly 5.70 while greedy-token agreement falls
from 61.3% to 21.5% under added logit noise. Perplexity is a geometric mean, so
losses on some tokens cancel gains on others. A 2.4% perplexity spread across
quantization schemes covered a ~250x KL range.

Our whole evaluation rested on that one number. This adds the metrics that
actually detect behavioural damage:

  flip_rate     fraction of positions where the compressed model's argmax
                differs from the teacher's -- what the user actually experiences
  kl            KL(teacher || student) over the vocabulary, from top-K mass
  nll_delta     per-token NLL change, STRATIFIED BY TOKEN FREQUENCY, which tests
                the rare-token hypothesis directly instead of assuming it
  row_coverage  how many embedding rows the eval even reads -- if it touches 8%
                of the table, it cannot evaluate compression of the other 92%

Run the teacher first (--save-teacher), then any number of students against it.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F

TOPK = 64  # enough mass for a good KL estimate; keeps the cache small


def token_frequencies(tok):
    """Frequency of each token id in wikitext TRAIN, for stratification."""
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n".join(ds["text"])
    counts = Counter()
    for i in range(0, len(text), 400_000):
        counts.update(tok(text[i:i + 400_000], add_special_tokens=False).input_ids)
    return counts


@torch.inference_mode()
def collect(model, input_ids, window=2048, stride=1024, device="cuda", verbose=True):
    """Per-position statistics, reduced immediately so nothing large is stored."""
    out = {"target": [], "nll": [], "argmax": [], "topk_lp": [], "topk_idx": []}
    prev_end = 0
    for begin in range(0, input_ids.size(1), stride):
        end = min(begin + window, input_ids.size(1))
        new = end - prev_end
        if new <= 0:
            continue
        chunk = input_ids[:, begin:end].to(device)
        logits = model(chunk, use_cache=False).logits[0].float()

        # Score only positions this window newly reveals.
        lo = max(0, logits.shape[0] - new)
        lp = F.log_softmax(logits[lo:-1] if lo < logits.shape[0] - 1 else logits[lo:], dim=-1)
        tgt = chunk[0, lo + 1:end - begin]
        n = min(lp.shape[0], tgt.shape[0])
        lp, tgt = lp[:n], tgt[:n]

        out["target"].append(tgt.cpu())
        out["nll"].append((-lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)).cpu())
        out["argmax"].append(lp.argmax(-1).cpu())
        vals, idx = lp.topk(TOPK, dim=-1)
        out["topk_lp"].append(vals.to(torch.float16).cpu())
        out["topk_idx"].append(idx.to(torch.int32).cpu())

        prev_end = end
        if verbose and begin % (stride * 20) == 0:
            print(f"    {end}/{input_ids.size(1)}", flush=True)
        if end == input_ids.size(1):
            break
    return {k: torch.cat(v) for k, v in out.items()}


def behaviour_delta(teacher, student):
    """(flip_rate, mean KL) between two `collect` outputs.

    The cheap two-number form of `compare`, for callers that need to rank or
    sweep something and cannot afford a scored perplexity pass per point. Same
    top-K convention as `compare`, so the numbers are directly comparable.
    """
    n = min(teacher["argmax"].shape[0], student["argmax"].shape[0])
    flips = (teacher["argmax"][:n] != student["argmax"][:n]).float().mean()

    t_lp, t_idx = teacher["topk_lp"][:n].float(), teacher["topk_idx"][:n].long()
    s_lp, s_idx = student["topk_lp"][:n].float(), student["topk_idx"][:n].long()

    kl = torch.zeros(n, dtype=torch.float64)
    for k in range(t_idx.shape[1]):
        hit = (s_idx == t_idx[:, k:k + 1])
        s_at = torch.where(hit.any(-1), (s_lp * hit).sum(-1),
                           torch.full((n,), -20.0))
        kl += t_lp[:, k].exp().double() * (t_lp[:, k].double() - s_at.double())

    return round(float(flips), 5), round(float(kl.mean()), 6)


def compare(teacher, student, freqs, vocab_size):
    """Behavioural deltas, plus the frequency decomposition."""
    n = min(teacher["nll"].shape[0], student["nll"].shape[0])
    t_nll, s_nll = teacher["nll"][:n].double(), student["nll"][:n].double()
    tgt = teacher["target"][:n]

    flips = (teacher["argmax"][:n] != student["argmax"][:n])

    # KL(teacher || student) restricted to the teacher's top-K support.
    t_lp, t_idx = teacher["topk_lp"][:n].float(), teacher["topk_idx"][:n].long()
    s_lp_full = student["topk_lp"][:n].float()
    s_idx = student["topk_idx"][:n].long()
    # Build a lookup of the student's logprob at the teacher's top-K indices.
    kl_sum = torch.zeros(n, dtype=torch.float64)
    for k in range(t_idx.shape[1]):
        want = t_idx[:, k:k + 1]
        hit = (s_idx == want)
        s_at = torch.where(hit.any(-1),
                           (s_lp_full * hit).sum(-1),
                           torch.full((n,), -20.0))
        p = t_lp[:, k].exp().double()
        kl_sum += p * (t_lp[:, k].double() - s_at.double())

    # Stratify by how often each target token appears in the training corpus.
    buckets = [(0, 10), (10, 100), (100, 1000), (1000, 10000), (10000, 10**12)]
    strat = []
    freq_of = torch.tensor([freqs.get(int(t), 0) for t in tgt], dtype=torch.float64)
    for lo, hi in buckets:
        m = (freq_of >= lo) & (freq_of < hi)
        if m.sum() == 0:
            continue
        strat.append({
            "freq_range": f"{lo}-{hi if hi < 10**12 else 'inf'}",
            "n_tokens": int(m.sum()),
            "share_of_eval": round(float(m.float().mean()), 4),
            "teacher_nll": round(float(t_nll[m].mean()), 5),
            "delta_nll": round(float((s_nll[m] - t_nll[m]).mean()), 5),
            "flip_rate": round(float(flips[m].float().mean()), 5),
        })

    return {
        "n_positions": int(n),
        "teacher_ppl": round(float(t_nll.mean().exp()), 4),
        "student_ppl": round(float(s_nll.mean().exp()), 4),
        "delta_nll_nats": round(float((s_nll - t_nll).mean()), 6),
        "flip_rate": round(float(flips.float().mean()), 5),
        "kl_mean": round(float(kl_sum.mean()), 6),
        "kl_p99": round(float(kl_sum.quantile(0.99)), 6),
        "rows_read": int(torch.unique(tgt).numel()),
        "row_coverage": round(torch.unique(tgt).numel() / vocab_size, 5),
        "by_frequency": strat,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--quant", default="none", choices=["none", "int8", "nf4"])
    ap.add_argument("--embed-method", default=None,
                    choices=[None, "int8", "int4", "svd"])
    ap.add_argument("--embed-group", type=int, default=128)
    ap.add_argument("--embed-rank", type=int, default=2048)
    ap.add_argument("--untie", action="store_true")
    ap.add_argument("--split", default="test",
                    help="wikitext split. Use 'train' to build a teacher for "
                         "KD training that is disjoint from the test set the "
                         "flip rate is reported on.")
    ap.add_argument("--save-teacher", default=None)
    ap.add_argument("--teacher", default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--out", default="/ephemeral/work/out/deep_eval.jsonl")
    args = ap.parse_args()

    from transformers import AutoConfig, AutoTokenizer
    from perplexity import build_model, load_wikitext

    tok = AutoTokenizer.from_pretrained(args.model)
    cfg = AutoConfig.from_pretrained(args.model)
    tcfg = getattr(cfg, "text_config", cfg)
    vocab = getattr(tcfg, "vocab_size")

    tag = args.tag or f"{args.model.split('/')[-1]}-{args.quant}" + \
          (f"-embed{args.embed_method}" if args.embed_method else "")
    print(f"=== deep eval: {tag} ===", flush=True)

    model = build_model(args.model, args.quant)
    if args.embed_method:
        from embed_ablation import apply_method
        st = apply_method(model, args.embed_method, 8 if args.embed_method == "int8" else 4,
                          args.embed_group, args.embed_rank, args.untie)
        print("  embed:", json.dumps(st), flush=True)

    ids, _nb = load_wikitext(tok, split=args.split)
    stats = collect(model, ids, device=next(model.parameters()).device)

    if args.save_teacher:
        torch.save(stats, args.save_teacher)
        print(f"  saved teacher stats -> {args.save_teacher}", flush=True)

    if args.teacher:
        print("  computing frequency table...", flush=True)
        freqs = token_frequencies(tok)
        teacher = torch.load(args.teacher)
        rec = {"tag": tag, "model": args.model, "quant": args.quant,
               "embed_method": args.embed_method, "untie": args.untie,
               **compare(teacher, stats, freqs, vocab)}
        print("\n" + json.dumps({k: v for k, v in rec.items()
                                 if k != "by_frequency"}, indent=2), flush=True)
        print("\n  by token frequency in training corpus:")
        print(f"    {'freq':<14}{'share':>8}{'t_nll':>9}{'d_nll':>10}{'flips':>9}")
        for b in rec["by_frequency"]:
            print(f"    {b['freq_range']:<14}{b['share_of_eval']:>8.3f}"
                  f"{b['teacher_nll']:>9.3f}{b['delta_nll']:>+10.4f}"
                  f"{b['flip_rate']:>9.4f}")
        with open(args.out, "a") as h:
            h.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    main()
