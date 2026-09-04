"""A/B report over the registry. Flags any treatment lacking a control."""

import argparse
from collections import defaultdict

from registry import load

METRIC_ORDER = ["perplexity", "bits_per_byte", "mmlu", "hellaswag",
                "arc_challenge", "gsm8k"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="perplexity")
    args = ap.parse_args()

    recs = load()
    if not recs:
        print("registry empty")
        return

    pairs = defaultdict(lambda: {"control": [], "treatment": []})
    for r in recs:
        pairs[r["pair_id"]][r["arm"]].append(r)

    print(f"{len(recs)} records, {len(pairs)} pairs, metric={args.metric}\n")
    orphans = []

    for pid in sorted(pairs):
        p = pairs[pid]
        if not p["control"]:
            orphans.extend(t["tag"] for t in p["treatment"])
            continue
        ctrl = p["control"][-1]
        base = ctrl["results"].get(args.metric)
        if base is None:
            continue
        print(f"=== {pid} ===   control: {ctrl['tag']} = {base:.4f}"
              f"   [code {ctrl['code_sha']}]")
        rows = sorted(p["treatment"],
                      key=lambda r: r["results"].get(args.metric, float("inf")))
        for t in rows:
            v = t["results"].get(args.metric)
            if v is None:
                continue
            d = (v / base - 1) * 100
            print(f"    {t['tag']:<40} {v:>12.4f}  {d:>+9.2f}%  "
                  f"[{t['code_sha']} {t['utc'][5:16]}]")
        print()

    if orphans:
        print("!! TREATMENTS WITH NO CONTROL (not comparable, do not report):")
        for t in sorted(set(orphans)):
            print(f"    {t}")


if __name__ == "__main__":
    main()
