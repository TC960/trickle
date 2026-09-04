"""Import results collected before the registry existed, with correct A/B pairing."""
import json
import pathlib

import registry

OUT = pathlib.Path("/ephemeral/work/out")


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


def classify(rec, kind):
    """Assign the pair and arm. Controls are the uncompressed references."""
    model = rec.get("model", "?")
    short = model.split("/")[-1]
    tag = rec.get("tag", "?")
    if kind == "embed":
        pair = f"{short}::embedding"
        arm = "control" if rec.get("method") == "none" else "treatment"
    else:
        pair = f"{short}::whole-model"
        arm = "control" if rec.get("quant") in ("none", None) else "treatment"
    return pair, arm, tag


seen = 0
for name, kind in (("ppl.jsonl", "perplexity"), ("embed.jsonl", "embed")):
    for rec in rows(name):
        if rec.get("tag", "").startswith("smoke"):
            continue
        pair, arm, tag = classify(rec, kind)
        results = {k: rec[k] for k in
                   ("perplexity", "bits_per_byte", "footprint_gb", "net_mb",
                    "embed_mb_after", "peak_gpu_gb", "eval_seconds",
                    "tokens_scored", "reconstruction_cosine")
                   if k in rec and rec[k] is not None}
        params = {k: rec[k] for k in
                  ("quant", "method", "group_size", "rank", "untied", "steps",
                   "window", "stride", "blocks_distilled")
                  if k in rec and rec[k] is not None}
        registry.record(tag=tag, kind=kind, model=rec.get("model", "?"),
                        arm=arm, pair_id=pair, params=params, results=results,
                        notes="backfilled: predates registry")
        seen += 1
print(f"backfilled {seen} records")
