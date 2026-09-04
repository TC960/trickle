#!/bin/bash
# Score with the swebench v5 CLI. The legacy
# `python -m swebench.harness.run_evaluation --dataset_name princeton-nlp/...`
# path fails with KeyError: 'image' -- v5 expects its own dataset build, which
# carries the container image reference. `swebench eval verified` resolves the
# right dataset via its alias.
set -ux
W=$HOME/swebench
RUNID=${RUNID:-qwen_q4km_bashonly}
cd "$W"
source venv/bin/activate

# v5 wants JSONL; mini-swe-agent writes a dict-keyed JSON.
python3 - <<'PY'
import json, os
W = os.path.expanduser("~/swebench")
d = json.load(open(f"{W}/results/preds.json"))
rows = [{"instance_id": k, **v} for k, v in d.items()]
with open(f"{W}/results/preds.jsonl", "w") as f:
    f.write("\n".join(json.dumps(r) for r in rows))
print("rows:", len(rows), "with patch:",
      sum(1 for r in rows if (r.get("model_patch") or "").strip()))
PY

swebench eval verified -p "$W/results/preds.jsonl" --run-id "$RUNID" -j 4 2>&1 | tail -40
echo "SCORE_EXIT=$?"
echo "=== report files ==="
find "$HOME" -maxdepth 3 -name "*${RUNID}*" -newermt '-2 hours' 2>/dev/null | head
touch "$W/SCORED"
