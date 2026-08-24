# Final stage on box 1: the PTQ -> QAT bridge, then the report.
cd /ephemeral/work/code
export HF_HOME=/ephemeral/work/hf TOKENIZERS_PARALLELISM=false
PY=/ephemeral/work/venv/bin/python
L=/ephemeral/work/logs

while [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -gt 0 ]; do sleep 120; done
echo "[$(date +%H:%M)] GPUs free -- starting end-to-end scale distillation"

# The bridge between PTQ and full QAT: codes frozen, only the ~230M per-group
# scales trained, objective = KL against the bf16 teacher's logits. This is the
# strongest thing that fits without 375 GB of optimizer state.
for CFG in "0 0" "2 2"; do
  set -- $CFG
  echo "[$(date +%H:%M)] === e2e scales, skip_first=$1 skip_last=$2 ==="
  $PY distill_e2e.py --model google/gemma-4-31B --steps 300 --lr 1e-3 \
    --seq-len 1024 --skip-first $1 --skip-last $2 \
    --tag "gemma-ternary-e2e-s$1$2" > $L/e2e_$1$2.log 2>&1
  grep -E "PERPLEXITY|KL " $L/e2e_$1$2.log | tail -4
done

echo "[$(date +%H:%M)] === generating report ==="
$PY make_report.py > /ephemeral/work/out/REPORT.md 2>&1
wc -l /ephemeral/work/out/REPORT.md
echo "[$(date +%H:%M)] === QUEUE3 DONE ==="
