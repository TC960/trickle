# Bit-width sweep through the sequential drift-aware pipeline.
# Weight-level reconstruction error says the cliff is between 2 and 3 bits:
#   ternary 0.5159 | 2-bit 0.5118 | 3-bit 0.2212 | 4-bit 0.1022
# So the question is not "can we rescue ternary" but "where does this model
# actually become viable". Same pipeline, same calibration, only bits change.
cd /ephemeral/work/code
export HF_HOME=/ephemeral/work/hf TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/ephemeral/work/venv/bin/python
L=/ephemeral/work/logs
M=google/gemma-4-31B

wait_idle () { local n=0
  while [ "$n" -lt 4 ]; do
    if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ]
      then n=$((n+1)); else n=0; fi
    sleep 30
  done; }

for B in 3 2 4; do
  wait_idle
  echo "[$(date +%H:%M)] ===== ${B}-bit, all 60 blocks, sequential ====="
  $PY distill_seq.py --model $M --bits $B --steps 200 --n-calib 32 \
    --tag "gemma-w${B}g128-seq" > $L/seq_w$B.log 2>&1
  grep -E "PERPLEXITY|block +5?9:" $L/seq_w$B.log | tail -2
done

$PY make_report.py > /ephemeral/work/out/REPORT.md 2>&1
echo "[$(date +%H:%M)] === QUEUE6 DONE ==="
