# Mixed-precision: profile every layer, then allocate by measured tolerance.
cd /ephemeral/work/code
export HF_HOME=/ephemeral/work/hf TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/ephemeral/work/venv/bin/python
L=/ephemeral/work/logs
M=google/gemma-4-31B

# Single-writer lock: only one queue may hold the GPUs at a time.
LOCK=/ephemeral/work/.gpu.lock
while ! mkdir "$LOCK" 2>/dev/null; do
  echo "[$(date +%H:%M)] waiting for GPU lock held by $(cat $LOCK/owner 2>/dev/null)"
  sleep 120
done
echo "queue4 $$" > "$LOCK/owner"
trap 'rm -rf "$LOCK"' EXIT
# And require the cards to look genuinely idle for 3 consecutive checks, since
# a process can be starting up and not yet registered.
IDLE=0
while [ "$IDLE" -lt 3 ]; do
  if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ]; then
    IDLE=$((IDLE+1))
  else
    IDLE=0
  fi
  sleep 20
done
echo "[$(date +%H:%M)] === per-layer sensitivity profile (60 layers) ==="
$PY sensitivity.py --model $M --mode profile > $L/sens_profile.log 2>&1
tail -24 $L/sens_profile.log

# Allocation sweep against the SAME ranking, so the comparison is clean.
for N in 20 30 40 45; do
  echo "[$(date +%H:%M)] === mixed precision: $N tolerant layers ternary ==="
  $PY sensitivity.py --model $M --mode allocate --n-ternary $N \
    > $L/mixed_$N.log 2>&1
  grep -E "MIXED|ternarizing" $L/mixed_$N.log | tail -2
done

$PY make_report.py > /ephemeral/work/out/REPORT.md 2>&1
echo "[$(date +%H:%M)] === QUEUE4 DONE ==="
