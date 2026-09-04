# Serialized: wait for EVERY GPU process (including q2's benchmarks), then run
# the two experiments that matter, one at a time.
cd /ephemeral/work/code
export HF_HOME=/ephemeral/work/hf TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/ephemeral/work/venv/bin/python
L=/ephemeral/work/logs

wait_idle () {
  local n=0
  while [ "$n" -lt 4 ]; do
    if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ]; then
      n=$((n+1)); else n=0; fi
    sleep 30
  done
}

echo "[$(date +%H:%M)] waiting for all GPU work to finish..."
wait_idle
echo "[$(date +%H:%M)] GPUs idle"

# EXPERIMENT 1 -- the literature's predicted 2x: learnable THRESHOLD, not just
# scale. PV-Tuning Table 1 says scale-only plateaus; moving the assignment does
# not. `mu` moves the assignment.
echo "[$(date +%H:%M)] === e2e ternary: learnable scale + THRESHOLD ==="
$PY distill_e2e.py --model google/gemma-4-31B --steps 300 --lr 1e-3 \
  --seq-len 512 --tag gemma-ternary-e2e-mu > $L/e2e_mu.log 2>&1
grep -E "PERPLEXITY|step +[0-9]+" $L/e2e_mu.log | tail -6

wait_idle
# EXPERIMENT 2 -- which layers are actually tolerant, vs my crude "first N".
echo "[$(date +%H:%M)] === per-layer sensitivity profile ==="
$PY sensitivity.py --model google/gemma-4-31B --mode profile > $L/sens_profile.log 2>&1
tail -22 $L/sens_profile.log

for N in 20 30 40; do
  wait_idle
  echo "[$(date +%H:%M)] === mixed precision: $N most-tolerant layers ternary ==="
  $PY sensitivity.py --model google/gemma-4-31B --mode allocate --n-ternary $N \
    > $L/mixed_$N.log 2>&1
  grep -E "MIXED|ternarizing" $L/mixed_$N.log | tail -2
done

$PY make_report.py > /ephemeral/work/out/REPORT.md 2>&1
echo "[$(date +%H:%M)] === QUEUE5 DONE ==="
