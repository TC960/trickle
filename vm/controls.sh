# The matched controls for the mixed-precision comparison.
#
# sensitivity-ranked selection scored 210.56 / 9584 / 50431 at N=30/40/45. It
# is tempting to set that against the old depth sweep's 7.71 at N=30 and call
# ranked selection 27x worse -- but that sweep ran distill_seq with 200 steps
# of block reconstruction, and this path is pure round-to-nearest. Comparing
# them measures reconstruction, not selection.
#
# These runs ternarize the FIRST N layers through the identical code path, so
# the only thing that differs is which layers were chosen.
set -u
cd /ephemeral/work/code
export HF_HOME=/ephemeral/work/hf TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/ephemeral/work/venv/bin/python
L=/ephemeral/work/logs
M=google/gemma-4-31B

step () { local name=$1 log=$2; shift 2
  echo "[$(date +%H:%M)] === $name ===" | tee -a $L/h200_queue.log
  "$@" > "$log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M)]     exit $rc  ($log)" | tee -a $L/h200_queue.log
  tail -4 "$log" | tr '\r' '\n' | tail -4 | sed 's/^/      /' | tee -a $L/h200_queue.log
  return 0; }

for N in 30 40 45; do
  FIRST=$(python3 -c "print(','.join(str(i) for i in range($N)))")
  step "CONTROL: first-$N layers ternary, no reconstruction" $L/h_ctrl_$N.log \
    $PY sensitivity.py --model $M --mode allocate --layers "$FIRST"
done

echo "[$(date +%H:%M)] === CONTROLS DONE ===" | tee -a $L/h200_queue.log
