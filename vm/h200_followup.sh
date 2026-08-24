# Runs after the main H200 queue. Re-does the 8-bit sanity check with the
# behavioural metrics attached.
#
# The first run passed on perplexity (5.1889 vs 5.1876, +0.025%), which is the
# criterion CLAUDE.md states -- but perplexity is precisely the metric this
# project distrusts, and int8 on this same model measured +0.49% perplexity
# against a 3.8% flip rate. A pipeline that reproduces bf16 perplexity while
# changing 1% of token choices is not a faithful pipeline, and the first run
# could not tell the difference.
set -u
cd /ephemeral/work/code
export HF_HOME=/ephemeral/work/hf TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/ephemeral/work/venv/bin/python
L=/ephemeral/work/logs
M=google/gemma-4-31B
T=/ephemeral/work/out/teacher_gemma.pt

step () { local name=$1 log=$2; shift 2
  echo "[$(date +%H:%M)] === $name ===" | tee -a $L/h200_queue.log
  "$@" > "$log" 2>&1
  # Capture immediately: a $(...) inside the echo runs first and clobbers $?,
  # which made every failing step report "exit 0" on 2026-08-22.
  local rc=$?
  echo "[$(date +%H:%M)]     exit $rc  ($log)" | tee -a $L/h200_queue.log
  tail -5 "$log" | tr '\r' '\n' | tail -5 | sed 's/^/      /' | tee -a $L/h200_queue.log
  return 0; }

[ -f "$T" ] || step "bf16 teacher stats" $L/hf_teacher.log \
  $PY deep_eval.py --model $M --quant none --save-teacher $T --tag gemma-bf16-teacher

step "8-bit sanity check WITH flip rate + KL" $L/hf_sanity_w8_behav.log \
  $PY distill_seq.py --model $M --bits 8 --steps 0 --n-calib 8 \
     --teacher $T --tag gemma-w8g128-sanity-behav

echo "[$(date +%H:%M)] === H200 FOLLOWUP DONE ===" | tee -a $L/h200_queue.log
