# Box2 chain, round 2.
#
# Round 1 failed on a stale checkout: --bits and --save-dir were added to
# distill_seq.py on box1 and never synced here, so the whole 4-bit chain died
# on an argparse error. All 19 files are now md5-verified identical to local.
#
# MLP pruning re-runs first because round 1 reported it on PERPLEXITY ONLY --
# the exact failure CLAUDE.md exists to prevent. It now reports flip rate and
# KL against the bf16 teacher, with perplexity kept last for comparability.
set -u
cd /ephemeral/work/code
export HF_HOME=/ephemeral/work/hf TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/ephemeral/work/venv/bin/python
L=/ephemeral/work/logs
M=google/gemma-4-31B
D=/ephemeral/work/models/gemma4-w4g128
T=/ephemeral/work/out/teacher_gemma.pt

step () { local name=$1 log=$2; shift 2
  echo "[$(date +%H:%M)] === $name ===" | tee -a $L/box2_queue.log
  "$@" > "$log" 2>&1
  # Capture immediately: a $(...) inside the echo runs first and clobbers $?,
  # which made every failing step report "exit 0" on 2026-08-22.
  local rc=$?
  echo "[$(date +%H:%M)]     exit $rc  ($log)" | tee -a $L/box2_queue.log
  tail -4 "$log" | tr '\r' '\n' | tail -4 | sed 's/^/      /' | tee -a $L/box2_queue.log
  return 0; }

# MLP pruning already completed with flip rate + KL at 22:22; not repeated.

step "gemma 4-bit quantize + save" $L/q2b_save_w4.log \
  $PY distill_seq.py --model $M --bits 4 --steps 200 --n-calib 32 \
     --save-dir $D --tag gemma-w4g128-saved

step "gemma 4-bit flip rate + KL" $L/q2b_behav_w4.log \
  $PY deep_eval.py --model $D --teacher $T --tag gemma-w4g128

step "gemma 4-bit downstream (GSM8K vs the 86.0 bf16 reference)" $L/q2b_bench_w4.log \
  $PY benchmarks.py --model $D --quant none --limit 200 --batch-size 1 \
     --tag gemma-w4g128

echo "[$(date +%H:%M)] === BOX2 ROUND 2 DONE ===" | tee -a $L/box2_queue.log
