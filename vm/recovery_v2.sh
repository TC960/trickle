# KD-LoRA recovery, run 2 -- the honest version.
#
# Run 1 reported 9.18% -> 3.95% flips, but trained and evaluated on the same
# wikitext TEST split, because load_wikitext defaults to split="test" and I
# passed the same token stream to both. That number is measured on training
# data. It is an upper bound, not a generalization estimate.
#
# This run:
#   teacher_train  wikitext TRAIN  -> the KD signal
#   teacher_gemma  wikitext TEST   -> what flip rate is reported on (exists)
#   GSM8K          restored; run 1 dropped it and never measured whether the
#                  6-point capability loss actually recovered
#
# Control (nf4, no adapters) is already measured: 9.18% flips, KL 0.1213 on test.
set -uo pipefail
W=/ephemeral/work
export HF_HOME=$W/hf TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $W/code
PY=$W/venv/bin/python
L=$W/logs
M=google/gemma-4-31B
T_TEST=$W/out/teacher_gemma.pt
T_TRAIN=$W/out/teacher_gemma_train.pt

step () { local name=$1 log=$2; shift 2
  echo "[$(date +%H:%M)] === $name ===" | tee -a $L/recovery2.log
  "$@" > "$log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M)]     exit $rc  ($log)" | tee -a $L/recovery2.log
  tail -5 "$log" | tr '\r' '\n' | tail -5 | sed 's/^/      /' | tee -a $L/recovery2.log
  return 0; }

[ -f "$T_TRAIN" ] || step "bf16 teacher on the TRAIN split" $L/r2_teacher_train.log \
  $PY deep_eval.py --model $M --quant none --split train \
     --save-teacher $T_TRAIN --tag gemma-bf16-teacher-train

step "KD-LoRA r=32, train on train / eval on test" $L/r2_qlora_r32.log \
  $PY qlora_recover.py --model $M --load-4bit \
     --teacher $T_TEST --train-teacher $T_TRAIN \
     --rank 32 --alpha 64 --steps 500 --lr 1e-4 \
     --save-dir $W/models/gemma4-nf4-lora-r32 --tag gemma-nf4-kdlora-r32-clean

step "downstream after recovery (does GSM8K come back?)" $L/r2_bench_r32.log \
  $PY benchmarks.py --model $W/models/gemma4-nf4-lora-r32 --quant none \
     --limit 200 --batch-size 1 --tag gemma-nf4-kdlora-r32-clean

echo "[$(date +%H:%M)] === RECOVERY v2 DONE ===" | tee -a $L/recovery2.log
echo "  control  nf4          : flips 9.18%  KL 0.1213"
echo "  run 1    nf4+LoRA     : flips 3.95%  KL 0.0693   (CONTAMINATED)"
echo "  bf16 ref                                          GSM8K 86.0"
echo "  nf4 has no GSM8K measured; our w4g128 scored 80.0"
