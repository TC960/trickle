# QLoRA recovery run. Assumes a fresh box bootstrapped with bootstrap_h200.sh.
#
# Control/treatment pair, per the project rule that a treatment without a
# control does not get reported:
#   control    nf4, no adapters   -- already measured: 9.18% flips, KL 0.1213
#   treatment  nf4 + KD-LoRA      -- same base, same eval, adapters only
#
# Base is bitsandbytes nf4 rather than our better w4g128 (8.25% flips) for a
# blunt reason: this box has 29 GB of disk free and save_pretrained writes the
# model back in bf16, which is 62.5 GB. nf4 quantizes on load and needs no
# artifact. If KD-LoRA works here it is worth porting onto w4g128; if it does
# not, we saved a rebuild.
#
# The teacher logits SURVIVED the stop (/ephemeral persists on this provider),
# so no regeneration is needed.
set -uo pipefail
W=/ephemeral/work
[ -d /ephemeral ] || W=$HOME/work
export HF_HOME=$W/hf TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd $W/code
PY=$W/venv/bin/python
L=$W/logs
M=google/gemma-4-31B
D=$W/models/gemma4-w4g128
T=$W/out/teacher_gemma.pt

$PY -c "import peft" 2>/dev/null || $W/venv/bin/pip install -q peft

step () { local name=$1 log=$2; shift 2
  echo "[$(date +%H:%M)] === $name ===" | tee -a $L/recovery.log
  "$@" > "$log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M)]     exit $rc  ($log)" | tee -a $L/recovery.log
  tail -5 "$log" | tr '\r' '\n' | tail -5 | sed 's/^/      /' | tee -a $L/recovery.log
  return 0; }

[ -f "$T" ] || step "bf16 teacher logits (top-K)" $L/r_teacher.log \
  $PY deep_eval.py --model $M --quant none --save-teacher $T --tag gemma-bf16-teacher

step "TREATMENT: KD-LoRA recovery on nf4 (r=32)" $L/r_qlora.log \
  $PY qlora_recover.py --model $M --load-4bit --teacher $T \
     --rank 32 --alpha 64 --steps 500 --lr 1e-4 \
     --save-dir $W/models/gemma4-nf4-lora --tag gemma-nf4-kdlora-r32

echo "[$(date +%H:%M)] === RECOVERY DONE ===" | tee -a $L/recovery.log
echo "  control  nf4 no adapters : flips 9.18%  KL 0.1213"
echo "  ref      our w4g128       : flips 8.25%  KL 0.0896  GSM8K 80.0"
echo "  ref      bf16             :                         GSM8K 86.0"
