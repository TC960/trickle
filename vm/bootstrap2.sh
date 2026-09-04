# Box 2 (single A100): full setup then the Qwen workstream, unattended.
set -uo pipefail
W=/ephemeral/work
export HF_HOME=$W/hf HF_XET_HIGH_PERFORMANCE=1 TOKENIZERS_PARALLELISM=false
mkdir -p $W/{hf,out,logs,code}

echo "[$(date +%H:%M)] installing uv + venv"
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH="$HOME/.local/bin:$PATH"
uv venv $W/venv --python 3.12 >/dev/null 2>&1
export VIRTUAL_ENV=$W/venv
uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2
uv pip install -q transformers accelerate safetensors huggingface_hub datasets \
  sentencepiece protobuf bitsandbytes einops lm-eval 2>&1 | tail -2
$W/venv/bin/python -c "import torch;print('torch',torch.__version__,'gpus',torch.cuda.device_count())"

echo "[$(date +%H:%M)] downloading Qwen3.8-27B"
$W/venv/bin/python -c "
from huggingface_hub import snapshot_download
print(snapshot_download('Qwen/Qwen3.8-27B', allow_patterns=['*.json','*.safetensors','*.txt','*.model'], max_workers=16))
" 2>&1 | tail -1

cd $W/code
PY=$W/venv/bin/python
L=$W/logs
M=Qwen/Qwen3.8-27B
T=$W/out/teacher_qwen.pt

echo "[$(date +%H:%M)] === qwen behavioural teacher ==="
$PY deep_eval.py --model $M --quant none --save-teacher $T \
  --tag qwen-bf16-teacher > $L/deep_teacher_qwen.log 2>&1
tail -2 $L/deep_teacher_qwen.log

echo "[$(date +%H:%M)] === qwen treatments (flip rate + KL + frequency strata) ==="
for SPEC in "int8:none" "nf4:none" "none:int8" "none:int4" "none:svd"; do
  Q="${SPEC%%:*}"; E="${SPEC##*:}"
  A="--quant $Q"; [ "$E" != "none" ] && A="$A --embed-method $E"
  echo "[$(date +%H:%M)] -- quant=$Q embed=$E"
  $PY deep_eval.py --model $M $A --teacher $T > $L/deep_qwen_${Q}_${E}.log 2>&1
  grep -E 'flip_rate|kl_mean|student_ppl|row_coverage' $L/deep_qwen_${Q}_${E}.log | tail -5
done

echo "[$(date +%H:%M)] === qwen vocab trim ==="
for TGT in 32000 64000; do
  $PY vocab_trim.py --model $M --target $TGT >> $L/trim_qwen.log 2>&1
done
grep -E 'table:|round-trip' $L/trim_qwen.log | tail -6

echo "[$(date +%H:%M)] === qwen downstream benchmarks ==="
for Q in none int8 nf4; do
  $PY benchmarks.py --model $M --quant $Q --limit 250 --batch-size 4 \
    > $L/bench_qwen_$Q.log 2>&1
  grep -A8 '{' $L/bench_qwen_$Q.log | tail -8
done

echo "[$(date +%H:%M)] === BOX2 QUEUE DONE ==="
