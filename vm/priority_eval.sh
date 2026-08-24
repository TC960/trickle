# The headline configs (3-bit, 4-bit) currently have PERPLEXITY ONLY -- the
# metric this project established is inadequate. This re-quantizes them WITH
# saving, then evaluates on what actually matters:
#   flip rate  - how often the model picks a different token than bf16
#   GSM8K      - generative arithmetic; one wrong token kills the answer, so
#                this is where quantization damage surfaces first
#   MMLU / HellaSwag / ARC-C
cd /ephemeral/work/code
export HF_HOME=/ephemeral/work/hf TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/ephemeral/work/venv/bin/python
L=/ephemeral/work/logs
M=google/gemma-4-31B
T=/ephemeral/work/out/teacher_gemma.pt

wait_idle () { local n=0
  while [ "$n" -lt 3 ]; do
    if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ]
      then n=$((n+1)); else n=0; fi
    sleep 30
  done; }

for B in 3 4; do
  D=/ephemeral/work/models/gemma4-w${B}g128
  wait_idle
  echo "[$(date +%H:%M)] === ${B}-bit: quantize + SAVE ==="
  $PY distill_seq.py --model $M --bits $B --steps 200 --n-calib 32 \
    --save-dir $D --tag "gemma-w${B}g128-saved" > $L/save_w$B.log 2>&1
  grep -E "PERPLEXITY|saved" $L/save_w$B.log | tail -2

  wait_idle
  echo "[$(date +%H:%M)] === ${B}-bit: FLIP RATE + KL ==="
  [ -f "$T" ] || $PY deep_eval.py --model $M --quant none --save-teacher $T \
      --tag gemma-bf16-teacher > $L/pe_teacher.log 2>&1
  $PY deep_eval.py --model "$D" --teacher $T --tag "gemma-w${B}g128" \
    > $L/pe_behav_w$B.log 2>&1
  grep -E "flip_rate|kl_mean|student_ppl|^ +[0-9]" $L/pe_behav_w$B.log | tail -8

  wait_idle
  echo "[$(date +%H:%M)] === ${B}-bit: GSM8K + MMLU + HellaSwag + ARC ==="
  $PY benchmarks.py --model "$D" --quant none --limit 200 --batch-size 2 \
    --tag "gemma-w${B}g128" > $L/pe_bench_w$B.log 2>&1
  grep -A8 '{' $L/pe_bench_w$B.log | tail -8
done

wait_idle
echo "[$(date +%H:%M)] === bf16 reference benchmarks ==="
$PY benchmarks.py --model $M --quant none --limit 200 --batch-size 2 \
  --tag gemma-bf16 > $L/pe_bench_bf16.log 2>&1
grep -A8 '{' $L/pe_bench_bf16.log | tail -8

$PY make_report.py > /ephemeral/work/out/REPORT.md 2>&1
echo "[$(date +%H:%M)] === PRIORITY EVAL DONE ==="
