# Behavioural evaluation queue. Runs after the depth sweep releases the GPUs.
cd /ephemeral/work/code
export HF_HOME=/ephemeral/work/hf TOKENIZERS_PARALLELISM=false
PY=/ephemeral/work/venv/bin/python
L=/ephemeral/work/logs
T=/ephemeral/work/out/teacher_gemma.pt

while [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -gt 0 ]; do sleep 90; done
echo "[$(date +%H:%M)] GPUs free, starting behavioural eval"

# Teacher reference: per-token NLL, argmax and top-64 logprobs from the bf16 model.
echo "[$(date +%H:%M)] === teacher reference ==="
$PY deep_eval.py --model google/gemma-4-31B --quant none \
  --save-teacher $T --tag gemma-bf16-teacher > $L/deep_teacher.log 2>&1
tail -3 $L/deep_teacher.log

# Now the treatments, each compared against that teacher. The question for each:
# does perplexity agree with flip rate, or does it hide damage?
for SPEC in "int8:none" "nf4:none" "none:int8" "none:int4" "none:svd"; do
  Q="${SPEC%%:*}"; E="${SPEC##*:}"
  ARGS="--quant $Q"; [ "$E" != "none" ] && ARGS="$ARGS --embed-method $E"
  echo "[$(date +%H:%M)] === treatment quant=$Q embed=$E ==="
  $PY deep_eval.py --model google/gemma-4-31B $ARGS --teacher $T \
    > $L/deep_${Q}_${E}.log 2>&1
  grep -E 'flip_rate|kl_mean|student_ppl|row_coverage|^ +[0-9]' $L/deep_${Q}_${E}.log | tail -12
done

# Downstream tasks, smaller batch so it does not thrash an almost-full card.
echo "[$(date +%H:%M)] === downstream benchmarks ==="
for Q in none int8 nf4; do
  $PY benchmarks.py --model google/gemma-4-31B --quant $Q --limit 250 \
    --batch-size 4 > $L/bench_$Q.log 2>&1
  grep -A8 '{' $L/bench_$Q.log | tail -8
done

echo "[$(date +%H:%M)] === QUEUE2 DONE ==="
