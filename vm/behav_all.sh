# Re-judge every surviving config on flip rate, not perplexity.
cd /ephemeral/work/code
export HF_HOME=/ephemeral/work/hf TOKENIZERS_PARALLELISM=false
PY=/ephemeral/work/venv/bin/python
L=/ephemeral/work/logs
T=/ephemeral/work/out/teacher_gemma.pt
M=google/gemma-4-31B

while ! grep -q "INTEGRATION DONE" $L/integrate.log 2>/dev/null; do sleep 120; done

[ -f "$T" ] || $PY deep_eval.py --model $M --quant none --save-teacher $T \
  --tag gemma-bf16-teacher > $L/behav_teacher.log 2>&1

for D in /ephemeral/work/models/*/; do
  N=$(basename "$D")
  echo "[$(date +%H:%M)] === behavioural: $N ==="
  $PY deep_eval.py --model "$D" --teacher $T --tag "$N" > $L/behav_$N.log 2>&1
  grep -E "flip_rate|kl_mean|student_ppl" $L/behav_$N.log | tail -3
done
echo "[$(date +%H:%M)] === BEHAVIOURAL SWEEP DONE ==="
