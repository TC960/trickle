# How many blocks can go ternary before the model breaks?
# Each run is an independent treatment against the same bf16 control, so the
# result is a degradation curve rather than a single pass/fail number.
cd /ephemeral/work/code
export HF_HOME=/ephemeral/work/hf TOKENIZERS_PARALLELISM=false
PY=/ephemeral/work/venv/bin/python
LOG=/ephemeral/work/logs

for N in 60 45 30 15 8; do
  echo "[$(date +%H:%M)] ===== ternary depth: $N of 60 blocks ====="
  $PY distill_seq.py --model google/gemma-4-31B \
    --steps 200 --n-calib 32 --max-blocks $N \
    --tag "gemma-ternary-seq-d$N" \
    --out $LOG/../out/seq_depth.jsonl > $LOG/seq_d$N.log 2>&1
  grep -E "PERPLEXITY" $LOG/seq_d$N.log | tail -1
done

echo "[$(date +%H:%M)] ===== qwen, full depth ====="
$PY distill_seq.py --model Qwen/Qwen3.8-27B --steps 200 --n-calib 32 \
  --tag "qwen-ternary-seq-full" --out $LOG/../out/seq_depth.jsonl \
  > $LOG/seq_qwen.log 2>&1
grep -E "PERPLEXITY" $LOG/seq_qwen.log | tail -1

echo "[$(date +%H:%M)] ===== gemma embedding svd (OOM retry) ====="
$PY embed_ablation.py --model google/gemma-4-31B --method svd --rank 2048 \
  --out $LOG/../out/embed.jsonl > $LOG/gemma_svd.log 2>&1
grep -E "PERPLEXITY" $LOG/gemma_svd.log | tail -1

echo "[$(date +%H:%M)] ===== DEPTH SWEEP DONE ====="
