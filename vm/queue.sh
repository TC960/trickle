# Unattended experiment queue. Waits for the GPUs, then runs in priority order.
cd /ephemeral/work/code
export HF_HOME=/ephemeral/work/hf TOKENIZERS_PARALLELISM=false
PY=/ephemeral/work/venv/bin/python
LOG=/ephemeral/work/logs

wait_for_gpus () {
  while true; do
    BUSY=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
    [ "$BUSY" -eq 0 ] && break
    sleep 60
  done
}

echo "[$(date +%H:%M)] waiting for GPUs to free..."
wait_for_gpus
echo "[$(date +%H:%M)] GPUs free"

# 1. THE fix for error compounding -- spans both GPUs.
echo "[$(date +%H:%M)] === sequential ternary: gemma ==="
$PY distill_seq.py --model google/gemma-4-31B --steps 300 --lr 2e-5 \
  > $LOG/seq_gemma.log 2>&1
grep -E "PERPLEXITY|block  ?[0-9]+:" $LOG/seq_gemma.log | tail -5

# 2. Same, but leave the first and last 2 blocks in bf16 -- those are the
#    most quantization-sensitive and cost little to keep.
echo "[$(date +%H:%M)] === sequential ternary + bf16 edges: gemma ==="
$PY distill_seq.py --model google/gemma-4-31B --steps 300 --lr 2e-5 \
  --skip-first 2 --skip-last 2 > $LOG/seq_gemma_skip.log 2>&1
grep -E "PERPLEXITY" $LOG/seq_gemma_skip.log | tail -2

# 3. Does it transfer to a different architecture?
echo "[$(date +%H:%M)] === sequential ternary: qwen ==="
$PY distill_seq.py --model Qwen/Qwen3.8-27B --steps 300 --lr 2e-5 \
  > $LOG/seq_qwen.log 2>&1
grep -E "PERPLEXITY" $LOG/seq_qwen.log | tail -2

# 4. Gemma SVD -- the datapoint the earlier OOM cost us. CPU-side SVD.
echo "[$(date +%H:%M)] === gemma embedding svd (retry) ==="
$PY embed_ablation.py --model google/gemma-4-31B --method svd --rank 2048 \
  --out /ephemeral/work/out/embed.jsonl > $LOG/gemma_svd.log 2>&1
grep -E "PERPLEXITY" $LOG/gemma_svd.log | tail -1

echo "[$(date +%H:%M)] === QUEUE DONE ==="
