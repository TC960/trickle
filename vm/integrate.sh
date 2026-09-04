# THE DELIVERABLE: 3-bit Gemma 4, sharded, streamed, verified.
#
# The project's original goal was ternary + AirLLM-style layer streaming.
# Ternary measured +1780% on this model; 3-bit measures +17.27% through the same
# pipeline. So the artifact is 3-bit, and this is the end-to-end integration:
# quantize -> shard per layer -> stream under a byte budget -> prove the streamed
# output matches the unstreamed one exactly.
cd /ephemeral/work/code
export HF_HOME=/ephemeral/work/hf TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/ephemeral/work/venv/bin/python
L=/ephemeral/work/logs
M=google/gemma-4-31B
Q=/ephemeral/work/models/gemma4-w3g128
S=/ephemeral/work/shards/gemma4-w3g128

wait_idle () { local n=0
  while [ "$n" -lt 4 ]; do
    if [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -eq 0 ]
      then n=$((n+1)); else n=0; fi
    sleep 30
  done; }

wait_idle
echo "[$(date +%H:%M)] === 1. quantize to 3-bit and SAVE ==="
$PY distill_seq.py --model $M --bits 3 --steps 200 --n-calib 32 \
  --save-dir $Q --tag gemma-w3g128-artifact > $L/artifact_quant.log 2>&1
grep -E "PERPLEXITY|saved" $L/artifact_quant.log | tail -2

# Perplexity is a geometric mean and cancels per-token damage; NeurIPS 2024
# measured it flat at 5.70 while token agreement fell 61%->21%. So the artifact
# is judged on FLIP RATE (how often it picks a different token than bf16) and
# KL, with perplexity kept only for comparability.
echo "[$(date +%H:%M)] === 1b. BEHAVIOURAL eval of the 3-bit artifact ==="
T=/ephemeral/work/out/teacher_gemma.pt
if [ ! -f "$T" ]; then
  $PY deep_eval.py --model $M --quant none --save-teacher $T \
    --tag gemma-bf16-teacher > $L/artifact_teacher.log 2>&1
fi
$PY deep_eval.py --model $Q --teacher $T --tag gemma-w3g128-3bit \
  > $L/artifact_behav.log 2>&1
grep -E "flip_rate|kl_mean|student_ppl|freq|^ +[0-9]" $L/artifact_behav.log | tail -10

echo "[$(date +%H:%M)] === 2. shard per layer for streaming ==="
$PY - <<'PY2' > $L/artifact_shard.log 2>&1
from airllm_ternary.policy import PrecisionPolicy
from airllm_ternary.shard import build_shards
pol = PrecisionPolicy(num_layers=60, group_size=128, pack_mode="2bit",
                      skip_first=0, skip_last=0)
m = build_shards("/ephemeral/work/models/gemma4-w3g128",
                 "/ephemeral/work/shards/gemma4-w3g128", pol,
                 measure_error=False, verbose=False)
s = m["summary"]
print(f"shards: {s['num_layer_shards'] if 'num_layer_shards' in s else '?'} layers, "
      f"largest {s['largest_layer_bytes']/1e6:.0f} MB, total {s['total_bytes']/1e9:.2f} GB")
PY2
tail -3 $L/artifact_shard.log

echo "[$(date +%H:%M)] === 3. stream it and verify bit-exactness ==="
$PY experiments/verify_streaming.py --model $Q --shards $S --budget-gb 4 \
  > $L/artifact_stream.log 2>&1 || \
$PY verify_streaming.py --model $Q --shards $S --budget-gb 4 > $L/artifact_stream.log 2>&1
grep -E "PASS|FAIL|tok/s|peak RSS|bytes read|budget" $L/artifact_stream.log | tail -8

echo "[$(date +%H:%M)] === INTEGRATION DONE ==="
