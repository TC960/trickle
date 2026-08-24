# The three measurements that decide whether this project's approach works.
#
#   1. Does the streaming engine reproduce the model exactly at 31B, and how
#      fast is it?  <- the deliverable, never once built
#   2. nf4 GSM8K, the missing control that makes the KD-LoRA result reportable
#   3. (phase 2, separate) the MoE alternative
#
# Phase 1 only. Qwen needs ~70 GB and this box has 28 GB free, so it runs after
# the Gemma artifacts are backed up and cleared.
set -uo pipefail
W=/ephemeral/work
export HF_HOME=$W/hf TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$W/code
cd $W/code
PY=$W/venv/bin/python
L=$W/logs
M=google/gemma-4-31B
S=$W/shards-gemma4-w4g128

step () { local name=$1 log=$2; shift 2
  echo "[$(date +%H:%M)] === $name ===" | tee -a $L/session.log
  "$@" > "$log" 2>&1
  local rc=$?
  echo "[$(date +%H:%M)]     exit $rc  ($log)" | tee -a $L/session.log
  tail -6 "$log" | tr '\r' '\n' | tail -6 | sed 's/^/      /' | tee -a $L/session.log
  return 0; }

# --- gate: the served quantizer must equal the trained one ------------------
step "unit tests: uniform quantization + QAT agreement" $L/s_tests.log \
  $PY -m pytest tests/test_uniform.py -q -p no:cacheprovider

# --- 1. build the streaming artifact ----------------------------------------
step "shard gemma-4-31B at 4 bits" $L/s_shard.log \
  $PY build_shards.py --model $M --out $S --bits 4 --group-size 128 \
     --num-layers 60 --skip-first 0 --skip-last 0

echo "  disk after sharding: $(df -h / | awk 'NR==2{print $4}') free" | tee -a $L/session.log

# --- 2. correctness + throughput --------------------------------------------
step "stream: bit-exactness across budgets, then tok/s" $L/s_stream.log \
  $PY stream_bench.py --shards $S --model $M --device cuda \
     --budgets-gb 0.5,1,2,4,64 --new-tokens 32 --tag gemma-w4g128-stream

# --- 3. the missing control -------------------------------------------------
step "CONTROL: nf4 downstream (no adapter)" $L/s_bench_nf4.log \
  $PY benchmarks.py --model $M --quant nf4 --limit 200 --batch-size 1 \
     --tag gemma-nf4-control

echo "[$(date +%H:%M)] === PHASE 1 DONE ===" | tee -a $L/session.log
echo "  compare: nf4+KD-LoRA scored GSM8K 83.5 / MMLU 82.97"
echo "           bf16 reference       GSM8K 86.0 / MMLU 82.99"
