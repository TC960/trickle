#!/usr/bin/env bash
# Compression sweep for one model on one GPU. Every result lands in the same
# JSONL so comparisons are apples-to-apples.
#   MODEL=... GPU=0 bash sweep.sh
set -uo pipefail
cd /ephemeral/work/code
export HF_HOME=/ephemeral/work/hf
export TOKENIZERS_PARALLELISM=false
PY=/ephemeral/work/venv/bin/python
OUT=/ephemeral/work/out
MODEL="${MODEL:?}"; GPU="${GPU:-0}"
SHORT=$(basename "$MODEL")

step() {
  local name="$1"; shift
  echo ""; echo "########## $SHORT :: $name :: $(date +%H:%M:%S)"
  CUDA_VISIBLE_DEVICES="$GPU" $PY "$@" 2>&1 \
    | grep -vE "^(Loading weights|Fetching|Generating|Downloading|Resolving)" \
    | grep -vE "it/s\]$|examples/s\]$" | tail -14
  echo "########## exit=${PIPESTATUS[0]} $name"
}

# Baseline first -- every other number is a delta from this.
step "bf16-baseline" perplexity.py --model "$MODEL" --tag "$SHORT-bf16" --out "$OUT/ppl.jsonl"
step "int8"          perplexity.py --model "$MODEL" --quant int8 --tag "$SHORT-int8" --out "$OUT/ppl.jsonl"
step "nf4"           perplexity.py --model "$MODEL" --quant nf4  --tag "$SHORT-nf4"  --out "$OUT/ppl.jsonl"

# Embedding ablation -- the measured bottleneck. The tied/untied pair is the
# question: does keeping a full-precision output head pay for its memory?
step "embed-control"    embed_ablation.py --model "$MODEL" --method none --out "$OUT/embed.jsonl"
step "embed-int8"       embed_ablation.py --model "$MODEL" --method int8 --out "$OUT/embed.jsonl"
step "embed-int8-untie" embed_ablation.py --model "$MODEL" --method int8 --untie --out "$OUT/embed.jsonl"
step "embed-int4-g128"  embed_ablation.py --model "$MODEL" --method int4 --group-size 128 --out "$OUT/embed.jsonl"
step "embed-int4-untie" embed_ablation.py --model "$MODEL" --method int4 --group-size 128 --untie --out "$OUT/embed.jsonl"
step "embed-int4-g32"   embed_ablation.py --model "$MODEL" --method int4 --group-size 32 --out "$OUT/embed.jsonl"
step "embed-svd-r2048"  embed_ablation.py --model "$MODEL" --method svd --rank 2048 --out "$OUT/embed.jsonl"

echo ""; echo "===== SWEEP DONE $SHORT $(date +%H:%M:%S) ====="
