#!/usr/bin/env bash
# Experiment driver. Runs a compression sweep and appends every result to one
# JSONL so the comparison table is apples-to-apples.
#
# Gemma-4-31B at bf16 is ~62 GB, which fits on ONE A100 80GB. So we pin each
# experiment to a single GPU and run two streams concurrently rather than
# sharding one model across both -- twice the experiments in the same wall time.
set -uo pipefail

WORK="${WORK:-$HOME/work}"
export HF_HOME="$WORK/hf"
export HF_HUB_ENABLE_HF_TRANSFER=1
export TOKENIZERS_PARALLELISM=false
OUT="$WORK/out"
mkdir -p "$OUT"

MODEL="${MODEL:?set MODEL}"
GPU="${GPU:-0}"
LIMIT="${LIMIT:-}"        # e.g. LIMIT="--limit-chars 400000" for a fast pass
SHORT=$(basename "$MODEL")

run() {
    local name="$1"; shift
    echo ""
    echo "##### [$SHORT/gpu$GPU] $name  $(date +%H:%M:%S)"
    CUDA_VISIBLE_DEVICES="$GPU" timeout "${STEP_TIMEOUT:-3600}" python "$@" \
        2>&1 | tail -25
    echo "##### exit=$? $name"
}

echo "model=$MODEL gpu=$GPU out=$OUT"

# 1. Baseline. Everything else is measured as a delta from this.
run "baseline-bf16" perplexity.py --model "$MODEL" --tag "$SHORT-bf16" \
    --out "$OUT/ppl.jsonl" $LIMIT

# 2. Standard quantization reference points.
run "int8" perplexity.py --model "$MODEL" --quant int8 --tag "$SHORT-int8" \
    --out "$OUT/ppl.jsonl" $LIMIT
run "nf4"  perplexity.py --model "$MODEL" --quant nf4  --tag "$SHORT-nf4" \
    --out "$OUT/ppl.jsonl" $LIMIT

# 3. Embedding ablation -- our measured bottleneck. The tied-vs-untied pair is
#    the interesting comparison: does a full-precision output head pay for itself?
run "embed-none"        embed_ablation.py --model "$MODEL" --method none  --out "$OUT/embed.jsonl" $LIMIT
run "embed-int8"        embed_ablation.py --model "$MODEL" --method int8  --out "$OUT/embed.jsonl" $LIMIT
run "embed-int8-untied" embed_ablation.py --model "$MODEL" --method int8 --untie --out "$OUT/embed.jsonl" $LIMIT
run "embed-int4-g128"   embed_ablation.py --model "$MODEL" --method int4 --group-size 128 --out "$OUT/embed.jsonl" $LIMIT
run "embed-int4-untied" embed_ablation.py --model "$MODEL" --method int4 --group-size 128 --untie --out "$OUT/embed.jsonl" $LIMIT

echo ""
echo "===== done $SHORT $(date +%H:%M:%S) ====="
