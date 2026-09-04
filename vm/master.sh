# ONE queue. No locks needed because nothing else runs on this box.
# Ordered by how much each result changes the project's direction.
cd /ephemeral/work/code
export HF_HOME=/ephemeral/work/hf TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY=/ephemeral/work/venv/bin/python
L=/ephemeral/work/logs
M=google/gemma-4-31B
step () { echo ""; echo "[$(date +%H:%M)] ##### $*"; }

# 1. THE decisive one. Weight-level reconstruction error says the cliff sits
#    between 2 and 3 bits (ternary 0.5159 | 2-bit 0.5118 | 3-bit 0.2212).
#    If 3-bit lands usable, the answer was never "rescue ternary".
for B in 3 2 4; do
  step "${B}-bit, 60 blocks, sequential drift-aware"
  $PY distill_seq.py --model $M --bits $B --steps 200 --n-calib 32 \
    --tag "gemma-w${B}g128-seq" > $L/seq_w$B.log 2>&1
  grep -E "PERPLEXITY" $L/seq_w$B.log | tail -1
done

# 2. Which layers are actually tolerant. Neither layer_scalar nor the
#    full_attention indices predicted the knee, so it has to be measured.
step "per-layer sensitivity profile"
$PY sensitivity.py --model $M --mode profile > $L/sens_profile.log 2>&1
tail -20 $L/sens_profile.log

for N in 30 40 45; do
  step "mixed precision: $N most-tolerant layers ternary"
  $PY sensitivity.py --model $M --mode allocate --n-ternary $N > $L/mixed_$N.log 2>&1
  grep -E "MIXED|ternarizing" $L/mixed_$N.log | tail -2
done

# 3. Learnable threshold -- tests PV-Tuning's prediction that scale-only plateaus.
step "e2e ternary with learnable threshold"
$PY distill_e2e.py --model $M --steps 300 --lr 1e-3 --seq-len 512 \
  --tag gemma-ternary-e2e-mu > $L/e2e_mu.log 2>&1
grep -E "PERPLEXITY|zeros=" $L/e2e_mu.log | tail -3

# 4. Downstream numbers last -- they characterize, they don't decide.
step "downstream benchmarks"
for Q in none int8 nf4; do
  $PY benchmarks.py --model $M --quant $Q --limit 200 --batch-size 2 \
    > $L/bench_$Q.log 2>&1
  grep -A6 '{' $L/bench_$Q.log | tail -6
done

$PY make_report.py > /ephemeral/work/out/REPORT.md 2>&1
step "MASTER QUEUE DONE"
