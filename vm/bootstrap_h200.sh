# Replacement box (H200, 143 GB VRAM). Setup, then the workstream that died
# with gorgeous-copper-mite.
#
# DISK CONSTRAINT, discovered on arrival: this instance has a single 120 GB
# root volume and no separate scratch disk (the old box had 738 GB at
# /ephemeral). Gemma 4 is 62.5 GB and `save_pretrained` writes another 62.5 GB
# in bf16, so any --save-dir step would exhaust the disk. Everything here is
# therefore chosen to need no on-disk artifact; the save-and-reload chains stay
# on box2, which has 572 GB free.
#
# 143 GB of VRAM on ONE card is the real win: Gemma 4 fits with room for
# activations, so no device_map split and no disk offload.
set -uo pipefail
W=/ephemeral/work
[ -d /ephemeral ] || W=$HOME/work
export HF_HOME=$W/hf HF_XET_HIGH_PERFORMANCE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $W/{hf,out,logs,code,models}
cd $W/code
PY=$W/venv/bin/python
L=$W/logs
M=google/gemma-4-31B

if [ ! -x "$PY" ]; then
  echo "[$(date +%H:%M)] installing uv + venv"
  curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
  export PATH="$HOME/.local/bin:$PATH"
  uv venv $W/venv --python 3.12 >/dev/null 2>&1
  export VIRTUAL_ENV=$W/venv
  uv pip install -q torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -2
  uv pip install -q transformers accelerate safetensors huggingface_hub datasets \
    sentencepiece protobuf bitsandbytes einops lm-eval 2>&1 | tail -2
fi
$PY -c "import torch;print('torch',torch.__version__,torch.cuda.get_device_name(0),
      round(torch.cuda.get_device_properties(0).total_memory/2**30),'GiB')"

echo "[$(date +%H:%M)] downloading gemma-4-31B  (free: $(df -h $W | awk 'NR==2{print $4}'))"
$PY -c "
from huggingface_hub import snapshot_download
print(snapshot_download('google/gemma-4-31B', allow_patterns=['*.json','*.safetensors','*.txt','*.model'], max_workers=16))
" 2>&1 | tail -1
echo "[$(date +%H:%M)] free after download: $(df -h $W | awk 'NR==2{print $4}')"

step () { local name=$1 log=$2; shift 2
  local free_gb; free_gb=$(df --output=avail -BG "$W" | tail -1 | tr -dc '0-9')
  if [ "${free_gb:-0}" -lt 10 ]; then
    echo "[$(date +%H:%M)] !! SKIP $name -- only ${free_gb}GB free" | tee -a $L/h200_queue.log
    return 0
  fi
  echo "[$(date +%H:%M)] === $name  (${free_gb}GB free) ===" | tee -a $L/h200_queue.log
  "$@" > "$log" 2>&1
  # Capture immediately: a $(...) inside the echo runs first and clobbers $?,
  # which made every failing step report "exit 0" on 2026-08-22.
  local rc=$?
  echo "[$(date +%H:%M)]     exit $rc  ($log)" | tee -a $L/h200_queue.log
  tail -4 "$log" | sed 's/^/      /' | tee -a $L/h200_queue.log
  return 0; }

# ---------------------------------------------------------------------------
# 1. PIPELINE SANITY CHECK -- first, because it gates the credibility of every
# number above it. CLAUDE.md: quantizing at 8 bits should reproduce bf16 to
# within ~0.1%. If it does not, the whole bit-width curve is suspect. Queued
# three times on the old box, never once completed.
# ---------------------------------------------------------------------------
step "8-bit pipeline sanity check (MUST land within ~0.1% of 5.1876)" \
  $L/h_sanity_w8.log \
  $PY distill_seq.py --model $M --bits 8 --steps 0 --n-calib 8 \
     --tag gemma-w8g128-sanity

# ---------------------------------------------------------------------------
# 2. Per-layer sensitivity, re-profiled. The old profile was lost with box1 and
# was perplexity-ranked, which CLAUDE.md says is the wrong basis for exactly
# this kind of decision. Now ranked by flip rate and KL against the bf16
# teacher: better aligned with the doctrine AND ~40x cheaper, because one
# cached teacher pass replaces a scored perplexity pass per layer.
# ---------------------------------------------------------------------------
step "per-layer sensitivity profile (flip rate + KL)" \
  $L/h_sens_profile.log \
  $PY sensitivity.py --model $M --mode profile --metric kl --profile-chars 400000

# ---------------------------------------------------------------------------
# 3. The direct test of mixed precision: does sensitivity-ranked selection beat
# the crude "first N layers" heuristic? The two overlapped 27/30 last time, so
# the honest prior is that it gains little. Worth measuring rather than
# assuming in either direction.
# ---------------------------------------------------------------------------
for N in 30 40 45; do
  step "mixed precision: $N most-tolerant layers ternary" $L/h_alloc_$N.log \
    $PY sensitivity.py --model $M --mode allocate --n-ternary $N
done

# ---------------------------------------------------------------------------
# 4. GPTQ on real hooked activations. The implementation reproduces RTN exactly
# at H=I, so it is correct; it has still never beaten RTN on real data, and the
# one test that said it was 33% worse used a rank-deficient Hessian.
# ---------------------------------------------------------------------------
step "GPTQ with real activation Hessians" $L/h_gptq_real.log \
  $PY gptq_real.py --model $M --bits 2,3,4 --group 128,64

echo "[$(date +%H:%M)] === H200 QUEUE DONE ===" | tee -a $L/h200_queue.log
