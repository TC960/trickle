#!/usr/bin/env bash
# Provision the A100 box. Idempotent; safe to re-run.
set -euo pipefail

echo "=== hardware ==="
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
echo "CPUs: $(nproc)  RAM: $(free -g | awk '/^Mem:/{print $2}')GB"
df -h /  | tail -1

# Keep every large artifact on the big volume, not the root disk.
export WORK="${WORK:-$HOME/work}"
mkdir -p "$WORK/hf" "$WORK/out" "$WORK/shards"
export HF_HOME="$WORK/hf"
echo "WORK=$WORK  HF_HOME=$HF_HOME"

echo "=== python env ==="
python3 -V
pip install -q --upgrade pip
# torch is normally preinstalled on Brev GPU images; only install if absent.
python3 -c "import torch" 2>/dev/null || pip install -q torch --index-url https://download.pytorch.org/whl/cu124
pip install -q transformers accelerate safetensors huggingface_hub datasets sentencepiece protobuf hf_transfer

python3 - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("devices:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name}  {p.total_memory/1e9:.0f} GB  sm_{p.major}{p.minor}")
PY
echo "=== bootstrap done ==="
