#!/usr/bin/env bash
# Provision the box and pull both target models. Long-running; run detached.
set -uo pipefail
WORK=/ephemeral/work
export HF_HOME=$WORK/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$WORK"/{hf,out,shards,logs}

echo "=== [$(date +%H:%M:%S)] installing uv ==="
curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
export PATH="$HOME/.local/bin:$PATH"

echo "=== [$(date +%H:%M:%S)] creating venv ==="
uv venv "$WORK/venv" --python 3.12
export VIRTUAL_ENV="$WORK/venv"
PIP="uv pip install"

echo "=== [$(date +%H:%M:%S)] torch (cu128, driver is 12.8) ==="
$PIP torch --index-url https://download.pytorch.org/whl/cu128 2>&1 | tail -3

echo "=== [$(date +%H:%M:%S)] libs ==="
$PIP transformers accelerate safetensors huggingface_hub hf_transfer \
     datasets sentencepiece protobuf bitsandbytes einops 2>&1 | tail -3

"$WORK/venv/bin/python" - <<'PY'
import torch, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda, "devices", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name} {p.total_memory/1e9:.0f}GB sm_{p.major}{p.minor}")
print("transformers", transformers.__version__)
PY

echo "=== [$(date +%H:%M:%S)] downloading models in parallel ==="
for M in google/gemma-4-31B Qwen/Qwen3.8-27B; do
  TAG=$(echo "$M" | tr '/' '_')
  nohup "$WORK/venv/bin/python" -c "
from huggingface_hub import snapshot_download
p = snapshot_download('$M', allow_patterns=['*.json','*.safetensors','*.txt','*.model'], max_workers=16)
print('DONE $M ->', p)
" > "$WORK/logs/dl_$TAG.log" 2>&1 &
  echo "  launched $M"
done
wait
echo "=== [$(date +%H:%M:%S)] setup complete ==="
du -sh "$WORK/hf"
