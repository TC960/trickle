#!/bin/bash
# Agentic-browsing box: model serving + browser automation stack.
#
# Built before the benchmark choice is finalised, because the serving stack is
# identical whichever benchmark wins (BrowserGym/MiniWoB++/WebArena all drive an
# OpenAI-compatible endpoint). Installs the browser tooling too so the only
# remaining step is the benchmark itself.
#
# Every fix from the SWE-bench boxes is pre-applied:
#   * gcc/g++ version mismatch -> install g++-12 and pin all three compilers
#     (image ships gcc 12.3 but g++ 11.4, and cc1plus exists only for 11)
#   * apt failures were silent under `set -ux` -> this runs `set -eux` and
#     VERIFIES cmake exists rather than assuming the install worked
#   * llama-server is an ~18KB thin exe, so size is not a validity test -> use
#     `[ -x ]`, never `cmd | head || die` (in a pipeline `||` binds to head)
#   * Qwen3.6 routes reasoning to `reasoning_content`, leaving `content` EMPTY
#     under a small budget -> --reasoning-budget 0, and the smoke test ASSERTS
#     non-empty
#   * -c is split across --parallel slots; 32k total over 4 slots gave 8k each
#     and truncated every action -> keep 131072 total
set -eux
set -o pipefail
H=$HOME
W=$H/browse
mkdir -p "$W/logs" "$W/models" "$W/results"
exec > "$W/logs/bootstrap.log" 2>&1

step () { echo "=== $(date '+%F %H:%M:%S') STEP: $* ==="; }
die  () { echo "FATAL: $*"; exit 1; }

step "arm 8h hard deadline"
date -d "+8 hours" +%s > "$H/DEADLINE"
cat > "$H/deadline_guard.sh" <<'EOF'
#!/bin/bash
set -u
D=$HOME/DEADLINE; L=$HOME/deadline_guard.log
[ -f "$D" ] || exit 0
now=$(date +%s); dl=$(cat "$D" 2>/dev/null || echo 0)
echo "$(date '+%F %H:%M:%S') remaining=$(( dl - now ))s" >> "$L"
[ "$now" -ge "$dl" ] && { echo "$(date '+%F %H:%M:%S') DEADLINE -- POWERING OFF" >> "$L"; sync; sudo shutdown -h now; }
exit 0
EOF
sudo tee /etc/systemd/system/deadline-guard.service > /dev/null <<EOF
[Unit]
Description=Hard wall-clock shutdown
After=multi-user.target
[Service]
Type=oneshot
User=$USER
Environment=HOME=$H
ExecStart=/bin/bash $H/deadline_guard.sh
[Install]
WantedBy=multi-user.target
EOF
sudo tee /etc/systemd/system/deadline-guard.timer > /dev/null <<'EOF'
[Unit]
Description=Check hard deadline every 5 minutes
[Timer]
OnBootSec=30s
OnUnitActiveSec=5min
AccuracySec=30s
[Install]
WantedBy=timers.target
EOF
sudo systemctl daemon-reload
sudo systemctl enable deadline-guard.service
sudo systemctl enable --now deadline-guard.timer
bash "$H/deadline_guard.sh"; cat "$H/deadline_guard.log"

step "system packages (VERIFY, do not assume apt succeeded)"
sudo apt-get update -qq
sudo apt-get install -y -qq software-properties-common build-essential git curl jq \
  cmake ninja-build gcc-12 g++-12
command -v cmake || die "cmake missing"
command -v ninja || die "ninja missing"
ls /usr/lib/gcc/x86_64-linux-gnu/12/cc1plus || die "cc1plus for gcc-12 missing"

sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -qq
sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev
python3.11 --version

step "python env + browser stack"
python3.11 -m venv "$W/venv"
source "$W/venv/bin/activate"
pip install --upgrade pip -q
pip install -q huggingface_hub hf_transfer datasets litellm
# BrowserGym is the unified harness fronting MiniWoB++, WebArena and WorkArena;
# browser-use is the alternative driver. Install both so the benchmark decision
# does not require another provisioning round.
pip install -q playwright browsergym browsergym-miniwob browsergym-webarena || \
  pip install -q playwright browsergym || echo "WARN: some browsergym extras unavailable"
pip install -q browser-use || echo "WARN: browser-use unavailable"
python3 -m playwright install chromium
python3 -m playwright install-deps chromium || true
python3 -c "import playwright; print('playwright ok')"
python3 -c "import browsergym; print('browsergym ok')" || echo "WARN: browsergym import failed"

step "CUDA toolkit"
if ! command -v nvcc >/dev/null 2>&1 && [ ! -x /usr/local/cuda-12.8/bin/nvcc ]; then
  sudo apt-get install -y -qq cuda-toolkit-12-8 || sudo apt-get install -y -qq cuda-toolkit-12-6
fi
for c in /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda; do
  [ -x "$c/bin/nvcc" ] && { export CUDA_HOME=$c; break; }
done
[ -n "${CUDA_HOME:-}" ] || die "no CUDA toolkit"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
nvcc --version | tail -2

step "build llama-server"
git clone --depth 1 https://github.com/ggml-org/llama.cpp "$W/llama.cpp"
cd "$W/llama.cpp"
cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
      -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT="$CUDA_HOME" \
      -DCMAKE_C_COMPILER=gcc-12 -DCMAKE_CXX_COMPILER=g++-12 \
      -DCMAKE_CUDA_HOST_COMPILER=g++-12
cmake --build build --config Release -j"$(nproc)" --target llama-server
BIN="$W/llama.cpp/build/bin/llama-server"
[ -x "$BIN" ] || die "llama-server not built"
"$BIN" --version 2>&1 | head -2

step "download Qwen3.6-35B-A3B Q4_K_M"
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download("unsloth/Qwen3.6-35B-A3B-GGUF", allow_patterns=["*Q4_K_M*"],
                  local_dir=os.path.expanduser("~/browse/models/qwen-q4"))
PY
find "$W/models/qwen-q4" -name '*.gguf' | sort | head -1 > "$W/GGUF_PATH"
GGUF=$(cat "$W/GGUF_PATH"); echo "GGUF=$GGUF"
[ -s "$GGUF" ] || die "gguf missing"

step "env for benchmark runs"
cat > "$W/env.sh" <<EOF
export OPENAI_API_KEY=dummy
export OPENAI_API_BASE=http://127.0.0.1:8080/v1
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export LITELLM_MODEL_REGISTRY_PATH=$W/registry.json
EOF
cat > "$W/registry.json" <<'EOF'
{"local":{"max_tokens":32768,"max_input_tokens":32768,"max_output_tokens":8192,
 "input_cost_per_token":0.0,"output_cost_per_token":0.0,
 "litellm_provider":"openai","mode":"chat"},
 "openai/local":{"max_tokens":32768,"max_input_tokens":32768,"max_output_tokens":8192,
 "input_cost_per_token":0.0,"output_cost_per_token":0.0,
 "litellm_provider":"openai","mode":"chat"}}
EOF

step "serve model (131072 ctx across 4 slots = 32k each; smaller truncates actions)"
"$BIN" -m "$GGUF" -ngl 999 -c 131072 --parallel 4 \
  --host 127.0.0.1 --port 8080 --jinja --reasoning-budget 0 \
  > "$W/logs/server.log" 2>&1 &
echo $! > "$W/server.pid"
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && { echo "healthy"; break; }
  sleep 10
done
curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 || die "server never healthy"

step "smoke test (must be non-empty)"
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
 -d '{"model":"local","messages":[{"role":"user","content":"Reply with exactly: PONG"}],"max_tokens":400,"temperature":0}' \
 | python3 -c "
import sys,json
c=(json.load(sys.stdin)['choices'][0]['message'].get('content') or '').strip()
print('SMOKE:',repr(c)[:200]); assert c, 'FATAL empty content'
"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
touch "$W/READY"
step "BROWSE BOX READY -- serving on :8080, browser stack installed"
