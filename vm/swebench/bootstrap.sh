#!/bin/bash
# SWE-bench Verified on Qwen3.6-35B-A3B, bare box to scored result, one script.
#
# Every blocker hit on the previous attempt is handled here:
#   * PyPI `sweagent` is a dead 0.0.1 stub (typo'd dep `togetherunidiff`), and
#     real SWE-agent asserts on three repo-relative dirs a pip install never
#     creates. Its own README says use mini-swe-agent -- which we do. It is also
#     the scaffold behind SWE-bench's official "Bash Only" leaderboard, so the
#     number is comparable to published results.
#   * mini-swe-agent v2 DEFAULTS TO NATIVE TOOL CALLING. `swebench_backticks.yaml`
#     sets backtick prompts but does NOT set model_class, so without an explicit
#     `--model-class litellm_textbased` you get backtick prompts driving a
#     tool-calling model -- a silent mismatch scoring ~0 for scaffold reasons.
#   * litellm demands a non-empty API key and chokes on unknown-model pricing.
#   * The box has an NVIDIA driver but no CUDA toolkit; llama.cpp cannot build a
#     GPU backend without nvcc.
#   * Previous run split setup and a watcher across two processes and detected
#     liveness with `ps | grep -qF <pattern>`, which matched grep's own argv
#     (bug 17). This is ONE linear process: no cross-process liveness to botch.
set -ux
H=$HOME
W=$H/swebench
mkdir -p "$W/logs" "$W/models" "$W/results" "$W/preflight"
exec > "$W/logs/bootstrap.log" 2>&1

step () { echo "=== $(date '+%F %H:%M:%S') STEP: $* ==="; }
die  () { echo "FATAL: $*"; exit 1; }
need () {
  local f=$1 min=${2:-1000} sz
  sz=$(stat -c %s "$f" 2>/dev/null || echo 0)
  [ "$sz" -ge "$min" ] || die "$f missing/small (${sz}B)"
  echo "OK: $f ($(numfmt --to=iec "$sz"))"
}
SRV=""
trap '[ -n "$SRV" ] && kill "$SRV" 2>/dev/null; true' EXIT

# ---------- 0. hard deadline, before anything else ----------
step "arm hard deadline (7h)"
date -d "+7 hours" +%s > "$H/DEADLINE"
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
echo "DEADLINE_ARMED"

# ---------- 1. system + python 3.11 ----------
step "system packages and python 3.11"
sudo apt-get update -qq
sudo apt-get install -y -qq software-properties-common build-essential git curl jq cmake ninja-build
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt-get update -qq
sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev
python3.11 --version

step "venv + agent stack (verify imports BEFORE paying for build/download)"
python3.11 -m venv "$W/venv"
source "$W/venv/bin/activate"
pip install --upgrade pip -q
pip install -q huggingface_hub hf_transfer datasets swebench mini-swe-agent
python3 -c "import minisweagent, swebench; print('minisweagent + swebench import OK')"
which mini-extra || die "mini-extra entrypoint missing"
echo "STACK_OK"

step "litellm registry + env"
cat > "$W/registry.json" <<'EOF'
{
 "local": {"max_tokens":32768,"max_input_tokens":32768,"max_output_tokens":8192,
           "input_cost_per_token":0.0,"output_cost_per_token":0.0,
           "litellm_provider":"openai","mode":"chat"},
 "openai/local": {"max_tokens":32768,"max_input_tokens":32768,"max_output_tokens":8192,
           "input_cost_per_token":0.0,"output_cost_per_token":0.0,
           "litellm_provider":"openai","mode":"chat"}
}
EOF
cat > "$W/env.sh" <<EOF
export OPENAI_API_KEY=dummy
export MSWEA_COST_TRACKING=ignore_errors
export LITELLM_MODEL_REGISTRY_PATH=$W/registry.json
export MSWEA_CONFIGURED=true
EOF
source "$W/env.sh"

# ---------- 2. docker ----------
step "docker (SWE-bench scoring runs each repo's tests in a container)"
docker --version || sudo apt-get install -y -qq docker.io
sudo systemctl start docker || true
sudo docker run --rm hello-world > /dev/null 2>&1 && echo "docker OK" || echo "WARN: docker smoke failed"

# ---------- 3. CUDA toolkit ----------
step "CUDA toolkit (driver present, nvcc absent on this image)"
if ! command -v nvcc >/dev/null 2>&1 && [ ! -x /usr/local/cuda-12.8/bin/nvcc ]; then
  sudo apt-get install -y -qq cuda-toolkit-12-8 || sudo apt-get install -y -qq cuda-toolkit-12-6
fi
for c in /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda; do
  [ -x "$c/bin/nvcc" ] && { export CUDA_HOME=$c; break; }
done
[ -n "${CUDA_HOME:-}" ] || die "no CUDA toolkit after install"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
nvcc --version | tail -2

# ---------- 4. llama.cpp ----------
step "build llama-server (L40S is sm_89)"
git clone --depth 1 https://github.com/ggml-org/llama.cpp "$W/llama.cpp"
cd "$W/llama.cpp"
cmake -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
      -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT="$CUDA_HOME"
cmake --build build --config Release -j"$(nproc)" --target llama-server
need "$W/llama.cpp/build/bin/llama-server" 100000

# ---------- 5. model + subset ----------
step "download Qwen3.6-35B-A3B Q4_K_M (Jetson deployment target)"
export HF_HUB_ENABLE_HF_TRANSFER=1
python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download("unsloth/Qwen3.6-35B-A3B-GGUF", allow_patterns=["*Q4_K_M*"],
                  local_dir=os.path.expanduser("~/swebench/models/qwen-q4"))
PY
find "$W/models/qwen-q4" -name '*.gguf' | sort | head -1 > "$W/GGUF_PATH"
GGUF=$(cat "$W/GGUF_PATH"); echo "GGUF=$GGUF"
need "$GGUF" 1000000000

step "fixed 50-instance subset (seed 0, reproducible across quant tiers)"
python3 - <<'PY'
import json, os, random
from datasets import load_dataset
ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
idx = sorted(random.Random(0).sample(range(len(ds)), 50))
ids = [ds[i]["instance_id"] for i in idx]
h = os.path.expanduser("~/swebench")
json.dump(ids, open(f"{h}/subset50.json","w"), indent=2)
open(f"{h}/filter50.txt","w").write("^(" + "|".join(ids) + ")$")
print("total", len(ds), "picked", len(ids), "first", ids[0])
PY
need "$W/filter50.txt" 100
touch "$W/SETUP_DONE"

# ---------- 6. serve ----------
step "start llama-server"
"$W/llama.cpp/build/bin/llama-server" -m "$GGUF" -ngl 999 -c 131072 \
  --parallel 4 --host 127.0.0.1 --port 8080 --jinja > "$W/logs/server.log" 2>&1 &
SRV=$!
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && { echo "server healthy"; break; }
  sleep 10
done
curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 || die "server never healthy"
touch "$W/SERVING"

step "endpoint smoke test"
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
 -d '{"model":"local","messages":[{"role":"user","content":"Reply with exactly: PONG"}],"max_tokens":50,"temperature":0}' \
 | python3 -c "import sys,json;print('SMOKE:',repr(json.load(sys.stdin)['choices'][0]['message']['content'][:300]))"

# ---------- 7. pre-flight ----------
step "PRE-FLIGHT: 1 instance, 30 steps"
FIRST=$(python3 -c "import json,os;print(json.load(open(os.path.expanduser('~/swebench/subset50.json')))[0])")
echo "preflight instance: $FIRST"
cd "$W"
timeout 2400 mini-extra swebench --subset verified --split test \
  --filter "^${FIRST}$" --model-class litellm_textbased -m openai/local \
  -c swebench_backticks.yaml \
  -c model.model_kwargs.api_base=http://127.0.0.1:8080/v1 \
  -c model.model_kwargs.drop_params=true \
  -c agent.step_limit=30 -o "$W/preflight" -w 1 2>&1 | tail -30 || true

step "GO/NO-GO"
GO=$(python3 - <<'PY'
import json, os
p = os.path.expanduser("~/swebench/preflight/preds.json")
if not os.path.exists(p): print("NOGO_NO_PREDS"); raise SystemExit
d = json.load(open(p))
if not d: print("NOGO_EMPTY_PREDS"); raise SystemExit
print("GO" if (list(d.values())[0].get("model_patch") or "").strip() else "NOGO_EMPTY_PATCH")
PY
)
echo "GO_DECISION=$GO"
python3 - <<'PY'
import glob, json, os
for f in sorted(glob.glob(os.path.expanduser("~/swebench/preflight/*/*.traj.json")))[:1]:
    d = json.load(open(f)); m = d.get("messages") or d.get("trajectory") or []
    print("messages:", len(m))
    for x in m[:8]:
        print("---", x.get("role"), "---"); print(str(x.get("content"))[:700])
PY
[ "$GO" = "GO" ] || { echo "STOPPING: scaffold cannot drive this model ($GO)."; touch "$W/NOGO"; exit 1; }
touch "$W/PREFLIGHT_OK"

# ---------- 8. batch ----------
# step_limit 60 vs the leaderboard's 250: 250 at local speed will not finish in
# the deadline. This DEPRESSES the resolve rate and must be reported with it.
step "BATCH: 50 instances, 4 workers, step_limit 60"
timeout 10800 mini-extra swebench --subset verified --split test \
  --filter "$(cat "$W/filter50.txt")" --model-class litellm_textbased -m openai/local \
  -c swebench_backticks.yaml \
  -c model.model_kwargs.api_base=http://127.0.0.1:8080/v1 \
  -c model.model_kwargs.drop_params=true \
  -c agent.step_limit=60 -o "$W/results" -w 4 2>&1 | tail -40 || true
touch "$W/BATCH_DONE"
python3 - <<'PY'
import json, os
p = os.path.expanduser("~/swebench/results/preds.json")
if os.path.exists(p):
    d = json.load(open(p))
    print("attempted:", len(d), "non-empty patches:",
          sum(1 for v in d.values() if (v.get("model_patch") or "").strip()))
PY

# ---------- 9. score ----------
step "SCORING with the official SWE-bench Docker harness"
kill "$SRV" 2>/dev/null || true; SRV=""; sleep 10
cd "$W"
timeout 5400 python3 -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path "$W/results/preds.json" \
  --max_workers 4 --run_id qwen_q4km_bashonly 2>&1 | tail -30 || true

step "FINAL"
find "$H" -maxdepth 3 -name '*qwen_q4km_bashonly*' 2>/dev/null | head
touch "$W/ALL_DONE"
step "ALL DONE"
