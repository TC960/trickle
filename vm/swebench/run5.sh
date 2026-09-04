#!/bin/bash
# Resume from the llama.cpp build. Everything before it succeeded: deadline
# armed, Python 3.11, mini-swe-agent + swebench imports verified, Docker, CUDA
# toolkit 12.8.
#
# The build failed on a compiler mismatch baked into this image: `gcc` is
# 12.3.0 but `g++` is 11.4.0, and cc1plus exists only under
# /usr/lib/gcc/x86_64-linux-gnu/11/. nvcc invokes gcc for C++, gcc-12 looks for
# a v12 cc1plus that was never installed, and dies with
# "cannot execute 'cc1plus'". Install g++-12 and pin host/CXX compilers so
# every layer agrees on one version instead of silently disagreeing.
set -ux
H=$HOME
W=$H/swebench
exec > "$W/logs/run5.log" 2>&1

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

source "$W/venv/bin/activate"
source "$W/env.sh"

step "install matching g++-12 (gcc is 12.3.0, g++ was 11.4.0)"
sudo apt-get install -y -qq g++-12 gcc-12
ls /usr/lib/gcc/x86_64-linux-gnu/12/cc1plus || die "cc1plus for gcc-12 still missing"
gcc-12 --version | head -1
g++-12 --version | head -1

for c in /usr/local/cuda-12.8 /usr/local/cuda-12.6 /usr/local/cuda; do
  [ -x "$c/bin/nvcc" ] && { export CUDA_HOME=$c; break; }
done
[ -n "${CUDA_HOME:-}" ] || die "no CUDA toolkit"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
nvcc --version | tail -2

step "verify llama-server (already built; size is NOT a validity test -- it is
# an ~18KB thin exe against libllama-server-impl.so, which my earlier 100KB
# gate false-failed. Check that it RUNS.)"
"$W/llama.cpp/build/bin/llama-server" --version 2>&1 | head -3 || die "llama-server will not run"

step "download Qwen3.6-35B-A3B Q4_K_M"
export HF_HUB_ENABLE_HF_TRANSFER=1
if [ ! -s "$W/GGUF_PATH" ]; then
  python3 - <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download("unsloth/Qwen3.6-35B-A3B-GGUF", allow_patterns=["*Q4_K_M*"],
                  local_dir=os.path.expanduser("~/swebench/models/qwen-q4"))
PY
  find "$W/models/qwen-q4" -name '*.gguf' | sort | head -1 > "$W/GGUF_PATH"
fi
GGUF=$(cat "$W/GGUF_PATH"); echo "GGUF=$GGUF"
need "$GGUF" 1000000000

step "50-instance subset (seed 0)"
if [ ! -s "$W/filter50.txt" ]; then
  python3 - <<'PY'
import json, os, random
from datasets import load_dataset
ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
idx = sorted(random.Random(0).sample(range(len(ds)), 50))
ids = [ds[i]["instance_id"] for i in idx]
h = os.path.expanduser("~/swebench")
json.dump(ids, open(f"{h}/subset50.json","w"), indent=2)
open(f"{h}/filter50.txt","w").write("^(" + "|".join(ids) + ")$")
print("picked", len(ids), "first", ids[0])
PY
fi
need "$W/filter50.txt" 100

step "start llama-server"
"$W/llama.cpp/build/bin/llama-server" -m "$GGUF" -ngl 999 -c 131072 \
  --parallel 4 --host 127.0.0.1 --port 8080 --jinja --reasoning-budget 0 \
  > "$W/logs/server.log" 2>&1 &
SRV=$!
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && { echo "server healthy"; break; }
  sleep 10
done
curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 || die "server never healthy"
touch "$W/SERVING"

step "smoke test"
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
 -d '{"model":"local","messages":[{"role":"user","content":"Reply with exactly: PONG"}],"max_tokens":400,"temperature":0}' \
 | python3 -c "
import sys,json
m=json.load(sys.stdin)['choices'][0]['message']
c=(m.get('content') or '').strip()
print('SMOKE:',repr(c)[:300])
assert c, 'FATAL: model returned EMPTY content -- thinking is eating the budget'
"

step "BATCH: 50 instances, 4 workers, step_limit 100"
timeout 10800 mini-extra swebench --subset verified --split test \
  --filter "$(cat "$W/filter50.txt")" --model-class litellm_textbased -m openai/local \
  -c swebench_backticks.yaml \
  -c model.model_kwargs.api_base=http://127.0.0.1:8080/v1 \
  -c model.model_kwargs.drop_params=true \
  -c agent.step_limit=100 -o "$W/results" -w 4 2>&1 | tail -40 || true
touch "$W/BATCH_DONE"
python3 - <<'PY'
import json, os
p = os.path.expanduser("~/swebench/results/preds.json")
if os.path.exists(p):
    d = json.load(open(p))
    print("attempted:", len(d), "non-empty patches:",
          sum(1 for v in d.values() if (v.get("model_patch") or "").strip()))
PY

step "SCORING (official SWE-bench Docker harness)"
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
