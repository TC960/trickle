#!/bin/bash
# Rerun the Q8 batch with the SAME context as the Q4 run.
#
# The first Q8 attempt scored 0/50 because I cut -c to 32768. llama.cpp divides
# the context across --parallel slots, so 32768/4 gave each agent 8k -- prompt
# plus history consumed nearly all of it and every response was truncated
# mid-action (finish_reason=length), ending in RepeatedFormatError with no
# patch. The Q4 run used -c 131072, i.e. 32k per slot.
#
# I cut it fearing 35GB of Q8_0 weights plus KV would not fit 46GB. The Q4 run
# disproves that: 22GB model + 131072 ctx totalled 24GB, so KV is only ~2GB --
# 30 of this model's 40 layers use linear attention with fixed-size state, so
# KV barely grows with context. 35 + 2 = 37GB fits comfortably.
#
# Also wipes results/: mini-swe-agent skips instances already in preds.json, and
# 50 stale empty entries would make this rerun a no-op.
set -eux
set -o pipefail
H=$HOME
W=$H/swebench
exec > "$W/logs/q8rerun.log" 2>&1
step () { echo "=== $(date '+%F %H:%M:%S') STEP: $* ==="; }
die  () { echo "FATAL: $*"; exit 1; }
SRV=""
trap '[ -n "$SRV" ] && kill "$SRV" 2>/dev/null; true' EXIT

source "$W/venv/bin/activate"; source "$W/env.sh"

step "kill any running server and clear stale zero-patch results"
for p in $(ps -eo pid,args | grep -F 'llama-server' | grep -vF grep | awk '{print $1}'); do
  kill "$p" 2>/dev/null || true
done
sleep 8
rm -rf "$W/results"; mkdir -p "$W/results"

step "serve Q8_0 with -c 131072 (32k per slot, identical to the Q4 run)"
BIN="$W/llama.cpp/build/bin/llama-server"
[ -x "$BIN" ] || die "llama-server missing"
GGUF=$(cat "$W/GGUF_PATH")
"$BIN" -m "$GGUF" -ngl 999 -c 131072 --parallel 4 \
  --host 127.0.0.1 --port 8080 --jinja --reasoning-budget 0 \
  > "$W/logs/server.log" 2>&1 &
SRV=$!
for i in $(seq 1 180); do
  curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 && { echo "healthy"; break; }
  sleep 10
done
curl -sf http://127.0.0.1:8080/health >/dev/null 2>&1 || die "server never healthy"
nvidia-smi --query-gpu=memory.used --format=csv,noheader

step "smoke test"
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
 -d '{"model":"local","messages":[{"role":"user","content":"Reply with exactly: PONG"}],"max_tokens":400,"temperature":0}' \
 | python3 -c "
import sys,json
c=(json.load(sys.stdin)['choices'][0]['message'].get('content') or '').strip()
print('SMOKE:',repr(c)[:200]); assert c, 'FATAL empty content'
"

step "BATCH: 50 instances, 4 workers, step_limit 100"
set +e
cd "$W"
timeout 10800 mini-extra swebench --subset verified --split test \
  --filter "$(cat "$W/filter50.txt")" --model-class litellm_textbased -m openai/local \
  -c swebench_backticks.yaml \
  -c model.model_kwargs.api_base=http://127.0.0.1:8080/v1 \
  -c model.model_kwargs.drop_params=true \
  -c agent.step_limit=100 -o "$W/results" -w 4 2>&1 | tail -30
touch "$W/BATCH_DONE"
python3 -c "
import json,os
d=json.load(open(os.path.expanduser('~/swebench/results/preds.json')))
print('attempted:',len(d),'with-patch:',sum(1 for v in d.values() if (v.get('model_patch') or '').strip()))
"

step "SCORE"
kill "$SRV" 2>/dev/null; SRV=""; sleep 10
RUNID=qwen_q8_bashonly bash "$H/score.sh" 2>&1 | tail -20
touch "$W/ALL_DONE"
step "Q8 RERUN DONE"
