#!/bin/bash
# VAB-WebArena-Lite harness + a 3-task pilot.
#
# Pilot first, deliberately. Today five separate harness bugs each produced a
# confident zero (dead PyPI package, silent tool-calling mismatch, empty
# `content` from thinking mode, stale preds skipping every instance, a v5 API
# change). A 3-task run costs minutes and distinguishes "model is bad at
# browsing" from "harness is mis-driving the model" -- which look identical in
# an aggregate score.
#
# Also selects the DETERMINISTIC task subset. WebArena grades ~14% of tasks with
# an LLM judge, and in the default config that judge is the SAME local endpoint
# -- i.e. the model grading its own work. Rather than patch in a separate judge,
# just exclude those tasks and report an honest denominator.
set -ux
H=$HOME
W=$H/browse
V=$W/VisualAgentBench/VAB-WebArena-Lite
mkdir -p "$W/logs"
exec > "$W/logs/harness.log" 2>&1
step () { echo "=== $(date '+%F %H:%M:%S') STEP: $* ==="; }
die  () { echo "FATAL: $*"; exit 1; }

IP=$(cat "$W/HOST_IP" | cut -d= -f2)
echo "IP=$IP"

step "clone harness"
cd "$W"
[ -d "$W/VisualAgentBench" ] || git clone --depth 1 https://github.com/THUDM/VisualAgentBench.git
cd "$V"
[ -d visualwebarena ] || git clone https://github.com/web-arena-x/visualwebarena.git visualwebarena
(cd visualwebarena && git reset --hard ad57aae4dad71531504726900b80db02e0526158)
bash replace.sh || echo "WARN replace.sh returned nonzero"

step "python env (harness pins want 3.10, which is the system python here)"
python3 -m venv "$V/venv"
source "$V/venv/bin/activate"
python3 --version
pip install --upgrade pip -q
pip install -q -r requirements.txt || echo "WARN: some requirements failed"
python3 -m playwright install chromium
python3 -m playwright install-deps chromium || true
pip install -q -e . || echo "WARN: editable install returned nonzero"

step "site env vars (ALL seven are asserted non-empty even when unused)"
cat > "$W/wa_env.sh" <<EOF
export DATASET=webarena
export SHOPPING="http://$IP:7770"
export SHOPPING_ADMIN="http://$IP:7780/admin"
export REDDIT="http://$IP:9999"
export GITLAB="http://$IP:8023"
export MAP="http://localhost:3000"
export WIKIPEDIA="http://localhost:8888"
export HOMEPAGE="http://localhost:4399"
export OPENAI_API_URL="http://127.0.0.1:8080/v1"
export OPENAI_API_BASE="http://127.0.0.1:8080/v1"
export OPENAI_BASE_URL="http://127.0.0.1:8080/v1"
export OPENAI_API_KEY="sk-noop"
EOF
source "$W/wa_env.sh"

step "generate test data"
python3 scripts/generate_test_data.py 2>&1 | tail -5
bash prepare.sh 2>&1 | tail -5 || echo "WARN prepare.sh nonzero"

step "select DETERMINISTIC, NON-MAP tasks"
python3 - <<'PY'
import glob, json, os
d = os.path.expanduser("~/browse/VisualAgentBench/VAB-WebArena-Lite/config_files/wa/test_webarena_lite")
files = sorted(glob.glob(f"{d}/*.json"), key=lambda p: int(os.path.basename(p).split('.')[0]))
det, fuzzy, mapped = [], [], []
for f in files:
    try: c = json.load(open(f))
    except Exception: continue
    idx = c.get("task_id", int(os.path.basename(f).split('.')[0]))
    sites = c.get("sites", [])
    if "map" in sites or "wikipedia" in sites:
        mapped.append(idx); continue
    ev = json.dumps(c.get("eval", {}))
    (fuzzy if ("fuzzy_match" in ev or "must_include" in ev and "string_note" in ev) else det).append(idx)
print(f"total={len(files)} map/wiki={len(mapped)} fuzzy(judge)={len(fuzzy)} deterministic={len(det)}")
out = os.path.expanduser("~/browse")
json.dump(det,   open(f"{out}/tasks_deterministic.json","w"))
json.dump(fuzzy, open(f"{out}/tasks_fuzzy.json","w"))
print("first 15 deterministic:", det[:15])
PY

step "PILOT: 3 deterministic tasks"
PILOT=$(python3 -c "import json,os;print(' '.join(map(str,json.load(open(os.path.expanduser('~/browse/tasks_deterministic.json')))[:3])))")
echo "pilot tasks: $PILOT"
mkdir -p "$W/results_pilot"
for t in $PILOT; do
  echo "--- task $t ---"
  timeout 900 python3 run.py \
    --instruction_path agent/prompts/jsons/p_webrl_chat.json \
    --test_config_base_dir config_files/wa/test_webarena_lite \
    --test_start_idx "$t" --test_end_idx "$((t+1))" \
    --result_dir "$W/results_pilot" \
    --provider openai --mode chat --model local --planner_ip '' \
    --temperature 0.0 --max_obs_length 0 --max_tokens 2048 \
    --parsing_failure_th 5 --repeating_action_failure_th 5 \
    --viewport_width 1280 --viewport_height 720 \
    --action_set_tag webrl_id --observation_type webrl 2>&1 | tail -25
done

step "PILOT ARTIFACTS"
ls -la "$W/results_pilot" 2>/dev/null | head
find "$W/results_pilot" -name '*.json' -o -name '*.html' 2>/dev/null | head
touch "$W/PILOT_DONE"
step "PILOT DONE"
