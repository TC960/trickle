#!/bin/bash
# Wait for ternary-h200, sync code, start phase 1.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$ROOT" || exit 1
BOX=ternary-h200
LOG="$ROOT/results-backup/_launch_session.log"
say () { echo "$(date '+%H:%M:%S')  $*" | tee -a "$LOG"; }
SSHOPTS=(-n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
         -o LogLevel=ERROR -o ConnectTimeout=15)
SCPOPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
         -o LogLevel=ERROR -o ConnectTimeout=15)

say "waiting for $BOX"
for i in $(seq 1 60); do
  brev refresh >/dev/null 2>&1
  ssh "${SSHOPTS[@]}" "$BOX" 'echo ok' 2>/dev/null | grep -q ok && { say "up after $((i*20))s"; break; }
  sleep 20
done
ssh "${SSHOPTS[@]}" "$BOX" 'echo ok' 2>/dev/null | grep -q ok || { say "!! unreachable"; exit 1; }

say "disk: $(ssh "${SSHOPTS[@]}" "$BOX" 'df -h / | tail -1')"

say "syncing code"
ssh "${SSHOPTS[@]}" "$BOX" 'mkdir -p /ephemeral/work/code/tests'
scp -q "${SCPOPTS[@]}" "$ROOT"/vm/*.py "$BOX:/ephemeral/work/code/"
scp -q "${SCPOPTS[@]}" "$ROOT"/vm/session_queue.sh "$BOX:/ephemeral/work/code/"
scp -q "${SCPOPTS[@]}" "$ROOT"/tests/*.py "$BOX:/ephemeral/work/code/tests/"
ssh "${SSHOPTS[@]}" "$BOX" 'rm -rf /ephemeral/work/code/airllm_ternary'
scp -qr "${SCPOPTS[@]}" "$ROOT"/airllm_ternary "$BOX:/ephemeral/work/code/"

# Verify the package actually arrived intact -- a stale checkout has already
# cost this project two failed runs (missing --bits, missing uniform_ste).
REMOTE_OK=$(ssh "${SSHOPTS[@]}" "$BOX" \
  'cd /ephemeral/work/code && /ephemeral/work/venv/bin/python -c "
from airllm_ternary.uniform import quantize_uniform, dequantize_uniform
from airllm_ternary.linear import UniformLinear
from airllm_ternary.policy import PrecisionPolicy
assert PrecisionPolicy(bits=4).bits == 4
print(\"imports-ok\")" 2>&1 | tail -1')
say "import check: $REMOTE_OK"
case "$REMOTE_OK" in *imports-ok*) ;; *) say "!! aborting, code not importable"; exit 1;; esac

ssh "${SSHOPTS[@]}" "$BOX" '/ephemeral/work/venv/bin/python -c "import pytest" 2>/dev/null' \
  || ssh "${SSHOPTS[@]}" "$BOX" 'export PATH=$HOME/.local/bin:$PATH; VIRTUAL_ENV=/ephemeral/work/venv uv pip install -q pytest' 2>/dev/null

say "starting phase 1"
ssh "${SSHOPTS[@]}" "$BOX" \
  'tmux kill-session -t sess 2>/dev/null; tmux new-session -d -s sess "bash /ephemeral/work/code/session_queue.sh 2>&1 | tee -a /ephemeral/work/logs/session_stdout.log"'
sleep 10
say "tmux: $(ssh "${SSHOPTS[@]}" "$BOX" 'tmux ls 2>&1 | head -2')"
say "=== launched ==="
