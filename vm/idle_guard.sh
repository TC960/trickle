#!/bin/bash
# Self-shutdown guard. Runs ON THE BOX, every few minutes, via systemd timer.
#
# Every previous version of this lived on the laptop, which meant it died when
# the laptop slept, when a session ended, or when someone forgot to re-arm it.
# That has cost real money twice: ~10 hours of idle A100+H200 once, and several
# more hours after a restart where re-arming was simply forgotten.
#
# Running it here removes the human and the laptop from the loop entirely.
#
# "Idle" requires BOTH signals to agree, because either alone gives false
# positives: GPU memory is near zero while a job is loading a 59 GB checkpoint
# from disk, and a stray python process can linger after work is finished.
#
# Escape hatches:
#   touch /ephemeral/work/KEEPALIVE     never shut down
#   systemctl --user stop idle-guard.timer
set -u
W=/ephemeral/work
STATE=$W/.idle_guard_state
LOG=$W/logs/idle_guard.log
IDLE_MB=${IDLE_MB:-2000}          # GPU memory below this counts as idle
NEEDED=${NEEDED:-3}               # consecutive idle checks before shutdown
mkdir -p "$W/logs"

say () { echo "$(date '+%F %H:%M:%S')  $*" >> "$LOG"; }

if [ -f "$W/KEEPALIVE" ]; then
  say "KEEPALIVE present -- not shutting down"
  echo 0 > "$STATE"
  exit 0
fi

# Signal 1: GPU memory across all devices.
gpu_mb=0
while read -r mb; do
  [[ "$mb" =~ ^[0-9]+$ ]] && gpu_mb=$((gpu_mb + mb))
done < <(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)

# Signal 2: any of our python work still running.
# pgrep -c prints "0" AND exits non-zero when nothing matches, so a `|| echo 0`
# fallback appends a second line and breaks the integer test. Take line one.
procs=$(pgrep -fc "python.*(distill_seq|deep_eval|benchmarks|stream_bench|build_shards|qlora_recover|sensitivity|mlp_prune|perplexity)" 2>/dev/null | head -1)
procs=${procs:-0}

if [ "$gpu_mb" -ge "$IDLE_MB" ] || [ "$procs" -gt 0 ]; then
  say "busy (gpu ${gpu_mb}MB, procs ${procs}) -- resetting counter"
  echo 0 > "$STATE"
  exit 0
fi

n=$(( $(cat "$STATE" 2>/dev/null || echo 0) + 1 ))
echo "$n" > "$STATE"
say "idle (gpu ${gpu_mb}MB, procs ${procs}) -- streak ${n}/${NEEDED}"

if [ "$n" -ge "$NEEDED" ]; then
  say "=== IDLE CONFIRMED -- SHUTTING DOWN ==="
  say "results are on /ephemeral, which survives a stop on this provider"
  sync
  sudo shutdown -h now
fi
