#!/bin/bash
# Poll the boxes, back results up CONTINUOUSLY, and stop everything once the
# work is done or the deadline passes.
#
# Replaces autoshutdown.sh, which had three faults that together cost ~5 hours
# of unrecoverable results on 2026-08-22:
#
#   1. `B1=${B1:-1}` gave an unreachable box the same reading as a busy one, so
#      a box that had silently died looked like it was working. The "both idle"
#      exit could then never fire, and the run only ended on the deadline.
#   2. Business was detected via `nvidia-smi --query-compute-apps=pid`, which
#      returns nothing inside many VMs regardless of load -- box2 read as idle
#      the entire time it was running benchmarks. Memory and utilization are
#      visible where PIDs are not.
#   3. Backup ran only at the start and the end. When the box died in between,
#      everything since the last backup went with it.
#
# So: reachability is its own state, load is measured by memory, and backups
# run every cycle.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1
LOG="$ROOT/results-backup/_watchdog.log"
# Override with:  BOXES="box-a box-b" bash watchdog.sh
read -r -a BOXES <<< "${BOXES:-military-turquoise-crawdad}"
MAX_HOURS=${MAX_HOURS:-8}
POLL=${POLL:-300}
IDLE_MB=${IDLE_MB:-2000}        # below this much GPU memory = not working
STOP_WHEN_DONE=${STOP_WHEN_DONE:-1}

say () { echo "$(date '+%H:%M:%S')  $*" | tee -a "$LOG"; }
SSHOPTS=(-n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
         -o LogLevel=ERROR -o ConnectTimeout=15)

# Echoes "busy", "idle", or "unreachable" -- three states, never conflated.
box_state () {
  local out
  out=$(ssh "${SSHOPTS[@]}" "$1" \
        "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits" 2>/dev/null)
  [ -z "$out" ] && { echo unreachable; return; }
  local total=0
  while read -r mb; do
    [[ "$mb" =~ ^[0-9]+$ ]] && total=$((total + mb))
  done <<< "$out"
  [ "$total" -ge "$IDLE_MB" ] && echo busy || echo idle
}

sync_box () {
  local box=$1
  mkdir -p "$ROOT/results-backup/$box/logs"
  scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o LogLevel=ERROR -o ConnectTimeout=15 \
      "$box:/ephemeral/work/out/*.jsonl" "$ROOT/results-backup/$box/" 2>/dev/null
  scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o LogLevel=ERROR -o ConnectTimeout=15 \
      "$box:/ephemeral/work/out/*.md" "$ROOT/results-backup/$box/" 2>/dev/null
  # Logs can be hundreds of MB of progress bars; take the tail only.
  ssh "${SSHOPTS[@]}" "$box" \
      'cd /ephemeral/work/logs 2>/dev/null && for f in *.log; do tail -c 200000 "$f" > /tmp/t_$f; done' 2>/dev/null
  scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o LogLevel=ERROR -o ConnectTimeout=15 \
      "$box:/tmp/t_*.log" "$ROOT/results-backup/$box/logs/" 2>/dev/null
}

say "=== watchdog armed (max ${MAX_HOURS}h, poll ${POLL}s, idle<${IDLE_MB}MB) ==="
DEADLINE=$(( $(date +%s) + MAX_HOURS*3600 ))
IDLE_STREAK=0

while true; do
  LINE=""; ANY_BUSY=0; ANY_REACHABLE=0
  for BOX in "${BOXES[@]}"; do
    ST=$(box_state "$BOX")
    LINE="$LINE  ${BOX%%-*}=$ST"
    [ "$ST" = busy ] && ANY_BUSY=1
    [ "$ST" != unreachable ] && { ANY_REACHABLE=1; sync_box "$BOX"; }
  done
  NFILES=$(find "$ROOT/results-backup" -name '*.jsonl' | wc -l | tr -d ' ')
  NOW=$(date +%s)
  say "$LINE  | backed up $NFILES jsonl  | $(( (DEADLINE-NOW)/60 ))m left"

  if [ "$ANY_BUSY" -eq 0 ] && [ "$ANY_REACHABLE" -eq 1 ]; then
    IDLE_STREAK=$((IDLE_STREAK + 1))
    say "  idle streak $IDLE_STREAK/3"
    [ "$IDLE_STREAK" -ge 3 ] && { say "confirmed idle"; break; }
  else
    IDLE_STREAK=0
  fi

  [ "$NOW" -ge "$DEADLINE" ] && { say "deadline reached"; break; }
  sleep "$POLL"
done

say "=== final sync ==="
for BOX in "${BOXES[@]}"; do sync_box "$BOX"; done
NFILES=$(find "$ROOT/results-backup" -name '*.jsonl' | wc -l | tr -d ' ')
say "backed up $NFILES jsonl files, $(du -sh "$ROOT/results-backup" | cut -f1)"

if [ "$NFILES" -lt 10 ]; then
  say "!! ABORT: only $NFILES result files. NOT stopping anything."
  exit 1
fi
if [ "$STOP_WHEN_DONE" != "1" ]; then
  say "STOP_WHEN_DONE=0 -- leaving instances running."
  exit 0
fi

say "=== stopping instances ==="
for BOX in "${BOXES[@]}"; do brev stop "$BOX" 2>&1 | tail -1 | tee -a "$LOG"; done
sleep 20
brev ls 2>&1 | tail -4 | tee -a "$LOG"
say "=== DONE ==="
