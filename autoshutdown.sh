#!/bin/bash
# Wait for all work to finish, back everything up, VERIFY the backup, then stop
# both instances. Never stops without a verified backup -- /ephemeral is wiped
# on stop, so an unverified shutdown would destroy the results.
# Derived from the script's own location -- a hardcoded absolute path
# leaked the local username and employer directory into a public repo.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1
LOG="$ROOT/results-backup/_autoshutdown.log"
BOX1=gorgeous-copper-mite
BOX2=military-turquoise-crawdad
MAX_HOURS=${MAX_HOURS:-7}
say () { echo "$(date '+%H:%M:%S')  $*" | tee -a "$LOG"; }

ssh_q () { ssh -n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
             -o LogLevel=ERROR -o ConnectTimeout=20 "$1" "$2" 2>/dev/null; }

say "=== autoshutdown armed (max ${MAX_HOURS}h) ==="
DEADLINE=$(( $(date +%s) + MAX_HOURS*3600 ))

while true; do
  # Done when neither box has a GPU process running.
  B1=$(ssh_q $BOX1 "nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l")
  B2=$(ssh_q $BOX2 "nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l")
  B1=${B1:-1}; B2=${B2:-1}
  NOW=$(date +%s)
  say "box1 gpu procs=$B1  box2 gpu procs=$B2  ($(( (DEADLINE-NOW)/60 ))m left)"

  if [ "$B1" -eq 0 ] && [ "$B2" -eq 0 ]; then
    say "both idle -- confirming (5 min)"
    sleep 300
    B1=$(ssh_q $BOX1 "nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l")
    B2=$(ssh_q $BOX2 "nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l")
    [ "${B1:-1}" -eq 0 ] && [ "${B2:-1}" -eq 0 ] && { say "confirmed idle"; break; }
    say "work resumed, continuing to wait"
  fi
  [ "$NOW" -ge "$DEADLINE" ] && { say "deadline reached -- shutting down anyway"; break; }
  sleep 300
done

say "=== regenerating report ==="
ssh_q $BOX1 "cd /ephemeral/work/code && /ephemeral/work/venv/bin/python make_report.py > /ephemeral/work/out/REPORT.md 2>&1; wc -l /ephemeral/work/out/REPORT.md"

say "=== final backup ==="
BEFORE=$(find "$ROOT/results-backup" -name '*.jsonl' | wc -l | tr -d ' ')
bash "$ROOT/backup.sh"
# Grab the report and logs too, not just the metrics.
for BOX in $BOX1 $BOX2; do
  mkdir -p "$ROOT/results-backup/$BOX/logs"
  scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
      "$BOX:/ephemeral/work/out/*.md" "$ROOT/results-backup/$BOX/" 2>/dev/null
  scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
      "$BOX:/ephemeral/work/logs/*.log" "$ROOT/results-backup/$BOX/logs/" 2>/dev/null
done
AFTER=$(find "$ROOT/results-backup" -name '*.jsonl' | wc -l | tr -d ' ')
REPORTS=$(find "$ROOT/results-backup" -name 'REPORT.md' | wc -l | tr -d ' ')
say "backup: $BEFORE -> $AFTER jsonl files, $REPORTS report(s), $(du -sh "$ROOT/results-backup" | cut -f1)"

# SAFETY GATE: refuse to stop if the backup looks empty.
if [ "$AFTER" -lt 10 ]; then
  say "!! ABORT: only $AFTER result files backed up. NOT stopping instances."
  say "!! Investigate before stopping - /ephemeral is wiped on stop."
  exit 1
fi

say "=== stopping instances ==="
for BOX in $BOX1 $BOX2; do
  say "stopping $BOX ..."
  brev stop "$BOX" 2>&1 | tail -2 | tee -a "$LOG"
done
sleep 20
brev ls 2>&1 | tail -4 | tee -a "$LOG"
say "=== DONE. results in $ROOT/results-backup ==="
