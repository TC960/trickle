#!/bin/bash
# Layer 3: pull results from both cloud boxes to this laptop, where OneDrive
# then syncs them off-machine. Run by launchd every 10 minutes; also fine to run
# by hand. Silent and idempotent -- never touches the GPUs.
# Derived from the script's own location -- a hardcoded absolute path
# leaked the local username and employer directory into a public repo.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$ROOT/results-backup"
LOG="$DEST/_backup.log"
mkdir -p "$DEST"
OPTS="-q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -F $HOME/.ssh/config"
for BOX in gorgeous-copper-mite military-turquoise-crawdad; do
  mkdir -p "$DEST/$BOX"
  /usr/bin/scp $OPTS "$BOX:/ephemeral/work/out/*.jsonl" "$DEST/$BOX/" 2>/dev/null
  /usr/bin/scp $OPTS "$BOX:/ephemeral/work/out/*.md"    "$DEST/$BOX/" 2>/dev/null
  /usr/bin/scp $OPTS "$BOX:\$HOME/results-safe/*.jsonl" "$DEST/$BOX/" 2>/dev/null
done
echo "$(date '+%Y-%m-%d %H:%M:%S')  files=$(find "$DEST" -name '*.jsonl' | wc -l | tr -d ' ')" >> "$LOG"
tail -200 "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"
