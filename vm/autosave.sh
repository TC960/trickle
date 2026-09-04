# Layer 1+2: snapshot results off the scratch disk, every 5 minutes, forever.
#   $HOME/results-safe  -- survives /ephemeral being wiped
#   the OTHER box        -- survives this instance dying
# Results are a few MB of JSONL, so this is essentially free.
PEER="${PEER:-}"
mkdir -p "$HOME/results-safe"
while true; do
  cp -f /ephemeral/work/out/*.jsonl /ephemeral/work/out/*.md \
        "$HOME/results-safe/" 2>/dev/null
  # Keep a timestamped weekly-rotating copy so a corrupt write can't erase history.
  STAMP=$(date +%H)
  mkdir -p "$HOME/results-safe/hourly-$STAMP"
  cp -f /ephemeral/work/out/*.jsonl "$HOME/results-safe/hourly-$STAMP/" 2>/dev/null
  if [ -n "$PEER" ]; then
    scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o LogLevel=ERROR -o ConnectTimeout=15 \
        /ephemeral/work/out/*.jsonl "$PEER:$HOME/results-safe/peer/" 2>/dev/null
  fi
  sleep 300
done
