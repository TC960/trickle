#!/bin/bash
# Wait for ternary-h200 to become reachable, push the code, start the queue.
# Backgrounded because provisioning takes minutes and there is other work to do.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT" || exit 1
BOX=ternary-h200
LOG="$ROOT/results-backup/_launch_h200.log"
say () { echo "$(date '+%H:%M:%S')  $*" | tee -a "$LOG"; }
SSHOPTS=(-n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
         -o LogLevel=ERROR -o ConnectTimeout=15)
SCPOPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
         -o LogLevel=ERROR -o ConnectTimeout=15)

say "waiting for $BOX to accept ssh"
for i in $(seq 1 90); do
  STATUS=$(brev ls 2>/dev/null | awk -v b="$BOX" '$1==b {print $2}')
  if [ "$STATUS" = "RUNNING" ]; then
    brev refresh >/dev/null 2>&1
    if ssh "${SSHOPTS[@]}" "$BOX" 'echo ok' 2>/dev/null | grep -q ok; then
      say "reachable after $((i*20))s"; break
    fi
  fi
  [ $((i % 6)) -eq 0 ] && say "  status=$STATUS  ($((i*20))s elapsed)"
  sleep 20
done

ssh "${SSHOPTS[@]}" "$BOX" 'echo ok' 2>/dev/null | grep -q ok || {
  say "!! never became reachable -- giving up"; exit 1; }

# Where the big disk is. nebius may not use /ephemeral like Hyperstack did.
W=$(ssh "${SSHOPTS[@]}" "$BOX" '[ -d /ephemeral ] && echo /ephemeral/work || echo $HOME/work')
say "work dir on box: $W"
ssh "${SSHOPTS[@]}" "$BOX" "mkdir -p $W/code $W/out $W/logs $W/models $W/hf"
say "disk: $(ssh "${SSHOPTS[@]}" "$BOX" "df -h $W | tail -1")"

say "pushing code"
scp -q "${SCPOPTS[@]}" "$ROOT"/vm/*.py "$BOX:$W/code/" 2>&1 | tail -2
scp -q "${SCPOPTS[@]}" "$ROOT"/vm/bootstrap_h200.sh "$BOX:$W/code/" 2>&1 | tail -2
scp -qr "${SCPOPTS[@]}" "$ROOT"/airllm_ternary "$BOX:$W/code/" 2>&1 | tail -2
say "pushed $(ssh "${SSHOPTS[@]}" "$BOX" "ls $W/code/*.py | wc -l") python files"

say "starting queue in tmux"
ssh "${SSHOPTS[@]}" "$BOX" \
  "tmux new-session -d -s h200 'bash $W/code/bootstrap_h200.sh 2>&1 | tee -a $W/logs/h200_stdout.log'"
sleep 10
say "tmux: $(ssh "${SSHOPTS[@]}" "$BOX" 'tmux ls 2>&1')"
say "=== launched ==="
