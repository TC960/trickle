#!/usr/bin/env bash
# One command to pick the session back up:   bash reconnect.sh
#
# Shows what both cloud boxes are doing, backs their results up to this laptop,
# reopens the dashboard tunnel, and prints the report if it exists.
# Safe to run repeatedly; it starts nothing on the GPUs and stops nothing.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOX1=gorgeous-copper-mite
BOX2=military-turquoise-crawdad
SSHOPTS=(-n -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null
         -o LogLevel=ERROR -o ConnectTimeout=20)

say () { printf '\n\033[1m%s\033[0m\n' "$*"; }

say "instances"
brev ls 2>/dev/null | tail -4

for BOX in "$BOX1" "$BOX2"; do
  say "$BOX"
  ssh "${SSHOPTS[@]}" "$BOX" '
    echo -n "  running: "; tmux ls 2>/dev/null | cut -d: -f1 | tr "\n" " "; echo
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu \
      --format=csv,noheader | sed "s/^/  gpu /"
    echo "  latest results:"
    for f in /ephemeral/work/out/*.jsonl; do
      [ -f "$f" ] && printf "    %-26s %s lines\n" "$(basename "$f")" "$(wc -l < "$f")"
    done 2>/dev/null
    for L in /ephemeral/work/logs/queue*.log /ephemeral/work/logs/depth.log \
             /ephemeral/work/logs/box2.log; do
      [ -f "$L" ] && echo "  $(basename "$L"): $(tail -1 "$L" | cut -c1-90)"
    done 2>/dev/null
  ' 2>/dev/null || echo "  (unreachable)"
done

# Back the results up locally -- /ephemeral does not survive an instance stop.
say "backing up results to ./results-backup/"
mkdir -p "$ROOT/results-backup"
for BOX in "$BOX1" "$BOX2"; do
  scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o LogLevel=ERROR "$BOX:/ephemeral/work/out/*.jsonl" \
      "$ROOT/results-backup/" 2>/dev/null
  scp -q -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o LogLevel=ERROR "$BOX:/ephemeral/work/out/*.md" \
      "$ROOT/results-backup/" 2>/dev/null
done
ls -1 "$ROOT/results-backup" 2>/dev/null | sed 's/^/  /' | head -20

say "dashboard"
if curl -s -m 3 localhost:8777/api/status >/dev/null 2>&1; then
  echo "  already up -> http://localhost:8777"
else
  ssh -f -N -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
      -o LogLevel=ERROR -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
      -L 8777:localhost:8777 "$BOX1" 2>/dev/null \
    && echo "  tunnel opened -> http://localhost:8777" \
    || echo "  could not open tunnel (is $BOX1 running?)"
fi

say "report"
ssh "${SSHOPTS[@]}" "$BOX1" 'test -f /ephemeral/work/out/REPORT.md \
  && /ephemeral/work/venv/bin/python /ephemeral/work/code/make_report.py 2>/dev/null \
  || echo "  (report generates when the queue finishes; regenerate any time with:
   ./vm/vmsh \"cd /ephemeral/work/code \&\& /ephemeral/work/venv/bin/python make_report.py\")"' 2>/dev/null

cat <<'TIPS'

  useful:
    ./vm/vmsh "<cmd>"                              run on box 1
    BREV_HOST=military-turquoise-crawdad ./vm/vmsh "<cmd>"   run on box 2
    ./vm/vmsh "tmux ls"                            what's running
    ./vm/vmsh "cd /ephemeral/work/code && /ephemeral/work/venv/bin/python report.py"
                                                   A/B table with provenance
TIPS
