# Wait for the master queue to finish, THEN build the artifact. Waiting on the
# completion marker rather than GPU idleness avoids grabbing the cards during a
# gap between master's own steps.
L=/ephemeral/work/logs
echo "[$(date +%H:%M)] waiting for master queue to finish..."
while ! grep -q "MASTER QUEUE DONE" $L/master.log 2>/dev/null; do
  sleep 120
done
echo "[$(date +%H:%M)] master done -- starting integration"
bash /ephemeral/work/code/integrate.sh 2>&1 | tee $L/integrate.log
