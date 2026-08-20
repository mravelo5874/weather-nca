#!/usr/bin/env bash
# Run a list of training jobs back-to-back on one instance, then stop the instance.
#
# Exists because `autostop.sh` shuts the machine down when a run ends, which is right for a
# single run and wrong for a queue: with GPU quota at 1 the phase-2d arms have to share one
# instance, and a shutdown between them would need a manual restart (into a possible capacity
# STOCKOUT) for every arm.
#
# Waits for any training already in flight before starting the queue, so it can be armed
# against a run that is already going -- which is the normal case here.
#
#   JOBS="configs/phase2d_control.yaml:0 configs/phase2d_control.yaml:1" \
#     nohup bash scripts/run_queue.sh > ~/queue.log 2>&1 < /dev/null &
#
# Each job is `<config>:<seed>`. Cancel with: pkill -f run_queue.sh
set -uo pipefail          # deliberately NOT -e: one failing job must not skip the shutdown

JOBS="${JOBS:?set JOBS='<config>:<seed> <config>:<seed> ...'}"
EPOCHS="${EPOCHS:-8}"
GRACE="${GRACE:-600}"
STOP_WHEN_DONE="${STOP_WHEN_DONE:-1}"
PY="${PY:-$HOME/venv/bin/python}"
PATTERN='wnca[.]cli train'

cd "$(dirname "$0")/.."
log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

if pgrep -f "$PATTERN" >/dev/null; then
  log "a training run is already in flight (pid $(pgrep -f "$PATTERN" | head -1)); waiting"
  while pgrep -f "$PATTERN" >/dev/null; do sleep 60; done
  log "in-flight run finished"
fi

i=0
for job in $JOBS; do
  i=$((i + 1))
  cfg="${job%%:*}"
  seed="${job##*:}"
  name="$(basename "$cfg" .yaml)_seed${seed}"
  out="runs/${name}"
  jlog="$HOME/${name}.log"

  log "=== job $i: $cfg seed=$seed epochs=$EPOCHS -> $out"
  "$PY" -u -m wnca.cli train -c "$cfg" --out "$out" \
      --set "train.seed=$seed" "train.epochs=$EPOCHS" > "$jlog" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    log "job $i OK: $(tail -2 "$jlog" | head -1)"
  else
    # Keep going. A diverged or crashed arm should not cost the remaining arms their
    # slot -- the queue is the scarce resource, not the job.
    log "job $i FAILED rc=$rc -- continuing to the next job. Last lines:"
    tail -5 "$jlog" | sed 's/^/    /'
  fi
done

log "queue drained ($i job(s))"
if [ "$STOP_WHEN_DONE" = "1" ]; then
  log "stopping the instance in ${GRACE}s -- cancel with: pkill -f run_queue.sh"
  sleep "$GRACE"
  log "shutting down (disk and all artifacts preserved)"
  sudo shutdown -h now
else
  log "STOP_WHEN_DONE=0 -- leaving the instance up"
fi
