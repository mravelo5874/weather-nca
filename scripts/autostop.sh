#!/usr/bin/env bash
# Stop the instance once training exits, so an unattended run cannot idle-bill.
#
# Why this exists: the training process is detached (nohup), so it survives the laptop being
# closed -- but NOTHING stops the *instance* when training ends. Phase 2c attempt 4 diverged
# and then held the GPU at 100% for ten hours taking zero optimizer steps, ~$9 of nothing.
# The new guards abort on divergence now, but aborting only ends the PROCESS: the instance
# keeps billing at ~$0.89/h until someone notices. Overnight that is ~$7, and over a weekend
# ~$45. This closes that gap.
#
# A guest-initiated `shutdown -h` puts a GCE VM into TERMINATED state: vCPU and GPU billing
# stop, the persistent disk survives untouched (cache, checkpoints, traces, logs), and
# `gcloud compute instances start` brings it back in about a minute.
#
#   nohup bash scripts/autostop.sh > ~/autostop.log 2>&1 < /dev/null &
#   GRACE=1800 nohup bash scripts/autostop.sh > ~/autostop.log 2>&1 < /dev/null &
#
# To cancel: pkill -f autostop.sh
set -euo pipefail

# Bracketed so this pattern cannot match the watchdog's own process line.
PATTERN='wnca[.]cli train'
GRACE="${GRACE:-600}"     # seconds between training exiting and the shutdown
WAIT_START="${WAIT_START:-600}"   # how long to wait for training to appear at all

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

log "watchdog up; waiting for a process matching /$PATTERN/"
waited=0
until pgrep -f "$PATTERN" >/dev/null; do
  sleep 10
  waited=$((waited + 10))
  if [ "$waited" -ge "$WAIT_START" ]; then
    log "no training process appeared within ${WAIT_START}s -- exiting WITHOUT shutting down."
    log "  (refusing to stop an instance whose job never started: that would look like a"
    log "   crash and destroy an interactive session someone may be using)"
    exit 0
  fi
done

log "training running (pid $(pgrep -f "$PATTERN" | head -1)); watching until it exits"
while pgrep -f "$PATTERN" >/dev/null; do
  sleep 60
done

log "training exited. Shutting down in ${GRACE}s -- cancel with: pkill -f autostop.sh"
sleep "$GRACE"
log "stopping instance now (disk and all artifacts are preserved)"
sudo shutdown -h now
