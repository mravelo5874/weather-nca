#!/usr/bin/env bash
# Run a phase on a spot/preemptible instance, surviving preemption and crashes.
#
# Two things can kill a run on a spot instance:
#
#   1. Preemption. GCP sends ACPI soft-off, which the OS delivers as SIGTERM. The training
#      loop catches it, checkpoints to `preempted.pt` and exits cleanly. The instance then
#      goes away, so RESTARTING THE VM IS NOT SOMETHING THIS SCRIPT CAN DO -- see the outer
#      watchdog in docs/cloud-setup.md. What this script does is make sure that when the VM
#      comes back, training resumes instead of starting over.
#   2. A crash inside training (2b' diverged in epoch 3 and exited non-zero). Here a retry is
#      the right response, but only at a lower learning rate -- retrying an identical diverged
#      configuration just diverges again.
#
# Usage:
#   ./scripts/spot_train.sh configs/phase2c_full.yaml [gs://bucket/prefix]
#
# The optional GCS prefix is where checkpoints are mirrored after every attempt, so a
# reclaimed instance does not take the run's only copy with it.

set -uo pipefail

CONFIG="${1:?usage: spot_train.sh <config.yaml> [gs://bucket/prefix]}"
GCS_PREFIX="${2:-}"
RUN_DIR="${RUN_DIR:-runs/spot}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-6}"
LR_DECAY_ON_CRASH="${LR_DECAY_ON_CRASH:-0.5}"

mkdir -p "$RUN_DIR"

# Pull any prior state back down first: on a fresh VM this is what makes resume possible.
if [[ -n "$GCS_PREFIX" ]]; then
  echo "[spot] syncing prior state from $GCS_PREFIX"
  gsutil -m rsync -r "$GCS_PREFIX/runs" "$RUN_DIR" 2>/dev/null || echo "[spot] nothing to restore"
  gsutil -m rsync -r "$GCS_PREFIX/wnca_cache" ./wnca_cache 2>/dev/null || echo "[spot] no cached data"
fi

lr_override=""
attempt=1

while (( attempt <= MAX_ATTEMPTS )); do
  # Resume whenever a checkpoint exists -- covers both preemption and crash restarts.
  resume=""
  if compgen -G "$RUN_DIR/*.pt" > /dev/null; then
    resume="--resume auto"
    echo "[spot] attempt $attempt: resuming from an existing checkpoint"
  else
    echo "[spot] attempt $attempt: starting fresh"
  fi

  # shellcheck disable=SC2086
  wnca train --config "$CONFIG" --out "$RUN_DIR" $resume $lr_override
  status=$?

  if [[ -n "$GCS_PREFIX" ]]; then
    echo "[spot] mirroring results to $GCS_PREFIX"
    gsutil -m rsync -r "$RUN_DIR" "$GCS_PREFIX/runs" || true
  fi

  if (( status == 0 )); then
    echo "[spot] training completed"
    exit 0
  fi

  # A preemption checkpoint means the VM is going away; exiting cleanly lets the outer
  # watchdog recreate it. Retrying here would race the shutdown.
  if [[ -f "$RUN_DIR/preempted.pt" ]]; then
    echo "[spot] preempted -- state saved, exiting for the watchdog to recreate the VM"
    exit 75          # EX_TEMPFAIL: retryable, distinct from a real failure
  fi

  # A genuine crash. Retrying the identical configuration reproduces the divergence, so back
  # the learning rate off -- this is what actually rescued phase 2b'.
  lr=$(python - "$CONFIG" "$LR_DECAY_ON_CRASH" "$attempt" <<'PY'
import sys, yaml, pathlib
cfg = yaml.safe_load(pathlib.Path(sys.argv[1]).read_text()) or {}
base = float((cfg.get("train") or {}).get("lr", 1e-3))
print(f"{base * float(sys.argv[2]) ** int(sys.argv[3]):.3e}")
PY
)
  lr_override="--set train.lr=$lr"
  echo "[spot] exited $status -- retrying at lr=$lr"
  attempt=$(( attempt + 1 ))
  sleep 10
done

echo "[spot] gave up after $MAX_ATTEMPTS attempts"
exit 1
