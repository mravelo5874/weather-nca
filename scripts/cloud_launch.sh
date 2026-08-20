#!/usr/bin/env bash
# Clone the prepared disk snapshot into a fresh GPU instance and start one training run on it.
#
# Exists so that adding a parallel arm is one command instead of a six-step ritual. The
# snapshot (`wnca-base-snap`, taken from wnca-2c) already carries the 64 GB ERA5 cache, the
# Python 3.12 venv and the repo, so a clone is ready to train about three minutes after
# creation -- no cache rebuild, no environment setup.
#
#   bash scripts/cloud_launch.sh wnca-2d-s0 configs/phase2d_control.yaml 0
#   bash scripts/cloud_launch.sh wnca-2d-s1 configs/phase2d_control.yaml 1
#
# REQUIRES GPU QUOTA. With GPUS_ALL_REGIONS at 1 this fails at the create step with
# "limit = 1.0" -- that is a quota error, not a capacity error. Check with:
#   gcloud compute project-info describe --format=json | grep -A2 GPUS_ALL_REGIONS
# Capacity STOCKOUT is a different failure and is transient; just retry.
set -euo pipefail

NAME="${1:?usage: cloud_launch.sh <instance-name> <config> <seed> [epochs]}"
CONFIG="${2:?}"
SEED="${3:?}"
EPOCHS="${4:-8}"

ZONE="${ZONE:-us-east1-c}"
SNAP="${SNAP:-wnca-base-snap}"
BRANCH="${BRANCH:-phase2c-stability-fixes}"
REPO="/home/mrave/weather-nca"
PY="/home/mrave/venv/bin/python"
LOG="/home/mrave/${NAME}.log"

say() { echo "[$(date -u +%H:%M:%S)] $*"; }
ssh_() { gcloud compute ssh "$NAME" --zone="$ZONE" --quiet \
           --strict-host-key-checking=no --command="$1"; }

say "creating $NAME from $SNAP"
gcloud compute instances create "$NAME" \
  --zone="$ZONE" \
  --machine-type=g2-standard-8 \
  --maintenance-policy=TERMINATE \
  --create-disk="boot=yes,source-snapshot=$SNAP,size=300,type=pd-balanced,auto-delete=yes" \
  --quiet

say "waiting for sshd"
for i in $(seq 1 40); do
  if ssh_ "true" >/dev/null 2>&1; then say "ssh up after ${i} tries"; break; fi
  sleep 15
  [ "$i" -eq 40 ] && { say "ssh never came up"; exit 1; }
done

say "syncing repo to $BRANCH"
# The snapshot was taken from a machine mid-run, so its working tree can carry a stray
# deletion; check out cleanly rather than assuming a fast-forward restores files.
ssh_ "cd $REPO && git fetch --quiet origin && git checkout -- . && \
      git checkout $BRANCH 2>/dev/null || git checkout -B $BRANCH origin/$BRANCH; \
      git reset --hard origin/$BRANCH --quiet && git log --oneline -1"

say "launching: $CONFIG seed=$SEED epochs=$EPOCHS"
# --out is explicit: the cloned disk carries phase-2c run directories, and a later
# `--resume auto` / eval that globbed best_*.pt could otherwise pick up an NCA checkpoint.
# It would fail loudly on the arch_hash assert, but keeping the runs apart is cleaner.
OUT="runs/${NAME}"
ssh_ "cd $REPO && nohup $PY -u -m wnca.cli train -c $CONFIG --out $OUT \
        --set train.seed=$SEED train.epochs=$EPOCHS > $LOG 2>&1 < /dev/null & sleep 20; \
      nohup bash scripts/autostop.sh > /home/mrave/autostop.log 2>&1 < /dev/null & sleep 60; \
      grep -vE 'Warning|warnings.warn' $LOG | tail -6; \
      echo '--- watchdog ---'; tail -1 /home/mrave/autostop.log; true"

say "done. watch with:"
echo "  gcloud compute ssh $NAME --zone=$ZONE --strict-host-key-checking=no \\"
echo "    --command \"cat $REPO/$OUT/metrics.jsonl; true\""
echo
say "autostop is armed: the instance stops itself when training exits."
echo "  delete when finished:  gcloud compute instances delete $NAME --zone=$ZONE --quiet"
