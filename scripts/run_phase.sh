#!/usr/bin/env bash
# Build the cache then train a phase, in one detached run.
#
# Exists because `gcloud compute ssh --command` mangles nested quoting when invoked from
# PowerShell or cmd -- ship a file, do not inline shell.
#
#   ./scripts/run_phase.sh configs/phase2c_full.yaml
set -euo pipefail
CONFIG="${1:?usage: run_phase.sh <config.yaml>}"
cd "$(dirname "$0")/.." || exit 1
export PATH="$HOME/.local/bin:$PATH"

echo "=== [$(date -u +%H:%M:%S)] CACHE BUILD: $CONFIG ==="
python3 -m wnca.cli cache --config "$CONFIG"

echo "=== [$(date -u +%H:%M:%S)] TRAINING: $CONFIG ==="
python3 -m wnca.cli train --config "$CONFIG"

echo "=== [$(date -u +%H:%M:%S)] DONE ==="
