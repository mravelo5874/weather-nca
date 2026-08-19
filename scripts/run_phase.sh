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
# Prefer the venv built by scripts/cloud_upgrade_python.sh (modern Python, no
# google-api-core version warnings). Falls back to system python3 if absent.
PY="${PY:-$( [ -x "$HOME/venv/bin/python" ] && echo "$HOME/venv/bin/python" || echo python3 )}"
echo "interpreter: $($PY --version 2>&1) ($PY)"

echo "=== [$(date -u +%H:%M:%S)] CACHE BUILD: $CONFIG ==="
"$PY" -u -m wnca.cli cache --config "$CONFIG"

echo "=== [$(date -u +%H:%M:%S)] TRAINING: $CONFIG ==="
"$PY" -u -m wnca.cli train --config "$CONFIG"

echo "=== [$(date -u +%H:%M:%S)] DONE ==="
