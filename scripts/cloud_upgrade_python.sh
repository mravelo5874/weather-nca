#!/usr/bin/env bash
# Rebuild the cloud instance's Python environment on a modern interpreter, in a venv.
#
# Why: Ubuntu 22.04 ships Python 3.10, and google-api-core emits a FutureWarning on EVERY
# import saying Google drops 3.10 support on 2026-10-04. Those warnings landed in the middle
# of every training log and had to be grepped out to read anything. 22.04's own python3.11
# candidate is `3.11.0~rc1` -- a release candidate, which is not what a multi-hour training
# run should sit on -- so this uses deadsnakes for a stable build.
#
# Also moves off `pip install --user` (packages were in ~/.local/lib/python3.10) and into a
# real venv, so the interpreter and its packages can be replaced together next time.
#
# Idempotent: safe to re-run. Override any of the variables from the environment.
#
#   bash scripts/cloud_upgrade_python.sh
#   PYV=3.11 bash scripts/cloud_upgrade_python.sh
set -euo pipefail

PYV="${PYV:-3.12}"
VENV="${VENV:-$HOME/venv}"
TORCH_VER="${TORCH_VER:-2.9.1}"
# cu129 matches the driver on the L4 image. Pinned deliberately: the spectral-norm state-dict
# keys are load-bearing for arch_hash, and they were verified against this torch version.
TORCH_IDX="${TORCH_IDX:-https://download.pytorch.org/whl/cu129}"

cd "$(dirname "$0")/.."
REPO="$(pwd)"

echo "=== [$(date -u +%H:%M:%S)] installing python $PYV ==="
if ! command -v "python$PYV" >/dev/null 2>&1; then
  sudo add-apt-repository -y ppa:deadsnakes/ppa
  sudo apt-get update -qq
  sudo apt-get install -y "python$PYV" "python$PYV-venv" "python$PYV-dev"
fi
"python$PYV" --version

echo "=== [$(date -u +%H:%M:%S)] venv -> $VENV ==="
[ -x "$VENV/bin/python" ] || "python$PYV" -m venv "$VENV"
"$VENV/bin/python" -m pip install --upgrade -q pip setuptools wheel

echo "=== [$(date -u +%H:%M:%S)] torch $TORCH_VER ==="
"$VENV/bin/pip" install --index-url "$TORCH_IDX" "torch==$TORCH_VER"

echo "=== [$(date -u +%H:%M:%S)] project (editable, with dev extras) ==="
"$VENV/bin/pip" install -e "$REPO[dev]"

echo "=== [$(date -u +%H:%M:%S)] verify ==="
"$VENV/bin/python" - <<'PY'
import sys, warnings
import torch
print("python", sys.version.split()[0])
print("torch ", torch.__version__, "| cuda", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no gpu")
# The whole point of the upgrade: importing the GCS stack must be warning-free now.
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    import gcsfs  # noqa: F401
    py_warnings = [str(x.message) for x in w if "Python version" in str(x.message)]
print("python-version warnings on gcsfs import:", len(py_warnings))
for m in py_warnings:
    print("  !", m[:120])
PY

echo "=== [$(date -u +%H:%M:%S)] DONE -- use $VENV/bin/python (run_phase.sh picks it up) ==="
