#!/usr/bin/env bash
# Health check for a running phase: is it progressing, and are non-finite steps settling or not?
#
# Exists because gcloud ssh --command mangles anything with quotes, redirects or awk from a
# Windows shell. Ship a file.
#
#   ./scripts/check_run.sh ~/p2c.log
set -uo pipefail
LOG="${1:-$HOME/p2c.log}"

echo "=== epochs ==="
grep -E '^epoch' "$LOG" || echo "  (none yet)"

echo
echo "=== non-finite ==="
grad=$(grep -c 'non-finite gradient' "$LOG" || true)
batch=$(grep -c 'skipped .* non-finite batches' "$LOG" || true)
echo "  gradients skipped: $grad     (forward was fine -- backward overflowed)"
echo "  batches skipped:   $batch     (forward itself went non-finite)"

if [[ "$grad" -gt 0 ]]; then
  grep -n 'non-finite gradient' "$LOG" | cut -d: -f1 > /tmp/_nf.txt
  first=$(head -1 /tmp/_nf.txt); last=$(tail -1 /tmp/_nf.txt)
  echo "  first at log line $first, last at $last"
  echo "  distribution across that span (settling = front-loaded):"
  awk -v f="$first" -v l="$last" '
    { b = int(($1 - f) / ((l - f + 1) / 5)); if (b > 4) b = 4; c[b]++ }
    END { for (i = 0; i < 5; i++) printf "    fifth %d: %d\n", i + 1, c[i] + 0 }' /tmp/_nf.txt
fi

echo
echo "=== progress ==="
ps -o etime= -C python3 | head -1 | sed 's/^/  elapsed: /'
nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw --format=csv,noheader | sed 's/^/  gpu: /'
