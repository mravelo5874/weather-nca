#!/usr/bin/env python
"""Compare two checkpoints on 2 m temperature at every 6 h lead.

**Read this before using the output.** It was written to re-score the milestone-2 claim that
solar forcing is "worth 19% on 2t", on the theory that the number had been distorted by the
diurnal-alignment artefact phase 2d uncovered. **That theory was wrong.** The 19% came from a
WITHIN-MODEL ablation -- one 2b' checkpoint scored with real forcing against forcing zeroed, in
raw RMSE -- so persistence never enters it and the artefact cannot apply. The original ablation
also already spanned off-grid leads (7% at 6 h, 9% at 12 h, 14% at 18 h), so the effect was
never confined to 24 h multiples either.

What this script actually measures is a DIFFERENT comparison: two separately-trained
checkpoints. Run on 2b (no solar) against 2b' (solar), it shows 2b' worse at every lead, which
mostly restates that 2b' is undertrained -- it doubled n_substeps and had its LR halved mid-run
after a divergence.

The experiment that would settle the solar claim is the forcing-zeroed ablation repeated on a
HEALTHY model, i.e. the phase-2c checkpoint. That needs a way to zero the forcing at inference
without changing arch_hash, which this script does not implement.

Kept because a two-checkpoint per-lead comparison on one channel is a useful shape, and because
the correction is worth keeping visible.

    python scripts/diag_2t_solar.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wnca.config import load_config  # noqa: E402
from wnca.eval.metrics import _scaled, evaluate  # noqa: E402
from wnca.train.checkpoint import load_checkpoint  # noqa: E402
from wnca.train.phases import setup  # noqa: E402

CHANNEL = "2m_temperature"


def score(cfg_path: str, ckpt: str, split: str, device, n_starts, max_windows):
    cfg = load_config(cfg_path)
    mesh, cache, model, _bands, dev = setup(cfg, device, verbose=False)
    load_checkpoint(ckpt, model, cfg, map_location=dev)
    sc = evaluate(model, cfg, cache, mesh, split=split, device=dev, n_starts=n_starts,
                  max_windows=max_windows)
    keys = list(sc.channels)
    i = keys.index(CHANNEL)
    return sc.leads(), _scaled(sc, "model", cache.normalizer)[:, i], \
        _scaled(sc, "persistence", cache.normalizer)[:, i]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--off", default="configs/phase2b_multivar.yaml:"
                                     "runs/phase2b_multivar_20260816_231822/"
                                     "best_phase2b_multivar_20260816_231823.pt",
                    help="config:checkpoint for the NO-solar arm")
    ap.add_argument("--on", default="configs/phase2b_prime.yaml:"
                                    "runs/phase2b_prime_20260817_083304/"
                                    "best_phase2b_prime_20260817_121558.pt",
                    help="config:checkpoint for the WITH-solar arm")
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-starts", type=int, default=None)
    ap.add_argument("--max-lead", type=int, default=72,
                    help="also caps the rollout, not just the printout -- 2b' runs 40 "
                         "sub-steps per window, so a full 15-day rollout is needlessly slow "
                         "when the diurnal question lives in the first three days")
    ap.add_argument("--out", default="diag_2t_solar.json")
    a = ap.parse_args(argv)

    c_off, k_off = a.off.rsplit(":", 1)
    c_on, k_on = a.on.rsplit(":", 1)
    print(f"no solar : {c_off}\n           {Path(k_off).name}")
    print(f"solar on : {c_on}\n           {Path(k_on).name}")
    print(f"split    : {a.split}\n")

    mw = max(a.max_lead // 6, 1)
    leads, off, _ = score(c_off, k_off, a.split, a.device, a.n_starts, mw)
    _, on, per = score(c_on, k_on, a.split, a.device, a.n_starts, mw)

    n = min(len(leads), a.max_lead // 6)
    print(f"{CHANNEL} RMSE (K). '*' = lead is a multiple of 24 h, where persistence is")
    print("diurnally aligned and the original 19% was measured.\n")
    print(f"  {'lead':>6} {'no solar':>10} {'solar on':>10} {'solar gain':>12}")
    print("  " + "-" * 42)
    gains_on, gains_off = [], []
    rows = []
    for k in range(n):
        h = int(leads[k])
        g = 100 * (1 - on[k] / off[k])          # positive = solar forcing helps
        star = "*" if h % 24 == 0 else " "
        (gains_on if h % 24 == 0 else gains_off).append(g)
        rows.append({"lead": h, "no_solar": float(off[k]), "solar": float(on[k]),
                     "gain_pct": float(g), "on_grid": h % 24 == 0})
        print(f"  {h:>5}h{star}{off[k]:>10.3f} {on[k]:>10.3f} {g:>11.1f}%")

    m_on = float(np.mean(gains_on)) if gains_on else float("nan")
    m_off = float(np.mean(gains_off)) if gains_off else float("nan")
    print(f"\n  mean gain ON the 24 h grid  : {m_on:+.1f}%   <- where 19% was measured")
    print(f"  mean gain OFF the grid      : {m_off:+.1f}%")
    print(f"  difference                  : {m_on - m_off:+.1f} points")

    print()
    if m_on > 5 and m_off < m_on / 2:
        print("  -> LARGELY AN ARTEFACT of the sampling grid. The advantage is much smaller")
        print("     off the 24 h multiples, so the headline number was inflated by scoring")
        print("     2 m temperature exactly where its baseline is diurnally aligned.")
    elif m_off >= m_on / 2 and m_off > 0:
        print("  -> SURVIVES off the grid. The measurement was not distorted; the claim's")
        print("     remaining problem is its OTHER confounds (2b' also doubled n_substeps")
        print("     and was undertrained), not where it was sampled.")
    else:
        print("  -> NO ADVANTAGE either way at this split. The claim does not reproduce.")

    Path(a.out).write_text(json.dumps(
        {"channel": CHANNEL, "split": a.split, "no_solar": a.off, "solar": a.on,
         "rows": rows, "mean_gain_on_grid": m_on, "mean_gain_off_grid": m_off},
        indent=1), encoding="utf-8")
    print(f"\nwritten to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
