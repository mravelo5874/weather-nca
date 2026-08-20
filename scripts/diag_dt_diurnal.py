#!/usr/bin/env python
"""Two cheap pre-2d diagnostics. Neither needs a retrain -- both re-score an existing
checkpoint -- and either one can change how phase 2d should be designed.

1. **dt-invariance.** `dt` and `n_substeps` are NOT in `arch_hash`, so the same weights can be
   integrated with a different sub-step budget. Train at dt=0.05 x 20, evaluate at
   dt=0.025 x 40 (and dt=0.1 x 10). If the learned rule is a genuine differential operator,
   refining the integration should work out of the box -- that is the NCA claim. If skill
   collapses, the model has learned a fixed per-window MAP and "local PDE rule" is the wrong
   description of it; it is a 20-layer recurrent residual net. That distinction is what 2d is
   supposed to be testing, so it is worth settling first.

   Note dt x n_substeps = 1.0 is held constant in both variants: the total integrated "time"
   per window is identical and only the discretisation changes. That is the whole point.

2. **Diurnal alignment of the persistence baseline.** Every lead in the standard scorecard
   (24/72/120 h) is a multiple of 24, where persistence compares the SAME local solar time and
   therefore reproduces the diurnal cycle for free. For a diurnally-dominated field like 2 m
   temperature that makes persistence an unusually strong baseline at exactly those leads.
   phase 2c scores 2t at -31% skill at 24 h, which reads as a broken channel.

   The Scorecard already stores every 6 h window, so the test is free: print skill at ALL
   leads. If the deficit largely vanishes at 18 h and 30 h (off the 24 h grid) and returns at
   24/48/72, the channel is not broken -- the baseline is aligned. If it persists everywhere,
   it is a real model failure.

Usage:
    python scripts/diag_dt_diurnal.py -c configs/phase2c_full.yaml \\
        --checkpoint runs/<run>/best_*.pt --split test
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wnca.config import HOURS_PER_WINDOW, load_config  # noqa: E402
from wnca.eval.metrics import _scaled, evaluate  # noqa: E402
from wnca.models.nca import build_model  # noqa: E402
from wnca.train.checkpoint import latest_checkpoint, load_checkpoint  # noqa: E402
from wnca.train.phases import setup  # noqa: E402

OK, WARN, FAIL = "[  ok  ]", "[ warn ]", "[ FAIL ]"


def _hdr(t: str) -> None:
    print(f"\n{'=' * 74}\n== {t}\n{'=' * 74}")


def _ch(sc, name: str) -> int:
    keys = list(sc.channels)
    if name in keys:
        return keys.index(name)
    hits = [i for i, k in enumerate(keys) if name in k]
    if not hits:
        raise SystemExit(f"channel {name!r} not in {keys[:6]}...")
    return hits[0]


def stage_dt(cfg, ckpt, mesh, cache, device, split, n_starts):
    """Same weights, three sub-step discretisations, dt * n_substeps held at 1.0."""
    _hdr("1. dt-INVARIANCE  (is the learned rule a differential operator, or a map?)")
    base_dt, base_n = cfg.model.dt, cfg.model.n_substeps
    variants = [(base_dt, base_n, "trained"),
                (base_dt / 2, base_n * 2, "refined 2x"),
                (base_dt * 2, max(base_n // 2, 1), "coarsened 2x")]

    rows = {}
    for dt, n, tag in variants:
        c = dataclasses.replace(
            cfg, model=dataclasses.replace(cfg.model, dt=dt, n_substeps=n))
        m = build_model(c, mesh, device)
        load_checkpoint(ckpt, m, c, map_location=device)
        sc = evaluate(m, c, cache, mesh, split=split, device=device, n_starts=n_starts)
        z = _scaled(sc, "model", cache.normalizer)[:, _ch(sc, "geopotential_500")]
        rows[tag] = (dt, n, z, sc.leads())
        del m

    leads = rows["trained"][3]
    show = [h for h in (24, 72, 120, 240) if h // HOURS_PER_WINDOW - 1 < len(leads)]
    print(f"\n  z500 RMSE (m2/s2), split={split}\n")
    print(f"  {'variant':>14} {'dt':>7} {'n_sub':>6} " + " ".join(f"{h:>9}h" for h in show))
    print("  " + "-" * (30 + 11 * len(show)))
    for tag, (dt, n, z, _) in rows.items():
        print(f"  {tag:>14} {dt:>7.3f} {n:>6} "
              + " ".join(f"{z[h // HOURS_PER_WINDOW - 1]:>10.1f}" for h in show))

    base = rows["trained"][2]
    print()
    for tag in ("refined 2x", "coarsened 2x"):
        z = rows[tag][2]
        d = [100 * (z[h // HOURS_PER_WINDOW - 1] / base[h // HOURS_PER_WINDOW - 1] - 1)
             for h in show]
        worst = max(abs(x) for x in d)
        verdict = OK if worst < 5 else (WARN if worst < 25 else FAIL)
        print(f"  {verdict} {tag:>13}: " + "  ".join(f"{h}h {x:+.1f}%" for h, x in zip(show, d)))

    worst_ref = max(abs(100 * (rows['refined 2x'][2][h // HOURS_PER_WINDOW - 1]
                               / base[h // HOURS_PER_WINDOW - 1] - 1)) for h in show)
    print()
    if worst_ref < 5:
        print("  -> DIFFERENTIAL OPERATOR: refining the integration is ~free. The 'local PDE")
        print("     rule' framing holds, and sub-step count is a genuine accuracy/cost dial.")
    elif worst_ref < 25:
        print("  -> PARTIAL: the rule transfers but is not scale-free. Report the caveat; the")
        print("     model is somewhere between a discretised operator and a learned map.")
    else:
        print("  -> LEARNED MAP: the weights encode a fixed per-window update, not a")
        print("     time-continuous operator. 'NCA learns a local PDE rule' overstates it --")
        print("     it is a 20-layer recurrent residual net. Reframe 2d's claim accordingly.")
    return {t: {"dt": v[0], "n_substeps": v[1],
                "z500": {int(h): float(v[2][h // HOURS_PER_WINDOW - 1]) for h in show}}
            for t, v in rows.items()}


def stage_diurnal(sc, normalizer, channels):
    """Skill vs persistence at EVERY lead -- the 24 h grid is the thing under test."""
    _hdr("2. DIURNAL ALIGNMENT  (is 2t broken, or is persistence unusually strong at 24k h?)")
    m = _scaled(sc, "model", normalizer)
    p = _scaled(sc, "persistence", normalizer)
    leads = sc.leads()
    n = min(len(leads), 12)  # first 72 h is where the diurnal signal lives

    out = {}
    for name in channels:
        i = _ch(sc, name)
        skill = [100 * (1 - m[k, i] / p[k, i]) for k in range(n)]
        out[name] = {int(leads[k]): round(skill[k], 1) for k in range(n)}
        print(f"\n  {name}   skill vs persistence (%), * = lead is a multiple of 24 h\n")
        print("  " + " ".join(f"{int(leads[k]):>6}h" for k in range(n)))
        print("  " + " ".join(f"{skill[k]:>6.0f}" + ("*" if leads[k] % 24 == 0 else " ")
                              for k in range(n)))

    t2 = out.get("2m_temperature")
    if t2:
        al = [v for h, v in t2.items() if h % 24 == 0]
        off = [v for h, v in t2.items() if h % 24 != 0]
        if al and off:
            gap = float(np.mean(off) - np.mean(al))
            print(f"\n  mean skill on the 24 h grid : {np.mean(al):+.1f}%")
            print(f"  mean skill off the grid     : {np.mean(off):+.1f}%")
            print(f"  gap                         : {gap:+.1f} points")
            if gap > 15:
                print(f"\n  {OK} METRIC ARTEFACT (mostly). Persistence is diurnally aligned at")
                print("     24 h multiples and reproduces the cycle for free. 2t is far less")
                print("     broken than the standard scorecard implies -- quote off-grid leads")
                print("     too, and do NOT spend 2d effort 'fixing' this.")
            elif gap > 5:
                print(f"\n  {WARN} PARTIAL. Some of the deficit is baseline alignment, some is")
                print("     real. Both effects are present; report the pair.")
            else:
                print(f"\n  {FAIL} REAL MODEL FAILURE. The deficit is not explained by diurnal")
                print("     alignment -- 2t is genuinely mispredicted at every lead.")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", "-c", default="configs/phase2c_full.yaml")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-starts", type=int, default=None,
                    help="limit start times (faster, noisier)")
    ap.add_argument("--stages", default="dt,diurnal")
    ap.add_argument("--out", default="diag_dt_diurnal.json")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    mesh, cache, model, _bands, device = setup(cfg, a.device)
    ckpt = a.checkpoint or latest_checkpoint(cfg.tracking.out_dir)
    if ckpt is None:
        print("no checkpoint found -- pass --checkpoint", file=sys.stderr)
        return 2
    blob = load_checkpoint(ckpt, model, cfg, map_location=device)
    print(f"config {a.config} | split {a.split} | device {device}")
    print(f"checkpoint {Path(ckpt).name}  (epoch {blob.get('epoch')}, "
          f"metric {blob.get('metric')})")

    stages = [s.strip() for s in a.stages.split(",")]
    res = {}
    if "dt" in stages:
        res["dt"] = stage_dt(cfg, ckpt, mesh, cache, device, a.split, a.n_starts)
    if "diurnal" in stages:
        sc = evaluate(model, cfg, cache, mesh, split=a.split, device=device,
                      n_starts=a.n_starts)
        res["diurnal"] = stage_diurnal(
            sc, cache.normalizer, ["2m_temperature", "geopotential_500", "temperature_850"])

    import json
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwritten to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
