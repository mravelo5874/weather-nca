#!/usr/bin/env python
"""Two evaluation-only measurements on the phase-2c checkpoint. No training, ~$1 of instance
time, and between them they close a live milestone-2 claim and price phase 3b.

**Stage `solar`** re-runs the forcing-zeroed ablation on a HEALTHY model.

  The claim "solar forcing is worth 19% on 2 m temperature" comes from that ablation run on
  phase 2b' -- a model that also doubled n_substeps and had its learning rate halved mid-run
  after a divergence, i.e. undertrained and confounded. `milestone-2-findings.md` has flagged it
  as "no support from a healthy model" ever since. 2c has solar forcing on and is healthy, so
  the same ablation can simply be repeated.

  Note this is NOT the diurnal-artefact question. That artefact is about *persistence* being a
  strong baseline at 24 h multiples; this ablation compares a model against itself with an input
  switched off, so persistence never enters. See `phase2d-results.md` section 10, which records
  getting that distinction wrong the first time.

  Zeroing is done by patching the forcing tensor at source rather than by changing config:
  `state.solar_forcing` is in `arch_hash`, so turning it off would refuse to load the checkpoint.
  The model still receives three forcing channels; they are just all zero.

**Stage `spectrum`** measures per-band energy of the deterministic 2c rollout against ERA5.

  Exit criterion 4 is about individual ENSEMBLE MEMBERS, which do not exist yet. But if the
  deterministic model is already badly over-smoothed at 72 h, that predicts 3b will matter and
  says what a CRPS fine-tune has to fix; if it is already sharp, 3b's expected value drops. The
  point is to price 3b before spending on 3a's scale-up, not to pre-empt criterion 4.

    python scripts/diag_2c_closeout.py -c configs/phase2c_full.yaml --checkpoint <path>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from wnca.config import load_config  # noqa: E402
from wnca.eval.metrics import _scaled, evaluate  # noqa: E402
from wnca.mesh.spectral import build_band_filters  # noqa: E402
from wnca.losses.spectral import band_energy_ratio  # noqa: E402
from wnca.train.checkpoint import load_checkpoint  # noqa: E402
from wnca.train.phases import setup  # noqa: E402

OK, WARN, FAIL = "[  ok  ]", "[ warn ]", "[ FAIL ]"


def _hdr(t):
    print(f"\n{'=' * 76}\n== {t}\n{'=' * 76}")


def stage_solar(cfg, mesh, cache, model, device, split, n_starts, max_lead):
    """Same weights, same batches; only the solar forcing channels differ."""
    _hdr("SOLAR FORCING ABLATION on a healthy model (phase 2c)")
    import wnca.eval.metrics as M
    from wnca.data.forcing import SolarForcing

    mw = max(max_lead // 6, 1)
    real = evaluate(model, cfg, cache, mesh, split=split, device=device,
                    n_starts=n_starts, max_windows=mw)

    # Zero the forcing at source. Shapes and channel count are untouched, so arch_hash and the
    # checkpoint load are unaffected -- the model simply sees zeros where cos(zenith) was.
    orig = SolarForcing.window

    def zeroed(self, start_idx, n_windows):
        return torch.zeros_like(orig(self, start_idx, n_windows))

    SolarForcing.window = zeroed
    try:
        off = evaluate(model, cfg, cache, mesh, split=split, device=device,
                       n_starts=n_starts, max_windows=mw)
    finally:
        SolarForcing.window = orig

    keys = list(real.channels)
    out = {}
    for ch in ("2m_temperature", "geopotential_500"):
        i = keys.index(ch)
        r = _scaled(real, "model", cache.normalizer)[:, i]
        z = _scaled(off, "model", cache.normalizer)[:, i]
        leads = real.leads()
        n = min(len(leads), mw)
        print(f"\n  {ch}   RMSE with real forcing vs forcing zeroed")
        print(f"  {'lead':>6} {'real':>10} {'zeroed':>10} {'gain':>10}")
        print("  " + "-" * 40)
        gains = []
        for k in range(n):
            g = 100 * (1 - r[k] / z[k])
            gains.append(float(g))
            print(f"  {int(leads[k]):>5}h {r[k]:>10.3f} {z[k]:>10.3f} {g:>9.1f}%")
        out[ch] = {"leads": [int(x) for x in leads[:n]],
                   "real": [float(x) for x in r[:n]],
                   "zeroed": [float(x) for x in z[:n]], "gain_pct": gains}

    g24 = next((g for lead, g in zip(out["2m_temperature"]["leads"],
                                     out["2m_temperature"]["gain_pct"]) if lead == 24), None)
    print()
    if g24 is None:
        print("  (24 h not in the scored range)")
    elif g24 > 10:
        print(f"  {OK} The 2b' claim REPRODUCES on a healthy model: {g24:+.1f}% on 2t at 24 h")
        print("     against the original 18.6%. The mechanism is real and the magnitude is now")
        print("     established on a model that is not undertrained.")
    elif g24 > 2:
        print(f"  {WARN} PARTIALLY reproduces: {g24:+.1f}% on 2t at 24 h against 18.6%. Solar")
        print("     forcing helps, but the headline number was inflated by 2b's undertraining.")
    else:
        print(f"  {FAIL} DOES NOT reproduce: {g24:+.1f}% on 2t at 24 h against a claimed 18.6%.")
        print("     The claim rested on the undertrained 2b' run and should be retired.")
    return out


def stage_spectrum(cfg, mesh, cache, model, device, split, lead, n_starts):
    """Per-band energy of the deterministic rollout relative to ERA5 at one lead."""
    _hdr(f"SPECTRAL BAND ENERGY at {lead} h -- deterministic 2c rollout vs ERA5")
    bands = build_band_filters(mesh, cfg, device=device, cache_dir=cfg.data.cache_dir)
    arr = cache.split(split).array
    n_win = lead // 6
    rng = np.random.default_rng(0)
    starts = rng.choice(len(arr) - n_win - 1, size=min(n_starts or 24, 24), replace=False)

    sf = None
    if cfg.state.solar_forcing:
        from wnca.data.forcing import SolarForcing
        sf = SolarForcing(cache.times(split), mesh, device)

    ratios = []
    with torch.no_grad():
        for s0 in starts:
            phys = torch.as_tensor(arr[s0:s0 + 1], device=device).float()
            truth = torch.as_tensor(arr[s0 + n_win:s0 + n_win + 1], device=device).float()
            static = model.static.unsqueeze(0) if hasattr(model, "static") else \
                torch.zeros(1, len(mesh["v"]), cfg.state.c_static, device=device)
            fw = sf.window(torch.tensor([int(s0)], device=device), n_win) if sf else None
            pred, _ = model.rollout_ensemble(model.seed(phys), static, n_win,
                                             n_members=1, return_aux=True, forcing=fw)
            last = pred[:, 0, -1][..., :cfg.c_phys]
            ratios.append(band_energy_ratio(last, truth, bands).cpu().numpy())
    r = np.stack(ratios).mean(0)          # [n_bands, C]

    keys = list(cache.channels) if hasattr(cache, "channels") else \
        [c.key for c in cfg.variables.channels()]
    show = [k for k in ("geopotential_500", "temperature_850", "2m_temperature",
                        "specific_humidity_500") if k in keys]
    print(f"\n  energy ratio, model / ERA5. 1.00 = matched, <1 = over-smoothed.")
    print(f"  band 0 = largest scales, band {r.shape[0] - 1} = smallest.\n")
    print("  " + f"{'channel':>22}" + "".join(f"{'band ' + str(b):>10}"
                                              for b in range(r.shape[0])))
    print("  " + "-" * (22 + 10 * r.shape[0]))
    out = {}
    for k in show:
        i = keys.index(k)
        out[k] = [float(x) for x in r[:, i]]
        print("  " + f"{k:>22}" + "".join(f"{x:>10.3f}" for x in r[:, i]))

    # Judge the PROFILE, not one band. A first version keyed on the finest band alone and
    # called a model "sharp" at 1.26 while bands 2-3 sat at 0.18-0.45 -- the finest band can run
    # hot from small-scale noise while the mid-scales are smoothed away, which is the opposite
    # of what that verdict implied.
    prof = np.array([out[k] for k in show])          # [channels, bands]
    per_band = prof.mean(0)
    worst_b = int(np.argmin(per_band))
    worst = float(per_band[worst_b])
    finer_half = float(per_band[len(per_band) // 2:].mean())
    print()
    print("  mean over channels, by band: "
          + "  ".join(f"b{i}={v:.2f}" for i, v in enumerate(per_band)))
    print(f"  worst band: b{worst_b} at {worst:.2f}   |   mean over the finer half: "
          f"{finer_half:.2f}")
    print()
    if worst < 0.5:
        print(f"  {WARN} HEAVILY over-smoothed: band {worst_b} retains only {worst:.2f} of ERA5's")
        print("     energy, and criterion 4 (members within 20% of ERA5) starts a long way off.")
        print("     Do NOT read this as raising 3b's price: an MSE-trained model converges to the")
        print("     smooth conditional mean, so this is the objective behaving as designed. What")
        print("     it gives you is the baseline 3a's MEMBERS have to beat -- CRPS penalises")
        print("     over-smooth members directly, so it should move these ratios with w_spec=0.")
    elif worst < 0.8:
        print(f"  {WARN} Over-smoothed at band {worst_b} ({worst:.2f}). 3b would help; whether it")
        print("     can reach criterion 4's 20% band is open.")
    else:
        print(f"  {OK} Every band within 20% of ERA5 (worst {worst:.2f} at band {worst_b}).")
        print("     3b's expected value is LOW -- the deterministic model is not what limits")
        print("     member sharpness.")
    if per_band[-1] > 1.2 and worst < 0.8:
        print()
        print(f"  {WARN} Note the shape: the finest band runs HOT ({per_band[-1]:.2f}) while")
        print(f"     mid-scales are smoothed ({worst:.2f}). That is small-scale noise sitting on")
        print("     top of a damped spectrum, not sharpness -- a spectral loss would need to")
        print("     suppress the former as well as restore the latter.")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="configs/phase2c_full.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default=None)
    ap.add_argument("--n-starts", type=int, default=40)
    ap.add_argument("--max-lead", type=int, default=72)
    ap.add_argument("--stages", default="solar,spectrum")
    ap.add_argument("--out", default="diag_2c_closeout.json")
    ap.add_argument("--smoke", action="store_true", help="tiny mesh + synthetic data")
    ap.add_argument("--set", nargs="*", dest="sets", default=[], metavar="KEY=VAL")
    a = ap.parse_args(argv)

    overrides = {}
    for pair in a.sets:
        key, val = pair.split("=", 1)
        node = overrides
        parts = key.split(".")
        for k in parts[:-1]:
            node = node.setdefault(k, {})
        try:
            node[parts[-1]] = json.loads(val)
        except json.JSONDecodeError:
            node[parts[-1]] = val
    cfg = load_config(a.config, overrides, smoke=a.smoke)
    mesh, cache, model, _bands, device = setup(cfg, a.device)
    blob = load_checkpoint(a.checkpoint, model, cfg, map_location=device)
    print(f"config {a.config} | split {a.split} | device {device}")
    print(f"checkpoint {Path(a.checkpoint).name} (epoch {blob.get('epoch')})")

    res = {}
    stages = [s.strip() for s in a.stages.split(",")]
    if "solar" in stages:
        res["solar"] = stage_solar(cfg, mesh, cache, model, device, a.split,
                                   a.n_starts, a.max_lead)
    if "spectrum" in stages:
        res["spectrum"] = stage_spectrum(cfg, mesh, cache, model, device, a.split,
                                         72, a.n_starts)
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\nwritten to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
