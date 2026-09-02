#!/usr/bin/env python
"""Did CRPS training actually buy member sharpness?

The claim so far rests on a comparison that is not like-for-like: phase 3a's ENSEMBLE MEMBERS
against phase 2c's SINGLE DETERMINISTIC FORECAST. Now that an initial-condition ensemble is known
to calibrate on either checkpoint, the fair comparison exists -- 3a's members against 2c's
IC-perturbed members, two ensembles built the same way, differing only in whether the model was
trained with a probabilistic objective.

Reports per-band energy relative to ERA5 for members and for the ensemble mean. 1.00 means the
model carries as much energy as reality at that scale; below 1.00 is blur.

    python scripts/diag_member_sharpness.py --checkpoint <path> --mode ic --ic-eps 0.20
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from diag_noise_structure import smooth_noise, to_torch_sparse  # noqa: E402
from wnca.config import load_config  # noqa: E402
from wnca.losses.spectral import band_energy_ratio  # noqa: E402
from wnca.mesh.operators import laplacian_matrix  # noqa: E402
from wnca.mesh.spectral import build_band_filters  # noqa: E402
from wnca.train.checkpoint import load_checkpoint  # noqa: E402
from wnca.train.phases import setup  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="configs/phase3a_probe.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--mode", choices=("film", "ic", "deterministic"), default="ic")
    ap.add_argument("--ic-eps", type=float, default=0.20)
    ap.add_argument("--l-cut", type=int, default=20)
    ap.add_argument("--lead", type=int, default=72)
    ap.add_argument("--n-starts", type=int, default=16)
    ap.add_argument("--members", type=int, default=20)
    ap.add_argument("--label", default="")
    ap.add_argument("--out", default="member_sharpness.json")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    mesh, cache, model, _b, device = setup(cfg, None)
    blob = load_checkpoint(a.checkpoint, model, cfg, map_location=device)
    model.eval()
    bands = build_band_filters(mesh, cfg, device=device, cache_dir=cfg.data.cache_dir)
    arr = cache.split(a.split).array
    nw = a.lead // 6
    M = 1 if a.mode == "deterministic" else a.members

    solar = None
    if cfg.state.solar_forcing:
        from wnca.data.forcing import SolarForcing
        solar = SolarForcing(cache.times(a.split), mesh, device)
    static1 = torch.from_numpy(cache.static).float().to(device).unsqueeze(0)

    if a.mode == "ic":
        Lm = laplacian_matrix(mesh)
        L_t = to_torch_sparse(Lm, device)
        lam_p = Path(str(cfg.data.cache_dir)) / f"lambda_max_sub{cfg.mesh.n_sub}.npy"
        lam_max = float(np.load(lam_p)) if lam_p.exists() else 5302.0
        gen = torch.Generator(device=device)
        gen.manual_seed(0)
        static_M = static1.expand(M, -1, -1).contiguous()
        z_ic = torch.zeros(M, 1, cfg.model.noise_dim, device=device)

    starts = np.unique(np.linspace(1, len(arr) - nw - 2, a.n_starts).astype(int))
    lbl = a.label or f"{Path(a.checkpoint).parent.name} / {a.mode}"
    print(f"{lbl}  | epoch {blob.get('epoch')} | M={M} | {len(starts)} starts | +{a.lead}h")

    mem_acc, mean_acc = [], []
    with torch.no_grad():
        for s0 in starts:
            prev1 = torch.as_tensor(arr[s0 - 1:s0], device=device).float()
            cur1 = torch.as_tensor(arr[s0:s0 + 1], device=device).float()
            truth = torch.as_tensor(arr[s0 + nw:s0 + nw + 1], device=device).float()
            if a.mode == "ic":
                pert = torch.stack([smooth_noise(L_t, lam_max, cur1.shape[1], cur1.shape[2],
                                                 a.l_cut, gen, device) for _ in range(M)])
                cur = cur1 + a.ic_eps * pert
                prev = prev1.expand(M, -1, -1).contiguous()
                fw = (solar.window(torch.full((M,), int(s0), device=device), nw)
                      if solar else None)
                pred = model.rollout_ensemble(model.seed(cur), static_M, nw, prev_phys=prev,
                                              n_members=1, z=z_ic, forcing=fw).transpose(0, 1)
            else:
                z = (torch.zeros(1, M, cfg.model.noise_dim, device=device)
                     if a.mode == "deterministic" else None)
                fw = (solar.window(torch.tensor([int(s0)], device=device), nw)
                      if solar else None)
                pred = model.rollout_ensemble(model.seed(cur1), static1, nw, prev_phys=prev1,
                                              n_members=M, z=z, forcing=fw)
            last = pred[:, :, nw - 1][..., :cfg.c_phys]          # [1, M, N, C]
            members = last.reshape(M, *last.shape[2:])
            mem_acc.append(band_energy_ratio(members, truth.expand_as(members),
                                             bands).cpu().numpy())
            mean_acc.append(band_energy_ratio(last.mean(dim=1), truth, bands).cpu().numpy())

    mem = np.mean(mem_acc, axis=0)
    mn = np.mean(mean_acc, axis=0)
    keys = list(cache.channels) if hasattr(cache, "channels") else \
        [c.key for c in cfg.variables.channels()]
    i = keys.index("geopotential_500")
    print(f"\n  band energy vs ERA5, geopotential_500, +{a.lead}h")
    print(f"  {'band':>6} {'members':>10} {'ens mean':>10}")
    print("  " + "-" * 28)
    for b in range(mem.shape[0]):
        print(f"  {b:>6} {mem[b, i]:>10.3f} {mn[b, i]:>10.3f}")
    fine = float(mem[mem.shape[0] // 2:, i].mean())
    print(f"\n  mean over the finer half, members: {fine:.3f}")
    Path(a.out).write_text(json.dumps({
        "label": lbl, "checkpoint": a.checkpoint, "mode": a.mode, "lead": a.lead,
        "members": [float(x) for x in mem[:, i]], "ens_mean": [float(x) for x in mn[:, i]],
        "fine_half_members": fine}, indent=1), encoding="utf-8")
    print(f"  written to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
