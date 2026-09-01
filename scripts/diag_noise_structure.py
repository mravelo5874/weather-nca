#!/usr/bin/env python
"""Two experiments on phase 3a's noise design, prompted by an observation on the globe:
member 1 sat at one extreme of every field while member 2 tracked ERA5.

**Stage `uniformity`** asks what that is.

  Decision 0001 draws ONE `z` per member and holds it constant across all 10,242 cells and all
  sub-steps, injected through FiLM. A constant `z` modulates the update rule identically
  everywhere, so it can only bias the tendency uniformly -- and 800 sub-steps of a uniform
  tendency bias integrate into a uniform drift.

  Each member's deviation from the ensemble mean is split into a global offset and a spatial
  pattern, and the ensemble variance attributed between them. A high uniform share means the
  ensemble samples a global "run warmer / run colder" knob rather than alternative weather.

**Stage `icpert`** tests the cheap alternative.

  Perturb the INITIAL CONDITION with a spatially correlated field and run the noise pathway at
  zero. The model's own error growth (x1.142 per step) does the diversifying, which is how
  operational centres actually build ensembles. Needs no retraining -- inference only. The
  perturbation size is swept, so the answer is a curve rather than one tuned number.

    python scripts/diag_noise_structure.py -c configs/phase3a_probe.yaml --checkpoint <path>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import torch  # noqa: E402

from wnca.config import load_config  # noqa: E402
from wnca.losses.terms import area_weights  # noqa: E402
from wnca.mesh.operators import laplacian_matrix  # noqa: E402
from wnca.train.checkpoint import load_checkpoint  # noqa: E402
from wnca.train.phases import setup  # noqa: E402


def to_torch_sparse(A: sp.csr_matrix, device):
    A = A.tocoo()
    idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long, device=device)
    val = torch.tensor(A.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(idx, val, A.shape).coalesce()


def smooth_noise(L, n, c, k, gen, device):
    """Spatially correlated noise: white noise relaxed by `k` explicit diffusion steps.

    Diffusion damps high graph frequencies and leaves a smooth field whose correlation length
    grows with k -- the cheapest way to get SPPT-like structure out of the mesh the model already
    carries. Renormalised to unit variance so `eps` alone sets the amplitude.
    """
    x = torch.randn(n, c, generator=gen, device=device)
    for _ in range(k):
        x = x + 0.12 * torch.sparse.mm(L, x)
    return x / (x.std(dim=0, keepdim=True) + 1e-9)


def var_split(members, aw):
    """Split ensemble variance into a spatially-uniform mode and everything else.

    `members` is [M, N, C] in normalised units. Variances are pooled and rooted last, matching
    the RMSE convention, so the two components are comparable.
    """
    e = members - members.mean(dim=0, keepdim=True)      # [M, N, C]
    w = aw.view(1, -1, 1)
    c = (e * w).sum(dim=1, keepdim=True) / w.sum()       # one offset per member per channel
    r = e - c                                            # spatial residual
    vg = float((c[:, 0] ** 2).mean().item())
    vs = float(((r ** 2) * w).sum().item()
               / (w.sum().item() * e.shape[0] * e.shape[2]))
    return vg / (vg + vs + 1e-12), float(np.sqrt(vg)), float(np.sqrt(vs))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="configs/phase3a_probe.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--leads", default="24,72")
    ap.add_argument("--uni-starts", type=int, default=8)
    ap.add_argument("--uni-members", type=int, default=50)
    ap.add_argument("--ic-starts", type=int, default=24)
    ap.add_argument("--ic-members", type=int, default=20)
    ap.add_argument("--eps", default="0.02,0.05,0.10")
    ap.add_argument("--smooth-k", type=int, default=24)
    ap.add_argument("--stages", default="uniformity,icpert")
    ap.add_argument("--out", default="noise_structure.json")
    a = ap.parse_args(argv)

    cfg = load_config(a.config)
    mesh, cache, model, _bands, device = setup(cfg, None)
    blob = load_checkpoint(a.checkpoint, model, cfg, map_location=device)
    model.eval()

    leads = [int(x) for x in a.leads.split(",")]
    nw = max(leads) // 6
    aw = area_weights(mesh["area"], device)
    arr = cache.split(a.split).array
    T = len(arr)
    solar = None
    if cfg.state.solar_forcing:
        from wnca.data.forcing import SolarForcing
        solar = SolarForcing(cache.times(a.split), mesh, device)
    static1 = torch.from_numpy(cache.static).float().to(device).unsqueeze(0)
    stages = [s.strip() for s in a.stages.split(",")]
    print(f"checkpoint epoch {blob.get('epoch')} | split {a.split} | leads {leads}")
    out = {"epoch": blob.get("epoch"), "leads": leads, "checkpoint": a.checkpoint}

    # ------------------------------------------------------------------ stage 1
    if "uniformity" in stages:
        print(f"\n{'=' * 74}")
        print("== HOW MUCH OF THE ENSEMBLE IS ONE GLOBAL OFFSET?")
        print(f"{'=' * 74}")
        starts = np.unique(np.linspace(1, T - nw - 2, a.uni_starts).astype(int))
        M = a.uni_members
        acc = {L: [] for L in leads}
        with torch.no_grad():
            for s0 in starts:
                prev = torch.as_tensor(arr[s0 - 1:s0], device=device).float()
                cur = torch.as_tensor(arr[s0:s0 + 1], device=device).float()
                fw = solar.window(torch.tensor([int(s0)], device=device), nw) if solar else None
                pred = model.rollout_ensemble(model.seed(cur), static1, nw, prev_phys=prev,
                                              n_members=M, forcing=fw)[0]
                for L in leads:
                    acc[L].append(var_split(pred[:, L // 6 - 1][..., :cfg.c_phys], aw))
        print(f"\n  {'lead':>6} {'uniform share':>15} {'global sd':>11} {'spatial sd':>11}")
        print("  " + "-" * 47)
        out["uniformity"] = {}
        for L in leads:
            u = float(np.mean([x[0] for x in acc[L]]))
            g = float(np.mean([x[1] for x in acc[L]]))
            s = float(np.mean([x[2] for x in acc[L]]))
            print(f"  {L:>5}h {100 * u:>14.1f}% {g:>11.4f} {s:>11.4f}")
            out["uniformity"][f"{L}h"] = {"uniform_fraction": u,
                                          "global_sd": g, "spatial_sd": s}
        u0 = out["uniformity"][f"{leads[0]}h"]["uniform_fraction"]
        print()
        if u0 > 0.5:
            print(f"  [CONFIRMED] {100 * u0:.0f}% of ensemble variance at {leads[0]} h is ONE")
            print("     global offset. The ensemble samples a 'run warmer / run colder' knob,")
            print("     not alternative weather. Shrinking noise_std would calibrate the spread")
            print("     while leaving it structureless -- passing the criterion for the wrong")
            print("     reason, which is worse than failing it.")
        elif u0 > 0.2:
            print(f"  [PARTLY] {100 * u0:.0f}% of the variance is a global offset -- a real")
            print("     component, but spatial structure still carries the majority.")
        else:
            print(f"  [NOT CONFIRMED] only {100 * u0:.0f}% is a global offset. The perturbation")
            print("     has spatial structure and the reading off the globe was misleading.")

    # ------------------------------------------------------------------ stage 2
    if "icpert" in stages:
        print(f"\n{'=' * 74}")
        print("== INITIAL-CONDITION PERTURBATION, NOISE PATHWAY OFF")
        print(f"{'=' * 74}")
        L_t = to_torch_sparse(laplacian_matrix(mesh), device)
        gen = torch.Generator(device=device)
        gen.manual_seed(0)
        eps_list = [float(x) for x in a.eps.split(",")]
        Mi = a.ic_members
        ic_starts = np.unique(np.linspace(1, T - nw - 2, a.ic_starts).astype(int))
        static_M = static1.expand(Mi, -1, -1).contiguous()
        z0 = torch.zeros(Mi, 1, cfg.model.noise_dim, device=device)
        corr = float(np.sqrt((Mi + 1) / Mi))
        out["ic_pert"] = {}
        print(f"\n  {Mi} members | {len(ic_starts)} starts | smoothing k={a.smooth_k} | z = 0")
        print(f"  members are carried in the BATCH dim, so each gets its own perturbed state\n")
        head = f"  {'eps':>6} " + " ".join(f"{'ss@' + str(L) + 'h':>10}" for L in leads)
        print(head + f" {'uniform@' + str(leads[0]) + 'h':>16}")
        print("  " + "-" * (len(head) + 12))
        for eps in eps_list:
            vg = {L: [] for L in leads}
            sq = {L: [] for L in leads}
            un = []
            with torch.no_grad():
                for s0 in ic_starts:
                    cur1 = torch.as_tensor(arr[s0:s0 + 1], device=device).float()
                    prev1 = torch.as_tensor(arr[s0 - 1:s0], device=device).float()
                    pert = torch.stack([
                        smooth_noise(L_t, cur1.shape[1], cur1.shape[2], a.smooth_k, gen, device)
                        for _ in range(Mi)])
                    cur = cur1 + eps * pert
                    prev = prev1.expand(Mi, -1, -1).contiguous()
                    fw = (solar.window(torch.full((Mi,), int(s0), device=device), nw)
                          if solar else None)
                    pred = model.rollout_ensemble(model.seed(cur), static_M, nw,
                                                  prev_phys=prev, n_members=1, z=z0,
                                                  forcing=fw)[:, 0]
                    for L in leads:
                        k = L // 6 - 1
                        mem = pred[:, k][..., :cfg.c_phys]
                        truth = torch.as_tensor(
                            arr[s0 + k + 1:s0 + k + 2], device=device).float()[..., :cfg.c_phys]
                        w = aw.view(-1, 1)
                        vg[L].append(float(((mem.var(dim=0, unbiased=True) * w).sum()
                                            / (w.sum() * mem.shape[-1])).item()))
                        sq[L].append(float((((mem.mean(dim=0) - truth[0]) ** 2 * w).sum()
                                            / (w.sum() * mem.shape[-1])).item()))
                        if L == leads[0]:
                            un.append(var_split(mem, aw)[0])
            row = f"  {eps:>6.3f} "
            rec = {}
            for L in leads:
                ss = corr * float(np.sqrt(np.mean(vg[L]))) / float(np.sqrt(np.mean(sq[L])))
                rec[f"ss_{L}h"] = ss
                row += f" {ss:>10.3f}"
            rec["uniform_fraction"] = float(np.mean(un))
            row += f" {100 * rec['uniform_fraction']:>15.1f}%"
            print(row)
            out["ic_pert"][f"eps_{eps}"] = rec

        print("\n  spread-skill target is 0.80 - 1.25 at BOTH leads. A LOW uniform share here is")
        print("  the point: it would mean IC perturbation buys the spatial structure the FiLM")
        print("  pathway does not, whatever amplitude turns out to be needed.")

    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\n  written to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
