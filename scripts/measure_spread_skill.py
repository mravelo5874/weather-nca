"""Phase 3a's PRIMARY pre-registered criterion, which `wnca eval` does not compute.

Exit criterion 3 is spread-skill in 0.8-1.25 at 24 h and 72 h, on the TEST split at m_test=50,
reported with day-block bootstrap CIs (docs/phase3a-experiment.md section 3). `wnca eval` reports
RMSE, CRPS, perturbation growth and band energies -- but never spread. The only spread numbers in
this project came from the training-time probe, which reads ONE batch of 4 start times off the
VAL loader; that is why the epoch trace swung 1.04-1.90 with nothing else moving.

Pooling follows the project convention (CLAUDE.md): variances and squared errors are pooled
across start times and the square root is taken LAST. Averaging per-start ratios would be biased.
"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np
import torch

from wnca.config import load_config
from wnca.losses.terms import area_weights
from wnca.train.checkpoint import load_checkpoint
from wnca.train.phases import setup
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diag_noise_structure import (  # noqa: E402
    diffusion_plan, effective_degree, smooth_noise, to_torch_sparse, var_split)
from wnca.mesh.operators import laplacian_matrix  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="configs/phase3a_probe.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-starts", type=int, default=64)
    ap.add_argument("--members", type=int, default=None)
    ap.add_argument("--leads", default="24,72")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--ic-eps", type=float, default=0.0,
                    help="0 uses the model's own FiLM ensemble; >0 perturbs the INITIAL "
                         "CONDITION with a spatially correlated field and holds z = 0")
    ap.add_argument("--l-cut", type=int, default=20)
    ap.add_argument("--out", default="spread_skill.json")
    a = ap.parse_args()

    cfg = load_config(a.config)
    mesh, cache, model, _b, device = setup(cfg, None)
    blob = load_checkpoint(a.checkpoint, model, cfg, map_location=device)
    M = a.members or cfg.ensemble.m_test
    leads = [int(x) for x in a.leads.split(",")]
    nw_max = max(leads) // 6
    model.eval()

    aw = area_weights(mesh["area"], device)
    arr = cache.split(a.split).array
    T = len(arr)
    starts = np.unique(np.linspace(1, T - nw_max - 2, a.n_starts).astype(int))
    print(f"checkpoint {blob.get('epoch')} | split {a.split} | M={M} | {len(starts)} starts "
          f"| leads {leads}")

    solar = None
    if cfg.state.solar_forcing:
        from wnca.data.forcing import SolarForcing
        solar = SolarForcing(cache.times(a.split), mesh, device)
    static = torch.from_numpy(cache.static).float().to(device).unsqueeze(0)

    # IC mode. Members ride in the BATCH dimension so each gets its own perturbed analysis, and
    # z is pinned to zero so nothing enters through FiLM -- the model's own error growth is the
    # only thing separating members.
    ic = a.ic_eps > 0
    if ic:
        Lm = laplacian_matrix(mesh)
        L_t = to_torch_sparse(Lm, device)
        lam_p = Path(str(cfg.data.cache_dir)) / f"lambda_max_sub{cfg.mesh.n_sub}.npy"
        if lam_p.exists():
            lam_max = float(np.load(lam_p))
        else:
            from wnca.mesh.spectral import spectral_radius
            lam_max = float(spectral_radius((-Lm).tocsr()))
        dt_s, k_s = diffusion_plan(lam_max, a.l_cut)
        gen = torch.Generator(device=device)
        gen.manual_seed(0)
        static_M = static.expand(M, -1, -1).contiguous()
        z0 = torch.zeros(M, 1, cfg.model.noise_dim, device=device)
        probe = smooth_noise(L_t, lam_max, len(mesh["v"]), 1, a.l_cut, gen, device)
        print(f"IC perturbation: eps {a.ic_eps} | degree target {a.l_cut}, measured "
              f"{effective_degree(L_t, probe, aw):.1f} | dt {dt_s:.2e}, {k_s} steps | z = 0")
    uni = []

    # Per start, per lead: area-weighted mean ensemble VARIANCE and mean SQUARED ERROR of the
    # ensemble mean, both averaged over channels. Kept unrooted so pooling can root last.
    var_acc = {L: [] for L in leads}
    sq_acc = {L: [] for L in leads}

    with torch.no_grad():
        for n, s0 in enumerate(starts):
            prev = torch.as_tensor(arr[s0 - 1:s0], device=device).float()
            cur = torch.as_tensor(arr[s0:s0 + 1], device=device).float()
            if ic:
                pert = torch.stack([smooth_noise(L_t, lam_max, cur.shape[1], cur.shape[2],
                                                 a.l_cut, gen, device) for _ in range(M)])
                cur_m = cur + a.ic_eps * pert
                prev_m = prev.expand(M, -1, -1).contiguous()
                fw = (solar.window(torch.full((M,), int(s0), device=device), nw_max)
                      if solar else None)
                pred = model.rollout_ensemble(model.seed(cur_m), static_M, nw_max,
                                              prev_phys=prev_m, n_members=1, z=z0,
                                              forcing=fw).transpose(0, 1)
            else:
                fw = (solar.window(torch.tensor([int(s0)], device=device), nw_max)
                      if solar else None)
                pred = model.rollout_ensemble(model.seed(cur), static, nw_max, prev_phys=prev,
                                              n_members=M, forcing=fw)
            if not torch.isfinite(pred).all():
                raise FloatingPointError(f"non-finite rollout at start {s0}")
            for L in leads:
                k = L // 6 - 1
                mem = pred[:, :, k][..., :cfg.c_phys]          # [1, M, N, C]
                truth = torch.as_tensor(arr[s0 + k + 1:s0 + k + 2], device=device).float()
                truth = truth[..., :cfg.c_phys]
                var = mem.var(dim=1, unbiased=True)             # [1, N, C]
                sq = (mem.mean(dim=1) - truth) ** 2             # [1, N, C]
                w = aw.view(1, -1, 1)
                var_acc[L].append(((var * w).sum(dim=(0, 1)) / w.sum()).cpu().numpy())
                sq_acc[L].append(((sq * w).sum(dim=(0, 1)) / w.sum()).cpu().numpy())
                if L == leads[0]:
                    uni.append(var_split(mem[0], aw)[0])
            if (n + 1) % 8 == 0:
                print(f"  {n + 1}/{len(starts)} starts")

    corr = np.sqrt((M + 1) / M)   # finite-ensemble correction, as in losses/crps.py
    rng = np.random.default_rng(0)
    out = {"checkpoint": a.checkpoint, "epoch": blob.get("epoch"), "M": M,
           "n_starts": int(len(starts)), "split": a.split,
           "mode": f"ic_eps={a.ic_eps}" if ic else "film",
           "uniform_fraction": float(np.mean(uni)) if uni else None}

    def ratio(v, s, idx):
        """Pool variance and squared error over the selected starts, root LAST, then ratio."""
        sp = np.sqrt(v[idx].mean(axis=0).mean())
        rm = np.sqrt(s[idx].mean(axis=0).mean())
        return float(corr * sp / rm)

    # Day blocks: consecutive start times 6 h apart, so 4 starts span a day. Resampling whole
    # days rather than start times keeps the within-day correlation intact.
    print()
    for L in leads:
        v = np.stack(var_acc[L]); s = np.stack(sq_acc[L])
        point = ratio(v, s, np.arange(len(v)))
        blocks = [np.arange(i, min(i + 4, len(v))) for i in range(0, len(v), 4)]
        boots = []
        for _ in range(a.boot):
            pick = rng.integers(0, len(blocks), len(blocks))
            boots.append(ratio(v, s, np.concatenate([blocks[i] for i in pick])))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        verdict = "IN BAND" if 0.8 <= point <= 1.25 else (
            "OVER-dispersed" if point > 1.25 else "UNDER-dispersed")
        print(f"  spread-skill @{L:>3}h : {point:.3f}   95% CI [{lo:.3f}, {hi:.3f}]   {verdict}")
        out[f"lead_{L}h"] = {"ratio": point, "ci_low": float(lo), "ci_high": float(hi),
                            "verdict": verdict, "n_blocks": len(blocks)}

    if uni:
        print(f"\n  spatially-uniform share of ensemble variance at {leads[0]} h: "
              f"{100 * float(np.mean(uni)):.1f}%    (the FiLM ensemble measured 79.2%)")
    print("\n  criterion 3 band is 0.8-1.25 at BOTH 24 h and 72 h")
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"  written to {a.out}")


if __name__ == "__main__":
    main()
