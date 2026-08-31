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
sys.path.insert(0, "src")
import numpy as np
import torch

from wnca.config import load_config
from wnca.losses.terms import area_weights
from wnca.train.checkpoint import load_checkpoint
from wnca.train.phases import setup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", "-c", default="configs/phase3a_probe.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--n-starts", type=int, default=64)
    ap.add_argument("--members", type=int, default=None)
    ap.add_argument("--leads", default="24,72")
    ap.add_argument("--boot", type=int, default=2000)
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

    # Per start, per lead: area-weighted mean ensemble VARIANCE and mean SQUARED ERROR of the
    # ensemble mean, both averaged over channels. Kept unrooted so pooling can root last.
    var_acc = {L: [] for L in leads}
    sq_acc = {L: [] for L in leads}

    with torch.no_grad():
        for n, s0 in enumerate(starts):
            prev = torch.as_tensor(arr[s0 - 1:s0], device=device).float()
            cur = torch.as_tensor(arr[s0:s0 + 1], device=device).float()
            fw = solar.window(torch.tensor([int(s0)], device=device), nw_max) if solar else None
            pred = model.rollout_ensemble(model.seed(cur), static, nw_max, prev_phys=prev,
                                          n_members=M, forcing=fw)
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
            if (n + 1) % 8 == 0:
                print(f"  {n + 1}/{len(starts)} starts")

    corr = np.sqrt((M + 1) / M)   # finite-ensemble correction, as in losses/crps.py
    rng = np.random.default_rng(0)
    out = {"checkpoint": a.checkpoint, "epoch": blob.get("epoch"), "M": M,
           "n_starts": int(len(starts)), "split": a.split}

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

    print("\n  criterion 3 band is 0.8-1.25 at BOTH 24 h and 72 h")
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"  written to {a.out}")


if __name__ == "__main__":
    main()
