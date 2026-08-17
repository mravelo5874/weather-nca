"""perturbation_growth -- the M1 diagnostic that was always right.

Roll out from two nearly-identical starts and watch the gap. No linearization, no finite
differences: it measures what the model actually does. ~1.00 per window means the operator is
neutrally stable and any remaining error growth is *systematic* (bias, blurring) rather than
amplification. Above ~1.05 the operator itself is the problem.

The standing methodology note, from M1: prefer direct measurement over clever measurement.
`perturbation_growth` was right every time. The one-sided spectrum check and the
finite-difference spectral radius both misfired confidently -- the latter's Jacobian-vector
products sat on the float32 noise floor, so the answer scaled inversely with the step size and
the reported value was noise.

M2 runs this **per variable**: `n_substeps = 20` was measured on one band-limited field, and
coupled dynamics carry faster adjustment processes. A single-window CFL test cannot see
per-window amplification that compounds, which M1 established the hard way.
"""

from __future__ import annotations

import numpy as np
import torch

from ..config import Config


@torch.no_grad()
def perturbation_growth(
    model,
    cfg: Config,
    cache,
    split: str = "test",
    start: int = 5,
    n_windows: int = 12,
    amp: float = 1e-3,
    device: str = "cpu",
    per_channel: bool = True,
    seed: int = 0,
    threshold: float = 1.05,
    mesh: dict | None = None,
) -> dict:
    """Finite-amplitude growth: roll out from x0 and from x0 + delta, track ||difference||.

    Returns per-window norms (total and, with `per_channel`, per physical channel) plus the
    geometric-mean growth rate excluding window 1, which is routinely anomalous.
    """
    model.eval()
    series = cache.split(split).array
    st = torch.from_numpy(cache.static).float().to(device).unsqueeze(0)

    prev = torch.from_numpy(np.array(series[start - 1], dtype=np.float32)).unsqueeze(0).to(device)
    cur = torch.from_numpy(np.array(series[start], dtype=np.float32)).unsqueeze(0).to(device)
    x0 = model.seed(cur)

    g = torch.Generator(device="cpu").manual_seed(seed)
    d = torch.zeros_like(x0)
    src = torch.randn(x0[..., : cfg.c_phys].shape, generator=g).to(device)
    d[..., : cfg.c_phys] = amp * src / src.norm()

    z = None
    if cfg.model.stochastic:
        # One fixed noise vector for BOTH trajectories: this measures the dynamics' sensitivity
        # to the state, not to the member draw.
        z = torch.zeros(1, cfg.model.noise_dim, device=device)

    a, b, pa, pb = x0, x0 + d, prev, prev
    total, per_ch = [], []
    n_windows = min(n_windows, len(series) - start - 1)

    # Both trajectories get the SAME forcing: this measures sensitivity to the state, not to
    # a difference in external forcing.
    solar = None
    if cfg.state.solar_forcing:
        if mesh is None:
            raise ValueError("perturbation_growth needs `mesh` when state.solar_forcing is on")
        from ..data.forcing import SolarForcing

        solar = SolarForcing(cache.times(split), mesh, device)
    fw_all = solar.window(torch.tensor([start], device=device), n_windows) if solar else None

    for w in range(n_windows):
        fw = fw_all[:, w] if fw_all is not None else None
        a = model.forecast_step(a, st, pa, z, fw)
        b = model.forecast_step(b, st, pb, z, fw)
        pa, pb = a[..., : cfg.c_phys], b[..., : cfg.c_phys]
        diff = b - a
        total.append(float(diff.norm()))
        if per_channel:
            per_ch.append(diff[..., : cfg.c_phys].pow(2).mean(dim=(0, 1)).sqrt().cpu().numpy())
        if cfg.model.reseed_hidden:
            a, b = model.reseed_hidden(a), model.reseed_hidden(b)

    total = np.array(total)
    out = {"norms": total, "ratios": total[1:] / np.maximum(total[:-1], 1e-30)}
    tail = out["ratios"][1:]
    tail = tail[np.isfinite(tail) & (tail > 0)]
    out["geometric_mean"] = float(np.exp(np.mean(np.log(tail)))) if len(tail) else float("nan")
    out.update(_settled(out["ratios"], threshold))

    if per_channel and per_ch:
        pc = np.stack(per_ch)  # [W, C]
        ratios = pc[1:] / np.maximum(pc[:-1], 1e-30)
        # Drop the same settling transient the total-norm rate drops, so the per-channel
        # numbers answer the same question and are comparable to it.
        skip = out["transient_windows"]
        with np.errstate(divide="ignore", invalid="ignore"):
            logs = np.log(np.where(ratios[skip:] > 0, ratios[skip:], np.nan))
        out["per_channel"] = pc
        out["per_channel_growth"] = np.exp(np.nanmean(logs, axis=0))
        out["channels"] = tuple(c.key for c in cfg.variables.channels())
    return out


@torch.no_grad()
def hidden_rms(model, cfg: Config, cache, split: str = "test", start: int = 5,
               n_windows: int = 12, device: str = "cpu", mesh: dict | None = None) -> np.ndarray:
    """Hidden-channel RMS through a rollout.

    Unbounded growth means the automaton is driving its latent state off-distribution, a
    failure single-step training is structurally blind to. M1 sat flat at 0.135, far under the
    overflow threshold of 1.0.
    """
    model.eval()
    series = cache.split(split).array
    st = torch.from_numpy(cache.static).float().to(device).unsqueeze(0)
    prev = torch.from_numpy(np.array(series[start - 1], dtype=np.float32)).unsqueeze(0).to(device)
    cur = torch.from_numpy(np.array(series[start], dtype=np.float32)).unsqueeze(0).to(device)

    n_windows = min(n_windows, len(series) - start - 1)
    fw_all = None
    if cfg.state.solar_forcing:
        if mesh is None:
            raise ValueError("hidden_rms needs `mesh` when state.solar_forcing is on")
        from ..data.forcing import SolarForcing

        fw_all = SolarForcing(cache.times(split), mesh, device).window(
            torch.tensor([start], device=device), n_windows)

    state, p = model.seed(cur), prev
    out = []
    for w in range(n_windows):
        state = model.forecast_step(state, st, p, None,
                                    fw_all[:, w] if fw_all is not None else None)
        p = state[..., : cfg.c_phys]
        out.append(float(state[..., cfg.c_phys :].pow(2).mean().sqrt()))
        if cfg.model.reseed_hidden:
            state = model.reseed_hidden(state)
    return np.array(out)


def _settled(ratios: np.ndarray, threshold: float) -> dict:
    """Split a settling transient from the sustained growth rate.

    M1's own guidance: "read the per-window ratios, not a geometric mean over all windows --
    the first window is routinely anomalous." Excluding exactly one window is not enough. A
    perturbation injected into the seed excites hidden channels that are re-seeded each window,
    so the response takes several windows to relax; averaging across that transient reports a
    settled operator as amplifying.

    The transient is the leading run of ratios above `threshold`; the sustained rate is the
    geometric mean of everything after it. The sustained rate is the number that answers "is
    the operator amplifying?" -- the transient answers "how hard was it kicked?".
    """
    r = np.asarray(ratios, dtype=float)
    ok = np.isfinite(r) & (r > 0)
    n_trans = 0
    while n_trans < len(r) and (not ok[n_trans] or r[n_trans] > threshold):
        n_trans += 1
    rest = r[n_trans:][ok[n_trans:]]
    # Keep at least the back half, so a genuinely amplifying operator cannot hide by being
    # classified as one long "transient".
    if len(rest) < max(3, len(r) // 2):
        rest = r[len(r) // 2 :][ok[len(r) // 2 :]]
        n_trans = len(r) // 2
    return {
        "transient_windows": int(n_trans),
        "sustained_growth": float(np.exp(np.mean(np.log(rest)))) if len(rest) else float("nan"),
        "sustained_from_window": int(n_trans + 1),
    }


def format_per_channel(result: dict, cfg, threshold: float = 1.05) -> str:
    """Sustained growth per variable and level -- the phase 2b check.

    `n_substeps = 20` was measured on one band-limited variable. Coupled dynamics carry faster
    adjustment processes, and a single-window CFL test cannot see per-window amplification that
    compounds (M1 established that the hard way). If one variable is amplifying while the rest
    are neutral, that variable is setting the sub-step budget for the whole state.

    Mitigation order, unchanged: more sub-steps, larger perception radius, then semi-Lagrangian
    pre-advection.
    """
    if "per_channel_growth" not in result:
        return "(per-channel growth unavailable)"
    g = result["per_channel_growth"]
    keys = list(result["channels"])
    v = cfg.variables

    lines = [f"sustained per-window growth by variable (from window "
             f"{result.get('sustained_from_window', '?')}; > {threshold} = amplifying)",
             f"{'variable':>24} " + " ".join(f"{lv:>9}" for lv in v.levels)]
    lines.append("-" * (25 + 10 * len(v.levels)))
    for name in v.atmospheric:
        row = []
        for lv in v.levels:
            k = f"{name}_{lv}"
            if k in keys:
                x = g[keys.index(k)]
                row.append(f"{x:>8.3f}{'*' if x > threshold else ' '}")
            else:
                row.append(f"{'-':>9}")
        lines.append(f"{name:>24} " + " ".join(row))
    for name in v.surface:
        if name in keys:
            x = g[keys.index(name)]
            lines.append(f"{name:>24} {x:>8.3f}{'*' if x > threshold else ''}")

    bad = [(keys[i], g[i]) for i in range(len(keys)) if g[i] > threshold]
    lines.append("")
    if bad:
        worst = max(bad, key=lambda t: t[1])
        lines.append(f"{len(bad)}/{len(keys)} channels above {threshold}; worst "
                     f"{worst[0]} at x{worst[1]:.3f}")
        lines.append("  -> this variable sets the sub-step budget for the whole state")
    else:
        lines.append(f"all {len(keys)} channels at or below {threshold}")
    return "\n".join(lines)


def summarize(result: dict, threshold: float = 1.05) -> str:
    sus = result.get("sustained_growth", float("nan"))
    verdict = "stable" if sus <= threshold else "AMPLIFYING"
    lines = [
        "per-window growth: " + "  ".join(f"{r:.2f}" for r in result["ratios"]),
        f"sustained growth from window {result.get('sustained_from_window', '?')}: "
        f"x{sus:.3f}  ({verdict})",
        f"  settling transient: {result.get('transient_windows', 0)} window(s); "
        f"naive mean over all but window 1 was x{result['geometric_mean']:.3f}",
    ]
    if "per_channel_growth" in result:
        worst = np.argsort(-result["per_channel_growth"])[:5]
        lines.append("worst channels (sustained): " + "  ".join(
            f"{result['channels'][i]} x{result['per_channel_growth'][i]:.3f}" for i in worst
        ))
    return "\n".join(lines)
