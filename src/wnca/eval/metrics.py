"""RMSE, CRPS, spread-skill, rank histogram.

RMSE is pooled across start times with the square root taken last. Averaging per-start RMSE
instead is biased low by Jensen and is not comparable to published numbers. This is already M1
practice and it is preserved here in one place so no scoring path can quietly diverge from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..config import HOURS_PER_WINDOW, Config
from ..losses.crps import crps_per_channel, ensemble_spread
from ..losses.terms import area_weights


@dataclass
class Scorecard:
    """Accumulates squared error / CRPS across start times, pooling before the square root."""

    n_windows: int
    n_channels: int
    channels: tuple[str, ...]
    sq_model: np.ndarray = field(init=False)
    sq_persist: np.ndarray = field(init=False)
    sq_clim: np.ndarray = field(init=False)
    crps: np.ndarray = field(init=False)
    spread: np.ndarray = field(init=False)
    n: int = 0

    def __post_init__(self):
        z = lambda: np.zeros((self.n_windows, self.n_channels))  # noqa: E731
        self.sq_model, self.sq_persist, self.sq_clim = z(), z(), z()
        self.crps, self.spread = z(), z()

    def rmse(self, which: str = "model") -> np.ndarray:
        acc = {"model": self.sq_model, "persistence": self.sq_persist, "climatology": self.sq_clim}[which]
        return np.sqrt(acc / max(self.n, 1))

    def mean_crps(self) -> np.ndarray:
        return self.crps / max(self.n, 1)

    def mean_spread(self) -> np.ndarray:
        return np.sqrt(self.spread / max(self.n, 1))

    def spread_skill(self, m_members: int) -> np.ndarray:
        corr = np.sqrt((m_members + 1) / m_members)
        return self.mean_spread() * corr / np.maximum(self.rmse("model"), 1e-12)

    def leads(self) -> np.ndarray:
        return (np.arange(self.n_windows) + 1) * HOURS_PER_WINDOW


@torch.no_grad()
def evaluate(
    model,
    cfg: Config,
    cache,
    mesh,
    split: str = "test",
    device: str = "cpu",
    max_windows: int | None = None,
    n_starts: int | None = None,
    n_members: int | None = None,
) -> Scorecard:
    """Score a model against persistence and climatology, per lead and per channel.

    Values are returned in **normalized** units. `to_physical` converts the non-log channels;
    log-transformed channels (specific humidity) stay in log units and the scorecard says so,
    because a difference in log space has no single physical scale.
    """
    model.eval()
    series = cache.split(split).array
    T = len(series)
    max_windows = min(max_windows or cfg.eval.max_windows, T - 3)
    n_starts = n_starts or cfg.eval.n_starts
    M = n_members if n_members is not None else (cfg.ensemble.m_test if cfg.model.stochastic else 1)

    aw = area_weights(mesh["area"], device)
    clim = torch.from_numpy(cache.climatology()).float().to(device)  # [N, C] from TRAIN only
    channels = tuple(c.key for c in cfg.variables.channels())
    sc = Scorecard(max_windows, len(channels), channels)

    usable = T - max_windows - 2
    if usable < 1:
        raise ValueError(f"split {split!r} has {T} steps, too short for {max_windows} windows")
    starts = np.unique(np.linspace(1, usable, min(n_starts, usable)).astype(int))

    for s0 in starts:
        prev = torch.from_numpy(np.array(series[s0 - 1], dtype=np.float32)).unsqueeze(0).to(device)
        cur = torch.from_numpy(np.array(series[s0], dtype=np.float32)).unsqueeze(0).to(device)
        tgt = torch.from_numpy(
            np.array(series[s0 + 1 : s0 + 1 + max_windows], dtype=np.float32)
        ).unsqueeze(0).to(device)  # [1, W, N, C]

        st = torch.from_numpy(cache.static).float().to(device).unsqueeze(0)
        pred = model.rollout_ensemble(model.seed(cur), st, max_windows, prev_phys=prev, n_members=M)
        # pred [1, M, W, N, C]

        mean = pred.mean(dim=1)  # [1, W, N, C]
        sc.sq_model += _wmse(mean, tgt, aw)
        sc.sq_persist += _wmse(cur.unsqueeze(1).expand_as(tgt), tgt, aw)
        sc.sq_clim += _wmse(clim.view(1, 1, *clim.shape).expand_as(tgt), tgt, aw)

        if M > 1:
            for wi in range(max_windows):
                sc.crps[wi] += crps_per_channel(pred[:, :, wi], tgt[:, wi], aw,
                                                cfg.loss.crps_alpha).cpu().numpy()
                sc.spread[wi] += (ensemble_spread(pred[:, :, wi], aw) ** 2).cpu().numpy()
        sc.n += 1
    return sc


def _wmse(pred: torch.Tensor, tgt: torch.Tensor, aw: torch.Tensor) -> np.ndarray:
    """Area-weighted MSE per (window, channel). Squares accumulate; the root comes later."""
    sq = (pred - tgt) ** 2  # [1, W, N, C]
    num = (sq * aw).sum(dim=(0, 2))
    den = (torch.ones_like(sq) * aw).sum(dim=(0, 2))
    return (num / den).cpu().numpy()


def to_physical(values: np.ndarray, normalizer, log_units_ok: bool = True) -> np.ndarray:
    """Scale normalized RMSE/CRPS to physical units, per channel.

    A difference in normalized space maps to physical units by multiplying by the channel's
    standard deviation -- but only for channels without a log transform. Log channels are left
    in log units; the printed scorecard marks them.
    """
    out = np.asarray(values, dtype=np.float64) * normalizer.std[None, :]
    if not log_units_ok and normalizer.log_mask.any():
        out[:, normalizer.log_mask] = np.nan
    return out


def rank_histogram(members: torch.Tensor, truth: torch.Tensor, n_bins: int | None = None) -> np.ndarray:
    """Where truth falls among the sorted members. Flat = calibrated.

    Run this once metrics 1-4 pass: a rank histogram on an ensemble that is not yet
    approximately calibrated tells you nothing you did not already know from spread-skill.
    """
    M = members.shape[1]
    n_bins = n_bins or (M + 1)
    below = (members < truth.unsqueeze(1)).sum(dim=1)  # [B, ...]
    counts = torch.bincount(below.reshape(-1), minlength=M + 1).float().cpu().numpy()
    return counts / max(counts.sum(), 1.0)


def channel_units(cfg: Config, normalizer) -> list[str]:
    """Display unit per channel. Log-transformed channels are NOT in physical units."""
    out = []
    for i, c in enumerate(cfg.variables.channels()):
        if normalizer.log_mask[i]:
            out.append("log")
        elif c.name.startswith("geopotential"):
            out.append("m2/s2")
        elif "wind" in c.name:
            out.append("m/s")
        elif "temperature" in c.name:
            out.append("K")
        else:
            out.append("-")
    return out


def _scaled(sc: Scorecard, which: str, normalizer) -> np.ndarray:
    """RMSE per (window, channel) in each channel's display unit."""
    return sc.rmse(which) * normalizer.std[None, :]


def format_level_table(sc: Scorecard, cfg: Config, normalizer, lead_hours: int = 24) -> str:
    """Variable x level table at one lead. The natural layout for a multi-level atmosphere.

    Log-transformed channels (specific humidity) are reported in **log units**, because a
    difference in log space has no single physical scale. Usefully, d(log q) ~ dq/q, so an RMSE
    of 0.10 there reads as roughly a 10% error in q.
    """
    i = lead_hours // HOURS_PER_WINDOW - 1
    if not 0 <= i < sc.n_windows:
        return f"(lead {lead_hours}h outside the scored range)"
    m = _scaled(sc, "model", normalizer)[i]
    p = _scaled(sc, "persistence", normalizer)[i]
    keys = list(sc.channels)
    v = cfg.variables
    units = channel_units(cfg, normalizer)

    lines = [f"RMSE at +{lead_hours}h   (skill vs persistence in parentheses)",
             f"{'variable':>24} {'unit':>6} " + " ".join(f"{lv:>15}" for lv in v.levels)]
    lines.append("-" * (31 + 16 * len(v.levels)))
    for name in v.atmospheric:
        row, unit = [], ""
        for lv in v.levels:
            k = f"{name}_{lv}"
            if k in keys:
                j = keys.index(k)
                unit = units[j]
                skill = 1 - m[j] / max(p[j], 1e-12)
                row.append(f"{m[j]:>8.3g} ({skill:>4.0%})")
            else:
                row.append(f"{'-':>15}")
        lines.append(f"{name:>24} {unit:>6} " + " ".join(row))

    if v.surface:
        lines.append("")
        for name in v.surface:
            if name in keys:
                j = keys.index(name)
                skill = 1 - m[j] / max(p[j], 1e-12)
                lines.append(f"{name:>24} {units[j]:>6} {m[j]:>8.3g} ({skill:>4.0%})")
    return "\n".join(lines)


def format_channel_summary(sc: Scorecard, cfg: Config, normalizer,
                           leads: tuple[int, ...] = (24, 72, 120)) -> str:
    """One row per channel, RMSE across several leads. The full scorecard at a glance."""
    idx = [(h, h // HOURS_PER_WINDOW - 1) for h in leads]
    idx = [(h, i) for h, i in idx if 0 <= i < sc.n_windows]
    m = _scaled(sc, "model", normalizer)
    p = _scaled(sc, "persistence", normalizer)
    units = channel_units(cfg, normalizer)
    has_crps = sc.crps.any()

    head = f"{'channel':>28} {'unit':>6} " + " ".join(f"{f'+{h}h':>12}" for h, _ in idx)
    if has_crps:
        head += f" {'CRPS@' + str(idx[0][0]) + 'h':>12}"
    lines = [head, "-" * len(head)]
    for j, key in enumerate(sc.channels):
        row = " ".join(f"{m[i, j]:>7.3g} ({1 - m[i, j] / max(p[i, j], 1e-12):>3.0%})" for _, i in idx)
        line = f"{key:>28} {units[j]:>6} {row}"
        if has_crps:
            line += f" {sc.mean_crps()[idx[0][1], j] * normalizer.std[j]:>12.3g}"
        lines.append(line)
    return "\n".join(lines)


def format_scorecard(sc: Scorecard, cfg: Config, normalizer, channel: str = "geopotential_500",
                     physical: bool = True) -> str:
    """Human-readable table for one channel, in the M1 evaluation format."""
    try:
        ci = sc.channels.index(channel)
    except ValueError:
        ci = 0
        channel = sc.channels[0]
    scale = float(normalizer.std[ci]) if physical and not normalizer.log_mask[ci] else 1.0
    unit = "" if scale != 1.0 else " (normalized)"

    m, p, c = sc.rmse("model")[:, ci] * scale, sc.rmse("persistence")[:, ci] * scale, sc.rmse("climatology")[:, ci] * scale
    crps = sc.mean_crps()[:, ci] * scale
    leads = sc.leads()

    lines = [f"channel: {channel}{unit}",
             f"{'lead':>7} {'model':>10} {'persist':>10} {'clim':>10} {'CRPS':>10} {'skill':>8}"]
    lines.append("-" * 60)
    for h in cfg.eval.lead_hours:
        i = h // HOURS_PER_WINDOW - 1
        if 0 <= i < len(m):
            skill = 1 - m[i] / max(p[i], 1e-12)
            cr = f"{crps[i]:>10.1f}" if crps[i] > 0 else f"{'-':>10}"
            lines.append(f"{h:>5}h  {m[i]:>10.1f} {p[i]:>10.1f} {c[i]:>10.1f} {cr} {skill:>7.1%}")
    return "\n".join(lines)
