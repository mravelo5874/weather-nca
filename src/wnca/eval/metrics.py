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
