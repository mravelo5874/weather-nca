"""Area weighting, overflow penalty, per-channel weights.

Area weights come from the mesh's barycentric vertex areas. On an icosphere these are nearly
uniform, which is exactly why the mesh was chosen over a lat-lon grid -- but they are applied
anyway, so that the metric is defined the same way here as it is on the 1.5 degree WB2 grid
where the weights matter a great deal.
"""

from __future__ import annotations

import numpy as np
import torch

from ..config import Config


def area_weights(area: np.ndarray, device="cpu") -> torch.Tensor:
    """Barycentric vertex areas normalized to mean 1, shaped [N, 1] for broadcasting."""
    w = torch.from_numpy(np.asarray(area, dtype=np.float32)).to(device)
    return (w / w.mean()).view(-1, 1)


def channel_weights(cfg: Config, device="cpu") -> torch.Tensor:
    """Per-channel loss weights, [1, C]. Unlisted channels get 1.0."""
    keys = [c.key for c in cfg.variables.channels()]
    w = torch.ones(len(keys), dtype=torch.float32, device=device)
    for i, k in enumerate(keys):
        if k in cfg.loss.channel_weights:
            w[i] = float(cfg.loss.channel_weights[k])
    return w.view(1, -1)


def weighted_mean(x: torch.Tensor, area_w: torch.Tensor, chan_w: torch.Tensor | None = None) -> torch.Tensor:
    """Area- and channel-weighted mean of `[..., N, C]` down to a scalar.

    Weights are means, not sums: both weight tensors have mean 1 by construction, so the
    result stays on the same scale as an unweighted mean and is comparable across meshes.
    """
    w = area_w if chan_w is None else area_w * chan_w
    return (x * w).sum() / (torch.ones_like(x) * w).sum()


def area_weighted_mse(pred: torch.Tensor, target: torch.Tensor, area_w: torch.Tensor,
                      chan_w: torch.Tensor | None = None) -> torch.Tensor:
    """The M1 deterministic objective. Used unchanged by phases 2a-2d."""
    return weighted_mean((pred - target) ** 2, area_w, chan_w)


def overflow_penalty(hidden: torch.Tensor) -> torch.Tensor:
    """Penalty on |hidden| > 1, from the original NCA paper. Carried from M1."""
    return torch.relu(hidden.abs() - 1.0).mean()


def per_channel_rmse(pred: torch.Tensor, target: torch.Tensor, area_w: torch.Tensor) -> torch.Tensor:
    """Area-weighted RMSE per channel, pooled over batch and nodes: [..., N, C] -> [C].

    Squares are pooled and the square root taken last. Averaging per-start RMSE instead is
    biased low by Jensen and is not comparable to published numbers -- M1 practice, preserved.
    """
    sq = (pred - target) ** 2
    dims = tuple(range(sq.ndim - 1))
    num = (sq * area_w).sum(dim=dims)
    den = (torch.ones_like(sq) * area_w).sum(dim=dims)
    return torch.sqrt(num / den)
