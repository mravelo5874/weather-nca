"""Fair / almost-fair kernel CRPS.

    CRPS_fair = (1/M) sum_i |x_i - y|  -  1/(2 M (M-1)) sum_i sum_j |x_i - x_j|

The naive estimator divides the second term by 2M^2 instead, which under-weights spread and
trains toward over-confidence. At M = 4 the difference in that term is 33%, and small M is
exactly the regime where getting it wrong is invisible and fatal -- the loss goes down while
the ensemble collapses. `alpha` blends the two (AIFS uses ~0.95 for gradient stability);
`alpha = 1.0` is fully fair.

The pairwise sum is computed by sorting rather than by materializing the [B, M, M, N, C]
difference tensor. For sorted members,

    sum_{i,j} |x_i - x_j| = 2 sum_k (2k - M + 1) x_(k)

which is exact, differentiable, and O(M log M) in memory instead of O(M^2). At the scorecard's
M = 50 on a full 28-channel state the pairwise form needs ~2.9 GB for a single batch element;
this form needs the size of `members`. `tests/test_crps.py` checks the two agree.
"""

from __future__ import annotations

import torch

from .terms import weighted_mean


def _pairwise_abs_sum(members: torch.Tensor) -> torch.Tensor:
    """sum_{i,j} |x_i - x_j| over the member axis. Reference form, O(M^2) memory."""
    return (members.unsqueeze(1) - members.unsqueeze(2)).abs().sum(dim=(1, 2))


def _sorted_abs_sum(members: torch.Tensor) -> torch.Tensor:
    """sum_{i,j} |x_i - x_j| over the member axis, via sorting. O(M log M) memory."""
    M = members.shape[1]
    xs, _ = torch.sort(members, dim=1)
    k = torch.arange(M, device=members.device, dtype=members.dtype)
    coef = (2.0 * k - M + 1.0).view(1, M, *([1] * (members.ndim - 2)))
    return 2.0 * (coef * xs).sum(dim=1)


def fair_crps_pointwise(members: torch.Tensor, truth: torch.Tensor, alpha: float = 1.0,
                        exact_pairwise: bool = False) -> torch.Tensor:
    """Per-element fair CRPS. members [B, M, ...], truth [B, ...] -> [B, ...]."""
    M = members.shape[1]
    if M < 2:
        raise ValueError(f"fair CRPS needs M >= 2 members, got {M}")

    skill = (members - truth.unsqueeze(1)).abs().mean(dim=1)
    pair = _pairwise_abs_sum(members) if exact_pairwise else _sorted_abs_sum(members)
    coef = alpha / (2 * M * (M - 1)) + (1 - alpha) / (2 * M * M)
    return skill - coef * pair


def fair_crps(members: torch.Tensor, truth: torch.Tensor, area_w: torch.Tensor,
              chan_w: torch.Tensor | None = None, alpha: float = 1.0,
              exact_pairwise: bool = False) -> torch.Tensor:
    """Area-weighted scalar fair CRPS.

    members [B, M, N, C] | truth [B, N, C] | area_w [N, 1] | chan_w [1, C].
    """
    return weighted_mean(
        fair_crps_pointwise(members, truth, alpha, exact_pairwise), area_w, chan_w
    )


def crps_per_channel(members: torch.Tensor, truth: torch.Tensor, area_w: torch.Tensor,
                     alpha: float = 1.0) -> torch.Tensor:
    """Fair CRPS per channel, area-weighted: [B, M, N, C] -> [C]. For the scorecard."""
    pw = fair_crps_pointwise(members, truth, alpha)
    dims = tuple(range(pw.ndim - 1))
    num = (pw * area_w).sum(dim=dims)
    den = (torch.ones_like(pw) * area_w).sum(dim=dims)
    return num / den


def ensemble_spread(members: torch.Tensor, area_w: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    """Area-weighted ensemble standard deviation per channel: [B, M, N, C] -> [C].

    Pooled as a variance and square-rooted last, matching the RMSE convention so that
    spread-skill is a ratio of two quantities computed the same way.
    """
    var = members.var(dim=1, unbiased=unbiased)  # [B, N, C]
    dims = tuple(range(var.ndim - 1))
    num = (var * area_w).sum(dim=dims)
    den = (torch.ones_like(var) * area_w).sum(dim=dims)
    return torch.sqrt(num / den)


def spread_skill_ratio(members: torch.Tensor, truth: torch.Tensor, area_w: torch.Tensor) -> torch.Tensor:
    """Spread / RMSE of the ensemble mean, per channel. Target 1.0.

    This is the check MSE cannot pass by construction, and it is phase 3a's gate. The
    sqrt((M+1)/M) factor is the standard finite-ensemble correction: for a calibrated ensemble
    of M members the expected error of the mean exceeds the sample spread by exactly that.
    """
    from .terms import per_channel_rmse

    M = members.shape[1]
    spread = ensemble_spread(members, area_w)
    rmse = per_channel_rmse(members.mean(dim=1), truth, area_w)
    corr = (torch.tensor((M + 1) / M, device=members.device, dtype=members.dtype)).sqrt()
    return (spread * corr) / rmse.clamp_min(1e-12)
