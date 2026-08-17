"""Fair CRPS: correctness, and the small-M bias that makes it necessary.

The plan is explicit that this must not be hand-rolled without a test, because the small-M
bias is exactly the regime where getting it wrong is invisible and fatal -- the loss goes down
while the ensemble collapses.

`scoringrules` is used when installed; the analytic CRPS of a standard normal is the fallback
so the check still runs in a bare environment.
"""
import numpy as np
import pytest
import torch

from wnca.losses.crps import (
    _pairwise_abs_sum, _sorted_abs_sum, ensemble_spread, fair_crps,
    fair_crps_pointwise, spread_skill_ratio,
)
from wnca.losses.terms import area_weights

# CRPS of N(0,1) evaluated at y=0:  (1/sqrt(pi)) (sqrt(2) - 1)
ANALYTIC_CRPS_N01_AT_0 = (np.sqrt(2) - 1) / np.sqrt(np.pi)


def test_sorted_equals_pairwise():
    """The O(M log M) form must equal the O(M^2) reference exactly (to float32)."""
    torch.manual_seed(0)
    m = torch.randn(3, 9, 17, 2, dtype=torch.float64)
    assert torch.allclose(_sorted_abs_sum(m), _pairwise_abs_sum(m), atol=1e-9)


def test_fair_estimator_is_unbiased_in_M():
    """The fair estimator must hit the analytic value at EVERY M; the naive one must not.

    This is the whole reason for the estimator. At M=4 the naive estimator over-states CRPS by
    ~60%, which during training means spread is under-credited and the model trains toward
    over-confidence.
    """
    torch.manual_seed(0)
    truth = torch.zeros(40000, 1)
    fair, naive = {}, {}
    for M in (2, 4, 8, 64):
        x = torch.randn(40000, M, 1)
        fair[M] = fair_crps_pointwise(x, truth, alpha=1.0).mean().item()
        naive[M] = fair_crps_pointwise(x, truth, alpha=0.0).mean().item()

    for M, v in fair.items():
        assert abs(v - ANALYTIC_CRPS_N01_AT_0) < 0.01, f"fair estimator biased at M={M}: {v}"
    # The naive estimator's bias must shrink with M -- i.e. it IS M-dependent, unlike the fair one.
    assert naive[2] > naive[4] > naive[8] > naive[64], f"naive estimator not converging: {naive}"
    assert naive[4] - ANALYTIC_CRPS_N01_AT_0 > 10 * abs(fair[4] - ANALYTIC_CRPS_N01_AT_0)


@pytest.mark.parametrize("M", [2, 5])
def test_against_scoringrules(M):
    scoringrules = pytest.importorskip("scoringrules")
    torch.manual_seed(0)
    x = torch.randn(64, M, 1, dtype=torch.float64)
    y = torch.randn(64, 1, dtype=torch.float64)
    ours = fair_crps_pointwise(x, y, alpha=0.0).squeeze(-1).numpy()  # naive == their default
    theirs = scoringrules.crps_ensemble(y.squeeze(-1).numpy(), x.squeeze(-1).numpy(), estimator="nrg")
    assert np.allclose(ours, theirs, atol=1e-8), f"max diff {np.abs(ours - theirs).max()}"


def test_perfect_forecast_scores_zero():
    x = torch.full((4, 6, 10, 2), 3.0)
    assert abs(fair_crps_pointwise(x, torch.full((4, 10, 2), 3.0)).abs().max().item()) < 1e-6


def test_crps_rewards_sharpness_when_correct():
    """A tight ensemble on the truth must beat a wide one. A proper score has to do this."""
    torch.manual_seed(0)
    truth = torch.zeros(500, 8, 1)
    aw = torch.ones(8, 1)
    tight = 0.1 * torch.randn(500, 6, 8, 1)
    wide = 2.0 * torch.randn(500, 6, 8, 1)
    assert fair_crps(tight, truth, aw).item() < fair_crps(wide, truth, aw).item()


def test_crps_penalizes_overconfidence():
    """A tight ensemble in the WRONG place must lose to a wide one covering the truth."""
    torch.manual_seed(0)
    truth = torch.zeros(500, 8, 1)
    aw = torch.ones(8, 1)
    confident_wrong = 3.0 + 0.05 * torch.randn(500, 6, 8, 1)
    honest_wide = 2.0 * torch.randn(500, 6, 8, 1)
    assert fair_crps(honest_wide, truth, aw).item() < fair_crps(confident_wrong, truth, aw).item()


def test_area_weighting_changes_the_answer(small_mesh):
    """If area weights were being ignored, this would silently pass everywhere else."""
    torch.manual_seed(0)
    N = len(small_mesh["v"])
    aw = area_weights(small_mesh["area"])
    x = torch.randn(2, 4, N, 1)
    y = torch.randn(2, N, 1)
    assert not np.isclose(fair_crps(x, y, aw).item(), fair_crps(x, y, torch.ones(N, 1)).item())


def test_spread_skill_of_a_calibrated_ensemble_is_one():
    """Construct a genuinely calibrated ensemble: truth drawn from the same law as members."""
    torch.manual_seed(0)
    B, M, N = 4000, 12, 4
    truth = torch.randn(B, N, 1)
    members = truth.unsqueeze(1) + torch.randn(B, M, N, 1)
    ss = spread_skill_ratio(members, truth + torch.randn(B, N, 1), torch.ones(N, 1))
    assert 0.9 < ss.mean().item() < 1.1, f"calibrated ensemble scored {ss.mean().item()}"


def test_spread_detects_collapse():
    collapsed = torch.zeros(4, 6, 10, 1) + torch.randn(4, 1, 10, 1)
    assert ensemble_spread(collapsed, torch.ones(10, 1)).max().item() < 1e-5


def test_fair_crps_rejects_single_member():
    with pytest.raises(ValueError, match="M >= 2"):
        fair_crps_pointwise(torch.randn(2, 1, 3, 1), torch.randn(2, 3, 1))


def test_gradients_flow():
    x = torch.randn(2, 4, 6, 1, requires_grad=True)
    fair_crps(x, torch.randn(2, 6, 1), torch.ones(6, 1)).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
