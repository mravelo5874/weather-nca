# 0002 — Fair CRPS, computed by sorting

**Date:** 2026-08-16
**Status:** accepted, implemented in `losses/crps.py`

## Decision

Use the **fair** (unbiased) kernel CRPS estimator

    CRPS_fair = (1/M) Σᵢ |xᵢ − y|  −  1/(2M(M−1)) Σᵢ Σⱼ |xᵢ − xⱼ|

with `alpha` blending toward the naive `2M²` denominator, defaulting to fully fair. Compute the
pairwise term by **sorting** rather than by materializing the difference tensor:

    Σᵢⱼ |xᵢ − xⱼ| = 2 Σₖ (2k − M + 1) x₍ₖ₎

## Why fair

The naive estimator under-weights spread and trains toward over-confidence. Measured here
(`tests/test_crps.py`): against the analytic CRPS of a standard normal, 0.23369 —

| M | fair | naive |
|---|------|-------|
| 2 | 0.2366 | 0.5189 |
| 4 | 0.2345 | 0.3750 |
| 8 | 0.2336 | 0.3041 |
| 64 | 0.2339 | 0.2515 |

The fair estimator is unbiased at every M. The naive one is off by 60% at M = 4, which is the
training regime. Small M is exactly where getting this wrong is invisible and fatal: the loss
falls while the ensemble collapses.

## Why by sorting

The plan's reference implementation materializes `[B, M, M, N, C]`. At the scorecard's M = 50 on
a 28-channel state that is ~2.9 GB for a *single* batch element. The sorted form is exact,
differentiable, and costs the size of `members`. Both forms are implemented and
`test_sorted_equals_pairwise` asserts they agree.

## Verification

`fair_crps_pointwise(..., alpha=0)` matches `scoringrules.crps_ensemble(estimator="nrg")` to
1e-8, and the fair form is checked against the analytic value at four values of M.
