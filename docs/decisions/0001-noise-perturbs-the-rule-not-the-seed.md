# 0001 — Noise perturbs the update rule, not the seed

**Date:** 2026-08-16
**Status:** accepted, implemented in `models/update.py`

## Decision

The probabilistic head is a per-member noise vector `z ∈ ℝ^16` that FiLM-modulates every hidden
layer of the update MLP. It is drawn **once per member per forecast trajectory** and held
constant across all cells and all sub-steps.

## Why

The NCA applies `n_substeps = 20` rounds of a smoothing local operator per forecast window.
Any perturbation placed in the seed field is exactly the thing that operator is built to damp.
The network does not have to learn to ignore seed noise — the dynamics erase it for free.
Ensemble collapse was therefore the *expected* outcome of the v1 design, not a risk of it.

Perturbing the rule cannot be damped, because it changes the operator rather than the state the
operator acts on.

The low dimensionality is load-bearing, not an efficiency choice: it is what forces the
perturbation to express itself as coherent, physically plausible spatial structure rather than
per-cell noise. This is the FGN (WeatherNext 2) result.

## Consequences

- `reseed_hidden = True` re-seeds hidden channels every window but must **never** re-draw `z`.
  If it did, member identity would dissolve and spread would stop meaning anything at long
  leads. Enforced by `tests/test_noise.py::test_reseed_hidden_does_not_touch_noise`.
- The FiLM projection is zero-initialized, so a stochastic model is numerically identical to
  the deterministic checkpoint it warm-starts from. That is what makes the 2c → 3a transition
  safe, and it is why phase 3a can drop straight to lr 1e-4 instead of re-warming.
- Members are folded into the batch axis (`[B, M, N, C] → [B*M, N, C]`), so one `z` per row
  broadcasts over all nodes by construction rather than by convention.

## What would reverse this

A measured ensemble collapse that survives the three phase-3a probes. The fix would be to move
where the FiLM modulation attaches — not to add more noise.
