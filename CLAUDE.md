# Agent operating instructions

This file is invariants and prior failures, not a description of the code — read the code for that.

## Three time granularities — do not confuse them

- `nca_step` — one pass of the update rule (one PDE sub-step).
- `forecast_step` — one 6h forecast window, composed of `n_substeps` (`= 20`) `nca_step`s.
- `rollout` — a chain of `forecast_step`s, e.g. 15 days.

Mixing these up is the most common error in this codebase. Sub-step budgets, noise draws, and
`reseed_hidden` all operate at different granularities — check which one before changing any of
them.

## Methodology

Prefer direct measurement over clever measurement. `perturbation_growth` was right every time in
M1; the one-sided spectrum check and the finite-difference spectral radius both misfired
confidently. Reach for the direct diagnostic first.

## Incidents that already happened (M1) — do not repeat

1. **Untrained-model evaluation.** Three M1 evaluation rounds ran on a zero-init identity model
   that reproduced persistence exactly, because a notebook cell rerun silently rebuilt the model.
   `test_checkpoint.py` and the load-time weight-norm assert exist because of this.
2. **Incommensurable checkpoint metrics.** Comparing selection metrics across phase boundaries
   produced meaningless comparisons. One fixed selection metric per phase, never compared across
   phases.
3. **One-sided spectrum check.** A clever diagnostic misfired confidently where the direct
   measurement wouldn't have. See Methodology above.
4. **`best.pt` clobbering.** A single fixed checkpoint path got silently overwritten mid-comparison.
   Checkpoint filenames are timestamped, always.

## Hard constraints

- Normalization statistics (per variable, per level) are computed on the **train split only**.
- RMSE is **pooled across start times with the square root taken last**. Averaging per-start RMSE
  is biased low by Jensen's inequality and is not comparable to published numbers.
- `reseed_hidden = True` re-seeds hidden channels every forecast window but must **never** re-draw
  the noise vector `z`. Re-drawing `z` dissolves member identity and spread stops meaning anything
  at long leads.
- The noise vector `z` is drawn once per member per trajectory and held constant across all cells
  and all sub-steps within that member.

## Do not, without a measurement justifying it

- Re-enable the rollout curriculum (`rollout_epochs > 0` in deterministic phases). M1 tried this
  twice; it contributed nothing and destroyed the long-lead metric both times.
- Expand perception to two hops.
- Add a semi-Lagrangian pre-advection step.

These were all considered and rejected in M1/M2 planning. An agent optimizing a loss will reach
for them — don't, unless a direct measurement (see Methodology) justifies it.

## Commands

```bash
make test                    # unit tests, must stay under 60s
make smoke                   # full train -> eval cycle, n_sub=3, one month of data, < 2 min
make train CONFIG=configs/phaseX.yaml
make eval CONFIG=configs/phaseX.yaml
```

Run `make smoke` before any phase goes to cloud.

## Decisions

Irreversible design choices go in `docs/decisions/`, one file per choice, at the time the choice
is made — not reconstructed after the fact.

## Regression tests

Whenever a numerical result changes unexpectedly, add a regression test for it before moving on.
