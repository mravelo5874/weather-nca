# weather-nca

A neural cellular automaton on an icosahedral mesh for global weather forecasting.

**Milestone 1** asked whether a strictly local update rule can advect at all. It can: 24 h z500
RMSE 344.7 vs. persistence 593.9 on real ERA5, stable to 7 days, spectrum preserved at 24 h.

**Milestone 2** asks the follow-up that decides whether to keep going:

1. Does restricting the update to a strictly local rule **cost accuracy** relative to a
   non-local model of equal budget?
2. Can that rule be made to produce a **calibrated, sharp ensemble**?

Question 1 is why there is a same-budget control GNN (phase 2d) rather than a comparison against
a frontier model — "how far are we from DeepMind" is unanswerable on four spot GPUs, while "does
locality cost accuracy, holding everything else fixed" is answerable and is the actual thesis.

See [docs/milestone-2-plan.md](docs/milestone-2-plan.md) for the full plan and
[docs/decisions/](docs/decisions/) for the irreversible choices and why they were made.

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cu124   # or /cpu
make install
make test      # 85 tests, ~17s
make smoke     # full train -> eval cycle on synthetic data, no network needed
```

## Running a phase

Phases are ordered so each isolates one variable. Do not skip the gates.

| Phase | Config | Isolates | Gate |
|---|---|---|---|
| 0 | `phase0_m1_repro.yaml` | — | 24 h z500 RMSE within 2% of 344.7 |
| 2a | `phase2a_data.yaml` | data volume | documented RMSE vs the 2-year number |
| 2b | `phase2b_multivar.yaml` | coupling | documented RMSE vs 2a |
| 2c | `phase2c_full.yaml` | combined + capacity | best deterministic checkpoint |
| 2d | `phase2d_control.yaml` | **locality** | control trained and scored |
| 3a | `phase3a_crps.yaml` | loss + noise | spread–skill > 0.3 by epoch 3 |
| 3b | `phase3b_spectral.yaml` | spectral term | ablation documented either way |

```bash
make benchmark CONFIG=configs/phase2c_full.yaml   # measure before committing compute
make cache     CONFIG=configs/phase2c_full.yaml   # resumable; safe to interrupt
make train     CONFIG=configs/phase2c_full.yaml
make eval      CONFIG=configs/phase2c_full.yaml
```

The CLI takes ad-hoc overrides, which is how the 2c capacity sweep is run:

```bash
wnca train -c configs/phase2c_full.yaml --set model.hidden_dim=768 model.n_layers=4
```

Every run writes its fully-resolved config next to its checkpoints. Reproducing a run means
pointing at that file, not remembering what was edited.

## Layout

```
src/wnca/
  config.py     frozen dataclasses, YAML load with `extends`, schema validation
  mesh/         icosphere, cotangent Laplacian + LS gradient, regrid, Chebyshev bands
  data/         WB2 ERA5 ingest (with the transpose fix), normalization, resumable memmap cache
  models/       MeshPerception, FiLM UpdateRule, WeatherNCA, same-budget ControlGNN
  losses/       fair CRPS, band-energy CRPS, area weighting + overflow penalty
  train/        loop, phases, timestamped/asserted/resumable checkpointing
  eval/         RMSE + CRPS + spread-skill scorecard, perturbation growth, spectra, WB2 regrid
configs/        base.yaml + one per phase
tests/          85 tests; `make test` must stay under 60s
docs/decisions/ one file per irreversible choice
notebooks/      analysis only, never authoritative
```

## Things worth knowing before changing anything

Read [CLAUDE.md](CLAUDE.md) — it carries the invariants and the four M1 incidents. The short
version:

- **Three time granularities** — `nca_step` (one PDE sub-step), `forecast_step` (`n_substeps` of
  them = one 6 h window), `rollout` (a chain of those). Confusing them is the most common error.
- **The noise vector is drawn once per member per trajectory** and held constant across all
  cells and sub-steps. `reseed_hidden` must never re-draw it.
- **Normalization stats come from the train split only**; RMSE pools before the square root.
- **The rollout curriculum is disabled and gated.** M1 ran it twice; it contributed nothing and
  destroyed the long-lead metric both times.
- **Prefer direct measurement over clever measurement.** `perturbation_growth` was right every
  time in M1; two cleverer diagnostics misfired confidently.
