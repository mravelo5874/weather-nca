# weather-nca

A neural cellular automaton on an icosahedral mesh for global weather forecasting.

Milestone 1 asked whether a strictly local update rule can advect at all — it can (24h z500 RMSE
344.7 vs. persistence 593.9, stable to 7 days). Milestone 2 asks whether that locality costs
accuracy relative to a same-budget non-local model, and whether the model can produce a
calibrated, sharp probabilistic ensemble. See [docs/milestone-2-plan.md](docs/milestone-2-plan.md).

## Layout

- `src/wnca/` — the package: mesh operators, data pipeline, models, losses, training, eval.
- `configs/` — one YAML per phase, layered on `base.yaml`.
- `scripts/` — entry points (`train.py`, `evaluate.py`, `build_mesh.py`, `build_cache.py`, `benchmark_step.py`).
- `tests/` — unit tests; `make test` must stay under 60s.
- `notebooks/` — analysis only, never authoritative. All results are produced by the package.
- `docs/` — milestone plans, findings, background reading, and `decisions/` (one file per
  irreversible design choice).

## Getting started

```bash
pip install -e ".[dev]"
make test
make smoke
```

See [CLAUDE.md](CLAUDE.md) for agent operating instructions and standing invariants.
