# Milestone 1 — Minimal Viable Weather NCA
 
**Goal:** A single-variable neural cellular automaton on the sphere that beats persistence at a 24-hour lead time.
 
This is deliberately the smallest thing that can fail informatively. It exists to answer one question — *can a strictly local update rule advect?* — before we commit to a full 84-channel atmospheric state.
 
---
 
## Scope
 
| | Milestone 1 | (Deferred) |
|---|---|---|
| Variables | z500 only | Full prognostic state, 13 levels |
| Resolution | 1° | 0.25° |
| Output | Deterministic | Probabilistic / ensemble |
| Cells | Fixed, Eulerian | + Lagrangian feature particles |
| Lead time | 24h | 15 days |
 
---
 
## Design decisions (locked)
 
**Geometry — icosahedral mesh, not lat-lon grid.**
Square lattices distort badly toward the poles. Use a 5-times-refined icosahedral mesh (~10k nodes at 1°) with either mesh-NCA perception or SPH perception in static-particle mode.
 
**Cell state — a vertical column, even though we only have one level now.**
Structure the state as `[physical channels | hidden channels]` from day one so adding levels later is a config change, not a rewrite. Milestone 1: 1 physical + ~15 hidden.
 
**Time — NCA steps are PDE sub-steps, not forecast steps.**
A 6h forecast step is composed of ~10 NCA updates at 1°. This is the CFL budget and it is the thing most likely to break.
 
**Update mask — deterministic (p = 1).**
The stochastic Bernoulli mask from the texture literature is unhelpful for a conservation-law system. Revisit only if training is unstable.
 
**Conditioning — second-order Markov.**
Feed the model the current and previous state. Cheap, and gives it a tendency estimate for free.
 
---
 
## Data
 
- **ERA5**, geopotential at 500 hPa, 6-hourly, downsampled to 1°.
- Access via ARCO-ERA5 (Zarr on GCS) or WeatherBench 2 — the latter also supplies the evaluation harness and baseline scores.
- Static channels, included even in Milestone 1: surface geopotential (orography), land-sea mask, latitude encoding.
- Train 1979–2017, validate 2018, freeze everything, test 2019.
---
 
## Evaluation
 
Primary bar: **RMSE below persistence at 24h.** Persistence is a low bar and if we can't clear it the locality assumption is wrong.
 
Secondary, in order:
1. RMSE vs. climatology at 24h, 48h, 72h.
2. Power spectrum of the forecast field vs. ERA5 — checks for blurring, which is the expected failure mode of deterministic MSE training.
3. Rollout stability out to 7 days. Does it blow up, or drift to a smooth attractor?
Reference points from WeatherBench 2 for context, not for parity.
 
---
 
## Sequence of work
 
1. **Data pipeline.** ERA5 → 1° → icosahedral mesh → batched tensors. Verify by round-tripping a field back to lat-lon and eyeballing it.
2. **Perception operator.** Get gradients on the mesh working in isolation. Sanity check: the operator applied to a known analytic field should recover the known gradient.
3. **Advection-only test.** Before any training, hand-set the update rule to pure advection with a constant wind and check the field translates cleanly. This isolates the CFL question from the learning question.
4. **Train.** Single 6h step, MSE loss, area-weighted.
5. **Rollout.** Extend to 24h autoregressively. Fine-tune on multi-step rollouts if single-step training doesn't hold up.
6. **Measure and decide.**
---
 
## Known risks
 
**CFL / advection speed.** The main one. If ~10 sub-steps per 6h isn't enough, the model will smear fast-moving features. Mitigations, in order of preference: more sub-steps, larger perception radius, then a semi-Lagrangian pre-advection step outside the NCA.
 
**Blurring.** Deterministic MSE will produce over-smooth fields at longer leads. Expected. This is the motivation for going probabilistic in Milestone 2 — don't over-engineer around it here.
 
**Rollout drift.** Small per-step biases compound. Watch for a slow collapse toward the mean state.
 
---
 
## Exit criteria
 
Move to Milestone 2 when we have: 24h RMSE below persistence, a stable 7-day rollout, and a documented answer on how many sub-steps per forecast step the CFL constraint actually demands.
 
Milestone 2 is the full prognostic state, 13 levels, 0.25° fine-tuning, and CRPS training with a latent noise vector for ensembles.