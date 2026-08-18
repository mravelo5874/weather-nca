# What this project is, and how it differs

Written 2026-08-18, mid-milestone-2. Numbers are measured unless marked otherwise.

---

## 1. The bet

Every competitive ML weather model moves information **non-locally** in a single forward pass.
This project asks whether that is necessary.

| model | mechanism | information reach per 6 h step |
|---|---|---|
| GraphCast | multi-scale GNN over an icosahedral mesh | global (coarse mesh levels span continents) |
| Pangu-Weather | 3D Earth-specific transformer | global (attention) |
| FourCastNet 3 | spherical Fourier neural operator | global (spectral transform) |
| GenCast | diffusion over a graph backbone | global, plus iterative refinement |
| AIFS-CRPS | encoder–processor–decoder GNN | global |
| **this project** | **strictly local rule, iterated** | **`n_substeps` cells — currently 20** |

The claim under test is that **weather is a PDE, and PDEs are local**. Advection, pressure
gradients, Coriolis, diffusion — every term in the primitive equations is a differential
operator acting on an infinitesimal neighbourhood. Global structure emerges from iterating a
local rule, not from a rule that sees everything at once.

If true, the local restriction is not a handicap but the correct inductive bias, and it should
buy sample efficiency and physical consistency. If false, the restriction costs accuracy and we
should know by how much.

**Phase 2d exists to answer exactly that** — a non-local GNN matched on mesh, variables, data,
optimizer, parameter count and wall-clock, so the only thing that varies is locality. It has
not run yet. **The central question of the milestone is still open.**

---

## 2. Where the NCA framing actually enters

A Neural Cellular Automaton is a learned local update rule applied identically at every cell
and iterated. Here that means:

```
state:      [ 28 physical channels | 32 hidden channels ]   per mesh node
perception: [ identity | ∇x | ∇y | ∇² ]  →  4 × 60 = 240 dims
update:     MLP(perceived ‖ conditioning) → residual delta,   dt = 0.05
forecast:   repeat the update n_substeps = 20 times = one 6 h window
```

Four properties do real work:

**Weight sharing across space.** One rule governs the entire globe — the tropics and the poles
run identical parameters. This is the mesh analogue of convolutional weight sharing, and it is
why the model is 1.03 M parameters against GraphCast's ~37 M.

**Locality by construction.** The perception stencil is one hop. The model *cannot* learn a
teleporting shortcut, so any long-range skill it shows is genuinely emergent from iteration.

**Depth without parameters.** Twenty sub-steps of the same rule is a 20-layer network with tied
weights. Capacity in *time* is free; capacity in *parameters* is not.

**Sub-steps are PDE sub-steps, not layers.** `dt × n_substeps` is the integration length of one
forecast window, and the CFL condition is a real constraint on it. This is the project's most
error-prone concept — the codebase enforces three distinct granularities (`nca_step`,
`forecast_step`, `rollout`) and CLAUDE.md leads with the warning that confusing them is the
most common mistake.

### Departures from classical NCA

Growing-NCA work (Mordvintsev et al.) uses a stochastic update mask, a learned Sobel-like
perception on a 2D grid, and no external conditioning. This project changes all of that:

- **deterministic update** (p = 1) — every cell updates every sub-step
- **geometric perception** — the cotangent Laplace–Beltrami operator and a least-squares tangent
  gradient, computed once from mesh geometry and never trained. A naive flat-space finite
  difference does not converge on a curved surface; M1 tried it and the error *grew* under
  refinement
- **second-order Markov conditioning** — the tendency `x_t − x_{t−1}` is supplied, so the rule
  sees velocity rather than only position
- **hidden channels re-seeded between windows**, because single-step training never presents a
  non-zero hidden state at the start of a window. Carrying it forward feeds the MLP inputs it
  was never trained on, and that mismatch — not blurring, not CFL — was M1's blow-up

---

## 3. Where the training differs

The architecture is the visible difference; the training regime turned out to matter more.

**Pushforward is the single largest effect measured in this project.** Unroll one window under
`no_grad`, then take the gradient step from *that* state, so the rule learns to contract error
from states it actually produces rather than only from clean ERA5 analyses. Measured at 2 years,
28 variables:

| lead | single-step MSE | + pushforward |
|---|---|---|
| 24 h | 297.6 | **282.1** |
| 168 h | 1862.0 | **1308.7** |
| 360 h | 3450.6 | **1860.5** (−46%) |

This is more consequential for an *iterated local* model than it would be for a single-pass
global one: a rule applied 20 × 60 = 1,200 times over a 15-day rollout compounds its own
error far more aggressively than a model applied 60 times.

**The probabilistic head perturbs the rule, not the state.** A 16-dimensional noise vector FiLM-
modulates every hidden layer of the update MLP, drawn once per member per trajectory and held
constant across all cells and sub-steps. Seed perturbations are exactly what 20 rounds of a
smoothing local operator are built to erase — ensemble collapse would be the *expected* outcome
of perturbing the state. This follows FGN / WeatherNext 2, and the low dimensionality is
load-bearing: it forces the perturbation into coherent spatial structure rather than per-cell
noise. Phases 3a/3b, not yet run.

**Fair CRPS, not the naive estimator.** At M = 4 members the naive estimator over-states CRPS by
~60% and under-credits spread, training toward over-confidence. Verified against the analytic
CRPS of a standard normal at four ensemble sizes.

---

## 4. Honest positioning against the frontier

**This is not a competitive forecast system and is not trying to be.**

| | 24 h z500 RMSE (m²/s²) | params | training data |
|---|---|---|---|
| GraphCast-class frontier | ~50–60 | ~37 M | 39 yr @ 0.25° |
| this project, best so far | **282** | 1.03 M | 2 yr @ ~2° |
| persistence | 590 | — | — |

Roughly **5× off the frontier**, on 1/36th the parameters, 1/20th the data, and ~8× coarser
resolution. The M1 plan's "6× off" framing also mislabelled the mesh: `n_sub = 5` is ~223 km
spacing, which is nearer **2°** than the 1° originally claimed, so part of the gap is a
resolution mismatch larger than assumed.

The deliverable is **a slope and a controlled comparison**, not a leaderboard position:

- does accuracy improve with data and coupling, or plateau?
- what does locality cost against a matched non-local control?

---

## 5. The questions, and their status

| # | question | phase | status |
|---|---|---|---|
| 1 | Can a strictly local rule advect at all? | M1 | ✅ yes — 24 h RMSE 344.7 vs persistence 593.9 |
| 2 | Does more data help, or is 2 years enough? | 2a | ✅ **−30%** at 24 h from 39 years, not yet saturated |
| 3 | Does coupling 28 variables help? | 2b | ⚠️ **yes to ~53 h, then worse** — and it destabilised long leads |
| 4 | Is the long-lead failure numerical (CFL)? | 2b′ | ❌ **no** — doubling sub-steps did not help and costs 2× |
| 5 | Is it the training objective? | 2b-pf | ✅ **yes** — pushforward cut 360 h error 46% |
| 6 | Do data volume and coupling compound? | **2c** | ⏳ **running now** |
| 7 | **What does locality cost vs a matched non-local model?** | 2d | ⛔ **not started — the central question** |
| 8 | Can it produce a calibrated, sharp ensemble? | 3a/3b | ⛔ not started |

---

## 6. Two findings that generalise beyond this architecture

**Perturbation growth and forecast error decouple.** The model with the *best* forecasts at
every meaningful lead has the *worst* score on the milestone's stability criterion. Perturbation
growth measures sensitivity to initial conditions — a Lyapunov exponent — while RMSE measures
total error. The earlier models were not stable, they were **over-damped**: error-doubling times
of 5.3 d (phase 0) and 2.9 d (2b) against the real atmosphere's 1.5–2.5 d. Pushforward removed
systematic drift and made the dynamics genuinely chaotic, landing at 1.3 d.

Consequence: exit criterion 5's "growth ≤ 1.05 per window" rewards over-damped models and,
applied literally, would select the worst forecaster in the ladder. A proposed replacement —
*bounded* (error saturates near climatology) plus *physical* (doubling time 1.5–3 d) — is in the
findings document awaiting a plan decision.

**Offline proxies mislead; direct diagnostics do not.** `n_substeps` was doubled on the strength
of an advection sweep showing a *fixed linear* operator overshooting 5.3% per window at jet
speed. The trained model learns the total per-window map, and since `dt × n_substeps` is pinned,
discretising that map more finely cannot change its amplification. The proxy was real and
irrelevant. This is the third time in the project's history that a clever measurement misfired
where `perturbation_growth` — two rollouts from nearly identical starts — was right.

---

## 7. Where it goes next

Beyond M2: mesh refinement toward 100 km and then ~28 km (with corrected node-count arithmetic
— 0.25° needs ~650k cells, not the 49k–163k the v1 plan assumed), the full 13-level state, and
calibrated extremes.

The original architectural bet, which the mesh design was chosen to permit, is **Lagrangian
feature particles layered on the Eulerian base** — the Neural Particle Automata direction. A
strictly local Eulerian rule is the substrate for that, not the destination.
