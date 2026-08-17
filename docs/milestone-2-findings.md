# Milestone 2 — Findings

Recorded as phases complete. Everything here is measured on the local GTX 1660 Ti (6 GB) unless
stated otherwise; re-measure on the cloud instance before budgeting.

Status: **phase 0 ✅ · 2a ✅ · 2b ✅ · 2b′ pending · 2c–5 not started.**

---

## 0. The port reproduces M1, and the baselines prove it

**Phase 0 result: 24 h z500 RMSE 325.3 against M1's 344.7** — 5.6% *better*, and therefore
outside the ±2% gate on the good side.

The gate was written to catch a broken migration, and the evidence that the migration is sound
is stronger than the headline number: **persistence and climatology match M1 to every printed
digit at all seven leads.** Persistence depends only on the data path — ERA5 ingest, the axis
transpose, bilinear regrid, train-only normalization, start-time selection, and the
pooled-before-square-root RMSE convention. Any drift there would move it. It did not move
anywhere.

Supporting checks: mesh operators converge monotonically under refinement; the sparse operator
assembly matches the scatter-add definition to 1e-12; area-weighted MSE is **bit-identical** to
M1's formulation (max difference exactly 0.0 over 28 leads); parameter count confirmed against
the M1 notebook's own output.

Two known divergences explain the model difference, neither avoidable:

1. **Initialization.** `UpdateRule` allocates the FiLM projection before the head, consuming
   different RNG draws. Bit-exact reproduction stopped being possible once the noise pathway
   existed.
2. **LR schedule granularity.** M1 stepped `CosineAnnealingLR` per epoch; the port decays per
   step. M1's selection metric oscillated (0.037, 0.028, 0.033, **0.023**, 0.032, 0.040, 0.028,
   0.027 — best at epoch 4) while the port descended monotonically to 0.02202 at epoch 7. The
   port simply found a better model.

Full reasoning in `decisions/0004-phase0-gate-outcome.md`.

### 0.1 M1's parameter count was 27,536, not "~100k"

The M1 findings said "a 100k-parameter local rule", and the M2 plan inherited the figure and
used it as a premise. The M1 configuration (`c_hidden=15, hidden_dim=128, n_layers=2`) gives
**27,536** parameters, confirmed against the notebook's own `parameters: 27,536` output.

The plan also justified raising capacity with "M1's ~100k parameters was a 6 GB-VRAM
constraint". Both halves were wrong: measured, the exact M1 configuration peaks at **0.37 GB**
— 6% of the card. VRAM was never the constraint.

The conclusion survives on the argument that holds: 28 physical + 32 hidden channels make the
perception output 240-dimensional against phase 0's 69, so capacity should scale with the
state. But the jump to 1–5M is **36–180× from 27.5k**, not the ~10–50× the old figure implied.

---

## 1. Data volume is worth ~30%, and it is not close to saturated

**Phase 2a (39 years, 1 variable) against phase 0 (2 years, 1 variable), both on test 2020:**

| lead | phase 0 | phase 2a | change |
|---|---|---|---|
| 6 h | 111.4 | 76.4 | −31% |
| 12 h | 168.9 | 122.6 | −27% |
| **24 h** | **327.4** | **229.2** | **−30%** |
| 48 h | 635.3 | 462.8 | −27% |
| 72 h | 889.2 | 688.1 | −23% |
| 120 h | 1242.1 | 1036.5 | −17% |

Skill over persistence at 24 h went 45.0% → **61.5%**, and the model now beats climatology out
to 120 h (was ~100 h).

**The run was deliberately capped at 3 epochs** so that data volume was not confounded with
training budget: 39 years is 7,117 batches/epoch against phase 0's 365, so 12 epochs would have
been 29× the gradient steps. 3 epochs is 7.3×. The selection metric was still falling ~11% per
epoch at the end, so **229.2 is a floor, not a converged number.**

### 1.1 A consistency check worth keeping

The 39-year normalizer differs from the 2-year one (z500 std 2735.5 vs 2796.0 — a 2-year window
is a different sample of the climate). Physical RMSE is recovered by multiplying normalized RMSE
by that std, so both runs should report **the same persistence RMSE on the same test year
despite normalizing differently**. They do: 595.4 in both. That validates the normalize/decode
path end to end, and it is cheap to re-check whenever the normalizer changes.

---

## 2. Coupling helps to ~53 h and hurts badly after

**Phase 2b (2 years, 28 variables) against phase 0 (2 years, 1 variable), test 2018, identical
eval settings:**

| lead | phase 0 | phase 2b | change |
|---|---|---|---|
| 24 h | 329.7 | **297.6** | **−9.7%** |
| 48 h | 639.1 | **624.0** | −2.4% |
| 72 h | 881.1 | 934.1 | +6.0% |
| 168 h | 1493.3 | 1862.0 | +24.7% |
| 360 h | 2208.5 | **3450.6** | **+56.2%** |

**Crossover at ~53 h.** At 15 days phase 2b sits at 3.2× climatology, which is active
divergence rather than graceful decay toward it. **Exit criterion 5 (stable to 15 days) fails.**

The short-lead gain is broad rather than a z500 artifact — every variable beats persistence at
24 h by 34–61%, with v-winds strongest (51–61%), consistent with the geostrophic coupling the
plan predicted.

Two caveats on the comparison. Capacity is **not** held fixed (27.5k → 1.03M params), which is
unavoidable rather than sloppy: a 128-wide MLP would be a bottleneck on a 240-dimensional
perception output, not a control. The honest claim is *"coupling plus the capacity it
requires"*. And 2b's control is **phase 0, not 2a** — the plan's table says "vs 2a", but 2a is
39 yr × 1 var while 2b is 2 yr × 28 var, which moves data volume and state together. The four
phases are a 2×2: phase 0 (2 yr, 1 var), 2a (39 yr, 1 var), 2b (2 yr, 28 var), 2c (39 yr, 28 var).

### 2.1 Root cause A — `n_substeps = 20` is under-resolved at jet speeds

`n_substeps = 20` came from an M1 sweep at **U = 30 m/s**. The 250 hPa jet in the 2b data
reaches **102 m/s**. Re-running M1's advection test — which bypasses the learned MLP entirely,
so it measures the *scheme* rather than the model — at that speed:

| sub-steps | CFL | RMSE (U=102) | max&#124;f&#124; | RMSE (U=30) |
|---|---|---|---|---|
| 10 | 0.99 | 0.01195 | 1.108 | 0.00136 |
| **20** | **0.49** | **0.00644** | **1.053** | **0.00111** |
| 40 | 0.25 | 0.00432 | 1.027 | 0.00103 |
| 80 | 0.12 | 0.00362 | 1.014 | 0.00101 |

At 30 m/s the error plateaus by 30 sub-steps and amplitude is 1.001 — the M1 budget was
justified for what it was measured on. At jet speed the error has **not plateaued even at 80**,
and 20 sub-steps overshoots amplitude by **5.3% per window**.

Three independent lines of evidence point at one mechanism:

- measured sustained perturbation growth is **×1.062** overall and **×1.094** for v_250 — the
  same magnitude as the 5.3% overshoot;
- growth is worst exactly where the winds are fastest (v_250, v_300, u_300);
- phase 0, single-variable with no fast winds, sat at **×1.033**.

Per-variable sustained growth, phase 2b (`* ` = above 1.05):

| variable | 850 | 700 | 500 | 300 | 250 |
|---|---|---|---|---|---|
| geopotential | 1.060* | 1.063* | 1.067* | 1.073* | 1.080* |
| u wind | 1.073* | 1.075* | 1.076* | 1.082* | 1.081* |
| v wind | 1.068* | 1.076* | 1.086* | 1.093* | **1.094\*** |
| temperature | 1.078* | 1.080* | 1.087* | 1.074* | 1.068* |
| humidity | 1.074* | 1.077* | 1.068* | 1.069* | 1.069* |

27 of 28 channels are above threshold; only `2m_temperature` (1.025) is not, and for the wrong
reason — see below.

### 2.2 Root cause B — no solar forcing, so the diurnal cycle is unrepresentable

`2m_temperature` scored **−48% skill at +24 h**: much worse than doing nothing. This was tested
rather than assumed.

| lead | 2t persistence | 2t model | 2t skill | t850 skill (control) |
|---|---|---|---|---|
| 6 h | 2.67 | 2.31 | +13.4% | +24.0% |
| 12 h | 3.61 | 2.92 | +19.1% | +32.3% |
| 18 h | 3.01 | 2.96 | +1.8% | +37.0% |
| **24 h** | **1.95** | 2.89 | **−48.2%** | +38.8% |
| 30 h | 3.31 | 3.40 | −2.8% | +37.1% |
| 36 h | 4.04 | 3.89 | +3.9% | +34.7% |
| 42 h | 3.49 | 4.10 | −17.6% | +32.4% |
| **48 h** | **2.58** | 4.19 | **−62.8%** | +29.1% |

Persistence RMSE for 2t is a clean **sawtooth**, dipping at 24 h and 48 h because those are the
same local solar time. The model's error is monotone — it has no way to know the hour. Static
conditioning was orography, land-sea mask, sin(lat), cos(lat): **nothing time-varying**.
`temperature_850` is the control — smooth, positive 24–39% at every lead, no sawtooth.

This also explains why 2t was the one channel *below* the growth threshold: a model that has
smoothed away the diurnal cycle has little left to amplify.

### 2.3 Perturbation growth is not simply "worse"

Converting sustained growth to error-doubling time:

| run | per 6 h | per day | doubling |
|---|---|---|---|
| phase 0 | 1.033 | 1.139 | **5.3 d** |
| phase 2a | 1.074 | 1.331 | 2.4 d |
| phase 2b | 1.062 | 1.272 | 2.9 d |

The real atmosphere doubles synoptic-scale errors in roughly **1.5–2.5 days**. Phase 0 was
*over-damped* at 5.3 days — too diffusive, consistent with the blurring M1 documented. So part
of the rise from 2a onward is the model becoming more physically realistic, not less stable.

**This matters for exit criterion 5.** Its ≤1.05 threshold was inherited from a weak,
over-smoothed single-variable model, and taken literally it selects *against* realistic
dynamics. Proposed reframing, not yet adopted: keep a stability requirement, but state it as
*bounded error saturating near climatology* plus a doubling time in the 1.5–3 day band. Phase 2b
fails on the first clause (3.2× climatology at 15 days) regardless of which threshold is used,
so nothing hinges on this yet — but phase 3a's CRPS objective will sharpen the model further and
the question will not stay academic.

---

## 3. Bugs found, and what they cost

Recorded because each was silent — no exception, no NaN, plausible-looking output.

1. **Pushforward target misalignment.** After the no-grad window, predictions start at lead +2
   while targets still started at +1, so the model would have trained against the wrong lead
   with every loss curve looking reasonable. Found by reading, not by failure — `pushforward`
   was off by default. Same edit also fixed `prev` being set to the stepped state's own
   physical channels, which zeroed the second-order tendency.
2. **Cache double-normalization on resume.** `normalized` flipped only after *all* splits were
   converted, so a build interrupted inside the pass re-encoded the finished ones. Normalization
   is not idempotent: measured, std goes **1.00 → 2.80**, finite and NaN-free. Worse, the
   normalizer was refitted on partially-normalized data, producing meaningless statistics. Did
   not trigger for 2b (verified mean ≈ 0, std ≈ 1) but the window widens to hours at 2c's 65 GB.
3. **Perturbation diagnostic misreporting.** Reported ×1.087 "AMPLIFYING" for an operator whose
   sustained rate was ×1.033. The summary excluded only window 1 while the settling transient
   ran 4 windows — exactly what M1's own findings warn about ("read the per-window ratios, not
   a geometric mean"). Now the transient and the sustained rate are reported separately.
4. **Comparability slips.** Two evaluations differed only in `eval.max_windows` (28 vs 60),
   which shifts the start-time selection and moved persistence by 0.7%. Baselines that *should*
   be identical are the cheapest possible check that two runs are comparable — persistence
   matching to the digit is what validated both the phase 0 port and the 2a normalizer.

---

## 4. Measured facts

| quantity | value | notes |
|---|---|---|
| M1 architecture | 27,536 params | not ~100k |
| M1 config peak VRAM | 0.37 GB | on a 6 GB card |
| phase 2b architecture | 1,028,156 params | `hidden_dim=512, n_layers=4` |
| phase 2b step (B=8, 20 sub-steps) | 3,722 ms, 2.10 GB | fwd+bwd |
| phase 2b′ step (B=8, 40 sub-steps) | 7,458 ms, 2.49 GB | exactly 2.0× |
| sub-step cost split | ~80% update MLP, ~20% perception | MLP scales with `hidden_dim²` |
| 28-channel 2-year cache | 6.7 GB | ~50 timestep-channels/s to stream |
| 39-year 28-channel cache | ~65 GB | est. ~15 h |
| 250 hPa jet max in data | 102 m/s | vs 30 m/s in the M1 CFL sweep |
| unit test suite | 133 tests, ~11 s | budget is 60 s |

**Things measured and rejected:**

- Fusing the three perception operators into one `[3N, N]` sparse matmul: **slower** (33 ms vs
  20 ms). Permuting the 3×-wider result costs more memory traffic than the saved kernel
  launches. Direct measurement beat the reasoning that motivated it.
- Reducing the selection-metric batch size to save memory: no effect on cost (337 s vs 353 s).
  What actually mattered was subsampling the start times (~4× on that pass).

---

## 5. What changed as a result

| change | driven by | where |
|---|---|---|
| Solar forcing: cos(zenith) + annual cycle as conditioning | §2.2 | `data/forcing.py`, `state.solar_forcing` |
| `n_substeps` 20 → 40, `dt` 0.05 → 0.025 | §2.1 | `configs/phase2b_prime.yaml` |
| Selection metric on a fixed subsample of val | selection cost > training cost | `train.ckpt_subsample` |
| Per-variable perturbation growth reporting | plan requirement at 2b | `eval/perturbation.py` |
| Multi-variable scorecards, log-unit labelling | 28 channels | `eval/metrics.py` |
| Chunk-level resumable normalization | §3.2 | `data/cache.py` |
| `wnca train --resume` | no recovery path existed | `train/phases.py` |

`dt` halves alongside `n_substeps` so that `dt × n_substeps` stays 1.0. Without that, doubling
the sub-step count would also double the effective integration length per window, confounding
"more, smaller steps" with "a longer step".

---

## 6. Open questions

1. **Does 2b′ close it?** Gates: sustained growth ≤1.05 with no single variable above it; 2t
   skill positive at 24 h; long-lead RMSE bounded near climatology.
2. **Is 40 sub-steps enough, or is this the first mitigation of three?** The plan's order is
   more sub-steps → larger perception radius → semi-Lagrangian pre-advection. The advection
   sweep had not plateaued at 80, so 40 may only halve the problem.
3. **How much of the long-lead divergence is CFL and how much is single-step training?**
   `pushforward` is implemented and now correct but has never been run. It is the designated
   mechanism for teaching the rule to contract error from states it actually produces.
4. **What is the converged 2a number?** Still falling ~11%/epoch when the run was cut, so the
   data-volume effect is understated.
5. **Does exit criterion 5's ≤1.05 threshold survive contact with a sharp model?** See §2.3.
