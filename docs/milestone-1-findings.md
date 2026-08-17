# Milestone 1 — Findings from the MVP scaffold

Recorded while building `claude/weather_nca_mvp.ipynb`. Two of the plan's open questions
now have preliminary, numerically-verified answers. Neither involved any training.

---

## 1. The perception operator: cotangent Laplacian, not finite differences

**Finding: a flat-space finite-difference Laplacian does not converge on the sphere.**

The first attempt used the standard irregular-point-cloud stencil
`∇²f ≈ (4/Σ|d|²)·Σ(f_j − f_i)` with neighbour offsets projected onto the local tangent
plane. Relative error against the analytic result **grew** with mesh refinement:

| n_sub | N | flat-FD Laplacian rel-err |
|---|---|---|
| 3 | 642 | 7.6e-01 |
| 4 | 2562 | 1.2e+00 |
| 5 | 10242 | 1.8e+00 |

The cause is curvature: projecting the chord onto the tangent plane discards a normal
displacement that enters at the same order as the term being measured, so the truncation
error never vanishes.

**Resolution:** use the **cotangent Laplace–Beltrami** operator from discrete differential
geometry, `ℓ_ij = (cot α_ij + cot β_ij)/2A_i`, with barycentric vertex areas. The
least-squares tangent-plane gradient was kept as-is — it is only first-order accurate and
converges fine.

Verified against the degree-1 spherical harmonics `f = x, y, z` (surface gradient
`â − f·r̂`, Laplace–Beltrami exactly `−2f`):

| n_sub | N | grad rel-err | lap rel-err |
|---|---|---|---|
| 3 | 642 | 1.13e-02 | 2.03e-02 |
| 4 | 2562 | 4.38e-03 | 1.01e-02 |
| 5 | 10242 | 1.61e-03 | 5.05e-03 |
| 6 | 40962 | 5.81e-04 | 2.51e-03 |

Gradient converges at ≈O(h^1.5), Laplacian at ≈O(h). This is the plan's step-2 sanity
check, passing. It lives in the notebook as `operator_convergence()` and should be re-run
whenever the mesh code is touched.

---

## 2. CFL budget: ~10 sub-steps per 6 h window is defensible at 1°

Plan step 3, run *before* any training: the update rule was hand-set to pure zonal
advection (`∂f/∂t = −u ∇ₓf`, U = 30 m/s, explicit Euler) and a Gaussian bump was advected
one 6 h window, sweeping the sub-step count. Error is against the analytically translated
bump.

**At n_sub = 5 (10 242 nodes, ~223 km spacing — the Milestone 1 mesh):**

| sub-steps | dt | CFL | RMSE |
|---|---|---|---|
| 1 | 21600 s | 2.90 | 0.0089 |
| 2 | 10800 s | 1.45 | 0.0047 |
| 5 | 4320 s | 0.58 | 0.0021 |
| **10** | **2160 s** | **0.29** | **0.0014** |
| 20 | 1080 s | 0.15 | 0.0011 |
| 40 | 540 s | 0.07 | 0.0010 |

**Error has essentially plateaued by 10 sub-steps** (CFL ≈ 0.29); going to 40 buys a
further ~30% at 4× the cost. This supports the plan's default of `n_substeps = 10` and
suggests the CFL constraint is **not** the binding risk it was flagged as — at least for
advection at jet speeds on a 1° mesh.

Two caveats before treating this as settled:

- It tests **one 6 h window**, not a rollout. Per-step advection error that looks small
  here still compounds over a 7-day rollout.
- Real z500 has faster and sharper features than a 0.35 rad Gaussian bump. A sharper
  initial condition will demand more sub-steps.

The honest version of the exit criterion — sub-steps needed for a *trained* model on
*real* data over a *full rollout* — still needs the trained model.

---

## 3. The first full run: an autoregressive failure, not a locality or CFL failure

Run configuration: synthetic data (`use_real_era5 = False`), `n_sub = 5` (10 242 nodes),
`n_substeps = 10`, `c_hidden = 15`, 8 single-step epochs + 3 rollout-fine-tune epochs,
GTX 1660 Ti, ~42 s/epoch.

### What happened

Single-step training converged cleanly (train 0.00105, val 0.00108) and the model is
excellent at one step. It then falls apart autoregressively:

| lead | NCA | persistence | climatology | skill vs persist |
|---|---|---|---|---|
| 6 h | 5.7 | 81.6 | 160.0 | **+93.0%** |
| 12 h | 61.8 | 149.4 | 160.0 | +58.6% |
| 24 h | **376.0** | **226.4** | 159.9 | **−66.1%** |
| 48 h | 3 125 | 261.7 | 159.8 | −1094% |
| 72 h | 2.17e6 | 262.7 | 159.5 | — |
| 168 h | inf | 222.1 | 158.6 | — |

Milestone 1 primary bar: **FAIL**. Spatial-std ratio against truth at 168 h: 2.35e5.

Error grows by roughly **×8 per 6 h window** (5.7 → 61.8 → 376 → 3125), which is a clean
exponential, not accumulating bias.

### Diagnosis

`WeatherNCA.seed` initializes `hidden = 0`. Single-step training uses `n_out = 1`, so every
forward pass starts from `hidden = 0` and discards the resulting hidden state. **Across all
8 epochs the update MLP never saw a non-zero hidden channel at the start of a forecast
step.**

At inference `rollout` carries the full state forward, so from window 2 onward 15 of the 16
input channels are values the network was never trained on — unbounded, unsupervised, and
fed through 10 more residual updates per window. Near-perfect at +6 h, catastrophic from
+12 h is exactly that signature.

This is a structural train/inference mismatch, not a statistical generalization gap.

### Two diagnostics that were misread, and are now fixed in the notebook

- **The spectrum check.** `power_spectrum_check` reported "high-wavenumber power retained:
  1441%" under a heading about blurring. The check was written one-sided: it only tests for
  the forecast falling *below* truth. 1441% is a 14× **excess**, present at every wavenumber
  including k = 1 — amplitude blow-up plus grid-scale noise, the opposite failure. The
  message now names both directions.
- **The training curve.** The rollout-fine-tune epochs sit ~200× above the single-step ones,
  which makes the fine-tune look like the cause. It was not: `best` was 0.00108 from phase 1,
  no fine-tune epoch beat it, so no fine-tune checkpoint was ever saved and evaluation ran on
  the *single-step* model. The fine-tune diverged independently and was correctly discarded.

Two real bugs sat behind that divergence:

- **Fine-tune LR.** `lr * 0.3` = 3e-4 while backpropagating through `2 * n_substeps` = 20
  residual updates of an already-amplifying map. `grad_clip = 1.0` bounds the gradient norm
  but AdamW's normalized step still moves the weights a long way per batch. Now
  `rollout_lr_scale = 0.03`, with a 1 → 2 → 4 window curriculum instead of one jump.
- **Checkpoint selection compared incommensurable numbers.** Phase 1 set `best` from a
  1-step val MSE; phase 2 tested a 2-step val MSE against it. "Did this epoch improve?" is
  meaningless across that boundary. There is now one fixed selection metric
  (`cfg.ckpt_windows`-window rollout wMSE on val) used identically in every phase.

### Changes made

| change | config | why |
|---|---|---|
| re-seed hidden channels between windows | `reseed_hidden = True` | makes rollout match the training distribution exactly |
| pushforward noise on the seeded field | `noise_std = 0.05` | trains the rule to contract error, not just to be accurate on-manifold |
| overflow penalty on `\|hidden\| > 1` | `overflow_w = 1e-2` | bounds the latent state (original NCA paper) |
| rollout curriculum 1 → 2 → 4 windows | `rollout_windows` | one jump to 2 windows diverged |
| fine-tune LR ÷ 10 | `rollout_lr_scale = 0.03` | see above |
| single fixed checkpoint metric | `ckpt_windows = 4` | see above |
| error-growth + hidden-RMS diagnostic | §8.1 | separates exponential amplification from drift |

Re-seeding is the cheapest test and needs **no retraining** — re-run evaluation on the
existing checkpoint with `cfg.reseed_hidden = True`. If the diagnosis holds, 24 h should land
in the 30–80 m²/s² range and clear the persistence bar outright. Second-order conditioning
still carries the tendency across windows, so less memory is lost than it looks.

Carrying hidden state across windows is the more genuinely NCA-ish design and is worth
returning to, but it requires training the way you infer: `reseed_hidden = False` **only**
together with the rollout curriculum, the overflow penalty and pushforward noise.

### What this run does *not* justify

**Do not escalate to two-hop perception or a semi-Lagrangian pre-advection step.** Those are
the plan's remedies for a locality failure. 93% skill at 6 h says locality and the perception
stencil are fine, and §2's sweep says the CFL budget is fine. The plan's escalation path is
addressed at a different failure than the one observed.

One smaller item from §2 that the rollout does implicate: the hand-set advection operator has
`max|f| = 1.005` at 10 sub-steps versus 0.999 at 40. The operator itself amplifies slightly
at the current setting. Over 28 windows that is only ~1.15×, nowhere near enough to explain
2.35e5, but it is a known growing mode removable for 2× compute by setting
`n_substeps = 20`.

### Caveat on scope

All of this is on the **synthetic** field — six Gaussian vortices on a solid-body zonal wind
plus a stationary orographic wave. It is deterministic and band-limited (its spectrum hits a
flat noise floor by k ≈ 15; real z500 does not). Conclusions about *stability* transfer to
real ERA5. Conclusions about *skill* do not: persistence being worse than climatology at 24 h
here is an artifact of the synthetic advection speed, not a property of z500.

---

## 4. Run 2: bar 1 passes, bar 2 does not

Same synthetic setup, with the §3 changes applied. **`reseed_hidden` was the fix.**

| lead | NCA | persistence | climatology | skill vs persist |
|---|---|---|---|---|
| 6 h | 5.7 | 81.6 | 160.0 | +93.0% |
| 12 h | 21.4 | 149.4 | 160.0 | +85.7% |
| 24 h | **61.6** | **226.4** | 159.9 | **+72.8%** |
| 48 h | 168.5 | 261.7 | 159.8 | +35.6% |
| 72 h | 516.5 | 262.7 | 159.5 | −96.6% |
| 168 h | 60 868 | 222.1 | 158.6 | — |

Milestone 1 primary bar: **PASS** (61.6 vs 226.4). Head weights alive (phys row norm 0.239,
hidden 0.102); hidden RMS grows 0.092 → 0.361 over 72 h and stays under the overflow
threshold. Crosses climatology at ≈46 h; useful to ≈36 h. Variance ratio at 168 h: 575, so
**the 7-day stability criterion still fails.**

### The residual instability is a single constant-rate mode

Per-window growth factors: ×3.75 (6→12 h), then 1.70, 1.28, 1.32, 1.30, 1.37. After the
first window it settles at **≈1.31× per window and never saturates** across seven days. The
large first ratio is not a separate effect — it is 5.7 being an unusually good single-step
number, not 21.4 being bad.

A geometric mean over all windows hides this. §8.1 now prints per-window ratios and a mean
excluding window 1; the earlier all-window mean was the wrong summary for a curve that is
either saturating or front-loaded.

### Why the synthetic field is worth exploiting before switching to ERA5

The synthetic data is deterministic periodic advection, so its **Lyapunov exponent is zero**
and a perfect model would show flat error at all leads. Every bit of the 1.31 is therefore
the operator's own amplifying mode, with no predictability limit mixed in. On real ERA5 the
two are inseparable. Measure the mode here, fix it here, then move.

### Changes made for run 3

| change | config | why |
|---|---|---|
| curriculum extended to 8 windows | `rollout_windows = (2, 4, 8)` | the objective had never seen past 24 h — exactly where it diverges |
| selection metric extended | `ckpt_windows = 8` | a 4-window metric selects models that are excellent to 24 h and diverge after |
| gradient checkpointing on the sub-step loop | `grad_ckpt = True` | 8 windows = 80 sub-steps; ~n_substeps less activation memory on 6 GB |
| spectral-radius power iteration | §8.1 | see below |
| checkpoint-loading cell with a `head |W| > 0` assert | §8.0 | see the incident below |

`n_substeps = 20` is still worth testing separately: §2 gives `max|f|` = 1.005 at 10 vs 1.001
at 20, a confirmed if small contributor, removable for 2× compute.

### The diagnostic that decides the next lever

`spectral_radius` power-iterates the Jacobian of one forecast window (including re-seeding,
since that is the map actually iterated), by finite differences.

- **λ ≈ 1.3, matching the observed tail** → the growth is linear. Attack it directly with
  Lipschitz control: spectral norm on the update MLP body, or a smaller `dt`.
- **λ ≈ 1.0 while the rollout still grows** → the growth is nonlinear and trajectory-
  dependent. Only rollout training at the failing lead, or explicit damping, will help.

Without this, "is 1.31 too fast?" is a judgment call. With it, it is a measurement.

### Incident: three runs were evaluated on an untrained model

Between §3 and this run, three evaluation rounds were reported on a zero-init model. Cell 19
(`model = WeatherNCA(...)`) rebuilds the model as an exact identity map, and re-running it
after training — or restarting the kernel and jumping to §8 — silently discards the trained
weights.

The identity map reproduces persistence exactly. The tell was subtle in the plots and
obvious in the table: NCA RMSE equal to persistence **to the decimal at all seven leads**,
skill exactly 0.0% rather than approximately zero, and spatial std a flat line at the initial
field's value. §8.1 was over-interpreted twice before the head-norm check caught it —
"saturating error growth, stability fixed" was persistence decorrelating, and the ×1.11
growth was a property of the synthetic field, not of any operator.

§8.0 now loads `best.pt` explicitly, asserts `head |W| > 0`, hard-fails on architecture
mismatch, and prints the trained-vs-evaluating values of the behavioural fields (which are
deliberately *not* enforced, since a `reseed_hidden` mismatch is a valid ablation).

Two process notes: `best.pt` is a single fixed path, so each run clobbers the last — add a
phase or timestamp before the next set of comparisons. And any cell downstream of cell 19
must either follow training in the same session or come through §8.0 first.

---

## 5. Run 3: Milestone 1 bar 1 cleared with margin; the fix was `dt`, not the curriculum

Config change from run 2: `dt = 0.05`, `n_substeps = 20` (same total integration per window,
gentler per update), plus the 8-window curriculum and `ckpt_windows = 8`.

| lead | run 2 | **run 3** | persistence | skill vs persist |
|---|---|---|---|---|
| 6 h | 5.7 | 19.3 | 81.6 | +76.4% |
| 12 h | 21.4 | 33.2 | 149.4 | +77.7% |
| 24 h | 61.6 | **56.0** | 226.4 | **+75.3%** |
| 48 h | 168.5 | 88.8 | 261.7 | +66.1% |
| 72 h | 516.5 | 108.2 | 262.7 | +58.8% |
| 120 h | 4 434.9 | 132.2 | 188.4 | +29.9% |
| 168 h | 60 868 | 148.6 | 222.1 | +33.1% |

**Skill stays positive at every lead out to 168 h**, bottoming at ~21% near 130 h. Error
saturates at 148.6, still under climatology (158.6) — run 2 crossed climatology at 46 h and
this one never does. Note the trade: 6 h got 3.4× *worse* while everything past 12 h improved
by up to 400×. Single-step accuracy was being bought at the cost of an amplifying operator.

### `dt` did the work

Finite-amplitude perturbation growth, random physical direction:
1.45, 0.98, 1.01, 0.96, 1.00, 0.99, 1.06, 0.99, 1.06, 0.95, 1.02 — **≈1.00 per window**,
against ≈1.27 in run 2. The operator went from amplifying to neutrally stable. The
power-iteration "eigenvector" gives the same ≈1.00, confirming again that no preferred
direction exists.

Forecast *error* still grows (1.69, 1.42, 1.31, 1.20, 1.13, ... 1.04; mean 1.138 excluding
window 1) while *perturbations* do not. Those are consistent, not contradictory: error growth
is now dominated by accumulating **systematic** error — the blurring drift below — rather than
by amplification of small differences. The concave log-axis curve is saturating drift, which
is the shape you want.

### The curriculum contributed nothing, twice

Best selection metric was **0.10749 at epoch 8 of single-step training**. Every fine-tune
epoch scored worse: 2-window bottomed at 0.139; 4-window ran 0.134 → 0.399 → 4.19; 8-window
ran 2.66 → 565 → 27 094. No fine-tune epoch saved a checkpoint. **The evaluated model is
single-step-trained only.**

The fixed selection metric (§3) is what made this safe — it caught a diverging phase and
refused to promote it, which is precisely the failure that silently wrecked run 1.

The 8-window stage held 1-step val flat at ~0.011 while destroying the 48 h metric by five
orders of magnitude: it degrades long-range behaviour while preserving short-range. Backprop
through 160 sub-steps at lr 3e-5 is still too aggressive. `rollout_epochs` now defaults to 0;
re-enable only with `rollout_lr_scale` another order of magnitude down, and treat the ~13 min
it costs as a bet, not a default.

### Bar 3 answered, and the answer changed

§2's single-window sweep said 10 sub-steps was the plateau. The rollout disagrees: **20
sub-steps at `dt = 0.05`** is what makes the operator neutrally stable. A one-window CFL test
cannot see a 1.005 per-window amplification that compounds over 28 windows. Treat §2's number
as a lower bound and the rollout perturbation test as the real criterion.

### Remaining failure: blurring, which is Milestone 2's problem

Spatial std drifts *down* to ~0.65 against truth's ~1.0, and the power spectrum sits 2–4×
below truth across wavenumbers 5–15 — the scales carrying the vortices. Maps confirm it: at
+48 h the features are in the right places but softer than truth.

This is the expected consequence of deterministic MSE training, which provably yields the
conditional mean. It is exactly what CRPS training with a latent noise vector exists to fix.
Do not engineer around it in Milestone 1.

One thing that is *not* blurring: visible **diagonal streaking** in the NCA panels at +24 h
and +30 h, aligned with the advection direction. Blurring is isotropic; this is structured.
Not blocking, but worth a look — plausibly the perception stencil's gradient operators
interacting with the zonal wind.

### Diagnostic retired

`spectral_radius` is removed. Its eps sweep reads 6.5 / 36.3 / 290 at eps 1e-2 / 1e-3 / 1e-4
— scaling inversely with step size, the signature of a finite-difference noise floor rather
than a property of the operator. It reported 39.2 in run 3 and 53 in run 2; both were
meaningless, and the run-2 number generated a whole false story about a latent mode in the
hidden channels. `perturbation_growth` measures the same quantity by running the model and
watching, needs no linearization, and has been right every time.

Standing lesson: three diagnostics were built for this failure, two misfired and pointed
confidently the wrong way, and the one that worked was the one that just ran the model and
looked at the output. Prefer direct measurement over clever measurement.

### Status against the Milestone 1 exit criteria

| bar | status |
|---|---|
| 24 h RMSE below persistence | **PASS** — 56.0 vs 226.4 (+75.3%) |
| stable 7-day rollout | **PASS on stability** — no blow-up, perturbation growth ≈1.00, hidden RMS flat at ~0.08. Fails on fidelity: variance decays to 0.65 of truth. |
| documented sub-step budget | **PASS** — 20 sub-steps at `dt = 0.05`, criterion revised (see above) |

All of it on the **synthetic** field: deterministic, periodic, band-limited, zero Lyapunov
exponent. These transfer as hypotheses, not measurements. The untested step is
`cfg.use_real_era5 = True` — real z500 has a genuine predictability limit, a power-law
spectrum with no noise floor, and sharper features than a 0.35 rad Gaussian, which is the
case §2 flagged as likely to demand still more sub-steps. Expect worse numbers and more
meaningful diagnostics.

---

## 6. Corrupted ERA5, then the real pass: an axis-order bug and the first honest numbers

### The transpose bug

The first real-data run produced diagonal-stripe "z500" fields and a sawtooth power
spectrum — alternating high/low power at every consecutive wavenumber across four orders of
magnitude. Both are signatures of a fixed-stride indexing error, not weather.

Cause: `regrid_to_mesh` and `latlon_to_mesh_weights` assume the grid is flattened row-major
as `lat*n_lon + lon`, i.e. dimension order `(..., latitude, longitude)`. The WeatherBench-2
zarr served the field as `(time, longitude, latitude)`. Reading one layout as the other
shears each row by a constant offset, which is exactly a diagonal stripe.

Fix: `load_era5` now transposes every field — z, orography, land-sea mask — to
`(time, latitude, longitude)` explicitly, making the contract with the regridder
unconditional rather than store-dependent. A `cfg.plot_raw_era5` sanity plot draws the first
native-grid frame before regridding, and the cache key was bumped to `_v2` so the corrupted
`.npz` is bypassed rather than silently reused.

The lesson is cheap and general: **plot the raw field once, on its native grid, before it
enters the pipeline.** Ten seconds would have caught this before a 45-minute training run.
Every downstream number from the corrupted run — skill, perturbation growth, spectrum — was
measuring the model's ability to fit an aliasing artifact and is void.

### The real pass

Corrected data, same model, 2 training years (2015–2016), test 2018:

| lead | NCA | persistence | climatology | skill vs persist |
|---|---|---|---|---|
| 6 h | 102.5 | 229.0 | 1068.4 | +55.2% |
| 12 h | 178.4 | 367.7 | 1071.0 | +51.5% |
| 24 h | **344.7** | **593.9** | 1072.7 | **+42.0%** |
| 48 h | 657.7 | 830.5 | 1077.4 | +20.8% |
| 72 h | 889.9 | 941.7 | 1074.5 | +5.5% |
| 120 h | 1172.6 | 1005.6 | 1069.7 | −16.6% |
| 168 h | 1367.2 | 1041.0 | 1066.6 | −31.3% |

**Milestone 1 primary bar: PASS** (344.7 vs 593.9). Positive skill holds to ~85 h; crosses
climatology at ~100 h.

### All three bars cleared, and the diagnostics agree for the first time

- **Bar 1 — beat persistence at 24 h:** PASS, +42% on real held-out data.
- **Bar 2 — stable 7-day rollout:** PASS. Perturbation growth starts at 1.63 and decays
  monotonically to 1.04 by 72 h — a settling curve, not an exponential. Hidden RMS flat at
  0.135, far under overflow. `dt = 0.05, n_substeps = 20` transferred from synthetic to real
  intact, which was the thing most likely to break.
- **Bar 3 — sub-step budget:** documented at 20 sub-steps / `dt = 0.05`, now validated on a
  field with genuine fine structure rather than a band-limited toy.

On synthetic data, error growth and perturbation growth *disagreed* — error grew while
perturbations didn't, because the growth was systematic blurring. On real data they *agree*:
error growth ~1.18/window, perturbation growth settling to ~1.04, spectrum sitting on top of
truth (100.9% retained at 24 h). That agreement is the signal that the remaining error is
genuine atmospheric divergence along the fronts — physics — rather than a model pathology.
The error maps concentrate exactly there, which is the correct place for a z500 forecast to
be wrong.

### Where it stands against the frontier

M1 was scored only against persistence and climatology, which it clears. Against the
WeatherBench-2 frontier, deterministic z500 RMSE at 24 h runs ~50–60 m²/s² (GraphCast beats
IFS HRES across nearly all leads; GenCast / U-Cast post z500 CRPS ~20 at 1 d against IFS
ENS's 22.4). At 344.7, this model is a factor of ~6 off.

That gap is expected and almost entirely attributable to things Milestone 2 addresses: two
training years against the frontier's ~forty, one variable against a coupled state, 1°
against 0.25°, deterministic MSE against CRPS. None of it is evidence the architecture is
wrong. A 28k-parameter local rule producing stable, spectrum-preserving 48 h z500 forecasts
on real data is a legitimate proof of concept that the iterated-local-PDE inductive bias does
real work. Whether it closes the 6× gap or plateaus at 2–3× is genuinely unknown and is the
question Milestone 2 exists to answer.

### Standing methodology note

Three diagnostics were built for the rollout instability across M1. Two — the one-sided
spectrum check and the finite-difference spectral radius — misfired and pointed confidently
the wrong way; the spectral radius sat on the float32 noise floor and generated an entirely
false story about a hidden-channel mode. The one that was right every time was
`perturbation_growth`, which just runs the model twice and watches the gap. **Prefer direct
measurement over clever measurement.** It is the single most useful thing this milestone
taught, and it is the diagnostic Milestone 2 should reach for first.

### Persistence is retired

It was cleared decisively on real data and is no longer informative. From Milestone 2 on, the
only baseline worth plotting is the published WeatherBench-2 z500 RMSE for the test year, so
every run reads as distance-to-frontier rather than distance-to-a-weak-baseline.

---

## 7. Notes for whoever picks this up

- Normalization statistics are computed from the **train split only**; climatology for the
  RMSE baseline likewise comes from train, not from the split being scored.
- Reported RMSEs pool MSE across start times and take the square root at the end. Averaging
  per-start RMSE instead is biased low by Jensen and is not comparable to published
  WeatherBench numbers.
- Area weighting uses the mesh's barycentric vertex areas. On an icosphere these are nearly
  uniform — which is the point of using it instead of a lat-lon grid.
- **The notebook has been executed end-to-end on real ERA5** through the Milestone 1 pass
  (§6). The spectral-radius cell was removed as unreliable; `perturbation_growth` replaces
  it. `load_era5` now transposes to `(time, latitude, longitude)` explicitly — this is
  load-bearing, and the `cfg.plot_raw_era5` sanity plot should stay in.
- Test year for Milestone 1 was 2018 on 2 training years. Milestone 2 moves the test year
  to 2020 to line up with the WeatherBench-2 probabilistic leaderboard.
- The stored `.ipynb` had `source` arrays whose lines were not newline-terminated, so the
  cells collapse to a single line when opened. Fixed in place; worth checking if the file
  passes through any tooling again.

> **Correction (2026-08-16, M2 port).** This document originally said "100k-parameter".
> The M1 configuration (`c_hidden=15, hidden_dim=128, n_layers=2`) gives **27,536** parameters,
> confirmed against the notebook's own `parameters: 27,536` output. The figure has been corrected
> above and in `milestone-2-plan.md`, where it was being used as a premise.
