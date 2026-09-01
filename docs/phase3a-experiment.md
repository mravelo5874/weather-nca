# Phase 3a — the probabilistic phase: experiment record

Written **before any 3a run starts** (2026-08-26), so the design, the success criteria and the
rules for reading a failure are on record rather than reconstructed afterwards. Results go in
`docs/phase3a-results.md`.

This pattern earned its place in 2d: the design doc is where the gated-placebo error was caught,
and it is why a null that would otherwise have been reported as "reach is worth 4%" got
retracted instead.

---

## 1. The question

Half of milestone 2 is probabilistic and **none of it has been measured**. Decision 0001's
entire noise architecture — one `z` per member per trajectory, held constant across cells and
sub-steps, injected through a zero-initialised FiLM projection — exists to prevent ensemble
collapse, and has never been run at scale.

> **Does the FiLM noise design produce a calibrated ensemble, or does it collapse?**

Exit criteria 3 and 4 are the only two of five that do not yet hold, and both live here.

### Why this is the phase worth the remaining budget

The deterministic model is finished: ~$50 remains and no affordable run moves 24 h z500 from
174. What is still winnable is the thing a local NCA uniquely offers — **a cheap stochastic
ensemble from a 1.03 M-parameter model.** Same-size deterministic baselines cannot produce an
ensemble at all without retraining. If spread is even roughly calibrated, that is the
differentiator for the write-up.

---

## 2. Design

| | |
|---|---|
| warm start | phase 2c, **arm A** (plain local model, seed 0) |
| architecture | unchanged from 2c — 4 perception groups, `spectral_norm: true` |
| what changes | **the loss only**: fair CRPS over a 4-member ensemble, FiLM noise active |
| ensemble | `m_train 4` · `m_val 16` · `m_test 50` |
| epochs | **16** (double 2c's 8) |
| lr | 1e-4 |
| pushforward | **ON**, mirroring 2c |
| probe split | **2 years** — 365 batches/epoch, 5,840 optimizer steps |
| configs | `phase3a_probe.yaml` (probe) `extends:` `phase3a_crps.yaml` (scale-up) |

### Two decisions taken deliberately, not inherited

**`epochs: 16`.** The FiLM projection is zero-initialised, so the whole noise pathway grows from
nothing during this fine-tune. The previous value of 15 had no measurement behind it; doubling
2c's budget is at least a stated rationale. It is still few enough steps that §5 matters.

**`pushforward: ON`.** Not because pushforward is better in the abstract — it measurably *hurts*
6 h skill (+43% in 2b) — but because **this phase must change one thing.** The checkpoint spent
8 epochs learning to operate on states it generated itself. Turning pushforward off would add
the CRPS objective *and* yank the state distribution back to clean analyses simultaneously. That
is the confound that spoiled 2b′ (solar forcing tangled with sub-step count) and 2d's GNN
control (four variables at once).

Cost is ~8%, not the ~33% it would be at `m=1`: the pushforward step passes `z=None`, so it is a
single deterministic forward that is **not** multiplied by `m_train`, and all members branch
from the state it produces.

### The probe split is contaminated, and the rule that follows

The probe reuses phase 2b's years — train 2015–16 / val 2017 / test 2018 — because that is the
only complete multi-year ERA5 cache on disk, and `cache_tag()` hashes all three splits together
so changing one costs a fresh ~6.3 GB download of the other two.

The cost: **probe test 2018 was 2c's validation year**, and **probe val 2017 sits inside 2c's
train range**. The warm-start model has seen both, so the probe's skill term is mildly
optimistic, and since spread–skill is spread ÷ skill that biases the ratio **up — toward the
0.8–1.25 band being tested for.**

> **A probe pass is not a reportable calibration result.** The quotable spread–skill number comes
> from the scale-up on the clean split (`phase3a_crps.yaml`: train 1979–2017 / val 2018 /
> test 2020). If the probe passes and the budget runs out, report it as a probe with this caveat
> attached — do not promote it.

A probe **failure** is unaffected: the leakage biases toward passing, so a miss here is a miss on
a clean split too, and §5 applies unchanged.

### Two launch-blocking bugs found while auditing this config

3a set only `stochastic` and `noise_dim` and inherited the rest from `base.yaml`. That gave
`spectral_norm: false` against 2c's `true`, so `arch_hash` differed (`5ef20621…` vs
`0f26ab65…`) and `warm_start` — which raises on unexpected keys — would have hit **12
spectral-norm buffer keys**. **Phase 3a could not have started at all.** The same inheritance
silently reverted `weight_decay` to 1e-5, the value 2c measured as ~11,000× too weak.

Both fixed and covered by tests in `tests/test_checkpoint.py`, verified by reverting the fix and
watching the arch test fail. Recorded here because it is the third config in a row whose
unexamined defaults would have cost a paid run.

**Second: the eval path crashes on this phase.** Smoking `phase3a_probe.yaml` end-to-end,
`wnca eval` died with `ValueError: state.solar_forcing is enabled but no forcing was supplied to
the rollout`. `eval/spectrum.py:member_spectra` never passed `forcing=` — it runs only when
`cfg.model.stochastic`, so **3a is the first config that is both stochastic and forced**, and
nothing had ever executed that combination. `evaluate` and `perturbation_growth` both handle
forcing; this one stage did not.

It would have crashed **after** training, i.e. having already spent ~9.5 h of paid instance per
seed. Fixed, with three regression tests in `tests/test_eval_spectrum.py`. Also cleared
`warm_start` on the smoke path, without which no warm-starting phase could be smoked at all —
a 64-dim smoke model cannot load a full-size checkpoint.

---

## 3. Success criteria, pre-registered

**Primary (exit criterion 3):**

1. **Spread–skill ratio in 0.8–1.25** at 24 h *and* 72 h, on the test split at `m_test = 50`.
2. **Zero-noise ablation measurably different from the ensemble mean** — the ensemble must be
   doing something a deterministic run does not.

**Reported with error bars, not as point estimates.** A spread–skill ratio computed over the
start times of one seed is *one correlated observation family* — the same trap as "nine leads
are not nine observations" (`phase2d-results.md` §3). Use a **day-block bootstrap** over start
times, and report the CI alongside the ratio.

**AMENDED 2026-08-26, before any run: one seed, not two.** Decided by the project owner on
budget grounds. Recorded here as a change to the pre-registration rather than applied silently,
because the previous line read "two seeds, non-negotiable" and a plan that quietly relaxes after
the fact is worth nothing.

What it costs. 2d measured seed spread as **arm-specific, not a project constant** — ±0.2%,
±0.8% and ±8.3% across three arms of the same codebase at the same budget — so a single run
cannot say which kind of arm 3a is. And 3a is a new arm whose measured quantity, spread, grows
from a zero-initialised projection over 5,840 steps; a growth process from zero is not obviously
stable across initialisations. The day-block bootstrap does **not** cover this: it resamples
start times *within* one run, and seed variance is *between* runs.

**The reporting rule this forces:**

| outcome | what may be claimed |
|---|---|
| **pass** (ratio in 0.8–1.25) | *"one seed, on a contaminated split, calibrated."* **Not** "the FiLM design calibrates." Unreplicated. |
| **fail, §5 design signature** | Reasonably strong. §5 reads the spread *trajectory* and the FiLM *gradient norm* — mechanism-level evidence that does not rest on one endpoint number. |
| **fail, §5 budget signature** | Weakest case: ambiguous between a bad draw and too few steps. Say so. |

**Both weaknesses of this probe — the contaminated split (§2) and n = 1 — bias toward passing.**
That makes it a decent *screening* instrument and a poor *confirmatory* one: a failure here is
robust to both, a pass is doubly weak. Which is an acceptable trade for a go/no-go probe,
provided the pass is never promoted into a result.

**Seed 1 (`--set train.seed=1`, ~$8) is the highest-value follow-up if the probe produces
anything worth quoting** — ahead of scaling seed 0 up. An unreplicated number at 5 years is
worth less than a replicated one at 2.

**Do not tune on train-side spread.** Fair CRPS at `m_train = 4` is unbiased but high-variance.
The `m_test = 50` estimate is the one to trust.

**One adjustment allowed.** If calibration misses, exactly **one** documented change to
`noise_std` / `noise_dim` may be made and the probe rerun. Iterating until the number lands in
the band is fitting to the criterion, not measuring it. Anything beyond one change gets reported
as a tuned result, not a measured one.

---

## 4. What would falsify the noise design

Stated now so it cannot be softened later:

- **Collapse.** Spread stays below 0.05 × RMSE with FiLM gradients vanishing → the conditioning
  cannot drive the dynamics. The `loop.py:551` gate stops the run itself.
- **Under-dispersion that does not close.** Spread–skill well below 0.8 and flat while RMSE
  improves → members are too similar; the ensemble is decorative.
- **Over-dispersion.** Ratio above 1.25 → noise is being injected faster than the dynamics
  organise it, which would point at `noise_std` rather than at the FiLM design.

A result in any of these categories is a genuine finding about decision 0001 and worth having
for $17 — **provided §5 rules out the boring explanation first.**

---

## 5. The negative branch is confounded — the discriminator, pre-registered

**This is the section most likely to be needed, and the one most likely to be skipped.**

The FiLM projection starts at zero. On the 2-year probe the noise pathway gets **5,840 optimizer
steps to grow from nothing**. A calibration failure at that budget has two explanations that
imply opposite responses:

| reading | signature at epoch 16 | correct response |
|---|---|---|
| **budget** — not enough steps | spread/skill still climbing monotonically; FiLM gradient norm healthy, non-vanishing | scale the probe up; the design is fine |
| **design** — conditioning cannot drive the dynamics | spread/skill flat or collapsing; FiLM gradient norm vanishing | fix the conditioning; more compute will not help |

All three quantities — spread, zero-noise gap, FiLM gradient norm — are printed every epoch by
the existing anti-collapse probe. Nothing new needs building; this just writes the rule down
first.

**Evidence the budget branch is live.** On a larger synthetic run spread reached 0.078 × RMSE at
epoch 1 and 0.503 by epoch 3. On the default smoke — 177 optimizer steps — it was still at 0.029
at epoch 3 and tripped the gate. **Step count alone decided which of those happened.**

A related trap: **the epoch-3 gate false-fires on the smoke path.** It is calibrated for a
full-data epoch (~7,100 batches). Do not read a smoke failure there as a broken model.

---

## 6. Cost, and what happens next

| | |
|---|---|
| probe, one seed | **~21.8 h, ~$19** — measured, not estimated |
| second seed, if warranted | +~$19 |
| remaining budget at launch | ~$47 |

**Measured on the L4, 2026-08-26, launch day.** Two prior estimates of this run were wrong — by
2.3× and then by more — so the numbers here come from `wnca benchmark` plus a direct timing of
the selection and val passes:

| stage | per epoch | share |
|---|---|---|
| training, 365 steps @ 8.647 s | 52.6 min | 64% |
| selection, 91 batches, 12 windows, `m_val 4` | 21.8 min | 27% |
| val pass, 183 batches, `m_val 4` | 7.4 min | 9% |
| **total** | **81.8 min** | 16 epochs = **21.8 h** |

Peak GPU memory 10.96 GB at `B=8 × M=4` — fits the L4's 23 GB, and confirms the local 6 GB card
was never an option.

**The step cost is 10.3× the `m=1` benchmark, not 4×.** `B×M = 32` reached via members costs
8.65 s where the same product reached via batch size costs 3.6 s. CLAUDE.md's "cost is linear in
`B × M`" holds for `B` and **not** for `M` — the table behind that claim only ever varied `B`.
Budget any future ensemble phase off a measurement, not off that rule.

**If it calibrates:** scale to 5 or 10 years (~$18 / ~$37) with whatever is left. Full 39 years
was never the point — 3a tests the *noise mechanism*, and 2a already established the data-scaling
result separately.

**If it does not:** apply §5 before calling it a design finding.

**3b stays optional either way.** A calibrated cheap ensemble is the differentiator; member
sharpness refines it. `docs/remaining-plan.md` §4 has the reasoning, including that 3b's
`w_spec: 0` ablation should be run **unconditionally** rather than gated on the treatment
winning — which is precisely the mistake 2d's first design made.

---

## 6b. RESULT — criterion 3 fails, decisively

Measured 2026-08-31 on the test split at `m_test = 50`, 64 start times, day-block bootstrap over
16 day-blocks (`scripts/measure_spread_skill.py`, `spread_skill_3a.json`):

| lead | spread–skill | 95% CI | band | verdict |
|---|---|---|---|---|
| 24 h | **2.120** | [2.088, 2.147] | 0.8–1.25 | over-dispersed, 1.70× the upper bound |
| 72 h | **2.567** | [2.510, 2.615] | 0.8–1.25 | over-dispersed, 2.05× the upper bound |

The intervals are ±1.5% wide. This is a property of the model, not estimator noise.

**This is the §4 over-dispersion branch, named in advance:** *"noise is being injected faster
than the dynamics organise it, which would point at `noise_std` rather than at the FiLM design."*
The conditioning demonstrably works — the zero-noise gap stayed at 0.17–0.27 for all 16 epochs
and spread never came near collapse. What is wrong is its magnitude, and it worsens with lead.

### `wnca eval` never computed this

The eval command reports RMSE, CRPS, perturbation growth and band energies; the string "spread"
appears **zero times** in its output. Every spread number this project produced before today came
from the training probe, which reads `next(iter(loader))` — **one batch of four start times** on
val. That, not the `m_val` reduction, is why the epoch trace swung 1.04–1.99 while the loss curves
ran smooth through it. `scripts/measure_spread_skill.py` exists because of this gap; it pools
variance and squared error across starts and takes the root last, per the CLAUDE.md convention.

### Correction to §2's caveat framing

§2 says the probe's two weaknesses "bias toward passing". **That is only true on the
under-dispersed side, and it was written before the direction was known.** Contamination lowers
the error term, and spread–skill is spread ÷ error, so it pushes the ratio *up* — for an
over-dispersed model that makes the reading worse, not better. The clean-split value is somewhat
below 2.120. Selection leakage moves error by a few percent; it would have to be ~10× larger to
reach 1.25. **The failure stands and the caveat works against the model, not for it.**

### The one permitted adjustment

§3 allows exactly one documented change to `noise_std` / `noise_dim`. Before spending ~$19 on a
retrain, the cheap test is **inference-time noise scaling**: re-measure spread–skill with `z`
scaled by 1/2.12, which costs one eval pass (~$2) and no training. If a single scalar brings both
leads into band, the diagnosis is confirmed as magnitude-only and the retrain is justified; if it
does not, the fault is in how spread grows with lead, not in its size, and a retrain at a smaller
`noise_std` would not fix it either. Report any such factor as a **calibration factor, not a
pass** — tuning until the number lands in the band is fitting to the criterion.

## 6c. WHY it fails — the ensemble is one global offset

Prompted by an observation on the globe visualisation: member 1 sat at one extreme of every
field while member 2 tracked ERA5. Measured with `scripts/diag_noise_structure.py`
(`noise_structure_3a.json`), 50 members, splitting each member's deviation from the ensemble
mean into a spatially-uniform offset plus a spatial pattern:

| lead | uniform share of ensemble variance | global sd | spatial sd |
|---|---|---|---|
| 24 h | **79.2%** | 0.4881 | 0.2499 |
| 72 h | **76.3%** | 1.1496 | 0.6397 |

**About four-fifths of the ensemble's variance is a single number added everywhere.** These are
not 50 alternative weather scenarios; they are one forecast run at 50 settings of a global
warmer/colder dial.

This is what decision 0001 must produce. One `z` per member, constant across all 10,242 cells
and all sub-steps, injected through FiLM, can only modulate the update rule identically
everywhere — so it biases the tendency uniformly, and 800 sub-steps integrate that into a drift.
The design prevented collapse by the crudest available means.

It reframes three earlier numbers rather than adding a fourth:

- The over-dispersion in §6b is now **explained**: uniform offsets inflate spread while
  contributing nothing that could reduce error.
- **The `noise_std` remedy proposed in §6b is dead.** Shrinking the offset would land
  spread–skill in band while leaving it 79% structureless — passing the criterion for the wrong
  reason, which is worse than failing it.
- The members' **1.250 at the coarsest band** (§ the closeout comparison), the highest of any
  band, is that DC offset showing up as excess planetary-scale energy — not resolved weather.

## 6d. The fix: move the noise to the initial condition

Tested on the *existing* checkpoint, inference only, no retraining (`ic_pert_3a.json`). Each
member starts from the analysis plus a spatially-correlated perturbation — white noise relaxed by
graph diffusion to spherical-harmonic degree ~14 — with the FiLM pathway held at **z = 0**, so
the model's own error growth does the diversifying. 20 members, 16 starts.

| eps | spread–skill @24 h | spread–skill @72 h | uniform share |
|---|---|---|---|
| 0.05 | 0.268 | 0.293 | 0.3% |
| 0.10 | 0.523 | 0.530 | 0.3% |
| **0.20** | **0.984** | **0.888** | **0.3%** |
| 0.40 | 1.546 | 1.235 | 0.4% |

Two results, and they should be quoted with different confidence.

**The structural one is robust and untuned.** The uniform share is 0.3–0.4% at *every* amplitude
— it is a property of where the noise enters, not of how much. Against the FiLM pathway's 79%,
that is the finding: **99.7% spatial structure versus 21%.**

**The calibration one is a swept curve, not a pass.** At `eps = 0.20` both leads land inside
0.8–1.25, and the ratio is nearly flat across lead (0.984 → 0.888) where the FiLM ensemble grew
2.12 → 2.57 — spread and error growing together is what a well-posed ensemble does. But that
amplitude was *selected from a sweep against the criterion*, on 20 members and 16 starts with no
bootstrap CIs, on the contaminated split. **It must be confirmed independently** — different
starts, 50 members, day-block CIs — before it is quoted as a calibrated result.

### What this changes for the project

The noise does not need a bigger budget or a different magnitude; **it needs to enter somewhere
else.** That is available today at inference cost, without touching the architecture, and it does
not violate the CLAUDE.md constraint on re-drawing `z` — `z` is simply unused.

A follow-up worth its ~$2: the same technique bracketed the band on the **2c deterministic**
checkpoint while the diffusion fix was being verified. If that holds up, a calibrated structured
ensemble may not require the CRPS phase at all — CRPS would then be buying member *sharpness*
(measured, real) rather than calibration.

## 7. Checklist before this is written up as a result

- [x] Seed 0 complete — 16 epochs, best selection 0.20636 at epoch 15
- [x] Spread–skill at 24 h and 72 h reported **with day-block bootstrap CIs** — §6b, fails
- [ ] **n = 1 stated wherever the ratio is quoted**, with the §3 reporting rule applied — a pass
      is not "the design calibrates"
- [ ] Contamination caveat (§2) carried alongside, since it biases the same direction
- [ ] Zero-noise ablation reported
- [ ] §5 discriminator applied and the verdict recorded, **whichever way it goes**
- [ ] Val RMSE tracked alongside spread — if it degraded on the 2-year split, only the
      *direction* of the spread result carries
- [ ] Any `noise_std` / `noise_dim` change declared, with the §3 one-adjustment rule honoured
- [ ] Short-lead spread checked specifically against pushforward's known 6 h regression (§2)
