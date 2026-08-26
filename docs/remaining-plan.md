# The rest of the project: phase 3, milestone 2 exit, and beyond

Written 2026-08-26, after phase 2d closed. Supersedes the phase-3 sections of
`milestone-2-plan.md` where they disagree, because that plan was costed before the ensemble
phases had a measured price and before the budget was half spent.

---

## 1. Where the project actually stands

| phase | what it tested | status |
|---|---|---|
| 0 | M1 reproduction, 1 var, 2 yr | ✅ |
| 2a | data volume, 1 var | ✅ worth ~30% |
| 2b | multivariate coupling, 2 yr | ✅ |
| 2b′ | solar forcing + 40 sub-steps | ❌ confounded, undertrained |
| 2b-pf | pushforward training | ✅ fixed long-lead divergence |
| **2c** | **39 years, 28 channels** | ✅ **the model: 24 h z500 = 174.0 ±0.8%** |
| **2d** | **locality** | ✅ **null to within ~8% — see the power caveat below** |
| 3a | ensembles, fair CRPS | ⬜ not started |
| 3b | spectral loss, member sharpness | ⬜ not started |

### Milestone-2 exit criteria

| # | criterion | status |
|---|---|---|
| 1 | Locality quantified against a same-budget control | ✅ 2d, both a GNN baseline and a minimal-delta reach control |
| 2 | Gap closes with scale (2a→2b→2c ladder) | ✅ 24 h z500 329.7 → 297.6 → 174.0 |
| 3 | Ensemble is real: spread–skill 0.8–1.25, zero-noise ablation differs | ❌ **needs 3a** |
| 4 | Members individually sharp: band energies within 20% at 72 h | ❌ **needs 3b** |
| 5 | Error growth in band + bounded at 15 days *(amended)* | ✅ 2c: 2.28 d doubling, 1.34× climatology |

**Three of five hold. The two that don't are the entire probabilistic half of the milestone**,
and they are both unmeasured rather than failing.

**Criterion 1 needs a caveat when it is written up.** 2d's minimum detectable effect was
**8.3%** (`phase2d-results.md` §2), so it rules out a large locality penalty and not a small
one. And the GNN control reached *less* far than the local model, so the only functioning
non-local arm was the dilated stencil. The claim that survives is "a strictly local rule is
sufficient at this resolution and budget, against isotropic long-range reach, to within ~8%" —
not "non-locality does not help".

---

## 2. The budget, stated plainly

**~$50 remains** of the $300 credit. Phase 3a as configured in `phase3a_crps.yaml` costs
**~$143**. That is the constraint the rest of this plan is written around.

Cost model: 2c measured 2.75 h/epoch at m=1 on an L4 at $0.89/h; cost is linear in B×M
(CLAUDE.md measured fact), and batches/epoch scales with the training-split length.

| 3a variant | batches/ep | wall clock | cost |
|---|---|---|---|
| 39 yr, m=4, 15 ep — **as written** | 7122 | 161 h | **$143** |
| 39 yr, m=4, 8 ep | 7122 | 86 h | $76 |
| 10 yr, m=4, 15 ep | 1826 | 41 h | $37 |
| 5 yr, m=4, 15 ep | 913 | 21 h | $18 |
| **2 yr, m=4, 15 ep** | **365** | **8 h** | **$7** |

Cutting **data** is a 20× lever; cutting `m_train` is a ~2× lever that damages the fair-CRPS
estimator, which is the phase's actual deliverable. So cut data first, members never.

---

## 3. Phase 3a — probabilistic forecasting

**The question:** does the FiLM noise design produce a *calibrated* ensemble, or does it
collapse? Decision 0001's entire noise architecture exists to prevent collapse and has never
been measured at scale.

**Already verified** (local smoke, free): the stochastic path runs end to end, and given enough
optimizer steps spread grows — 0.078 → 0.096 → 0.503 spread/RMSE over three epochs, with a
non-zero zero-noise gap. **The ensemble does not collapse.** That was the biggest code risk and
it is retired.

**One trap:** the hard gate at `loop.py:551` stops training if spread < 0.05 × RMSE at epoch 3.
It **false-fires on the default smoke** (177 optimizer steps). Do not read a smoke failure there
as a broken model.

### 3a.1 — the $8/seed probe (do this first)

Warm-start from the phase-2c checkpoint, fine-tune on a **2-year** split, 16 epochs at
`m_train = 4` with pushforward on. Answers the calibration question at ~1/17 the price of the
full-data run: **~7-9 h and ~$6-8 for the single seed now planned** (amended from two, §3a.3).

Design, success criteria and the failure-reading rules are pre-registered in
`docs/phase3a-experiment.md`.

**Pre-register before running:**

- **Success** = spread–skill in **0.8–1.25** at 24 h and 72 h, with a zero-noise ablation
  measurably different from the ensemble mean.
- **Report the ratio with a confidence interval, not as a point estimate.** A spread–skill ratio
  computed over `n_starts` initialisations from one seed is one correlated observation family —
  the same trap as "nine leads are not nine observations" in `phase2d-results.md` §3. Use a
  **day-block bootstrap** over start times. Note the bootstrap covers *within-run* sampling
  noise only; with the two-seed rule dropped (§3a.3) *between-run* variance is unmeasured.
- **Do not tune on train-side spread.** Fair CRPS at `m_train = 4` is unbiased but
  high-variance; the eval-side estimate at `m_test = 50` is the one to trust.
- Track **val RMSE alongside spread**. If RMSE degrades on the 2-year subset, the probe is
  confounded and only the *direction* of the spread result carries.
- **One** documented adjustment to `noise_std` / `noise_dim` is allowed in response. Iterating
  until the number lands in the band is fitting to the criterion, not measuring.

### 3a.1b — the negative branch is confounded, and needs its own discriminator

**Pre-register this too, because without it a failed probe will be over-read.**

The FiLM projection is **zero-initialised**. So a 2-year, 15-epoch fine-tune at lr 1e-4 gives the
entire noise pathway **~5,475 optimizer steps to grow from nothing**. A calibration failure at
that budget has two completely different explanations, and they imply opposite responses:

| reading | signature at epoch 15 | response |
|---|---|---|
| **budget** — not enough steps | spread/skill still climbing monotonically; FiLM gradient norm healthy and non-vanishing | scale the probe up; the design is fine |
| **design** — the conditioning cannot drive the dynamics | spread/skill flat or collapsing; FiLM gradient norm vanishing | fix the conditioning; more compute will not help |

The instrumentation already exists — spread, zero-noise gap and FiLM gradient norm are printed
every epoch by the anti-collapse probe. This just writes the discriminator down **before** the
result, so a $7 null cannot be reported as "a real finding about the FiLM design" when it may
only be a statement about 5,475 steps.

Local evidence that the budget branch is live: on a larger synthetic run spread reached
0.078 × RMSE at epoch 1 and 0.503 by epoch 3, but on the default smoke (177 steps) it was still
at 0.029 at epoch 3 and tripped the gate. **Step count, not design, decided which of those
happened.**

### 3a.2 — scale up, conditional

If the probe calibrates: run **5 or 10 years at 15 epochs ($18 / $37)** depending on what is
left. Full 39 years is out of reach and, given 3a tests the *noise mechanism* rather than data
scaling, is not what the phase is for.

If the probe does *not* calibrate: **apply the §3a.1b discriminator before calling it a design
finding.** Only the "design" branch — spread flat and FiLM gradients vanishing — justifies
fixing the conditioning. The "budget" branch justifies more steps and nothing else.

### 3a.3 — seeds

**Amended 2026-08-26 to ONE seed**, by the project owner, on budget grounds — before any run,
and recorded rather than applied silently.

The original rule and its reasoning stand as written: 2d's single seed produced a confident,
wrong, internally-consistent result, and seed spread there was **arm-specific** (±0.2%, ±0.8%,
±8.3% across three arms), so one run cannot establish which kind of arm 3a is.

What follows from n = 1 is a **reporting** constraint, not a redesign: the probe becomes a
screening instrument. Its two weaknesses — the contaminated split and the single seed — both
bias toward *passing*, so a failure is robust and a pass may not be promoted to "the FiLM design
calibrates". The full rule, including which §5 verdicts survive at n = 1, is in
`phase3a-experiment.md` §3. Seed 1 is the highest-value follow-up if the probe yields anything
quotable — ahead of scaling seed 0 up.

### 3a.4 — the config audit, done

Run against §3a's concerns, and it found a **launch-blocking bug plus a silent one**:

| field | 2c | 3a as written | consequence |
|---|---|---|---|
| `model.spectral_norm` | true | **false** | `arch_hash` differs; `warm_start` raises on 12 unexpected keys — **3a could not start** |
| `train.weight_decay` | 0.1 | **1e-5** | silently reverts the fix that saved phase 2c |
| `train.ckpt_every_steps` | 2000 | 0 | a 15-epoch run with no rolling checkpoint |

All three now mirror 2c explicitly, with the reason recorded in the config so nobody
"simplifies" them back. `tests/test_checkpoint.py` asserts the arch hashes match and the weight
decay agrees; verified by reverting the fix and watching the test fail.

**Two decided 2026-08-26**, and recorded in the config with their reasoning:

- **`pushforward: true`**, mirroring 2c — so the loss is the only thing that changes. Cost is
  ~8% here, not ~33%, because the pushforward step is deterministic and is not multiplied by
  `m_train`. Known risk: its 6 h regression could distort short-lead spread.
- **`epochs: 16`**, double 2c's 8, because the zero-initialised FiLM pathway must grow from
  nothing. On the 2-year probe that is 5,840 optimizer steps — few enough that §3a.1b's
  budget-vs-design discriminator is load-bearing.

Still unaudited: **`lr: 1e-4`** has no recorded provenance. 2c's learning rate came from a
fixed-LR sweep; this one was "dropped an order of magnitude". If the probe misbehaves, suspect
it first.

**Warm-start from the plain local model (phase 2c, arm A).** 2d settled this: best at every
lead, cheapest, most seed-stable (±0.8% against the dilated arm's ±8.3%), and its error growth
sits closest to the atmospheric band — which matters more in a probabilistic phase, where
over-damping directly suppresses the spread 3a exists to measure.

---

## 4. Phase 3b — spectral loss and member sharpness

Exit criterion 4: individual members' band energies within 20% of ERA5 at 72 h. The machinery
exists (`losses/spectral.py`, Chebyshev band filters) and `w_spec` is already wired.

**Only run this if 3a calibrates.** A sharp member from an uncalibrated ensemble is not a
result, and the criteria are ordered that way for a reason. Cost is the same as the 3a variant
it follows, plus one ablation arm with `w_spec: 0` — which, after 2d, should be treated as a
**placebo and run unconditionally**, not gated on the treatment winning.

At present budget this is likely out of reach. That is an honest stopping point, not a failure.

---

## 5. If the money runs out — and it probably does

**There is a complete, defensible result already**, and it does not need more compute:

- A strictly local update rule at 223 km, 1.03 M parameters, trained on 39 years, reaches
  **24 h z500 RMSE 174.0** — 41% better than the 2-year model, ~5× off the frontier.
- **Data volume dominates every architectural change tried.**
- **Non-locality does not help**: neither a same-budget multi-scale GNN nor a globally-reaching
  dilated stencil beats the local model, and the dilated arms are *worse*.
- Error growth lands **inside the atmospheric band** (2.28 d) — the first model in the ladder to
  do so, and it is also the most accurate, which contradicts the original stability criterion.

The probabilistic half would be a second paper's worth of work. Writing up the deterministic
half now is a reasonable end state for M2, with criteria 3 and 4 recorded as unmeasured.

### The findings most likely to outlive the project

These are methodological and cost nothing to write up:

1. **Seed spread is a property of the arm, not the phase** — 0.2%, 0.8% and 8.3% across three
   arms of one experiment, and nothing in the first two predicted the third.
2. **A model-selection metric can be stable to 0.3% while the test quantity moves 8%.**
   Selection-metric agreement is not evidence of seed stability for anything else.
3. **Persistence at 24 h multiples is diurnally aligned**, which makes it an unusually strong
   baseline for diurnally-dominated fields and produced a 60-point artefact in 2 m temperature.
4. **A stability gate defined as a growth *ceiling* selects for over-damping** — only the worst
   forecaster in the ladder passed the original criterion 5.
5. **Two diagnostics passed their own smoke tests while measuring the wrong thing.** Smoke
   coverage is not correctness; each needed an oracle it did not have.

---

## 6. Milestone 3, if it happens

`milestone-2-plan.md` puts mesh refinement (100 km → 28 km), the full 13-level state, and
calibrated extremes beyond M2, then the project's original bet: **Lagrangian feature particles
on the Eulerian base** — Neural Particle Automata, which the mesh design was chosen to allow.

Two things to carry forward:

- **The M3 compute budget was derived from the wrong node arithmetic** (the "1°" that was
  actually ~223 km ≈ 2°) and has never been recomputed. Do that before committing.
- **The Lagrangian direction is the strongest differentiator.** An Eulerian NCA at 223 km will
  likely lose the accuracy race regardless — 2d showed the architecture is not the bottleneck,
  data and resolution are. A learned-particle method with local interactions has no incumbent
  competitor. If the project continues, that is where the novelty is, and it deserves explicit
  pivot criteria rather than being reached by exhaustion.

---

## 7. Immediate next actions, in spend order

**Free — do all of these before any instance starts:**

1. ~~Audit `phase3a_crps.yaml`~~ — **done, and it found a launch-blocking bug.** 3a inherited
   `spectral_norm: false` against 2c's `true`, so `arch_hash` differed and `warm_start` would
   have raised on 12 unexpected keys: **3a could not have started at all.** The same inheritance
   silently reverted `weight_decay` to the value 2c measured as ~11,000× too weak. Both fixed,
   both now covered by tests in `tests/test_checkpoint.py`.
2. **Write `docs/phase3a-experiment.md`** before running — design, the §3a.1 success criteria,
   the §3a.1b budget-vs-design discriminator, the noise-tuning rule.
3. **Decide `pushforward` and `epochs`.** Both are inherited rather than chosen, and both are
   flagged in the config. 2c trained with pushforward ON; base defaults it OFF.
4. **Start the write-up skeleton** from §5. The methodological findings are the most citable
   part of the project and cost only time.

**Near-free — evaluation only; one instance-hour covers both. DONE 2026-08-26, ~$1, results in
`docs/phase2c-closeout.md`:**

5. ~~**Forcing-zeroed ablation on the 2c checkpoint.**~~ **Done.** The 2b′ claim reproduces at
   **52.2% on 2t at 24 h** against the claimed 18.6% — understated by ~3×, not inflated — plus
   ~6% on z500. Caveat retired. It also falsified the "conditioning too weak" branch of 2c's 2t
   deficit, leaving mesh resolution as the explanation.
6. ~~**Deterministic spectral band energies, 2c rollout vs ERA5 at 72 h.**~~ **Done.** Monotone
   over-smoothing: band means 0.98 / 0.71 / 0.48 / 0.30 / 0.25, with 2t collapsed to **0.045**.
   This does **not** raise 3b's price the way the script's printed verdict claims — an MSE model
   converging to the smooth conditional mean is the objective behaving as designed, not an
   architectural limit. What it buys is the baseline 3a's members must beat and a free
   diagnostic: CRPS should move these ratios with no spectral term at all.

**Then spend:**

7. **The two-year probe, ONE seed (~$6–8).** Amended from two on 2026-08-26 on budget grounds;
   see `phase3a-experiment.md` §3 for what n = 1 costs and the reporting rule it forces.
   Ready to launch: `configs/phase3a_probe.yaml`,
   warm-start wired to the 2c checkpoint (now pulled local, `arch_hash 5ef20621bbf047bb` matching
   the config), split resolving onto the cache that already exists, smoked end to end. Seed 1 is
   `--set train.seed=1`. **Cloud only** — 3a OOMs the local 6 GB card at stock batch size.
   Read the probe-contamination caveat in `phase3a-experiment.md` §2 before quoting any number
   it produces.
8. **Decide from the probe**, not from a plan written before it.

3b is **optional even if 3a calibrates** (§4). A cheap calibrated ensemble from a 1 M-parameter
local model is the differentiator worth having; member sharpness refines it.
