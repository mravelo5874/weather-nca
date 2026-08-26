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
| **2d** | **locality** | ✅ **null: reach buys nothing measurable** |
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

### 3a.1 — the $7 probe (do this first)

Warm-start from the phase-2c checkpoint, fine-tune on a **2-year** split, full 15 epochs at
`m_train = 4`. Answers the calibration question at 1/20 the price.

**Pre-register before running:**
- Success = spread–skill in **0.8–1.25** at 24 h and 72 h, with a zero-noise ablation
  measurably different from the ensemble mean.
- Track **val RMSE alongside spread**. If RMSE degrades on the 2-year subset, the probe is
  confounded and only the *direction* of the spread result carries.
- **Decide now** whether tuning `noise_std` / `noise_dim` in response is legitimate. Suggested
  rule: **one** documented adjustment allowed; iterating until the number lands in the band is
  fitting to the criterion, not measuring.

### 3a.2 — scale up, conditional

If the probe calibrates: run **5 or 10 years at 15 epochs ($18 / $37)** depending on what is
left. Full 39 years is out of reach and, given 3a tests the *noise mechanism* rather than data
scaling, is not what the phase is for.

If the probe does *not* calibrate: that is a real finding about the FiLM design and it cost $7.
Fix the conditioning rather than training through it — the gate's own advice.

### 3a.3 — seeds

**Two seeds, non-negotiable, after 2d.** At 2-year scale that is $14 total. Phase 2d's whole
lesson is that a single seed produced a confident, wrong, internally-consistent result — and the
arm that was noisiest was the one carrying the treatment. A spread–skill ratio at n = 1 is worth
very little.

### 3a.4 — the config needs an audit first

2d's config shipped with `epochs: 20` against 2c's 8, a control at 1.67× its own stated
parameter budget, and an inherited `weight_decay` 10,000× off. 3a's config has the same
provenance problem — values written down early and never tested:

- `epochs: 15` against 2c's 8, for what is a *warm-start fine-tune* at lr 1e-4.
- `lr: 1e-4`, "dropped an order of magnitude", from no recorded measurement. 2c's lr came from
  an actual sweep.
- The epoch-3 spread gate threshold of 0.05, provenance unknown.
- `warm_start: null` — literally unset, with a comment saying to fill it in.

**Warm-start from the plain local model (phase 2c, arm A).** 2d settled this: A is the best arm
at every lead, the cheapest, the most seed-stable (±0.8% vs the dilated arm's ±8.3%), and its
error growth sits closest to the atmospheric band — which matters more in a probabilistic phase,
where over-damping directly suppresses the spread 3a exists to measure.

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

## 7. Immediate next actions

1. **Audit `phase3a_crps.yaml`** against §3.4. Free.
2. **Write `docs/phase3a-experiment.md`** before running — design, pre-registered success
   criteria, the noise-tuning rule. The two-doc pattern caught the gated-placebo error in 2d and
   costs nothing. Free.
3. **Run the $7 two-year probe, two seeds ($14).**
4. **Decide from the probe**, not from a plan written before it.

Deferred and cheap, whenever an instance is up: the **forcing-zeroed ablation on the 2c
checkpoint**, which settles the 2b′ solar-forcing claim on a healthy model (`phase2d-results.md`
§10).
