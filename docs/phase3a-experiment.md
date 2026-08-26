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

### A launch-blocking bug found while auditing this config

3a set only `stochastic` and `noise_dim` and inherited the rest from `base.yaml`. That gave
`spectral_norm: false` against 2c's `true`, so `arch_hash` differed (`5ef20621…` vs
`0f26ab65…`) and `warm_start` — which raises on unexpected keys — would have hit **12
spectral-norm buffer keys**. **Phase 3a could not have started at all.** The same inheritance
silently reverted `weight_decay` to 1e-5, the value 2c measured as ~11,000× too weak.

Both fixed and covered by tests in `tests/test_checkpoint.py`, verified by reverting the fix and
watching the arch test fail. Recorded here because it is the third config in a row whose
unexamined defaults would have cost a paid run.

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

**Two seeds, non-negotiable.** $17 for the pair at probe scale. 2d's entire lesson is that one
seed produced a confident, internally consistent, *wrong* result, and that the noisiest arm was
the one carrying the treatment.

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
| probe, per seed | ~9.5 h, **~$8** |
| probe, two seeds | **~$17** |
| remaining budget | ~$50 |

**If it calibrates:** scale to 5 or 10 years (~$18 / ~$37) with whatever is left. Full 39 years
was never the point — 3a tests the *noise mechanism*, and 2a already established the data-scaling
result separately.

**If it does not:** apply §5 before calling it a design finding.

**3b stays optional either way.** A calibrated cheap ensemble is the differentiator; member
sharpness refines it. `docs/remaining-plan.md` §4 has the reasoning, including that 3b's
`w_spec: 0` ablation should be run **unconditionally** rather than gated on the treatment
winning — which is precisely the mistake 2d's first design made.

---

## 7. Checklist before this is written up as a result

- [ ] Both seeds complete
- [ ] Spread–skill at 24 h and 72 h reported **with day-block bootstrap CIs**, both seeds
- [ ] Zero-noise ablation reported
- [ ] §5 discriminator applied and the verdict recorded, **whichever way it goes**
- [ ] Val RMSE tracked alongside spread — if it degraded on the 2-year split, only the
      *direction* of the spread result carries
- [ ] Any `noise_std` / `noise_dim` change declared, with the §3 one-adjustment rule honoured
- [ ] Short-lead spread checked specifically against pushforward's known 6 h regression (§2)
