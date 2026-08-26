# Phase 2d — results: what locality costs, and what it does not

Companion to `phase2d-experiment.md`, which recorded the design **before** the outcome was
known. This file records the outcome. Evaluations are on **held-out test 2020** with
`--checkpoint` passed explicitly; the pre-registered quantity is 24 h z500 area-weighted RMSE.

Interactive version: `media/phase2d_report.html` (`scripts/report_2d.py`).

---

## 1. The headline, in one table

z500 area-weighted RMSE (m²/s²), held-out test 2020, 24 h lead:

| arm | what it is | 24 h RMSE |
|---|---|---|
| **A** | local NCA, 4 perception groups, seed 0 | **174.7** |
| **B** | identical to A, **seed 1** | **173.3** |
| **D** | local NCA + dilated ring at **8 hops** (reach 160/window) | **173.4** |
| **E** | local NCA + dilated ring at **1 hop** (reach 20/window) — the placebo | **180.4** |
| **D1** | as D, **seed 1** | **188.4** |
| **E1** | as E, **seed 1** | **180.0** |
| **C** | same-budget message-passing GNN | **291.2** |

Two-seed means: **local 174.0 ±0.8%**, **`d=8` 180.9 ±8.3%**, **`d=1` 180.2 ±0.2%**.

The seed spread is the yardstick every other number is measured against, and it is **not one
number** — it varies by arm as well as by lead (§2, §8).

---

## 2. The result: reach buys nothing measurable

Each arm is the mean of two seeds; ± is the within-arm seed spread.

| lead | local (A,B) | `d=8` (D,D1) | `d=1` (E,E1) | **d8 vs d1**<br>*reach alone* | d8 vs local |
|---|---|---|---|---|---|
| 6 h | 130.2 ±1.9% | 129.6 ±0.5% | 132.2 ±0.8% | −2.0% | −0.5% |
| 12 h | 110.2 ±0.1% | 111.9 ±6.1% | 111.6 ±0.9% | +0.3% | +1.5% |
| **24 h** | **174.0 ±0.8%** | **180.9 ±8.3%** | **180.2 ±0.2%** | **+0.4%** | **+4.0%** |
| 48 h | 318.8 ±0.4% | 334.7 ±8.7% | 331.9 ±0.6% | +0.8% | +5.0% |
| 72 h | 460.9 ±0.1% | 487.7 ±8.5% | 481.5 ±0.1% | +1.3% | +5.8% |
| 120 h | 738.8 ±1.2% | 790.5 ±6.9% | 779.0 ±0.2% | +1.5% | +7.0% |
| 168 h | 948.3 ±0.8% | 1011.5 ±6.7% | 996.8 ±1.2% | +1.5% | +6.7% |

**The pre-registered comparison (`d=8` vs `d=1`, identical in parameters and compute, differing
only in reach) is +0.4% at 24 h** — the wrong sign, and twenty times smaller than the `d=8`
arm's own 8.3% seed spread.

### The power bound, which has to travel with the null

With n = 2 per arm, the standard error of each arm mean is |seed difference| / 2:

| arm | mean | SE of mean |
|---|---|---|
| local | 174.0 | 0.40% |
| `d=8` | 180.9 | **4.15%** |
| `d=1` | 180.2 | 0.11% |

The SE of the `d8 − d1` difference is **4.16%**, so the **minimum effect this experiment could
detect at 2σ is 8.3%**.

That is the honest form of the result, and it is weaker than "reach buys nothing": **a 4% reach
effect — exactly the size originally claimed — sits below the detection threshold.** This
experiment cannot distinguish "reach is worth 4%" from "reach is worth 0%". It rules out large
effects, not small ones.

The defensible claim is therefore **"locality is sufficient at this budget and resolution,
against isotropic long-range reach, to within ~8%"** — not "non-locality does not help".

With n = 2 the SD estimate carries one degree of freedom, so the bound itself is rough. The
cheapest thing that would tighten it is **a third seed on `d=8` specifically** — the only noisy
arm — not a third seed everywhere.

**What does hold:** both dilated arms are worse than the plain local model — 4.0% and 3.6% at
24 h, widening with lead. **A fifth, near-collinear perception channel costs ~4%** whatever
radius it carries, and reach does not recover it. The wording matters: that cost is measured
for a channel whose 1-hop form is a uniform-weight Laplacian sitting beside the existing
cotangent one, and it should not be quoted as the cost of adding *any* channel (§3).

> **Adding a long-range channel to a strictly local update rule does not improve forecasts by
> more than this experiment can resolve (~8%). At 160 hops per window — 1.7× the mesh diameter,
> effectively global — the model is no better than at 20 hops, and both are worse than not
> adding the channel at all.**

Caveats that remain: one radius, isotropic, n = 2, one mesh, one training budget (§6). And the
channel's ~4% cost is confounded with its near-collinearity to the existing Laplacian (§3).

---

## 3. What was retracted, and what it cost to find out

**On seed 0 alone, `d=8` beat `d=1` by 3.9% at 24 h — consistently across all nine leads.**
Against a 0.8% seed spread that read as a real effect, and it was written up and reported as
one, with a narrative on top: the fifth channel costs ~4%, reach pays it back, two effects
cancelling.

Then `d=8` seed 1 scored **188.4** against seed 0's **173.4**. One run, an **8.3% within-arm
spread** — ten times the local model's — and the 3.9% lead becomes a 0.4% deficit.

**Three things this cost, worth recording so they are not repeated:**

1. **"Consistent across nine leads" is not nine observations.** A rollout is one trajectory
   family from one set of weights, so a seed's idiosyncrasy propagates through every lead. The
   nine-lead consistency was close to *one* correlated observation, and it was used as if it
   were nine.
2. **The decomposition was interpretation, not measurement.** `d=8` vs `d=1` isolates radius
   cleanly. "The channel costs 3.7%" came from `d=1` vs the local model, where the 1-hop ring is
   a uniform-weight Laplacian sitting beside the existing cotangent one — near-collinear inputs.
   That half was never clean, and the tidy story was built on it.
3. **Selection-metric agreement was mistaken for seed stability.** See §8; this is the most
   transferable of the three.

**The design held even though the reading did not.** The placebo caught the parameter confound —
and would have been skipped entirely under the first version of the plan, which gated it on
`d=8` winning. The pre-registered seed gate fired *precisely because* the gap was under 5%,
which is what surfaced the noise. Both guards did the job they were put there for. The failure
was in believing n = 1 before they finished.

## 4. What this means for the thesis

The project's claim is that *a strictly local update rule is enough to forecast global weather*.
The result is more interesting than either a clean confirmation or a refutation:

The seed pair settled it (§2), and the answer is cleaner than the n = 1 reading suggested:

- **Reach buys nothing this experiment can resolve.** `d=8` (160 hops/window) against `d=1`
  (20 hops) is **+0.4% at 24 h** on two-seed means — the wrong sign. But the minimum detectable
  effect here is **8.3%** (§2), so this rules out a large reach effect and says nothing about a
  small one.
- **The channel that carries reach costs ~4%.** Both dilated arms sit above the plain local
  model at every lead from 12 h out.
- **So locality is sufficient at this budget**, and the earlier "reach is not free skill" reading
  was one lucky seed.

What this still does **not** license is "non-locality doesn't help" in general — for three
separate reasons, and the write-up should carry all of them:

1. **Power.** The bound is ~8%; a 4% effect would be invisible here.
2. **Form.** This tested *isotropic* long-range diffusion at one radius. A directional ring
   gradient, or learned long-range weights, are untested.
3. **No working non-local baseline.** The GNN control turned out to reach *less* far than the
   local model (§6), so the only functioning non-local arm in this phase is the dilated stencil.
   A reviewer asking "where is the message-passing baseline that actually passes information?"
   has a fair question, and the answer is that it would need pooling/unpooling across icosphere
   levels, which was not built.

The claim that survives all three is narrow and worth stating exactly: **a strictly local update
rule is sufficient at this resolution and budget, against isotropic long-range reach, to within
~8%.**

---

## 5. Error growth

Sustained per-window perturbation growth, converted to error-doubling time. The real atmosphere
doubles synoptic-scale errors in roughly 1.5–2.5 days.

| arm | sustained growth | doubling time |
|---|---|---|
| A · local, seed 0 | ×1.079 | 2.28 d |
| B · local, seed 1 | ×1.090 | 2.01 d |
| E · d=1 placebo | ×1.080 | 2.25 d |
| D · d=8 | ×1.097 | **1.87 d** |
| C · control GNN | ×1.195 | **0.97 d** |

Two readings:

- **Do not compare doubling times between NCA arms at n = 2.** D at 1.87 d against the placebo's
  2.25 d looks like an 18.5% difference, but A and B — *the same model at different seeds* —
  differ by 12.6% on this exact quantity. The per-arm numbers are reported above because they
  were measured, not because they support a ranking. An earlier version of this section read a
  "starts better, degrades quicker" mechanism into them; that was the same over-reading that
  produced the retracted result in §3, one metric over.

  **The only comparison the data supports here is categorical:** all four local-rule arms sit in
  or near the atmospheric band; the GNN does not.
- **The GNN doubles error twice as fast as the atmosphere.** That is the signature of a model
  whose own states drift off-distribution, and it matches its collapse beyond 72 h.

**Seeds move this too:** A and B differ by 12% on doubling time (2.28 vs 2.01 d) despite being
the same model. Growth-rate comparisons need the same error bars the RMSE comparisons do.

---

## 6. The GNN control was never a locality test

The same-budget message-passing GNN scores **291.2 at 24 h (+67%)**, is worse than persistence
at 6 h and beyond 72 h, and doubles error in under a day.

It is reported as a **parameter-matched message-passing baseline**, not as a locality control,
because its receptive field was measured (`scripts/measure_reach.py`, gradient support on the
real mesh) at **9 fine-mesh hops per window against the local model's 20**. The "non-local" arm
moved information *less far* than the arm it was meant to control for.

The cause: a coarse icosphere level's edge list only connects the nodes that **exist** at that
level — level 1 holds 42 of 10,242 — and `ControlGNN` has no pooling/unpooling to bridge
levels, so a coarse layer is a no-op for ~99.6% of the mesh. Raising `gnn_levels` does not fix
it; measured, it makes coverage worse for typical nodes while handing 42 privileged nodes a
long reach. Full table in `configs/phase2d_control.yaml`.

---

## 7. What this does not test

- **The long-range channel is isotropic.** `mean(ring) − centre` is scalar diffusion; the local
  stencil additionally carries direction via ∇x and ∇y. So this measures the value of
  *isotropic* long-range information. A directional ring gradient — ∇x/∇y assembled over ring
  edges, two more groups by the same construction — is the natural follow-up, and given reach
  is worth 4% isotropically it is now worth running.
- **One radius, not a curve.** d = 8 gives 160 hops/window against a mesh diameter of ~93 — it
  is effectively global. The 4% is the value of *global* reach; d = 2 (0.43× diameter) and
  d = 4 (0.86×) would say where the effect saturates.
- **n = 1 per dilated arm.** See §7.

---

## 8. The pre-registered gate, and how it resolved

The rule, written in `phase2d-experiment.md` §6 **before** any dilated run finished:

> Seeds 1 of both run **if and only if** the seed-0 gap is under 5% on the test-2020 24 h z500
> RMSE.

The seed-0 gap was **3.9%** — under the threshold — so both second seeds ran. They cost ~$43 and
they overturned the result (§2, §3). The gate was written expecting to fire on a *null* branch;
it fired on what looked like a positive one, and that is exactly when it was most needed.

**Per-arm seed spread at 24 h**, the number that made the difference:

| arm | spread |
|---|---|
| local (A/B) | 0.8% |
| `d=1` placebo (E/E1) | 0.2% |
| **`d=8` treatment (D/D1)** | **8.3%** |

Seed spread is **not a property of the phase** — it is a property of the arm. Nothing in the
local or placebo arms predicted that the treatment arm would be ten times noisier.

---

## 9. The methodological finding: the noise floor is not one number

The selection metric put seed-to-seed spread at **0.21%**, and that figure carried planning
weight through this whole milestone. On the evaluation metric it holds only at short leads:

| lead | 24 h | 120 h | 168 h | 240 h | 360 h |
|---|---|---|---|---|---|
| \|A − B\| | 0.8% | 1.2% | 0.8% | **7.1%** | **12.2%** |

**At 240 h and beyond, nothing under ~10% is resolvable at n = 2.** This retracts a reading
given in an earlier status report, where D's apparent 4–12% long-lead deficit against A was
described as real; it is inside the noise.

Every future long-lead claim in this project needs seeds, and the 0.21% number should not be
quoted as *the* noise floor again.

---

## 10. Two-metre temperature: a metric artefact, confirmed

Phase 2c reported 2 m temperature at **−31% skill** at 24 h — the only channel worse than
persistence, and an apparent failure of the solar-forcing conditioning. Scoring at every 6 h
lead rather than only at multiples of 24 shows what was happening:

```
        6h   12h   18h   24h   30h   36h   42h   48h   54h   60h   66h   72h
2t     +36   +51   +28  -31*   +15   +24    +5  -42*    -4    +8   -12  -57*
z500   +43   +70   +66  +71*   +66   +66   +63  +62*   +58   +56   +53  +51*
```

Mean skill **−43.5% on the 24 h grid, +16.8% off it** — a 60-point sawtooth. Persistence at a
24 h multiple compares the *same local solar time*, so it reproduces the diurnal cycle for free;
for a diurnally-dominated surface field that makes it an unusually strong baseline at exactly
those leads. z500 shows no such pattern, so this is specific to the field, not an evaluation bug.

**Consequence:** 2 m temperature must be quoted off the 24 h grid from now on.

### Correction: this does *not* touch the 2b′ solar-forcing claim

An earlier version of this section said the unresolved 2b′ claim that solar forcing is "worth
19% on 2t" was "judged on the same bad metric". **That was wrong, and it was checked only after
being asserted.** The 19% came from a *within-model ablation* — the same 2b′ checkpoint scored
with real forcing against forcing zeroed, in raw RMSE:

| lead | real forcing | zeroed | gain | on the 24 h grid? |
|---|---|---|---|---|
| 6 h | 2.230 | 2.397 | 7.0% | no |
| 12 h | 2.895 | 3.195 | 9.4% | no |
| 18 h | 3.042 | 3.532 | 13.9% | no |
| 24 h | 3.273 | 4.019 | **18.6%** | yes |

Two things follow. **Persistence never enters that comparison**, so the diurnal-alignment
artefact cannot apply to it. And the ablation **already spans off-grid leads**, showing 7–14%
there — so the effect is not confined to 24 h multiples, it grows with lead.

The claim's real problem is the one the findings doc already names: it was measured on an
undertrained, confounded model (2b′ also doubled `n_substeps` and had its LR halved mid-run),
and **it has never been repeated on a healthy model**. Phase 2c has solar forcing on, so the
same forcing-zeroed ablation can be run against its checkpoint. That is the experiment that
settles it, and it is evaluation-only. Deferred while the instance trains the seed pair.

A separate measurement, run while chasing this: comparing the 2b (no solar) and 2b′ (solar)
*checkpoints* on 2 m temperature shows 2b′ **worse at every lead**, by 10% at 24 h rising to
47% at 72 h. That is a different comparison from the ablation — two differently-trained models
rather than one model with an input switched off — and it mostly restates that 2b′ is
undertrained. `scripts/diag_2t_solar.py`, `diag_2t_solar.json`.

---

## 11. A diagnostic that did not work, and was deleted

`scripts/diag_val_split.py` was written to separate two readings of the GNN's inverted
train/val relationship: rollout instability versus over-regularisation. It compared a
**one-window prediction against a two-window target**, so its two halves were not the same task
and its ratios were not interpretable. Its numbers were excluded from this write-up rather than
explained.

**Deleted rather than fixed.** A matched-lead rewrite would have to step the *analysis* forward
one window for the clean arm so both arms predict the same target from different starting
states — doable, but the question it answered died with the GNN's demotion to a baseline
(§5), and a broken tool with no live use case is worse than no tool. If the question returns,
the rewrite is specified in this paragraph.

This is the second diagnostic this milestone that passed its own smoke test and measured the
wrong thing — the first being the receptive-field arithmetic in §5, which was wrong by 1.5–3.6×
in two independent estimates before anyone measured it.

---

## 12. Before phase 3a

1. **Finish the seed pair** (running; ~48 h) and re-state §2 with error bars.
2. **Decide the 3a warm-start architecture.** `configs/phase3a_crps.yaml` sets
   `warm_start: null  # <- set to the phase 2c checkpoint`, so 3a inherits whichever
   architecture it starts from and locks to it for 15 epochs. A is the cheapest and matches D's
   accuracy; D is 4% better than its own placebo but not better than A. **A is the
   recommendation** unless the seed pair changes the picture.
3. ~~Fix or delete `diag_val_split.py`~~ — **done**, deleted (§10).
4. ~~Amend exit criterion 5~~ — **done**. Replaced with a doubling-time *band* (1.2–3.5 d) plus a
   long-lead boundedness test, in `milestone-2-plan.md`. The original ≤1.05 ceiling selected for
   over-damping: only phase 0 passed it, and phase 0 is the worst forecaster in the ladder. That
   matters more for 3a than it did here, because over-damping directly suppresses ensemble
   spread — the thing 3a exists to measure.
5. **Re-score the 2b′ solar-forcing claim off the 24 h grid** — see `scripts/diag_2t_solar.py`.
