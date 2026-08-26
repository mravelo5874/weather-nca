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

## 2. RETRACTED: the reach effect did not survive seeds

**Everything in this section below the line was written at n = 1 per dilated arm. The seed pair
has since landed and killed the headline.** Kept in place rather than rewritten, because the
retraction is the more useful record.

### What the two-seed means say

z500 RMSE, held-out test 2020, each arm the mean of two seeds, ± the within-arm spread:

| lead | local (A,B) | `d=8` (D,D1) | `d=1` (E,E1) | d8 vs d1 | d8 vs local |
|---|---|---|---|---|---|
| 6 h | 130.2 ±1.9% | 129.6 ±0.5% | 132.2 ±0.8% | −2.0% | −0.5% |
| 12 h | 110.2 ±0.1% | 111.9 ±6.1% | 111.6 ±0.9% | +0.3% | +1.5% |
| **24 h** | **174.0 ±0.8%** | **180.9 ±8.3%** | **180.2 ±0.2%** | **+0.4%** | **+4.0%** |
| 48 h | 318.8 ±0.4% | 334.7 ±8.7% | 331.9 ±0.6% | +0.8% | +5.0% |
| 72 h | 460.9 ±0.1% | 487.7 ±8.5% | 481.5 ±0.1% | +1.3% | +5.8% |
| 120 h | 738.8 ±1.2% | 790.5 ±6.9% | 779.0 ±0.2% | +1.5% | +7.0% |
| 168 h | 948.3 ±0.8% | 1011.5 ±6.7% | 996.8 ±1.2% | +1.5% | +6.7% |

**The pre-registered comparison comes back null.** Seed 0 alone said `d=8` beat `d=1` by 3.9% at
24 h. The two-seed means say **+0.4%** — the wrong sign, and an order of magnitude smaller than
the `d=8` arm's own 8.3% seed spread. **There is no measurable reach effect.**

### The 3.9% was seed noise, and one arm is far noisier than the others

| arm | 24 h seed spread |
|---|---|
| local (A/B) | 0.8% |
| `d=1` placebo (E/E1) | **0.2%** |
| `d=8` treatment (D/D1) | **8.3%** |

`d=8` seed 1 scored 188.4 against seed 0's 173.4. That single run is the worst dilated result in
the phase, and it is what turns a 3.9% lead into a 0.4% deficit. Reading `D` alone as "reach is
worth 4%" was reading one lucky seed.

**The selection metric completely failed to predict this.** On the 72 h val rollout the `d=8`
seeds agreed to 0.29% (0.11434 vs 0.11401); on 24 h test RMSE they differ by 8.3%. A model
selection metric that is stable to 0.3% can sit on top of a test quantity that moves 8%. That is
a stronger version of the val/selection decoupling seen in 2c, and it means **selection-metric
agreement is not evidence of seed stability** for anything else.

### What actually holds

Both dilated arms are **worse than the plain local model** — `d=8` by 4.0% and `d=1` by 3.6% at
24 h, and the gap widens with lead. The fifth perception channel costs ~4% whatever radius it
carries, and reach does not recover it.

So the phase's conclusion is simpler, and stronger for the thesis, than the n = 1 reading:

> **Adding a long-range channel to a strictly local rule does not improve forecasts. At 160 hops
> per window — 1.7× the mesh diameter, effectively global — the model is no better than at 20,
> and both are worse than not adding the channel at all.**

The honest caveats remain: one radius, isotropic, n = 2, and the channel's cost is confounded
with its near-collinearity to the existing Laplacian (below).

---

## 2b. The n = 1 reading, kept for the record

## 2. The result inverted once the placebo landed

**Read against A alone, D looks like a null.** 173.4 sits between the two local seeds
(174.7, 173.3). The obvious reading — the one written up and reported before E finished — is
that giving a strictly local model global reach buys nothing.

**The placebo says that null is two effects cancelling.**

E carries the *identical* fifth perception channel as D and the *identical* 1,060,412
parameters, but its ring sits at 1 hop, so it adds **no reach at all**. It scores **180.4 —
3.7% worse than the plain local model**. The extra channel, on its own, hurts: 30,720 more
input weights to fit and one more input the update MLP has to learn to ignore.

So the comparison that carries the result — same parameters, same wall-clock, **only reach
differs** — is:

| lead | D · reach 160 | E · reach 20 | D vs E |
|---|---|---|---|
| 6 h | 129.9 | 131.7 | −1.4% |
| 12 h | 108.5 | 112.1 | −3.2% |
| **24 h** | **173.4** | **180.4** | **−3.9%** |
| 48 h | 320.2 | 332.9 | −3.8% |
| 72 h | 467.0 | 481.6 | −3.0% |
| 120 h | 763.2 | 779.7 | −2.1% |
| 168 h | 977.9 | 1002.9 | −2.5% |
| 240 h | 1228.2 | 1290.7 | −4.8% |
| 360 h | 1556.7 | 1732.4 | −10.1% |

**D beats E at every one of the nine leads.** That is consistent with reach doing real work —
roughly enough to pay back what the architecture change costs, and no more.

**How much weight the nine-lead consistency carries: less than it looks.** These are not nine
independent observations. A rollout is one trajectory family from one set of weights, so a
seed's idiosyncrasy propagates through every lead — the nine leads are closer to **one
correlated observation** than to nine. With one seed per arm against a 0.8% seed spread measured
on a *different* architecture, 3.9% is suggestive and not established. The seed pair is running;
§7.

### The decomposition is shakier than the 3.9%

D vs E cleanly isolates ring **radius**: same parameters, same compute, one knob. That number is
solid modulo seeds.

The other half of the story — "the fifth channel costs 3.7%" — comes from E vs A, and it is
*not* clean. E's ring at 1 hop is a **uniform-weight Laplacian sitting alongside the existing
cotangent Laplacian**: two near-collinear inputs. So E's deficit may be specific to feeding the
update MLP a near-duplicate channel rather than to added capacity in general, and a placebo
carrying an uninformative-but-non-redundant channel might cost nothing at all.

The clean statement of the result is therefore: **increasing the ring radius from 1 to 8 hops
improves 24 h z500 RMSE by 3.9%.** The "two effects cancelling" narrative is interpretation
layered on one clean measurement and one confounded one.

### Why the earlier design would have missed this

The first version of this experiment **gated E on D winning**, on the argument that D's extra
3% of parameters made a D loss "conservative". That was wrong twice over: the sign of the
parameter confound was unknown, and it turns out to be **negative**. Gating E would have buried
a real ~4% reach effect under a fake null, and the write-up would have claimed "locality is
sufficient" on the strength of it.

This is the second time in this phase that a control caught a conclusion that was about to be
drawn wrongly. The first was the receptive-field measurement (§5 below).

---

## 3. What this means for the thesis

The project's claim is that *a strictly local update rule is enough to forecast global weather*.
The result is more interesting than either a clean confirmation or a refutation:

The seed pair settled it (§2), and the answer is cleaner than the n = 1 reading suggested:

- **Reach buys nothing measurable.** `d=8` (160 hops/window) against `d=1` (20 hops) is **+0.4%
  at 24 h** on two-seed means — the wrong sign, and 20× smaller than `d=8`'s own 8.3% seed
  spread.
- **The channel that carries reach costs ~4%.** Both dilated arms sit above the plain local
  model at every lead from 12 h out.
- **So locality is sufficient at this budget**, and the earlier "reach is not free skill" reading
  was one lucky seed.

What this still does **not** license is "non-locality doesn't help" in general. It tested one
radius, isotropically, at n = 2, on one mesh and one training budget (§6).

---

## 4. Error growth

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

- **Reach may make error grow faster — consistent with, not established.** D at 1.87 d against
  the placebo's 2.25 d is an 18.5% difference. But A and B, *the same model at different seeds*,
  differ by 12.6% on this exact quantity. So the D−E growth gap is only about 1.5× the observed
  seed spread, which is not enough to call it. It would fit a "starts better, degrades quicker"
  mechanism — the D−E RMSE gap does narrow from 3.9% at 24 h to 2.1% at 120 h — but the seed
  pair has to land before that reading means anything.
- **The GNN doubles error twice as fast as the atmosphere.** That is the signature of a model
  whose own states drift off-distribution, and it matches its collapse beyond 72 h.

**Seeds move this too:** A and B differ by 12% on doubling time (2.28 vs 2.01 d) despite being
the same model. Growth-rate comparisons need the same error bars the RMSE comparisons do.

---

## 5. The GNN control was never a locality test

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

## 6. What this does not test

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

## 7. Status, and the pre-registered gate

The gate written in `phase2d-experiment.md` §6 **before** any dilated run finished:

> Seeds 1 of both run **if and only if** the seed-0 gap is under 5% on the test-2020 24 h z500
> RMSE.

The gap is **3.9%** — under the threshold. **Seed 1 of both dilated arms is therefore
training**, exactly the branch §6 said to expect for a small-gap outcome. Until it lands:

- D−E is **n = 1 per arm** against a 0.8% seed spread at 24 h. A consistent sign across nine
  leads is reassuring; it is not an error bar.
- The long-lead half of that consistency is **inside the noise regardless** — see §8.

---

## 8. The methodological finding: the noise floor is not one number

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

## 9. Two-metre temperature: a metric artefact, confirmed

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

## 10. A diagnostic that did not work, and was deleted

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

## 11. Before phase 3a

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
