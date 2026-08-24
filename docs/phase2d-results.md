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
| **C** | same-budget message-passing GNN | **291.2** |

A and B differ only in random seed, so the **seed spread is 0.8%** at this lead. That is the
yardstick every other number here is measured against.

---

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

**D beats E at every one of the nine leads.** Reach is doing real work — roughly enough to pay
back what the architecture change costs, and no more.

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

- **Reach is worth ~4% at 24 h.** Locality is not free. A model that can see 160 mesh hops per
  window forecasts measurably better than the same model that can see 20.
- **But the plain local model matches it.** A and B score 174.7 and 173.3 against D's 173.4,
  with 3% fewer parameters and 10% less compute per window. The cheapest way to get D's
  accuracy is not to add reach — it is to not add the channel that carries it.
- **So locality is sufficient *at this budget*, and reach is not free skill.** Both statements
  are true and neither is the headline on its own.

What it does **not** license is "non-locality doesn't help" — 4% is a real effect, and this
tested exactly one radius, isotropically (§6).

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

- **Reach makes error grow faster.** D at 1.87 d against the placebo's 2.25 d. The longer-range
  channel makes the forecast start better and degrade quicker — consistent with the D−E gap
  narrowing from 3.9% at 24 h to 2.1% at 120 h before widening again at 15 days where noise
  dominates.
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

**Consequence:** 2 m temperature must be quoted off the 24 h grid from now on, and the
unresolved 2b′ claim that solar forcing is "worth 19% on 2t" was judged on the same bad metric.

---

## 10. A diagnostic that did not work

`scripts/diag_val_split.py` was written to separate two readings of the GNN's inverted
train/val relationship: rollout instability versus over-regularisation. It compares a
**one-window prediction against a two-window target**, so its two halves are not the same task
and its ratios are not interpretable. Its numbers are excluded from this write-up rather than
explained.

The fix is not a one-liner: a matched-lead version must step the *analysis* forward one window
for the clean arm, so both arms predict the same target from different starting states. Not yet
done. Recorded here because the tool is in the repo and should not be trusted until it is.

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
3. **Fix or delete `diag_val_split.py`** (§10).
4. **Amend exit criterion 5.** Still unadopted, and every arm here fails it as written
   (≤1.05 sustained growth) while being the best models in the ladder.
