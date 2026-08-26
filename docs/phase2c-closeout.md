# Phase 2c close-out: two evaluation-only measurements

Run 2026-08-26 on the phase 2c checkpoint (`best_phase2c_full_20260819_192050.pt`, epoch 7),
test split, 40 start times, leads to 72 h. No training. Instance time ~$1.
Script: `scripts/diag_2c_closeout.py`. Raw numbers: `diag_2c_closeout.json`.

These were items 5 and 6 of `remaining-plan.md` §7 — the two highest-value things left that cost
almost nothing. One settles a claim that has carried an open caveat since 2b′; the other prices
phase 3b. **Both changed the picture, and one of them in the opposite direction from the guess.**

---

## 1. Solar forcing: the claim reproduces, and it was *understated* by ~3×

The standing claim — "solar forcing is worth 19% on 2 m temperature at 24 h" — came from a
forcing-zeroed ablation on 2b′, a model that was undertrained and confounded (it also doubled
`n_substeps` and had its LR halved mid-run after a divergence). `milestone-2-findings.md` has
flagged it as *"no support from a healthy model"* ever since. 2c is healthy and has forcing on,
so the ablation was simply repeated: same weights, same start times, forcing channels zeroed at
source.

### 2 m temperature, RMSE (K)

| lead | real forcing | zeroed | gain | 2b′ gain |
|---|---|---|---|---|
| 6 h | 1.728 | 2.176 | 20.6% | 7.0% |
| 12 h | 1.795 | 2.888 | 37.9% | 9.4% |
| 18 h | 2.188 | 4.154 | 47.3% | 13.9% |
| **24 h** | **2.484** | **5.194** | **52.2%** | **18.6%** |
| 30 h | 2.744 | 5.900 | 53.5% | — |
| 48 h | 3.589 | 6.399 | 43.9% | — |
| 72 h | 4.436 | 8.419 | 47.3% | — |

The gain climbs to a peak at 30 h and then plateaus in the 44–48% band out to 72 h. Every 2b′
lead is beaten by a factor of 2.5–3.4.

### geopotential 500, RMSE (m²/s²)

| lead | real | zeroed | gain |
|---|---|---|---|
| 6 h | 130.3 | 131.7 | 1.1% |
| 24 h | 176.5 | 188.4 | 6.3% |
| 72 h | 465.5 | 498.2 | 6.6% |

Small but consistently non-zero, so the pathway is not 2t-only — it is worth ~6% on the
headline dynamical field too.

### The correction runs the unintuitive way

The natural expectation was that a healthy model would show a *smaller* forcing effect, with the
19% partly an artefact of 2b′ having failed to learn the dynamics. The opposite happened: the
undertrained model **under-used** the forcing input, and training it properly roughly tripled how
much the model leans on it. The caveat is retired; the mechanism is real and larger than claimed.

### What this ablation does *not* show

The three forcing channels are the model's **only clock**. Zeroing them does not isolate
radiative physics — it removes the time-of-day signal entirely, and 2 m temperature is dominated
by a diurnal swing. So the defensible statement is *"the forcing input is worth 52% on 2t at
24 h"*, not *"solar radiation is worth 52%"*. Separating the two would need an arm with a
time-of-day channel but no zenith geometry. Nobody has claimed the stronger version, but it is
the easy thing to slide into when quoting this number.

### It falsifies one branch of the open 2t question

`milestone-2-findings.md` frames 2c's worst channel as an either/or:

> 2 m temperature: −31% at 24 h vs persistence. Solar forcing is *on* in this run, so **either
> the conditioning is too weak to drive a field that swings on a 24 h period, or a 223 km mesh
> cannot resolve the land-surface contrast that sets it.**

**The first branch is now dead.** Conditioning that is worth 52% is not "too weak to drive the
field". 2t is the worst channel *despite* the forcing pathway working hard — which relocates the
problem to mesh resolution. §2 below independently corroborates that.

---

## 2. Spectral band energy: heavily over-smoothed, worst by far on 2t

Deterministic 2c rollout vs ERA5 at 72 h, energy ratio (model / ERA5); 1.00 = matched, <1 =
over-smoothed. Band 0 = largest scales, band 4 = smallest.

| channel | band 0 | band 1 | band 2 | band 3 | band 4 |
|---|---|---|---|---|---|
| geopotential_500 | 1.042 | 0.992 | 0.821 | 0.646 | 0.710 |
| specific_humidity_500 | 1.029 | 0.996 | 0.709 | 0.358 | 0.162 |
| temperature_850 | 0.958 | 0.612 | 0.307 | 0.149 | 0.096 |
| **2m_temperature** | 0.875 | 0.237 | 0.081 | 0.046 | **0.045** |
| *mean* | *0.98* | *0.71* | *0.48* | *0.30* | *0.25* |

The decline is monotone and there is no hot fine band, so this is genuine over-smoothing rather
than small-scale noise sitting on a damped spectrum — the failure mode the verdict logic was
specifically rewritten to distinguish after getting it wrong on a smoke run.

**2 m temperature retains 4.5% of ERA5's fine-scale energy.** That is a spectral collapse, not a
mild damping, and it is a second independent line of evidence for the mesh-resolution reading in
§1: 2t's variance lives at scales a 223 km mesh cannot carry, so the model reproduces the
large-scale envelope and almost nothing beneath it.

### Correcting the script's own verdict

`diag_2c_closeout.py` prints *"3b has real work to do — this RAISES 3b's expected value."*
**That framing is too strong and should not be quoted.** An MSE-trained deterministic model
converges toward the conditional mean, which is smooth by construction; over-smoothing here is
the *expected consequence of the objective*, not evidence of an architectural limit. It therefore
does not establish that a spectral loss is needed.

What the measurement genuinely delivers:

1. **The baseline 3a's members have to beat** — 0.25 mean at the finest band, 0.045 for 2t.
2. **Exit criterion 4 starts a long way off.** "Members within 20% of ERA5" means ≥0.80; three of
   four channels are below 0.20 at band 4.
3. **A prediction worth checking.** CRPS penalises over-smooth members directly, so if the FiLM
   design works at all, 3a should move these ratios *without* a spectral term. That is a free
   diagnostic to read off 3a's members.

**3b's pricing is therefore unchanged, not raised.** Decide it against 3a's member spectra, not
against this deterministic profile. The pre-registered rule stands: if 3b runs, its `w_spec: 0`
ablation runs unconditionally rather than gated on the treatment winning.

---

## 3. What changed in the standing claims

| claim | before | now |
|---|---|---|
| solar forcing worth 19% on 2t @ 24 h | undertrained model, no healthy support | **52.2% on 2c**; caveat retired |
| solar pathway is 2t-only | untested | ~6% on z500 as well |
| 2c's 2t deficit: weak conditioning *or* mesh | open either/or | **conditioning branch falsified** |
| deterministic member sharpness | unmeasured | 0.25 at finest band; 2t at 0.045 |
| 3b expected value | "optional" | still optional — decide from 3a's members |
