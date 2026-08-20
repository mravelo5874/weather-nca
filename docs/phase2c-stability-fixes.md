# Phase 2c stability fixes — implementation report

Date: 2026-08-19. Author: Kimi (at the user's request), following the analysis in
`docs/cloud-compute-incidents.md`. That doc's attempt-5 post-mortem ended with the fix
**"not yet chosen: that needs its own measurement."** This document records what was chosen,
what was changed, and the test results.

## The diagnosis being addressed (recap)

Attempt 5 showed the 2c divergence is **not** a precision or learning-rate problem:

- The 20-sub-step composed update map has a stability threshold. Below it the loss sits at
  ~0.06; above it *every* batch blows up, with perfectly ordinary-looking weights
  (`layers.0.weight` norm 54.92, frozen, producing losses of 1e24).
- The input layer's weight norm ratchets **linearly** (~1.745e-3/step at lr 2.9e-4) until it
  crosses that threshold. LR changes move the onset later (~2,000 steps at 1e-3 vs ~11,100 at
  3e-4) but do not change the destination.
- `weight_decay = 1e-5` is ~11,000× too weak to oppose the ratchet (decay pull 1.59e-7/step
  vs growth 1.745e-3/step).

So the fixes below bound the recurrence or the weights (the cause), and harden the guards and
checkpointing (the cost when a cause-level fix is insufficient).

## Changes

### 1. Spectral normalisation of the update MLP — the structural fix

`src/wnca/models/update.py`, gated by new `model.spectral_norm` (`src/wnca/config.py`),
enabled in `configs/phase2c_full.yaml`.

Every hidden layer of `UpdateRule` is wrapped in `torch.nn.utils.spectral_norm`, pinning
σ_max ≈ 1 per layer. The per-sub-step gain is then bounded by
`(1 + dt · σ_head · ~1.13)` regardless of where training drifts.

**Reviewer's correction (verified by measurement).** The original wording here claimed threshold
crossing "becomes impossible by construction." That is overstated: σ_head is *not* pinned, so
only the hidden-layer contribution is bounded. Measured on the tiny config with every hidden
layer pinned and the head inflated:

| head weight std | rel. state change, one 6h window |
|---|---|
| 0.1 | 1.76 |
| 1.0 | 115.6 |
| 10.0 | 7.4e4 |

The head is an unbounded amplification path, and in the attempt-5 trace it was the
**fastest-growing parameter**: +30% over 3,900 steps against `layers.0`'s +14%. Real heads sit
near std 0.015 and the growth was linear, so extrapolation does not reach std 0.1 inside the
57k-step schedule — there is margin, and `weight_decay` (fix #2) now acts on it. But the bound
rests on weight decay holding the head, which is **unmeasured**. `||head.weight||` must be watched
alongside σ.

Two scope decisions, both measured rather than assumed:

- **The head is NOT wrapped.** It is zero-init by design (untrained model = identity map),
  and `spectral_norm`'s power iteration divides by the weight's norm — **verified empirically
  on torch 2.6 that an exactly-zero weight produces NaN on the very first forward**, in both
  train and eval mode. Wrapping the head would have NaN'd batch 1 of the cloud run. The
  head's norm remains guarded by the existing checkpoint-load assert.
- **The FiLM projection is NOT wrapped.** Capping it would cap ensemble spread in phase 3a.

`arch_hash()` includes `spectral_norm` when enabled (the wrapper changes state-dict keys:
`weight_orig`, `weight_u`, `weight_v`), following the existing "only when enabled" policy so
older checkpoints stay loadable.

**Consequence for phase 3a:** 2c checkpoints now carry spectral-norm buffers, so phase 3a
must set `model.spectral_norm: true` to warm-start from them. A mismatch fails loudly at
`warm_start` (unexpected keys), not silently.

### 2. `weight_decay: 0.1` in `configs/phase2c_full.yaml`

Derived from the incident's own measurements: balance requires
`wd ≈ growth / (lr · ‖w‖) = 1.745e-3 / (2.9e-4 · 55) ≈ 0.1`.
The previous 1e-5 was decoration. With spectral norm pinning the hidden layers, decay now
acts mainly on the unwrapped parameters (head, FiLM, biases) — the residual ratchet path.
The two fixes are complementary: spectral norm bounds the map's gain; weight decay bounds
what spectral norm doesn't wrap.

### 3. Finite-but-absurd loss guard

`src/wnca/train/loop.py`, `Trainer.run_epoch`.

Attempt 4's destroyed model reported a **finite** 3.69e23 train loss for ten hours — neither
the non-finite-loss guard nor the (then loss-only) step guard applied. New rule: a batch loss
above **100× the running median** of recent losses (window 200, armed after 20 batches) is
skipped and counted; sustained occurrences (>2% of an epoch, same tolerance as the existing
guards) abort with a resume instruction. The onset trace goes 0.219 → 63.9 → 3.0e8 in four
steps, so this fires within a few batches of a runaway — before the first non-finite
gradient exists.

Design details that matter:

- The window lives on the **Trainer**, not the epoch — a model destroyed mid-epoch is still
  judged against the healthy losses that preceded it.
- Absurd values are **excluded** from the window, so the median cannot chase the runaway.

### 4. Rolling mid-epoch checkpoints

`ckpt_every_steps` existed in the config schema but was wired to nothing. It is now
implemented in `run_epoch` and set to **2000** in the 2c config. Attempt 4 diverged at step
11,103 of a 7,122-step epoch; the nearest resume point was ~4,000 steps (~1 h of L4 time)
back. The rolling save (`last.pt`, atomic via the existing tmp-and-replace
`save_checkpoint`) caps that at ~28 min. It is deliberately *not* matched by
`latest_checkpoint("best_*.pt")`, so `--resume auto` never picks it up accidentally.

Known limitation (pre-existing, unchanged): resume granularity is the epoch — resuming from
any checkpoint restarts at `epoch + 1`. Fixing that needs batch-index restore in the loader
and is out of scope.

### 5. SIGTERM: checkpoint first, score never

`src/wnca/train/loop.py`, `fit`. Incidents-doc §4.1: the handler ran validation + selection
(3+ min) before checkpointing, against spot's ~30 s notice — the reason every 2c run paid
on-demand rates. Now, when the preemption flag is set during the training epoch, the loop
writes `preempted.pt` **immediately after the training pass, before validation and the
selection metric run at all**. The end-of-epoch preemption path is retained for a SIGTERM
that lands during scoring. This unblocks spot provisioning (~60% discount; ≈$14 → ≈$6 on the
projected 16 h run).

### 6. `diagnose.py --stages lr` now measures the ratchet, not just NaN counts

Incidents-doc lesson 2 was "probe length must exceed onset" — a losing game, since fixes move
onsets later. The probe now tracks the input layer's weight norm per step and fits the slope:

- `spectral_norm` off: Frobenius norm (the incident's measured proxy), extrapolating the
  +14% growth that preceded the measured crossing → predicted onset step, compared against
  the schedule length. Verdict `RATCHET` when the crossing lands inside the schedule.
- `spectral_norm` on: **σ_max of the effective weight** instead. Measured during
  implementation: the Frobenius norm *still grows* under spectral norm (σ_max is pinned but
  smaller singular values spread), so the old proxy false-alarms. σ_max hovers at 1.00–1.03
  with power-iteration jitter, so the check is the wrapper's actual contract — σ_max must
  stay ≤ 1.1 — not a slope fit on noise.

This is also how fix #1 is validated before relaunch: on the cloud instance,
`scripts/diagnose.py --stages lr` should show `sig0/step` flat and `onset~ inf`.

## Deliberate confound: fixes #1 and #2 ship together

Spectral norm and the 10,000x weight-decay increase go into the same run, so **2c's outcome will
not attribute to either one alone.** This is the same confound the project criticises in phase 2b
(coupling entangled with a 37x capacity jump) and in the kimi critique's item 2.

Accepted knowingly, for two reasons: the two fixes are complementary rather than alternative
(spectral norm bounds what it wraps, weight decay bounds what it does not), and at ~$19/run
separating them costs a full extra run to answer a question that does not change the next
decision. If 2c converges, neither fix is thereby established as individually necessary; if it
underfits, weight decay at 0.1 is the first thing to back off, since it is the change with the
larger effect on the model's capacity to fit.

## Reviewer addendum: head-norm tracking in the ratchet probe

`stage_lr` tracked sigma_max of `layers[0]` only, and the relaunch checklist said to stop
watching weight norms because "the sigma pin is now the quantity to watch." That left the one
path spectral norm does not wrap -- the head -- unmonitored. It now also reports the head's RMS
slope (`head/step`) and a `HEAD-RATCHET` verdict, and `head_onset` joins the safe-LR filter.

Tracked as RMS (`norm / sqrt(numel)`) against an absolute threshold of 0.1, **not** as a ratio
from the starting value: the head is zero-init, so a ratio rule divides by ~0 and fires on every
fresh model. That false alarm was caught by running the probe rather than by reading it -- the
first implementation reported a crossing at ~500 steps for a healthy fresh model. Fits are also
skipped entirely while the head is still at its zero init (RMS < 1e-3), because the fast rise out
of zero is by design and a linear fit across that transient predicts a crossing that never
arrives. A trained head sits near 0.015, clear of both bounds.

Sanity check on the real numbers: the attempt-5 trace's head grew 2.06 -> 2.69 over 3,900 steps,
an RMS slope of ~9.2e-7/step from 0.0154, extrapolating to 0.1 at ~92,000 steps -- outside the
57k-step schedule. Consistent with "margin exists, but it is margin, not a guarantee."

## Test results

- **Full suite: 172 passed** (previously 158), ~25 s, well under the 60 s `make test` budget.
- New tests, `tests/test_spectral_norm.py` (6):
  - σ_max of every wrapped layer ≤ 1.05 after convergence, with raw weights inflated to
    std 2.0.
  - Zero-init head and FiLM stay unwrapped; fresh model produces finite output (regression
    guard for the measured NaN).
  - Composed-map gain: with identical inflated weights, the plain model's forecast step
    explodes (>10× state change) while the spectral-normed one stays bounded (<1×).
  - `arch_hash` differs across the flag.
  - Checkpoint save/load round trip with spectral-norm state-dict keys.
  - Integration: one real training epoch with spectral norm **and** gradient checkpointing
    (the recompute re-runs the power iteration — must not trip autograd's version counters).
- New tests, `tests/test_train_loop.py` (6):
  - Sustained finite-but-absurd losses raise (the attempt-4 regression, magnitude edition).
  - An isolated absurd spike is tolerated: skipped, counted, median unpolluted.
  - Healthy epochs report zero absurd batches (no false positives).
  - Rolling checkpoint written mid-epoch, resumable (weights + optimizer + step), and
    invisible to `--resume auto`.
  - Preemption checkpoints **before** scoring: with the flag set, neither validation nor the
    selection metric runs, and `preempted.pt` exists.
- **`make smoke` equivalents pass**: train + eval on synthetic data, plus a separate smoke
  train with `model.spectral_norm=true` and `ckpt_every_steps=20` end-to-end through the CLI
  (rolling `last.pt` written, selection checkpoint unaffected).
- **`stage_lr` verified on a synthetic tiny setup**: Frobenius proxy mode reproduces a
  measurable slope (1.19e-3/step) and an onset extrapolation; σ_max mode reads pinned with
  `onset~ inf`.

## Incidental findings

- `spectral_norm` is the **legacy** implementation (`weight_orig` / `weight_u` / `weight_v`,
  power iteration in train mode only), and the tests access `weight_orig` accordingly. This was
  flagged as a risk if torch were upgraded to the parametrization-based implementation.
  **Resolved: verified on torch 2.9.1** (the version the cloud instance runs) --
  `nn.utils.spectral_norm` is still the legacy API there, producing identical state-dict keys
  `['bias', 'weight_orig', 'weight_u', 'weight_v']`. `arch_hash` and checkpoint loading are
  unaffected, and all 172 tests pass on 2.9.1 both locally and on the instance. The
  parametrization-based version lives at `nn.utils.parametrizations.spectral_norm` and is a
  separate opt-in.
- The CLI's `--set` takes **space-separated** pairs under one flag
  (`--set a.b=1 c.d=2`); repeated `--set` flags silently keep only the last. A mistyped
  verification command briefly built an unneeded `era5_sub3` cache and two run dirs; all
  three were deleted.
- `scripts/run_phase.sh` and `docs/cloud-compute-incidents.md` carry **pre-existing
  uncommitted changes** from the previous agent (the `python3 -u` fix and the incidents doc
  itself). Not modified by this work.

## Deliberately not done

- **Perception block normalisation** (the Laplacian block's 1/h² scale into `layers.0`).
  Plausible contributor — 96% of the pre-explosion gradient mass sat in that layer — but it
  changes the model's effective inputs and needs its own measurement. Next candidate if the
  ratchet signal persists.
- **Raising `w_over`.** The overflow penalty watches hidden channels; the measured failure
  was gain in the composed map with ordinary-looking weights. Wrong mechanism.
- **Resuming mid-epoch.** Resume granularity is still the epoch (see #4).

## Outcome — the fixes held (2026-08-20)

2c ran to completion: 8 epochs, 56,976 steps, 22 h 05 m, **zero non-finite and zero
finite-but-absurd batches**. Full results in `docs/milestone-2-findings.md` §2c-r.

**Pre-launch probe.** `scripts/diagnose.py --stages lr` on the real config, 200 steps:

| lr | grad trend | max gn | non-finite | `sig0/step` | verdict |
|---|---|---|---|---|---|
| 3.0e-4 | 1.7× | 1.48 | 0 | 5.52e-06 | ok |
| 5.0e-4 | 1.9× | 2.31 | 0 | 3.60e-06 | ok |

Against 4.23e+12 max grad norm and 287/500 non-finite at lr 1e-3 before the fix. The σ slope is
~300× flatter than the pre-fix Frobenius ratchet of 1.745e-3/step — the pin holds.

**The head, tracked through the run.** The reviewer's correction above said the head is an
unbounded path held only by weight decay, and that "steps to 0.1" was still landing inside the
schedule at the last check. Measured across the real run, the crossing estimate receded far
faster than training advanced:

| checked at step | head RMS | latest-window slope | extrapolated steps to 0.1 |
|---|---|---|---|
| 2,160 | 0.01336 | 3.51e-06 | 24,656 |
| 5,010 | 0.02002 | 2.02e-06 | 39,528 |
| 8,107 | 0.02540 | 1.57e-06 | 47,428 |
| 47,378 | **0.04420** | **9.68e-08** | **576,435** |

The head gained 0.0189 in its first 8k steps and 0.0007 in the last 7k — a 27× slowdown. It
asymptotes; it does not ratchet. **The concern is resolved on the real architecture**, not just
inferred from the tiny config. The 0.1 threshold was always a tiny-config proxy and remains one,
but the run never approached it.

**Guard behaviour.** The gradient guard fired exactly once in anger — on the attempt-5
diagnostic resume, aborting after 143 non-finite gradients where attempt 4 had idled for ten
hours. It did not fire during the real run. `ckpt_every_steps` wrote `last.pt` throughout and was
never needed.

**Still open.** The confound (fix #1 and #2 together) is unresolved by design and recorded above.
Spot provisioning is unblocked by fix #5 but was **not used** — the run paid on-demand rates,
because switching provisioning model requires recreating the VM and would have destroyed the
64 GB cache.
