# Phase 2d — the locality control: experiment record

Written **while the runs are in flight** (2026-08-21), before the outcome is known, so the
design and its caveats are on record rather than reconstructed afterwards. Results go in
`docs/milestone-2-findings.md` when all four runs and their evaluations are done.

---

## 1. The question

Every phase so far has varied data, conditioning or training objective. The thesis itself —
*a strictly local update rule is enough to forecast global weather* — has never been tested,
because there has been nothing non-local to compare against.

2d supplies that: a same-budget non-local baseline on the same mesh, the same data and the same
optimizer.

**The evidence this design produces is asymmetric, and not in the convenient direction.**
Almost every uncontrolled factor tilts toward the local arm: `lr` and `weight_decay` were tuned
on it (§4), it is handed ∇x, ∇y and ∇² for free by fixed geometry while the control must infer
differential structure from raw neighbour values, it spends ~3× the compute per step, and —
measured, see §5 — it actually reaches *further* per window than the "non-local" control does.

So a **local win is close to uninformative**: it could be the learning rate, the compute, the
free differential filter bank, or simply the greater reach. A **control win, despite all of
that, would be damning** for the thesis. This experiment can falsify the locality claim far more
cleanly than it can support it, and any write-up has to say so.

---

## 2. The two arms

| | local arm | control arm |
|---|---|---|
| config | `configs/phase2c_full.yaml` | `configs/phase2d_control.yaml` |
| `model.kind` | `nca` | `control_gnn` |
| update rule | per-cell MLP over `[identity, ∇x, ∇y, ∇²]`, 1-hop | interaction-network message passing |
| applications per 6 h window | **20 sub-steps** | **1 pass**, 6 message-passing hops |
| non-locality comes from | nothing — strictly 1-hop | nested icosphere: coarse-level edges as long-range shortcuts. **But see §5** — measured, this reaches *less* far than the local arm |
| **parameters** | **1,029,692** | **1,010,780** (0.982×) |

Both inherit seeding, conditioning, rollout and ensemble machinery from `ForecastModel`, so the
update operator really is the only structural difference.

### Parameter matching was measured, not assumed

The shipped 2d config carried a gate in its own header — *"tune `gnn_hidden` / `gnn_hops` until
params and ms/step are within ~10% of the NCA's"* — that **had never been applied**:

| configuration | params | vs NCA |
|---|---|---|
| `gnn_hidden 256`, `gnn_hops 4` (as shipped) | 1,719,612 | **1.67×** |
| `gnn_hidden 192`, `gnn_hops 4` | 982,524 | 0.954× |
| **`gnn_hidden 160`, `gnn_hops 6`** (chosen) | **1,010,780** | **0.982×** |

160/6 was picked over 192/4 for two reasons: it is the closer match, and it is the **more
generous control** — 6 hops give the non-local arm more information flow than 4, so a win for
the strictly local arm against it is the stronger result.

`tests/test_amp.py::test_phase2d_control_is_parameter_matched_to_phase2c` now asserts the two
configs stay within 10%, so a later edit to either cannot silently break the match.

---

## 3. What is held identical

Every `train:`, `data:` and `state:` value in the 2d config is a verbatim mirror of 2c's,
written out explicitly rather than inherited so the match is auditable as a diff. Verified
field by field:

| | value |
|---|---|
| data | 1979–2017 train · 2018 val · 2020 test |
| mesh | `n_sub 5` — 10,242 nodes, ~223 km |
| channels | 28 physical + 4 static, solar forcing **on** |
| epochs / batch | 8 / 8 → 7,122 batches per epoch, 56,976 steps |
| lr · weight decay · warmup | 3e-4 · 0.1 · 500 |
| precision | bf16 (compute capability 8.9) |
| pushforward · noise_std | on · 0.05 |
| selection metric | 72 h rollout (`ckpt_windows 12`, `ckpt_subsample 0.25`) |
| seeds | 0 and 1 per arm |

Inheritance was a live hazard here, not a hypothetical: without the explicit mirror the control
would have silently taken `weight_decay: 1e-5` from `base.yaml` against 2c's 0.1.

---

## 4. What is NOT matched, and why

**Wall-clock cannot also be matched.** The control runs one pass per window against the NCA's
20 sub-steps, so at matched parameters it is simply faster — measured at **55 min/epoch versus
2 h 45 m**, a 3.0× gap. Parameters are the matched quantity; the wall-clock difference is
recorded as a result in its own right, since the compute cost of iterating a local rule is part
of what the thesis trades away.

**The hyperparameters were tuned on the local arm.** `lr 3e-4` came from an NCA sweep, and
`weight_decay: 0.1` was derived specifically to stop the NCA's weight-norm ratchet — a
pathology the control does not have, having no 20-sub-step recurrence. Applying it to the
control may over-regularise it.

This is the weakest point in the design. Until it is checked, the claim this experiment
supports is **"at the NCA's tuned hyperparameters"**, not "in general". A short probe on the
control arm (`wd ∈ {0, 0.01, 0.1} × lr ∈ {3e-4, 1e-3}`, two epochs each) costs ~$6 at 55
min/epoch and is planned before the result is written up as anything stronger.

---

## 5. What a difference can and cannot attribute to

The control changes **at least four things at once**, not two as an earlier version of this
section claimed:

1. **Locality** — the update sees shortcut edges rather than only immediate neighbours.
2. **Iteration count** — one pass per window against twenty sub-steps.
3. **Weight sharing** — the NCA applies *the same* MLP 20 times; the control has 6 distinct
   layer stacks. Recurrence-versus-depth is its own axis.
4. **Input featurisation** — the NCA is handed ∇x, ∇y and ∇² from fixed geometry; the control
   must infer differential structure from raw neighbour values. That is a physics prior the
   control does not get, and unlike the others it cannot be fixed by configuration.

So a gap measures:

> a strictly local, recurrent, differential-featurised rule iterated 20× **versus** a
> shortcut-edge message-passing network applied once, at matched parameters

and **not** locality in isolation. The clean single-variable control — same update MLP, same 20
sub-steps, same perception plus a uniform dilated ring — is the experiment that would actually
isolate the thesis, and it is unrun. Recorded here so the finding is never written up as more
than it is.

### The control reaches less far than the arm it controls for (measured)

Receptive field of one 6 h window, measured by gradient support on the real mesh
(`scripts/measure_reach.py`) — not by counting hop spans, which gets it wrong by 1.5–3.6×:

| | level-1 node | level-3 node | **fine-only node** (99.75% of mesh) |
|---|---|---|---|
| control, `gnn_levels 3` | 8 hops / 91 | 8 / 109 | **9 hops / 109 nodes** |
| control, `gnn_levels 4` | 16 / 156 | 20 / 224 | 13 / 30 |
| control, `gnn_levels 5` | 32 / 146 | 28 / 35 | 13 / 30 |
| **NCA** | 20 / ~1200 | 20 / ~1200 | **20 hops / ~1200 nodes** |

A coarse level's edge list only connects the nodes that **exist** at that level — level 1 has 42
vertices of 10,242 — and `ControlGNN` has no pooling/unpooling to carry information between
levels, so a coarse layer is a no-op for ~99.6% of the mesh. It is message passing on nested
subgraphs, not a multi-scale architecture, and `gnn_levels` does not fix it: raising it shrinks
coverage for typical nodes (109 → 30) while handing 42 privileged nodes 32 hops.

**Consequence for what this experiment is.** The control is not a locality control. A loss is
attributable to reach before topology. These runs should be reported as *a parameter-matched
message-passing baseline with ~9-hop reach*, and the clean single-variable control — same MLP,
same 20 sub-steps, perception widened by a uniform dilated ring — remains unrun.

A related caveat from the pre-2d diagnostics: measured dt-invariance is **partial**. Refining
the integration (dt 0.05×20 → 0.025×40) costs 3–7% RMSE, and coarsening blows up entirely. The
learned rule is closer to a discretised differential operator than to a fixed per-window map,
but it is not scale-free — so "local PDE rule" is defensible and "a PDE" is not. 2d is framed
as testing the former.

---

## 6. Seeds, and the noise floor

Both arms run seeds 0 and 1 (`train.seed`, which sets weight init, batch order and the input
noise draws — not the data or the splits). Without this, a gap is uninterpretable: the project's
own docs record ~5% swings from RNG ordering alone, and phase 2b's headline claims were made at
n=1.

**The local arm's two seeds are complete, and the noise floor is much tighter than feared:**

| | final selection metric |
|---|---|
| NCA seed 0 | 0.112575 |
| NCA seed 1 | 0.112813 |
| **spread** | **+0.21%** |

Per-epoch spread was noisy early (−5.5%, −13.2%, −3.5%) and converged steadily (−2.4%, −0.6%,
−1.6%, +0.5%) to 0.21% at epoch 8.

**This is the local arm's noise floor, and only that.** An earlier version of this section
concluded "n = 2 is sufficient", which silently assumed the control's seed variance is similar.
It may well not be: rollout-unstable models tend to have fatter seed distributions, and the
control's early diagnostics (val 4.5× train) point that way. The control's own two seeds will
measure it; until they land, no sufficiency claim is supported.

---

## 7. Execution

One `g2-standard-8` (1× NVIDIA L4, bf16) in `us-east1-c`. The four runs are **sequential, not
parallel**, because the project's GPU quota is 1:

```
GPUS_ALL_REGIONS          limit 1.0   usage 1.0
NVIDIA_L4_GPUS us-east1   limit 1.0   usage 1.0
```

An increase to 3 has been requested (up to 48 h). `scripts/run_queue.sh` runs the jobs
back-to-back and stops the instance only after the last one — `autostop.sh` is right for a
single run and wrong for a queue, where a shutdown between arms would need a manual restart
into a possible capacity stockout each time. `scripts/cloud_launch.sh` clones the prepared disk
snapshot for parallel arms once quota allows.

| run | wall clock | cost |
|---|---|---|
| NCA seed 0 (= phase 2c) | 22 h 05 m | ~$20 |
| NCA seed 1 | ~22 h | ~$20 |
| GNN seed 0 | ~7.4 h | ~$7 |
| GNN seed 1 | ~7.4 h | ~$7 |

### One bug this uncovered

The control GNN **had never been executed** when 2d was configured. It crashes on the first
training step under AMP:

```
RuntimeError: index_add_(): self (Float) and source (Half) must have the same scalar type
```

`index_add_` raises on mismatched scalar types rather than promoting. On CUDA, autocast sends
the edge MLP's output to half/bf16 while LayerNorm returns `h` in fp32, so from the second
message-passing layer onward an fp32 accumulator meets a half source. Found by smoke-testing
the config before committing a queue to it; it would otherwise have killed an overnight run on
its first step.

The regression test has to be a **CUDA** test — CPU autocast keeps LayerNorm in bf16, so the
dtypes agree there and a CPU version of the test passes with or without the fix.

---

## 8. Status as of 2026-08-21 21:00 UTC

- NCA seeds 0 and 1: **complete**
- GNN seed 0: **epoch 4 of 8**
- GNN seed 1: queued
- Evaluation: **not yet run for any arm** — training metrics are not the comparison quantity.
  The headline is RMSE vs lead on held-out test 2020, via `wnca eval`.

Partial training numbers, for orientation only — these are *not* the result, and the control is
half-trained:

| epoch | NCA seed 0 · selection | GNN seed 0 · selection |
|---|---|---|
| 1 | 0.22740 | 0.70265 |
| 2 | 0.19875 | 0.59331 |
| 3 | 0.15742 | 0.54636 |
| 4 | 0.14693 | 0.47589 |

The control is behind by ~3× on the 72 h rollout metric and is closing slowly. Its **train**
loss is only ~7% worse at epoch 4 (0.0568 vs 0.0415) while its **val** loss is 4.5× worse
(0.1119 vs 0.0249) — and the sign of the train/val relationship flips between arms: the NCA's
val sits *below* its train loss (it trains on harder pushforward states and validates on clean
single steps), the control's sits well above. It fits the training distribution and then falls
apart on rollout.

Whether that survives to epoch 8, evaluation on the test split, and the hyperparameter fairness
check of §4 is the whole question.

---

## 9. Checklist before this is written up as a result

- [ ] All four runs complete
- [ ] `wnca eval` on **test 2020** for all four checkpoints, `--checkpoint` passed explicitly
      (the disk carries 2c checkpoints too, and eval globs `best_*.pt`)
- [ ] Hyperparameter fairness probe on the control arm (§4)
- [ ] Report the gap against the **0.21% seed noise floor**, not against nothing
- [ ] State the two-variable caveat (§5) in the headline, not a footnote
- [ ] Quote 2 m temperature **off the 24 h grid** — persistence is diurnally aligned at 24 h
      multiples and the standard scorecard leads are all multiples of 24
- [ ] Exit criterion 5 is still unamended and 2c fails it as written (27/28 channels above
      1.05) while being the best model in the ladder. Fix the gate before applying it to 2d.
