# Cloud compute: environment, and how every run broke

Companion to `cloud-setup.md`, which says how to *start* a cloud run. This file records what
has gone *wrong* on cloud compute and why, so the same hour is not spent twice.

Scope: phase 2c only. Phases 0–2b ran locally on a GTX 1660 Ti and are not covered here.

---

## 1. Platform and configuration

Google Cloud Platform, Compute Engine. No managed ML service — a plain VM we SSH into, because
the workload is a single long-running process with a large local cache, and the managed
offerings add cost and indirection without adding anything we need.

| | |
|---|---|
| Instance | `wnca-2c`, zone `us-east1-c` |
| Machine type | `g2-standard-8` — 8 vCPU, 31 GB RAM |
| GPU | 1× NVIDIA L4, 23,034 MiB, driver 580.173.02, compute capability 8.9 |
| Provisioning | **on-demand** (`STANDARD`), not spot — see §4.1 |
| Disk | 300 GB balanced persistent disk, 91 GB used |
| Rate | ~$0.89/h |
| OS / Python | Ubuntu deep-learning image, Python 3.10.12 |
| PyTorch | 2.9.1+cu129 |
| Data source | WeatherBench-2 zarr on GCS, streamed once into a local 64 GB float32 cache |

Compute capability 8.9 ≥ 8.0, so **bf16 is native** and `GradScaler` is unnecessary. This is
checked at runtime via `torch.cuda.get_device_capability()` rather than
`torch.cuda.is_bf16_supported()`, which also returns `True` for slow emulated support on Turing.

Spend to date: **~$25** of the $300 free credit — ~$12 across attempts 1–3, ~$11.70 on attempt
4, remainder on benchmarks.

---

## 2. The runs, and how each one broke

All four are the same phase — 2c, 39 years × 28 channels, 7,122 batches/epoch, 8 epochs,
56,976 total steps. Only the marked variable changed between them.

### Attempt 1 — fp16 AMP, `lr 1e-3`

**Broke:** forward pass overflowed partway through epoch 1; 143 non-finite batches.

`torch.sparse.mm` has no fp16 CUDA kernel, so perception/spectral/wb2 were forced to fp32, and
the surviving fp16 path overflowed anyway. The cotangent Laplacian scales as 1/h², and at
n_sub=5 its block reaches ~13,097 against fp16's 65,504 ceiling — **5× headroom**, which the
weights consume as they grow.

**Read at the time:** "fp16 is too narrow." Correct, but not the cause of the run failing.

### Attempt 2 — bf16, `lr 1e-3`

**Broke:** ~700 non-finite gradients. bf16 has fp32's exponent range, so the overflow
explanation from attempt 1 could not apply, and it failed anyway.

### Attempt 3 — fp32, `lr 1e-3`

**Broke:** clean for the first minutes, diverged by ~40 min.

Three precisions, three failures. **The precision hypothesis was wrong.** A fixed-LR sweep
(`scripts/diagnose.py --stages lr`, 500 steps each) found the real cause:

| lr | max grad norm | non-finite / 500 |
|---|---|---|
| **1.0e-3** | **4.23e+12** | **287 (57%)** |
| 5.0e-4 | 10.8 | 0 |
| 3.0e-4 | 5.86 | 0 |

A cliff, not a slope — halving the LR drops the peak gradient norm by ~11 orders of magnitude.

**Why earlier phases survived the same nominal LR:** not the value, the **schedule length**.
2b-pushforward ran 2,912 steps, so cosine decay reached ~9e-5 by step 2,337. 2c's schedule spans
56,976 steps and sits within 0.3% of peak for thousands of steps — 16,817 steps above 8e-4
versus 859. Scaling the data 20× silently changed the optimisation problem. **An LR validated on
a short phase does not transfer to a long one.**

### Attempt 4 — bf16, `lr 3e-4` (the LR fix)

**Broke:** diverged during epoch 2, then ran **10 more hours at 100% GPU taking zero optimizer
steps.**

| epoch | train_loss | val_loss | selection |
|---|---|---|---|
| 1 | 19.9 | 0.253 | 7.28 |
| 2 | 3.69e23 | 1944617488.612022 | NaN |
| 3 | 8.40e23 | 1944617488.612022 | NaN |
| 4 | 8.36e23 | 1944617488.612022 | NaN |

`val_loss` byte-identical across epochs 2–4 is the signature: the weights were destroyed during
epoch 2 and then frozen, because every subsequent gradient was non-finite and every step
skipped. 23,458 non-finite gradient events ≈ one per batch.

Two distinct failures compounded here:

**(a) The LR sweep was underpowered.** It ran 500 steps; onset at 3e-4 is at ~7,000 steps. A
500-step probe cannot detect a 7,000-step onset — it could only ever have returned "clean." This
is M1 incident 3 recurring: a short proxy misfiring confidently where the direct measurement
would not have. The real warning was in plain sight and misread — epoch 1's `train_loss` of 19.9
was two orders of magnitude above every comparable phase, and was attributed to a warmup
artifact.

**(b) The divergence guard had a hole — this is what made it expensive.** The guard aborted on
non-finite **loss**, but 3.69e23 is finite in fp32. The gradient path only skipped the step:

```python
if assert_finite(float(gn), "gradient norm"):
    self.opt.step()          # no counter, no abort
```

So the loss-based counter stayed at 0, the abort never fired, and the run burned ~$9 doing
nothing. The RuntimeError it would have raised tells you to resume at a lower LR — it never got
the chance to print it.

**Fixed:** the gradient path now counts and aborts on the same 2%-of-epoch tolerance, with
regression tests in `tests/test_train_loop.py`
(`test_sustained_nonfinite_gradients_raise_even_though_the_loss_is_finite`, and its counterpart
asserting isolated spikes are still tolerated).

### Attempt 5 — onset measurement (diagnostic; succeeded)

Resumed from attempt 4's epoch-1 checkpoint with `WNCA_GRAD_TRACE=1`, same config and same
`epochs: 8`, so the cosine schedule position at step 7,122 was identical and the divergence was
reproduced rather than approximated. **The new guard aborted after 143 non-finite gradients**
instead of idling for 10 hours — the fix from attempt 4 paid for itself on its first run.

Divergence reproduced at **step 11,103**, mid-epoch-2. Onset is a four-step runaway:

| step | grad_norm | loss |
|---|---|---|
| 11,050 | 5.58 | 0.219 |
| 11,099 | 2.97e5 | 63.9 |
| 11,100 | 7.68e11 | 3.01e8 |
| 11,102 | 2.63e17 | 1.61e13 |
| 11,103 | **inf** | 7.00e23 |

For the preceding 3,900 steps the model was **entirely healthy** — median grad_norm 0.31, loss
~0.06, versus 2b's comparable 0.056. So epoch 1's `train_loss` of 19.9 was an early-epoch
transient the model recovered from, not evidence of a sick model.

**The weights are not what exploded.** Gradient clipping held throughout:

| | step 7,150 | step 11,050 | step 11,100 | after guard (11,200) |
|---|---|---|---|---|
| `layers.0.weight` norm | 47.95 | 54.75 | 54.91 | 54.92 (frozen) |

The weight norm moved **+0.3%** across the blowup and is frozen at a perfectly ordinary 54.92
afterwards — yet every subsequent batch produces a loss of ~1e24. A normal-looking weight tensor
that reliably produces 1e24 is the whole finding:

> **The 20-sub-step recurrence has a stability threshold, and the weights drifted across it.**
> Below it the composed update map is contractive and the loss sits at 0.06; above it the map is
> explosive and *every* batch blows up. Crossing is a property of the weights, not of any
> particular batch, which is why it is irreversible and why the frozen weights stay broken.

The per-layer breakdown at onset points at the input layer: post-clip gradient mass concentrates
almost entirely in `layers.0.weight` (0.96 of a 1.0-clipped total, up from 0.41 when the mass was
spread evenly across all four layers and the head). That is the layer consuming the perception
stencil, whose Laplacian block carries the 1/h² scaling.

**Why the LR fix only postponed it.** Weight norm grows ~linearly during healthy training:

| | measured |
|---|---|
| growth rate, `layers.0.weight` | 1.745e-3 /step at lr 2.9e-4 |
| growth over the traced window | 47.95 → 54.75 (+14.2% in 3,900 steps) |
| decay pull, `wd × lr × ‖w‖` | 1.59e-7 /step |
| **ratio** | **weight decay is ~11,000× too weak to oppose it** |

`weight_decay = 1e-5` is effectively zero here, so the weight norm ratchets upward until it
crosses the threshold. Lower LR slows the ratchet and moves the onset later — it does not change
the destination. The onset times are consistent with exactly that: ~2,000 steps at `lr 1e-3`
versus ~11,100 at `3e-4`, against an LR ratio of 3.4×.

**This reframes the whole 2c diagnosis.** The learning rate is the *driver*, not the root cause.
Attempts 1–4 tuned precision and then LR — both of which change *when* the threshold is reached,
neither of which stops it being reached. A fix has to bound the weights or the recurrence itself
(weight decay of a useful magnitude, spectral normalisation on the update MLP, or a stronger
`w_over` overflow penalty — currently 1e-2). **Chosen and implemented:** spectral norm on the
update MLP's hidden layers plus `weight_decay: 0.1` — see `docs/phase2c-stability-fixes.md`.
`w_over` was rejected: it watches hidden-channel overflow, and the measured failure was gain
in the composed map with ordinary-looking weights — the wrong mechanism.

---

## 3. Cross-cutting lessons

0. **Four of the five attempts treated a symptom.** Precision (×3) and then learning rate (×1)
   all change *how fast* the weights drift into the sub-step stability threshold; none of them
   stops the drift. Each fix "worked" in the sense that the run survived longer, which is
   exactly the signature of a driver rather than a cause — and is why the LR fix looked
   confirmed for a full clean epoch before failing.
1. **Three precision changes treated a symptom.** The variable under test was never isolated;
   the LR was constant across all three and was the actual cause of *those* three failures.
2. **Probe length must exceed the onset time you are trying to rule out.** A clean 500-step
   result is not evidence of stability at 7,000 steps, and reporting it as such is what turned a
   known-unknown into a 13-hour run.
3. **Every guard needs to cover the finite-but-absurd case.** Divergence does not always produce
   a NaN.
4. **On cloud, a silent failure costs money at a fixed hourly rate.** Locally a stuck run is
   free. This inverts the value of fail-fast instrumentation.

---

## 4. Operational and tooling problems

These cost real time and are unrelated to the model.

### 4.1 Spot machinery is not fit for purpose as written

The SIGTERM handler runs validation *and* the selection metric before checkpointing — 3+ minutes
— against spot's ~30-second preemption notice. Every 2c run therefore used **on-demand**
provisioning, forgoing the ~60% spot discount. **Fixed:** when the flag is set during the
training epoch, `fit` now writes `preempted.pt` immediately after the training pass, before any
scoring runs (regression test: `test_preemption_checkpoints_before_scoring`). Spot is usable
again.

### 4.2 Log output appeared frozen for 3 hours

`run_phase.sh` invoked `python3` without `-u`. Redirected to a file, stdout is block-buffered, so
`tail -f` showed nothing while training ran normally. Diagnosed by reading `metrics.jsonl` and
the checkpoint mtime instead. **Fixed:** `python3 -u` in `run_phase.sh`. When a log looks frozen,
check artifact mtimes before concluding the run is stuck.

### 4.3 `gcloud compute ssh --command` quirks

- A remote command exiting non-zero (e.g. `grep` with no match) surfaces as an SSH error via
  `plink exit 1`. **Always append `; true`.**
- Nested single quotes inside the command string break gcloud's parser. Use `--command="..."`
  with the `=` form, and avoid inline `python3 -c '...'`.
- `pkill -f 'wnca.cli train'` **matches the SSH command string that carries it** and kills its
  own session (exit 128). Use a pattern that cannot self-match.
- SIGTERM is caught by the handler in §4.1, so a kill can appear to do nothing. `pkill -9` when
  the process state is already worthless.

### 4.4 `gcloud compute scp` does not expand `~`

`pscp: unable to open ~/weather-nca/...`. Use absolute remote paths (`/home/<user>/...`).

### 4.5 `gsutil ls -L -b` returns 403

Bucket *metadata* permission is not granted on the public WB2 bucket, though object reads work
fine. Not a misconfiguration — test throughput by reading an object instead.

### 4.6 Python version warnings

The deep-learning image emits `FutureWarning`/version warnings on import. Benign; filter with
`grep -vE 'Warning|warnings.warn'`.

---

## 5. Standing checklist before a cloud phase

- [ ] `make smoke` passes locally.
- [ ] LR validated over a probe **at least as long as the schedule's near-peak region**, not a
      fixed 500 steps — or use `scripts/diagnose.py --stages lr`, which now extrapolates the
      weight-norm ratchet's crossing step from a short probe (σ_max tracking when
      `model.spectral_norm` is on).
- [ ] Divergence guards cover non-finite loss, non-finite gradients, *and* finite-but-absurd
      loss (100× running median).
- [ ] `python3 -u` so the log is readable in real time.
- [ ] A budget alert is set, and the expected wall-clock × hourly rate is written down first.
- [ ] Instance deletion command noted before launch, not after.

---

## 6. Teardown

Stopping preserves the 64 GB cache and checkpoints; deleting does not.

```bash
gcloud compute instances stop   wnca-2c --zone=us-east1-c          # keeps disk, stops GPU billing
gcloud compute instances delete wnca-2c --zone=us-east1-c --quiet  # when the phase is finished
```

An idle running instance still bills at the full ~$0.89/h.
