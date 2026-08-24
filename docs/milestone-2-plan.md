# Milestone 2 — Probabilistic Multi-Variable Weather NCA (v2)

**Goal:** Determine whether a strictly local update rule loses accuracy relative to a non-local model of equal budget, and whether it can be made to produce a calibrated, sharp ensemble.

Milestone 1 answered *can a strictly local rule advect?* — yes, on real ERA5: 24 h z500 RMSE 344.7 vs persistence 593.9, stable to 7 days, spectrum preserved at 24 h, on 27.5k parameters and two years of data. M2 asks the follow-up that actually decides whether to keep going.

---

## What changed from v1

| # | Change | Reason |
|---|---|---|
| 1 | **Added a same-budget control model** | Distance-to-frontier confounds architecture with compute. Without a control, a 2× gap is uninterpretable. |
| 2 | **0.25° fine-tune moved to M3** | It answers neither M2 question and carries most of the compute risk. |
| 3 | **Noise perturbs the update rule, not the seed** | 20 sub-steps of a local averaging operator will erase a seed perturbation. Ensemble collapse was the *expected* outcome of the v1 design, not a risk. |
| 4 | **Spectral term moved from Deferred into the M2 loss** | FCN3: pointwise CRPS produces well-scored ensembles whose members are spectrally wrong. |
| 5 | **Fair CRPS estimator mandated** | At M = 4–8 the naive estimator is biased toward under-dispersion by a large factor. |
| 6 | **Step 2 split into 2a / 2b** | v1 changed variables and data volume simultaneously and claimed to separate them. |
| 7 | **Parameter count freed** | M1's 27.5k (not ~100k — see Capacity) was neither a VRAM constraint nor a design principle. 20+ variables and 39 years need more capacity. |
| 8 | **Exit criterion is a slope, not a number** | "Under 2× frontier" may be unreachable on spot instances regardless of whether the architecture is sound. |
| 9 | **True pushforward added alongside existing training noise** | M1's `noise_std = 0.05` is Sanchez-Gonzalez-style input noise, which is not the same mechanism as Brandstetter's pushforward. |
| 10 | **Mesh resolution labels corrected** | See below — the "1°" mesh is nearer 2°, and this materially changes how the frontier gap should be read. |

### Resolution labelling (read this before quoting the gap again)

The M1 mesh is `n_sub = 5`, 10,242 nodes, ~223 km mean spacing. That is closer to **2°** than to 1°. The v1 plan's 0.25° target of "nside = 64, ~163k nodes" is also off: 0.25° is ~28 km spacing, which needs on the order of 650k equal-area cells (HEALPix nside ≈ 256), not 49k (nside = 64) or 163k.

Consequence: part of the 6× frontier gap is a resolution mismatch roughly 8× larger than assumed. This is good news for the architecture and bad news for the M3 compute estimate. Fix the node-count arithmetic before budgeting M3.

---

## Scope

| | M1 (done) | **M2** | Deferred (M3+) |
|---|---|---|---|
| Variables | z500 only | z, u, v, t, q at 5 levels + 3 surface | Full 13-level state |
| Mesh | ~223 km (10k nodes) | ~223 km, unchanged | ~100 km, then ~28 km |
| Output | Deterministic | Probabilistic ensemble | Calibrated extremes |
| Data | 2 years | 2 yr → 1979–2017 (laddered) | + operational analyses |
| Loss | Area-weighted MSE | Fair CRPS + spectral band CRPS | + energy/conservation terms |
| Params | 27.5k | 1–5M | — |
| Baselines | Persistence | **Same-budget GNN control**, climatology, WB2 frontier | GenCast parity |

Resolution is deliberately frozen for the whole milestone. Every result in M2 is comparable to every other result in M2.

---

## Design decisions

### Carried forward unchanged

Icosahedral mesh; `[physical | hidden]` state layout; NCA sub-steps as PDE sub-steps; deterministic update mask (p = 1); second-order Markov conditioning; `reseed_hidden = True`; overflow penalty on hidden channels; area weighting by barycentric vertex areas; gradient checkpointing on the sub-step loop; `dt = 0.05`, `n_substeps = 20` as the starting sub-step budget.

Also carried forward, as methodology: **prefer direct measurement over clever measurement.** `perturbation_growth` was right every time in M1; the one-sided spectrum check and the finite-difference spectral radius both misfired confidently. M2 reaches for the direct diagnostic first.

### Probabilistic head — functional perturbation, not seed noise

A per-member noise vector **z ∈ ℝ^16** (16–32; start at 16) is drawn once per member per forecast trajectory and used to FiLM-modulate every hidden layer of the update MLP. It is held constant across all cells and all sub-steps within a member.

The reasoning is architecture-specific and is the single most important change in v2. The NCA applies 20 rounds of a smoothing local operator per forecast window. Any perturbation placed in the seed field is exactly the thing that operator is built to damp; the network does not have to learn to ignore it, the dynamics erase it for free. Perturbing the rule itself cannot be damped, because it changes the operator rather than the state it acts on. This is the FGN result (WeatherNext 2) — a single low-dimensional vector modulating network functions across the whole field — and the low dimensionality is load-bearing: it is what forces the perturbation to express itself as coherent, physically plausible spatial structure rather than per-cell noise.

**Decision to record explicitly:** the noise vector persists for the entire trajectory, giving each member a stable identity across windows, analogous to a stochastic-physics ensemble. `reseed_hidden = True` re-seeds the hidden channels each window but must *not* re-draw z. If it does, member identity dissolves and spread stops meaning anything at long leads.

### Loss — composite, both terms proper

```
L = w_field · fairCRPS(members, truth)          # area-weighted, per variable/level
  + w_spec  · fairCRPS(log band energies)       # Chebyshev bands on the mesh Laplacian
  + w_over  · overflow_penalty(hidden)          # carried from M1
```

Field term uses the **fair** estimator:

```
CRPS_fair = (1/M) Σᵢ |xᵢ − y|  −  1/(2M(M−1)) Σᵢ Σⱼ |xᵢ − xⱼ|
```

The naive estimator divides the second term by 2M² instead, which under-weights spread and trains toward over-confidence. At M = 4 the difference is 33%. AIFS uses an *almost*-fair variant that blends the two with α ≈ 0.95 for gradient stability; keep α configurable and start at 1.0 (fully fair).

Spectral term: the cotangent Laplace–Beltrami operator already exists from M1, so band-pass filters come free via Chebyshev polynomials of the scaled Laplacian — no eigendecomposition, just a few sparse matvecs. Compute per-member log band energies, then score them against truth's with the same fair CRPS. That keeps the whole objective inside one proper-scoring framework rather than bolting an MSE term onto a CRPS loss.

Whether the spectral term is needed *for an NCA* is genuinely unknown — the local-rule inductive bias may already preserve the spectrum, as M1's 100.9% retained power at 24 h weakly suggests. Phase D runs it as an ablation.

### Multi-variable state

Prognostic: geopotential, u, v, temperature, specific humidity at **850 / 700 / 500 / 300 / 250 hPa**, plus surface **10u, 10v, 2t**. That is 25 + 3 = 28 physical channels. Hidden channels rise from 15 to ~32. Static channels unchanged.

The surface additions exist because the v1 evaluation plan scored CRPS on 10u without forecasting it. Either forecast it or drop it from the scorecard; forecasting it is more useful and costs three channels.

Levels chosen to bracket 500 hPa (geostrophic coupling to the wind is the main expected z500 gain), include 850 for the moisture and low-level thermal structure, and 250–300 for the jet.

### Capacity

M1's parameter count was **27,536**, not ~100k (corrected 2026-08-16 against the notebook's own output; see `milestone-1-findings.md`). And it was **not** a 6 GB-VRAM constraint: measured, the exact M1 configuration peaks at **0.37 GB** — 6% of the card. Both halves of the original justification were wrong.

The real reason to raise capacity is the one that survives: with 28 physical + 32 hidden channels the perception output alone is 240-dimensional, so the input layer is ~8× wider than M1's before any depth is added. Capacity should scale with the state, not because VRAM was previously binding.

Target **1–5M parameters** (`hidden_dim = 512`, `n_layers = 4`) — still an order of magnitude under DLWP-HPX's 9.8M, the nearest published small-model precedent. But note this is a **36–180× jump from 27.5k**, not the ~10–50× the old figure implied, and the measured profile says runtime is ~80% update-MLP scaling with `hidden_dim²`. Sweep capacity once, early, at fixed data — and treat the sweep as a compute-budget decision, not only an accuracy one. `hidden_dim = 256` (318k params) is a legitimate landing spot if the accuracy gain from 512 is small.

### The control model

A non-local model matched on mesh, variables, data, parameter count, optimizer, and wall-clock budget. Cheapest path is Neural-LAM's GraphCast reimplementation (`github.com/mllam/neural-lam`); a plain multi-scale message-passing GNN over the same mesh is acceptable and easier to budget-match.

This turns the milestone's central question from "how far are we from DeepMind" — unanswerable on four spot GPUs — into "does restricting the update to a strictly local rule cost accuracy, holding everything else fixed," which is answerable and is the actual thesis under test.

---

## Data

- **ERA5** via WeatherBench-2 Zarr on GCS. The M1 `load_era5` axis-order transpose fix is load-bearing; keep the raw-field sanity plot for every new variable.
- Splits: **train 1979–2017, validate 2018, test 2020.** 2020 matches the WB2 probabilistic leaderboard's 732 initial conditions.
- Normalization per variable **and per level**, computed on train only. Specific humidity spans orders of magnitude across levels — log-transform q before normalizing, or the 850 hPa channel will dominate every gradient.
- Climatology for the baseline comes from train only.
- Cache mesh-projected tensors to local disk as a memory-mapped array. Recomputing the regrid every epoch on a spot instance is the single biggest avoidable time sink.

---

## Evaluation

**Primary:** ensemble-mean z500 RMSE at 24 h and 72 h, plotted against (a) the same-budget control, (b) climatology, (c) published WB2 frontier numbers — in that order of interpretive weight.

**Secondary, in order:**

1. **CRPS at 1 d / 3 d / 10 d** for z500, t850, 10u. IFS ENS reference for z500: 22.4 / 58.3 / 262.
2. **Spread–skill ratio.** Target 1.0. Report per variable and per lead; this is the check MSE cannot pass by construction.
3. **Band-energy spectrum vs ERA5** at 24 / 72 / 120 h, for *individual members*, not the mean.
4. **Rollout stability to 15 days** via `perturbation_growth`, per variable.
5. **Rank histogram**, once 1–4 pass.

**Two comparability requirements, both work items rather than footnotes:**

- WB2 probabilistic numbers are at **1.5° lat-lon**. Scoring the mesh against them requires an explicit regrid, and the regrid operator must be conservative (area-weighted), not nearest-neighbour, or the RMSE will be biased. Build it once, test it with a round-trip, reuse it.
- WB2 scores operational models against IFS analysis and ML models against ERA5. This project is on the ERA5 side: the GenCast comparison is like-for-like, the IFS ENS comparison is slightly favourable. Put that in a comment in the scoring script so nobody has to rediscover it.

RMSE is pooled across start times with the square root taken last (per-start RMSE averaged instead is biased low by Jensen and is not comparable to published numbers). This is already M1 practice — preserve it.

---

## Sequence of work

Each phase has a gate. Do not proceed past a failed gate without writing down why.

| Phase | What | Isolates | Gate |
|---|---|---|---|
| **0** | Port M1 into the repo, reproduce 344.7 | — | 24 h z500 RMSE within 2% of M1 |
| **2a** | M1 config, single variable, **full 39 yr** | Data volume | Documented RMSE vs the 2-year number |
| **2b** | Multi-variable, **2 years** | Coupling | Documented RMSE vs 2a |
| **2c** | Multi-variable, full data, capacity sweep | Combined + capacity | Best deterministic checkpoint |
| **2d** | **Control GNN**, matched budget | Locality | Control trained and scored |
| **3a** | Probabilistic: FiLM noise + fair CRPS, warm-started from 2c | Loss + noise | Spread–skill > 0.3 by epoch 3 |
| **3b** | Add spectral band term | Whether it's needed | Ablation documented either way |
| **4** | Score on 2020, all metrics | — | Scorecard complete |
| **5** | Decide | — | — |

Phases 2a and 2b are cheap and answer a question v1 claimed to answer but could not. 2a in particular reuses a model that already works, so it is the lowest-risk run in the milestone.

---

## Training regimen

**Deterministic phases (2a–2c).** Single-step training, area-weighted MSE, AdamW, lr 1e-3 with cosine decay, grad clip 1.0, warmup 500 steps. Batch as large as memory allows; prefer larger batch over larger lr. `rollout_epochs = 0` by default — M1 showed the curriculum contributed nothing across two attempts and destroyed the long-lead metric twice.

**Rollout drift.** Before re-enabling any curriculum, add a **true pushforward step**: unroll one forecast window with `torch.no_grad()`, then take the training step from that state. This is a different mechanism from M1's `noise_std = 0.05` seed noise, which is Sanchez-Gonzalez-style input perturbation. Pushforward trains the rule to contract error from states it actually produces, at roughly 1.5× the cost of single-step and none of the instability of backpropagating through 160 sub-steps.

**Probabilistic phase (3a).** Warm-start from the best 2c checkpoint. Initialize FiLM scale to 1 and shift to 0 so the model begins as an exact copy of the deterministic one, and let the noise pathway grow. Drop lr to 1e-4. `M = 4` members in training, `M = 16` for validation scoring, `M = 50` for the final scorecard.

**Anti-collapse instrumentation, from step zero of 3a:**

- Log ensemble spread every validation pass. If spread < 0.05× RMSE by epoch 3, the noise is not reaching the dynamics — stop and fix the conditioning rather than training through it.
- Run a **zero-noise ablation** at every validation: forward with z = 0 and compare to the ensemble mean. If they are identical, the noise is decorative.
- Log the FiLM layer's gradient norm separately from the rest of the network.

**Checkpoint discipline (M1 incidents, do not repeat):**

- One fixed selection metric across every phase — `ckpt_windows`-window rollout score on validation — and never compare metrics across phase boundaries.
- Timestamped checkpoint filenames. `best.pt` as a single fixed path clobbered comparisons in M1.
- On load, assert the head weight norm is non-zero and the architecture hash matches. Three M1 evaluation rounds were run on a zero-init identity model that reproduced persistence exactly. This assert is not optional; it is a regression test for a bug that already happened.
- Finiteness guard on both the gradient step and checkpoint selection.

**Spot-instance resumption.** Checkpoint every N steps *and* on SIGTERM. Store optimizer state, RNG state, epoch, and step. The data cache must be rebuildable and resumable independently of training state.

**Compute cost — measured, 2026-08-16.** The dominant term is `M × n_substeps` network evaluations per forecast step — at M = 4 and 20 sub-steps that is 80 MLP passes over ~10k nodes per sample, before backward.

Measured with `scripts/benchmark_step.py` on the local **GTX 1660 Ti (6 GB, no tensor cores)**, fp32, gradient checkpointing on, `n_sub = 5` (10,242 nodes), 28 physical + 32 hidden channels, one forecast step forward **and** backward:

| hidden_dim | B | M | ms/step | peak GB | params |
|---|---|---|---|---|---|
| 512 | 1 | 1 | 1,195 | 0.29 | 1.03 M |
| 512 | 2 | 1 | 2,213 | 0.54 | 1.03 M |
| 512 | 1 | 4 | 4,800 | 1.05 | 1.03 M |
| 512 | 2 | 4 | 7,350 | 2.08 | 1.03 M |
| 256 | 2 | 4 | 3,864 | 1.33 | 318 K |
| 256 | 4 | 4 | 6,619 | 2.63 | 318 K |
| 128 | 8 | 4 | 10,632 | 3.87 | 110 K |

Two corrections to the paragraph this replaces:

1. **The memory estimate was right, and memory is not the constraint.** Peak stays under 4 GB in every configuration tried; cost is linear in `B × M` exactly as predicted. 6 GB is not binding.
2. **Time is the constraint, and it is worse than implied.** Cost is also linear in `B × M`, which means there is no batching efficiency to recover — at `hidden_dim = 512, B = 2, M = 4` a single training step costs 7.35 s, i.e. **3.7 s per sample**. One epoch over the full 39-year split (56,979 samples) is therefore ~58 h on this card, and phase 2c at 20 epochs is ~48 days. Locally infeasible by two orders of magnitude, exactly as the plan assumed — but the number is now measured rather than assumed.

Profiling the step (`hidden_dim = 512`, `B × M = 8` rows): the update MLP is **84 ms** and mesh perception is **20 ms** per sub-step, so ~80% of the cost is the MLP and it scales with `hidden_dim²`. The capacity sweep in 2c is therefore also the compute-budget decision, not just an accuracy one.

Implication for the cloud budget: an A100 runs fp32 roughly 15–25× this card, and more with TF32/AMP (both wired up; `train.amp` is off by default because Turing has no tensor cores and it is a no-op locally). That puts phase 2c near 2.5–4 h/epoch and ~2–3 days per full phase, so 2c + 2d + 3a + 3b is on the order of **10 GPU-days**. Re-run the benchmark on the actual cloud instance before committing to that.

**Cache sizing — measured.** 39 years, 6-hourly, 10,242 nodes, 28 channels, float32 is **65 GB** on disk (the 2-year phase-2b split is 3.3 GB). `data.cache_dtype: float16` halves it. This is memmapped and built resumably in 256-timestep chunks, so a preempted build restarts at a chunk boundary.

---

## Implementation

### Repo, not notebook

M1 ran as a notebook and produced three evaluation rounds on an untrained model because rerunning one cell silently rebuilt the model as an identity map. That failure mode is structural to notebooks and is not worth risking again on a milestone with eight phases and a control model. Notebooks stay, but only as read-only analysis surfaces over artifacts the package produced.

### Layout

```
weather-nca/
├── CLAUDE.md                  # agent operating instructions (see below)
├── README.md
├── pyproject.toml             # package + pinned deps
├── Makefile                   # make test / smoke / train / eval
├── docs/
│   ├── milestone-1-findings.md
│   ├── milestone-2-plan.md
│   └── decisions/             # one file per irreversible choice
├── configs/
│   ├── base.yaml
│   ├── phase0_m1_repro.yaml
│   ├── phase2a_data.yaml
│   ├── phase2b_multivar.yaml
│   ├── phase2c_full.yaml
│   ├── phase2d_control.yaml
│   ├── phase3a_crps.yaml
│   └── phase3b_spectral.yaml
├── src/wnca/
│   ├── config.py              # dataclasses, YAML load, schema validation
│   ├── mesh/
│   │   ├── icosphere.py       # construction, refinement, vertex areas
│   │   ├── operators.py       # cotangent Laplacian, LS tangent gradient
│   │   ├── regrid.py          # mesh <-> lat-lon sparse matrices, cached
│   │   └── spectral.py        # Chebyshev band-pass filters
│   ├── data/
│   │   ├── era5.py            # WB2 zarr -> mesh tensors (the transpose fix lives here)
│   │   ├── normalize.py       # per-var/per-level stats, q log-transform
│   │   ├── cache.py           # memmap cache, resumable
│   │   └── dataset.py
│   ├── models/
│   │   ├── perception.py      # MeshPerception, unchanged from M1
│   │   ├── update.py          # UpdateRule + FiLM noise conditioning
│   │   ├── nca.py             # WeatherNCA
│   │   └── control_gnn.py     # same-budget non-local baseline
│   ├── losses/
│   │   ├── crps.py            # fair / almost-fair kernel CRPS
│   │   ├── spectral.py        # band-energy CRPS
│   │   └── terms.py           # area weighting, overflow penalty
│   ├── train/
│   │   ├── loop.py
│   │   ├── checkpoint.py      # timestamped, asserted, resumable
│   │   └── phases.py
│   ├── eval/
│   │   ├── metrics.py         # RMSE, CRPS, spread-skill, rank histogram
│   │   ├── perturbation.py    # the M1 diagnostic — the one that was always right
│   │   ├── spectrum.py
│   │   └── wb2.py             # 1.5° regrid + leaderboard comparison
│   └── cli.py
├── scripts/
│   ├── build_mesh.py
│   ├── build_cache.py
│   ├── benchmark_step.py      # run this first
│   ├── train.py
│   └── evaluate.py
├── tests/
│   ├── test_operators.py      # analytic convergence, from M1 §1
│   ├── test_regrid.py         # round-trip + conservation
│   ├── test_crps.py           # vs scoringrules; fair-estimator bias at small M
│   ├── test_spectral.py       # band energies of a known field
│   ├── test_noise.py          # zero-noise == deterministic; nonzero => spread
│   ├── test_checkpoint.py     # the untrained-model guard
│   └── test_shapes.py
└── notebooks/                 # analysis only, never authoritative
```

### The three pieces that don't exist yet

**1. FiLM-conditioned update rule** (`models/update.py`)

```python
class UpdateRule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        in_dim = 4 * cfg.c_state + cfg.c_cond
        self.layers = nn.ModuleList(
            [nn.Linear(in_dim, cfg.hidden_dim)]
            + [nn.Linear(cfg.hidden_dim, cfg.hidden_dim) for _ in range(cfg.n_layers - 1)]
        )
        # One (scale, shift) pair per hidden layer, produced from the noise vector.
        self.film = nn.Linear(cfg.noise_dim, 2 * cfg.hidden_dim * cfg.n_layers)
        nn.init.zeros_(self.film.weight)     # starts as an exact identity modulation:
        nn.init.zeros_(self.film.bias)       # gamma = 1, beta = 0  ->  deterministic model
        self.head = nn.Linear(cfg.hidden_dim, cfg.c_state)
        nn.init.zeros_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, perceived, cond, z):
        # z: [B, noise_dim] — one draw per member, broadcast over ALL nodes and sub-steps.
        mod = self.film(z).view(z.shape[0], len(self.layers), 2, -1)
        h = torch.cat([perceived, cond], dim=-1)
        for i, layer in enumerate(self.layers):
            gamma = 1.0 + mod[:, i, 0].unsqueeze(1)   # [B, 1, hidden]
            beta = mod[:, i, 1].unsqueeze(1)
            h = F.gelu(gamma * layer(h) + beta)
        return self.head(h)
```

Zero-init on `film` means phase 3a starts numerically identical to the phase 2c checkpoint it warm-starts from. That is the same trick as the zero-init head from the original NCA paper, applied to the noise pathway, and it is what makes the deterministic → probabilistic transition safe.

**2. Fair kernel CRPS** (`losses/crps.py`)

```python
def fair_crps(members, truth, weights, alpha=1.0):
    """members: [B, M, N, C]; truth: [B, N, C]; weights: [N] barycentric areas."""
    M = members.shape[1]
    skill = (members - truth.unsqueeze(1)).abs().mean(dim=1)          # [B, N, C]
    diffs = (members.unsqueeze(1) - members.unsqueeze(2)).abs()       # [B, M, M, C...]
    coef = alpha / (2 * M * (M - 1)) + (1 - alpha) / (2 * M * M)      # fair <-> biased
    spread = coef * diffs.sum(dim=(1, 2))
    return _area_weighted_mean(skill - spread, weights)
```

Test it against `scoringrules` and assert the fair estimator is unbiased as M grows while the naive one is not. Do not hand-roll this without the test; the small-M bias is exactly the regime where getting it wrong is invisible and fatal.

**3. Chebyshev band-energy spectral term** (`mesh/spectral.py`, `losses/spectral.py`)

The cotangent Laplacian is already built. Scale it to `L̃ = 2L/λ_max − I` (λ_max by power iteration, computed once and cached), then recurse `T₀ = x`, `T₁ = L̃x`, `Tₖ = 2L̃Tₖ₋₁ − Tₖ₋₂`. A band-pass filter is a fixed linear combination of the `Tₖ`; four to six bands is plenty. Band energy is the area-weighted mean square of the filtered field. Take logs, then score per-member log band energies against truth's with the same `fair_crps`.

Cost is a handful of sparse matvecs per band per member. It reuses the operator that M1 already validated against spherical harmonics, so there is no new geometry to get wrong.

### Config strategy

One `base.yaml` plus per-phase overrides, loaded into frozen dataclasses with schema validation at startup. Every run writes its fully-resolved config next to its checkpoints. Reproducing a run means pointing at that file, not remembering which cell was edited.

### Testing

`make test` must run in under 60 seconds on the 1660 Ti. Two tests carry disproportionate weight:

- `test_operators.py` — the analytic convergence check from M1 §1, re-run whenever mesh code is touched. It caught the flat-space Laplacian failure.
- `test_checkpoint.py` — build, train one step, save, reload, assert the head norm is non-zero and predictions differ from the identity map. This is a regression test for the specific incident that wasted three M1 evaluation rounds.

Add `make smoke`: a full train → eval cycle on `n_sub = 3` (642 nodes) with one month of data, target under two minutes, runnable locally. Every phase runs smoke before it runs on cloud.

### CLAUDE.md

The agent's usefulness will be set almost entirely by this file. Keep it short and load it with invariants and prior failures, not with descriptions of what the code does — it can read the code.

Include: the three time granularities (`nca_step` / `forecast_step` / `rollout`) and that confusing them is the most common error; the standing methodology note about preferring direct measurement; the four M1 incidents (untrained-model evaluation, incommensurable checkpoint metrics, one-sided spectrum check, `best.pt` clobbering) as things that have already happened; hard constraints (normalization from train split only, RMSE pooled before square root, `reseed_hidden` semantics, noise vector constant across cells and sub-steps); commands (`make test`, `make smoke`, `make train CONFIG=...`); where decisions get written down (`docs/decisions/`); and an instruction to add a regression test whenever a numerical result changes unexpectedly.

Explicitly tell it *not* to re-enable the rollout curriculum, expand perception to two hops, or add a semi-Lagrangian pre-advection step without a measurement justifying it — M1 concluded all three were unjustified, and an agent optimizing a loss will reach for them.

### Tracking

Weights & Biases, one project, one run per phase, config logged as the resolved YAML. Log spread–skill, per-band energies, and the zero-noise ablation as first-class metrics from phase 3a, not as afterthoughts — they are the phase's gate.

### Migration order

1. Scaffold the repo, `pyproject.toml`, `Makefile`, `CLAUDE.md`, CI running `make test`.
2. Port mesh + operators from the notebook. Get `test_operators.py` green — this reproduces a known-good M1 result and validates the port.
3. Port data pipeline. Round-trip a raw field to lat-lon and eyeball it.
4. Port the model and training loop.
5. **Reproduce M1's 344.7 within 2%.** This is the migration's acceptance test. Do not start phase 2a until it passes.
6. `scripts/benchmark_step.py`, then write the real compute budget into this document.

---

## Risks

**Ensemble collapse.** Downgraded from v1's framing but not eliminated. Functional perturbation makes it much harder for the network to ignore the noise, but the low-dimensional bottleneck can still degenerate. Instrumented in phase 3a with three independent signals; a fix means reworking where the FiLM modulation attaches, not adding more noise.

**CRPS training instability.** CRPS over a small in-batch ensemble is noisier than MSE, and M1 diverged twice at learning rates that seemed conservative. Warm-starting from a deterministic checkpoint with zero-init FiLM is the main mitigation; low lr and the fixed selection metric are the second and third.

**Sub-step budget under coupling.** `n_substeps = 20` was measured on one band-limited variable. Coupled dynamics carry faster adjustment processes. Confirm with `perturbation_growth` on the multi-variable state at phase 2b, per variable — a single-window CFL test cannot see per-window amplification that compounds, which M1 established the hard way. Mitigation order unchanged: more sub-steps, larger perception radius, then semi-Lagrangian pre-advection.

**Control model consumes real budget.** Roughly 25% of the milestone's compute. It is still the highest-value spend here, because without it the headline result is uninterpretable.

**Spot preemption.** Checkpoint on SIGTERM; cache resumable; never let a run's only state live in GPU memory.

**Over-reading a matched spectrum.** M1's 100.9% retained power at 24 h means the right *amount* of variance per scale, not the right *placement* — the error maps showed real phase error along the fronts. Band energies are a necessary check, not a sufficient one. Reliability and CRPS are the calibration evidence.

---

## Exit criteria

Move to M3 when all five hold:

1. **Locality is quantified.** The NCA's 24 h and 72 h ensemble-mean z500 RMSE is documented against the same-budget control. Within 1.2× is a strong result; beyond 2× is a real finding that the local restriction costs accuracy, and is equally worth having.
2. **The gap closes with scale.** The 2a → 2b → 2c ladder shows RMSE improving with data and coupling rather than plateauing. The *slope* is the deliverable, not any single number.
3. **The ensemble is real.** Spread–skill ratio in 0.8–1.25 for z500 at 1 d and 3 d, and a zero-noise ablation that is measurably different from the ensemble mean.
4. **Members are individually sharp.** Band energies of individual members within 20% of ERA5 at 72 h, with the spectral-term ablation documented either way.
5. **Error growth is physically plausible, and the forecast is bounded.** Superseded 2026-08-24;
   see below.

### Criterion 5, amended

The original read: *stable to 15 days, `perturbation_growth` ≤ ~1.05 per window, per variable.*
That threshold is **wrong in kind, not just in value**, and every model in the ladder that is
any good fails it:

| model | sustained growth | doubling | 24 h z500 |
|---|---|---|---|
| phase 0 | ×1.033 | 5.34 d | 329.7 |
| 2b-pushforward | ×1.139 | 1.33 d | 282.1 |
| phase 2c (best) | ×1.079 | 2.28 d | **174.7** |

Only phase 0 passes ≤1.05, and phase 0 is the *worst* forecaster in the table. The criterion
selects for **over-damping**: a model that smooths its own errors away scores well on it and
forecasts badly. The real atmosphere doubles synoptic-scale errors in roughly 1.5–2.5 days, so a
model at 5.34 d is not stable, it is sluggish.

This matters more for phase 3 than it did for phase 2. **Over-damping directly suppresses
ensemble spread**, which is what 3a exists to measure — so applying the original criterion to a
probabilistic phase would select against the phase's own objective.

**Amended criterion 5.** Both must hold:

  a. **Error-doubling time in 1.2–3.5 days**, computed from sustained per-window
     `perturbation_growth` as `ln 2 / ln(g⁴)`, on the pooled state. A band, not a ceiling: too
     slow is over-damped, too fast is unstable. The 1.5–2.5 d synoptic range sits inside it with
     room for measurement error.
  b. **Bounded at long lead**: 15-day RMSE ≤ 1.5× climatology, which is the property "stable to
     15 days" was actually reaching for. Phase 2c sits at 1.34×; 2b was at 3.23× and genuinely
     diverging.

**Report doubling time with error bars.** Two seeds of the same phase-2c model differ by 12% on
this quantity (2.28 d vs 2.01 d), so a single run cannot place a model in or out of the band on
its own — see `docs/phase2d-results.md` §4 and §8.

Beyond M2: mesh refinement toward 100 km then 28 km (with the corrected node-count arithmetic), the full 13-level state, calibrated extremes, and the project's original architectural bet — Lagrangian feature particles layered on the Eulerian base, the Neural Particle Automata direction the mesh design was chosen to allow.
