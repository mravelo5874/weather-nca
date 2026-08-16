# Milestone 2 — Background Reading

Curated for the three things M2 actually has to solve: **a probabilistic head that doesn't collapse**, **coupled multi-variable dynamics**, and **staying calibrated and sharp under rollout**. Annotations say why each item matters here, not what it is.

Ordering within each tier is by priority, not date.

---

## Tier 0 — The four that change the M2 plan

**Skillful joint probabilistic weather forecasting from marginals (FGN)** — Alet, Price, El-Kadi, Masters et al., DeepMind, 2025. `arXiv:2506.10772`
The single most important paper for M2. FGN is the production model behind WeatherNext 2 and it is *exactly* the design M2 sketched — noise vector + CRPS, no diffusion — done at frontier scale. Three things to take from it:
- The noise perturbs the **model** (a shared low-dimensional vector modulating network functions), not just the input seed. This is the direct mitigation for the ensemble-collapse risk: a per-cell input perturbation is easy for the network to ignore; a global functional perturbation is not.
- The noise vector is **tiny** — 32 dimensions for an entire global field. The low dimensionality is what forces physically coherent joint structure instead of per-cell noise.
- Trained **only on per-location marginal CRPS**, yet learns joint spatial and cross-variable structure. Vindicates M2's plan to use pointwise area-weighted CRPS and not attempt a multivariate score.

**FourCastNet 3** — Bonev, Kurth et al., NVIDIA, 2025. `arXiv:2507.12144`
Read for one finding that contradicts an assumption in the current M2 plan: **pointwise CRPS training alone does not produce correct spectra.** Prior CRPS models scored well but had spatially uncorrelated, spectrally wrong ensemble members. FCN3's fix is a composite loss — pointwise CRPS *plus* a spectral term weighting coefficients by multiplicity so the distribution matches across wavelengths. M2 currently treats spectrum as a diagnostic; this paper argues it has to be in the loss. Also the best reference for spread–skill ≈ 1, flat rank histograms, and stable spectra at 30–60 day rollouts.

**AIFS-CRPS** — Lang, Alexe, Clare, Leutbecher et al., ECMWF, 2024. `arXiv:2412.15832` (published npj AI, 2026)
The practical recipe. Operational since mid-2025, so the engineering is battle-tested rather than aspirational. Specifically relevant:
- Uses an **almost-fair kernel CRPS**. With M ≈ 4–8 members the naive CRPS estimator's finite-ensemble bias is large and pushes toward under-dispersion. Do not use the plain estimator at small M.
- Gaussian noise injected into the processor, per member, at train *and* inference time.
- Rollout is used in training with gradients through the full chain — relevant to the M1 curriculum divergences.
- One forward pass per member per step. This is the argument for CRPS over diffusion on a budget.

**CRPS-LAM** — Larsson, Oskarsson, Landelius, Lindsten, 2025. `arXiv:2510.09484`
The academic-budget version of the above, and the closest analogue to this project's scale. Regional domain, single-forward-pass sampling, ~39× faster than the diffusion baseline it replaces. Also a useful precedent for a hybrid architecture: GNN where the geometry is irregular, cheaper local operators elsewhere. Companion: Diffusion-LAM and Graph-EFM from the same group, for what the alternatives cost.

---

## Tier 1 — Geometry, small compute, and long rollouts

**Advancing Parsimonious DLWP Using the HEALPix Mesh (DLWP-HPX)** — Karlbauer et al., JAMES 2024. `arXiv:2311.06253` · code: `github.com/CognitiveModeling/dlwp-hpx`
The most useful precedent for a small model that behaves. Seven prognostic variables, ~110 km mesh, 9.8M parameters, trained on four A100s — and one-week skill only about a day behind GraphCast/Pangu. It holds spectral power and stays realistic over 365-day rollouts without collapsing to climatology. Two structural echoes of this project: HEALPix for the same reason the icosahedral mesh was chosen, and a GRU carrying state at each level of the hierarchy — functionally the same bet as the hidden channels.

**Comparing and Contrasting DLWP Backbones on Navier-Stokes and Atmospheric Dynamics** — Karlbauer et al., 2024. `arXiv:2407.14129`
Controlled comparison of backbones on lat-lon vs. HEALPix, with 365-day rollout stability and zonal-wind physics checks as the evaluation axes. Read it for the evaluation design more than the rankings — it's a template for the "is the architecture wrong or is the training wrong" question M2 has to answer.

**Neural-LAM** — Oskarsson, Landelius, Lindsten, 2023. `arXiv:2309.17370` · code: `github.com/mllam/neural-lam`
Readable, modular, actively maintained graph-based NWP code with hierarchical mesh support and a re-implementation of GraphCast. The `prob_model_global` branch has Graph-EFM, a latent-variable ensemble model. If any external codebase is worth borrowing structure from for the M2 pipeline, this is it.

**Learning to Simulate Complex Physics with Graph Networks** — Sanchez-Gonzalez et al., ICML 2020. `arXiv:2002.09405`
**Message Passing Neural PDE Solvers** — Brandstetter, Worrall, Welling, ICLR 2022. `arXiv:2202.03376`
The two canonical treatments of autoregressive rollout drift in learned simulators: training-time noise injection, and the pushforward trick (one unrolled step, gradients blocked through it). Both are cheap to add to single-step training and both address the M1 drift risk more directly than lengthening the curriculum. The second is effectively the theory M1's `perturbation_growth` diagnostic is measuring.

**NeuralGCM** — Kochkov et al., Nature 2024
A differentiable dynamical core with learned physics, stable over long integrations. Structurally the closest thing in the literature to "NCA sub-steps as PDE sub-steps": an explicit numerical integrator wrapped in a learned correction. Worth reading against the current design to see which parts of the sub-step loop are doing solver work and which are doing parameterization work.

---

## Tier 2 — Frontier reference points (skim, don't study)

- **GenCast** — Price et al., *Nature* 637, 84–90 (2025). Already in the project docs. Re-read only §methods on the noise schedule and the 2nd-order Markov conditioning.
- **GraphCast** — Lam et al., *Science* 382 (2023). `doi:10.1126/science.adi2336`. Read the discussion of MSE-optimal blurring; it's the cleanest statement of why M1's spatial std drifted to 0.87.
- **AIFS** — Lang et al., 2024. `arXiv:2406.01465`. Deterministic baseline for AIFS-CRPS; encoder–processor–decoder on a reduced Gaussian grid.
- **WeatherNext 2** — DeepMind blog, Nov 2025. The productionized FGN. Useful for the marginals-vs-joints framing in plain language before reading the paper.
- **ECMWF AIFS blog** — `ecmwf.int/en/about/media-centre/aifs-blog`. Short, frequent, unusually candid posts on what breaks in operational ML forecasting: localized wind extremes, storm structure, training-strategy effects on TC forecasts.

---

## Tier 3 — Evaluation and scoring

**WeatherBench 2** — Rasp et al., JAMES 2024. `doi:10.1029/2023MS004019` · site: `sites.research.google/gr/weatherbench` · code: WeatherBench-X
The scoring harness M2 commits to. Note the probabilistic scorecards are CRPS relative to IFS ENS, and that operational models are scored against IFS analysis while ML models are scored against ERA5 — a comparability gotcha worth writing into the eval script's comments.

**Strictly Proper Scoring Rules, Prediction, and Estimation** — Gneiting & Raftery, JASA 2007
The foundation. Read enough to be able to state precisely why CRPS is proper and MSE-on-the-ensemble-mean is not.

**Ensemble size: How suboptimal is less than infinity?** — Leutbecher, QJRMS 2019
The fair-CRPS correction and why finite-M bias matters. Directly load-bearing at M ≈ 4–8. Pair with Ferro (2007) on fair scores.

**Probabilistic measures afford fair comparisons of AIWP and NWP model output** — 2025. `arXiv:2506.03744`
A careful treatment of what's actually being compared when ML and physics models are scored side by side. Relevant to M2's "distance to frontier" framing.

**Tools:** `scoringrules` (Python, CRPS estimators including fair variants), `xskillscore`, WeatherBench-X. Don't hand-roll the CRPS estimator — the fair/almost-fair variants are easy to get subtly wrong.

---

## Tier 4 — The NCA side

An honest framing first: **NCA applied to numerical weather prediction is essentially unoccupied territory.** Searching the literature turns up CA-ANN hybrids for land-surface temperature, self-learning CA for radar nowcasting, and not much else — nothing resembling a learned local update rule on a global atmospheric state. That is the project's genuine novelty, and also why the baselines and the failure-mode literature have to come from graph/message-passing PDE solvers (Tier 1) rather than from the NCA field itself.

**Growing Neural Cellular Automata** — Mordvintsev, Randazzo, Niklasson, Levin, *Distill* 2020. `distill.pub/2020/growing-ca`
The origin. Re-read specifically for the pool/sample-pool training pattern and the stochastic update mask — M1 correctly rejected the mask for a conservation-law system, and it's worth being able to articulate why to anyone who asks.

**Self-Organising Textures** — Niklasson, Mordvintsev et al., *Distill* 2021. `distill.pub/selforg/2021/textures`
The loss-function change from "reproduce this exact image" to "reproduce this distribution" is conceptually the same move M2 makes from MSE to CRPS. Short and worth the analogy.

**Neural Cellular Automata: applications to biology and beyond classical AI** — Hartl, Levin, Pio-Lopez, *Physics of Life Reviews* 56:94–108, 2026. `doi:10.1016/j.plrev.2025.11.010`
Current survey. Fastest way to see what's been tried and what hasn't.

**Learning Graph Cellular Automata** — Grattarola, Livi, Alippi, NeurIPS 2021
NCA generalized off the square lattice onto arbitrary graphs. The formal grounding for mesh-NCA perception.

**Physics-Informed Neural Cellular Automata for Electromagnetics (NCA-EM)** — IEEE, 2025
Rare direct precedent: discretized Maxwell's equations embedded as learnable local update rules, stable to 2,000 timesteps. Also has a physics-informed kernel initialization trick (initialize the perception kernels at the known finite-difference stencils) worth stealing — it's a ~45% error reduction there, and it directly complements the cotangent-Laplacian operator work from M1.

**Differentiable Logic Cellular Automata** — Google Research, 2025. `arXiv:2506.04912` · `google-research.github.io/self-organising-systems/difflogic-ca`
Peripheral to M2, but the clearest recent demonstration that NCA update rules can be made radically cheaper. Relevant if inference cost ever becomes the bottleneck.

**Neural Particle Automata** — already in the project. Re-read the seeding and particle-interaction sections *after* the FGN paper; the noise-into-seed question is common to both.

---

## Watch

- **The AI Weather Forecasting Revolution** — Christian Lessig (ECMWF), 2026. `youtube.com/watch?v=JUsFvifyZeM`
  Best single-hour overview of where data-driven NWP actually stands, from someone inside an operational centre.
- **ECMWF Training / AIFS lecture series** — `ecmwf.int/en/learning/training/search` and the ECMWF YouTube channel. Simon Lang on AIFS ensembles and Gabriel Moldovan on AIFS architecture are the two to find. Ensemble-specific and short.
- **ECMWF ML for weather prediction training course, Oct 2025** — materials and notebooks at `github.com/ecmwf-training/2025-ml-training`. The `6-Anemoi/training_aifs-ens.ipynb` notebook walks through converting a deterministic training pipeline to CRPS ensemble training, including the AlmostFairKernelCRPS loss. **This is the single most directly reusable artifact on this list for M2 step 3.**
- **Anemoi docs** — `anemoi.readthedocs.io`, specifically the kCRPS setup guide. Configuration-level detail on ensemble-size-per-device and the loss.
- **Alexander Mordvintsev: neural cellular automata from scratch** — `youtube.com/watch?v=kA7_LGjen7o`. Implementation-level walkthrough from the originator; useful if any part of the NCA formulation still feels like received wisdom.
- **Quanta, "Self-Assembly Gets Automated in Reverse of 'Game of Life'"** (Sep 2025) — non-technical NCA overview. Good for explaining the project to someone else.

---

## What this implies for the M2 plan

Four proposed amendments, in order of how much they'd change the work:

1. **Add a spectral term to the CRPS loss.** FCN3's result says pointwise CRPS alone yields spectrally wrong members. The M2 plan lists spectrum as a diagnostic gate; it likely has to be in the objective to pass that gate.
2. **Perturb the model, not the seed.** FGN's functional perturbation with a low-dimensional (~32) global noise vector is a stronger design than injecting noise into the seed state, and it directly targets the ensemble-collapse risk already flagged.
3. **Use a fair or almost-fair CRPS estimator.** At M ≈ 4–8 the naive estimator's bias is not a rounding error. `scoringrules` or the Anemoi implementation.
4. **Add pushforward or training-noise before lengthening the rollout curriculum.** Cheaper than curriculum extension and better targeted at drift, and the M1 curriculum already diverged twice.
