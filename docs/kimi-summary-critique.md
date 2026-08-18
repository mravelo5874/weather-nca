 ● Summary

   weather-nca is a research project testing whether a strictly local learned update rule — a neural
   cellular automaton on a 10k-node icosahedral mesh (~223 km) — can forecast global weather, against
   the field's consensus that you need non-local information flow (attention, multi-scale GNNs,
   spectral transforms). The thesis: "weather is a PDE, and PDEs are local." Architecture is a
   per-cell MLP update over a fixed geometric perception stencil (identity, ∇x, ∇y, ∇²), integrated
   as 20 Euler sub-steps per 6h window, 28 ERA5 channels + 32 hidden channels, ~1.03M params.

   Progress so far: M1 (single-var z500) hit 24h RMSE 325–345 vs persistence ~594. Phase 2a showed
   data volume matters a lot (−30%). 2b added multivariate coupling but diverged at long lead (3.2×
   climatology at 15 days). 2b-pushforward (Brandstetter-style training from the model's own
   rolled-out state) is the current best: 24h 282.1, 360h −46%. The crucial experiments are unrun:
   phase 2d (same-budget non-local GNN control — the actual thesis test) and 3a/3b (FiLM-noise
   ensembles, fair CRPS, spectral loss). The docs are unusually candid — they log voided runs, wrong
   diagnostics, and confounded experiments — and the engineering hygiene (fixed selection metrics,
   incidents-encoded-as-tests, resumable caches) is genuinely good.

   Real criticisms

   1. The headline thesis is untested, and current evidence cuts against it. Everything hinges on
      phase 2d, which hasn't run. Meanwhile the best model is ~5× off the WB2 frontier, and 2b
      confounded coupling with a 37× capacity jump. "Locality is fine because 6h skill was 93%" is
      weak — that mostly shows the advection operator works. The project currently has a good
      engineering story, not yet a science result.

   2. n=1 everywhere. One run per config, one checkpoint, one (inconsistent) test year — 2a tests on
      2020, 2b/2b′/2b-pf on 2018. The docs themselves show RNG ordering and LR granularity move
      results by ~5% (phase 0), yet headline deltas like 297.6 → 282.1 (−5.2%) are reported without
      variance estimates. Several "wins" are plausibly within run-to-run noise. The 2b′ episode
      proves the authors know this and the pushforward claim still rests on a single
      undertrained-adjacent run.

   3. The gates are known-broken and still nominally in force. Exit criterion 5 (perturbation growth
      ≤1.05) is documented as selecting the worst model in the ladder and rewarding over-damping —
      the amendment is "proposed, not adopted." Selection metric (48h wMSE) demonstrably failed to
      predict the pushforward gains at 120h+. When your own plan's gates would fail your best model,
      the plan needs editing, not just a footnote.

   4. The resolution was mislabeled for an entire milestone. "1°" was actually ~223 km ≈ 2°, and the
      0.25° node arithmetic was off 4–13×. Acknowledged, but the M3 compute budget derived from those
      numbers hasn't been recomputed in the docs — the budget figure (~54 GPU-h, $40) inherits the
      error.

   5. Zero probabilistic results. Half the milestone (ensemble calibration — the thing decision
      0001's entire noise design is built to de-risk) has no measurements. No spread-skill, no rank
      histograms, no CRPS numbers. Also, z is drawn once per trajectory and held constant across all
      sub-steps — that's a fixed random rule per member, which is a strong and unusual choice;
      whether 16 FiLM dims can generate enough spatially-coherent diversity without collapse or
      without spatially-uniform bias is exactly the open question, and it's completely unmeasured.

   6. Confounded "wins" cited as established. "Solar forcing is worth 19% on 2t" comes from an
      ablation on the confounded, diverged 2b′ run where 2t skill regressed (−48% → −62%). The
      mechanism is real; the magnitude is not established on a healthy model, and the docs overstate
      it.

   7. Code-level reproducibility bug: hash(split) in data/cache.py:199 is salted per process, so a
      synthetic cache built across interrupted/resumed runs silently mixes two different series, and
      cache_tag doesn't capture it. Plus an unexplained bf16-backward-inf failure that knocked the
      flagship 2c run off AMP (~+50% wall time), imageio_ffmpeg missing from deps, and eval/wb2.py —
      the module that produces the externally comparable numbers — has no tests.

   Suggestions

   • Run 2d before anything else, and pre-register the comparison. It's the highest-value experiment
     by the docs' own admission. Fix the budget-matching protocol (params? FLOPs? wall-clock?) in
     writing first, and run 2–3 seeds of both arms. ~25% of compute for the only number that makes
     the headline interpretable.
   • Amend exit criterion 5 and the selection metric now. Adopt the bounded-near-climatology +
     doubling-time-1.5–3d amendment, and move selection to a metric that includes a long-lead term
     (the pushforward lesson is that 48h selection is blind to 120h+ behavior). Do it before 2c
     results land, so it doesn't look post-hoc.
   • Cheap variance estimates: rerun phase 2b-pushforward with 2 extra seeds (it's the cheapest
     flagship run) to put error bars on the −5.2% / −46% claims before they're quoted further.
   • Fix the small stuff: replace hash() with a stable digest, add imageio_ffmpeg to deps, delete one
     venv, recompute the M3 budget at correct mesh counts, and either test eval/wb2.py against a
     known-answer grid or stop quoting WB2-adjacent numbers.
   • Diagnose the bf16 backward-inf properly (gradient histograms per layer around the overflow) —
     it's costing ~7 GPU-hours per 2c-scale run and the mechanism being unknown is a risk to all
     future cloud runs.
   • Retire persistence carefully: the docs retire it after M1, but it's the only baseline with
     verified bit-identical reproduction across phases. Keep it in every scorecard as a sanity anchor
     — it has already caught one voided run.

   Research avenues / insights

   • The growth–RMSE decoupling is the most interesting finding in the repo and it's currently
     treated as a methodology footnote. Pushforward made forecasts much better while perturbation
     growth got worse (×1.139, 1.33d doubling — close to the real atmosphere's 1.5–2.5d, while the
     "stable" models were over-damped at 5d). That suggests perturbation growth is measuring
     proximity to the atmosphere's error-growth dynamics, not forecast skill, and that stability
     gates actively select against physical dynamics. A short write-up of this (ladder of models ×
     growth rate × RMSE × doubling time) is a genuinely novel observation worth a workshop paper on
     its own.
   • dt × n_substeps = 1.0 pinning means the model learns the per-window map, not a dynamics. The
     docs draw this conclusion for the CFL sweep but don't follow through: if the sub-stepping isn't
     learning a time-continuous operator, the "NCA learns a local PDE rule" framing is weaker than
     claimed — it's a 20-layer recurrent residual net. A real test: train at dt=0.05×20, then
     evaluate at dt=0.025×40 on the same window. If the learned rule is truly a differential
     operator, the finer integration should work out of the box (that's the NCA superpower). If it
     breaks, the model has learned a map, and that's an important honest reframing of the thesis.
   • Ensemble design question worth answering explicitly in 3a: with z constant across sub-steps,
     members differ in their dynamics, not their initial conditions — closer to a
     stochastic-parametrization ensemble (like SPPT) than to perturbed-IC ensembles (IFS ENS). That's
     arguably the right design for a local model whose error is structural, but it should be framed
     and compared as such, and the spread-skill behavior will differ characteristically from IC
     ensembles.
   • The Lagrangian endgame (Neural Particle Automata) is the strongest differentiator and it's
     buried. An Eulerian NCA at 223 km will likely lose the accuracy race regardless of 2d's outcome;
     a learned-particle method with local interactions has no incumbent competitor at all. Consider
     timeboxing the Eulerian ladder and stating the pivot criteria now.
   • Nearest-neighbor precedent: DLWP-HPX (9.8M params, similar mesh family) is cited but never
     actually compared against at matched budget — a DLWP-style convolutional model on the same
     icosphere at 1M params would be a much stronger control than a GNN for the "locality" claim
     specifically, since convolutions are also local but non-learned-topology. The current 2d design
     conflates "local vs non-local" with "NCA-style iteration vs single-pass message passing."

   Bottom line: the project is well-instrumented and honest, but it is one control experiment and one
   seed away from knowing whether its central claim is true. The most valuable next moves are 2d with
   seeds, the growth-vs-skill write-up, and a decisive test of whether the learned rule is actually a
   differential operator.