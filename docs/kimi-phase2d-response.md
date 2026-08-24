 ● I read the report, both configs, and the two model implementations (src/wnca/models/control_gnn.py, src/wnca/models/nca.py, perception.py, icosphere.py). Short version: it's a real experiment and the
   report is unusually honest about its own weaknesses — but no, the GNN is not an accurate control for the locality hypothesis, and there's one problem with it that the report doesn't flag, which I'd
   consider the biggest issue in the document. Here's the critique, roughly in order of severity.

   1. The "non-local" arm may have a smaller per-window receptive field than the "local" arm

   This is the thing the report missed. Trace the actual construction:

   • multiscale_edges(5, n_levels=3) (control_gnn.py:79) yields levels [5, 4, 3], cycled coarse-to-fine as [3, 4, 5, 3, 4, 5] over the 6 hops. The coarsest level ever used is 3, whose edges span 2^(5−3) = 4
     fine cells ≈ 900 km. The docstring's claim of "one hop at level n_sub − 3 spans roughly eight fine cells" (control_gnn.py:16) is wrong — level 2 is never reached with n_levels=3.
   • Best-case information travel per 6 h window for the GNN: 4+2+1+4+2+1 ≈ 14 fine-cell widths (~3,100 km), and mean-aggregation diffuses that further downward in practice.
   • The NCA: 20 sub-steps × 1 hop = 20 fine cells ≈ 4,500 km per window.

   So the control is non-local only in the sense of topological shortcuts (jumping over intermediate nodes), not in reach. Per window it arguably moves information less far than the strictly local model. The
   strongest statement a GNN win could support is "shortcut edges of ~900–1,800 km help," not "non-locality helps." Planetary-scale teleconnections — the actual reason to doubt strict locality — are untested:
   no edge spans more than ~1/20 of the globe. If you want the control to genuinely threaten the thesis, it needs edges at level 0–1 (or a couple of global attention layers), and n_levels=3 looks chosen by
   default, not by argument.

   2. The two-variable confound in §5 is really four variables

   §5 is the best section in the report, but it undercounts. The arms also differ in:

   • Weight sharing: the NCA applies the same MLP 20 times; the GNN has 6 distinct layer stacks. Recurrence-vs-depth is its own axis, independent of locality.
   • Temporal discretization: the NCA integrates the window (dt 0.05 × 20); the GNN is a one-shot 6 h map. Given your own dt-invariance diagnostic (§5, "partial"), any gap could come from
     integrator-vs-emulator structure rather than locality.
   • Input featurization: the NCA is handed ∇x, ∇y, ∇² from fixed geometry (perception.py); the GNN must infer differential structure from raw neighbor values. That's a physics prior the control doesn't get.
     You can call it part of "being a PDE rule," but then the experiment is testing "PDE-flavored local rule" vs "generic GNN," which is not what the headline will say.

   Your proposed follow-up (same MLP, 20 sub-steps, widened stencil) is indeed the right single-variable experiment — I'd go further than §5 and say the current design can produce no clean attribution at all,
   only a sign.

   3. The interpretability is asymmetric, and the report has it backwards in §1

   §1 says "if the local model holds up, the locality claim has support." That's the weak direction. Tally the uncontrolled factors:

   • Favoring the NCA: lr and wd tuned on the NCA (§4 admits this), spectral norm ON for NCA / OFF for GNN (a real architectural difference, not just an implementation detail — the NCA's layers are pinned to
     σ_max ≈ 1), ~3× the compute per step, and the free differential filter bank.
   • Favoring the GNN: shortcut edges, 6 hops chosen as the "generous" option.

   Almost every uncontrolled factor tilts toward the local arm. So an NCA win is nearly uninformative about locality — it could be lr, compute, spectral norm, or featurization. A GNN win, despite all that,
   would be damning. The experiment can falsify the thesis far more cleanly than it can support it, and the write-up should be framed that way. Relatedly, "parameters are the matched quantity" is a choice,
   not a neutral fact: for a forecasting thesis, skill-per-FLOP is at least as decision-relevant, and right now the NCA is spending 3× the FLOPs for its result.

   4. §4 undersells the hyperparameter problem

   lr 3e-4 wasn't just "tuned on the NCA" — per the 2c config header, it sits at the bottom of the NCA's stability cliff (5e-4 and 1e-3 diverge). The GNN, with no 20-step recurrence, has no such cliff and
   plausibly wants a much higher lr. The epoch-4 numbers look exactly like an under-trained control: train loss only ~37% worse while the rollout metric is 3× worse. The planned probe (wd × lr, 2 epochs each)
   is the right instinct, but 2-epoch probes are a screen for "obviously mis-tuned," not evidence of fairness — learning-rate rankings routinely invert between epoch 2 and epoch 8. I'd state in the findings
   that the control is screened, not tuned.

   5. The seed logic in §6 is a non-sequitur

   "The local arm's seed spread is 0.21%, therefore n=2 is sufficient" assumes the GNN's seed variance is similar. But the control's own early diagnostics (val 4.5× train) hint at higher variance, and
   rollout-unstable models typically have fatter seed distributions. The 0.21% is the local arm's noise floor; say so. With a projected gap in the hundreds of percent this won't change the conclusion, but the
   reasoning as written doesn't support the third-seed decision.

   6. One unexplained anomaly deserves a sentence before it's a result

   The sign flip you note — NCA val below train, GNN val 4.5× above train — has at least two readings: "falls apart on rollout" (your reading) or plain overfitting/distribution shift on the clean single-step
   val, potentially worsened by applying the NCA's anti-ratchet wd=0.1 to a model with no ratchet. These imply different fixes and different interpretations of the final gap. Worth one diagnostic (e.g.,
   val-with-pushforward vs val-clean) before the narrative hardens.

   Smaller items

   • control_gnn.py:8-12 docstring still says "Matched to the NCA on … parameter count and wall-clock" — contradicted by both the config header and report §4. For a project whose convention is
     docs-match-measurements, fix it alongside the level-3 docstring error above.
   • Parameter matching across a hidden-512 wide/recurrent net and a hidden-160 6-layer GNN is a weak notion of "same capacity." Unavoidable, but it deserves one sentence.
   • Checklist item on the diurnal trap for 2 m temperature and the broken exit criterion 5 — both good catches, keep them.

   Verdict

   Is it a valid experiment? Yes — as a same-data, same-optimizer, parameter-matched baseline comparison, and the pre-registration-while-running discipline, the measured parameter gate with a regression test,
   and the caught config-inheritance bug are all genuinely good practice. Is the GNN an accurate control variable for the locality hypothesis? No. It's a defensible baseline but a confounded control:
   reach-limited (point 1), multi-variable (point 2), and biased toward the local arm in nearly every uncontrolled dimension (point 3). The report already concedes half of this in §5; I'd strengthen §5 to
   include the receptive-field arithmetic, reframe §1 so the asymmetry of evidence is explicit, and write the finding as "a strictly local rule, iterated 20× with 3× the compute and tuned hyperparameters,
   beats/holds/loses to a ~900-km-shortcut GNN applied once" — anything shorter will overclaim in whichever direction it lands.