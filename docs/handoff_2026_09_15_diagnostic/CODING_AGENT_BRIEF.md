# Stage 2 diagnostic handoff: explain the fixed-GP result

15 September 2026. The user authorizes this diagnostic follow-up and requires no local numerical runs. Implement locally if convenient; execute all model checks, checkpoint evaluation, Jacobians and other numerical diagnostics on the existing cluster GPU. This is analysis of saved models, not a new training study.

## Objective and scope

Explain what the failed Stage 2 screen says about the fixed generalized response, separating:

1. The response change imposed by the equation at unchanged weights.
2. The different weights and responses learned by each arm.
3. Current-input transmission, retained history, and gradient propagation.
4. The short learning-rate schedule and the limits of the available checkpoints.

The Stage 2 result remains a failed screen. No training, optimizer updates, resume, new seeds, hyperparameter/physical-coefficient sweep, test-set scoring, spatial-only training, or meta-learning is included. Derivative evaluation without a parameter update is authorized. Do not change the architecture or the physical/readout convention to improve a score.

Proceed with implementation and the bounded cluster diagnostic without another planning-only confirmation. If you cannot execute on the cluster, deliver one short launcher command for the user; distinguish prepared from executed work.

## Inspected state and provenance

Coordinator statically inspected:

- Checkout: /Users/alessandrodurso/Documents/final-prospective-s5
- Branch: cluster-gp-rawat
- Commit: 227648e6d64a9b786b59e4bbfb64bfa8cda8d6ff
- Working tree was clean at inspection.
- Report: docs/GP_IMPLEMENTATION_REPORT.md, section 12.12.
- Reported saved runs: /Users/durso/s5-runs/stage2/, manifest.jsonl.
- Reported execution job 65870 is historical provenance, not authorization to assume that allocation is still active.

Preserve existing work. Create a diagnostic feature branch from the inspected revision, or record the newer base and explain any relevant differences. Push code and the compact report to the existing public repository after checks. Do not publish data, full checkpoints, raw speech features, or credentials.

Use /Local/durso/prospective_ssm_project/.venv/bin/python on the verified cluster environment. No environment upgrades and no CPU fallback on the laptop. Obtain/use an available allocation by the established workflow, without interfering with other jobs.

## A. Preserve the evidence and make loading genuinely read-only

Inventory all nine Stage 2 run directories. Save a diagnostic manifest listing original run identity, configuration, original code/environment provenance, metrics file, best/last checkpoint filenames, epochs/steps, and SHA-256 hashes of these files.

Use a NEW output directory outside Stage 2, such as /Users/durso/s5-runs/stage2-diagnostics/<unique-run-id>. Never write within the source run directories. Compare source hashes before and after the diagnostic.

Two static findings require attention:

1. experiments/gp/rawat_benchmark.py calls write_config before its evaluate-mode branch and append_metrics inside evaluate. Do NOT invoke that CLI on original run directories: it would overwrite original configuration/provenance and append evaluation records.
2. s5/gp_diagnostics.py:core_from_module dispatches using the old mechanism attribute. SubstrateSSM instead uses response and coefficients(); silently falling back to inherited native coefficients can report the wrong dynamics. The old run_diagnostics.py also initializes older models rather than these saved Stage 2 models. Add a dedicated adapter for the actual executed substrate; do not use those paths unchanged.

Restore the saved parameters AND the matching batch statistics using a structurally correct template. Preserve dtype/configuration and record all casts. Existing checkpoint metadata/optimizer state may be loaded for provenance, but do not apply an optimizer update.

Load only the train/validation assets needed. The current SC.load opens all three splits; use a split-selective read-only path for this diagnostic. Historical statements should distinguish “test not scored/used for selection” from “test files never opened”; do not imply that merely loading a file trained on it.

## B. Reconstruct the screen before interpreting it

Read existing metrics/configuration for all nine runs. Do not choose different checkpoints. Publish:

- Exact selected validation accuracy/unsmoothed CE, best epoch, last epoch, and full-precision paired deltas.
- Train loss, train accuracy, validation accuracy and validation CE versus epoch/update.
- Learning-rate schedule reconstructed from the saved configuration and update count.
- Reported epoch durations, with first-epoch/compile effects distinguished where the logs permit.
- Whether best is the last checkpoint; whether any accuracy ties occurred; the rule actually used by the runner. Report any discrepancy rather than retroactively selecting a different checkpoint.

The runner's training CE is label-smoothed and validation CE is unsmoothed. Do not subtract them and label the difference a directly comparable generalization gap. Its logged grad_norm is the last minibatch value, not an epoch average or a record of clipping frequency.

The script passed epochs=10, and make_optimizer uses steps_per_epoch * epochs as the cosine decay duration. Describe Stage 2 as a TEN-EPOCH COSINE SCHEDULE, not a ten-epoch prefix of the 300-epoch paper schedule. Plot the two schedule definitions analytically for illustration if useful; do not run the alternative schedule or claim this caused the ranking.

Re-evaluate each of the five originally selected 1e-3 best checkpoints once on the SAME validation split, only to corroborate restoration. Do not evaluate test or reselect a model. Use original batching and inference batch statistics. Check exact correct counts; declare a CE tolerance appropriate to recorded dtype before execution. If counts differ, report the discrepancy and resolve loading/dtype/data issues before explaining performance.

The report's actual epoch times are approximately native 2.75 s, gain_clip 2.80 s, alpha-P 2.70 s, mass 6.85 s. Use measured end-to-end comparisons (~2.5x) separately from the earlier 1.93x step microbenchmark. State whether checkpoint-write time is included; the current epoch_s is captured before checkpoint writes.

## C. Extract the actual temporal response

Primary models: the five selected 1e-3 best checkpoints, all four layers. Also reconstruct their seed-100 initialization from the recorded code/configuration, without training. Verify common ordinary-parameter initialization rather than assuming it; label reconstructed initializations, not saved checkpoints. Use last checkpoints for lightweight pole/parameter summaries only when distinct from best.

Extract from bound executed modules:

- Raw and clipped Lambda, Delta, a=Delta*Lambda, and gain-scaled b=Delta*B_c.
- Actual discrete transition and input matrices, C_tilde and native D.
- Fixed T/gamma/rho/M and evidence that none occurs in trainable parameters.
- Native versus effective poles, decay times in input frames, oscillation frequencies, clipping activity, and log-step changes.

Read physical s only in the positive-mass model. Do not reinterpret the auxiliary filtered velocity as an extra readout or replace s by a prospective firing observable.

For each layer, compute REAL input-to-output impulse matrices K_l at lags 0..511, where lag zero means output after consuming the current held token. Separate the native D contribution from the dynamical current tap. Report energy in the fixed lag bands:

  0; 1–4; 5–16; 17–64; 65–160; 161–511.

Report both absolute gain and fractions. Fractions alone can hide a collapse in total response. Values beyond 511 are not assumed negligible: report a tail estimate or explicitly mark the window as truncated. No monotonic-Hankel or “longer poles imply better useful memory” claim.

Required impulse identities, with R(Z)=2 Re(C_tilde Z) for a real-input complex-pair realization (adjust to executed conjugate convention):

- One tap: K_0=R(B_bar)+diag(D); K_l=R(A_bar^l B_bar), l>=1.
- Alpha-P two tap: K_0=R(B_plus)+diag(D);
  K_l=R(A_bar^(l-1)(A_bar B_plus+B_minus)), l>=1.
- GP M=0: K_0=R(b_bar+d_x)+diag(D);
  K_l=R(a_bar^l b_bar), l>=1.
- GP mass: K_l is the physical-s readout of A_block^l B_block, plus native D only at l=0.

Check these against actual forward impulse calls for every layer/arm. “Current tap” includes the within-interval dynamic update; it is not synonymous with continuous-time algebraic feedthrough.

Compute discrete frequency responses on a fixed 129-point grid from 0 to Nyquist using the exact real state-space realization or properly realified two-tap transfer. Do NOT take 2 Re of a complex-frequency response as if that were complex-pair realification: the resulting real system still has a complex Fourier response. An impulse-DFT cross-check must account for truncation tails. Skip expensive full block-Hankel SVDs.

Add within-checkpoint equation comparisons: for each alpha-P/GP best checkpoint, keep Lambda, Delta, B_tilde, C_tilde, D, gain and clipping fixed and evaluate the one-tap law's linear response alongside its executed response. These are untrained counterfactual response calculations, not new accuracy scores or trained baselines. They separate the direct law change from coadaptation of learned weights.

## D. Small signal and gradient diagnostic, with no updates

Choose one common 32-example TRAINING subset by a fixed permutation seed 20260915, independent of losses/labels. Store indices and the source split hash. Reuse it for every arm. This is a post-screen diagnostic; do not relabel it an independent confirmation sample.

At initialization and at the five selected best checkpoints:

1. Compute one full-BPTT gradient of the original smoothed mean-pooled classification loss. No optimizer call. Report per-layer ordinary-parameter gradient norms, relative to parameter norms where meaningful, plus activation RMS and loss. Preserve complex parameter conventions.
2. Capture gradients with respect to recurrent-core inputs and pre-pooling block activations. Report where current transmission comes from: recurrent current tap, native D, and the residual path, using the executed architecture. Magnitude alone is not a gradient-quality measure, and gradient size also depends on current error and weight scale.
3. Probe the FINAL ENCODER OUTPUT BEFORE TEMPORAL POOLING at anchor frames 80 and 160. Use four fixed unit-normalized Rademacher VJP directions (seed 20260916), shared across arms, to estimate input sensitivity by lag. Aggregate the same lag bands as above within the available history, and separately record any future-input sensitivity.

The pre-pooling probe is essential: pooled classification logits depend directly on every earlier output, so sensitivity of pooled logits to early inputs does not by itself demonstrate recurrent memory. These probes measure local sensitivity, not delayed-recall accuracy or semantic usefulness.

Primary response/sensitivity measurements use inference mode: saved batch statistics fixed, dropout disabled. For the best checkpoints only, one additional gradient pass may use the original training-mode normalization and fixed dropout RNG, discarding all mutable updates. Label it separately: training BatchNorm can couple samples/times, so it is not a purely causal recurrence-gradient diagnostic.

Do not implement or call this “spatial-only training.” All gradient paths here remain full BPTT. A fixed-state local core Jacobian can be calculated analytically from the current tap, but that is a transmission diagnostic, not a trained alternative.

## E. Mechanistic predictions to examine, not assume

For normalized scalar fixed coefficients and r=j s-bx:

  G_mass(p)=b(1+Tp)/[rho*T*p^2+(1+Tj)p+j].

The same law can be written, for compatible zero prehistory,

  [j+p*(rho+(1-rho)/(1+Tp))] S(p)=b X(p).

At rho=.75, its high-frequency core gain relative to the same ordinary mode tends to 1/rho=4/3; it still decays as 1/p. Input-only prospectivity's continuous core has a nonzero high-frequency limit. Native D and residual skips are separate. These are continuous-time facts, not claims about actual discrete Nyquist gain or the cause of the accuracy gap.

For M=0, the pole mapping is a_eff=a/(1-Ta). For Re(a)<0 and scalar T>0, its image lies in the disk centered at -1/(2T) with radius 1/(2T). At T=5 this bounds the continuous effective imaginary magnitude by .1 per frame and the real part between -.2 and 0. Examine whether relevant mode frequencies/residues are being compressed; do not infer relevance from the bound alone.

Determine what the saved-model evidence supports:

- excessive change to learned temporal filtering;
- insufficient current-path benefit over the common D/residual substrate;
- learning curves consistent with optimization differences (not proof of a schedule cause);
- normalization/readout effects or a concrete diagnostic/implementation defect;
- or insufficient evidence to attribute the gap.

Dense learned-T local gains are a distinct model family. Do not silently switch to that family, arbitrary off-diagonal coefficients, learned physical constants, or another readout as a “fix.”

## F. Checks, resource bound and completion

Before data-dependent diagnostics, freeze defaults, probe seeds, bands and numerical checks in docs/GP_STAGE2_DIAGNOSTIC_PROTOCOL.md. Keep tolerances scoped to actual dtype and independent-reference checks; use the already declared production tolerances when applicable. Verify:

- read-only restore reproduces the saved evaluation;
- coefficient/impulse adapter equals executed forward for all five arms;
- diagnostic hooks leave predictions unchanged;
- one fixed input-direction JVP/VJP agrees with an independent finite difference at declared step sizes;
- inference pre-pooling probes have no future dependency beyond numerical tolerance;
- parameters, saved batch statistics and all original source files remain unchanged.

Execute only new focused checks and this diagnostic, not the entire historical test/train suite. The cap is 20 minutes of allocated GPU wall time INCLUDING compilation and checks, one GPU worker, with timeouts/checkpoints between diagnostic phases. Start with A/B and core extraction. If time is tight, omit reconstructed-initialization gradient probes, then optional training-mode gradients, before dropping selected-model response measurements. Record incomplete sections; do not extend the budget automatically or restart completed phases.

Host-side NumPy/SciPy work on the allocated cluster node is fine. No laptop numerical execution, no local plotting from checkpoints, no test scoring, and no hidden optimizer steps. Static editing and report preparation locally are fine.

Deliver:

- docs/GP_STAGE2_DIAGNOSTIC_PROTOCOL.md
- docs/GP_STAGE2_DIAGNOSTIC_REPORT.md with actual checkout/branch/commit, execution identity, original-data/checkpoint identity, findings, uncertainty and failures.
- Compact machine-readable manifest, curve/response/gradient summaries, and a few plots: training/LR curves; pole/impulse response; current-versus-history sensitivity. Put large artifacts on persistent cluster storage.
- A source-hash invariance result before/after, actual wall time and one restartable cluster launcher/status command.
- Append the diagnostic report link to GP_IMPLEMENTATION_REPORT.md. Preserve the failed Stage 2 verdict.

Correct the accompanying prose: ten-epoch cosine schedule; approximately 2.5x observed epoch cost; conditional gain/tap comparisons rather than a universal 50/50 decomposition. A future declared experiment is not inherently invalid after a negative screen, but none is authorized by this diagnostic brief.

End with a short conclusion identifying a supported next theoretical/implementation question, or stating that the available evidence cannot distinguish the hypotheses. A diagnosis is the deliverable; an improved score is not required.
