# Cluster handoff: generalized prospective S5 versus Rawat's S5 baseline

Prepared 15 September 2026. This replaces the next-step priority of the September14 brief; it does not rewrite that protocol or its evidence. The user wants implementation and actual cluster training, no local training, with the aim of improving on Rawat's prospective S5 reference. Improvement is the research objective, not an assumed result.

Read `DERIVATION_CONTRACT.md` first. It incorporates the user's latest instruction to stay faithful to the physical and biological derivation. Its symmetric fixed reference and coefficient normalization govern the new experiment. No learned response or mass is authorized.

## 1. Start from the inspected code; proceed with concrete work

Coordinator inspected the clean local checkout `/Users/alessandrodurso/Documents/final-prospective-s5`, branch `generalized-prospective-s5`, commit `a4c5b12599c247923ee0425c25a09d92e8556ce3`. Inspect status again before changing it. Preserve existing work and historical outputs. Create a separate feature branch from this state, record its name, implement there, and push code/tests/report to the existing public repository. Do not merge to main or overwrite prior results.

What is present: diagonal M=0 GP, a matched modal control, tokenwise cue/recall code, and a separate scalar-mode finite-inertia prototype. The latter is not wired into training. What is absent: the coupled response used in the strongest local preliminary results, a faithful Rawat S5 reference, and a completed new-task GPU comparison.

Your reported178 CPU tests are evidence at a4c5b12; they are not GPU tests. The earlier GPU evidence belongs to its recorded earlier commits. The coordinator ran no training or test suite during this handoff.

Training, numerical suites and comparative runs now belong on the cluster. Local activity is limited to editing, static inspection and preparing artifacts. Do not launch CPU training, rerun old local sweeps, or train in a local fallback if GPU access fails. Do not introduce an RQF architecture or a meta-learner in this round.

Proceed with implementation, cluster checks, and the bounded first batch below without asking again whether to start. If you cannot execute remotely, prepare one short launcher the user can run; distinguish prepared commands from executed jobs. Never invent a job ID, backend, metric or successful push.

## 2. Evidence that changes the priority

The positive empirical lead is a **coupled learned response**, not the claim that adding arbitrary positive mass always helps. In the archived width64 experiment, learned mass scored .136700 versus .188372 for the ordinary state-count control (27.43% lower); the depth2 figures were .250197 versus approximately .31490. Those recipes learned T and gamma, sometimes rho, and ordinary controls used a different fixed clock. They are candidates for transfer, not an isolated physics or S5 improvement theorem.

In the later fixed-coefficient experiment, current-input and eight-step recall errors improved45.7% and72.5% at one matched clock, but joint error worsened3.43%. The newest fixed-coefficient, pole-preserving study obtained .00677726 versus ordinary64 .00520387 on fresh confirmation:30.24% worse, zero of three paired wins. Its numerical audits passed. Do not implement the newest pole-preserving parameterization as the default merely because it is newest.

Read the reference documents supplied with this packet. Older studies and checkpoints remain untouched. No S5/Mamba/Rawat benchmark advantage has been established.

Coefficient policy is now RESOLVED. Asked to choose fixed or learned responses, the user replied **"just do as in nla and/or tss"**. Follow their earlier experimental convention: the added physical coefficients are fixed declared model hyperparameters; train the ordinary S5 parameters, including its existing learned steps, using BPTT. Do not learn T, gamma, rho or mass independently, and do not start a learned-response ablation. Implement an explicit fixed-response path with immutable buffers and tests. Existing learned mechanisms remain preserved but are not the current main experiment. No further permission question is needed for this decision.

This convention follows a declared neuronal model; it does not imply NLA published our finite-dendrite numerical tuple. Component equalities and normalization assumptions must remain explicit. The original learned-response gains motivate research but cannot be claimed for this fixed version.

## 3. The general equation and the fixed S5 implementation

For state s, learned residual r=Js-Bx, scalar gamma>0 and symmetric positive-definite T, the family is

\[
M\ddot s+\gamma\dot s+r+T\dot r=0,
\qquad M=0\quad\hbox{or}\quad M=\rho\gamma T,\quad0<\rho\le1.
\]

Coefficients are constant within each sequence. BPTT differentiates all declared trainable coefficients and ordinary weights through the exact discretization. Do not put an input-only derivative in place of T times the **total residual derivative**. Do not add a second prospective input correction on top of this equation by default.

Start from native S5 once: a_i=Delta_i Lambda_i, b_i=Delta_i Btilde_i. In the gain-scaled comparison below use Btilde replaced by B_c consistently. Realify each complex pair with [[Re a,-Im a],[Im a,Re a]], and realify the corresponding input rows in the same order. Set J=-F0. Native S5 clipping makes sym(J)>0. A free learned dense J with only eigenvalue stability would require a different argument.

For the currently authorized fixed identical-cell model, T=5I: no learned off-diagonal coupling follows from it. Reuse the scalar/diagonal M=0 implementation and wire the existing per-mode finite-inertia prototype with exact fixed constants. This is substantially less code than a new coupled model. The matrix formulas below explain the general family and provide a reference; a new full-matrix implementation is not required in this batch. Do not add arbitrary fixed off-diagonal elements to manufacture a coupled candidate.

For M=0, use ordered matrix products:

\[
W=\gamma I+TJ,\quad D_x=W^{-1}TB,\quad
F=-W^{-1}J,\quad B_h=W^{-1}(B-JD_x).
\]

Compute solves, never elementwise b/(1-ta)^2 for noncommuting matrices. Exact ZOH gives h_k=Abar h_(k-1)+Bbar x_k and physical s_k=h_k+D_x x_k. Compute Abar and integral exp(Ft)dt using a small augmented exponential independent of feature width, then multiply by B_h. Preserve the correct current-input term at the initial token and across chunk boundaries.

For positive mass carry z=(s,v), with

\[
\gamma\rho\dot s=-Js-\gamma(1-\rho)v+Bx,
\qquad T\dot v=\dot s-v.
\]

In block form:

\[
\mathcal A=\begin{pmatrix}
-J/(\gamma\rho)&-(1-\rho)I/\rho\\
-T^{-1}J/(\gamma\rho)&-T^{-1}/\rho
\end{pmatrix},\qquad
\mathcal B=\begin{pmatrix}B/(\gamma\rho)\\T^{-1}B/(\gamma\rho)\end{pmatrix}.
\]

Use a separate block affine associative scan: (A2,b2) after (A1,b1) means (A2*A1,A2*b1+b2). Positive mass doubles internal state. Read **s only**, after consuming the held token; no additional M=0 state feedthrough. Retain native S5 D, residual paths and the exact conjugate real readout factor. With zero prehistory, rho=1 has the ordinary driven response at the same gamma; arbitrary auxiliary initial states need not.

Stability is supported by T SPD, sym(J)>0, scalar gamma>0 and scalar rho in(0,1]. Do not extend the scalar-rho construction to arbitrary learned mass matrices without proof. Positive components are a constitutive restriction, not a global biological derivation of the entire nonlinear network.

For standard S5 transfer work use normalized gamma=1 and native learned Delta on every arm. Preserve gamma explicitly in the coefficient API for physical checks. The new symmetric reference in DERIVATION_CONTRACT.md has T=5I, physical gamma=5 and M=18.75, giving normalized mass M/gamma=3.75 and normalized residual matrices J/gamma,B/gamma. Set physical J=-gamma F0 and B=gamma B0 to recover the comparator's ordinary generator and drive exactly. This is an explicit current-coupling normalization, not a claim to identify native S5 weights as biological conductances. The older a=.1 tuple is historical; no empirical choice of capacitance ratio is needed for this normalized test.

Freeze added response values exactly, using configuration/buffers rather than trainable raw parameters with nominal zero learning rate. Ordinary S5 parameters still learn. The M=0 test with positive gamma is a separate reduced family member; it is not the exact fast-dendrite circuit limit, which sends both gamma and M to zero. Use separate mechanism labels and retain old checkpoints unchanged.

Suggested new mechanism names are `gp_fixed_m0` and `gp_fixed_mass`. Static no-op/plain routing must remain the original path. For the finite-mass prototype, replace the feature-width-sized augmented exponential with the equivalent4x4 per-mode exponential for transition and integral, followed by multiplication by the2xH drive. This avoids an expm of size(2+H) at every mode/update. Verify values and all derivatives against the existing prototype before switching production execution. Do not use an eigenvector derivative near coalescing poles.

## 4. Benchmark reference: Rawat alpha-P-S5, not the existing input switch

Primary source: [Rawat et al., Appendix E.3–E.4 and Table6](https://arxiv.org/html/2609.04134v1). Obtain and pin the linked [reference code](https://github.com/Sequel-Institute/prospective-rqf); the coordinator's web fetch failed, so its implementation was not inspected. If unavailable, label the port paper-based and record unresolved details.

In native coordinates, the published construction is

\[
B_c=\operatorname{diag}(-\Re\Lambda)\widetilde B,\quad
\bar A=e^{\Delta\Lambda},\quad
\bar B=\Lambda^{-1}(\bar A-I)B_c,\quad
\bar B_\pm=\{\bar B+5\Delta\bar A B_c,\ -5\Delta\bar A B_c\}.
\]

It uses clipped poles; native S5 uses unscaled input and no such clipping. Thus both gain and clipping are attribution controls. Implement the two-tap law explicitly with zero previous input; preserve instantaneous native D. The existing `prospective_input` is an ingredient control, not the complete baseline.

Start with four-layer width32 S5, MFCC Speech Commands10, not the existing35-class loader. Table6 reports96.31±.32% alpha-P-S5 and95.84±.32% native. Use the source's architecture, preprocessing, split, optimizer, schedule and checkpoint selection; resolve code-level defaults before freezing. These published numbers are reference measurements, not an acceptance threshold for a short run. No RQF replacement.

Create `docs/RAWAT_BASELINE_MAP.md`: exact reference commit or unavailable status; every architecture/loader/training setting; corresponding local code location; verified equality or declared difference. Save filename-based split manifests, label mapping, preprocessing configuration, normalization statistics and hashes. Never tune on published test agreement or silently change class selection/splits.

Compare the new GP on the **same gain-scaled, clipped S5 substrate** as alpha-P-S5, replacing the recurrent response law. Add gain-scaled clipped S5 without either prospective correction. Native paper-style S5 remains a separate complete-recipe reference. Keep a learned-step ordinary comparator; do not recreate a deliberately slow baseline.

## 5. Fix runner infrastructure before comparative training

Direct coordinator inspection found these remaining issues at a4c5b12:

- `cue_recall_runner.py` imports save_checkpoint but never calls it. It records best development metrics without retaining best parameters; it has no actual selected-checkpoint test/extrapolation evaluation. Implement separate train/select/evaluate modes, an atomic best checkpoint, complete resumable last state, and guarded final evaluation. A summary scalar is not a checkpoint.
- `n_state_coords=2*SSM_SIZE_BASE*N_LAYERS` double counts the ordinary conjugate-symmetric physical real state. The existing two-layer width32 model has64 recurrent real coordinates, before counting any additional buffer. Derive counts from executed carries; report physical state, auxiliary state, previous-input buffers and parameters separately.
- Nonlinear end-to-end state/timing and gradient checks must include the executed runner, not only standalone coefficient units.
- Epoch continuation needs optimizer, scheduler, batch statistics, shuffle/dataloader and model RNG states, progress and selection counters. Existing checkpointing.py explicitly defers loop resume. Finish this for cluster use; exact metadata-only seeds do not reconstruct a partially advanced RNG.
- Do not pipeline a test process into `tail` and report tail's exit code as success. Use pipefail or save the log, check the actual process result, then display its tail.
- Production mixed precision is not established by a double coefficient fixture. The old positive-mass dense code failed pure float32 checks; separate actual coefficient dtype, parameter dtype and state execution dtype. Use a measured working precision policy consistently on comparator arms and report its cost.

Do not execute the old complete cue/recall sweep as well as a new benchmark sweep. Preserve the old protocol as unexecuted/historical if that is still true. Use only a short cluster integration check of the runner, then prioritize the benchmark.

## 6. Cluster correctness and limited experiment order

All execution is GPU-only with an explicit backend guard. Predeclare tolerances before running, using the existing independently audited coefficient gates as references. Required coverage: initial and trained-like admissible fixtures; repeated poles; input jumps; values and all parameter/clock gradients against independent real-valued SciPy evolution; sequential versus block-scan; chunks/reset/nonzero initialization; fixed-buffer invariance; ordinary/rho1 and scalar-diagonal reductions. Test both actual production dtype and a high-precision reference. Fix genuine defects and retain failed-check records; do not loosen thresholds to clear a failure. Noncommuting matrix tests become necessary only if a future authorized full-matrix mechanism is added.

Before any score-driven selection, commit `docs/GP_RAWAT_CLUSTER_PROTOCOL.md` with the following bounded plan and resolved coefficient policy:

1. **One initialization/two-update integration check per executable arm**, then a cluster throughput/memory measurement. No local execution. This stage establishes execution, not accuracy superiority.
2. **A small validation-only MFCC development batch**, depth4 width32, one development seed100, at most two candidate configurations per family, at most10 epochs each. Families: gain-scaled clipped ordinary; alpha-P-S5; fixed GP M=0; fixed GP M>0. Native paper-style S5 gets one unswept reference configuration. All four candidate families use base learning rates1e-3 and3e-4 with the same schedule definition. Added physical coefficients are fixed, not swept or optimized: T=5I, normalized gamma=1, M=0 or normalized M=3.75I. Ordinary/reference parameters follow their declared common/published optimizer recipe. Freeze all groups and settings before execution.
3. Cap this first development batch at **two GPU-hours total**, including compile/validation. Estimate from stage1; if it will exceed the cap, reduce the common epoch budget for all families and record that decision before development. Never stop only the currently losing arm. Save resumable state and report if the cap prevents a complete balanced batch.
4. Select configurations by validation accuracy, then validation cross-entropy, then the declared candidate order. Screen favorable candidates on validation only. If none exceeds the best reference, report that within this bounded batch; do not start an undeclared rescue sweep.
5. The primary physical candidate is finite-mass GP. If it improves validation, freeze its setting and compare it with native S5, alpha-P-S5 and gain-scaled clipped ordinary using fresh paired seeds0,1,2 and the source's full training/early-stopping schedule. M=0 success alone can trigger a separately labeled reduced-model confirmation, never replacement of the physical candidate's recorded result. Maximum **eight additional GPU-hours total** for this initial cluster phase. If the full balanced comparison cannot fit, retain resumable runs and report them as incomplete; do not call a shortened run a full reproduction. Test each selected checkpoint once after all corresponding training/configuration choices are fixed.

The full comparison is authorized within these caps; it need not await another planning-only approval if the criteria and access are satisfied. The first screen is deliberately small. Three seeds are preliminary and differ from the paper's five-seed aggregate. More seeds, widths, raw audio, Path-X, Mamba, meta-learning and new tuning are outside this batch.

Positive preliminary outcome: same-task mean accuracy improvement over the reproduced alpha-P-S5 **and** the matched gain-scaled ordinary control, positive paired difference in every confirmation seed, with parameter/state/runtime costs disclosed. Predeclare a practical target of at least .3 percentage points in mean accuracy. This is a screening target, not proof of statistical significance. If only faster learning or selected temporal objectives improve, report that narrower result. Never suppress the other objectives or comparison arms.

For a finite-mass winner, an ordinary SSM with the same internal state count is required before claiming a benefit beyond added dynamic capacity; it has a different trainable parameter count, which must be reported. Design and record that comparator before its outcomes. It can be a subsequent bounded study after the first benchmark batch; the initial claim is only same-external-width, equal-added-parameter comparison. The existing modal control remains useful for M=0. Do not claim that parameter count alone establishes equal representation or compute.

## 7. Spatial-only request: second priority, clearly separated

Prepare the gradient-mode API while editing if straightforward; do not multiply the first benchmark grid. Use the physical(s,v) carry for positive mass and state the exact carry for M=0. Detach every previous recurrent/auxiliary component, retain current coefficient derivatives and same-time layer gradients. For two-tap inputs, detach previous activations, not the current weights multiplying them. A parameter-dependent carry-coordinate change can change a truncated parameter gradient.

See the supplied analytical note for the finite-mass limitation: at fixed positive M the recurrent-core current-input gain is O(h), so a direct-lead small-step argument cannot be assumed unchanged. State what residual skips/native D and temporal normalization contribute; full-sequence batch statistics can create other cross-time dependencies. Do not label a graph spatial-only until these paths are audited. Compare gradient modes at the same loss placement for attribution; separately reproduce any source-specific loss convention. No spatial-only training in the initial capped batch, and no meta-learning now.

## 8. Execution, persistence and reporting

Use the verified shared cluster environment `/Local/durso/prospective_ssm_project/.venv`, not a presumed repo `.venv`. Verify its current packages/backend without blindly upgrading or replacing the shared environment. Use `/Local/durso/final-prospective-s5` for checkout if present and `/Users/durso/s5-runs/` for persistent run artifacts, after verifying both. The reported host is `pgi15-gpu3.iff.kfa-juelich.de`, RTX3090; verify actual access and hardware. Do not silently train on a login node or someone else's occupied allocation.

Use a single detached cluster launcher, resumable manifest and one GPU worker by default. Preserve existing unrelated jobs. Set and record deterministic XLA, highest matmul precision, preallocation and offline logging consistently. Keep historical smokes unchanged. Record compile versus steady-state time and real peak memory. Respect the predeclared GPU-hour caps even if the laptop disconnects. If access is unavailable, give the user short single-line setup and launch commands with the exact new pushed commit; do not claim execution.

Deliverables:

- Working branch/commit plus pushed public link; source provenance for ported/reference material.
- `docs/GP_IMPLEMENTATION_REPORT.md` updated with actual equations, clock normalization, coefficient policy, carry/output timing, and new executed checks.
- `docs/RAWAT_BASELINE_MAP.md`, `docs/GP_RAWAT_CLUSTER_PROTOCOL.md` and machine-readable run manifest with every candidate, seed and status.
- One cluster launch command and a resume/status command, documented without claiming unexecuted flags work.
- Best and resumable checkpoints, full-precision metrics, split hashes, code/environment/hardware identity and artifact path for every run. Public Git gets code/reports/manifests, not datasets, large checkpoints or credentials.
- An honest table separating historical local positives, unsuccessful fixed-coefficient confirmation, numerical verification, GPU integration, validation screening and final benchmark comparisons.

Send the coordinator the pushed commit and report path after implementation and the first actual cluster batch. Preserve incomplete/failing results and explain the next concrete blocker if one exists; do not return merely another proposed plan when executable work is available.
