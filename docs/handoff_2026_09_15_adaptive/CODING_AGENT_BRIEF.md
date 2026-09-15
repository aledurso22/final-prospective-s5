# Adaptive generalized prospective coding in the recurrence: bounded follow-up

Coordinator brief, 15 September 2026. No numerical runs were performed to prepare this brief.

## User-authorized objective

Reuse the Stage 2 Speech Commands 10 experiment, add activity-dependent physical adaptation to generalized prospective coding in the recurrence, and train prospective coding in the recurrence (the professor equation) once as an explicit cancellation control. Continue full BPTT. Do not implement spatial-only BP, a separate TSS experiment, or a MAML training loop in this batch.

The user wants adaptation consistent with the circuit derivation, not independent learned mass/damping/horizon arrays. Static response parameters optimized between batches are NOT the activity-dependent adaptation requested here. Ordinary neural adaptation is also not, by itself, evidence of meta-learning.

Implement in an isolated branch/worktree from the current diagnostic work. Record checkout path, branch, starting commit, final commit, and the exact execution commit. The coordinator inspected a clean `stage2-diagnostic` checkout and the current `s5/rawat_s5.py`, `s5/gp_fixed.py`, `s5/physical_coefficients.py`, `experiments/gp/rawat_benchmark.py`, and Stage 2 protocol. Preserve historical artifacts and results. Push code, the new protocol, and the eventual report to the existing public repository after checking that no data, credentials, or host-private artifacts are included.

## What is already reusable, and what is not

Reusable: cached train/validation data and split identity, the Rawat/S5 architecture, ordinary parameter initialization, optimizer, full-BPTT training loop, checkpoints, manifests, and literature comparator implementations. The older `full_state_pc` implementation provides a cancellation reference but is not yet a selectable arm in the Stage 2 substrate.

New scientific work: the activity-dependent circuit-to-S5 map and its causal discrete recurrence. `mass_block_generator` and `mass_block_zoh` currently implement CONSTANT response parameters. Broadcasting new coefficients across tokens does not establish that they implement the requested circuit.

Complete the derivation and focused validation before starting adaptive-arm training. This is a correctness dependency, not a request to seek another permission for work the user has already authorized. If a faithful map cannot be completed, finish independent professor-control work and report the precise missing map; do not silently substitute a generic gate or static learned response.

## A. Required adaptive-model contract

### Published starting point

NLA Appendix 6, Eqs. 80–81:

\[
c_s\dot u=g_L(E_L-u)+g_{sd}(v-u),
\]
\[
c_d\dot v=g_L(E_L-v)+g_E(t)(E_E-v)+g_I(t)(E_I-v)+g_{ds}(u-v).
\]

The paper specifies conductance inputs of the form `g_E = W_E rates`, and similarly for inhibition. Its Eq. 82 eliminates the fast dendrite. Our extension retains finite `c_d`. The resulting instantaneous dendritic response time is `c_d / (g_L + g_E + g_I + g_ds)`.

Sources:
- https://elifesciences.org/articles/89674 (Appendix 6, Eqs. 80–95; the publisher PDF or PMC copy can be used if HTML is inaccessible).
- https://arxiv.org/html/2511.14917v2 (TSS Eq. 5 and its actual memory experiment, Section 3.3).
- https://arxiv.org/html/2609.04134v1 (Rawat baseline).
- Existing `docs/handoff_2026_09_15/DERIVATION_CONTRACT.md` and its referenced circuit audit.

### Separate three levels explicitly

1. Published compartment/conductance equations.
2. The finite-dendrite prospective source/feedback construction in our derivation.
3. The computational mapping to a complex S5 mode and its input channels.

The passive compartment alone does not generate arbitrary prospective recurrent feedback. Keeping Eq. 81 while dropping the prospective source relation would be a different conductance RNN. Conversely, arbitrary complex S5 weights are not automatically nonnegative biological conductances.

Specify before training:

- Which capacitances and intrinsic conductances remain fixed, using the existing symmetric reference where compatible.
- Which existing synaptic/input quantities determine the nonnegative variable conductances; their units, reversal potentials, and excitation/inhibition map.
- How a complex mode is represented by real coordinates, and how the two quadratures share or do not share physical components.
- Which quantities learn through BPTT and which adapt within the sequence. Prefer reuse of existing drive parameters; do not claim zero added parameters until counted.
- Both the drive-current and conductance effects of each synapse. Keeping only a favorable time-constant effect while omitting the associated current is not the stated circuit.
- The precise prospective source convention and its time-dependent version. Distinguish prospecting a voltage from prospecting a current divided by a variable conductance.
- The continuous update, normalization, causal discretization, initial conditions, input-jump handling, reset and chunk boundary semantics.

Any dimensionless calibration or voltage scale not fixed by the existing model must be stated as a modeling assumption BEFORE validation scores, not attributed to NLA or selected from a sweep. If a new parameter map is necessary, expose its added parameters and give the adaptive control the same opportunity.

### Important algebraic checks

In the reciprocal frozen-conductance sector:

\[
\kappa=G_s-h^2/G_d,quad T=c_s/\kappa,
\quad\tau_d=c_d/G_d,
\quad\gamma=G_s\tau_d/\kappa,
\quad M=T\tau_d.
\]

These are tied coefficients, not independent knobs. Re-derive their role when conductances vary; an instantaneous parameter identity does not alone establish a time-varying state realization.

For example, the EXISTING block realization writes

\[
A_c(t)\dot s=-r-B_c(t)v,\qquad T(t)\dot v=\dot s-v,
\quad A_c=\gamma\rho,\quad B_c=\gamma(1-\rho).
\]

Direct differentiation gives

\[
M\ddot s+\gamma\dot s+r+T\dot r
=-T\left(\dot A_c\dot s+\dot B_c v\right).
\]

Thus simply making this block's coefficients time dependent generally changes the eliminated law. This identity diagnoses a possible shortcut; it does NOT assert that all variable-conductance eliminations require these particular extra terms. Derive the chosen realization from its physical coordinates. Parameter-dependent coordinate changes also require correct transformations of the carried state when parameters jump.

Do not silently renormalize by `gamma(t)` while retaining the old `J` and `B`; show the whole-equation mapping. Absorb native learned Delta once, preserving the existing sample-time convention.

Use a layer-input-dependent realization if the physical mapping supports one: precomputable per-token affine blocks can retain associative scan. If dependence on the same layer's recurrent state is essential, use a sequential recurrence and report its cost; a new parallel scan does not follow automatically. Positive conductances/frozen stable poles do not prove stability of an arbitrary active, time-varying feedback loop.

## B. Prospective coding in the recurrence: exact professor control

Use the SAME gain-scaled, clipped S5 substrate as the matched control. Let

\[
a=\Delta\Lambda,\quad b=\Delta B_c,\quad J=-\operatorname{diag}(a),
\quad r=Js-bx.
\]

Implement `r + T r_dot = 0`, with `T=5` and consistent zero residual prehistory. For invertible J its driven realization is exactly

\[
s_k=J^{-1}b x_k,
\qquad y_k=2\operatorname{Re}(\widetilde C s_k)+D\odot x_k
\]

under conjugate symmetry. The native D, output timing, and rest of the architecture remain unchanged. Use a solve or the exact diagonal division, not a backward-difference derivative that creates an extra recurrent root. Ensure all dispatchers reject unknown laws instead of falling through to the mass case.

Required observations:
- The standalone core has zero driven history beyond lag zero and zero dependence on earlier inputs with current input fixed.
- Its ordinary weights can still receive gradients and learn a static mapping.
- A pooled speech classifier can perform usefully without a recurrent memory. Do NOT make low accuracy or failure to optimize a correctness condition.
- Run history checks on the isolated core or inference model with fixed normalization statistics. Global pooling and training-time batch normalization can otherwise obscure what the test measures.

This control illustrates the fully matched recurrence placement; it does not represent the entire TSS paper, which explicitly includes memory-bearing non-prospective neurons in its temporal task.

## C. Comparison and training scope

Keep the Stage 2 problem unchanged: cached SC10 MFCC (161 frames, 20 features), depth 4, width 32, 8 HiPPO blocks, conjugate symmetry, same normalization/residual/half-GLU/readout, batch 32, full BPTT. Use seed 100, learning rate 1e-3, 10 epochs with the EXACT existing 10-epoch schedule, optimizer and highest-precision deterministic settings. No new learning-rate, horizon, or physical-scale sweep. Test split remains unopened.

Primary new treatment: adaptive generalized prospective coding in the recurrence, with positive retained dendritic capacitance.

Additional new arms:
1. Prospective coding in the recurrence, once, as above.
2. A matched ordinary adaptive SSM using the same modulation information and comparable adaptation parameters. Document the response mapping and any state/cost mismatch; this is needed to separate general adaptivity from the prospective law.

References: existing Rawat prospective-input S5, gain-scaled/clipped ordinary S5, native S5, and fixed positive-mass generalized recurrence. The old first-order zero-mass ablation is not rerun or deleted.

Historical reference scores may be reused to keep the batch short only when the old configurations, data identity, training schedule, initialization and comparator paths remain unchanged. Label reused measurements as historical, with their exact commits. Validate unchanged paths at parameter, forward, gradient and update level. If shared code changes comparator behavior, rerun the affected comparator within the cap or report that the comparison is incomplete; never splice incompatible scores.

If freezing modulation reproduces the previous positive-mass model, verify that identity. If the adaptive mapping changes the underlying architecture or parameterization even when frozen, add a frozen version of THAT new model as its within-family control; the old mass result cannot substitute for it. Count this extra arm in the budget before training.

The same SC10 screen measures classification under adaptive dynamics. It does not establish meta-learning, causally useful memory allocation, or performance on unseen temporal regimes. Keep those as later questions.

## D. Small correctness and cost preflight, on the cluster

Reuse existing tests; add focused tests for the new risks rather than another broad diagnostic project:

1. Conductance admissibility, coefficient/source ties, and claimed frozen-response limit.
2. Independent reference integration of the unreduced specified circuit versus the implemented continuous/discrete model, including changing inputs AND changing conductances. A frozen matrix exponential alone is not a sufficient reference for adaptation. Predeclare numerical tolerances appropriate to integration error and dtype.
3. Derivatives through adaptation against finite differences/JVPs in actual production float32/complex64; do not detach the coefficient or modulation path.
4. Streaming/chunk/reset equivalence, plus causal dependence of modulation itself; record the scope of any normalization-induced cross-time dependence.
5. Professor-control history cancellation and nonzero ordinary parameter gradient paths.
6. Two optimizer updates and measured steady-state time/memory for new arms. Cost an actual training batch, including the adaptive kernel, before projecting the batch runtime.

No local numerical runs by the coordinator or coding agent for this batch. Static editing/inspection is fine; numerical tests, integration and training run on the cluster. Reuse the existing environment; do not install or replace shared packages incidentally.

Set a HARD TOTAL cluster budget of 20 minutes for focused numerical checks, compilation, preflight and training, with an actual watchdog and cleanup reserve. The previous entire nine-run Stage 2 batch took 716 seconds, but that is NOT a timing estimate for an unimplemented adaptive kernel. If the measured projection cannot fit, report the cost and finish correctness work; do not start a long run, silently shorten selected arms, or automatically widen the budget. Record incomplete work honestly.

## E. Report and deliverables

Before training, commit the final model contract, arm definitions, numerical criteria and bounded run plan. This is a NEW development experiment informed by Stage 2, not a reopening of Stage 2's failed screen.

After execution write `docs/ADAPTIVE_PROSPECTIVE_RECURRENCE_REPORT.md`, link it from the existing implementation report, and include:
- checkout, branch, commits, complete model equations and assumptions;
- exact data/environment/hardware/seed/commands/artifact paths;
- actual correctness results and statuses, including omissions;
- parameter, state, memory and runtime costs;
- validation accuracy and CE, with differences against the Rawat reference, ordinary matched control and ordinary adaptive control;
- the professor control's actual training score alongside its verified zero-history property;
- adaptation activity (distribution of physical response quantities and their context dependence), with no inference that variability alone proves useful memory selection;
- whether each reference was rerun or reused, and why comparisons are valid.

Retain the existing +0.3 percentage-point margin over BOTH Rawat's prospective-input reference and the gain-scaled/clipped ordinary reference as a practical screening target. Additionally report the comparison to the ordinary adaptive control. No single-seed outcome is statistical confirmation. No automatic larger run or rescue sweep follows.

Once implementation, the model contract and preflight CODE are ready, return the public commit link and a SHORT cluster launcher for the user. The launcher runs preflight first and then the bounded training batch only if its checks pass and the measured projection fits. Do not describe unexecuted tests as passed. Return the report link after actual execution. The coordinator cannot directly control the user's Claude Code session or the cluster.
