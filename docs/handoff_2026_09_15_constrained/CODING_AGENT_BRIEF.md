# Corrected scope: literature-analogous constrained response learning

15 September 2026. This brief SUPERSEDES the activity-dependent adaptation brief at `outputs/adaptive_recurrence_handoff_2026_09_15/CODING_AGENT_BRIEF.md`.

The user explicitly clarified: use the analogous coefficient policy of TSS/Rawat FIRST. Keep the derived dynamical law; learn designated constrained model parameters with full BPTT; derive the remaining coefficients. Parameters are constant during each forward sequence. Do not implement within-sequence conductance modulation, a new adaptive controller, spatial-only BP, or meta-learning for this batch.

The coordinator observed that the coding agent has already created branch `adaptive-recurrence` with uncommitted work. Preserve that work; do not discard it or overwrite it indiscriminately. Redirect the implementation to the policy below and record which superseded work is retained but unused. No numerical runs were performed by the coordinator.

## 1. Exact model, keeping the existing clock and source convention

Keep the published-comparison horizon FIXED at `T=5` input intervals. Existing clipped S5 poles, gain-scaled input map, output, learned native steps, and full BPTT remain unchanged. Absorb Delta once:

\[
F_0=\operatorname{diag}(\Delta\Lambda),\qquad
B_0=\operatorname{diag}(\Delta)B_c,\qquad
r_n=-F_0s-B_0x.
\]

For each stored complex mode i, learn two real quantities, `gamma_n_i > 0` and `0 < rho_i < 1`, shared by its conjugate partner. They determine the response:

\[
\boxed{\mu_i\ddot s_i+\gamma_{n,i}\dot s_i+r_{n,i}+T\dot r_{n,i}=0,
\qquad \mu_i=T\gamma_{n,i}\rho_i.}
\]

Initialize `gamma_n=1`, `rho=0.75`, so `mu=3.75` EXACTLY matches the prior fixed positive-mass model. Do not learn a separate mass or prospective horizon. Do not normalize gamma_n back to one after an update: that would change which physical coupling is held fixed.

Per-mode learning is compatible with the existing diagonal S5 sector: each complex mode has a scalar positive gamma_n and scalar rho. No claim about arbitrary noncommuting learned response matrices is needed.

## 2. Positive physical realization of every allowed coefficient pair

This section gives the missing concrete component map. It is a family within the existing reciprocal, equal-leak, additive-current operating-point model, not an identified biological parameter set.

Choose conductance units with `kappa_0=3/2`, the original reference conductance. Fix the whole-equation normalization `gamma_ref=5`. For each mode set

\[
c_s=T\kappa_0=7.5,\qquad
c_{d,i}=\gamma_{\rm ref}\gamma_{n,i}\kappa_0=7.5\gamma_{n,i},
\]
\[
G_{s,i}=G_{d,i}=\kappa_0/\rho_i,\qquad
h_i=G_{s,i}\sqrt{1-\rho_i},\qquad
g_{L,i}=G_{s,i}-h_i>0.
\]

For stable evaluation, the equal leak can also be written
`g_L = kappa_0 / (1 + sqrt(1-rho))`.

These are positive capacitances/conductances, with reciprocal axial coupling and the same leak in the two compartments. They give IDENTICALLY

\[
\kappa_i=G_{s,i}-h_i^2/G_{d,i}=\kappa_0,
\qquad T_i=c_s/\kappa_i=T,
\]
\[
\tau_{d,i}=c_{d,i}/G_{d,i}=\gamma_{\rm ref}\gamma_{n,i}\rho_i,
\]
\[
\gamma_{{\rm phys},i}=G_{s,i}\tau_{d,i}/\kappa_i
=\gamma_{\rm ref}\gamma_{n,i},
\quad M_{{\rm phys},i}=T\tau_{d,i}.
\]

Define `r_phys = gamma_ref * r_n`, with gamma_ref FIXED. Dividing the WHOLE physical equation by gamma_ref gives exactly Section 1 and `mu = M_phys/gamma_ref = T gamma_n rho`.

At initialization the components are `c_s=c_d=7.5`, `h=g_L=1`, `G_s=G_d=2`, recovering the original symmetric reference. Learning departs from equality of capacitances and from equality of leak and axial conductances, while maintaining the displayed physical relationships and fixed prospective horizon. These equalities were initialization/modeling assumptions, not universal physical laws.

The construction still uses the previously declared prospective source/active feedback and learned additive-current interpretation of S5. Preserve its input/current conversion when expressing the circuit reference. A physical parameter manifold does not turn arbitrary complex S5 feedback or BPTT into a fully biological neural circuit or plasticity rule.

Since all parameters are held constant during a forward sequence, the existing constant-coefficient derivation applies. No new time-dependent coefficient terms, gate states, parameter-switching state transformations, or online conductance equations are part of this implementation. Update weights only between complete independent sequence batches as in the current benchmark.

## 3. Parameterization and optimizer policy

Use positive log parameterizations, analogous to the constrained learned filter quantities in Rawat:

\[
\gamma_{n,i}=\exp(\eta_i),\qquad \rho_i=\exp(\zeta_i).
\]

Initial leaves: `eta=0`, `zeta=log(0.75)`. Suggested explicit names: `log_response_gamma`, `log_response_rho`, each with shape `(P,)`, the number of stored complex modes.

For this single bounded screen, fix numerical admissibility bounds BEFORE scores:
- `1e-2 <= gamma_n <= 1e2`;
- `1e-4 <= rho <= 1-1e-4`.

These broad bounds are declared numerical design choices, not physiological measurements or numbers supplied by NLA. There is no bound/init sweep. Clip log values before exponentiation as a forward safety guard, and project the corresponding parameter leaves after optimizer updates so that raw leaves do not drift outside the allowed interval. Preserve the optimizer state and report the projection policy.

Use the existing paper-reproduction AdamW policy on these new leaves, including its common learning rate and weight decay. Do not add a response-specific learning-rate search. Record that weight decay is in log coordinates: this is an optimizer choice, not a derived plasticity law. All ordinary S5 parameters retain their existing optimizer policy.

Only the new learned-response arm may contain these two named parameter leaves. Keep fixed-response assertions for all historical fixed arms; replace the blanket prohibition for the new arm with an exact allowed-name/shape/count check. Do not silently relax the assertions globally.

At the current configuration (`P=16`, 4 layers), expect 128 added real trainable parameters, i.e. 35,178 versus 35,050, but verify from the actual tree. No new recurrent states are added relative to the fixed positive-mass model.

## 4. Reuse the existing numerical implementation

The constant-coefficient block equations remain

\[
\gamma_n\rho\dot s=-r_n-\gamma_n(1-\rho)v,
\qquad T\dot v=\dot s-v.
\]

Use `mass_block_generator`, `mass_block_zoh`, the existing block associative scan and physical-state readout. Forward trajectories are still linear within a layer and coefficients are computed once per layer/parameter update, not per token. Differentiate through both physical response parameters and the existing matrix exponential.

Specific integration pitfall: existing `gp_fixed.py` uses scalar gamma/rho. After introducing `(P,)` vectors, input-matrix divisions must use `(P,1)` row broadcasting, not accidental division along feature dimension H. Include a test with P != H. Keep T scalar. Do not wrap traced learned arrays in the immutable host-side `PhysicalResponse` dataclass or call its Python-value validation on tracers.

All response dispatches must be exhaustive. The current substrate has fallthroughs assuming a non-first-order new arm is the mass arm; repair or explicitly extend these without mislabelling diagnostics. Preserve all historical paths and the exact output timing.

## 5. Prospective coding in the recurrence: professor control

Keep this requested addition. Same gain-scaled/clipped input and pole substrate as the matched ordinary S5. With `J=-diag(Delta Lambda)` and `b=diag(Delta) B_c`, implement

\[
r=Js-bx,\quad r+T\dot r=0,
\qquad s_k=J^{-1}b x_k
\]

for consistent zero residual prehistory, followed by the common conjugate readout and native D. Reuse the old `full_state_pc` implementation as a reference, not as an unexamined differently normalized model. Avoid backward-difference parasitic recurrences.

Validate zero core history at nonzero lag. Train it once; its static weights can learn and a pooled speech classifier can be useful without recurrent memory, so low accuracy or failure to optimize is not a correctness requirement. This arm illustrates the fully matched recurrence, not the whole TSS memory architecture.

## 6. Small cluster batch, retaining the existing benchmark

Use exactly Stage 2 SC10 MFCC data, architecture and training schedule: depth 4, width 32, 8 HiPPO blocks, batch 32, seed 100, lr=1e-3, 10 epochs, full BPTT, highest precision, deterministic XLA, validation only. No zero-mass rerun, new task, wider sweep or within-sequence adaptation study.

New training runs:
1. Generalized prospective coding in the recurrence with constrained learned response, initialized at the existing fixed reference.
2. Prospective coding in the recurrence (professor equation).

References: Rawat prospective-input S5, matched gain-scaled/clipped ordinary S5, native S5, and the old fixed positive-mass generalized recurrence. Existing scores can be reused with explicit historical provenance if code, data, schedule, common initial parameters and optimizer behavior are unchanged. Verify this; rerun an affected comparator within the cap or label the comparison incomplete if equivalence fails.

Adding Flax parameter leaves must not silently change common-parameter initialization. If initialization differs, explicitly align the common parameter tree to the existing baseline initialization and record how. Prove that freezing gamma_n=1 and rho=0.75 reproduces the old fixed arm's outputs and gradients for shared parameters under the same initial state. Preserve the previous fixed-arm result; do not retrofit its protocol to permit learning.

The frozen-versus-learned comparison isolates allowing response learning within this model family. Literature comparisons measure the complete construction. Disclose the 128 expected additional parameters and the already doubled carry versus ordinary S5; this screen does not establish an advantage independent of parameter/state capacity. Do not insert arbitrary dummy parameters into a control to claim capacity matching.

## 7. Focused tests and cost, cluster only

No local numerical runs. Required new checks, reusing prior tests where possible:
- component reconstruction and coefficient identities, including the reference point;
- frozen-response equivalence to existing mass model;
- actual float32/complex64 gradient checks for BOTH new leaves, with finite-difference directions/steps that exceed rounding noise;
- both response leaves change after a nondegenerate small training update, not merely because of weight decay;
- log-parameter projection and admissibility at the bounds; finite matrix exponentials and derivatives over representative interior/boundary cases;
- P != H broadcasting, conjugate sharing, scan/stream/reset consistency;
- exact professor-core history cancellation and ordinary weight gradient paths;
- shared initialization/unchanged historical comparator behavior.

Keep a hard total 20-minute cluster budget including focused tests, compilation and training, with a real watchdog/cleanup reserve. Measure new-arm throughput before launching the batch. The recurrence and carry are unchanged from the fixed-mass arm, so reuse is substantial; do not state an unmeasured runtime as fact. No automatic long run or budget expansion.

## 8. Provenance and report

Record in the new protocol that the user's clarification superseded the prior activity-dependent brief BEFORE this study's numerical execution; if any superseded execution already occurred, retain and label it rather than claiming it did not happen. Preserve work from that direction without continuing it for this batch.

Commit/push the corrected protocol and code before numerical execution. Give the user one short cluster launcher that runs tests/preflight and then the two training runs if they pass and fit the budget. Use a new artifact directory. Never write into old Stage 2 run directories or open test arrays.

Write `docs/CONSTRAINED_PROSPECTIVE_RESPONSE_REPORT.md` from actual results, linked from the main implementation report. Include checkout, branch, commits, exact commands, hardware/environment, dataset identity, all test results, validation metrics, runtime/state/parameter counts, and final learned gamma_n/rho/derived mu distributions. State whether baselines were reused or rerun. Push public code/report and give their commit-specific links.

Keep the earlier +0.3 percentage-point margin above BOTH Rawat and matched ordinary S5 as the practical screen, not a statistical significance test. Report all observed differences and the fixed-versus-learned comparison. No outcome is assumed in advance and no larger run follows automatically.

Literature coefficient policy: TSS Eq. 5 ties the prospective coefficient to neuronal tau; its memory experiment learns memory-neuron timescales. Rawat Appendix E fixes the RQF/S5 prospective horizon while learning constrained underlying response quantities; the input correction adds no independent parameter. Our precise two-parameter component family above is our modeling choice within the derived circuit, not a coefficient prescription quoted from either paper.

Sources: https://arxiv.org/html/2511.14917v2 ; https://arxiv.org/html/2609.04134v1 ; https://elifesciences.org/articles/89674 ; existing `docs/handoff_2026_09_15/DERIVATION_CONTRACT.md`.
