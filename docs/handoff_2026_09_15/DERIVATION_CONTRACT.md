# Derivation contract for the cluster model

This document governs the coding brief. The user requires prospective and generalized prospective dynamics to remain faithful to their derivation. Benchmark improvement is not permission to add unmotivated coefficients, hide approximations or claim an unestablished biological derivation.

## Starting circuit and retained component

NLA Appendix6 Equations80–81, in the reciprocal constant-conductance/small-signal sector, read

\[
c\dot u+G_su-hv=I_s,\qquad c_d\dot v+G_dv-hu=I_d.
\]

Retaining dendritic capacitance c_d is the modification. Define kappa=G_s-h²/G_d>0, T=c/kappa and tau_d=c_d/G_d. With constant I_s, exact elimination gives

\[
c\tau_d\ddot u+(c+G_s\tau_d)\dot u+\kappa u=b,
\qquad b=I_s+(h/G_d)I_d.
\]

Using the original prospective-source relation b=(1+TD)bbar with consistent causal initialization, and r=u-bbar/kappa, yields

\[
M\ddot u+\gamma\dot u+r+T\dot r=0,\quad
\gamma=G_s\tau_d/\kappa,\quad M=T\tau_d,\quad
\rho=M/(\gamma T)=\kappa/G_s.
\]

The passive plant supplies these tied coefficients. It does not itself prescribe the prospective closed-loop source. Substituting bbar/kappa=f_theta(u,x) is an explicit feedback/source convention; do not infer that a passive capacitance alone generates the total residual derivative for arbitrary f.

Changing I_s adds its derivative; changing conductances adds coefficient/source derivatives. The S5 experiment assumes fixed intrinsic/background conductances and learned additive-current coupling. Literal conductance-synapse learning would require the unreduced circuit or a new derivation.

## Fixed reference and exact S5 normalization

Use the simplest symmetric intrinsic reference: c_s=c_d=c, g_L=h=g>0, zero background synaptic conductance. Then G_s=G_d=2g, kappa=3g/2, tau_d=3T/4, gamma=T, M=3T²/4 and rho=3/4. Component equalities are declared idealizations, not an identified numerical NLA cell.

Set T=5 input intervals to align the computational horizon with the benchmark. This is a model-time convention; MFCC frames and audio samples do not both last the2ms assumed in earlier synthetic work. Unit-sample coefficients are T=5, gamma=5, M=18.75. They are not trained or chosen from validation results.

For native S5 generator F0 and input B0, choose physical current residual matrices J_phys=-gamma F0 and B_phys=gamma B0. Dividing the **whole** equation by gamma gives

\[
(M/\gamma)\ddot s+\dot s+r_n+T\dot r_n=0,
\qquad r_n=r/\gamma=-F_0s-B_0x,
\qquad M/\gamma=3.75.
\]

Thus normalized code uses gamma_n=1, T=5 and mu=3.75. Scale residual/input matrices with the mass; changing only gamma or M changes the model. Other positive capacitance ratios can produce this same normalized transfer when physical current couplings are rescaled accordingly; that is not equality at fixed physical synapses.

Native S5 pole/input/readout/step parameters train as in the comparator. Added response coefficients remain immutable. A nonsymmetric S5 residual is not automatically the gradient of the original NLA mismatch energy. Its active closed-loop current feedback needs the declared matrix stability constraints; it does not inherit the passive plant's entire interpretation.

In this mapping, learned native Delta changes effective recurrent/input coupling; it does not change the physical sample interval or independently retime the fixed compartment. Absorb Delta once in F0/B0. When checking the actual c_d→0 circuit limit, hold physical current couplings fixed; do not continue the singular gamma=1 normalization through gamma=0 or rescale the physical couplings to zero while calling it the same limit.

## Meaning of the comparison arms

- **Finite-dendrite primary:** the full positive-mass law above. Preserve circuit-compatible state/velocity and source-jump initialization. Read only physical state, with the benchmark's common native output skip.
- **M=0 with gamma>0:** a separate reduced/constitutive ablation. Dropping acceleration is a low-frequency approximation only when its omitted response is small in the relevant band. A score improvement outside that regime is not validation of the biological approximation.
- **True fast-dendrite limit:** c_d→0 sends both gamma and M to zero at fixed intrinsic parameters. The resulting r+T rdot=0 has the previously verified driven-memory cancellation in the fully matched linear placement. It is not Rawat's input-only law.
- **Rawat reference:** retain its independently specified input-side prospective equation and entire S5 recipe; do not rewrite it into our residual law.

## Traceability required before comparative training

Write a source-to-code table: circuit equation, retained component, fixed assumption, normalized coefficient, initial condition, exact discrete update, buffer/parameter location, and what can learn. Check normalized versus unnormalized trajectories and input jumps on the cluster. Check exclusion of fixed coefficients from gradients, optimizer updates and weight decay.

The scalar passive circuit reduction/variational representation is derived in the stated sector. Applying its response law to learned S5 current feedback is a computational extension. The full nonlinear NLA learning theorem and literal conductance-network interpretation are not inherited. Galley's framework organizes a specified nonconservative interaction; it does not select an arbitrary neural feedback law or guarantee improved task learning.

If a mapping omits a required term or constraint, repair it algebraically or report the specific gap before training that arm. Do not substitute an unrelated trainable correction. If the full physical candidate loses while M=0 wins, claim only the reduced-model outcome.

Sources: references/CIRCUIT_DERIVED_NLA_ACTION.md, references/CIRCUIT_PARAMETER_IDENTIFICATION.md and references/FIXED_MODEL.md. Their older numeric experiments remain historical; this symmetric reference governs the new cluster test. No new training was performed for this handoff.
