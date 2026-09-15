# Post-run algebra: learned damping duplicates the S5 clock

15 September 2026. Static source inspection and analytic derivation only. No new numerical experiments or training. Reviewed the reported execution `4848ee4989474bdb2bd1ddaacc9095b6301bb507`, response-summary output, and report/code at `d6c4a222568dcdc93974164c1a8d39015e08aa2e` in `/Users/alessandrodurso/Documents/final-prospective-s5`.

The constrained-response screen remains unsuccessful against its stated literature comparators. The new facts below change the interpretation of parameter learning, not the recorded scores. The coordinator should have identified the clock redundancy before specifying both learned leaves.

## 1. Exact redundancy for every rho, not just a limiting case

For one mode, write the implemented equation as

\[
\gamma_n\rho T\ddot s+\gamma_n\dot s+r_n+T\dot r_n=0,
\quad r_n=-\Delta\lambda s-\Delta B_c x.
\]

All parameters are constant during a forward sequence; T is fixed independently of Delta. Divide the entire equation by gamma_n and define

\[
\widehat\Delta=\Delta/\gamma_n,\qquad
\widehat r=-\widehat\Delta\lambda s-\widehat\Delta B_c x.
\]

The SAME equation becomes

\[
\rho T\ddot s+\dot s+\widehat r+T\dot{\widehat r}=0.
\]

Thus `(Delta, gamma_n, rho)` is input-output equivalent to
`(Delta/gamma_n, 1, rho)` for every admissible rho. This is not just agreement of a few poles or of the DC gain.

It also holds directly in the implemented carry `(s,v)`:

\[
\rho\dot s=-r_n/\gamma_n-(1-\rho)v,\qquad
T\dot v=\dot s-v.
\]

Both continuous state matrices and input matrices, and therefore their exact ZOH discretization and scan, are identical after the transformation. No carry-coordinate transformation is needed here. The same initial carry and readout produce the same output trajectory. This does not contradict the separate need to transform state when changing a physical coordinate definition in other realizations.

The code satisfies the required conditions:
- `rawat_s5._native` uses `Delta=exp(log_step)`;
- `B_c=diag(-Re(lambda)) B_tilde` depends on lambda and B_tilde, not Delta or gamma_n;
- `gp_fixed.mass_block_generator` uses a and b through `a/(gamma_n rho)` and `b/(gamma_n rho)`, with all other entries independent of gamma_n;
- native log_step is learned, and its initialization range is not a hard forward bound.

In particular, the transformation does not alter native pole clipping, because it changes Delta while leaving lambda unchanged.

The family still has positive component realizations. The point is that distinct physical realizations can have indistinguishable responses at the chosen input/output ports. Physical admissibility does not guarantee parameter identifiability.

## 2. What the 128 extra parameters mean

The implementation has 128 additional real leaves at 64 stored modes, and that parameter count remains correct. But only the 64 rho coordinates add an independent per-mode response-shape freedom relative to the previously trainable clock. The 64 gamma_n coordinates introduce a redundant parameterization of that clock.

The existing S5 parameterization may contain other redundancies; the statement above concerns this particular extra freedom, not the rank of the entire network parameterization.

This equivalence concerns the forward model/data loss. It does not imply the two training procedures are identical. AdamW, log-coordinate weight decay, bounds, and global gradient clipping can change optimization when a redundant parameter is added.

In the interior of the response clipping bounds, with all other parameters held fixed and for the differentiable data loss,

\[
\mathcal L(\ell,\eta,\rho)=\widetilde{\mathcal L}(\ell-\eta,\rho),
\quad\ell=\log\Delta,\ \eta=\log\gamma_n,
\]
\[
\boxed{\partial_{\ell_i}\mathcal L=-\partial_{\eta_i}\mathcal L.}
\]

Finite precision, clipping boundaries, or explicitly parameter-dependent regularizers require the corresponding qualifications. AdamW decay is not evidence against this data-loss identity.

For any subsequent concise parameterization, gamma_n can be fixed to one and its fitted value absorbed into log_step. That preserves the current trained function, in exact arithmetic. It is not a justification to restart training now, and it would not preserve optimizer continuation without separately addressing optimizer state and update rules.

## 3. Meaning of rho approaching one

For fixed J=j and input coefficient b the transfer is

\[
G(p)=\frac{b(1+Tp)}{\gamma_n\rho T p^2+(\gamma_n+Tj)p+j}.
\]

At rho=1 the denominator factorizes, giving

\[
G(p)=\frac{b}{\gamma_n p+j}.
\]

This is an ordinary memory-bearing SSM response, with its clock rescaled by gamma_n. It is NOT the professor equation's memoryless equilibrium map. The rho=1 boundary can be considered algebraically even though the implemented admissible interval ends below one.

Observed median rho increased from .75 to .83998; layer 2's median is .8824 and one mode reaches .9984. This shows movement toward that limit in parts of the population. It does not prove the full trained network became equivalent to ordinary S5, that all responses accelerated, or why the accuracy screen failed. Those depend on joint learned poles, drives, readouts and response residues.

## 4. The professor control has another exact cancellation

Its implemented zero-history map is

\[
K=-\frac{\Delta B_c}{\Delta\lambda}=-\frac{B_c}{\lambda}.
\]

Therefore its exact data-loss gradient with respect to log_step is ZERO. The report's Section 5 assertion of a nonzero log_step gradient is not a correct property of this model. Floating-point cancellation can give a small numerical residual; a test asserting merely `>0` is not evidence of a genuine gradient path.

Ordinary B/C/D and appropriate pole degrees of freedom can still learn; no claim that all native parameters remain identifiable is needed. A future implementation can form `-B_c/lambda` directly, leaving log_step structurally unused in this arm. Record the old implementation/score as executed; do not silently rewrite its provenance or rerun training to make this reporting correction.

## 5. Corrections needed in the report

1. Replace causal/general statements that response learning 'made it worse' or 'cost .052 pp' with the observation that this run scored .052 pp lower (three examples out of 5,783), with no established degradation or improvement. The failed predeclared screen remains explicit.
2. Delete the inference of a flat objective or that the optimizer 'explored the family and found nothing better'. Similar final accuracy does not establish objective curvature or search completeness. The exact redundancy above establishes one flat reparameterization direction for the data loss; it says nothing comparable about the rho direction or the full training objective.
3. The saved best checkpoint has no response values at a bound. This does not establish that bounds were never active during training; use checkpoint-specific language unless trajectory evidence exists.
4. Marginal medians of gamma_n and rho do not establish per-mode compensatory motion or an approximately constant-mass trajectory. The mass distribution is observed; a within-mode joint/trajectory claim needs the corresponding data. No new run is needed just to soften the claim.
5. The 84.85% memoryless-core classifier score is not an additive decomposition of accuracy into static and memory contributions, a performance floor, or a ceiling on what recurrence could contribute. It measures one trained architecture/configuration on this pooled classification task.
6. Deltas of -.398 and -.795 pp are differences from comparator accuracy. Shortfalls relative to a +.3 pp target are approximately .698 and 1.095 pp, respectively; do not confuse these quantities.
7. Describe the excluded tiny-mass corner as a non-finite result of the tested numerical exponential implementation on the probed matrices. A finite matrix has a finite mathematical exponential; this is not a physical mass threshold or a universal numerical frontier determined by mass alone. Full matrix norm, pole scale, conditioning and the implementation's scaling/squaring limits also matter.

## 6. Next action without training

Correct the report and record this analytic redundancy. Use the already saved gamma_n, rho and native log_step arrays when describing what was learned: the observable clock coordinate is `log_step - log(gamma_n)`, alongside rho, rather than gamma_n alone. If convenient, add this to the existing read-only summary. No new numerical test suite, training, sweep, or activity-dependent adaptation is requested by this review.

Do not claim numerical verification of the reparameterization unless it is separately performed and recorded. The proof and inspected coefficient formulas establish the exact-arithmetic identity; the recorded training measurements remain the user's supplied cluster results.
