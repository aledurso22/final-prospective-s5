# Prospective dynamics on the MDN write path: recurrence audit

19 September 2026. Branch `mdn-nesterov-qhm-audit`. This extends
`docs/MDN_NESTEROV_QHM_AUDIT.md` and leaves unchanged every full-held-out-set,
stability and preregistration correction of `e134ab1`. No completed result
is modified.

Exact checks are in `tests/test_prospective_realizations.py` (8 tests, pure
Python `Fraction`, passing locally). Production float64 checks, which run in
the cluster's fail-closed stage, are in
`tests/test_mdn_nesterov_qhm_production.py`.

## 0. Result

| Item | Result |
|---|---|
| Classification | **`EXACTLY_EQUIVALENT_REALIZATIONS`**: the direct finite-difference realization of the generalized law **is** the executed generalized recurrence, token by token, with the same transfer function |
| New arm | **none**. A generalized finite-difference arm would duplicate the executed one; **six arms** are retained |
| Zucchet learned-mismatch boundary | `M = 0`, `gamma + T = h`, `kappa = T/h`. Derived: it is the only place where the recurrence carries no auxiliary state |
| Zucchet matched-time-constants boundary | `M = gamma = 0`: Zucchet et al. Eqs. (5) and (17) |
| Zucchet adaptive current (Eq. 7) | **exact at `tau_a = h`** under forward Euler, for both the ordinary and the generalized law. The paper's own exactness condition, `tau_a -> 0`, is **not** what is executed |

## 1. The placement and the law

The executed placement (`filtered.py`) uses:
- the processing state `s = y`;
- the input `f = R_t = m_t (alpha_t W_(t-1) k_t - v_t) k_t^T`, the masked MDN
  residual evaluated after the native decay;
- the law's residual `r = s - f`;
- the output `y`, which feeds the unchanged native momentum
  `U_t = mu_t U_(t-1) + eta_t y_t`, `W_t = alpha_t W_(t-1) - beta_t U_t`.

The law is

    M s'' + gamma s' + (s - f) + T (s' - f') = 0.

Same-token output: token `t` produces `s_(t+1) = y_t` from `f_t = R_t`.

## 2. Eliminating the auxiliary state

The executed recurrence carries `y`, `y_prev` and `R_prev`:

    y_t = a y_(t-1) - b y_(t-2) + c R_t - d R_(t-1),
    A = M + h(gamma + T),  a = [2M + h(gamma+T) - h^2]/A,  b = M/A,
    c = [h^2 + hT]/A,  d = hT/A.

Eliminating the states gives the causal convolution
`y_t = sum_j h_j R_(t-j)` with

    H(z) = (c - d z^-1) / (1 - a z^-1 + b z^-2).

The exact impulse response, by long division, reproduces the executed
recurrence
(`test_eliminating_the_auxiliary_state_gives_the_same_transfer_function`).

## 3. The direct finite-difference realization

Take the difference choices of Zucchet et al.'s Eq. (17) (arXiv:2511.14917v2,
the "Euler-like implicit" scheme):
- a forward difference for `s'`;
- a backward difference for `f'`;
- the zeroth-order residual at `(s_t, f_t)`;

and add a centred second difference for `s''`:

    M (s_(t+1) - 2 s_t + s_(t-1))/h^2 + gamma (s_(t+1) - s_t)/h + (s_t - f_t)
        + T [(s_(t+1) - s_t) - (f_t - f_(t-1))]/h = 0.

Multiplying by `h^2` and solving for `s_(t+1)`:

    A s_(t+1) = [2M + h(gamma+T) - h^2] s_t - M s_(t-1) + (h^2 + hT) f_t - hT f_(t-1),

which is **exactly** the executed recurrence, coefficient for coefficient. It
was checked by solving the difference equation from its own terms, with no
use of `a, b, c, d`, for every admissible `(M, gamma, T)` including
`gamma < 0` (`test_direct_finite_difference_realization_is_the_executed_recurrence`),
and in float64 on the production rollout
(`test_executed_generalized_is_the_direct_fd_realization`).

The same equation is (Y) of `docs/PROSPECTIVE_TSS_CONTAINMENT_SPEC.md` §1.1.
Other difference choices would define *different* schemes: evaluating the
residual at `s_(t+1)`, or a central `f'`, both change `a`, `b`, `c` and `d`.
The executed scheme is the one consistent with Eq. (17), and none of the
others is executed.

**Classification: `EXACTLY_EQUIVALENT_REALIZATIONS`.** Its transfer function
is `H(z)` above, the same as §2. No generalized finite-difference arm is
added.

## 4. Boundaries (derived)

**Zucchet prospective dynamics, learned time-constant mismatch
\(\tau'\neq\tau\).** `a = 0` and `b = 0` hold together
**iff** `M = 0` and `gamma + T = h`
(`test_ordinary_finite_difference_boundary_is_gamma_plus_T_equal_h`). There,

    c = 1 + T/h,  d = T/h:   y_t = R_t + (T/h)(R_t - R_(t-1)),

which is the executed learned-mismatch arm (`ordinary.py`) with `kappa = T/h`,
membrane `tau = h` and `tau' = kappa h`. It is exact in closed loop on the full MDN write path, with
token gates and masks, for every `kappa`, including `kappa > 1`, where
`gamma = h(1 - kappa) < 0`:
- exact: `test_generalized_recurrence_recovers_the_ordinary_fd_arm_closed_loop`;
- production float64:
  `test_generalized_at_its_fd_boundary_is_the_executed_ordinary_fd_arm`.

**Domain caveat.** The executed generalized arm keeps the completed
protocol's passive repair (`gamma >= 0`), so inside the ladder it can reach
this boundary only for `kappa <= 1`. The completed ordinary
finite-difference runs learned `kappa ~ 1.87-2.14`. The containment is
algebraic; within the executed domain it is partial.

**Zucchet prospective dynamics, matched time constants \(\tau'=\tau\).**
`M = gamma = 0` gives
`a = 1 - h/T`, `b = 0`, `c = 1 + h/T`, `d = 1`, which is Eq. (17), the
executed `tss_processing` arm. It coincides with the learned-mismatch
boundary only at `T = h`
(`test_the_ordinary_adaptive_state_boundary_is_M_gamma_zero`).

So both Zucchet prospective dynamics arms are `M = 0` points of **one** recurrence, at
`gamma = h - T` and `gamma = 0`. They differ by the damping boundary, not by
a change of discretization
(`test_the_two_ordinary_arms_differ_by_gamma_not_by_realization`).

## 5. The mapping to Zucchet's adaptive current

Zucchet et al.'s Eq. (7):

    tau u' = -u + (1 + tau/tau_a) f - (tau/tau_a) a,    tau_a a' = -a + f.

The paper states that it approaches Eq. (5) as `tau_a -> 0`, and gives no
discrete scheme for it.

Discretize `u` with the Eq. (17) choices and `a` with forward Euler, step `h`.
**At `tau_a = h`**, `a_(t+1) = a_t + (h/tau_a)(f_t - a_t) = f_t`, so
`a_t = f_(t-1)` is exactly the stored `R_(t-1)`. The coefficient map is:

| Eq. (7) | executed |
|---|---|
| `tau` | `T` |
| `1 + tau/tau_a` | `1 + T/h` |
| `tau/tau_a` | `T/h` |
| `a_t` | `R_(t-1)` |

and the update becomes Eq. (17) term by term. The same holds for the
generalized law, with `M s'' + (gamma + tau) s' + s` on the left:
`test_adaptive_current_with_tau_a_equal_step_is_exactly_eq17`.

**What this does and does not establish.**
- The executed matched-time-constants arm is **exactly** Zucchet's
  adaptive-current realization at `tau_a = h`, and simultaneously **exactly**
  his Eq. (17). The paper itself presents Eq. (17) as its finite-difference
  scheme.
- This adaptive-current equivalence holds only with the qualification
  `tau_a = h` (forward Euler). It is **not** the paper's `tau_a -> 0` limit,
  and adaptive-current dynamics are not an arm in this experiment.
- Any other `tau_a` gives a distinct realization. Minimal counterexample: at
  `tau_a = 2h` the first write has amplitude `1 + T/tau_a = 3/2` against
  Eq. (17)'s `2` (`T = h`).
- `tau_a < h/2` makes the forward-Euler map of `a` unstable, so the
  `tau_a -> 0` limit is not reachable at a fixed step.
- **No** off-step adaptive-current realization is executed. Adding one would
  be a new model, with a new parameter `tau_a`, and is not done here.

## 6. Consequence for the accepted recurrence audit

The earlier realization question reduces to parameter boundaries of one
accepted recurrence:

| Scientific arm | Recurrence boundary |
|---|---|
| Zucchet prospective dynamics, matched time constants \(\tau'=\tau\) | `M = gamma = 0` |
| Zucchet prospective dynamics, learned time-constant mismatch \(\tau'\neq\tau\) | `M = 0`, `gamma + T = h` |
| Generalized prospective dynamics \((M,\gamma,T)\) | the recurrence with trainable `M, gamma, T` on the declared `gamma >= 0` domain |

The accepted proof therefore justifies one generalized arm, not an additional
arm. The final experiment does not label the Zucchet arms by implementation
style and does not report a law-by-realization factorial. It reports the five
pre-registered comparisons in §7.

## 7. Final pre-registered primary contrasts

| Scientific contrast | Executed as | Status |
|---|---|---|
| Generalized prospective dynamics \((M,\gamma,T)\) vs Native Momentum DeltaNet | `generalized_processing_vs_native_full` | primary |
| Generalized prospective dynamics \((M,\gamma,T)\) vs Zucchet prospective dynamics, matched time constants \(\tau'=\tau\) | `generalized_processing_vs_tss_processing` | primary |
| Generalized prospective dynamics \((M,\gamma,T)\) vs Zucchet prospective dynamics, learned time-constant mismatch \(\tau'\neq\tau\) | `generalized_processing_vs_operator_full` | primary |
| Generalized prospective dynamics \((M,\gamma,T)\) vs Literal Nesterov Momentum DeltaNet | `generalized_processing_vs_literal_nesterov` | primary |
| Generalized prospective dynamics \((M,\gamma,T)\) vs QHM Momentum DeltaNet | `generalized_processing_vs_qhm` | primary |

Everything else is unchanged from `e134ab1`:
- the six arms;
- sources, streams, seeds, selection ordering and update budget;
- the full held-out set as the primary set, with non-finite outcomes kept in
  the denominator;
- the recommendation rule (Generalized prospective dynamics \((M,\gamma,T)\) must beat every
  applicable control on the full-set primary analysis);
- the Nesterov applicability records;
- the secondary stable-subset diagnostic.

The remaining pairs are descriptive. The measured preflight still covers six
arms and still refuses rather than trims.

## 8. Tests

| Required | Where |
|---|---|
| elimination and direct FD realization | `test_eliminating_the_auxiliary_state_gives_the_same_transfer_function`, `test_direct_finite_difference_realization_is_the_executed_recurrence`, production `test_executed_generalized_is_the_direct_fd_realization` |
| Zucchet learned-mismatch boundary | `test_ordinary_finite_difference_boundary_is_gamma_plus_T_equal_h`, `test_generalized_recurrence_recovers_the_ordinary_fd_arm_closed_loop`, production `test_generalized_at_its_fd_boundary_is_the_executed_ordinary_fd_arm` |
| Zucchet matched-time-constants boundary | `test_the_ordinary_adaptive_state_boundary_is_M_gamma_zero` |
| Zucchet mapping and its limit | `test_adaptive_current_with_tau_a_equal_step_is_exactly_eq17`, `test_adaptive_current_with_tau_a_off_the_step_is_a_distinct_realization` |
| exact naming, display order and five primary contrasts | production `test_every_arm_carries_only_the_declared_scientific_names`, `test_display_order_and_primary_contrasts_are_preregistered_exactly` |
| discrete stability domain | unchanged: `docs/PROSPECTIVE_COEFFICIENT_DOMAIN.md`. The learned-mismatch boundary has `a = b = 0`, an FIR filter, stable for every `kappa` |
| unchanged outputs of every existing arm | `test_completed_arms_still_execute_their_documented_laws`, `test_two_tap_arm_is_the_completed_implementation`, `test_two_tap_ladder_evaluation_equals_the_completed_evaluation` |

Because no arm is added, the new-arm tests (hand-computed updates,
causality, reference unroll, Jury domain) do not apply. The executed
generalized recurrence already carries them: `test_every_form_is_causal`,
`test_completed_arms_still_execute_their_documented_laws`, and the domain
document.
