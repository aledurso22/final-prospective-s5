# Prior art: what the implemented mechanism already is

17 September 2026. Documentation only. Nothing here changes an executed
equation, and no completed study's verdict is affected.

**The mechanism is not novel as an equation.** Every rule this repository has
executed is a known optimizer or a known continuous law, in a different
placement. What remains open is the **placement** (inside an associative
fast-weight memory, and — separately — on the read path) and the
**substrate**, not the algebra.

## How the identifications below were checked

Each algebraic identification was reproduced here in **exact rational
arithmetic** (Python `Fraction`), which is stronger than the float64
reproduction the brief asked for: an exact zero rather than a small residual.
The same identities are also reproduced numerically in float64 against the
production code by `tests/test_prospective_readout_probe.py`, which runs on
the cluster. Statements that are **citations to external papers** cannot be verified from
this environment. They were **verified externally by the coordinator against
the arXiv versions on 17 September 2026** — Titans' update (Eq. 1 and 13),
MDN's Appendix C future-work sentence naming Nesterov momentum, Adam and Muon,
An et al.'s PID rule, Adan, and Shi et al.'s gradient-correction term — and
are marked accordingly below. The non-exhaustive-search caveat stands
unchanged.

Notation: the operator's residual `R_t = m_t (alpha_t W_prev k_t - v_t) k_t^T`
is the gradient of `1/2 ||W k - v||^2` with respect to `W` at the masked
write, so "gradient" below means `R`.

## 1. The two-tap operator is momentum plus a gradient difference

Executed rule (`ordinary.py`), fixed gates:

    U_t = mu U_(t-1) + eta [(1 + kappa) g_t - kappa g_(t-1)],
    W_t = alpha W_(t-1) - beta U_t.

**At `kappa = mu` and `alpha = 1` this is exactly Nesterov's accelerated
gradient in gradient-correction form**

    x_(k+1) = x_k + mu (x_k - x_(k-1)) - s g_k - mu s (g_k - g_(k-1)),

with `s = beta eta`. **Verified exactly** (residual 0 in rational
arithmetic); off the line `kappa = mu` the two trajectories differ, as they
must. The condition `alpha = 1` matters: with `alpha < 1` the rule is NAG
with an extra multiplicative decay on the iterate, which NAG does not have.

**At general `kappa`** the accumulator splits exactly as

    U_t = I_t + kappa D_t,
    I_t = mu I_(t-1) + eta g_t,      D_t = mu D_(t-1) + eta (g_t - g_(t-1)),

i.e. momentum plus `K_d` times a **moving average (pole `mu`) of the gradient
difference**. **Verified exactly.** That is the PID optimizer's structure
(An et al., CVPR 2018): the integral/momentum term plus `K_d (g_c - g_(c-1))`
with an average on the derivative term. *The attribution to that paper, and the statement that Adan (Xie et al.,
2022) is the adaptive-gradient version, were verified externally against the
arXiv versions (17 September 2026); they are not re-checkable here.*

## 2. The residual-processing family is PID with an averaged D term

The executed processing filter `y_t = a y_(t-1) - b y_(t-2) + c R_t - d R_(t-1)`
decomposes **exactly** as `y_t = R_t + E_t` with

    E_t = a E_(t-1) - b E_(t-2) + (c - 1) dR_t + b dR_(t-1),

which requires the identity `a + c = 1 + b + d`; that identity holds for every
`(M, gamma, T)` (**verified exactly**, including `gamma < 0`). The total
weight the filter puts on the gradient difference is

    ((c - 1) + b) / (1 - a + b) = (h - gamma) / h     (verified exactly).

So the whole implemented family is **momentum plus `K_d` times a smoothed
gradient difference**, with `K_d = (h - gamma)/h`. Under the old passive
domain `gamma >= 0` this forces `K_d <= 1`; `gamma < 0` is exactly
`K_d > 1`. The mass `M` supplies a **second-order smoother** of the
difference in place of an EMA: with `M = 0` the `E` recursion is first order,
with `M > 0` it is second order.

## 3. The continuous law is inertial dynamics with Hessian-driven damping

For `M s'' + gamma s' + (1 + T D)(s - f) = 0` with the residual read as a
gradient `r = grad E(W)`, the term `T dr/dt = T grad^2 E(W) Wdot` is a
Hessian-driven damping term (chain rule; trivially verified). This is the
inertial-dynamics-with-Hessian-damping family (Alvarez et al. 2002; Attouch
et al., already cited in this repository) and, specifically, the
high-resolution ODE of NAG-SC (Shi, Du, Jordan, Su), where `sqrt(s)
grad^2 f(X) Xdot` is called the gradient correction. *The identification of Shi et al.'s gradient-correction term was verified
externally against the arXiv version (17 September 2026); the papers are not
available here.*

## 4. Delta-rule boundaries (one correction to the brief)

**In the PM parameterization** (`dynamics.py`), `kappa = mu/(1 - mu)` makes the
transfer numerator `(1 - mu z^-1)/(1 - mu)`, which cancels the momentum pole
and leaves a **first-order delta rule with step `beta eta/(1 - mu)`** — QHM at
`nu = 0`. **Verified exactly**, both as a transfer function and by running the
executed recurrence. This is the passive-circuit boundary of that
parameterization.

**The brief, and the handoff's s6.3 framing, call this "the same cancellation
as `M = gamma T`, in different coordinates".** That is true of the
**continuous** law and **not** of the implemented discrete coefficient form,
so the s6.3 framing is corrected here as well, not only the brief:

- continuous `H(s) = (1 + T s)/(M s^2 + (gamma + T) s + 1)`: the denominator
  at the numerator root `s = -1/T` equals `(M - gamma T)/T^2`, so the pole
  cancels exactly at `M = gamma T` (**verified exactly**); the completed
  meta-delta study records the same line as `rho = M/(gamma T) = 1`, its exact
  delta boundary;
- the executed TSS-compatible discretisation `z^2 - a z + b` with numerator
  `c z - d`: at `M = gamma T` the denominator at the numerator root
  `z = d/c = T/(h + T)` is **not** zero (for example `3/176` at
  `gamma = 2, T = 3, h = 1`, a value the coordinator independently
  reproduced). **The claim does not hold there**, and this document marks it
  as such. The two statements are therefore independent: `kappa = mu/(1-mu)`
  cancels the momentum pole exactly in the discrete PM recurrence, while
  `M = gamma T` is a continuous-only statement, where the denominator factors
  so that the `(1 + T p)` zero meets a pole. In that discretisation the only first-order
  point of the processing family is the native point `M = 0, gamma = h,
  T = 0`.

## 5. The backbone

*Verified externally against the arXiv versions (17 September 2026), not
re-checkable here:* the native rule (N) is the linear-memory case of Titans
(Eq. 1 and 13) (momentum plus forgetting on the associative
loss), and MDN is its chunkwise-parallel form. MDN's Appendix C names
Nesterov momentum, Adam and Muon scaling as future work and does not
implement them; no published delta-rule or linear-attention model uses a
gradient-correction inner rule, **as far as a non-exhaustive search found**.
That caveat is part of the claim: the search was not exhaustive, and this
document does not assert priority.

## 6. What this means for the project

- **The equation is prior art.** Two-tap = PID/Nesterov-type gradient
  correction; the filtered family = PID with an averaged derivative term;
  the continuous law = inertial dynamics with Hessian-driven damping.
- **What is not settled by prior art** is the *placement*: using such a rule
  as the inner update of an associative fast-weight memory, and (the Stage A
  and B question) on the **read path** of that memory. Whether either
  placement helps is an empirical question that this repository's completed
  studies have so far answered negatively or left infeasible.
- Nothing here should be read as a novelty claim, and no result in this
  repository is described as expected to be positive.
