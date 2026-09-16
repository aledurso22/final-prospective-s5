# Independent audit: prospective correction of Momentum DeltaNet

16 September 2026. Brief audited:
`PROSPECTIVE_MOMENTUM_NEXT_STEP_2026_09_16.md`, equations (2)–(12).

**Method.** Every result below was re-derived by hand from the stated
definitions. Nothing was copied from the brief's intermediate lines. No
numerical computation was run locally. The cluster checks in
`tests/test_prospective_momentum.py` test the same identities with separate
formulas; they have not been executed yet.

**Verdict.** Equations (3)→(6), (7), (8), (9), (10) and (11)/(12) are correct
within their stated scope. The audit adds these precisions, which the brief
does not contradict:

- the zero-initial-state requirement in (7)/(8);
- the exact pole–zero cancellations in (8);
- the `nu ≠ 0` condition on the QHM state map;
- why three Jury expressions are sufficient;
- the `q = 0` and `m = 0` cases.

No discrepancy requires a change of equation.

Notation: `q = beta eta`, `c = 1 - kappa (1-mu)/mu`, `d = (1-mu)/mu`.

## 1. From the canonical step (3) to the update (6)

Map (4): `M = h/beta`, `gamma = (1-mu)/(beta mu)`, `s = eta/(mu h)`,
`kappa = T/h`, `Q = -P`. For `beta > 0` and `0 < mu < 1`, the coefficients
M, gamma and s are strictly positive, and `T = kappa h ≥ 0` for `kappa ≥ 0`.

Substitute `V_t = (P_t - T s R_t)/M` into the momentum line of (3):

    P_t (1 + h gamma / M) = P_(t-1) + (h gamma T s / M) R_t - h s R_t.

The coefficients reduce as follows:

- `h gamma / M = beta gamma = (1-mu)/mu`, so `1 + h gamma/M = 1/mu`.
- `mu h s = eta`.
- `mu (h gamma T s / M) = mu · ((1-mu)/mu) · kappa h · eta/(mu h) = kappa eta d`.

Hence

    P_t = mu P_(t-1) - eta R_t + kappa eta d R_t,
    Q_t = mu Q_(t-1) + eta R_t - kappa eta d R_t = Qnative_t - kappa eta d R_t.   ✓

For the position line:

    W_t = Wbar_t + (h/M)(P_t - T s R_t) = Wbar_t - beta Q_t - beta T s R_t.

Here `beta T s = kappa beta eta / mu`. Substituting `Q_t`:

    W_t = Wbar_t - beta Qnative_t + beta kappa eta d R_t - kappa beta eta R_t / mu
        = Wbar_t - beta Qnative_t - kappa beta eta R_t [1/mu - (1-mu)/mu]
        = Wbar_t - beta Qnative_t - kappa beta eta R_t.                    ✓

Scope of the scheme:

- R is explicit, evaluated at `Wbar = alpha W_(t-1)`, after the native
  forgetting.
- Friction is implicit.
- The position update uses the new velocity.
- Forgetting multiplies W but not P (or Q). This is the native convention
  and is kept.

This is a declared first-order discretization, as the brief states. It is
not an exponential integrator. With gates that vary by token, it is a
piecewise-frozen modelling choice, not the ODE with time-varying
coefficients.

**At kappa = 0** the update is the native recurrence. In floating point the
implementation evaluates every native operation first, in native order,
then subtracts `kappa · (...)`.

- At kappa = 0 the subtracted value is an exact (signed) zero, so values are
  unchanged. Only the sign of a zero entry can differ.
- `R = m (Wbar k - v) k^T` with `m ∈ {0, 1}`, so `eta (m E)` and the native
  `(eta m) E` are equal.
- The kappa gradient is not disconnected: no branch is taken on kappa.

## 2. Identity (7) and transfer (8)

Fixed gates, prescribed R. From (6):

    Q_t = mu Q_(t-1) + eta c R_t
    W_t = alpha W_(t-1) - beta mu Q_(t-1) - q (1+kappa) R_t.

From the second line at t and t−1:

    beta mu Q_(t-1) = alpha W_(t-1) - W_t - q(1+kappa) R_t
    beta mu Q_(t-2) = alpha W_(t-2) - W_(t-1) - q(1+kappa) R_(t-1).

Multiply the Q recursion at t−1 by `beta mu` and substitute:

    alpha W_(t-1) - W_t - q(1+kappa)R_t
      = mu alpha W_(t-2) - mu W_(t-1) - mu q (1+kappa) R_(t-1) + mu q c R_(t-1).

The `R_(t-1)` coefficient is `q[-mu - mu kappa + mu - kappa(1-mu)] = -q kappa`.
Hence

    W_t - (alpha+mu) W_(t-1) + alpha mu W_(t-2) = -q [R_t + kappa (R_t - R_(t-1))].   ✓

This holds for t ≥ 2 unconditionally. For t = 0, 1 it holds with
`W_(-1) = W_(-2) = 0`, `Q_(-1) = 0` and `R_(-1) = 0`: the zero-initial-state
condition. Both cases were checked by hand. The z-transform then gives (8)
exactly:

    W(z)/R(z) = -q [1 + kappa (1 - z^-1)] / [(1 - alpha z^-1)(1 - mu z^-1)].   ✓

Consequences:

- **DC gain.** For alpha, mu < 1 the DC gain `-q/((1-alpha)(1-mu))` does not
  depend on kappa. ✓
- **Pulse at alpha = 1.** Immediate displacement `-q(1+kappa) R_0`. The total
  follows from `W_0 = -q(1+kappa)R_0`, `Q_0 = eta c R_0` and a geometric idle
  tail:
  `W_inf = -q(1+kappa)R_0 - beta mu eta c R_0/(1-mu) = -q R_0/(1-mu)`, using
  `mu c = mu - kappa(1-mu)`. It is independent of kappa. ✓ At mu = 1/2 and
  kappa = 1/2 the immediate ratio is 3/2. ✓
- **Pole–zero precision.** The numerator zero is at `z = kappa/(1+kappa)`. It
  cancels the mu pole exactly when `kappa = mu/(1-mu)`, i.e. `c = 0`: Q then
  receives no residual, and the R→W response is first order. It cancels the
  alpha pole when `kappa = alpha/(1-alpha)`. "The poles do not change" is
  correct for the system poles. At these isolated values one mode becomes
  unexcited by R.
- **Closed loop.** In the associative loop R depends on W, and neither (8)
  nor untouched-key retention transfers. This matches the brief.

## 3. Passive sector (9)

`M ≤ gamma T` ⇔ `h/beta ≤ (1-mu) kappa h/(beta mu)` ⇔ `kappa ≥ mu/(1-mu)`.
This uses beta > 0 and 0 < mu < 1. ✓ Equivalently `c ≤ 0` (and `nu ≤ 0` in
§4). The native kappa = 0 lies outside this sector, because mu > 0. ✓

## 4. QHM equivalence (10)

This section assumes alpha = 1 and fixed gates.

**Transfer.** QHM with `g_t = mu g_(t-1) + (1-mu) R_t` gives

    (1 - z^-1) W = -a [(1-nu) + nu(1-mu)/(1 - mu z^-1)] R
                 = -a [1 - nu mu - (1-nu) mu z^-1] / (1 - mu z^-1) R.

Matching (8) at alpha = 1 requires `a(1 - nu mu) = q(1+kappa)` and
`a(1-nu) mu = q kappa`. With `a = q/(1-mu)` the second condition gives
`nu = 1 - kappa(1-mu)/mu = c`. The first then holds identically:
`1 - mu + kappa(1-mu) = (1+kappa)(1-mu)`. ✓

**State map (closed loop, including state-dependent R).** Expand QHM:

    W_t = W_(t-1) - a nu mu g_(t-1) - a(1 - nu mu) R_t.

Compare with `W_t = W_(t-1) - beta mu Q_(t-1) - q(1+kappa) R_t`. The R
coefficients agree, and the state terms agree if `beta Q = a nu g`, i.e.
`g = (1-mu) Q / (eta nu)`. Then `g_t = mu g_(t-1) + (1-mu) c R_t / nu`, which
is the QHM recursion because `c = nu`. ✓

The residual is evaluated at the same W in both forms, so the equivalence
holds for the state-dependent associative residual step by step. It is not
limited to prescribed R.

Scope added by this audit:

- **Matched initialization** means `g_(-1) = (1-mu) Q_(-1)/(eta nu)`. This
  requires `nu ≠ 0`, or `Q_(-1) = 0`. At `nu = 0` (the passive boundary) QHM
  is plain SGD with step `q/(1-mu) = q(1+kappa)`. The candidate agrees only
  if Q starts at zero, since otherwise the decaying Q still moves W.
- The usual QHM range `nu ∈ [0,1]` corresponds to `kappa ∈ [0, mu/(1-mu)]`.
  Larger kappa gives `nu < 0`: the same algebraic form outside the usual
  range.
- alpha < 1 and gates that vary by token are not covered. Token-varying
  rescaling of g is not the canonical carry. This matches the brief.
- No optimizer novelty is claimed. See the brief's citations.

## 5. Frozen-token transition and Jury conditions (11)/(12)

One fixed unit key k, v = 0 (homogeneous), m = 1. Row-wise component
`x = W k`:

- `R k = alpha x`.
- `W_t k = alpha x - beta(mu Q k + eta alpha x) - kappa beta eta alpha x = alpha(1 - q(1+kappa)) x - beta mu (Qk)`.
- `Q_t k = mu Qk + eta alpha x - kappa eta d alpha x = alpha eta c x + mu Qk`.

Hence `A = [[alpha(1-q(1+kappa)), -beta mu], [alpha eta c, mu]]`. ✓

Trace and determinant:

- `tr = alpha + mu - alpha q(1+kappa)`. ✓
- `det = alpha mu(1 - q(1+kappa)) + beta mu alpha eta c = alpha mu - alpha q kappa`,
  using `mu c = mu - kappa(1-mu)`. ✓

Jury expressions:

- `p(1) = 1 - tr + det = (1-alpha)(1-mu) + alpha q`. ✓
- `p(-1) = 1 + tr + det = (1+alpha)(1+mu) - alpha q(1+2kappa)`. ✓
- `1 - det = 1 - alpha mu + alpha q kappa`. ✓

**Sufficiency.** For a real monic quadratic, both roots lie strictly inside
the unit circle iff `p(1) > 0`, `p(-1) > 0` and `|det| < 1`. Since
`p(1) + p(-1) = 2(1 + det)`, the first two already imply `det > -1`. The
three listed expressions are therefore necessary and sufficient. ✓

For `0 < alpha, mu < 1`, `q > 0` and `kappa ≥ 0`:

- `p(1) > 0` and `1 - det > 0` automatically.
- `p(-1) > 0` ⇔ (12). ✓

**Native point strictly inside.** In the pinned gate family:

- `beta = sin^2(theta) sigmoid(b) ≤ 1` and `eta = tanh(e) + 1 < 2`, so
  `q < 2` in exact arithmetic.
- `alpha ≤ 1` and `mu ≥ exp(-2)`, so `alpha q < 2 alpha ≤ (1+alpha)(1+mu)`.

The numerator of (12) is therefore strictly positive, and kappa = 0 is
strictly admissible.

- **alpha = 1** (only possible by rounding): `p(1) = q`. This is still
  positive when q > 0.
- **q = 0** (beta = 0): the unsimplified `p(-1) = (1+alpha)(1+mu)` does not
  depend on kappa, so that token imposes no bound. The residual still enters
  Q through `alpha eta c`: beta = 0 removes only its immediate effect on W. ✓
- **m = 0 tokens and key-perpendicular components:** triangular, with
  eigenvalues alpha and mu. ✓
- **Neutral rounded cases:** `p(1) = 0` needs `(alpha = 1 or mu = 1)` and
  `q = 0`; `1 - det = 0` needs `alpha mu = 1` and `alpha q kappa = 0`. Both
  can arise only from rounded native gates. The implementation classifies
  the rounded executed entries exactly, in rational arithmetic, as stable,
  neutral, unstable or non-finite.

**Scope.** This is a frozen-token test for one fixed key and gate. It is
neither a common Lyapunov function nor a switching certificate. The
continuous certificate of the meta-delta audit does not transfer to this
discretization. ✓ (brief)

## 6. Projection margin

Projection: `kappa ← min(max(kappa, 0), (1 - delta) Kmax)`, with
`delta = PROJ_REL_MARGIN = 1e-3`. Kmax is the minimum of (12) over all write
settings, computed in float32 inside the compiled step from the updated gates.

**Slack at the cap.** With `kappa = (1-delta) B_i ≤ (1-delta) B_i` for the
minimizing setting, and `kappa ≤ B_i` elsewhere, every setting satisfies

    p(-1) = N_i - 2 alpha q kappa ≥ delta · N_i,   N_i = (1+alpha)(1+mu) - alpha q.

Using the executed float32 gates, `alpha ≤ 1`, `mu ≥ exp(-2) ≈ 0.1353` and
`q ≤ 2` (rounded η may equal 2):

    N_i ≥ min over alpha ∈ [0,1] of (1+alpha)(1.1353) - 2 alpha = 0.2706 (alpha = 1).

So the exact slack is at least `2.7e-4`.

**Error budget.** These must be covered by that slack:

1. **Kmax in float32.** The product and sum `(1+alpha)(1+mu) - alpha q` have
   relative error of a few eps32 because `N_i ≥ 0.27` (at most 4·4 eps/0.27 ≈
   7e-6 relative). A further division adds eps. Total relative error on Kmax
   is ≲ 1e-5, i.e. ≲ 1% of delta.
2. **Rounding of the executed transition.** Each entry is a sum of at most
   three rounded products. At the cap, `alpha q ≤ 2` and
   `alpha q kappa ≤ N/2 ≤ 2`, so `|a11| ≤ 5`, `|a12| ≤ 1` and `|a22| ≤ 1`.
   `a21 = alpha eta c` can be large only when q is small (kappa ~ 1/q); then
   `a12 = -beta mu` is proportionally small, and the product `a12 a21`
   (bounded by `alpha q kappa / mu`-type terms ≤ 15) carries only a few eps
   of relative rounding. The absolute perturbation of `p(-1)` from the
   rounded entries is below roughly `100 eps32 ≈ 1.2e-5`, more than 20×
   smaller than the exact slack.
3. **Gate recomputation inside the scan versus on the table.** The same
   one-hot features and projection vectors are used, with possibly different
   reduction order. The perturbation of a, b, m and e is ≤ 2 eps |x|. It
   propagates to alpha, mu, beta and eta with relative size ≲ 32 |x| eps32
   (A_log ≤ log 16). This is ≲ 4e-6 for |x| ≤ 1, still ≪ delta.

delta = 1e-3 therefore leaves a factor of more than 20 over the worst
estimate. It is not copied from the earlier continuous rho margin.

**Checks.** The rounded executed transitions are verified in exact rational
arithmetic on the float32 entries (float32 probe and every validation), and
the effective margin `1 - kappa/B_f64` is recorded. Zero is the exact native
fallback: `max(·, 0)`, and `Kmax` cannot be ≤ 0 in exact arithmetic (§5).

**Gain control.** `alpha q g < (1+alpha)(1+mu)` from `A_g` with
`tr = alpha + mu - alpha q g` and `det = alpha mu`:

- `p(1) = (1-alpha)(1-mu) + alpha q g`;
- `p(-1) = (1+alpha)(1+mu) - alpha q g`;
- `1 - det = 1 - alpha mu`.

The projection is `log g ≤ log Gmax + log1p(-delta)`. The slack
`delta (1+alpha)(1+mu) ≥ delta` is larger than the candidate's.

## 7. Items for the reviewer

- The write table includes the absent value on WRITE events: 32 × 9
  settings, a superset of the task's writes. This is conservative.
- Neutral rounded native cases are counted, not failed. Unstable or
  non-finite executed transitions fail validation. This applies to the
  native arm too, with a diagnosis, because exact arithmetic excludes them.
- No derivative of the gates is involved in the projection. The optimizer
  state is untouched.
