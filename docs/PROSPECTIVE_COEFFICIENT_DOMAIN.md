# Corrected coefficient domain: stability is sign-agnostic in gamma

17 September 2026. Derivation and specification. **No completed study is
re-run, reinterpreted or re-evaluated.** Every executed verdict stands,
including the infeasible and unavailable ones.

## 1. What was wrong

The implemented family executes

    y_t = a y_(t-1) - b y_(t-2) + c R_t - d R_(t-1),
    A = M + h(gamma + T),  a = [2M + h(gamma+T) - h^2]/A,  b = M/A,
    c = [h^2 + hT]/A,      d = hT/A.

Its admissible set carried `gamma >= 0`. **That is a passivity condition, not
a stability condition.** It excludes the best rule this project has measured:
the learned two-tap operator with `kappa > 1`.

## 2. Derivation

For the real quadratic `z^2 - a z + b`, the Jury conditions are
`p(1) > 0`, `p(-1) > 0`, `|b| < 1`. Substituting the coefficients gives, for
`A > 0`:

| Jury condition | Value | Equivalent to |
|---|---|---|
| `p(1) = 1 - a + b` | `h^2 / A` | `A > 0` |
| `p(-1) = 1 + a + b` | `[4M + 2h(gamma+T) - h^2] / A` | `4M + 2h(gamma+T) > h^2` |
| `1 - b` | `[A - M]/A = h(gamma+T)/A` | `gamma + T > 0` |

(The `|b| < 1` lower side, `b > -1`, follows from `p(1) + p(-1) = 2(1 + b) > 0`.)
So the corrected admissible set is

    A > 0,    gamma + T > 0,    4M + 2h(gamma + T) > h^2,

**verified exactly over a grid including negative `gamma`**. These conditions
are **sign-agnostic in `gamma`** once `A > 0`: only the combination
`gamma + T` and the mass appear.

This project keeps two further **declared** (not stability-derived) choices:
`M >= 0` (a mass) and `T >= 0` (a horizon). `gamma` is free.

## 3. What the corrected family contains

| Member | Coefficients | Note |
|---|---|---|
| Native Momentum | `M = 0, gamma = h, T = 0` | `a = b = d = 0`, `c = 1` |
| Literal TSS | `M = gamma = 0`, `T > h/2` | `a = 1 - h/T`, `b = 0`, `c = 1 + h/T`, `d = 1` |
| Two-tap operator, any `kappa` | `M = 0`, `gamma + T = h`, i.e. `gamma = h(1-kappa)`, `T = kappa h` | **`a = b = 0`: an FIR filter, unconditionally stable** (verified exactly for `kappa = 1/2, 1, 2, 5/2`); `kappa > 1` needs `gamma < 0` |
| Nesterov line | the two-tap line at `kappa = mu` | see `docs/PROSPECTIVE_PRIOR_ART.md` |
| Second-order interior | `M > 0` | the smoothed-difference region |

The learned operator (`kappa` about 1.9–2.2 in the completed runs) therefore
lies **inside** the corrected family and **outside** the old passive one.

## 4. Corrected repair and gate

**Executed acceptance gate: unchanged.** It already classifies the rounded
`a, b` that the compiled program executed, and that classification never
looked at the sign of `gamma`. The only change is the domain check attached
to it.

- `filtered.repair` and `filtered.filter_failure` (passive: `gamma >= 0`)
  are **left exactly as the completed studies executed them**.
- `filtered.repair_corrected` clamps `M` and `T` at zero, **never clamps
  `gamma`**, and raises `T` minimally for the declared numerical gaps, in
  this order:
  1. `M <- max(M, 0)`, `T <- max(T, 0)`;
  2. `T <- max(T, g_min - gamma)` for `gamma + T >= g_min`;
  3. `T <- max(T, (h^2 (1 + delta) - 4M)/(2h) - gamma)` for the filter gap.
  `A >= h g_min > 0` then follows whatever the sign of `gamma`.
- `filtered.filter_failure_corrected` requires finiteness, `M >= 0`,
  `T >= 0` and a strictly stable executed polynomial. `gamma < 0` is
  admissible.

Declared numerical gaps are unchanged: `g_min = 2^-10 h`, `delta = 1e-3`.
They remain a robustness policy, not a certificate; the certificate is the
executed-coefficient classification. As before, this certifies the isolated
filter only.

## 5. Status of the completed studies

- The containment, temporal-response and retention-aware studies **ran under
  the old passive domain**. Their verdicts are unchanged and are not
  re-evaluated.
- The `gamma`-boundary pressure recorded in those runs (the optimizer
  repeatedly proposing `gamma < 0`, with the repair clamping it to zero) now
  reads as **the optimizer being constrained out of an admissible region**.
  It is **not** evidence that `gamma < 0` would have helped: no run evaluated
  that region, and the one rule known to live there (the learned two-tap
  operator, `kappa > 1`) had the **lowest** retention of all arms in the
  containment run.
- Any future use of the corrected domain must say which domain it executed.
