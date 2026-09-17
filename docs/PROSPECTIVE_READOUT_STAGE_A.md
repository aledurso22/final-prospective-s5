# Stage A: static derivation of the prospective readout

17 September 2026. Derivation and redundancy audit. No training, no
execution. Results here are algebraic; each identity below was **verified
exactly in rational arithmetic** and is re-verified numerically in float64 on
the cluster by `tests/test_prospective_readout_probe.py`.

## 1. The placement

The backbone stays **exactly native**: residuals keep using the native fast
weight `W`, never the readout state `X`:

    R_t = m_t (alpha_t W_(t-1) k_t - v_t) k_t^T          (native, unchanged)
    U_t = mu_t U_(t-1) + eta_t R_t                        (native, unchanged)
    W_t = alpha_t W_(t-1) - beta_t U_t                    (native, unchanged)
    X_t = a X_(t-1) - b X_(t-2) + c W_t - d W_(t-1)       (readout only)

Queries are answered from `X_t q_t` through the existing learned readout map.
The coefficients `a, b, c, d` are the same functions of `(M, gamma, T, h)` as
before, now over the corrected domain
(`docs/PROSPECTIVE_COEFFICIENT_DOMAIN.md`).

## 2. Exact recoveries (verified exactly, any gate schedule, any event
pattern, matched histories `X_(-1) = X_(-2) = 0`)

- **Native:** `M = 0, gamma = h, T = 0` gives `a = b = d = 0`, `c = 1`, so
  `X_t = W_t` **identically**, token by token.
- **Literal TSS:** `M = gamma = 0`, `T > 0` gives `a = 1 - h/T`, `b = 0`,
  `c = 1 + h/T`, `d = 1`, i.e.

      X_t = X_(t-1) + (h/T)(W_t - X_(t-1)) + (W_t - W_(t-1)),

  which is literal TSS Eq. (17) driven by `W`.

## 3. One-way coupling

`X` appears in no right-hand side of `R`, `U` or `W`: the three native lines
above are written entirely in `W`, `U` and the token inputs, and `X` is
computed after them. Therefore, **with a frozen backbone, `W` and `U` are
bitwise identical across every readout choice**, including the native
identity. The Stage B probe exploits exactly this: one backbone rollout
supplies the trajectory for every readout arm.

## 4. Decomposition, and what each coefficient does

With `X_t = W_t + E_t`:

    E_t = a E_(t-1) - b E_(t-2) + (c - 1) dW_t + b dW_(t-1)

(verified exactly; it requires the identity `a + c = 1 + b + d`, which holds
for all `(M, gamma, T)` including `gamma < 0`), and the **total weight on the
one-step difference** is

    ((c - 1) + b)/(1 - a + b) = (h - gamma)/h.

Since `W_t = alpha_t W_(t-1) - beta_t U_t`,

    dW_t = (alpha_t - 1) W_(t-1) - beta_t U_t                (verified exactly)

so the readout extrapolates a velocity built from **the write term
`-beta_t U_t` and the decay term `(alpha_t - 1) W_(t-1)` together**.

**Conclusion, stated explicitly:** `M` and `T` shape the *velocity
estimator* — how `dW` is smoothed and over what horizon — while **only
`gamma` sets the first-order lead** `(h - gamma)/h`. Two filters with the
same `gamma` extrapolate the same total amount and differ only in how they
average.

## 5. Redundancy audit

Carry counts are **extra state carried between tokens**, beyond what the
native step already holds. `W_(t-1)` is free: the step has it in hand while
computing `W_t`.

| Readout | Family position | `a, b` | Extra carry |
|---|---|---|---|
| Native identity `X = W` | `M = 0, gamma = h, T = 0` | `0, 0` | 0 |
| Two-tap `(1+kappa) W_t - kappa W_(t-1)` | `M = 0`, `gamma + T = h`, `kappa = T/h` (verified exactly) | `0, 0` | 0 |
| `T = h, gamma = M = 0` (`X = 2W_t - W_(t-1)`) | the two-tap point `kappa = 1` | `0, 0` | 0 |
| Literal TSS, `T != h` | `M = gamma = 0` | `a != 0, b = 0` | 1 matrix (`X_(t-1)`) |
| Generalized interior | `M > 0` | `a, b != 0` | 2 matrices (`X_(t-1), X_(t-2)`) |
| `U`-lookahead `X_t = W_t - lambda beta_t U_t` | **not** a member; see below | — | 0 (uses the carried `U`) |

**The one substantive difference.** Using `dW_t = (alpha_t - 1) W_(t-1) -
beta_t U_t`,

    two-tap(kappa = lambda) - U-lookahead(lambda) = lambda (alpha_t - 1) W_(t-1)
                                                                (verified exactly)

The two-tap readout **extrapolates the forgetting term** as part of the
velocity; the `U`-lookahead extrapolates **only the write term**. They
coincide exactly when `alpha_t = 1`.

## 6. Is the proposal redundant?

Partly, and this must be said plainly:

- The **FIR line** of the family (`M = 0`, `gamma + T = h`, any `kappa`,
  including `kappa > 1`) is computable from quantities the native step
  already has: `W_t`, the in-step `W_(t-1)`, and equivalently `U_t`. It adds
  **no carried state**. It is a cheap control, not a new mechanism.
- The **`U`-lookahead** is likewise computable from `(W, U)` with no extra
  carry, and differs from the two-tap only by the decay term.
- What is **not** computable from `(W, U)` at a single token is the
  **smoothed velocity**: the first-order average (literal TSS, `a != 0`, one
  extra matrix) and the second-order average (`M > 0`, two extra matrices).
  Those are the only members that require new state.

**A distinct hypothesis therefore survives**, and it is narrow: *does
averaging the velocity estimate over time (first or second order) predict the
near-future fast weight better than the raw one-step difference, and does it
do so by extrapolating the write term while not extrapolating the decay
term?* Stage B tests exactly that against ground truth, with the cheap
controls (native identity, two-tap, `U`-lookahead) in the same sweep. If the
smoothed members do not beat the cheap controls, there is nothing here to
train.

Nothing in this document claims the readout placement is novel beyond what
`docs/PROSPECTIVE_PRIOR_ART.md` supports, and nothing here is expected to
come out positive.
