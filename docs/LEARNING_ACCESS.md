# Learning access: inference-perfect, learning-wrong

Branch `learning-access-prospectivity`, branched from `closure-prospectivity`
`338d47e`. Independent of S5. The closure suite (B-F) is preserved unchanged and
serves as the regression suite.

**Claim under test.** A prospective realization can be exactly correct for
nominal inference while lacking physical information required for faithful
learning. Under restricted access no local causal corrector can recover the
missing parameter-induced direction; weak access carries an unavoidable noise
cost; and exact BPTT can therefore learn the wrong intended recurrent memory
despite zero training loss.

Reproduce: `python -m experiments.learning_access.<name>` ·
Tests: `python -m pytest tests/test_learning_access.py -q` (30 tests)

## Relation to the earlier closure work

`docs/TSS_CLOSURE.md` §E-B (`H_theta(p) = 1/(p+1) + theta/(p+3)`) is the
**simplest precursor** of this result: nominal response closure is not the same
thing as local parameter-tangent closure. It is retained unchanged. What the
three-state system adds is an actual prospective *architecture* and a
consequence for BPTT and learned memory, rather than a transfer-function
curiosity.

## 1. The exact three-state example

```
z_{k+1} = (rho+theta) z_k + x_k                     intended computation
b_{k+1} = beta b_k + v_k                            hidden physical mode
s_{k+1} = a s_k + (1-a)(z_{k+1} + theta b_{k+1})    physical realization

w_k = (s_k - a s_{k-1})/(1-a) = z_k + theta b_k     local prospective inverse
```

| gate | result |
|---|---|
| `max abs(w - z)` at theta=0, 200 randomized histories | **1.78e-15** |
| `max abs(w - (z + theta b))` at random theta | 2.84e-14 |
| tangent: autodiff vs analytic | 1.42e-14 |
| tangent: finite difference vs analytic | 2.01e-09 |
| `max abs((dw - dz) - b)` | **1.67e-15** |

Inference is perfect. The tangent is wrong, and **the gap is exactly `b`**.

## 2. The impossibility witness

Two histories with the same `x` and `z0` but different `b0, v`:

| | value |
|---|---|
| `max abs(w_A - w_B)` | **0.000e+00** (bit-identical) |
| `max abs(b_A - b_B)` | 5.308 |
| `max abs((-b_A) - (-b_B))` | 5.308 |

At theta=0 the b-subsystem is completely decoupled from `s`, so `b` leaves no
trace in `w`. The last `n` taps of `w` are identical for n = 1, 2, 4, 8, 16 -
every local corrector, at any capacity, receives the same input and must emit
the same correction, while the correct corrections differ by 5.308.

**More taps, local recurrence or network capacity cannot manufacture missing
information.** This numerical witness illustrates the analytical theorem; it
does not prove it.

## 3. Weak access and the noise law

`m_k = delta b_k + n_k`, correction `z_hat_k = w_k - (theta/delta) m_k`.

Noiseless, this reproduces `z` for the **full parameterized family** (max error
1.7e-13 over theta in [-0.3, 0.35] and delta in [0.05, 7]), so its BPTT
gradient equals the intended z-system gradient (agreement 5.3e-15).

| delta | measured Var | predicted `l'(z)^2 sigma^2/delta^2` | rel. err |
|---|---|---|---|
| 0.02 | 1.325e+02 | 1.315e+02 | 0.008 |
| 0.10 | 5.276e+00 | 5.259e+00 | 0.003 |
| 1.00 | 5.443e-02 | 5.259e-02 | 0.035 |
| 10.0 | 5.233e-04 | 5.259e-04 | 0.005 |

Fitted log-log slope **-1.9970** against the predicted **-2**.

Approximate gain `z_hat = w - theta k m`: the tradeoff
`eps_g + delta sqrt(V)/(|l'(z)| sigma) >= 1` is satisfied and **tight**, with
minimum **0.9994 +/- 0.0035** (Monte Carlo error on `sqrt(V)`, 40k samples).

| access | nominal inference | learning |
|---|---|---|
| none | exact | impossible |
| weak | exact | faithful but noisy |
| strong | exact | faithful and bounded |

## 4. The general port-access criterion

`w = z + sum_a theta_a U_a b + O(||theta||^2)`, `m = R b`. Faithful learning is
possible iff `ker R` is contained in `intersection_a ker U_a`, equivalently iff
`L_a R = U_a` is solvable; then `z_hat = w - sum_a theta_a L_a m`.

| case | holds | rank R | p_learning | evidence |
|---|---|---|---|---|
| three-state, delta=1 | yes | 1 | 1 | residual tangent 0.00e+00 |
| three-state, delta=0 | **no** | 0 | 1 | witness `b=[1]`, `Rb=0`, `Ub=1` |
| 3-D, port misses coord 3 | **no** | 2 | 1 | witness `b=[0,0,1]`, `Ub=2` |
| 3-D, full port | yes | 3 | 1 | 0.00e+00 |
| 3-D, minimal port | yes | 1 | 1 | 3.28e-16 |
| 2 params, rank-1 port | **no** | 1 | 2 | witness `b=[0,1,0]`, `U_1 b=1` |
| 2 params, rank-2 port | yes | 2 | 2 | 0.00e+00 |

`p_learning = rank([U_1; ...; U_d])` held in every case, and the minimal port
has exactly that rank. Treated as synthetic theorem validation, not a universal
neural port count.

## 5. Exact BPTT learns the wrong memory

Controlled case: `rho=0.9, beta=0.8, a=0.7, z0=1, b0=-4`, two steps, zero
drives, `y* = 0.95^2 = 0.9025`.

Initial gradients at theta=0, agreeing to 9 decimals across autodiff, finite
differences and closed form:

| model | gradient | direction |
|---|---|---|
| LOCAL `w_2 = z_2 + theta b_2` | **+0.070300000** | pushes theta DOWN |
| COLLECTIVE `z_hat_2 = z_2` | **-0.166500000** | pushes theta UP |

After training with the same autodiff stack, same lr, same iterations:

| quantity | LOCAL | COLLECTIVE |
|---|---|---|
| training loss | 0.000000 | 0.000000 |
| learned theta | **-0.106724** | **+0.050000** |
| learned pole | **0.793276** | **0.950000** |
| \|pole - 0.95\| | 0.156724 | 0.000000 |
| e-folding horizon (steps) | 4.32 | 19.50 |
| output, nuisance removed | 0.629287 | 0.902500 |
| **loss, nuisance removed** | **0.037323** | **0.000000** |

Both hit the analytic roots exactly: local `theta = (0.76 - sqrt(0.76^2 +
4*0.0925))/2 = -0.106724`, collective `theta = +0.05`.

**Both reach zero training loss. Only the local model does so by exploiting the
physical nuisance**, and its task-visible memory is 4.5x too short.

Randomized sweep, 200 nuisance histories and initial conditions, 0 diverged:
median `|pole-0.95|` local 0.0296 vs collective 0.00e+00; median clean loss
local 1.80e-03 vs collective 0.00e+00; **local worse in 100% of runs**.

A note on method: at `lr=0.35` the local objective (a quartic in theta)
diverged in 15/40 randomized configurations. The reported runs use `lr=0.03`,
where 0/40 diverge, applied identically to both models. Divergences are
excluded and counted explicitly, never averaged away as NaN.

## Gate status for a future S5 experiment

| gate | status |
|---|---|
| 1. local scalar prospectivity is nominally exact | **PASS** (1.78e-15) |
| 2. its learning tangent is analytically wrong | **PASS** (gap = b, 1.67e-15) |
| 3. extra local capacity cannot resolve the witness | **PASS** (identical taps) |
| 4. cross-channel access restores the tangent | **PASS** (exact for all theta) |
| 5. exact BPTT then learns the intended memory | **PASS** (pole 0.95 vs 0.793) |
| 6. the access criterion predicts the needed channel | **PASS** (7/7 cases) |

All six gates pass. S5 remains deliberately not started.

## Not done

Section 6 (reciprocal/passive continuous control) and section 8 (the older
`M s' + K s = K z` realization, with target motion `-M z'` as explicit forcing)
are next. Section 9 (generalized S5) stays gated.
