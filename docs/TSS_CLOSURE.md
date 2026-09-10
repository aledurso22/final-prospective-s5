# Closure prospectivity: (L, g) -> P_g

Branch `closure-prospectivity`, branched from `main` `156e428`. Independent of
S5 and of `professor-state-pc`; nothing here imports S5.

The central move is to derive the prospective law for whatever residual
observable actually closes, instead of assuming the scalar one-pole law
`(I + tau D) g = 0`.

Reproduce: `python -m experiments.tss_closure.<name>` · Tests:
`python -m pytest tests/test_tss_closure.py -q` (14 tests)

## Model classes, kept distinct

```
scalar one-pole -> diagonal first-order -> collective first-order
                -> augmented hidden-state closure
```

`prospective/identification.py` fits all four. The augmented class is an
OVERPARAMETERIZED CONTROL: if selection ever prefers it on exactly first-order
data, the selection procedure is broken.

## B. Ideal TSS - the gate

Integrating the real TSS law `s' = J_r^-1 [f_t - r/tau]`:

| system | prescribed rate | measured | rel. error |
|---|---|---|---|
| affine | -2.8571 | -2.8608 | 1.3e-03 |
| nonlinear tanh | -2.8571 | -2.8710 | 4.9e-03 |
| affine 2-D | -2.8571 | -2.8569 | 7.6e-05 |

On stochastically driven ideal-TSS data, BIC selects **order 1**. Innovations
are white at every order, so extra modes buy nothing. **Gate passed.**

## B2. Collective first-order - closure order is not the same as pole count

Data obey `r' = -Gamma r` with `Gamma = [[1, 0.8], [0.8, 3]]` exactly. No
hidden state. True poles `-3.281, -0.719`.

| model class | k | fitted poles | held-out MSE | BIC | pole error |
|---|---|---|---|---|---|
| scalar | 1 | -1.273, -1.273 | 1.11e-02 | -336004 | 2.008 |
| diagonal | 2 | -2.470, -0.769 | 4.35e-03 | -345726 | 0.811 |
| **full** | 4 | **-3.283, -0.716** | 1.64e-04 | **-357704** | **0.004** |
| augmented | 8 | -474, -474 | 1.64e-04 | -357279 | 473.7 |

BIC selects `full`. The augmented model's held-out edge is **+0.14 %** -
immaterial - and its poles are spurious.

**A zero closure defect does not mean scalar one-pole dynamics. It means the
chosen observable space is closed.** A scalar or diagonal model cannot
represent a coupled first-order system, and no hidden state is required to fix
that - only a richer first-order coupling.

Note also that raw held-out rollout MSE alone is NOT a sufficient selection
criterion here: it ranked the augmented model first on a 0.14 % difference.
BIC and pole error both identified the correct class.

## C. The fundamental multimode test

`r' = -gamma r + z b`, `b' = -a b + z r`, with `gamma=1, a=0.4, z=0.5`
(`a*gamma = 0.4 > z^2 = 0.25`, stable). Generator eigenvalues
`-1.283, -0.117`.

From `r(0) = 0`, `b(0) = 1`:

| quantity | value |
|---|---|
| `r'(0)` predicted `z*b0` | 0.500000 |
| `r'(0)` numerical | 0.499301 (rel. 1.4e-03) |
| peak `|r|` reached | **0.3065**, starting from zero |

| closure | RMS residual-prediction error |
|---|---|
| C1 matched one-pole | 2.136e-01 |
| C2 best visible first-order | 2.136e-01 |
| C3 two-mode | 0 |

C1 and C2 both predict `r(t) == 0` forever, because they start from the
visible `r(0)=0` and have no hidden state. C2's best-fit visible pole is
`+0.026` - positive, i.e. meaningless, because the data are not first order.

## D. Active synchronization

Oracle control `u_c(t) = -z b0 exp(-a t)`:

| quantity | value |
|---|---|
| max abs r | 6.1e-05 (ZOH artifact) |
| max abs b | **1.0000** |
| max abs u_c | **0.5000** |
| effort V numerical | 0.156249048 |
| effort V analytic | 0.156249040 |
| relative error | **4.8e-08** |

The residual is not exactly zero only because of zero-order hold. Verified
first order in dt - `max|r|/dt` is constant at 0.0613 across dt from 0.008 to
0.001.

**Zero visible tracking error coexists with non-zero hidden mismatch and
non-zero control cost.**

## E-A. Least-Control learning signal

Objective is control effort, not squared tracking error.

| gradient dV/dz at z=0.5 | value |
|---|---|
| autodiff (full) | +0.624996190 |
| finite difference | +0.624996190 |
| analytic | +0.624996160 |
| **reduced one-pole** | **+0.000000000** |

Autodiff agrees with finite differences to 8e-12 and with the closed form to
3e-08, converging as O(dt^2). The reduced model has no hidden state, so from
`r(0)=0` it needs no control: its objective and gradient are identically zero.

**`r(t) = 0` throughout, and the Least-Control learning signal is still 0.625,
which the reduced closure reports as exactly zero.**

## E-B. Forward response vs parameter sensitivity

`H_theta(p) = 1/(p+1) + theta/(p+3)`. Nominally one mode at -1; the tangent is
a different mode at -3.

| model | forward RMSE (impulse) | forward RMSE (step, held out) | sensitivity RMSE (impulse) | sensitivity RMSE (step, held out) |
|---|---|---|---|---|
| reduced one-mode | 1.28e-14 | 5.92e-14 | **7.23e-02** | **2.40e-01** |
| two-mode closure | 1.04e-15 | 5.48e-15 | 4.55e-12 | 2.08e-11 |

Forward error ratio 12x; **sensitivity error ratio 1.6e10x**. Calibration used
the impulse response, evaluation the step response.

**A reduced model can match the nominal response to machine precision and
still fail its parameter tangent.**

> This experiment is the simplest **precursor** of the learning-access result.
> It shows nominal response closure is not local parameter-tangent closure. The
> three-state system in [LEARNING_ACCESS.md](LEARNING_ACCESS.md) strengthens it
> into an actual prospective architecture with a consequence for BPTT and
> learned memory.

A further nuance worth recording: at `theta = 0` the hidden mode has zero
amplitude and is **not identifiable at all** - AR(2) returns a spurious pole at
-69. It becomes exactly identifiable (`-3, -1`) as soon as `theta != 0`. The
hidden mode is invisible in the nominal response by construction, not by
accident of fitting.

## Status

Implemented and passing: B, B2, C, D, E-A, E-B, F. 14 unit tests covering
specification items 1-6 plus the collective first-order case.

**Not yet done:** G, the small recurrent realization experiment. When built it
must use the correct error equation, in which target motion is an explicit
forcing term:

```
M r' + K r = -M z'
```

and memory preservation must be judged by input-output transfer, task-visible
memory horizon and parameter sensitivities - not by `A_star`'s eigenvalues
alone.

H (generalized S5) is deliberately not started.
