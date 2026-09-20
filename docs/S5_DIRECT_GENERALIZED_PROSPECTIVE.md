# Direct prospective S5 recurrences: derivation, and a stability finding

20 September 2026. Branch `s5-direct-generalized-prospective`, from
`6d55765`. Additive: Native S5, the Zucchet arm, the old two-compartment arm
and the post-scan WWJ readout arms are untouched.

**This document reports a negative result that must be read before any
implementation is adopted.** Every causal direct-recurrence form examined
here — matched WWJ and partially matched two-compartment, under either target
construction — has companion spectral radius ≥ 1 over the S5 mode disc, and
the exactly matched form destroys the driven S5 response outright. Both
findings are proved exactly, not inferred from a failed run.

## 1. Two instructions, and which one this follows

Two specifications arrived together and they contradict each other on the
central equation:

* **(A)** the inertial two-compartment equation
  $M\ddot s+\Gamma\dot s+s=f+T\dot f$, $\Gamma=\gamma+T$ — *"deliberately
  contains no $M\ddot f$. Do not add one."*
* **(B)** the matched WWJ equation
  $M\ddot s+\tau\dot s+s=f+\tau\dot f+M\ddot f$ — *"The $M\ddot f$ term is
  mandatory"*, and *"This supersedes my previous instruction"*, describing
  (A) as the pre-existing partially matched two-compartment model.

(B) is later and explicitly supersedes, so **(B) is followed**: the matched
equation is implemented as the subject, with the professor-consistent target
and the required exact-matching diagnostic. (A)'s equation is implemented
**alongside** it, under its own name, because (A) asks for a fast parallel
scan of it and because it is the natural control for (B). Nothing is renamed:
`generalized_prospective_s5` in the production tree is untouched.

## 2. The three equations, kept apart

| name | equation | transfer function $S/F$ |
|---|---|---|
| ordinary (professor / Zucchet) | $(1+\tau D)(s-f)=0$ | $1$ |
| **matched WWJ** | $(1+\tau D+MD^2)(s-f)=0$, i.e. $M\ddot s+\tau\dot s+s=f+\tau\dot f+M\ddot f$ | $1$ |
| **partially matched** (two-compartment) | $M\ddot s+\Gamma\dot s+s=f+T\dot f$, $\Gamma=\gamma+T$ | $\dfrac{1+Tp}{1+\Gamma p+Mp^2}$ |

The matched form's numerator and denominator are the *same* polynomial, so
$S/F=1$ identically. The partially matched form's first-order numerator
cannot cancel two second-order poles: **prospective correction plus retained
inertial memory**. That difference is the whole point, and it is why the two
are never given the same name.

## 3. Causal discretization, rederived

State forward, target backward, $h=1$ **token** (never S5's learned
$\Delta$):

$$\dot s_t\approx\frac{s_{t+1}-s_t}{h},\quad
\ddot s_t\approx\frac{s_{t+1}-2s_t+s_{t-1}}{h^2},\quad
\dot f_t\approx\frac{f_t-f_{t-1}}{h},\quad
\ddot f_t\approx\frac{f_t-2f_{t-1}+f_{t-2}}{h^2}.$$

With $q=M+h\tau$ (matched) the identity gives

$$s_{t+1}=as_t-bs_{t-1}+c_0f_t+c_1f_{t-1}+c_2f_{t-2},$$
$$a=\frac{2M+h\tau-h^2}{q},\ b=\frac{M}{q},\
c_0=\frac{M+h\tau+h^2}{q},\ c_1=-\frac{2M+h\tau}{q},\ c_2=\frac{M}{q},$$

and with $q=M+h\Gamma$ (partially matched)

$$s_{t+1}=as_t-bs_{t-1}+c_0f_t+c_1f_{t-1},\qquad
c_0=\frac{h(h+T)}{q},\quad c_1=-\frac{hT}{q}.$$

Both are verified on a basis of the free variables in exact rational
arithmetic. **This is a consistent causal approximation, not the exact
finite-step identity $P(D_h)(s-f)=0$**, because the state and target stencils
differ — stated here, in the module docstring, in the names and in the result
metadata.

At $M=0$, $\Gamma=T=\tau$ **both** reduce exactly to the professor update

$$s_{t+1}=\Big(1-\frac h\tau\Big)s_t+\Big(1+\frac h\tau\Big)f_t-f_{t-1},$$

and with $f=\bar As+\bar Bx$,
$A_0=aI+c_0\bar A$, $A_1=-bI+c_1\bar A$, $A_2=c_2\bar A$, $C_i=c_i\bar B$.

**Discrepancy to flag.** The repository's existing `zucchet_coefficients`
returns $a_1=(1-k)+(1+k)\bar A$ but $a_2=(k-1)-k\bar A$ and $c_2=-k\bar B$,
whereas the boundary above gives $a_2=-\bar A$ and $c_2=-\bar B$. They agree
only at $k=1$. So the claim "the $M=0$ boundary *is* the existing Zucchet
recurrence" does **not** hold against the code as written; the existing arm
is a different discretization. It is left untouched, and the difference is
recorded here rather than silently reconciled.

## 4. The exact-matching consequence, proved

With the common stencil $D_h$, the matched operator on the residual
$r=s-f$ is

$$(1+k+m)r_t-(k+2m)r_{t-1}+mr_{t-2}=0,\qquad k=\tau/h,\ m=M/h^2 .$$

With zero residual history this forces $r_t=0$ **for every $t$** — exact
matching means $s=f$. With $f=\bar As+\bar Bx$ that is the algebraic relation

$$(I-\bar A)s_t=\bar Bx_t\quad\Longrightarrow\quad
s_t=(I-\bar A)^{-1}\bar Bx_t,$$

i.e. a **memoryless** map: the impulse response is a single nonzero token,
against Native S5's $\bar A^t\bar B$ decay. The second-order residual does
have two transient modes, but they are *residual* transients and they do not
carry input-driven memory. Proved exactly in
`tests/test_direct_prospective_algebra.py`, reproduced numerically in
`tests/test_direct_prospective_scan.py`. This is a required scientific
diagnostic, and it is the reason the exactly matched model cannot simply be
adopted as an improved S5 recurrence.

## 5. Stability: a structural obstruction, not a tuning problem

**Fact 1 (exact).** At a mode $\bar A=1$, the characteristic polynomial of
**both** causal recurrences vanishes at $z=1$, for every $M$, $\Gamma$, $T$:
$1-A_0-A_1-A_2=0$ identically. No parameter choice moves that root. The
recurrence is marginal at best on a unit mode.

**Fact 2 (computed, `docs/analysis/`).** Maximum companion radius over a
disc of S5 modes ($|\bar A|\in\{0.5\ldots0.9999\}$, 96–128 phases),
$\varepsilon=1/4$:

| $\tau$ | mixed-stencil, native-matched target | mixed-stencil, professor target | two-compartment (no $M\ddot f$) |
|---|---|---|---|
| 0.01 | 1.7320 | 199.49 | 199.49 |
| 0.05 | 1.7321 | 39.52 | 39.50 |
| 0.25 | 1.7361 | 7.60 | 7.53 |
| 1.0 | 1.9673 | 1.9673 | 1.7319 |
| 10 | 19.52 | 1.0752 | 1.0513 |
| 100 | 199.07 | 1.0053 | 1.0050 |
| 1000 | 1998.9 | 1.0000 | 1.0005 |

Three conclusions:

1. **The native-matched target did not cause the earlier instability.** It
   was the *more* stable of the two constructions, and it still never drops
   below ≈ 1.73. The cluster's 1.705–1.877 is reproduced here (1.732–1.967)
   from the same formulas, so the mechanism is understood.
2. **The professor-consistent target is worse**, not better, for small
   $\tau$: the radius grows like $2h/\tau$ (199 at $\tau=0.01$), because
   $A_0=a+c_0\bar A$ contains $(h/\tau)(\bar A-1)$, which the native-matched
   $F_\tau\to I$ suppresses and the professor target does not.
3. **No form reaches the unit disc.** The radius approaches 1 only from
   above, as $\tau\to\infty$ — where the prospective term vanishes. The
   partially matched (no $M\ddot f$) equation behaves the same way, so
   dropping $M\ddot f$ does not rescue stability either.

The declared bound is tied to the sequence length rather than guessed:
$\rho\le1+10^{-5}$, which permits at most $(1+10^{-5})^{16000}=1.17$ of
growth end to end. **It is not relaxed to obtain an admissible cell.**

## 6. What is implemented

* `s5/direct_prospective.py` — both equations' coefficients, both target
  constructions, the exact common-stencil residual operator and its
  collapsed state, a generic **constant-operator doubling scan** (order 2 or
  3, $O(\log L)$ depth, no token loop, no sequential `lax.scan`, no
  $L\times P\times n\times n$ tensor, only the $(L,P,n)$ lifted state
  forward), and diagnostics.
* Backward memory is documented honestly: `remat="level"` retains each level
  **boundary**, so residuals are $O(LP\log L)$; `remat="whole"` retains only
  the inputs, $O(LPn)$, at roughly twice the forward flops; measured peak
  device memory is the authority.
* `tests/direct_prospective_reference.py` — frozen sequential oracles, for
  tests only.
* `experiments/s5_direct_prospective/stability_grid.py` — the cluster grid
  on **real production-initialized modes**: radius estimate cross-checked
  against exact CPU roots, nonfinite coefficients, the worst offending mode,
  float32/float64 production-length rollouts and gradient finiteness for
  admissible cells, and impulse responses for Native, the repository's
  Zucchet arm, the exact common-stencil collapse and the causal
  mixed-stencil recurrence.

**No training arm and no benchmark were built.** Building a production arm
whose own stability gate is predicted to fail would be premature; the grid
decides first. The previous generalized run's 2.98 steps/minute is the
execution failure being fixed, and the doubling scan is the fix — but
throughput is only worth measuring for a recurrence that is stable enough to
train.

## 7. What remains for the cluster

1. `stability_grid.py` on real HiPPO modes: does any declared cell satisfy
   $\rho\le1+10^{-5}$? The analysis predicts **no**, and predicts the worst
   modes lie near $\bar A\approx-1$ and $\bar A\to1$.
2. `tests/test_direct_prospective_scan.py`: scan-versus-oracle, gradient
   agreement, orientation, alignment, determinism, the radius estimate
   against exact roots, and the numerical collapse of the exactly matched
   model.
3. If, as predicted, no cell is admissible, the recorded outcome is
   `NO_ADMISSIBLE_INITIALIZATION` with the analytic reason above — **not** a
   relaxed threshold, and **not** a silent fall back to the old
   two-compartment arm.

Nothing here launches anything.
