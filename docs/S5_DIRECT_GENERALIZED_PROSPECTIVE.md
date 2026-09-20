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

## 2a. Naming: two independent axes

The first cluster run failed a test called
`..._matches_the_oracle(construction="professor", M>0)`, a name that reads
as "the Professor model" when it meant "the WWJ model on the professor
target". The two axes are now separate everywhere — code, tests and result
metadata:

| axis | values |
|---|---|
| **model** (which equation) | `professor_tss` ($M=0$, two-state), `wwj_generalized_tss` ($M>0$, three-state, with the mandatory $M\ddot f$) |
| **target construction** (how $f$ is built) | `professor_linear_target` ($f=\bar As+\bar Bx$), `native_matched_target` ($F_\tau,G_\tau$) |

`DP.model_of(mass)` returns the model from the mass, and the argument is
`target_construction=`, never `construction=`. The legacy repository arm
keeps its own label (`zucchet_repository_legacy`) until §3's discrepancy is
resolved by its author.

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

**Discrepancy, now identified exactly.** The repository's
`zucchet_coefficients` returns $a_1=(1-k)+(1+k)\bar A$, $a_2=(k-1)-k\bar A$,
$c_1=(1+k)\bar B$, $c_2=-k\bar B$. The derivation above, with $h/\tau=k$,
gives $A_0=(1-k)+(1+k)\bar A$, $A_1=-\bar A$, $C_0=(1+k)\bar B$,
$C_1=-\bar B$. The first taps agree exactly; the differences are

$$a_2-A_1=-(1-k)(1-\bar A),\qquad c_2-C_1=(1-k)\bar B,$$

which together add exactly $-(1-k)\,(s_{t-1}-f_{t-1})=-(1-k)\,r_{t-1}$ to the
update: **an extra residual-feedback term**, vanishing only at $k=1$.

So the cause is **not indexing** (both emit $s_{t+1}$ from $s_t,f_t,f_{t-1}$)
and **not the target construction** (both use $f=\bar As+\bar Bx$ in the
comparison). It is (i) the meaning of the stored parameter — the legacy $k$
matches the derivation only if `response` denotes the **rate** $h/\tau$, not
the timescale $\tau$ — and (ii) a term the stated discretization does not
contain. Whether (ii) is deliberate or a legacy error is for its author to
say; the arm is left untouched and the difference is proved in
`tests/test_direct_prospective_algebra.py`.

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

## 6a. The first cluster failure was the fixture, not the scan

`test_the_matched_scan_matches_the_oracle_at_many_lengths` failed at
`length=1000`, $M=\tau^2/4$, professor target. Reproducing that fixture's
arithmetic locally in IEEE double (`docs/analysis/`, pure Python — no JAX
needed for this question):

| mode | companion radius | first nonfinite, scan / oracle | common finite prefix | max **relative** error on the prefix |
|---|---|---|---|---|
| $0.9e^{2.0i}$ | 3.1107 | **626 / 626** | 626 tokens | 1.9e-14 |
| $-0.95$ | 3.5778 | **558 / 558** | 558 tokens | 2.3e-14 |
| $0.7e^{-1.2i}$ | 2.0583 | **983 / 983** | 983 tokens | 1.2e-14 |

The scan and the oracle agree to ~1e-14 relative over the entire finite
prefix and go nonfinite at the *same token*. The absolute error looks
enormous (~2e294) only because the values themselves are ~1e308. **The
recurrence and scan code were therefore not changed**; what changed is the
test design:

* scan correctness is now tested on **analytically certified stable**
  companion fixtures, built from roots inside the unit disc, at lengths
  through 1500, orders 2 and 3, real and complex, values *and* gradients,
  with a comparison that **fails on any nonfinite value** — `equal_nan` is
  never used;
* the divergent production-derived fixture is kept as an **unstable-model
  regression** asserting agreement over the complete common finite prefix,
  the same first nonfinite token and mode, and that the cell can never be
  classified as admissible;
* the real coefficient builders are searched for stable cells, and the scan
  is checked against the oracle on any that exist — the analysis predicts
  none do, and that is recorded rather than skipped.

## 6b. A prediction falsified, and what the stable cell actually is

**My prediction that no cell of the real coefficient builders would be
stable was wrong.** The corrected suite found one on the cluster:

$$\tau=1000,\quad \varepsilon=1/4,\quad
\text{target}=\texttt{professor\_linear\_target},\quad
\rho=0.9980477224938169<1 .$$

It stays in the test suite and is not replaced by an easier synthetic
fixture.

### The finite disagreement at that cell: diagnosed against exact arithmetic

Exact complex-rational ground truth (`docs/analysis/`, no JAX needed), the
scan and the oracle both in IEEE double:

| mode | L | max abs err | max rel err | token of max | value there | max ULP | **scan vs exact** | **oracle vs exact** |
|---|---|---|---|---|---|---|---|---|
| $0.9$ | 17 | 1.1e-12 | 3.6e-13 | 16 | 3.1 | 2 516 | 3.2e-13 | 7.1e-14 |
| $0.9$ | 257 | 3.9e-07 | 9.9e-07 | 216 | 20.1 | 4.7e9 | **9.9e-07** | **1.2e-09** |
| $0.9$ | 1000 | 1.1e-05 | 2.2e-04 | 815 | 4.8 | 1.9e12 | — | — |
| $0.5{+}0.25i$ | 257 | 4.1e-09 | 6.2e-09 | 244 | 3.6 | 1.8e9 | 6.2e-09 | 1.9e-11 |
| $-0.6{+}0.4i$ | 257 | 7.6e-11 | 8.0e-11 | 227 | 5.8 | 2.3e7 | 8.0e-11 | 1.5e-12 |

Finite masks identical everywhere; no nonfinite values at this cell.
**The oracle is right and the scan is the inaccurate one**, by two to three
orders. The forward recurrence residual confirms it: the oracle's is
**exactly zero** by construction, the scan's is 4.6e-8 at L=257 and 2.1e-6 at
L=1000. Gradient disagreement (derivative sequence, same recurrence with
source $s_{t-1}$) tracks the forward error: 4.4e-9 at L=257, 2.1e-5 at
L=1000.

Both algorithms are correct; the difference is conditioning. The companion at
this cell is strongly **non-normal** — $A_0\approx2.896$, $A_1\approx-2.792$,
$A_2\approx0.896$ with all roots near the unit circle — so
$\lVert H^{2^k}\rVert_F$ peaks at $7.5\times10^3$ even though $\rho<1$, and
the scan's repeated squaring cancels catastrophically. The naive
absolute-value condition number is useless here (it grows as $6.6^L$, giving
$10^{128}$ at L=257).

**The tolerance is now derived, not chosen:**
$\text{tol}=\text{safety}\cdot\epsilon\cdot\max_k\lVert H^{2^k}\rVert_F^2$,
floored at $64\epsilon$. Measured against exact ground truth the constant
never exceeded 300, so safety = 1000 leaves ~3× margin. It is computed from
the actual coefficients at test time, not hardcoded.

Error versus length **saturates** rather than diverging (ρ<1 forgets):
1.7e-8 at 257, 4.4e-7 at 1000, 3.5e-6 at 2000 and flat at 3.5e-6 through
16 000 — about 5.5 correct digits at production length, in float64.

### The decisive pre-training finding: float32 breaks the scan here

Emulated float32, same cell (`direct_prospective_float32_emulation.txt`):

| mode | L | **doubling scan** rel err | sequential recurrence rel err |
|---|---|---|---|
| $0.9$ | 257 | **7.2** | 5.7e-04 |
| $0.9$ | 1000 | **1.5e+05** | 2.1e-03 |
| $0.9$ | 4000 | **2.3e+21** | 2.6e-03 |
| $0.5{+}0.25i$ | 4000 | **4.6e+03** | 2.4e-04 |

**In float32 — the precision the production path uses — the constant-operator
doubling scan is unusable at the only stable cell that exists**, while the
sequential recurrence is fine. This is a property of squaring a non-normal
operator, not a bug, and no tolerance can repair it. The options are float64
for the scan (slower, more memory), a **chunked scan** (sequential within
chunks of size $c$, doubling across $L/c$ chunk summaries, which caps the
number of squarings at $\log_2(L/c)$ and is the natural next design), or the
sequential path that was already too slow. None of them is chosen here.

### Is $\tau=1000$ a real prospective effect?

Coefficients at this cell (with $\bar B=1$): $C_0=1.000004$,
$C_1=-1.996016$, $C_2=0.996016$, and for $\bar A=0.9$:
$A_0=2.896016$, $A_1=-2.792430$, $A_2=0.896414$.

| $\bar A$ | roots | timescales (tokens) | vs Native | vs Professor/TSS | centroid shift |
|---|---|---|---|---|---|
| 0.9 | 0.899966, 0.998025±0.000283i | 9.5, **505.7**, 505.7 | 1.9e-03 | 6.9e-03 | +0.01 |
| 0.5+0.25i | 0.5+0.25i, 0.99798, 0.99804 | 1.7, **494.5**, 508.3 | 1.2e-04 | 1.7e-03 | +0.00 |
| 0.99 | 0.989439, 0.998289±0.000919i | 94.2, **584.0**, 584.0 | **5.1e-02** | 5.4e-02 | **+4.56** |

The native root survives almost exactly (0.899966 vs 0.9), and the recurrence
**adds a pair of residual modes with ≈500–584 token timescales**. The
impulse response differs from Native by 0.01–5%, largest for the
longest-memory mode. The response centroid moves **later**, not earlier
(+4.56 tokens at $\bar A=0.99$): at this $\tau$ the effect is *retained
long-timescale inertial memory*, **not** a temporal advance. Whether that is
the intended prospective effect is a scientific decision, not a numerical
one — but it should not be described as "prospective compensation".

## 6c. The block/chunked scan: implemented, and measured honestly

The full-sequence doubling scan is **rejected for float32 production** (error
1.5e5 by L=1000, 2.3e21 by L=4000). It is kept only as a labelled diagnostic
and regression, and no states function selects it implicitly — every default
is `scan_kind="block"`.

**The algorithm**, orders two and three, in `s5/direct_prospective.py`:

1. the sequence is padded to a multiple of $C$ with **zero drive at the
   end**; the recurrence is causal, so no real token sees the padding and
   there is no wraparound, and the outputs are truncated back to $L$;
2. each chunk's affine summary $z_\text{end}=H^Cz_\text{start}+g_\text{chunk}$
   is computed **sequentially in token order with `lax.scan`, vmapped across
   chunks** — a JAX primitive, never a Python token loop;
3. $H^C$ is built from `order` zero-drive **basis runs of the ordinary
   recurrence**, never by repeated squaring;
4. chunk boundaries are propagated with a **sequential `lax.scan` over the
   $L/C$ summaries** — deliberately not a doubling scan over summaries,
   which would reintroduce the same problem;
5. each chunk is re-run from its own boundary, vmapped, giving every token;
6. `remat="chunk"` wraps the in-chunk scan; peak backward memory is reported
   by the chunk study.

Sequential depth falls from $L$ to $C+L/C$ — about 253 instead of 16 000 at
$C=126$.

**Measured in emulated float32 at the stable cell**
(`docs/analysis/direct_prospective_block.txt`), relative to float64
sequential:

| $\bar A$ | L | doubling | block C=2 | C=8 | C=16 | C=64 | C=256 | **sequential** |
|---|---|---|---|---|---|---|---|---|
| 0.9 | 1000 | 3.1e+05 | 4.8e-02 | 2.4e-02 | 1.6e-01 | 4.5e+00 | 1.1e+00 | **1.1e-03** |
| 0.9 | 4000 | 3.2e+21 | — | — | 2.6e-01 | 8.0e+02 | 5.6e+00 | **2.1e-03** |
| 0.5+0.25i | 1000 | 3.3e+00 | 1.1e-03 | 3.6e-03 | 2.4e-03 | 1.8e-02 | 8.8e-02 | **8.5e-05** |

So the block scan is an improvement of up to **eighteen orders of magnitude**
over doubling — but at this cell it is still 20–400× worse than the ordinary
sequential recurrence, and its absolute error (1.6e-1 at C=16, L=1000) is not
usable. The cause is the same one, one step removed: $|H^C|$ reaches 1.7e2 at
$C=16$ and 2.9e3 at $C=256$ while the states are $O(1)$
(`direct_prospective_chunk_transition_norms.txt`), so every boundary
application injects about $\epsilon_{32}|H^C|$, and those errors are then
amplified by later boundary applications.

**Provisional conclusion, to be confirmed on the GPU**: chunking fixes the
catastrophe but not the marginality. At this cell even the plain sequential
float32 recurrence carries ~1e-3 relative error — the recurrence is
ill-conditioned in production precision, independently of the scan. The chunk
study is built to measure exactly this on real modes and applies its decision
rule in code (float32 forward *and* gradient error within 10× the sequential
float32 error, finite, deterministic); if nothing qualifies it reports
`BLOCK_SCAN_NOT_USABLE_IN_FLOAT32`.

## 6d. Three gates, never conflated

| gate | question | where it is decided |
|---|---|---|
| **1 mathematical stability** | companion radius over the production horizon | `stability_grid.py`, exact CPU roots, bound $1+10^{-5}$ |
| **2 numerical computability** | float32 forward *and* backward accuracy of the scan that would actually run | `chunk_study.py` |
| **3 scientific usefulness** | a measurable, correctly labelled temporal effect | `stability_grid.py`, `scientific_usefulness` + `label_effect` |

Gate 3 reports the response difference from Native and from the corrected
Professor/TSS, the impulse-response **centroid shift**, the **peak-response
shift**, the **early-response error after an innovation** and the **retained
long-delay response**, and labels the result. A cell that only moves the
response later is labelled
`ADDITIONAL_LAG_NOT_PROSPECTIVE_COMPENSATION` — which is what $\tau=1000$
looks like so far (+4.56 tokens at $\bar A=0.99$).

**Eligibility for training** requires all three gates plus throughput and
memory. The grid now records `eligible_for_training: false` with the reason,
because no cell can be eligible on gate 1 alone.

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
