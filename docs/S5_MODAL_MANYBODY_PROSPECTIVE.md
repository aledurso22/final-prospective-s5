# Heterogeneous modal prospective S5: a stable cascade

20 September 2026. Branch `s5-modal-manybody-prospective`, from the clean
production commit `ef004cda025b4e098041cfe5970c217fa0015bba`. Native S5 is
byte-identical and every file here is new.

## 0. What this is, stated accurately

A **heterogeneous many-body, WWJ-inspired modal action/cascade**. The local
first-order factors are **variationally motivated** by the WWJ operator; the
particular **coupling topology** — one independent cascade per native S5
mode — and the **learned gates** that select each mode's regime are an
**explicit model construction**, not a consequence of the variational
principle. This is not the matched residual law and does not claim to be
derived from one.

## 1. The negative result this replaces

The direct matched route is stopped and preserved on branches
`s5-direct-generalized-prospective` (tip `a08c858`) and
`s5-three-arm-discrete-prospective`. What it established, and why no more
time is being spent on it:

* the **exactly matched** law `P(D)(s−f)=0` has `S/F = 1`, and with
  `f = Ā s + B̄ x` it collapses the driven response to `(I−Ā)s = B̄x` — a
  **memoryless map**, proved exactly;
* the **mixed-stencil** realization escapes that cancellation only by using
  inconsistent stencils, so its poles are discretization artefacts. They
  left the unit disc on the real S5 modes: 1.705–1.877 with the
  native-matched target, growing like `2h/τ` with the professor-consistent
  one, **1.4416 at τ = 2** with float32 divergence at token 242 — the
  *sequential* path too, so no scan could repair it;
* the surviving cells were **nearly marginal** (ρ = 0.998 at τ = 1000) and
  numerically fragile: the full-sequence doubling scan lost everything in
  float32 (measured 4.18e7 relative on the cluster), and even a chunked scan
  injects `ε·|H^C|` per boundary;
* three separate certification errors came from measuring a **mode subset**
  rather than the production inventory.

The lesson carried into this design: **do not let the discretization choose
the poles.** Here it cannot.

## 2. The stage, and why it is stable by construction

With the backward derivative `D_h y_t = (y_t − y_{t−1})/h`, the first-order
mechanical stage `(1 + d D_h) u = (1 + n D_h) v` is, multiplying by `h` and
solving for `u_t`,

$$u_t=\frac{d}{h+d}\,u_{t-1}+\frac{h+n}{h+d}\,v_t-\frac{n}{h+d}\,v_{t-1}.$$

Its **only** recurrent pole is

$$p=\frac{d}{h+d}\in(0,1)\quad\text{for every }d>0,\;h>0,$$

with no parameter able to move it out. The numerator parameter `n` moves a
**zero**, at `n/(h+n)`, and never a pole — so gates and numerator learning
cannot destabilize anything. At `n = d` the stage is the **exact identity**
for zero-consistent history.

## 3. The three regimes, selected by gates

Per mode `j` and stage `ℓ`:

$$d_{\ell j}=\text{D\_MIN}+\mathrm{softplus}(\cdot)>0,\qquad
g_{\ell j}=\sigma(\cdot)\in(0,1),\qquad
n_{\ell j}=d_{\ell j}+g_{\ell j}\,\mathrm{softplus}(\delta_{\ell j}).$$

| gates | cascade | regime |
|---|---|---|
| `g₁ = g₂ = 0` | both stages identities | **exactly Native S5** |
| one gate on | `q → u` | ordinary prospectivity |
| both gates on | `q → u → w` | generalized WWJ |

The two-stage transfer is
`(1+n₁D)(1+n₂D) / [(1+d₁D)(1+d₂D)]`, whose numerator expands to
`1 + Γⁿ D + Mⁿ D²` with

$$\Gamma^n_j=n_{1j}+n_{2j},\qquad M^n_j=n_{1j}n_{2j},$$

and the **critical numerator branch** `n₁ = n₂ = τ/2` gives `Γⁿ = τ`,
`Mⁿ = τ²/4` — the WWJ critical mass, appearing as a **numerator**, where it
shapes the response without owning a pole.

Gates are **per mode**, deliberately: the construction exists so that
different modes may choose different regimes. A gate penalty (mean gate)
keeps the model from switching mechanics on everywhere for free, and gate
values are reported per layer and per mode.

## 4. Cancellation is monitored, not assumed away

Each stage's numerator zero sits at `n/(h+n)`. If that lands on a native
pole the mode is **annihilated**. The layer therefore reports
`|L_j(λ̄_j)|` — the effective filter's gain at each native pole — penalizes
it quadratically below a declared floor, and counts the modes beneath that
floor. Exact cancellation cannot pass silently; a test constructs a
cancelled mode and checks the gain detects it.

## 5. Implementation

Each stage is a **scalar (per-mode diagonal) affine associative scan**, run
with `s5/ssm.py`'s own `binary_operator` through
`jax.lax.associative_scan` — the repository's scan primitive, reused. The
generalized branch is **two sequential scalar scans**. **No dense per-mode
3×3 or 4×4 companion is ever constructed.** The S5 recurrence itself is
untouched; `h = 1` token, never identified with S5's learned `Δ`.

## 6. What is proved, and where

`tests/test_modal_prospective_algebra.py` — exact rational arithmetic, no
JAX, runs anywhere (10 tests): the update solves the stage equation on a
basis; every added pole is `d/(h+d)` and lies strictly inside the disc; the
numerator and gate never move a pole; `n = d` is the exact identity and a
cascade of identity stages is too; one active stage is the ordinary lead
operator; two stages expand exactly to `1 + ΓD + MD²`, with the critical
branch giving `Γ = τ`, `M = τ²/4`; the gate selects the regime exactly; zero
prehistory with no wraparound at either boundary; the reverse branch is the
flipped causal cascade; and a cancelled native mode is detected by the gain.

`tests/test_modal_prospective_jax.py` — the cluster suite (13 tests): scan
versus sequential oracle for real and complex modes at awkward lengths
**through 16 000**; the two-stage cascade against the oracle; identity
stages reproducing Native sequences **and gradients**; gradients against the
oracle for every parameter; zero prehistory and no wraparound; bidirectional
orientation; poles inside the disc with gates unable to move them;
production-length **float32** values and gradients finite; the cancellation
report; the layer starting at Native with its penalty increasing in the
gates; and Native S5 byte-identity against `ef004cd`.

## 7. First deliverables, and what is deliberately absent

Delivered: the derivation above, the implementation, both test suites, a
Native-versus-one-stage-versus-two-stage benchmark
(`experiments/s5_modal/benchmark.py`, 16 000 tokens, compile timed
separately, peak memory, forward and backward), and a tiny synthetic
memory-plus-switch experiment (`experiments/s5_modal/synthetic_gates.py`).

**Not** delivered, deliberately: the 15-epoch experiment is not wired and not
launched, Native S5 is not rerun, and **no scientific benefit is claimed**.
The synthetic experiment reports
`DIFFERENTIATION_DEMONSTRATED` only when the fitted gates are heterogeneous
across modes **and** the gated model beats the all-Native baseline on the
same data and budget; it separately reports the long-delay-memory error and
the lead error, because the scientific claim requires **both** retained
long-delay memory and reduced response lag, and neither is assumed.

Nothing in this commit launches anything.
