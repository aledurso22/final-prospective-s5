# The three prospective codings, derived from one action

Every line is checked by `python -m experiments.s5_modal.wwj_lagrangian`.
They are not three models. They are the **same action** with one ingredient
added each time.

## 0. The action

Generalized coordinate: the **prediction error** `e = s − f`.

$$T=\tfrac12 M\dot e^{\,2},\qquad V=\tfrac12 k e^{2},\qquad \mathcal R=\tfrac12\tau\dot e^{\,2}$$

kinetic, potential, Rayleigh dissipation. With `L = T − V` and

$$\frac{d}{dt}\frac{\partial L}{\partial\dot e}-\frac{\partial L}{\partial e}+\frac{\partial\mathcal R}{\partial\dot e}=0$$

the Euler–Lagrange equation is, once and for all,

$$\boxed{M\ddot e+\tau\dot e+k e=0}\qquad\Longleftrightarrow\qquad (k+\tau D+MD^{2})\,e=0 .$$

---

## 1. Ordinary prospective coding — **drop the kinetic term**

`M = 0`: no inertia. Euler–Lagrange degenerates to gradient flow of `V`
against `R`:

$$\tau\dot e+e=0\;\Longrightarrow\;(1+\tau D)(s-f)=0\;\Longrightarrow\;s-f+\tau(\dot s-\dot f)=0$$

$$\boxed{\tau\dot s=-s+f_\theta(s,t)+\tau\tfrac{d}{dt}f_\theta(s,t)}$$

which is **the note's Eq. (1), exactly**. The prospective term `τ ḟ` is not
an addition to the model — it is what first-order relaxation of the
prediction error *looks like* once you write it as an equation for `s`.

**One root, therefore one timescale** (`τ/k`). That single number must be
both the mode's memory and its prospective horizon. It cannot be both.

---

## 2. Generalized prospective coding — **keep the kinetic term**

`M > 0`: the full Euler–Lagrange equation survives.

$$M\ddot e+\tau\dot e+e=0\;\Longrightarrow\;(1+\tau D+MD^{2})(s-f)=0$$

Second order, so **two roots, two relaxation times**:

| M | τ | regime | timescales |
|---|---|---|---|
| 0 | 5 | no kinetic term | 5.000 |
| 4 | 5 | overdamped | 4.000, 1.000 |
| **6.25** | 5 | **critical** | **2.500, 2.500** |
| 10 | 5 | underdamped | 2.50 ± 1.94i |

**The critical mass is the whole point.** `M = τ²/4` is where the two roots
*merge*: a critically damped body has one timescale again and is the
ordinary case in disguise. `M < τ²/4` **splits them into two distinct real
relaxation times** — and that split is what lets one hold the memory while
the other provides the lead.

**The inverse map is how the model is actually parameterized.**
`P(D) = (1 + T₁D)(1 + T₂D) = 1 + ΓD + MD²` gives

$$\Gamma=T_1+T_2,\qquad M=T_1T_2$$

which is *exactly* `gamma_and_mass` in `s5/modal_prospective.py`. **The
cascade's two numerator parameters `n₁, n₂` are the action's two relaxation
times.** Setting `n₁ = n₂` puts the body on the critical line — which is
why tied stage initialization had to be broken: tied stages are ordinary
prospectivity wearing a second stage.

---

## 3. Why both of them cancel the memory

The action is written in `e`; the model is written in `s`; they are related
by `e = (I − A)s − Bx` — for constant `A`, an **affine change of
coordinates with constant Jacobian `(I − A)`**.

Euler–Lagrange equations are covariant under a constant linear change of
variables. Substituting:

$$(I-A)\big[M\ddot s+\tau\dot s+k s\big]=(\text{terms in }x\text{ only}),$$

and the constant `(I − A)` divides straight out. **The dynamics of `s` are
the roots of the operator and nothing else.** `A` never enters them, at any
order in `D`.

That is the cancellation theorem in the language it came from: **`A` is a
change of coordinates, and no Lagrangian written purely in the error can
see a change of coordinates.** Raising the order of `P` cannot help,
because the proof never reads `P`.

---

## 4. Many-body / per-mode — **one action term per body**

S5's `A` is **diagonal**, so the modes are already the *normal coordinates*
of the system: `e_j = (1 − a_j)s_j − b_j x`. The many-body action is the
sum over them, each with its own constants:

$$L=\sum_j\Big[\tfrac12 M_j\dot e_j^{\,2}-\tfrac12 k_j e_j^{2}\Big],\qquad \mathcal R=\sum_j\tfrac12\tau_j\dot e_j^{\,2}$$

and Euler–Lagrange, now one equation **per body**:

$$M_j\ddot e_j+\tau_j\dot e_j+k_je_j=0\;\Longleftrightarrow\;\big(k_j+\tau_jD+M_jD^{2}\big)(s_j-f_j)=0 .$$

§3 still applies to each body separately — `a_j` still divides out. What
has changed is that **the operator itself now depends on the body**. The
dynamics of `s_j` are the roots of `P_j`, and `P_j` is ours to choose.
**The memory does not come back through the residual; it comes back through
the per-mode constants of the action.**

And this is why it needed the generalized form first:

| order | roots per body | can hold memory | can add lead | **both at once** |
|---|---|---|---|---|
| ordinary, `M_j = 0` | 1 | yes | yes | **NO** |
| generalized, `M_j > 0` | 2 | yes | yes | **yes** |

With one root per body the single timescale must serve both demands, so a
per-mode gate would have **nothing to select between** — `a_j` already
fixes what is available. With two roots there is a genuine two-parameter
family `(Γ_j, M_j) = (T₁ⱼ+T₂ⱼ, T₁ⱼT₂ⱼ)`, and each body places its two times
where it needs them.

**"Some modes use it and some do not" is then a statement about each body's
phase in the `(M_j, τ_j)` plane:**

| body | T_memory | T_lead | Γ | M | M/(Γ/2)² | phase |
|---|---|---|---|---|---|---|
| 0 | 300 | 300 | 600 | 90000 | **1.0000** | critical |
| 1 | 300 | 60 | 360 | 18000 | 0.5556 | overdamped |
| 2 | 300 | 3 | 303 | 900 | **0.0392** | overdamped |
| 3 | 8 | 8 | 16 | 64 | **1.0000** | critical |
| 4 | 8 | 0.5 | 8.5 | 4 | 0.2215 | overdamped |

A body **on** the critical line has `T_memory = T_lead` — it is ordinary
prospectivity and has given up the separation. A body **far below** it
holds a long memory time and a short lead time at once, which is exactly
what the construction was built to allow and exactly what a single body
cannot do.

---

## 5. Summary

1. One action, `M ë + τ ė + k e = 0`, in the prediction error.
2. **Drop the mass** → the note's Eq. (1). One timescale per body.
3. **Keep the mass** → the generalized WWJ operator. Two timescales, split
   below the critical mass `M = τ²/4`.
4. **Both cancel `A`** — it is a constant change of coordinates, invisible
   to an action written in the error. No order of `P` repairs that.
5. **Sum over bodies with per-mode constants** → the operator becomes
   mode-dependent, so the memory returns through `(Γ_j, M_j)` rather than
   through the residual, and each body chooses its phase.
6. The two roots are `n₁, n₂` in the code; `Γ = n₁+n₂`, `M = n₁n₂`.
