# Why the generalized operator could not save the equation, and why it was still the step that made the per-mode treatment possible

Every number here is reproduced by
`python -m experiments.s5_modal.why_generalized` — pure complex arithmetic,
no fitting, no seeds.

## 1. The equation

The note's prediction-correction RNN,

$$\tau\dot s=-s+f_\theta(s,t)+\underbrace{\tau\tfrac{d}{dt}f_\theta(s,t)}_{\text{prospective}},\qquad f_\theta(s,t)=As+Bx,$$

is exactly `P(D)(s − f) = 0` with `P(D) = 1 + τD`. That is worth saying
plainly, because it is what makes the next line inevitable.

## 2. The cancellation is structural, and it does not read P

$$s-f=(I-A)s-Bx.$$

`(I − A)` is a **constant** matrix, so it commutes with every `D`:

$$P(D)\big[(I-A)s-Bx\big]=0\;\Longrightarrow\;(I-A)P(D)s=P(D)Bx\;\Longrightarrow\;P(D)s=(I-A)^{-1}P(D)Bx.$$

The homogeneous dynamics are `P(D)s = 0`. **`A` has left the dynamics
entirely** and survives only as a static input gain: every mode relaxes
with whatever timescales `P` has, whatever `A` was.

The proof never reads `P`. So `1 + τD + Mτ²D²` cancels the memory in
*exactly* the same way as `1 + τD`. **Generalizing the operator cannot save
the equation**, and no higher order would have either. That is not a
failure of the generalization — it is a property of the residual `(s − f)`.

## 3. What the one knob actually does: lead and memory are the same object

The Euler discretization is not the exact law — its stencils are
inconsistent — so its poles do retain `a`. They are the roots of
`λ² − Ã₁λ + a`, and their **product is exactly `a`**. But that does not
give a second degree of freedom. There is still one knob, `ε = Δt/τ`, and
it is simultaneously the lead **and** the position of the memory pole.

Most lead the cell can reach before leaving the unit disc, and where the
memory pole has been dragged to when it gets there:

| mode | native pole | best lead (tokens) | pole then | timescale |
|---|---|---|---|---|
| 0.50 e^{0.2i} | 0.5000 | −0.024 | 0.9845 | 1.4 → **63.9** |
| 0.90 e^{0.2i} | 0.9000 | −1.385 | 0.9971 | 9.5 → **340.8** |
| 0.90 e^{1.0i} | 0.9000 | −0.316 | 0.9970 | 9.5 → **336.1** |
| 0.99 e^{0.2i} | 0.9900 | −2.794 | 0.9981 | 99.5 → **514.1** |
| 0.99 e^{1.0i} | 0.9900 | −0.353 | 0.9973 | 99.5 → **369.2** |

The lead is small and the price is the entire memory. A mode with a
1.4-token timescale comes back with a 64-token one. **The pole *is* the
knob** — pushing lead moves it, and `ε_max(a)` is where it stops:

$$\varepsilon_{\max}(a)=\frac{2\,(1-|a|^2)\,|1-a|^2}{\big|2a-1-|a|^2\big|^2},$$

exact (ρ crosses 1 there, verified against the true roots on 2000 random
complex modes, zero mismatches).

## 4. What generalizing changes: the lead moves into the zeros

The same modes, with a two-stage cascade `n = (8,8)`, `d = (1,1)` on top of
the untouched native mode:

| mode | lead | native pole | added poles | vs. the note's cell |
|---|---|---|---|---|
| 0.50 e^{0.2i} | −13.109 | **0.5000** | 0.500 | **556×** |
| 0.90 e^{0.2i} | −12.412 | **0.9000** | 0.500 | 9.0× |
| 0.90 e^{1.0i} | −14.385 | **0.9000** | 0.500 | 45.5× |
| 0.99 e^{0.2i} | −14.245 | **0.9900** | 0.500 | 5.1× |
| 0.99 e^{1.0i} | −14.488 | **0.9900** | 0.500 | 41.0× |

The native pole does not move **at all**. The added poles sit at
`d/(h+d) = 0.5` whatever the numerators do. The lead is five to five
hundred times larger.

This is not a better approximation of the professor's law. It is a
different object: an operator whose **poles and zeros are placed
independently**, instead of an operator applied to a residual that ties
them together.

## 5. Why that is what makes a per-mode choice possible

With the first-order law a mode has **one** free parameter, `ε`, whose
range `(0, ε_max(a))` is a function of the mode itself. The mode does not
choose anything: `a` fixes how much lead is available and what it costs.
There is no regime to select, so **there is nothing for a gate to gate**.

With poles and zeros placed independently the mode has a **two**-parameter
family `(Γ, M) = (n₁+n₂, n₁n₂)` over a denominator that is stable whatever
they are. "How much memory" and "how much lead" become separate
coordinates; a mode can hold one fixed and vary the other; and *some modes
use it and some do not* becomes an actual choice rather than a consequence
of `a`.

**That is what the generalized operator bought.** Not a repair of the
cancellation — §2 proves no operator repairs that — but the separation of
the two quantities the per-mode story has to be about.

## 6. What this argument does *not* establish

At matched DC lead, one zero and two zeros are close:

| target lead | one zero | two zeros | ω=0.05 | ω=0.2 | ω=0.5 |
|---|---|---|---|---|---|
| −2.5 | n = 5.087 | n = (3.043, 3.043) | −1.05 / −1.25 | 7.89 / 7.02 | 0.91 / 0.80 |
| −5.0 | n = 7.587 | n = (4.294, 4.294) | −2.82 / −3.44 | 8.14 / 6.75 | 1.08 / 1.10 |

A single zero can reach **any** DC lead on its own. So this argument
justifies moving the lead into the zeros; it does **not**, by itself,
justify the *second* zero — the `M f̈` term.

The evidence for the second zero is **empirical, not analytic**: in the
frontier run (§6i) the two-stage arm beat the one-stage arm it contains on
lead at every delay — 12–15 of 15 paired seeds, medians 1.34–1.63×, at an
identical initialization draw — while staying memory non-inferior at every
delay. That is the justification, and it should be quoted as a measurement
rather than dressed up as a derivation.

## 7. Summary

1. `P(D)(s−f) = 0` cancels `A` from the dynamics for **every** `P`. Proved.
2. So the generalized operator could not, and cannot, save the equation.
3. The Euler form keeps `a` in the poles but gives only **one** knob, which
   is the lead and the memory pole at once — so buying lead costs the
   memory, and the trade has a hard wall at `ε_max(a)`.
4. Placing poles and zeros independently separates the two, at 5–556× the
   lead with the native pole untouched.
5. **Only then is there a per-mode choice to make** — which is what the
   many-body term is for.
6. The second zero specifically rests on measurement, not on this argument.
