# Professor Ideas 1 and 2 - deterministic mechanism study

No SGD, no S5, no GPU. A 2-mode diagonal linear system integrated exactly, so
every theoretical claim is checked against closed form.

Reproduce: `python experiments/run_professor_mechanism.py`
Verify: `python -m pytest tests/test_mechanism.py -q` (13 tests)

## 1. Continuous equations

Plain plant:

```
tau ds/dt = (A - I) s + B x                       A = diag(a_i)
```

Idea 1, full state PC:

```
tau ds/dt = -s + f(s,x) + tau d/dt f(s,x),        f(s,x) = A s + B x
```

Substituting `f`, using `df/dt = A ds/dt + B dx/dt`, collecting `ds/dt` and
left-multiplying by `(I-A)^-1`, with `K = (I-A)^-1 B`:

```
tau (I - A) ds/dt = (A - I) s + B x + tau B dx/dt
         tau ds/dt = -s + K x + tau K dx/dt                     (*)
```

Idea 2, projected state PC (`P_M + P_T = I`):

```
tau ds/dt = -s + f(s,x) + tau P_T d/dt f(s,x)
```

For a diagonal system with a modal projector this decouples exactly:

```
memory mode   (P_T,ii = 0):  tau ds_i/dt = (a_i - 1) s_i + B_i x     -> pole (a_i-1)/tau
tracking mode (P_T,ii = 1):  tau ds_i/dt = -s_i + K_i x + tau K_i dx/dt -> pole -1/tau
```

## 2. Discrete equations actually coded

`rho = exp(-dt/tau)`, `lam_i = exp((a_i-1) dt/tau)`,
`bbar_i = B_i (lam_i - 1)/(a_i - 1)`, `K_i = B_i/(1-a_i)`.

```
plain / memory  :  s_k = lam_i s_{k-1} + bbar_i x_k
prospective     :  s_k = rho   s_{k-1} + K_i x_k - rho K_i x_{k-1}
static bypass   :  s_k = K_i x_k
readout lead    :  y_k = y_k + alpha (y_k - y_{k-1})
```

The prospective form is **exact**, not a finite difference. Writing `s = Kx + r`
in (*) gives `tau dr/dt = -r`, hence `r_k = rho r_{k-1}` and

```
s_k - K x_k = rho (s_{k-1} - K x_{k-1})
```

which rearranges to the coded recursion. It is first order: one state per mode,
no parasitic second-order state. Asserted by
`test_no_parasitic_second_order_state`.

## 3. System

| mode | role | a | B | K | plain pole | lam | half-life |
|---|---|---|---|---|---|---|---|
| 0 | memory | 0.98 | 1.0 | 50.0 | -0.020 | 0.999000 | 34.657 s |
| 1 | tracking | 0.20 | 1.3 | 1.625 | -0.800 | 0.960789 | 0.866 s |

`tau = 1.0`, `dt = 0.05`, `rho = 0.951229425`, readout `C = [1, 1]`.

## 4. Poles: theory vs measured

Measured from the **homogeneous** response (zero input, non-zero initial
state). Poles are a property of the homogeneous dynamics; the forced impulse
response is the wrong place to measure them here, because a fully prospective
mode has `s = K x` exactly and therefore an impulse response of length 1.

| model | mode | theory (disc) | measured | theory (cont) | half-life | impulse support |
|---|---|---|---|---|---|---|
| A plain | 0 | 0.999000 | 0.999000 | -0.0200 | 34.657 | >3000 |
| A plain | 1 | 0.960789 | 0.960789 | -0.8000 | 0.866 | 622 |
| B full PC | 0 | 0.951229 | 0.951229 | -1.0000 | 0.693 | **1** |
| B full PC | 1 | 0.951229 | 0.951229 | -1.0000 | 0.693 | **1** |
| C projected PC | 0 | 0.999000 | 0.999000 | -0.0200 | **34.657** | >3000 |
| C projected PC | 1 | 0.951229 | 0.951229 | -1.0000 | 0.693 | **1** |
| D static bypass | 0 | 0.999000 | 0.999000 | -0.0200 | 34.657 | >3000 |
| D static bypass | 1 | 0.0 | collapsed | -inf | 0 | 1 |

Theory matches measurement to 6 decimals everywhere.

## 5. Findings

**Full PC destroys memory, exactly as predicted.** Every pole becomes `-1/tau`;
the learned mode structure `A` vanishes from the homogeneous dynamics. The
memory mode's impulse retention drops from 0.67 at lag 400 to **0**, and its
forced impulse response is **one sample long**.

**Projected PC preserves the designated memory mode exactly.** Mode 0 is
bit-identical to plain S5 (`atol=0`), half-life 34.657 s unchanged, while the
tracking mode becomes prospective. This is the predicted behaviour and it is
what the mechanism was supposed to deliver.

**Tracking improves to zero lag in both.** Step-response time-to-correct falls
from 3.70 s (plain) to 0.00 s.

### The hostile controls all fire

| comparison | max abs difference | verdict |
|---|---|---|
| C projected PC vs D static bypass (tracking) | 2.9e-15 | **IDENTICAL** |
| B full PC vs D static bypass (tracking) | 2.9e-15 | **IDENTICAL** |
| E1 readout lead (matched alpha) vs C projected PC | 4.0e-15 | **IDENTICAL** |
| E1 readout lead vs D static bypass | 2.4e-15 | **IDENTICAL** |

A fully prospective tracking mode **is** the instantaneous equilibrium `K x`
plus a decaying initialization transient. Nothing else survives. And for a
first-order mode an ordinary output-side lead reproduces it *exactly*, because
the lead's zero cancels the pole when

```
alpha_i = lam_i / (1 - lam_i)
```

giving `P(z) S_plain(z) = K X(z)`. This is algebra, not an approximation.

### Where the mechanisms genuinely differ

The exact readout-lead equivalence is **per mode**. S5 applies one scalar alpha
per block, downstream of a readout that has already mixed modes with different
`lam_i`, and one zero cannot cancel several distinct poles:

```
matched alpha, tracking mode (lam=0.960789):    24.50
matched alpha, memory   mode (lam=0.999000):   999.50
```

With a single shared alpha on the mixed readout, the least-squares optimum is
`alpha = 13.79`, leaving **5.8 % RMS** residual against the projected-PC
readout. So a shared alpha does not reproduce projected state PC exactly - but
it gets within a few percent here, because the difference operator
`y - y_prev` is small for the slow memory mode and the lead therefore acts
almost selectively on the fast mode. The gap would widen as the two timescales
approach each other.

### Robustness

| model | noise gain | reset peak | reset final |
|---|---|---|---|
| A plain | 1.09 | 0.961 | 3.4e-04 |
| B full PC | **51.63** | 0.951 | 4.5e-05 |
| C projected PC | 1.94 | 0.951 | 4.5e-05 |
| D static bypass | 1.94 | 0.000 | 0 |
| E1 readout lead | 1.94 | 0.961 | 4.3e-19 |

Full PC amplifies white noise ~47x relative to plain, because it applies the
equilibrium gain `(I-A)^-1 B` and that gain is 50 for the slow mode. This is a
direct consequence of destroying the low-pass memory dynamics, and it is a
second independent reason not to apply prospectivity to slow modes.

All mechanisms remain affine in `(state, input)`, so all stay compatible with
an associative scan (`test_all_mechanisms_are_affine_and_scan_compatible`).

## 6. Figures

- `results/professor_mechanism/fig1_mechanism_overview.png` - memory (top) and
  tracking (bottom) for plain / full PC / projected PC.
- `results/professor_mechanism/fig2_hostile_control.png` - projected PC, static
  bypass and matched readout lead on one trajectory, with residuals at float64
  rounding.
