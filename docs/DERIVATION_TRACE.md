# Source-to-code traceability

Required by `docs/handoff_2026_09_15/DERIVATION_CONTRACT.md` ("Write a
source-to-code table") **before comparative training**. Every added physical
coefficient is traced from its circuit equation to the line that consumes it,
with what may learn stated explicitly.

## 1. The chain

| step | statement | where |
|---|---|---|
| circuit | `c u' + G_s u - h v = I_s`, `c_d v' + G_d v - h u = I_d` (NLA App. 6, Eqs. 80-81, reciprocal constant-conductance small-signal sector) | `s5/physical_coefficients.py` docstring |
| retained component | dendritic capacitance `c_d` — this is the modification under test | same |
| exact elimination | `c tau_d u'' + (c + G_s tau_d) u' + kappa u = b`, `b = I_s + (h/G_d) I_d`, `kappa = G_s - h^2/G_d > 0` | same |
| prospective source | `b = (1 + T D) bbar`, `r = u - bbar/kappa` gives `M u'' + gamma u' + r + T r' = 0` | same |
| tied coefficients | `gamma = G_s tau_d/kappa`, `M = T tau_d`, `rho = M/(gamma T) = kappa/G_s` | `PhysicalResponse.validate` enforces `M = rho gamma T` |
| symmetric reference | `c_s = c_d = c`, `g_L = h = g`, no background synaptic conductance ⇒ `G_s = G_d = 2g`, `kappa = 3g/2`, `tau_d = 3T/4`, `gamma = T`, `M = 3T^2/4`, `rho = 3/4` | `SYMMETRIC_REFERENCE` |
| model-time convention | `T = 5` input intervals | `T_INTERVALS = 5.0` |
| unit-sample physical tuple | `(T, gamma, M) = (5, 5, 18.75)` | `gamma_physical`, `mass_physical` |
| S5 normalization | `J_phys = -gamma F0`, `B_phys = gamma B0`; divide the **whole** equation by gamma | `fixed_m0_coefficients`, `mass_block_generator` |
| normalized coefficients | `gamma_n = 1`, `T = 5`, `mu = M/gamma = 3.75`, `rho = 0.75` | `PhysicalResponse.mass` |
| clock absorption | `a = Delta * lambda`, `b = Delta * B_c`, applied EXACTLY ONCE | `SubstrateSSM.coefficients` |

Verified numerically: `T=5.0, gamma_n=1.0, rho=0.75, mu=3.75, M_phys=18.75,
gamma_phys=5.0` (`test_coefficients_are_the_contract_values_and_immutable`).

## 2. Exact discrete update and where each quantity lives

**M = 0** (`s5/gp_fixed.py:fixed_m0_coefficients`). General form, then the
scalar-T diagonal-J reduction actually executed:

```
W = gamma I + T J        D_x = W^-1 T B      F = -W^-1 J      B_h = W^-1(B - J D_x)
J = -a diagonal, T scalar  =>  W = gamma - T a
F = a/(gamma - T a)    D_x = T b/(gamma - T a)    B_h = gamma b/(gamma - T a)^2
a_bar = exp(F)         b_bar = phi1(F) B_h
h_k = a_bar h_{k-1} + b_bar x_k    (h_{-1} = 0)
s_k = h_k + D_x x_k
y_k = 2 Re{C_tilde s_k} + D (*) x_k
```

Elementwise division is used only because T = 5I and J are simultaneously
diagonal, so the solve *is* the division. A full-matrix mechanism is not added
in this batch, and the file says so.

**M > 0** (`s5/gp_fixed.py:mass_block_zoh`), carry `z = (s, v)`:

```
gamma rho s' = -J s - gamma(1-rho) v + B x
T v'         = s' - v

A = [[ -J/(gamma rho),      -(1-rho) I/rho ],
     [ -T^-1 J/(gamma rho), -T^-1/rho      ]]
B_blk = [ B/(gamma rho) ; T^-1 B/(gamma rho) ]

aug = [[A, I2],[0,0]] (4x4 per mode)   expm(aug) = [[A_bar, Phi],[0, I]]
A_bar = e^A,  Phi = int_0^1 e^{As} ds,  B_bar = Phi B_blk
z_k = A_bar z_{k-1} + B_bar x_k        (z_{-1} = 0)
s_k = z_k[0]                            READ FIRST COMPONENT ONLY
y_k = 2 Re{C_tilde s_k} + D (*) x_k     NO D_x term at positive mass
```

Eliminating `v` returns `M s'' + gamma s' + r + T r' = 0` with `M = rho gamma T`
exactly — verified against independent SciPy evolution with derivatives taken
from the ODE rather than by finite differences
(`test_mass_block_satisfies_the_target_law_against_scipy`).

The per-mode **4x4** exponential replaces the earlier prototype's
`(2+H)x(2+H)` augmented exponential; cost is now independent of feature width.
`expm` is used throughout, never an eigendecomposition, because admissible
exceptional points exist where eigenvectors coalesce while the exponential and
its parameter derivatives stay analytic
(`test_repeated_poles_do_not_break_the_exponential_or_its_gradient`).

## 3. Initial conditions

Zero prehistory, `z_{-1} = 0`: the circuit starts at rest and the source jumps
on at the first token. This is a **declared convention**. Its check is that
with zero prehistory `rho = 1` reproduces the ordinary driven response at the
same gamma — verified to 1e-10 in float64 and 3.9e-07 in float32
(`test_rho_one_with_zero_prehistory_is_the_ordinary_driven_response`).

For M = 0 the current-input term `D_x x_k` is present at the FIRST token and
across chunk boundaries (`test_input_jump_is_handled_at_the_first_token_and_at_a_step`,
`test_chunked_streaming_equals_the_full_sequence`).

## 4. Buffer and parameter locations; what can learn

| quantity | where it lives | trainable |
|---|---|---|
| `T`, `gamma`, `rho`, `mu` | `PhysicalResponse`, a frozen dataclass passed as static module config | **NO** |
| `Lambda_re`, `Lambda_im` | `S5SSM` parameters | yes |
| `B`, `C`, `D` | `S5SSM` parameters | yes |
| `log_step` (the native learned clock) | `S5SSM` parameter | yes |
| `alpha` gain | derived from `Lambda`, never stored | no (derived) |
| two-tap coefficients | derived from `Lambda`, `Delta`, `B_c` | no (derived) |

Checked, not asserted: for both fixed arms the parameter tree is exactly
`{B, C, D, Lambda_im, Lambda_re, log_step}` — no response parameter exists, so
no gradient, no optimizer slot and no weight-decay term can reach the physical
coefficients (`test_fixed_coefficients_are_outside_the_gradient_and_the_optimizer`).
The runner independently refuses to start if a response-like parameter appears
(`assert_response_is_not_trainable`).

The converse is also checked, so that "fixed physical response" is not an empty
claim: changing `T` or `rho` changes the model output
(`test_changing_a_fixed_coefficient_changes_the_model`).

## 5. Normalized versus unnormalized, and the c_d -> 0 limit

`gamma_physical = 5` and `mass_physical = 18.75` are retained in the API
alongside the normalized `gamma_n = 1`, `mu = 3.75`, so the physical check
remains possible. Dividing only one term would be a different model;
`validate()` enforces the tie.

**The `gp_fixed_m0` arm is NOT the fast-dendrite circuit limit.** The true
`c_d -> 0` limit sends gamma and M to zero together at fixed intrinsic
parameters; `gp_fixed_m0` keeps `gamma = 1 > 0`. It is a reduced/constitutive
ablation and is labelled as one everywhere, including in `M0_REFERENCE.label`.

## 6. Known gap, stated before training

The passive plant supplies the tied coefficients. It does **not** prescribe the
prospective closed-loop source: substituting `bbar/kappa = f_theta(u, x)` with
a learned S5 residual is an explicit feedback convention and a **computational
extension**, not something inherited from the circuit derivation. A
nonsymmetric S5 residual is not automatically the gradient of the original NLA
mismatch energy. Its stability here rests on the declared constraints — `T` SPD
(here `5I`), `sym(J) > 0` (from native clipping, enforced by a raise if
`clip_eigs=False`), scalar `gamma > 0`, scalar `rho` in (0, 1] — and not on a
global biological derivation of the nonlinear network.

This gap is **not repaired by an added trainable correction**, per the
contract. It is recorded here, and it bounds what any benchmark outcome can be
claimed to show.
