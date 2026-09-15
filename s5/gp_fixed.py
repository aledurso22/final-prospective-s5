"""FIXED-coefficient generalized prospective response for S5.

Implements the family of `docs/handoff_2026_09_15/CLUSTER_CODING_BRIEF.md` s3

    M s_ddot + gamma s_dot + r + T r_dot = 0,
    r = J s - B x,   M = 0  or  M = rho gamma T,  0 < rho <= 1

with the coefficients FIXED by `s5.physical_coefficients`. There is no
trainable response here and no `gp_response_raw` parameter: the values live in
frozen configuration, so they cannot be reached by gradients, by the optimizer
or by weight decay. Ordinary S5 parameters (Lambda, B, C, D, log_step) train
normally through the exact discretization.

Two mechanisms
--------------
``gp_fixed_m0``    M = 0, gamma > 0. A REDUCED / constitutive ablation, not the
                   true fast-dendrite circuit limit (which sends gamma and M to
                   zero together at fixed intrinsic parameters).
``gp_fixed_mass``  the primary physical candidate, M = rho gamma T > 0.

Why this file may use elementwise division
------------------------------------------
The brief requires ordered matrix solves, never elementwise `b/(1-ta)^2`, for
NONCOMMUTING matrices. The currently authorized model has T = 5I (scalar) and
J diagonal in the native modal basis, so every matrix in sight commutes and the
solve IS the elementwise division. The general formulas are written below in
solve form and reduce to the scalar expressions used here; a full-matrix
mechanism is deliberately not added in this batch.

    W   = gamma I + T J
    D_x = W^-1 T B
    F   = -W^-1 J
    B_h = W^-1 (B - J D_x)

With J = -a diagonal and T scalar: W = gamma - T a, F = a/(gamma - T a),
D_x = T b/(gamma - T a), B_h = gamma b/(gamma - T a)^2.

Positive mass carries z = (s, v), per the brief:

    gamma rho s_dot = -J s - gamma (1 - rho) v + B x
    T v_dot         = s_dot - v

    A = [[ -J/(gamma rho),    -(1-rho) I/rho ],
         [ -T^-1 J/(gamma rho), -T^-1/rho    ]]
    B = [ B/(gamma rho) ; T^-1 B/(gamma rho) ]

Eliminating v returns M s_ddot + gamma s_dot + r + T r_dot = 0 with
M = rho gamma T exactly; `tests/` checks this numerically rather than trusting
the algebra.

Discretization. Unit-interval ZOH via a per-mode 4x4 augmented exponential

    aug = [[A, I2], [0, 0]],   expm(aug) = [[A_bar, Phi], [0, I]]

so A_bar = e^A and Phi = int_0^1 e^{As} ds, then B_bar = Phi @ B_blk. The cost
is INDEPENDENT of feature width H: the earlier prototype exponentiated a
(2+H)x(2+H) matrix per mode. Never an eigendecomposition - admissible
exceptional points exist where eigenvectors coalesce while the exponential and
its parameter derivatives stay analytic.

Timing. Consume the held token, advance the block state, read the FIRST
component s. At positive mass NO first-order `D_x x` feedthrough is added: the
positive-mass state transfer is strictly proper and adding it would
double-count a different family member's feedthrough. Native learned D and the
conjugate real readout factor are retained on every arm.

Initialization. Zero prehistory z_{-1} = 0, i.e. the circuit starts at rest and
the source jumps on at the first token. That is a DECLARED convention; with
zero prehistory rho = 1 reproduces the ordinary driven response at the same
gamma, which is the reduction test.
"""

import jax
import jax.numpy as np
from jax.scipy.linalg import expm

from .gp_coefficients import phi1
from .gp_second_order import block_binary_operator
from .physical_coefficients import PhysicalResponse

FIXED_MECHANISMS = ("gp_fixed_m0", "gp_fixed_mass")


# ---------------------------------------------------------------- M = 0 ----
def fixed_m0_coefficients(a, b, T, gamma):
    """Exact M = 0 coefficients. `a`, `b` are ALREADY clock-absorbed."""
    T_c = np.asarray(T, dtype=a.dtype)
    g_c = np.asarray(gamma, dtype=a.dtype)
    W = g_c - T_c * a                      # = gamma I + T J, J = -a
    F = a / W                              # = -W^-1 J
    d_x = (T_c * b) / W[:, None]           # = W^-1 T B
    B_h = (g_c * b) / (W ** 2)[:, None]    # = W^-1 (B - J D_x)
    a_bar = np.exp(F)
    b_bar = phi1(F)[:, None] * B_h
    return dict(a_eff=F, a_bar=a_bar, b_hist=B_h, b_bar=b_bar, d_x=d_x, W=W)


# ---------------------------------------------------------------- M > 0 ----
def _row_factor(v):
    """Shape a scalar or per-mode (P,) coefficient for dividing (P, H) ROWS.

    Dividing a (P, H) input matrix by a (P,) vector broadcasts along the
    FEATURE axis H. That is silently valid whenever H == P and WRONG in every
    other case, so a per-mode coefficient must be reshaped to (P, 1) first.
    A scalar is returned unchanged. Tested with P != H.
    """
    v = np.asarray(v)
    return v if v.ndim == 0 else v[..., None]


def mass_block_generator(a, b, T, gamma, rho):
    """Continuous per-mode (A, B) for the (s, v) carry. (P,2,2), (P,2,H).

    `gamma` and `rho` may be scalars (the fixed arms) or per-mode (P,) arrays
    (the learned-response arm). `T` is always scalar: the published-comparison
    horizon is FIXED and is not learned.
    """
    T_c = np.asarray(T, dtype=a.dtype)
    g_c = np.asarray(gamma, dtype=a.dtype)
    r_c = np.asarray(rho, dtype=a.dtype)
    J = -a
    gr = g_c * r_c
    one = np.ones_like(J)
    A = np.stack([
        np.stack([-J / gr, -((1.0 - r_c) / r_c) * one], axis=-1),
        np.stack([-J / (gr * T_c), -(one / (r_c * T_c))], axis=-1),
    ], axis=-2)
    B = np.stack([b / _row_factor(gr), b / _row_factor(gr * T_c)], axis=-2)
    return A, B


def mass_block_zoh(a, b, T, gamma, rho):
    """Unit-interval ZOH by a per-mode 4x4 exponential (width-independent)."""
    A, B = mass_block_generator(a, b, T, gamma, rho)
    P = A.shape[0]
    aug = np.zeros((P, 4, 4), dtype=A.dtype)
    aug = aug.at[:, :2, :2].set(A)
    aug = aug.at[:, :2, 2:].set(np.broadcast_to(np.eye(2, dtype=A.dtype),
                                                (P, 2, 2)))
    E = jax.vmap(expm)(aug)
    A_bar = E[:, :2, :2]
    Phi = E[:, :2, 2:]                      # int_0^1 exp(A s) ds
    B_bar = np.einsum("pij,pjh->pih", Phi, B)
    return dict(A=A, B=B, A_bar=A_bar, Phi=Phi, B_bar=B_bar)


def mass_scan(A_bar, B_bar, input_sequence, z0=None, reset_mask=None):
    """z_k = A_bar z_{k-1} + B_bar x_k over 2x2 blocks; z_{-1} = 0 by default."""
    L = input_sequence.shape[0]
    P = A_bar.shape[0]
    A_elems = np.broadcast_to(A_bar, (L, P, 2, 2))
    if reset_mask is not None:
        keep = (~np.asarray(reset_mask, dtype=bool))[:, None, None, None]
        A_elems = A_elems * keep.astype(A_bar.dtype)
    B_elems = jax.vmap(lambda u: B_bar @ u)(input_sequence)       # (L,P,2)
    if z0 is not None:
        B_elems = B_elems.at[0].add((A_elems[0] @ z0[..., None])[..., 0])
    _, zs = jax.lax.associative_scan(block_binary_operator, (A_elems, B_elems))
    return zs


def mass_scan_sequential(A_bar, B_bar, input_sequence, z0=None,
                         reset_mask=None):
    """Same recurrence by `lax.scan`; differentiable, for scan-order checks."""
    P = A_bar.shape[0]
    z_init = np.zeros((P, 2), dtype=A_bar.dtype) if z0 is None else z0
    mask = (np.zeros(input_sequence.shape[0], dtype=bool)
            if reset_mask is None else np.asarray(reset_mask, dtype=bool))

    def step(z, xm):
        u, drop = xm
        z_prev = np.where(drop, np.zeros_like(z), z)
        z_next = (A_bar @ z_prev[..., None])[..., 0] + B_bar @ u
        return z_next, z_next

    _, zs = jax.lax.scan(step, z_init, (input_sequence, mask))
    return zs


def mass_readout(zs, C_tilde, conj_sym):
    """Read the FIRST block component s only; no positive-mass feedthrough."""
    s = zs[..., 0]
    if conj_sym:
        return jax.vmap(lambda si: 2 * (C_tilde @ si).real)(s)
    return jax.vmap(lambda si: (C_tilde @ si).real)(s)


def state_counts(P, conj_sym, mechanism):
    """Executed carry sizes, in REAL coordinates, derived not assumed.

    The stored complex modes are P; conjugate symmetry means the physical real
    state has 2P coordinates per layer, and that is ALSO the whole ordinary
    carry - it is not doubled again. Positive mass adds an auxiliary velocity
    carry of the same size.
    """
    physical = 2 * P if conj_sym else P
    auxiliary = physical if mechanism == "gp_fixed_mass" else 0
    return dict(physical_real=physical, auxiliary_real=auxiliary,
                previous_input_buffer=0,
                total_real=physical + auxiliary)


def fixed_coefficients(mechanism, a, b, response: PhysicalResponse):
    """Dispatch. `response` is a frozen PhysicalResponse; nothing here learns."""
    if mechanism == "gp_fixed_m0":
        return fixed_m0_coefficients(a, b, response.T, response.gamma)
    if mechanism == "gp_fixed_mass":
        return mass_block_zoh(a, b, response.T, response.gamma, response.rho)
    raise ValueError(f"unknown fixed mechanism {mechanism!r}; expected one of "
                     f"{FIXED_MECHANISMS}")


# ------------------------------------------- prospective INPUT correction ---
#: Rawat's input horizon, in native-clock intervals. FIXED, never learned.
PROSPECTIVE_INPUT_HORIZON = 5.0


def mass_block_two_tap(a, b, T, gamma, rho, horizon_in=PROSPECTIVE_INPUT_HORIZON):
    """Exact interval law for q' = A q + B (x + T_in x'), held tokens + jumps.

    The input correction is a DISTRIBUTIONAL jump, not a sampled derivative.
    With the token held on [k, k+1) and jumping from x_{k-1} to x_k at the
    start of interval k,

        q(0+) - q(0-) = T_in B (x_k - x_{k-1}),

    followed by ordinary held-input evolution over one unit interval. Hence

        q_k = A_bar q_{k-1} + B_plus x_k + B_minus x_{k-1},
        J_in    = T_in * A_bar @ B          (CONTINUOUS B, not B_bar),
        B_plus  = B_bar + J_in,
        B_minus = -J_in.

    Three ways to get this wrong, all excluded here:

    * `T_in * B_bar` - that integrates the jump as if it were a held input;
    * `T_in * A_bar @ B_bar` - that advances the jump twice;
    * an extra `Delta` factor - the clock is absorbed exactly once, upstream,
      when `a` and `b` are formed. There is no second absorption here.

    The autonomous poles are untouched: only the input coupling and the output
    residues change. `B_plus + B_minus = B_bar`, so the DC response is
    unchanged. At `horizon_in = 0` this returns the plain generalized
    recurrence, and at `rho = 1` the s component reduces to Rawat's two-tap
    alpha-P-S5 for the same common parameters and zero prehistory; both are
    tested rather than asserted.
    """
    d = mass_block_zoh(a, b, T, gamma, rho)
    T_in = np.asarray(horizon_in, dtype=d["A_bar"].dtype)
    J_in = T_in * np.einsum("pij,pjh->pih", d["A_bar"], d["B"])
    return dict(d, J_in=J_in, B_plus=d["B_bar"] + J_in, B_minus=-J_in,
                horizon_in=horizon_in)


def _delayed_inputs(input_sequence, prev_x=None, reset_mask=None):
    """x_{k-1} with the carried previous token and per-token resets.

    Clearing only the block state `q` at a sequence boundary is NOT a reset:
    the second tap would still read the last token of the PREVIOUS sequence.
    Both carries are cleared here.
    """
    if prev_x is None:
        head = np.zeros_like(input_sequence[:1])
    else:
        head = np.asarray(prev_x, dtype=input_sequence.dtype)[None]
    x_prev = np.concatenate([head, input_sequence[:-1]], axis=0)
    if reset_mask is not None:
        keep = (~np.asarray(reset_mask, dtype=bool))[:, None]
        x_prev = x_prev * keep.astype(x_prev.dtype)
    return x_prev


def mass_two_tap_drive(B_plus, B_minus, input_sequence, prev_x=None,
                       reset_mask=None):
    """B_plus x_k + B_minus x_{k-1} over 2x2 blocks; shape (L, P, 2)."""
    x_prev = _delayed_inputs(input_sequence, prev_x, reset_mask)
    cur = jax.vmap(lambda u: B_plus @ u)(input_sequence)
    prev = jax.vmap(lambda u: B_minus @ u)(x_prev)
    return cur + prev


def mass_scan_two_tap(A_bar, B_plus, B_minus, input_sequence, z0=None,
                      prev_x=None, reset_mask=None):
    """q_k = A_bar q_{k-1} + B_plus x_k + B_minus x_{k-1}, parallel scan."""
    L = input_sequence.shape[0]
    P = A_bar.shape[0]
    A_elems = np.broadcast_to(A_bar, (L, P, 2, 2))
    if reset_mask is not None:
        keep = (~np.asarray(reset_mask, dtype=bool))[:, None, None, None]
        A_elems = A_elems * keep.astype(A_bar.dtype)
    B_elems = mass_two_tap_drive(B_plus, B_minus, input_sequence, prev_x,
                                 reset_mask)
    if z0 is not None:
        B_elems = B_elems.at[0].add((A_elems[0] @ z0[..., None])[..., 0])
    _, zs = jax.lax.associative_scan(block_binary_operator, (A_elems, B_elems))
    return zs


def mass_scan_two_tap_sequential(A_bar, B_plus, B_minus, input_sequence,
                                 z0=None, prev_x=None, reset_mask=None):
    """The same recurrence by `lax.scan`. Differentiable; for scan-order checks.

    Written independently of the parallel path - it carries the previous token
    in the scan state rather than building a shifted array - so agreement
    between the two is evidence, not a restatement.
    """
    P = A_bar.shape[0]
    H = input_sequence.shape[-1]
    z_init = np.zeros((P, 2), dtype=A_bar.dtype) if z0 is None else z0
    x_init = (np.zeros((H,), dtype=input_sequence.dtype) if prev_x is None
              else np.asarray(prev_x, dtype=input_sequence.dtype))
    mask = (np.zeros(input_sequence.shape[0], dtype=bool)
            if reset_mask is None else np.asarray(reset_mask, dtype=bool))

    def step(carry, xm):
        z, x_prev = carry
        u, drop = xm
        z_prev = np.where(drop, np.zeros_like(z), z)
        x_del = np.where(drop, np.zeros_like(x_prev), x_prev)
        z_next = ((A_bar @ z_prev[..., None])[..., 0]
                  + B_plus @ u + B_minus @ x_del)
        return (z_next, u), z_next

    _, zs = jax.lax.scan(step, (z_init, x_init), (input_sequence, mask))
    return zs
