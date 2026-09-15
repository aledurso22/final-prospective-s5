"""Activity-dependent generalized prospective coding in the recurrence.

Three levels are kept explicitly separate, as the contract requires.

LEVEL 1 - published compartment equations (NLA Appendix 6, Eqs. 80-81)
----------------------------------------------------------------------
    c_s u' = g_L(E_L - u) + g_sd(v - u)
    c_d v' = g_L(E_L - v) + g_E(t)(E_E - v) + g_I(t)(E_I - v) + g_ds(u - v)

With E_L as the reference potential, g_sd = g_ds = h, and

    G_s   = g_L + h                    FIXED  (intrinsic)
    G_d(t)= g_L + h + g_E(t) + g_I(t)  VARIES (activity dependent)
    I_d(t)= g_E(t) E_E + g_I(t) E_I    the synaptic DRIVE CURRENT

so a synapse has BOTH effects: it raises G_d (shunting, a shorter dendritic
time constant) and it injects I_d. Keeping only the time-constant effect would
not be this circuit.

LEVEL 2 - finite-dendrite prospective construction
--------------------------------------------------
Retaining c_d, with I_s = 0, eliminate v EXACTLY. From the somatic equation
v = (c u' + G_s u)/h, whose coefficients c, G_s, h are all FIXED, so

    v' = (c u'' + G_s u')/h

and substituting into the dendritic equation gives

    c^2 u'' + c(G_s + G_d(t)) u' + (G_s G_d(t) - h^2) u = h I_d(t).      (*)

**No dG_d/dt term appears.** The varying conductance multiplies v
algebraically, and v maps to u' with constant coefficients. This is the reason
to realize the model in PHYSICAL coordinates: the naive route of making the
computational (s, w) block's coefficients time dependent does NOT satisfy the
same law - direct differentiation of that block leaves
-T(A_c' s' + B_c' w) on the right-hand side. Here there is nothing to correct.

Dividing (*) by G_d(t), with tau_d(t) = c/G_d(t) and
kappa(t) = G_s - h^2/G_d(t):

    c tau_d(t) u'' + (c + G_s tau_d(t)) u' + kappa(t) u = b(t),
    b(t) = (h tau_d(t)/c) I_d(t)

which is the frozen form of the contract with time-varying coefficients.

The prospective source, time-dependent version. Writing the source as the
prospective image of b_bar with the horizon INSIDE the derivative,

    b = b_bar + d(T(t) b_bar)/dt ,          r = u - b_bar/kappa(t)

and using T(t) kappa(t) = c_s (a constant, since T = c_s/kappa), the residual
law comes out with NO extra terms:

    M(t) u'' + gamma(t) u' + r + T(t) r' = 0
    T = c/kappa,  tau_d = c/G_d,  gamma = G_s tau_d/kappa,  M = T tau_d.

The alternative convention b = b_bar + T(t) b_bar' instead leaves
+(T kappa'/kappa)(u - r) on the right. The operator form (1 + D o T) is
therefore not a matter of taste: it is the one that preserves the law under
adaptation, because d(T kappa)/dt = 0. This is recorded as the chosen
convention, with its reason.

LEVEL 3 - computational mapping to a complex S5 mode
-----------------------------------------------------
The residual is the LEARNED feedback/source: r = gamma_0 (J s - beta x) with
J = -diag(a), a = Delta*Lambda, beta = Delta*B_c, absorbing the native clock
exactly once. This substitution is a COMPUTATIONAL EXTENSION, not something the
passive plant prescribes, and a complex S5 weight is not a nonnegative
conductance.

Quadrature sharing. Conductances are real, nonnegative, physical quantities. A
complex mode's two real quadratures belong to the SAME compartment, so they
share one real G_d(t); the modulation multiplies the complex state by a real
scalar and the complex structure is untouched. The complex pole itself is
computational: a passive reciprocal two-compartment circuit has real
eigenvalues, so the complex J is learned feedback and is NOT claimed to be a
passive conductance.

Realization actually integrated
-------------------------------
Physical coordinates z = (s, s'). The map to (u, v) is
v = (c s' + G_s s)/h with CONSTANT coefficients, so z carries no
parameter-dependent coordinate change and needs no correction when the
modulation jumps between tokens.

    M z2' = -(gamma + T gamma_0 J) z2 - gamma_0 J z1
            + gamma_0 beta x + T gamma_0 beta x'
    z1' = z2

Within a token the modulation is held constant (it is a function of that
token's layer input), so the generator is constant and the update is an exact
matrix exponential; the x' term is impulsive at token boundaries and enters as
a jump in s'. Because the coefficients depend only on the layer INPUT, all
per-token blocks are precomputable and the associative scan is retained.

Declared modeling assumptions, fixed BEFORE any validation score
----------------------------------------------------------------
* c_s = c_d = c = 1 and g_L = h = g = 2/15 (time in input intervals). These
  are not free: they are the unique symmetric-reference values reproducing the
  contract tuple T = 5, tau_d = 3.75, gamma_phys = 5, M_phys = 18.75,
  rho = 3/4. `validate_reference()` checks this.
* The conductance modulation reuses the EXISTING synaptic weights and adds no
  parameters:

      drive_{p,k} = |beta_p . x_k|                       (existing complex drive)
      q_{p,k}     = MOD_SCALE * G_d0 * drive/(DRIVE_REF + drive)   >= 0

  so q is nonnegative and bounded by MOD_SCALE*G_d0: G_d can at most double and
  tau_d at least halves. Boundedness stands in for finite synaptic resources.
  MOD_SCALE = 1.0 and DRIVE_REF = 1.0 are DIMENSIONLESS CALIBRATION CONSTANTS
  declared here; they are not from NLA and are not swept.
* The current effect of the synapse is the existing complex drive beta.x (the
  level-3 computational source); the conductance effect is q, from the same
  weights. Both effects of each synapse are therefore present. What is NOT
  claimed is that the complex drive literally equals g_E E_E + g_I E_I with
  physical reversal potentials; that identification is not made.
"""

import jax
import jax.numpy as jnp
import numpy as onp
from jax.scipy.linalg import expm

#: intrinsic circuit constants, in input-interval time units
C_MEM = 1.0
G_LEAK = 2.0 / 15.0          # g_L = h = g
H_COUPLE = 2.0 / 15.0
G_S = G_LEAK + H_COUPLE      # 4/15
G_D0 = G_LEAK + H_COUPLE     # 4/15 at rest
#: frozen physical gamma, used to map the normalized S5 residual to physical
GAMMA0 = 5.0

#: declared dimensionless calibration (see module docstring)
MOD_SCALE = 1.0
DRIVE_REF = 1.0


def validate_reference(tol=1e-12):
    """The intrinsic constants must reproduce the contract tuple exactly."""
    q = 0.0
    c = coefficients_from_conductance(onp.asarray(q))
    got = dict(T=float(c["T"]), tau_d=float(c["tau_d"]),
               gamma=float(c["gamma"]), M=float(c["M"]), rho=float(c["rho"]))
    want = dict(T=5.0, tau_d=3.75, gamma=5.0, M=18.75, rho=0.75)
    bad = {k: (got[k], want[k]) for k in want if abs(got[k] - want[k]) > tol}
    if bad:
        raise ValueError(f"intrinsic constants do not reproduce the contract "
                         f"reference: {bad}")
    return got


def coefficients_from_conductance(q):
    """Tied coefficients from the dendritic conductance. q >= 0, any shape.

    These are TIED, not independent knobs: every one of them is a function of
    the single varying quantity G_d = G_d0 + q.
    """
    xp = jnp if isinstance(q, jnp.ndarray) else onp
    G_d = G_D0 + q
    tau_d = C_MEM / G_d
    kappa = G_S - H_COUPLE ** 2 / G_d
    T = C_MEM / kappa
    gamma = G_S * tau_d / kappa
    M = T * tau_d
    return dict(G_d=G_d, tau_d=tau_d, kappa=kappa, T=T, gamma=gamma, M=M,
                rho=M / (gamma * T), xp=xp)


def conductance_from_input(b, x, mod_scale=MOD_SCALE, drive_ref=DRIVE_REF):
    """q_{k,p} >= 0 from the SAME synaptic weights that carry the drive.

    `b` is the clock-absorbed complex input coupling (P, H); `x` is (L, H).
    Returns (L, P), real, nonnegative, bounded by mod_scale * G_D0.

    Depends only on the layer input at that token, so every per-token block is
    precomputable and the associative scan is retained.
    """
    drive = jnp.abs(x @ b.T)                       # (L, P), real >= 0
    return mod_scale * G_D0 * drive / (drive_ref + drive)


def adaptive_generator(a, b, q):
    """Per-token physical generator in z = (s, s') coordinates.

    Returns A (L,P,2,2), Bx (L,P,2,H), and the impulsive jump coefficient
    d_jump (L,P,H) multiplying the input STEP at each token boundary.
    """
    c = coefficients_from_conductance(q)                  # each (L,P)
    T, gamma, M = c["T"], c["gamma"], c["M"]
    J = -a                                                 # (P,) complex
    gJ = GAMMA0 * J[None, :]                               # (1,P)
    zero = jnp.zeros_like(gJ + 0j)
    one = jnp.ones_like(gJ + 0j)
    a11 = zero
    a12 = one
    a21 = -gJ / M
    a22 = -(gamma + T * gJ) / M
    A = jnp.stack([jnp.stack([a11, a12], axis=-1),
                   jnp.stack([a21, a22], axis=-1)], axis=-2)   # (L,P,2,2)
    drive = (GAMMA0 / M)[..., None] * b[None, :, :]            # (L,P,H)
    Bx = jnp.stack([jnp.zeros_like(drive), drive], axis=-2)    # (L,P,2,H)
    d_jump = (T * GAMMA0 / M)[..., None] * b[None, :, :]       # (L,P,H)
    return A, Bx, d_jump, c


def adaptive_zoh(A, Bx):
    """Exact per-token ZOH by a 4x4 augmented exponential per (token, mode).

    Width independent: the augmented block is [[A, I],[0,0]], so A_bar = e^A
    and Phi = int_0^1 e^{A s} ds, then B_bar = Phi @ Bx.
    """
    L, P = A.shape[0], A.shape[1]
    aug = jnp.zeros(A.shape[:2] + (4, 4), dtype=A.dtype)
    aug = aug.at[..., :2, :2].set(A)
    eye = jnp.broadcast_to(jnp.eye(2, dtype=A.dtype), A.shape[:2] + (2, 2))
    aug = aug.at[..., :2, 2:].set(eye)
    E = jax.vmap(jax.vmap(expm))(aug)
    A_bar = E[..., :2, :2]
    Phi = E[..., :2, 2:]
    B_bar = jnp.einsum("lpij,lpjh->lpih", Phi, Bx)
    return A_bar, B_bar


def adaptive_scan(A_bar, B_bar, d_jump, x, z0=None, reset_mask=None):
    """z_k = A_bar_k (z_{k-1} + jump_k) + B_bar_k x_k, by a block scan.

    jump_k = [0 ; d_jump_k (x_k - x_{k-1})] is the impulsive contribution of
    the prospective x' term at the token boundary, with x_{-1} = 0.
    """
    from .gp_second_order import block_binary_operator
    L, P = A_bar.shape[0], A_bar.shape[1]
    x_prev = jnp.concatenate([jnp.zeros_like(x[:1]), x[:-1]], axis=0)
    dx = x - x_prev                                            # (L,H)
    jump_p = jnp.einsum("lph,lh->lp", d_jump, dx)              # (L,P)
    jump = jnp.stack([jnp.zeros_like(jump_p), jump_p], axis=-1)  # (L,P,2)
    drive = jnp.einsum("lpih,lh->lpi", B_bar, x)               # (L,P,2)
    b_elems = (A_bar @ jump[..., None])[..., 0] + drive
    A_elems = A_bar
    if reset_mask is not None:
        keep = (~jnp.asarray(reset_mask, dtype=bool))[:, None, None, None]
        A_elems = A_elems * keep.astype(A_bar.dtype)
    if z0 is not None:
        b_elems = b_elems.at[0].add((A_elems[0] @ z0[..., None])[..., 0])
    _, zs = jax.lax.associative_scan(block_binary_operator, (A_elems, b_elems))
    return zs                                                   # (L,P,2)


def adaptive_scan_sequential(A_bar, B_bar, d_jump, x, z0=None,
                             reset_mask=None):
    """Same recurrence by lax.scan; differentiable, for scan-order checks."""
    L, P = A_bar.shape[0], A_bar.shape[1]
    x_prev = jnp.concatenate([jnp.zeros_like(x[:1]), x[:-1]], axis=0)
    dx = x - x_prev
    z_init = jnp.zeros((P, 2), dtype=A_bar.dtype) if z0 is None else z0
    mask = (jnp.zeros(L, dtype=bool) if reset_mask is None
            else jnp.asarray(reset_mask, dtype=bool))

    def step(z, inp):
        Ab, Bb, dj, xk, dxk, drop = inp
        z = jnp.where(drop, jnp.zeros_like(z), z)
        jp = jnp.einsum("ph,h->p", dj, dxk)
        z = z + jnp.stack([jnp.zeros_like(jp), jp], axis=-1)
        z = (Ab @ z[..., None])[..., 0] + jnp.einsum("pih,h->pi", Bb, xk)
        return z, z

    _, zs = jax.lax.scan(step, z_init,
                         (A_bar, B_bar, d_jump, x, dx, mask))
    return zs


def reference_unreduced_trajectory(a_p, b_p, x_fn, dx_fn, q_fn, t_span,
                                   rtol=1e-10, atol=1e-12, n_eval=200):
    """Integrate the UNREDUCED two-compartment circuit for ONE mode.

    Independent reference for the cluster check: this integrates level 1 with a
    time-varying conductance and the derived synaptic current, NOT the reduced
    model. Returns (t, u) with u the somatic voltage, to be compared with the
    implemented s.

        c u' = -G_s u + h v
        c v' = -G_d(t) v + h u + I_d(t),     I_d = (G_d/h) b_src
        b_src = kappa (u - R) + c (u' - R'),   R = gamma0 (J u - beta x)

    using T kappa = c so that T*b_bar = c (u - R) exactly.
    """
    from scipy.integrate import solve_ivp
    J = -a_p

    def rhs(t, y):
        u = y[0] + 1j * y[2]
        v = y[1] + 1j * y[3]
        q = float(q_fn(t))
        G_d = G_D0 + q
        kappa = G_S - H_COUPLE ** 2 / G_d
        xt = onp.asarray(x_fn(t))
        dxt = onp.asarray(dx_fn(t))
        du = (-G_S * u + H_COUPLE * v) / C_MEM
        R = GAMMA0 * (J * u - onp.dot(b_p, xt))
        dR = GAMMA0 * (J * du - onp.dot(b_p, dxt))
        b_src = kappa * (u - R) + C_MEM * (du - dR)
        I_d = (G_d / H_COUPLE) * b_src
        dv = (-G_d * v + H_COUPLE * u + I_d) / C_MEM
        return onp.array([du.real, dv.real, du.imag, dv.imag])

    ts = onp.linspace(t_span[0], t_span[1], n_eval)
    sol = solve_ivp(rhs, t_span, onp.zeros(4), t_eval=ts, rtol=rtol, atol=atol)
    return sol.t, sol.y[0] + 1j * sol.y[2]


def reference_discrete_via_unreduced(a_p, b_p, x, q, rtol=1e-11, atol=1e-13):
    """INDEPENDENT reference for one mode: integrate the UNREDUCED circuit.

    Segment by segment in the physical (u, v) coordinates, with BOTH the input
    and the conductance changing between tokens:

      * at token k apply the impulsive prospective current, which shows up as a
        jump in the dendritic voltage, Delta v = c * Delta s' / h, with
        Delta s' = T_k gamma0 (beta . Delta x) / M_k;
      * then integrate  c u' = -G_s u + h v,
                        c v' = -G_d(q_k) v + h u + I_d  over one unit interval
        with x held, where I_d = (G_d/h)[kappa (u - R) + c (u' - R')],
        R = gamma0 (J u - beta.x), R' = gamma0 J u' (x held, so x' = 0).

    Returns s_k = u at the END of each interval, to compare against the
    implemented discrete recurrence. This is scipy on level 1, versus a matrix
    exponential on the reduced 2-state model: an independent path, not the same
    code twice. A frozen matrix exponential alone would not test adaptation.
    """
    from scipy.integrate import solve_ivp
    J = -complex(a_p)
    bp = onp.asarray(b_p)
    x = onp.asarray(x)
    q = onp.asarray(q)
    L = x.shape[0]
    out = onp.zeros(L, dtype=complex)
    u = 0.0 + 0j
    v = 0.0 + 0j
    x_prev = onp.zeros_like(x[0])
    for k in range(L):
        G_d = G_D0 + float(q[k])
        kappa = G_S - H_COUPLE ** 2 / G_d
        tau_d = C_MEM / G_d
        T = C_MEM / kappa
        M = T * tau_d
        dx = x[k] - x_prev
        ds = T * GAMMA0 * onp.dot(bp, dx) / M
        v = v + C_MEM * ds / H_COUPLE          # impulsive current -> v jump
        drive = onp.dot(bp, x[k])

        def rhs(t, y):
            uu = y[0] + 1j * y[2]
            vv = y[1] + 1j * y[3]
            du = (-G_S * uu + H_COUPLE * vv) / C_MEM
            R = GAMMA0 * (J * uu - drive)
            dR = GAMMA0 * (J * du)             # x held within the interval
            b_src = kappa * (uu - R) + C_MEM * (du - dR)
            I_d = (G_d / H_COUPLE) * b_src
            dv = (-G_d * vv + H_COUPLE * uu + I_d) / C_MEM
            return onp.array([du.real, dv.real, du.imag, dv.imag])

        sol = solve_ivp(rhs, (0.0, 1.0),
                        onp.array([u.real, v.real, u.imag, v.imag]),
                        rtol=rtol, atol=atol, dense_output=False)
        y = sol.y[:, -1]
        u = y[0] + 1j * y[2]
        v = y[1] + 1j * y[3]
        out[k] = u
        x_prev = x[k]
    return out
