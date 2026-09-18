"""Pure-Python reference forms of the write-path updates on the MDN memory.

NO JAX, NO NumPy: every function works on nested lists of any number type,
so `fractions.Fraction` gives EXACT arithmetic. Used by the exact identity
tests and, converted to float64, as the independent reference for the
production rollouts. Audit: docs/MDN_NESTEROV_QHM_AUDIT.md.

Orientation is the repository's (`experiments/nested_memory/dynamics.py`):
the fast weight W is d_v x d_k and predicts v as W k, i.e. W = S^T of the MDN
paper, whose S, M are d_k x d_v. The masked residual

    R(P) = m (P k - v) k^T

is the gradient of 1/2 ||P k - v||^2 with respect to P, i.e. (grad_S L)^T.
Every update below applies the NATIVE decay first, Wbar_t = alpha_t W_(t-1),
as MDN Eqs. (4)-(5) do.

    native      U_t = mu U + eta R(Wbar),          W_t = Wbar - beta U_t
    nesterov    L_t = Wbar - beta mu U             (lookahead, c_t = beta_t mu_t)
                U_t = mu U + eta R(L_t),           W_t = Wbar - beta U_t
    qhm         U_t = mu U + eta R(Wbar),
                W_t = Wbar - beta [nu U_t + (1 - nu) eta R(Wbar)]
    two_tap     U_t = mu U + eta [(1 + kappa) R_t - kappa R_(t-1)],
                W_t = Wbar - beta U_t              (ordinary.py)
    filtered    y_t = a y - b y_prev + c R_t - d R_(t-1),
                U_t = mu U + eta y_t, W_t = Wbar - beta U_t   (filtered.py)
"""


# ------------------------------------------------------------ list algebra --
def zeros(d_v, d_k, z=0):
    return [[z for _ in range(d_k)] for _ in range(d_v)]


def lin(*terms):
    """sum_i c_i X_i for matrices X_i (lists of lists) and scalars c_i."""
    c0, X0 = terms[0]
    out = [[c0 * x for x in row] for row in X0]
    for c, X in terms[1:]:
        out = [[o + c * x for o, x in zip(orow, xrow)]
               for orow, xrow in zip(out, X)]
    return out


def matvec(P, k):
    return [sum(p * kk for p, kk in zip(row, k)) for row in P]


def outer(a, b):
    return [[x * y for y in b] for x in a]


def residual(P, k, v, m):
    """m (P k - v) k^T."""
    e = [pk - vv for pk, vv in zip(matvec(P, k), v)]
    return [[m * x for x in row] for row in outer(e, k)]


# ------------------------------------------------------------------- steps --
def native_step(carry, tok, g):
    W, U = carry
    a, b, mu, eta = g
    Wb = lin((a, W))
    R = residual(Wb, *tok)
    U2 = lin((mu, U), (eta, R))
    return (lin((1, Wb), (-b, U2)), U2)


def nesterov_step(carry, tok, g):
    """Literal Nesterov: the residual at the point the momentum part of THIS
    token's step reaches, L_t = Wbar_t - beta_t mu_t U_(t-1)."""
    W, U = carry
    a, b, mu, eta = g
    Wb = lin((a, W))
    L = lin((1, Wb), (-(b * mu), U))
    R = residual(L, *tok)
    U2 = lin((mu, U), (eta, R))
    return (lin((1, Wb), (-b, U2)), U2)


def lookahead(carry, g):
    """The Nesterov lookahead of the NEXT token, given its gates."""
    W, U = carry
    a, b, mu, _ = g
    return lin((a, W), (-(b * mu), U))


def qhm_step(carry, tok, g, nu):
    W, U = carry
    a, b, mu, eta = g
    Wb = lin((a, W))
    R = residual(Wb, *tok)
    U2 = lin((mu, U), (eta, R))
    step = lin((nu, U2), ((1 - nu) * eta, R))
    return (lin((1, Wb), (-b, step)), U2)


def qhm_raw_step(carry, tok, g, nu, scale):
    """QHM with an explicit step scale: W_t = Wbar - scale beta [nu U +
    (1 - nu) eta R] (used only for the exact bridge to the two-tap)."""
    W, U = carry
    a, b, mu, eta = g
    Wb = lin((a, W))
    R = residual(Wb, *tok)
    U2 = lin((mu, U), (eta, R))
    step = lin((nu, U2), ((1 - nu) * eta, R))
    return (lin((1, Wb), (-(scale * b), step)), U2)


def two_tap_step(carry, tok, g, kappa):
    W, U, Rp = carry
    a, b, mu, eta = g
    Wb = lin((a, W))
    R = residual(Wb, *tok)
    Rpros = lin((1 + kappa, R), (-kappa, Rp))
    U2 = lin((mu, U), (eta, Rpros))
    return (lin((1, Wb), (-b, U2)), U2, R)


def filter_coefficients(M, gamma, T, h=1):
    """The executed coefficient form of filtered.py (exact in Fractions)."""
    A = M + h * (gamma + T)
    return ((2 * M + h * (gamma + T) - h * h) / A, M / A,
            (h * h + h * T) / A, (h * T) / A)


def filtered_step(carry, tok, g, coeff):
    W, U, y, yp, Rp = carry
    a, b, mu, eta = g
    ca, cb, cc, cd = coeff
    Wb = lin((a, W))
    R = residual(Wb, *tok)
    y2 = lin((ca, y), (-cb, yp), (cc, R), (-cd, Rp))
    U2 = lin((mu, U), (eta, y2))
    return (lin((1, Wb), (-b, U2)), U2, y2, y, R)


# ---------------------------------------------------------------- rollouts --
def run(step, carry0, tokens, gates, *extra):
    """Carries after each token; `tokens[t] = (k, v, m)`, `gates[t] =
    (alpha, beta, mu, eta)`. Strictly causal: carry t uses tokens <= t."""
    out, c = [], carry0
    for tok, g in zip(tokens, gates):
        c = step(c, tok, g, *extra)
        out.append(c)
    return out


def readout(W, k):
    """What the shell reads at a token: W k (before the readout matrix)."""
    return matvec(W, k)


# ----------------------------------------------------- frozen-token forms --
def key_aligned_2x2(rule, a, b, mu, eta, nu=None):
    """Frozen-token transition of (w, u) along a unit key, with m = 1.
    Rows are (w_t, u_t) in terms of (w_(t-1), u_(t-1))."""
    q = b * eta
    if rule == "native":
        return [[a * (1 - q), -b * mu], [a * eta, mu]]
    if rule == "nesterov":
        return [[a * (1 - q), -b * mu * (1 - q)], [a * eta, mu * (1 - q)]]
    if rule == "qhm":
        return [[a * (1 - q), -nu * b * mu], [a * eta, mu]]
    raise ValueError(rule)


def jury(A):
    """(1 - tr + det, 1 + tr + det, 1 - det); all > 0 iff Schur stable."""
    tr = A[0][0] + A[1][1]
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    return (1 - tr + det, 1 + tr + det, 1 - det)
