"""Deterministic initialization calibration. Runs on the cluster, in float64.

The new response has a sub-token decay time at the original reference while
the task asks for recall over tens of intervals, so BOTH regimes are declared
before any result: a short configuration A and a long configuration B. This is
an initialization design decision, not a rescue run added after seeing scores.

TARGET. Match the single isolated unit write, queried one idle interval later,
to the OLD prospective reference:

    beta_* = 1 - F11_ref + (1 - exp(-1/(3/4))) F21_ref

with `F_ref` the `(e, z)` matrix exponential at `nu = 4/3`, `rho = 3/4`,
`tau = 3/4`, `h = 1`. The full-precision computed value is used, never a
rounded decimal. It matches ONE scalar observable at zero initial carry - not
the transfer function, and not the response under interfering writes, so
matched initial performance is NOT implied and is reported separately.

ALGORITHM, declared before execution: scan the positive grid
`exp(linspace(-12, 12, 257))` in ascending order; take the FIRST consecutive
bracket where `amplitude - beta_*` goes from nonpositive to nonnegative; bisect
inside it to absolute observable error <= 1e-10 in float64. An exactly matching
grid point is accepted. If no bracket exists, or the response or its derivative
is non-finite, STOP before training and report an initialization-design
obstruction - do not move the target or the interval.

The same policy, target and tolerance are applied to the TSS finite-adaptation
reference, solving for `q = 1/tau_m` at each declared `epsilon/tau_m` ratio
through its OWN `(W, P)` observable `1 - F11 - (h/M) F21`. The ideal
equilibrium reference is deliberately NOT calibrated: its equality constraint
fixes the write strength at one.

This is a declared numerical procedure. It is not a claim that the inertial or
TSS response root is globally unique.
"""

import math

import numpy as onp

GRID_LO, GRID_HI, GRID_N = -12.0, 12.0, 257
TOL = 1e-10
NU_REF, TAU_REF, RHO_REF = 4.0 / 3.0, 0.75, 0.75
H = 1.0


def _expm2_f64(G, h=H):
    """Closed-form 2x2 exponential in float64, for calibration only.

    Calibration is a host-side root solve on a scalar observable, so it does
    not need to be differentiable; the EXECUTED path uses the differentiable
    `dynamics.expm2` and never this routine.
    """
    G = onp.asarray(G, dtype=onp.float64)
    tr = G[0, 0] + G[1, 1]
    mu = ((G[0, 0] - G[1, 1]) ** 2 + 4.0 * G[0, 1] * G[1, 0]) / 4.0
    m = mu * h * h
    if abs(m) < 1e-12:
        ch, sh = 1.0 + m / 2.0, 1.0 + m / 6.0
    elif m > 0:
        r = math.sqrt(m); ch, sh = math.cosh(r), math.sinh(r) / r
    else:
        r = math.sqrt(-m); ch, sh = math.cos(r), math.sin(r) / r
    N = G - (tr / 2.0) * onp.eye(2)
    return math.exp(h * tr / 2.0) * (ch * onp.eye(2) + h * sh * N)


def prospective_G(nu, tau, rho, w=1.0):
    return onp.array([[-nu * w, -1.0 / tau],
                      [-nu * (1.0 - rho) * w, -1.0 / tau]], dtype=onp.float64)


def inertial_G(eta, tau, w=1.0):
    return onp.array([[0.0, -1.0 / tau],
                      [eta * w, -1.0 / tau]], dtype=onp.float64)


def tss_G(q, ratio, w=1.0):
    """The TSS reference generator in `(e, p)` from the rate `q = 1/tau_m`.

    `tau_m = 1/q`, `epsilon = ratio/q`, hence `M = tau_m*epsilon = ratio/q^2`
    and `T = tau_m + epsilon = (1+ratio)/q`. `gamma = 0`: this reference is
    outside the generalized candidate's admissible sector, by construction.
    """
    tau_m, eps = 1.0 / q, ratio / q
    M, T = tau_m * eps, tau_m + eps
    return onp.array([[-T * w / M, 1.0 / M], [-w, 0.0]], dtype=onp.float64), M


def tss_amplitude(q, ratio, h=H):
    """`1 - F11 - (h/M) F21`, the SAME observable in `(W, P)` coordinates.

    The sign differs from the `(W, Z)` formula because the auxiliary and its
    idle dynamics differ; the generalized formula is not reused here.
    """
    G, M = tss_G(q, ratio)
    F = _expm2_f64(G, h)
    return 1.0 - F[0, 0] - (h / M) * F[1, 0]


def amplitude(G, tau, h=H):
    """`1 - F11 + (1 - exp(-h/tau)) F21`, the single-write query observable."""
    F = _expm2_f64(G, h)
    return 1.0 - F[0, 0] + (1.0 - math.exp(-h / tau)) * F[1, 0]


def beta_star():
    """The reference amplitude, at full precision."""
    F = _expm2_f64(prospective_G(NU_REF, TAU_REF, RHO_REF, 1.0), H)
    return (1.0 - F[0, 0]
            + (1.0 - math.exp(-H / TAU_REF)) * F[1, 0]), F


def solve_rate(kind, tau, rho=None, target=None, tol=TOL, ratio=None):
    """Bracket on the declared grid, then bisect. Returns the rate and a record.

    `kind` is "prospective" (solves for `nu`), "inertial" (solves for `eta`) or
    "tss" (solves for `q = 1/tau_m` at a fixed `epsilon/tau_m` ratio). The
    SAME ascending grid, first-bracket and bisection policy applies to all
    three; the target `beta_*` is never replaced. Raises with an explicit
    obstruction message rather than relaxing anything.
    """
    target = beta_star()[0] if target is None else target

    def f(rate):
        if kind == "tss":
            a = tss_amplitude(rate, ratio)
        else:
            G = (prospective_G(rate, tau, rho) if kind == "prospective"
                 else inertial_G(rate, tau))
            a = amplitude(G, tau)
        if not onp.isfinite(a):
            raise FloatingPointError(
                f"non-finite calibration response at {kind} rate {rate}, "
                f"tau={tau}, rho={rho}: initialization-design obstruction")
        return a - target

    grid = onp.exp(onp.linspace(GRID_LO, GRID_HI, GRID_N))
    vals = [f(g) for g in grid]
    for i in range(GRID_N - 1):
        if vals[i] == 0.0:
            return float(grid[i]), dict(kind=kind, tau=tau, rho=rho,
                                        ratio=ratio,
                                        target=target, rate=float(grid[i]),
                                        bracket=[float(grid[i])] * 2,
                                        observable_error=0.0,
                                        exact_grid_point=True, iterations=0)
        if vals[i] <= 0.0 <= vals[i + 1]:
            lo, hi = float(grid[i]), float(grid[i + 1])
            flo = vals[i]
            it = 0
            while it < 200:
                mid = 0.5 * (lo + hi)
                fm = f(mid)
                if abs(fm) <= tol:
                    return mid, dict(kind=kind, tau=tau, rho=rho,
                                     ratio=ratio,
                                     target=target, rate=mid,
                                     bracket=[float(grid[i]),
                                              float(grid[i + 1])],
                                     observable_error=abs(fm),
                                     exact_grid_point=False, iterations=it)
                if (flo <= 0.0) == (fm <= 0.0):
                    lo, flo = mid, fm
                else:
                    hi = mid
                it += 1
            raise FloatingPointError(
                f"bisection did not reach {tol} for {kind} tau={tau}: "
                f"initialization-design obstruction")
    raise FloatingPointError(
        f"no sign-changing bracket on the declared grid for {kind} "
        f"tau={tau}, rho={rho}, ratio={ratio}: initialization-design "
        f"obstruction")


def configurations():
    """The fourteen declared development slots: two per family, seven families.

    Five original families plus the two ordinary-prospective references of the
    16 September 2026 amendment. The TSS reference is calibrated to the SAME
    `beta_*` on the SAME grid/bracket/bisection policy; the ideal-equilibrium
    reference is not calibrated at all, because its equality constraint fixes
    the write strength at one, which is reported openly rather than matched.
    """
    bs, F_ref = beta_star()
    out = {"beta_star": bs, "F_ref": F_ref.tolist(), "slots": {}, "records": []}

    for tag, tau in (("A", 0.75), ("B", 32.0)):
        nu, rec = solve_rate("prospective", tau, rho=RHO_REF, target=bs)
        out["records"].append(rec)
        out["slots"][f"adaptive_prospective/{tag}"] = dict(
            rule="adaptive_prospective", config=tag, lr=0.003,
            raw_nu=math.log(nu), raw_tau=math.log(tau),
            raw_rho=math.log(RHO_REF / (1.0 - RHO_REF)),
            nu=nu, tau=tau, rho=RHO_REF)
        eta, rec = solve_rate("inertial", tau, target=bs)
        out["records"].append(rec)
        out["slots"][f"adaptive_inertial/{tag}"] = dict(
            rule="adaptive_inertial", config=tag, lr=0.003,
            raw_eta=math.log(eta), raw_tau=math.log(tau), eta=eta, tau=tau)

    # --- amendment reference 1: TSS finite adaptation, two ratio slots -----
    for tag, ratio in (("A", 0.1), ("B", 0.5)):
        q, rec = solve_rate("tss", None, target=bs, ratio=ratio)
        out["records"].append(rec)
        tau_m, eps = 1.0 / q, ratio / q
        out["slots"][f"tss_prospective/{tag}"] = dict(
            rule="tss_prospective", config=tag, lr=0.003,
            raw_tau_m=math.log(tau_m),
            raw_ratio=math.log(ratio / (1.0 - ratio)),
            q=q, tau_m=tau_m, epsilon=eps, ratio=ratio,
            M=tau_m * eps, T=tau_m + eps, gamma=0.0)

    eta0 = -math.log(1.0 - bs)
    for tag, lr in (("A", 0.003), ("B", 0.01)):
        out["slots"][f"adaptive_delta/{tag}"] = dict(
            rule="adaptive_delta", config=tag, lr=lr,
            raw_eta=math.log(eta0), eta=eta0)
        # --- amendment reference 2: ideal equilibrium. NO beta_* matching:
        # the exact zero-residual constraint fixes the write strength at one.
        out["slots"][f"ideal_projection/{tag}"] = dict(
            rule="ideal_projection", config=tag, lr=lr, beta=1.0,
            calibrated=False,
            note="write strength is fixed at 1 by the equality constraint; "
                 "there is no initial-write matching to beta_star")
        for rule in ("gated_delta", "momentum_delta"):
            out["slots"][f"{rule}/{tag}"] = dict(rule=rule, config=tag, lr=lr)

    # the short prospective slot must recover the reference rate exactly
    nu_a = out["slots"]["adaptive_prospective/A"]["nu"]
    out["reference_recovery"] = dict(
        nu=nu_a, reference=NU_REF, absolute_error=abs(nu_a - NU_REF),
        tolerance=1e-8, passed=bool(abs(nu_a - NU_REF) < 1e-8))
    return out
