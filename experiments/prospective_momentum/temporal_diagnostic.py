"""Controlled mechanism diagnostic: equal initial correction, different
subsequent dynamics. Protocol s5 (docs/PROSPECTIVE_TEMPORAL_RESPONSE_PROTOCOL.md).

For h = 1 and zero processing histories the executed coefficient form
y_t = a y_(t-1) - b y_(t-2) + c R_t - d R_(t-1) gives, for a unit residual
impulse at t = 0:

    TSS, T = 1                     a = 0, b = 0,   c = 2, d = 1
                                   y = (2, -1, 0, 0, ...)
    generalized, M = 1/4, gamma = 0, T = 1/2
                                   A = 3/4, a = 0, b = 1/3, c = 2, d = 2/3
                                   y = (2, -2/3, -2/3, 2/9, 2/9, -2/27, ...)

Both are strictly stable with unit DC gain and the SAME immediate amplitude
c = 2. These are DIAGNOSTIC settings only; they are never imposed on the
trained arms.

Two separate checks, neither of which alone establishes a task improvement:

1. OPEN LOOP: the production coefficient formation and update applied to a
   PRESCRIBED residual impulse, against the exact rational impulse response.
2. CLOSED LOOP: the actual Momentum rollout on a common frozen backbone (the
   development source), with common incoming W and U taken from the same
   episode, identical events and gates, and explicitly matched zero processing
   state. Later residuals depend on W. The first-write change in W is verified
   to agree as a MATRIX (not a norm or a coefficient) between TSS and the
   generalized setting before later revised and untouched readouts are
   compared.
"""

from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as onp

from experiments.nested_memory import dynamics as NMD
from experiments.nested_memory.task import FAMILIES, N_VALUES

from . import filtered as FL
from . import model as PM
from . import temporal_task as TT
from . import tss_containment as TC

TRAJ32 = TC.TRAJ32
#: (name, (M, gamma, T)); the native point is a reference, not a contender
SETTINGS = (("native_point", (0.0, 1.0, 0.0)),
            ("tss_T1", (0.0, 0.0, 1.0)),
            ("generalized_M1q_T1h", (0.25, 0.0, 0.5)))
CONTRAST = ("tss_T1", "generalized_M1q_T1h")
#: the values stated in the brief, checked EXACTLY against the recursion
DECLARED_IMPULSE = {"tss_T1": (Fraction(2), Fraction(-1), Fraction(0)),
                    "generalized_M1q_T1h": (Fraction(2), Fraction(-2, 3),
                                            Fraction(-2, 3))}
N_IMPULSE = 18
#: the closed-loop diagnostic uses each episode's delay-16 block
DIAG_DELAY = 16
SUFFIX_LEN = DIAG_DELAY + 2


def exact_coefficients(M, gamma, T, h=1):
    M, gamma, T, h = (Fraction(x) for x in (M, gamma, T, h))
    A = M + h * (gamma + T)
    return dict(A=A, a=(2 * M + h * (gamma + T) - h * h) / A, b=M / A,
                c=(h * h + h * T) / A, d=h * T / A)


def exact_impulse(M, gamma, T, n=N_IMPULSE):
    """Exact rational response to R = (1, 0, 0, ...), zero histories."""
    k = exact_coefficients(M, gamma, T)
    y1 = y2 = Fraction(0)
    r_prev = Fraction(0)
    out = []
    for t in range(n):
        r = Fraction(1 if t == 0 else 0)
        y = k["a"] * y1 - k["b"] * y2 + k["c"] * r - k["d"] * r_prev
        out.append(y)
        y2, y1, r_prev = y1, y, r
    return out


def leaves(M, gamma, T, dtype):
    return {"fil_M": jnp.full((1,), M, dtype=dtype),
            "fil_gamma": jnp.full((1,), gamma, dtype=dtype),
            "fil_T": jnp.full((1,), T, dtype=dtype)}


def open_loop(dtype=jnp.float32):
    """Production coefficient formation and production update on a prescribed
    impulse, versus the exact rational response. Returns (failures, rows)."""
    fails, rows = [], {}
    for name, (M, gamma, T) in SETTINGS:
        ex = exact_impulse(M, gamma, T)
        if name in DECLARED_IMPULSE and tuple(
                ex[:len(DECLARED_IMPULSE[name])]) != DECLARED_IMPULSE[name]:
            fails.append(f"{name}: exact recursion {ex[:3]} differs from the "
                         f"declared {DECLARED_IMPULSE[name]}")
        coeff = FL.coefficients(leaves(M, gamma, T, dtype))
        z = jnp.zeros((1, 1), dtype=dtype)
        one = jnp.ones((1, 1), dtype=dtype)
        W = U = y = y_prev = r_prev = z
        got = []
        for t in range(N_IMPULSE):
            R = one if t == 0 else z
            W, U, y_new, y_prev, r_prev = FL.filtered_update(
                W, U, y, y_prev, r_prev, jnp.asarray(0.0, dtype),
                jnp.asarray(1.0, dtype), jnp.asarray(1.0, dtype), coeff, R)
            y = y_new
            got.append(float(onp.asarray(y)[0, 0]))
        errs = [abs(g - float(e)) for g, e in zip(got, ex)]
        bad = [t for t, (g, e, er) in enumerate(zip(got, ex, errs))
               if not (onp.isfinite(g) and er <= TRAJ32 * max(abs(float(e)),
                                                              1.0))]
        if bad:
            fails.append(f"{name}: open-loop response differs at steps {bad}")
        rows[name] = dict(
            coefficients=dict(M=M, gamma=gamma, T=T),
            exact_coefficients={k: str(v) for k, v in
                                exact_coefficients(M, gamma, T).items()},
            executed_coefficients=FL.coefficient_values(coeff),
            exact_response=[str(v) for v in ex],
            executed_response=got, max_abs_error=float(max(errs)))
    return fails, rows


# ------------------------------------------------------------- closed loop --
def _arrays(out):
    """Only array outputs may leave a compiled function. The processing-state
    summary is carried through so the diagnostic can check it (review R1)."""
    return dict(logits=out["logits"], W_trace=out["W_trace"],
                U_trace=out["U_trace"], proc_max_abs=out["proc_max_abs"],
                coeff=out["coeff"])


# ------------------------------------------- finite-first guards (review R1) -
#: arrays every returned diagnostic rollout must carry, in production dtype
PAYLOAD_KEYS = ("logits", "W_trace", "U_trace", "proc_max_abs")


def finite_failures(name, arr):
    """A named failure for an empty or non-finite array; [] otherwise."""
    a = onp.asarray(arr)
    if a.size == 0:
        return [f"{name}: empty"]
    if not onp.all(onp.isfinite(a)):
        return [f"{name}: non-finite values"]
    return []


def payload_failures(label, out, dtype):
    """Production-dtype and finiteness checks of EVERY returned rollout array
    and every returned coefficient copy, before any host float64 analysis."""
    fails = []
    want = onp.dtype(dtype)
    for key in PAYLOAD_KEYS:
        if key not in out:
            fails.append(f"{label}: missing {key}")
            continue
        a = onp.asarray(out[key])
        if a.dtype != want:
            fails.append(f"{label}: {key} dtype {a.dtype} is not the "
                         f"production dtype {want}")
        fails += finite_failures(f"{label}: {key}", a)
    coeff = out.get("coeff") or {}
    for k in FL.COEFF_NAMES:
        if k not in coeff:
            fails.append(f"{label}: missing coefficient {k}")
            continue
        a = onp.asarray(coeff[k])
        if a.dtype != want:
            fails.append(f"{label}: coefficient {k} dtype {a.dtype} is not "
                         f"the production dtype {want}")
        fails += finite_failures(f"{label}: coefficient {k} (all copies)", a)
    return fails


def curve_failures(label, curves):
    """Non-empty cells and finite values in every reported curve."""
    fails = []
    for g, c in curves.items():
        if c.get("n", 1) <= 0:
            fails.append(f"{label}/{g}: empty cell")
            continue
        for k, v in c.items():
            if k != "n":
                fails += finite_failures(f"{label}/{g}/{k}", v)
    return fails


@jax.jit
def _traced_full(p, eps):
    return jax.vmap(lambda e: _arrays(PM.rollout(FL.FILTERED, p, e,
                                                 trace=True)))(eps)


@jax.jit
def _native_logits(p, eps):
    return jax.vmap(lambda e: PM.rollout("momentum_delta", p, e)["logits"])(
        eps)


@jax.jit
def _traced_suffix(p, eps, W0, U0):
    def one(e, w, u):
        z = jnp.zeros_like(w)
        return _arrays(PM.rollout(FL.FILTERED, p, e, carry0=(w, u, z, z, z),
                                  trace=True))
    return jax.vmap(one)(eps, W0, U0)


def _rel(a, b):
    a, b = onp.asarray(a, onp.float64), onp.asarray(b, onp.float64)
    if not (onp.all(onp.isfinite(a)) and onp.all(onp.isfinite(b))):
        return float("inf")
    n = float(onp.linalg.norm(b))
    return float(onp.linalg.norm(a - b)) / (n if n > 0 else 1.0)


def _softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = onp.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def closed_loop(native_p, batch):
    """Returns (failures, report). `native_p` is the frozen development
    source; `batch` a temporal-task batch from the diagnostic stream. Every
    returned state, logit, coefficient copy and derived readout is checked
    finite first; the first failing stage returns named failures instead of
    continuing with invalid evidence (review R1)."""
    dt = native_p["A_log"].dtype
    inputs = {k: jnp.asarray(batch[k]) for k in TT.MODEL_INPUTS}
    nat_pt = TC.native_point_tree(native_p)

    # the incoming state comes from the processing law at its exact native
    # point, whose agreement with the native rule is verified right here
    full = _traced_full(nat_pt, inputs)
    ref = _native_logits(native_p, inputs)
    fails = payload_failures("native-point full rollout", full, dt)
    if onp.asarray(ref).dtype != onp.dtype(dt):
        fails.append(f"native rule logits dtype {onp.asarray(ref).dtype}")
    fails += finite_failures("native rule logits", ref)
    if fails:
        return fails, dict(stage="incoming rollouts")
    rec_err = _rel(full["logits"], ref)
    if not rec_err <= TRAJ32:
        return ([f"native-point incoming state: logits differ from the "
                 f"native rule by {rec_err:.3e} (TRAJ32 {TRAJ32})"],
                dict(stage="native-point agreement"))

    B = batch["event"].shape[0]
    probe = (batch["delay"] == DIAG_DELAY) & (batch["offset"] == 0)
    starts, rev_key, unt_key, rev_lab, unt_lab = [], [], [], [], []
    for b in range(B):
        t0 = int(onp.flatnonzero(probe[b])[0])
        r = t0 - int(batch["since_revision"][b, t0])
        starts.append(r)
        rev_key.append(int(batch["key_id"][b, r]))
        rev_lab.append(int(batch["val_id"][b, r]))
        ut = [t for t in (r + DIAG_DELAY, r + DIAG_DELAY + 1)
              if batch["kind"][b, t] == 1][0]
        unt_key.append(int(batch["key_id"][b, ut]))
        unt_lab.append(int(batch["label"][b, ut]))
    starts = onp.asarray(starts)
    if not onp.all(starts >= 1):
        return (["a diagnostic block starts at token 0: no incoming state"],
                dict(stage="block extraction"))
    idx = starts[:, None] + onp.arange(SUFFIX_LEN)[None, :]
    rows_ = onp.arange(B)[:, None]
    suffix = {k: jnp.asarray(onp.asarray(batch[k])[rows_, idx])
              for k in TT.MODEL_INPUTS}
    W_tr = onp.asarray(full["W_trace"])
    U_tr = onp.asarray(full["U_trace"])
    W0 = jnp.asarray(W_tr[onp.arange(B), starts - 1], dtype=dt)
    U0 = jnp.asarray(U_tr[onp.arange(B), starts - 1], dtype=dt)

    k_all, _ = NMD.safe_normalize(native_p["key_raw"])
    readout_inputs = dict(keys=k_all, readout_W=native_p["readout_W"],
                          readout_b=native_p["readout_b"],
                          incoming_W=W0, incoming_U=U0)
    fails = []
    for name, arr in readout_inputs.items():
        fails += finite_failures(f"incoming/readout input {name}", arr)
    if fails:
        return fails, dict(stage="incoming state and readout inputs")
    # host float64 ANALYSIS of production outputs starts here; the production
    # arithmetic itself stayed in the checked production dtype above
    k_all = onp.asarray(k_all, onp.float64)
    RW = onp.asarray(native_p["readout_W"], onp.float64)
    rb = onp.asarray(native_p["readout_b"], onp.float64)
    traces, executed, proc = {}, {}, {}
    for name, (M, gamma, T) in SETTINGS:
        tree = dict(native_p, **leaves(M, gamma, T, dt))
        out = _traced_suffix(tree, suffix, W0, U0)
        fails += payload_failures(f"{name} suffix rollout", out, dt)
        if fails:
            continue
        traces[name] = onp.asarray(out["W_trace"], onp.float64)
        proc[name] = float(onp.asarray(out["proc_max_abs"]).max())
        copies = onp.stack([onp.asarray(out["coeff"][k]).ravel()
                            for k in FL.COEFF_NAMES], axis=1)
        distinct = onp.unique(copies, axis=0)
        executed[name] = dict(zip(FL.COEFF_NAMES,
                                  (float(x) for x in distinct[0])))
        for row in distinct:
            bad = FL.filter_failure(FL.filter_report(
                dict(zip(FL.COEFF_NAMES, (float(x) for x in row)))))
            if bad:
                fails.append(f"{name}: {bad}")
    if fails:
        return fails, dict(stage="suffix rollouts", executed=executed)

    # first-write agreement, as a MATRIX, before anything later is compared
    W0n = onp.asarray(W0, onp.float64)
    dW = {n: traces[n][:, 0] - W0n for n in CONTRAST}
    per_ep = [_rel(dW[CONTRAST[1]][b], dW[CONTRAST[0]][b]) for b in range(B)]
    bitwise = [bool(onp.array_equal(traces[CONTRAST[1]][b, 0],
                                    traces[CONTRAST[0]][b, 0]))
               for b in range(B)]
    first_write = dict(max_relative_difference=float(max(per_ep)),
                       episodes=B, bitwise_identical=int(sum(bitwise)),
                       tolerance=TRAJ32,
                       nonzero_first_write=bool(all(
                           onp.linalg.norm(dW[CONTRAST[0]][b]) > 0
                           for b in range(B))))
    if not (onp.isfinite(first_write["max_relative_difference"])
            and first_write["max_relative_difference"] <= TRAJ32
            and first_write["nonzero_first_write"]):
        return ([f"first-write change in W does not agree: {first_write}"],
                dict(stage="first write", first_write=first_write,
                     executed=executed))

    fam = onp.asarray(batch["family"])
    cond = onp.asarray(batch["condition"])
    curves = {}
    for name in traces:
        Wt = traces[name]
        kr = k_all[onp.asarray(rev_key)]
        ku = k_all[onp.asarray(unt_key)]
        lr_ = onp.einsum("vw,btwk,bk->btv", RW, Wt, kr) + rb
        lu_ = onp.einsum("vw,btwk,bk->btv", RW, Wt, ku) + rb
        fails += finite_failures(f"{name}: revised readout logits", lr_)
        fails += finite_failures(f"{name}: untouched readout logits", lu_)
        if fails:
            continue
        pr, pu = _softmax(lr_), _softmax(lu_)
        fails += finite_failures(f"{name}: revised readout probabilities", pr)
        fails += finite_failures(f"{name}: untouched readout probabilities",
                                 pu)
        if fails:
            continue
        lab_r, lab_u = onp.asarray(rev_lab), onp.asarray(unt_lab)
        p_rev = pr[onp.arange(B)[:, None], onp.arange(SUFFIX_LEN)[None, :],
                   lab_r[:, None]]
        p_unt = pu[onp.arange(B)[:, None], onp.arange(SUFFIX_LEN)[None, :],
                   lab_u[:, None]]
        a_rev = lr_.argmax(-1) == lab_r[:, None]
        a_unt = lu_.argmax(-1) == lab_u[:, None]
        curves[name] = {}
        for fi, fname in enumerate(FAMILIES):
            for ci, cn in enumerate(TT.CONDITIONS):
                m = (fam == fi) & (cond == ci)
                if not m.any():
                    fails.append(f"{name}/{fname}/{cn}: empty cell")
                    continue
                curves[name][f"{fname}/{cn}"] = dict(
                    n=int(m.sum()),
                    revised_label_probability=p_rev[m].mean(0).tolist(),
                    revised_accuracy=a_rev[m].mean(0).tolist(),
                    untouched_label_probability=p_unt[m].mean(0).tolist(),
                    untouched_accuracy=a_unt[m].mean(0).tolist())
        fails += curve_failures(f"{name} curves", curves[name])
    if fails:
        return fails, dict(stage="readouts", first_write=first_write,
                           executed=executed)
    diffs = {}
    for a, b in ((CONTRAST[1], CONTRAST[0]), (CONTRAST[0], "native_point"),
                 (CONTRAST[1], "native_point")):
        diffs[f"{a}_minus_{b}"] = {
            g: {k: (onp.asarray(curves[a][g][k])
                    - onp.asarray(curves[b][g][k])).tolist()
                for k in ("revised_label_probability", "revised_accuracy",
                          "untouched_label_probability",
                          "untouched_accuracy")}
            for g in curves[a]}
        fails += curve_failures(f"difference {a}_minus_{b}",
                                diffs[f"{a}_minus_{b}"])
    if fails:
        return fails, dict(stage="differences", first_write=first_write,
                           executed=executed)
    report = dict(
        token_offsets=list(range(SUFFIX_LEN)),
        offset_meaning=("0 = the revision write; 1..15 = the declared fill; "
                        "16, 17 = the probe pair tokens of the delay-16 block"),
        incoming_state=("W, U after the token before the revision, from the "
                        "same episode under the frozen backbone at the native "
                        "point; processing state y, y_prev, R_prev = 0"),
        native_point_logit_agreement=rec_err,
        executed_coefficients=executed, first_write=first_write,
        processing_state_max_abs=proc,
        finite_first=("every returned logit, W/U trace, processing-state "
                      "summary and coefficient copy (production dtype), the "
                      "incoming states and readout inputs, derived readout "
                      "logits, probabilities, cell means and differences "
                      "were checked finite before use"),
        curves=curves, differences=diffs,
        scope=("diagnostic only: a frozen backbone and prescribed filter "
               "coefficients; neither this closed-loop response nor the "
               "open-loop check establishes a task improvement"))
    return [], report


def run(native_p, batch):
    f1, ol = open_loop(native_p["A_log"].dtype)
    f2, cl = closed_loop(native_p, batch)
    return f1 + f2, dict(open_loop=ol, closed_loop=cl, settings=[
        dict(name=n, M=c[0], gamma=c[1], T=c[2]) for n, c in SETTINGS])
