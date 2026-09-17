"""Stage B: frozen-checkpoint predictive diagnostic for the prospective
readout. NO TRAINING.

Protocol: docs/PROSPECTIVE_READOUT_STAGE_B_PROTOCOL.md. Derivation:
docs/PROSPECTIVE_READOUT_STAGE_A.md. Corrected domain:
docs/PROSPECTIVE_COEFFICIENT_DOMAIN.md. NOT AUTHORIZED TO RUN until static
review clears it. No completed study is re-run or re-evaluated.

On saved native checkpoints, with the backbone frozen, the readout's job has
a ground truth: estimate W_(t+k) from the trajectory up to t. Because X never
enters R, U or W (Stage A s3), ONE backbone rollout per checkpoint supplies
the trajectory for every readout arm.

Primary measurement, per arm and per offset k:

    ||X_t - W_(t+k)||_F / ||W_t - W_(t+k)||_F

so 1.0 means "no better than reading the native fast weight". Reported
separately by direction (revised key versus untouched target keys), by fill
condition (idle versus intervening writes), and with dW split into its write
component -beta_t U_t and its decay component (alpha_t - 1) W_(t-1).
"""

import argparse
import json
import os
import signal
import sys
import time

import jax
import jax.numpy as jnp
import numpy as onp

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from experiments.nested_memory import dynamics as NMD              # noqa: E402
from experiments.nested_memory import model as NM                  # noqa: E402
from experiments.nested_memory.task import IDLE, QUERY, WRITE      # noqa: E402
from experiments.prospective_momentum import filtered as FL        # noqa: E402
from experiments.prospective_momentum import model as PM           # noqa: E402
from experiments.prospective_momentum import replication_sources as RS  # noqa
from experiments.prospective_momentum import retention_aware as RA  # noqa
from experiments.prospective_momentum import study as ST           # noqa: E402
from experiments.prospective_momentum import temporal_diagnostic as TD  # noqa
from experiments.prospective_momentum import temporal_task as TT   # noqa: E402
from experiments.prospective_momentum import tss_containment as TC  # noqa

TRAJ32 = TC.TRAJ32
H = FL.H
#: the frozen checkpoints: the declared, read-only native endpoints of the
#: completed temporal-response run (seed 500 development, 501-503 final)
CHECKPOINT_RUN = RA.REFERENCE_RUN
SOURCE_RUN = RA.SOURCE_RUN
SEEDS = (RA.SOURCE_DEV,) + RA.SOURCE_FINAL
#: fresh probe stream, asserted disjoint from every earlier range
STREAM = dict(probe=720_000_000)
N_PROBE_PER_FAMILY = 32
#: prediction offsets
K_OFFSETS = (1, 2, 4, 8, 16)
#: a pair (t, t+k) is scored only if the native change is above this floor
BASE_FLOOR = 1e-6
#: the declared margin of the stopping rule, in units of the normalized error
STOP_MARGIN = 0.01
#: how many of the five offsets the interior must win by that margin
STOP_MIN_OFFSETS = 3
#: how many of the four checkpoints must pass the rule (no pooled mean)
STOP_MIN_CHECKPOINTS = 3
#: DECLARED BEFORE THE RUN. The idle roll-forward is closed form:
#: target_k = alpha^k W_t - c_k U_t, c_k = beta mu (alpha^k - mu^k)/(alpha-mu),
#: so the U-lookahead family contains its U part exactly and differs only by
#: (1 - alpha^k) W_t, the decay term. If the best lookahead arm gets within
#: NEAR_EXACT of the primary target on the SELECTION half at at least
#: NEAR_EXACT_MIN_OFFSETS offsets, the primary target is (near) trivially
#: predictable and the interior-versus-baseline decision moves to the
#: SECONDARY target (the actual W_(t+k)), where real writes and varying gates
#: make the trend estimate non-trivial. Both are always reported.
NEAR_EXACT = 0.05
NEAR_EXACT_MIN_OFFSETS = 3
#: a reported cell is flagged underpowered if its standard error exceeds half
#: the decision margin
MIN_CELL_N = 100
#: offsets after a revision for the secondary label-probability curves
SECONDARY_OFFSETS = tuple(range(TD.SUFFIX_LEN))
DIAG_DELAY = TD.DIAG_DELAY

# ------------------------------------------------------------ frozen grids --
# MATCHED SEARCH BUDGET: every family that enters the stopping rule has
# exactly N_MATCHED members, declared here. The wider interior grid is kept
# as a DESCRIPTIVE family that never enters the rule.
N_MATCHED = 7
TWO_TAP_KAPPA = (0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
TSS_T = (0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0)
LOOKAHEAD_LAMBDA = (0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
#: seven interior points around (M, gamma, T) = (0.25, 0, 1), spanning the
#: mass, both signs of gamma and the horizon
GEN_MATCHED = ((0.1, 0.0, 1.0), (0.25, 0.0, 1.0), (0.5, 0.0, 1.0),
               (0.25, -0.25, 1.0), (0.25, 0.25, 1.0), (0.25, 0.0, 0.5),
               (0.25, 0.0, 2.0))
#: descriptive only, excluded from the stopping rule
GEN_EXTENDED_M = (0.1, 0.25, 0.5, 1.0)
GEN_EXTENDED_GAMMA = (-0.5, -0.25, 0.0, 0.25, 0.5)
GEN_EXTENDED_T = (0.5, 1.0, 2.0)
#: families that may enter the stopping rule (equal cardinality)
MATCHED_FAMILIES = ("two_tap", "literal_tss", "lookahead",
                    "generalized_interior")
#: the baseline the interior must beat: the best CARRY-FREE or one-matrix
#: control, i.e. the better of the literal-TSS line and the U-lookahead
BASELINE_FAMILIES = ("literal_tss", "lookahead")
KINDS = ("all", "revised", "untouched")
CONDITIONS = ("both",) + TT.CONDITIONS
HALVES = ("selection", "confirmation")
#: query kinds of the temporal task: 0 revised probe, 2 late selected are
#: "revised"; 1 untouched probe, 3 late untouched are "untouched"
REVISED_KINDS, UNTOUCHED_KINDS = (0, 2), (1, 3)


def arms():
    """The frozen arm list. Grid members outside the corrected admissible set
    are excluded HERE, by the declared rule, not after seeing results."""
    out = [dict(name="native_identity", family="identity", kind="identity")]
    for k in TWO_TAP_KAPPA:
        out.append(dict(name=f"two_tap@kappa={k}", family="two_tap",
                        kind="filter", M=0.0, gamma=H * (1.0 - k), T=k * H,
                        kappa=k))
    for T in TSS_T:
        out.append(dict(name=f"tss@T={T}", family="literal_tss",
                        kind="filter", M=0.0, gamma=0.0, T=T))
    for lam in LOOKAHEAD_LAMBDA:
        out.append(dict(name=f"lookahead@lambda={lam}", family="lookahead",
                        kind="lookahead", lam=lam))
    for (M, g, T) in GEN_MATCHED:
        if not FL.admissible(M, g, T):
            raise ValueError(f"declared interior point ({M},{g},{T}) is "
                             "not admissible")
        out.append(dict(name=f"generalized@M={M},gamma={g},T={T}",
                        family="generalized_interior", kind="filter",
                        M=M, gamma=g, T=T))
    named = {a["name"] for a in out}
    for M in GEN_EXTENDED_M:
        for g in GEN_EXTENDED_GAMMA:
            for T in GEN_EXTENDED_T:
                name = f"generalized@M={M},gamma={g},T={T}"
                if FL.admissible(M, g, T) and name not in named:
                    out.append(dict(name=name, family="generalized_extended",
                                    kind="filter", M=M, gamma=g, T=T))
    sizes = {f: sum(1 for a in out if a["family"] == f)
             for f in MATCHED_FAMILIES}
    if set(sizes.values()) != {N_MATCHED}:
        raise ValueError(f"matched families must have {N_MATCHED} members "
                         f"each: {sizes}")
    return out


def arm_coefficients(arm, dtype=jnp.float32):
    """Executed coefficients of a filter arm, formed by the production
    function in the production dtype."""
    p = {"fil_M": jnp.full((1,), arm["M"], dtype=dtype),
         "fil_gamma": jnp.full((1,), arm["gamma"], dtype=dtype),
         "fil_T": jnp.full((1,), arm["T"], dtype=dtype)}
    return FL.coefficient_values(FL.coefficients(p))


def arm_admissibility(arm_list):
    """Executed-coefficient acceptance of every filter arm under the
    CORRECTED domain. Returns (failures, rows)."""
    fails, rows = [], []
    for a in arm_list:
        if a["kind"] != "filter":
            rows.append(dict(name=a["name"], family=a["family"],
                             executed=None))
            continue
        ex = arm_coefficients(a)
        rep = FL.filter_report(ex, source="grid member, production dtype")
        bad = FL.filter_failure_corrected(rep)
        # a repaired point is a DIFFERENT hypothesis (T also sets the velocity
        # smoother's window), so proposed and repaired coordinates are logged
        # separately and a moved grid point FAILS rather than being reported
        # at its nominal location
        proposed = {"fil_M": jnp.full((1,), a["M"], jnp.float32),
                    "fil_gamma": jnp.full((1,), a["gamma"], jnp.float32),
                    "fil_T": jnp.full((1,), a["T"], jnp.float32)}
        repaired, tel = FL.repair_corrected(proposed)
        moved = int(tel["n_repaired"]) > 0
        if moved:
            fails.append(f"{a['name']}: the corrected repair moves this grid "
                         f"point to M={float(repaired['fil_M'][0])}, "
                         f"gamma={float(repaired['fil_gamma'][0])}, "
                         f"T={float(repaired['fil_T'][0])}; a repaired point "
                         "is a different hypothesis and is not reported at "
                         "its nominal location")
        if bad:
            fails.append(f"{a['name']}: {bad}")
        rows.append(dict(name=a["name"], family=a["family"],
                         proposed=dict(M=a["M"], gamma=a["gamma"], T=a["T"]),
                         repaired=dict(M=float(repaired["fil_M"][0]),
                                       gamma=float(repaired["fil_gamma"][0]),
                                       T=float(repaired["fil_T"][0])),
                         repair_moved=moved, executed=ex,
                         executed_dtype="float32",
                         classification=rep["classification"],
                         jury_slacks=rep["jury_slacks_exact"], failure=bad))
    return fails, rows


# ------------------------------------------------------------- trajectories --
@jax.jit
def _traces(p, eps):
    def one(e):
        out = PM.rollout(FL.FILTERED, p, e, trace=True)
        a, b, mu, eta = out["gates"]
        return dict(W=out["W_trace"], U=out["U_trace"], alpha=a, beta=b,
                    mu=mu, logits=out["logits"], coeff=out["coeff"])
    return jax.vmap(one)(eps)


@jax.jit
def _native_logits(p, eps):
    return jax.vmap(lambda e: PM.rollout("momentum_delta", p, e)["logits"])(
        eps)


@jax.jit
def _idle_gates(p):
    """The gates of an IDLE token (key 0, no value): the token the
    counterfactual roll-forward uses."""
    x = NM.gate_features(jnp.zeros((1,), jnp.int32),
                         -jnp.ones((1,), jnp.int32),
                         jnp.full((1,), IDLE, jnp.int32)).astype(
                             p["key_raw"].dtype)
    a, b, mu, eta = NM._momentum_gates(p, x)
    return a[0], b[0], mu[0], eta[0]


def trajectories(native_p, batch):
    """W, U, alpha, beta for every token, from the processing law at its exact
    native point, whose agreement with the native rule is verified here."""
    nat_pt = TC.native_point_tree(native_p)
    eps = {k: jnp.asarray(batch[k]) for k in TT.MODEL_INPUTS}
    tr = _traces(nat_pt, eps)
    ref = _native_logits(native_p, eps)
    want = onp.dtype(native_p["A_log"].dtype)
    fails = []
    for name in ("W", "U", "alpha", "beta", "mu", "logits"):
        arr = onp.asarray(tr[name])
        if arr.dtype != want:
            fails.append(f"{name} dtype {arr.dtype} is not the production "
                         f"dtype {want}")
        fails += TD.finite_failures(f"native-point trace {name}", arr)
    for k in FL.COEFF_NAMES:
        fails += TD.finite_failures(f"executed coefficient {k}",
                                    onp.asarray(tr["coeff"][k]))
    fails += TD.finite_failures("native rule logits", ref)
    ex = FL.coefficient_values({k: onp.asarray(v).ravel()[0]
                                for k, v in tr["coeff"].items()})
    bad = FL.filter_failure_corrected(FL.filter_report(
        ex, source="native-point trajectory"))
    if bad:
        fails.append(f"native-point trajectory coefficients: {bad}")
    err = TD._rel(tr["logits"], ref)
    if not err <= TRAJ32:
        fails.append(f"native-point trajectory differs from the native rule "
                     f"by {err:.3e} (TRAJ32 {TRAJ32})")
    ig = [float(x) for x in _idle_gates(native_p)]
    fails += TD.finite_failures("idle gates", onp.asarray(ig))
    return fails, dict(W=onp.asarray(tr["W"]), U=onp.asarray(tr["U"]),
                       alpha=onp.asarray(tr["alpha"]),
                       beta=onp.asarray(tr["beta"]),
                       idle_gates=dict(alpha=ig[0], beta=ig[1], mu=ig[2],
                                       eta=ig[3]),
                       native_point_logit_agreement=err)


def roll_forward(tr):
    """PRIMARY TARGET: the native recurrence rolled forward from (W_t, U_t)
    over IDLE tokens only, k steps ahead. This is what a prospective readout
    is entitled to anticipate: the transient already in flight, with no
    unobserved future write. On an idle token R = 0, so
    U <- mu U and W <- alpha W - beta U."""
    g = tr["idle_gates"]
    al = onp.float32(g["alpha"]); be = onp.float32(g["beta"])
    mu = onp.float32(g["mu"])
    W = tr["W"].copy(); U = tr["U"].copy()
    out = {}
    for step in range(1, max(K_OFFSETS) + 1):
        U = mu * U
        W = al * W - be * U
        if step in K_OFFSETS:
            out[step] = W.copy()
    return out


def split_halves(batch):
    """Declared BEFORE the run: inside every (family, condition) stratum the
    first half of the episodes is the SELECTION half and the second half the
    CONFIRMATION half. Arms are chosen on selection and reported on
    confirmation."""
    fam = onp.asarray(batch["family"]); cond = onp.asarray(batch["condition"])
    half = onp.empty(fam.shape, dtype=object)
    for f in onp.unique(fam):
        for c in onp.unique(cond):
            idx = onp.flatnonzero((fam == f) & (cond == c))
            cut = len(idx) // 2
            half[idx[:cut]] = "selection"
            half[idx[cut:]] = "confirmation"
    return half


def readout(arm, tr):
    """X for one arm, in the production dtype, from the frozen trajectory.
    X never enters R, U or W (Stage A s3), so every arm reads the SAME
    trajectory."""
    W, U, beta = tr["W"], tr["U"], tr["beta"]
    if arm["kind"] == "identity":
        return W.copy()
    if arm["kind"] == "lookahead":
        return W - onp.float32(arm["lam"]) * beta[..., None, None] * U
    c = arm_coefficients(arm)
    a, b, cc, d = (onp.float32(c["a"]), onp.float32(c["b"]),
                   onp.float32(c["c"]), onp.float32(c["d"]))
    X = onp.zeros_like(W)
    x1 = onp.zeros_like(W[:, 0])
    x2 = onp.zeros_like(W[:, 0])
    wp = onp.zeros_like(W[:, 0])
    for t in range(W.shape[1]):
        x = a * x1 - b * x2 + cc * W[:, t] - d * wp
        X[:, t] = x
        x2, x1, wp = x1, x, W[:, t]
    return X

# ----------------------------------------------------------------- metrics --
def _ratio(num, den, mask):
    """Mean normalized error, with the standard error of that mean and an
    explicit underpowered flag: a cell cannot support the 0.01 decision
    margin if its standard error exceeds half of it."""
    ok = mask & (den > BASE_FLOOR) & onp.isfinite(num) & onp.isfinite(den)
    n = int(ok.sum())
    if n == 0:
        return dict(mean=None, sem=None, n=0, excluded=int(mask.sum()),
                    underpowered=True)
    r = num[ok] / den[ok]
    sem = float(r.std(ddof=1) / onp.sqrt(n)) if n > 1 else None
    return dict(mean=float(r.mean()), sem=sem, n=n,
                excluded=int(mask.sum()) - n,
                underpowered=bool(n < MIN_CELL_N or sem is None
                                  or sem > STOP_MARGIN / 2))


def arm_metrics(X, tr, batch, targets, keys, half):
    """Normalized prediction error. PRIMARY: key-projected, against the
    counterfactual roll-forward, at the tokens where a query is actually
    answered, split by query kind, fill condition and half. SECONDARY: the
    same against the ACTUAL W_(t+k), and the full-matrix (Frobenius) norms."""
    W = tr["W"]
    B, L = W.shape[0], W.shape[1]
    ev, kd = onp.asarray(batch["event"]), onp.asarray(batch["kind"])
    cond = onp.asarray(batch["condition"])
    q_tok = ev == QUERY
    qkeys = keys[onp.asarray(batch["key_id"])]                 # (B, L, d_k)
    out = {}
    for k in K_OFFSETS:
        roll = targets[k]
        pairs = [("rollforward", roll, onp.ones((B, L), dtype=bool))]
        act = onp.zeros_like(W)
        act[:, :L - k] = W[:, k:]
        ok_act = onp.zeros((B, L), dtype=bool)
        ok_act[:, :L - k] = True
        pairs.append(("actual", act, ok_act))
        for tname, target, valid in pairs:
            D = (X - target).astype(onp.float64)
            Bs = (W - target).astype(onp.float64)
            key_num = onp.linalg.norm(onp.einsum("btvw,btw->btv", D, qkeys),
                                      axis=-1)
            key_den = onp.linalg.norm(onp.einsum("btvw,btw->btv", Bs, qkeys),
                                      axis=-1)
            fro_num = onp.linalg.norm(D, axis=(-2, -1))
            fro_den = onp.linalg.norm(Bs, axis=(-2, -1))
            for hname in HALVES + ("both",):
                hm = (onp.ones((B, 1), dtype=bool) if hname == "both"
                      else (half == hname)[:, None])
                for cname in CONDITIONS:
                    cm = (onp.ones((B, 1), dtype=bool) if cname == "both"
                          else (cond == TT.CONDITIONS.index(cname))[:, None])
                    for kind in KINDS:
                        if kind == "all":
                            km = q_tok
                        elif kind == "revised":
                            km = q_tok & onp.isin(kd, REVISED_KINDS)
                        else:
                            km = q_tok & onp.isin(kd, UNTOUCHED_KINDS)
                        m = valid & hm & cm & km
                        out[f"{tname}/key/{kind}/{cname}/{hname}/k{k}"] = \
                            _ratio(key_num, key_den, m)
                    if cname == "both" and hname == "both":
                        out[f"{tname}/frobenius/all/both/both/k{k}"] = _ratio(
                            fro_num, fro_den, valid)
    return out


PRIMARY_TARGET, SECONDARY_TARGET = "rollforward", "actual"
PRIMARY_PROJECTION = "key"


def primary_key(kind="all", cond="both", half="confirmation", k=1,
                target=PRIMARY_TARGET):
    return f"{target}/{PRIMARY_PROJECTION}/{kind}/{cond}/{half}/k{k}"


def closed_form_diagnostic(tr, metrics_by_arm):
    """How predictable the primary target is, per checkpoint, and the
    DECLARED consequence. Reports the closed-form coefficients of the idle
    roll-forward, how much the per-token gates vary (which is why a fixed
    lambda can or cannot match), and the best lookahead arm's SELECTION-half
    error at each offset."""
    g = tr["idle_gates"]
    al, be, mu = g["alpha"], g["beta"], g["mu"]
    coeffs = {}
    for k in K_OFFSETS:
        cu = (be * mu * (al ** k - mu ** k) / (al - mu) if al != mu
              else k * be * mu * al ** (k - 1))
        coeffs[f"k{k}"] = dict(coefficient_on_W=float(al ** k),
                               coefficient_on_U=float(cu),
                               decay_gap_on_W=float(1.0 - al ** k))
    best = {}
    for k in K_OFFSETS:
        sel = primary_key(half="selection", k=k)
        vals = [(n, rec["metrics"][sel]["mean"])
                for n, rec in metrics_by_arm.items()
                if rec["family"] == "lookahead"
                and rec["metrics"][sel]["mean"] is not None]
        best[f"k{k}"] = (dict(arm=min(vals, key=lambda x: x[1])[0],
                              selection=min(v for _, v in vals))
                         if vals else None)
    near = sum(1 for v in best.values()
               if v is not None and v["selection"] <= NEAR_EXACT)
    decisive = (SECONDARY_TARGET if near >= NEAR_EXACT_MIN_OFFSETS
                else PRIMARY_TARGET)
    return dict(
        idle_gates=g, closed_form_coefficients=coeffs,
        gate_variation={n: dict(mean=float(onp.mean(tr[n])),
                                std=float(onp.std(tr[n])),
                                min=float(onp.min(tr[n])),
                                max=float(onp.max(tr[n])))
                        for n in ("alpha", "beta")},
        best_lookahead_on_primary=best, near_exact_threshold=NEAR_EXACT,
        offsets_near_exact=near, decisive_target=decisive,
        note=("the idle roll-forward is an exact linear function of "
              "(W_t, U_t): alpha^k W_t - c_k U_t. The U-lookahead family "
              "contains its U part exactly and differs only by the decay "
              "term (1 - alpha^k) W_t, which the filtered arms can "
              "extrapolate and the lookahead cannot. If the lookahead is "
              "already near-exact the primary target is close to trivially "
              "predictable, and by the DECLARED rule the decision moves to "
              "the secondary target; both remain reported."))


def component_split(tr, batch):
    """The two parts of dW, reported so a filter that beats the baseline can
    be read as extrapolating the write term, the decay term, or both."""
    W, U, alpha, beta = tr["W"], tr["U"], tr["alpha"], tr["beta"]
    write = -beta[..., None, None] * U
    prev = onp.concatenate([onp.zeros_like(W[:, :1]), W[:, :-1]], axis=1)
    decay = (alpha[..., None, None] - 1.0) * prev
    dW = W - prev
    nw = onp.linalg.norm(write.astype(onp.float64), axis=(-2, -1))
    nd = onp.linalg.norm(decay.astype(onp.float64), axis=(-2, -1))
    nt = onp.linalg.norm(dW.astype(onp.float64), axis=(-2, -1))
    ev = onp.asarray(batch["event"]); cond = onp.asarray(batch["condition"])
    out = dict(identity_max_abs=float(onp.max(onp.abs(
        (write + decay - dW).astype(onp.float64)))))
    for cname in CONDITIONS:
        m = (onp.ones_like(nt, dtype=bool) if cname == "both"
             else onp.broadcast_to(
                 (cond == TT.CONDITIONS.index(cname))[:, None], nt.shape))
        for tname, tm in (("write_tokens", ev == WRITE),
                          ("non_write_tokens", ev != WRITE)):
            mm = m & tm
            if not mm.any():
                continue
            out[f"{cname}/{tname}"] = dict(
                write_component=float(nw[mm].mean()),
                decay_component=float(nd[mm].mean()),
                total_dW=float(nt[mm].mean()), n=int(mm.sum()))
    return out


def secondary_curves(X, tr, batch, keys, readout_W, readout_b):
    """Revised-label and untouched-label probabilities through the existing
    readout map, at offsets after each delay-16 revision."""
    B, L = tr["W"].shape[0], tr["W"].shape[1]
    probe = (batch["delay"] == DIAG_DELAY) & (batch["offset"] == 0)
    rows, labs_r, labs_u, keys_r, keys_u = [], [], [], [], []
    for b in range(B):
        t0 = int(onp.flatnonzero(probe[b])[0])
        r = t0 - int(batch["since_revision"][b, t0])
        if r + len(SECONDARY_OFFSETS) > L:
            continue
        ut = [t for t in (r + DIAG_DELAY, r + DIAG_DELAY + 1)
              if batch["kind"][b, t] == 1][0]
        rows.append((b, r))
        keys_r.append(int(batch["key_id"][b, r]))
        labs_r.append(int(batch["val_id"][b, r]))
        keys_u.append(int(batch["key_id"][b, ut]))
        labs_u.append(int(batch["label"][b, ut]))
    if not rows:
        return None
    idx_b = onp.asarray([b for b, _ in rows])
    idx_t = onp.asarray([[r + o for o in SECONDARY_OFFSETS] for _, r in rows])
    Xs = X[idx_b[:, None], idx_t].astype(onp.float64)
    out = {"offsets": list(SECONDARY_OFFSETS)}
    for tag, kk, ll in (("revised", keys_r, labs_r),
                        ("untouched", keys_u, labs_u)):
        q = keys[onp.asarray(kk)]
        logits = onp.einsum("vw,btwk,bk->btv", readout_W, Xs, q) + readout_b
        pr = TD._softmax(logits)
        lab = onp.asarray(ll)
        out[f"{tag}_label_probability"] = pr[
            onp.arange(len(rows))[:, None],
            onp.arange(len(SECONDARY_OFFSETS))[None, :], lab[:, None]
        ].mean(0).tolist()
        out[f"{tag}_accuracy"] = (logits.argmax(-1)
                                  == lab[:, None]).mean(0).tolist()
    return out


# ------------------------------------------------------- report and verdict --
def argmins(metrics_by_arm, families=None, keys=None):
    """Best arm per reported key. `families` restricts the pool (the matched
    budget); `keys` restricts which keys are ranked."""
    out = {}
    pool = [n for n in metrics_by_arm
            if families is None or metrics_by_arm[n]["family"] in families]
    all_keys = keys or [k for k in metrics_by_arm[pool[0]]["metrics"]]
    for key in all_keys:
        best = None
        for n in pool:
            v = metrics_by_arm[n]["metrics"][key]["mean"]
            if v is None:
                continue
            if best is None or v < best[1]:
                best = (n, v)
        out[key] = dict(arm=best[0], mean=best[1]) if best else None
    return out


def stopping_rule(metrics_by_arm, target=PRIMARY_TARGET):
    """PREDECLARED (protocol s6), for ONE checkpoint.

    Baseline: the better of the literal-TSS line and the U-lookahead - the
    strongest control, since the lookahead carries no extra state and does
    not extrapolate the decay term. Only the matched-budget families take
    part (equal cardinality), so the interior cannot win by grid size.

    Arms are chosen on the SELECTION half and compared on the CONFIRMATION
    half. The interior wins an offset only if its confirmation number is at
    least STOP_MARGIN below the baseline's."""
    def pick(families, k):
        sel = primary_key(half="selection", k=k, target=target)
        conf = primary_key(half="confirmation", k=k, target=target)
        best = None
        for n, rec in metrics_by_arm.items():
            if rec["family"] not in families:
                continue
            v = rec["metrics"][sel]["mean"]
            if v is None:
                continue
            if best is None or v < best[1]:
                best = (n, v, rec["metrics"][conf]["mean"])
        return (dict(arm=best[0], selection=best[1], confirmation=best[2])
                if best else None)
    per_k, wins = {}, 0
    for k in K_OFFSETS:
        gi = pick(("generalized_interior",), k)
        base = pick(BASELINE_FAMILIES, k)
        if (gi is None or base is None or gi["confirmation"] is None
                or base["confirmation"] is None):
            per_k[f"k{k}"] = dict(interior=gi, baseline=base, wins=False)
            continue
        win = bool(gi["confirmation"] <= base["confirmation"] - STOP_MARGIN)
        wins += int(win)
        conf_key = primary_key(half="confirmation", k=k, target=target)
        under = [n for n in (gi["arm"], base["arm"])
                 if metrics_by_arm[n]["metrics"][conf_key]["underpowered"]]
        per_k[f"k{k}"] = dict(interior=gi, baseline=base,
                              difference=gi["confirmation"]
                              - base["confirmation"], wins=win,
                              underpowered_arms=under)
    return dict(target=target, margin=STOP_MARGIN,
                offsets_required=STOP_MIN_OFFSETS, offsets_won=wins,
                per_offset=per_k, baseline_families=list(BASELINE_FAMILIES),
                passes=bool(wins >= STOP_MIN_OFFSETS))


def overall_verdict(per_seed):
    """The rule is evaluated PER CHECKPOINT, on that checkpoint's DECLARED
    decisive target; the interior helps only if it passes on at least
    STOP_MIN_CHECKPOINTS of the four."""
    passed = [s for s, r in per_seed.items() if r["stopping_rule"]["passes"]]
    helped = len(passed) >= STOP_MIN_CHECKPOINTS
    return dict(checkpoints_passed=sorted(passed),
                checkpoints_required=STOP_MIN_CHECKPOINTS,
                decisive_target={s: r["stopping_rule"]["target"]
                                 for s, r in per_seed.items()},
                interior_beats_baseline=helped,
                verdict=("the generalized interior beats the best carry-free "
                         "or one-matrix control at predicting the rolled-"
                         "forward fast weight" if helped else
                         "STOP: no generalized-interior member beats the best "
                         "of the literal-TSS line and the U-lookahead; the "
                         "added freedom does not help on the read path"),
                scope=("a negative verdict is a statement about the SEVEN "
                       "DECLARED interior points, on this task, these "
                       "checkpoints and these offsets - not about the "
                       "interior of the family as a whole"),
                note=("declared before execution: targets and the rule for "
                      "choosing between them, projection, baseline, matched "
                      "budget, selection/confirmation split, margin and the "
                      "per-checkpoint requirement. Grids and categories are "
                      "not widened or re-cut afterwards, and passing "
                      "authorizes no trained comparison."))


# -------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root",
                    default="/Users/durso/s5-runs/prospective-readout-probe")
    ap.add_argument("--source_run", default=SOURCE_RUN)
    ap.add_argument("--checkpoint_run", default=CHECKPOINT_RUN)
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--budget_s", type=float, default=600.0)
    ap.add_argument("--reserve_s", type=float, default=30.0)
    args = ap.parse_args()
    t0 = time.time()
    deadline = args.deadline if args.deadline else t0 + args.budget_s
    if jax.default_backend() != "gpu":
        raise SystemExit("REFUSING: backend is not 'gpu'.")
    if jax.config.read("jax_enable_x64") or jnp.zeros(1).dtype != jnp.float32:
        raise SystemExit("REFUSING: production is float32 with x64 off.")
    if os.path.realpath(args.checkpoint_run) != os.path.realpath(
            CHECKPOINT_RUN):
        raise SystemExit("REFUSING: the checkpoint run is declared")
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(args.out_root, run_id)
    os.makedirs(out, exist_ok=True)
    arm_list = arms()
    status = dict(
        run_id=run_id, out=out, backend=jax.default_backend(),
        study=("Stage B: frozen-checkpoint predictive diagnostic for the "
               "prospective readout (no training)"),
        protocol="docs/PROSPECTIVE_READOUT_STAGE_B_PROTOCOL.md",
        stage_a="docs/PROSPECTIVE_READOUT_STAGE_A.md",
        domain="docs/PROSPECTIVE_COEFFICIENT_DOMAIN.md",
        source_run=args.source_run, checkpoint_run=args.checkpoint_run,
        seeds=list(SEEDS), stream=STREAM,
        episodes_per_family=N_PROBE_PER_FAMILY, offsets=list(K_OFFSETS),
        base_floor=BASE_FLOOR,
        grids=dict(matched_size=N_MATCHED, two_tap_kappa=list(TWO_TAP_KAPPA),
                   tss_T=list(TSS_T),
                   lookahead_lambda=list(LOOKAHEAD_LAMBDA),
                   generalized_matched=[list(x) for x in GEN_MATCHED],
                   generalized_extended=dict(M=list(GEN_EXTENDED_M),
                                             gamma=list(GEN_EXTENDED_GAMMA),
                                             T=list(GEN_EXTENDED_T)),
                   matched_families=list(MATCHED_FAMILIES),
                   baseline_families=list(BASELINE_FAMILIES)),
        arms=[a["name"] for a in arm_list], n_arms=len(arm_list),
        measurement=(
            "PRIMARY: ||(X_t - target_k) q|| / ||(W_t - target_k) q|| at the "
            "tokens where a query is answered, q that query's key, with "
            "target_k the native recurrence rolled forward k IDLE steps from "
            "(W_t, U_t) - the transient already in flight, with no "
            "unobserved future write. 1.0 means no better than reading the "
            "native fast weight. SECONDARY: the actual W_(t+k) as target, "
            "and the full-matrix norms."),
        baseline=("the stopping rule measures the generalized interior "
                  "against the BEST of the literal-TSS line and the "
                  "U-lookahead, at matched search budget"),
        training="none: the backbone is frozen and nothing is optimized",
        source=dict(hashes_at_restore=None), incomplete=[])
    status_path = os.path.join(out, "status.json")

    def persist(st):
        st["wall_s"] = time.time() - t0
        ST.write(status_path, st)

    def rehash():
        keys = (status.get("source") or {}).get("hashes_at_restore")
        if not keys:
            return {}
        d = RS.sources_dir(args.source_run)
        got = {}
        for k in keys:
            path = (k[len("reference:"):] if k.startswith("reference:")
                    else os.path.join(d, k))
            got[k] = RS.sha256(path) if os.path.isfile(path) else None
        return got

    signal.signal(signal.SIGTERM, ST._on_sigterm)
    code, _ = ST.guarded(lambda: run_probe(args, status, arm_list, deadline,
                                           lambda: persist(status)),
                         status, rehash, persist)
    return code


def run_probe(args, status, arm_list, deadline, save):
    batch = TT.generate_batch(STREAM["probe"], N_PROBE_PER_FAMILY)
    check = TT.structure_check(batch)
    status["task_check"] = check
    status["task_digest"] = TT.episode_digest(batch)
    if not (check["oracle_exact"] and check["block_cells_balanced"]):
        status["failed"] = f"probe task check failed: {check}"
        return 4, "FAILED"
    fails, rows = arm_admissibility(arm_list)
    status["arm_table"] = rows
    half = split_halves(batch)
    status["split"] = dict(
        selection=int((half == "selection").sum()),
        confirmation=int((half == "confirmation").sum()),
        rule=("declared before the run: inside every (family, condition) "
              "stratum the first half of the episodes selects and the second "
              "half confirms"))
    save()
    if fails:
        status["failed"] = f"inadmissible or unstable grid members: {fails}"
        return 4, "FAILED"
    try:
        sources = TC.load_sources(args.source_run, status)
        refs, ref_hashes = RA.load_references(args.checkpoint_run, sources,
                                              status)
    except (RS.SourceRefusal, RA.ReferenceRefusal) as e:
        status["failed"] = f"checkpoint refusal: {e}"
        print(f"[!] {status['failed']}")
        return 4, "FAILED"
    status["source"]["hashes_at_restore"].update(ref_hashes)
    print(f"[checkpoints] {len(refs)} frozen native checkpoints verified")
    save()

    #: one arm of each structural class is timed, because the classes differ
    #: in carried state and cost
    classes = ("identity", "two_tap", "lookahead", "literal_tss",
               "generalized_interior")
    per_seed, detail, class_s = {}, {}, {}
    for i, seed in enumerate(SEEDS):
        t_seed = time.time()
        p = refs[seed]
        fails, tr = trajectories(p, batch)
        if fails:
            status["failed"] = f"trajectory checks failed for {seed}: {fails}"
            return 4, "FAILED"
        targets = roll_forward(tr)
        keys = onp.asarray(NMD.safe_normalize(p["key_raw"])[0], onp.float64)
        rw = onp.asarray(p["readout_W"], onp.float64)
        rb = onp.asarray(p["readout_b"], onp.float64)
        metrics, secondary = {}, {}
        for a in arm_list:
            t_arm = time.time()
            X = readout(a, tr)
            if not onp.all(onp.isfinite(X)):
                status["failed"] = f"non-finite readout {a['name']} ({seed})"
                return 4, "FAILED"
            metrics[a["name"]] = dict(
                family=a["family"],
                metrics=arm_metrics(X, tr, batch, targets, keys, half))
            secondary[a["name"]] = secondary_curves(X, tr, batch, keys, rw,
                                                    rb)
            if a["family"] in classes and a["family"] not in class_s:
                class_s[a["family"]] = time.time() - t_arm
                if len(class_s) == len(classes):
                    need = sum(
                        class_s.get(b["family"], max(class_s.values()))
                        for b in arm_list) * (len(SEEDS) - i)
                    left = deadline - time.time() - args.reserve_s
                    status["projection"] = dict(
                        seconds_per_arm_by_class=dict(class_s),
                        projected_remaining_s=need, remaining_s=left,
                        note=("projected from one arm of EVERY structural "
                              "class, since carried state and cost differ"))
                    print(f"PROBE_PROJECTED_REMAINING_S={need:.1f}")
                    save()
                    if need > left:
                        why = (f"projected {need:.0f}s > remaining "
                               f"{left:.0f}s; not started, nothing reduced")
                        status["incomplete"].append(why)
                        print(f"[!] {why}")
                        return 3, "INCOMPLETE"
            if time.time() > deadline - args.reserve_s:
                status["incomplete"].append(
                    f"stopped at arm {a['name']} of seed {seed}")
                return 3, "INCOMPLETE"
        primary = [primary_key(kind, cond, half_, k, target)
                   for target in (PRIMARY_TARGET, SECONDARY_TARGET)
                   for kind in KINDS for cond in CONDITIONS
                   for half_ in HALVES for k in K_OFFSETS]
        matched = {n: m for n, m in metrics.items()
                   if m["family"] in MATCHED_FAMILIES}
        closed = closed_form_diagnostic(tr, matched)
        sr = stopping_rule(matched, closed["decisive_target"])
        other_target = (SECONDARY_TARGET
                        if closed["decisive_target"] == PRIMARY_TARGET
                        else PRIMARY_TARGET)
        other = stopping_rule(matched, other_target)
        cells = {key: metrics["native_identity"]["metrics"][key]
                 for key in primary}
        under = sorted(key for key, v in cells.items() if v["underpowered"])
        per_seed[str(seed)] = dict(
            stopping_rule=sr, rule_on_the_other_target=other,
            closed_form=closed, underpowered_cells=under,
            cell_counts={key: dict(n=v["n"], sem=v["sem"])
                         for key, v in cells.items()},
            argmins_matched=argmins(metrics, MATCHED_FAMILIES, primary),
            argmins_all_families=argmins(metrics, None, primary),
            component_split=component_split(tr, batch),
            idle_gates=tr["idle_gates"],
            native_point_logit_agreement=tr[
                "native_point_logit_agreement"],
            wall_s=time.time() - t_seed)
        detail[str(seed)] = dict(metrics={n: m["metrics"]
                                          for n, m in metrics.items()},
                                 secondary=secondary)
        status["per_seed"] = per_seed
        save()
        print(f"[seed {seed}] decisive target {closed['decisive_target']} "
              f"(lookahead near-exact at {closed['offsets_near_exact']} of "
              f"{len(K_OFFSETS)}); passes {sr['passes']} (won "
              f"{sr['offsets_won']}); underpowered cells {len(under)} of "
              f"{len(primary)}")
    ST.write(os.path.join(status["out"], "metrics.json"), detail)
    status["overall"] = overall_verdict(per_seed)
    status["work_completed"] = dict(seeds=len(SEEDS), arms=len(arm_list),
                                    episodes=int(batch["event"].shape[0]),
                                    offsets=list(K_OFFSETS), updates=0)
    ov = status["overall"]
    print(f"[stopping rule] interior_beats_baseline="
          f"{ov['interior_beats_baseline']} checkpoints passed "
          f"{ov['checkpoints_passed']}")
    status["complete"] = True
    return 0, "PASS"


if __name__ == "__main__":
    sys.exit(main())
