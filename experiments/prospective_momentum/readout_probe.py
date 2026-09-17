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
from experiments.nested_memory.task import WRITE                  # noqa: E402
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
#: offsets after a revision for the secondary label-probability curves
SECONDARY_OFFSETS = tuple(range(TD.SUFFIX_LEN))
DIAG_DELAY = TD.DIAG_DELAY

# ------------------------------------------------------------ frozen grids --
TWO_TAP_KAPPA = (0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
TSS_T = (0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 8.0)
GEN_M = (0.1, 0.25, 0.5, 1.0)
GEN_GAMMA = (-0.5, -0.25, 0.0, 0.25, 0.5)
GEN_T = (0.5, 1.0, 2.0)
LOOKAHEAD_LAMBDA = (0.25, 0.5, 1.0, 1.5, 2.0)
CATEGORIES = ("overall", "revised_direction", "untouched_direction")
CONDITIONS = ("both",) + TT.CONDITIONS


def arms():
    """The frozen arm list, built once and printed before any measurement.
    Grid members outside the corrected admissible set are excluded HERE, by
    the declared rule, not after seeing results."""
    out = [dict(name="native_identity", family="identity", kind="identity",
                coefficients=None)]
    for k in TWO_TAP_KAPPA:
        out.append(dict(name=f"two_tap@kappa={k}", family="two_tap",
                        kind="filter", M=0.0, gamma=H * (1.0 - k), T=k * H,
                        kappa=k))
    for T in TSS_T:
        out.append(dict(name=f"tss@T={T}", family="literal_tss",
                        kind="filter", M=0.0, gamma=0.0, T=T))
    for M in GEN_M:
        for g in GEN_GAMMA:
            for T in GEN_T:
                if FL.admissible(M, g, T):
                    out.append(dict(name=f"generalized@M={M},gamma={g},T={T}",
                                    family="generalized_interior",
                                    kind="filter", M=M, gamma=g, T=T))
    for lam in LOOKAHEAD_LAMBDA:
        out.append(dict(name=f"lookahead@lambda={lam}", family="lookahead",
                        kind="lookahead", lam=lam))
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
                             coefficients=None))
            continue
        ex = arm_coefficients(a)
        rep = FL.filter_report(ex, source="grid member, production dtype")
        bad = FL.filter_failure_corrected(rep)
        if bad:
            fails.append(f"{a['name']}: {bad}")
        rows.append(dict(name=a["name"], family=a["family"], executed=ex,
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
                    logits=out["logits"], coeff=out["coeff"])
    return jax.vmap(one)(eps)


@jax.jit
def _native_logits(p, eps):
    return jax.vmap(lambda e: PM.rollout("momentum_delta", p, e)["logits"])(
        eps)


def trajectories(native_p, batch):
    """W, U, alpha, beta for every token, from the processing law at its exact
    native point, whose agreement with the native rule is verified here."""
    nat_pt = TC.native_point_tree(native_p)
    eps = {k: jnp.asarray(batch[k]) for k in TT.MODEL_INPUTS}
    tr = _traces(nat_pt, eps)
    ref = _native_logits(native_p, eps)
    want = onp.dtype(native_p["A_log"].dtype)
    fails = []
    for name in ("W", "U", "alpha", "beta", "logits"):
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
    return fails, dict(W=onp.asarray(tr["W"]), U=onp.asarray(tr["U"]),
                       alpha=onp.asarray(tr["alpha"]),
                       beta=onp.asarray(tr["beta"]),
                       native_point_logit_agreement=err)


def readout(arm, tr):
    """X for one arm, in the production dtype, from the frozen trajectory."""
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
def episode_directions(batch, keys):
    """Per episode: the key vector of the most recent revision at each token
    (or None before the first), and the untouched target key vectors."""
    B, L = batch["event"].shape
    rev_at = onp.full((B, L), -1, dtype=onp.int32)
    unt = []
    for b in range(B):
        probes = onp.flatnonzero(batch["kind"][b] == 0)
        rev_tokens = sorted({int(t - batch["since_revision"][b, t])
                             for t in probes})
        cur = -1
        for t in range(L):
            if cur + 1 < len(rev_tokens) and t >= rev_tokens[cur + 1]:
                cur += 1
            rev_at[b, t] = (int(batch["key_id"][b, rev_tokens[cur]])
                            if cur >= 0 else -1)
        unt.append(sorted({int(batch["key_id"][b, t]) for t in
                           onp.flatnonzero((batch["kind"][b] == 1)
                                           | (batch["kind"][b] == 3))}))
    return rev_at, onp.asarray(unt), keys


def _ratio(num, den, mask):
    ok = mask & (den > BASE_FLOOR) & onp.isfinite(num) & onp.isfinite(den)
    n = int(ok.sum())
    if n == 0:
        return dict(mean=None, n=0, excluded=int(mask.sum()))
    return dict(mean=float((num[ok] / den[ok]).mean()), n=n,
                excluded=int(mask.sum()) - n)


def arm_metrics(X, tr, batch, rev_at, unt_keys, keys):
    """Normalized prediction error per offset, category and condition."""
    W = tr["W"]
    B, L = W.shape[0], W.shape[1]
    cond = onp.asarray(batch["condition"])
    out = {}
    for k in K_OFFSETS:
        T_ = L - k
        D = (X[:, :T_] - W[:, k:]).astype(onp.float64)
        Bs = (W[:, :T_] - W[:, k:]).astype(onp.float64)
        fro_num = onp.linalg.norm(D, axis=(-2, -1))
        fro_den = onp.linalg.norm(Bs, axis=(-2, -1))
        rev_q = keys[onp.clip(rev_at[:, :T_], 0, None)]
        has_rev = rev_at[:, :T_] >= 0
        rev_num = onp.linalg.norm(onp.einsum("btvw,btw->btv", D, rev_q),
                                  axis=-1)
        rev_den = onp.linalg.norm(onp.einsum("btvw,btw->btv", Bs, rev_q),
                                  axis=-1)
        uq = keys[unt_keys]                                   # (B, n_unt, dk)
        unt_num = onp.linalg.norm(onp.einsum("btvw,buw->btuv", D, uq),
                                  axis=-1).mean(-1)
        unt_den = onp.linalg.norm(onp.einsum("btvw,buw->btuv", Bs, uq),
                                  axis=-1).mean(-1)
        for cname in CONDITIONS:
            m = onp.ones((B, T_), dtype=bool)
            if cname != "both":
                m = m & (cond == TT.CONDITIONS.index(cname))[:, None]
            out[f"overall/{cname}/k{k}"] = _ratio(fro_num, fro_den, m)
            out[f"revised_direction/{cname}/k{k}"] = _ratio(
                rev_num, rev_den, m & has_rev)
            out[f"untouched_direction/{cname}/k{k}"] = _ratio(
                unt_num, unt_den, m)
    return out


def component_split(tr, batch):
    """The two parts of dW, reported so a filter that beats literal TSS can be
    read as extrapolating the write term, the decay term, or both."""
    W, U, alpha, beta = tr["W"], tr["U"], tr["alpha"], tr["beta"]
    write = -beta[..., None, None] * U
    prev = onp.concatenate([onp.zeros_like(W[:, :1]), W[:, :-1]], axis=1)
    decay = (alpha[..., None, None] - 1.0) * prev
    dW = W - prev
    nw = onp.linalg.norm(write.astype(onp.float64), axis=(-2, -1))
    nd = onp.linalg.norm(decay.astype(onp.float64), axis=(-2, -1))
    nt = onp.linalg.norm(dW.astype(onp.float64), axis=(-2, -1))
    ev = onp.asarray(batch["event"])
    cond = onp.asarray(batch["condition"])
    out = dict(identity_max_abs=float(onp.max(onp.abs(
        (write + decay - dW).astype(onp.float64)))))
    for cname in CONDITIONS:
        m = onp.ones_like(nt, dtype=bool)
        if cname != "both":
            m = m & (cond == TT.CONDITIONS.index(cname))[:, None]
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
    idx_t = onp.asarray([[r + o for o in SECONDARY_OFFSETS]
                         for _, r in rows])
    Xs = X[idx_b[:, None], idx_t].astype(onp.float64)
    out = {}
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
    out["offsets"] = list(SECONDARY_OFFSETS)
    return out


# ------------------------------------------------------- report and verdict --
def argmins(metrics_by_arm):
    """The best arm per category, condition and offset - reported instead of
    a single aggregate."""
    out = {}
    keys = next(iter(metrics_by_arm.values())).keys()
    for key in keys:
        best = None
        for name, m in metrics_by_arm.items():
            v = m[key]["mean"]
            if v is None:
                continue
            if best is None or v < best[1]:
                best = (name, v)
        out[key] = dict(arm=best[0], mean=best[1]) if best else None
    return out


def stopping_rule(metrics_by_arm, arm_list):
    """PREDECLARED (protocol s6): the generalized interior helps on the read
    path only if its best member beats the best literal-TSS member on the
    overall/both category by at least STOP_MARGIN at at least
    STOP_MIN_OFFSETS of the five offsets. Otherwise the stage stops."""
    fam = {a["name"]: a["family"] for a in arm_list}

    def best(family, key):
        vals = [(n, m[key]["mean"]) for n, m in metrics_by_arm.items()
                if fam[n] == family and m[key]["mean"] is not None]
        return min(vals, key=lambda x: x[1]) if vals else None
    per_k, wins = {}, 0
    for k in K_OFFSETS:
        key = f"overall/both/k{k}"
        gi, tss = best("generalized_interior", key), best("literal_tss", key)
        if gi is None or tss is None:
            per_k[key] = dict(generalized_interior=gi, literal_tss=tss,
                              wins=False)
            continue
        win = bool(gi[1] <= tss[1] - STOP_MARGIN)
        wins += int(win)
        per_k[key] = dict(generalized_interior=dict(arm=gi[0], mean=gi[1]),
                          literal_tss=dict(arm=tss[0], mean=tss[1]),
                          difference=gi[1] - tss[1], wins=win)
    helped = bool(wins >= STOP_MIN_OFFSETS)
    return dict(margin=STOP_MARGIN, offsets_required=STOP_MIN_OFFSETS,
                offsets_won=wins, per_offset=per_k,
                interior_beats_literal_tss=helped,
                verdict=("the generalized interior beats the literal-TSS line "
                         "at predicting the near-future fast weight"
                         if helped else
                         "STOP: no generalized-interior member beats the "
                         "literal-TSS line at predicting W_(t+k); the added "
                         "freedom does not help on the read path"),
                note=("declared before execution; grids and categories are "
                      "not widened or re-cut after seeing results, and no "
                      "trained comparison is authorized by this result"))


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
        grids=dict(two_tap_kappa=list(TWO_TAP_KAPPA), tss_T=list(TSS_T),
                   generalized_M=list(GEN_M), generalized_gamma=list(GEN_GAMMA),
                   generalized_T=list(GEN_T),
                   lookahead_lambda=list(LOOKAHEAD_LAMBDA)),
        arms=[a["name"] for a in arm_list], n_arms=len(arm_list),
        measurement=("||X_t - W_(t+k)||_F / ||W_t - W_(t+k)||_F; 1.0 means no "
                     "better than reading the native fast weight"),
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

    per_seed, timings = {}, {}
    for i, seed in enumerate(SEEDS):
        t_seed = time.time()
        p = refs[seed]
        fails, tr = trajectories(p, batch)
        if fails:
            status["failed"] = f"trajectory checks failed for {seed}: {fails}"
            return 4, "FAILED"
        keys = onp.asarray(NMD.safe_normalize(p["key_raw"])[0], onp.float64)
        rev_at, unt_keys, keys = episode_directions(batch, keys)
        rw = onp.asarray(p["readout_W"], onp.float64)
        rb = onp.asarray(p["readout_b"], onp.float64)
        metrics, secondary = {}, {}
        t_arm0 = time.time()
        for j, a in enumerate(arm_list):
            X = readout(a, tr)
            if not onp.all(onp.isfinite(X)):
                status["failed"] = f"non-finite readout {a['name']} ({seed})"
                return 4, "FAILED"
            metrics[a["name"]] = arm_metrics(X, tr, batch, rev_at, unt_keys,
                                             keys)
            secondary[a["name"]] = secondary_curves(X, tr, batch, keys, rw,
                                                    rb)
            if j == 0:
                timings["seconds_per_arm"] = time.time() - t_arm0
                left = deadline - time.time() - args.reserve_s
                need = timings["seconds_per_arm"] * len(arm_list) * (
                    len(SEEDS) - i)
                timings["projected_remaining_s"] = need
                print(f"PROBE_PROJECTED_REMAINING_S={need:.1f}")
                if need > left:
                    why = (f"projected {need:.0f}s > remaining {left:.0f}s; "
                           "not started, nothing reduced")
                    status["incomplete"].append(why)
                    print(f"[!] {why}")
                    return 3, "INCOMPLETE"
            if time.time() > deadline - args.reserve_s:
                status["incomplete"].append(
                    f"stopped at arm {a['name']} of seed {seed}")
                return 3, "INCOMPLETE"
        per_seed[str(seed)] = dict(
            metrics=metrics, secondary=secondary,
            component_split=component_split(tr, batch),
            native_point_logit_agreement=tr["native_point_logit_agreement"],
            argmins=argmins(metrics), stopping_rule=stopping_rule(metrics,
                                                                  arm_list),
            wall_s=time.time() - t_seed)
        status["per_seed"] = per_seed
        save()
        print(f"[seed {seed}] interior beats literal TSS: "
              f"{per_seed[str(seed)]['stopping_rule']['interior_beats_literal_tss']}")
    pooled = {}
    for name in (a["name"] for a in arm_list):
        pooled[name] = {}
        for key in per_seed[str(SEEDS[0])]["metrics"][name]:
            vals = [per_seed[str(s)]["metrics"][name][key]["mean"]
                    for s in SEEDS]
            pooled[name][key] = dict(
                mean=(float(onp.mean(vals)) if all(v is not None for v in
                                                   vals) else None),
                per_seed={str(s): per_seed[str(s)]["metrics"][name][key][
                    "mean"] for s in SEEDS})
    status["pooled"] = dict(metrics=pooled, argmins=argmins(pooled),
                            stopping_rule=stopping_rule(pooled, arm_list))
    status["timings"] = timings
    status["work_completed"] = dict(seeds=len(SEEDS), arms=len(arm_list),
                                    episodes=int(batch["event"].shape[0]),
                                    offsets=list(K_OFFSETS), updates=0)
    sr = status["pooled"]["stopping_rule"]
    print(f"[stopping rule] interior_beats_literal_tss="
          f"{sr['interior_beats_literal_tss']} offsets_won={sr['offsets_won']}")
    status["complete"] = True
    return 0, "PASS"


if __name__ == "__main__":
    sys.exit(main())
