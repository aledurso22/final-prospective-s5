"""Literal Nesterov and QHM on the MDN write path, and the controlled ladder.

Audit and pre-registered rules: docs/MDN_NESTEROV_QHM_AUDIT.md. NOT AUTHORIZED
TO RUN until that document's protocol section is cleared.

Two NEW write-path rules, executed in exactly the shell of
`experiments.prospective_momentum.model.rollout` (same helpers, same order;
checked numerically against the native rollout in float64):

    nesterov_momentum  (literal Nesterov; no parameter)
        Wbar = alpha W ;  L = Wbar - (beta mu) U
        R = m (L k - v) k^T ;  U' = mu U + eta R ;  W' = Wbar - beta U'
    qhm_momentum       (quasi-hyperbolic momentum; one scalar nu in [0, 1])
        Wbar = alpha W ;  R = m (Wbar k - v) k^T ;  U' = mu U + eta R
        W' = Wbar - beta [nu U' + (1 - nu) eta R]

The lookahead coefficient c_t = beta_t mu_t is DERIVED, not chosen: it is the
momentum part of this token's own step, so L_t is the point the step reaches
before its gradient correction (Sutskever's convention, translated to the MDN
gates, after the native decay as MDN Eqs. (4)-(5) apply it).

The ladder reuses the completed temporal-response protocol UNCHANGED: task,
episode construction, sources (seed 500 development, 501-503 final, read
only), stream seeds, learning rates, checkpoints, selection ordering and
metric definitions. In display order the arms are Native Momentum
DeltaNet, Zucchet prospective dynamics with matched time constants, Zucchet
prospective dynamics with a learned time-constant mismatch (the completed
implementation, unchanged), Generalized prospective dynamics (M, gamma, T),
Literal Nesterov Momentum DeltaNet and QHM Momentum DeltaNet. The first four
arms are executed by the completed study's own
code paths (`temporal_response.host_step`, `tss_containment`), not
re-implemented here. Every completed module is left
byte-identical; the two new rules live in this file. Registry additions are
`setdefault` only.
"""

import argparse
import copy
import math
import os
import signal
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as onp
import optax

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from experiments.adaptive_memory import model as AM               # noqa: E402
from experiments.meta_delta import study as MS                    # noqa: E402
from experiments.nested_memory import dynamics as NMD             # noqa: E402
from experiments.nested_memory import model as NM                 # noqa: E402
from experiments.nested_memory import task as TK                  # noqa: E402
from experiments.nested_memory.task import WRITE                  # noqa: E402
from experiments.prospective_momentum import dynamics as PD       # noqa: E402
from experiments.prospective_momentum import filtered as FL       # noqa: E402
from experiments.prospective_momentum import model as PM          # noqa: E402
from experiments.prospective_momentum import ordinary as OD       # noqa: E402
from experiments.prospective_momentum import replication_sources as RS  # noqa
from experiments.prospective_momentum import study as ST          # noqa: E402
from experiments.prospective_momentum import temporal_response as TR  # noqa
from experiments.prospective_momentum import temporal_task as TT  # noqa: E402
from experiments.prospective_momentum import tss_containment as TC  # noqa

# ------------------------------------------------------------------ rules --
NESTEROV = "nesterov_momentum"
QHM = "qhm_momentum"
#: TEST-ONLY: the native step executed in THIS file's shell, to check that the
#: shell is the production one
NATIVE_SHELL = "native_in_ladder_shell"
NEW_RULES = (NESTEROV, QHM)
QHM_LEAF = "qhm_nu"
#: declared QHM start: nu = 1 is the native function exactly
QHM_NU0 = 1.0
QHM_DOMAIN = (0.0, 1.0)
CARRY_REALS = 2 * NM.D_V * NM.D_K                 # (W, U), as native

PD.DISPLAY.setdefault(NESTEROV, "Literal Nesterov Momentum DeltaNet")
PD.DISPLAY.setdefault(QHM, "QHM Momentum DeltaNet")
PD.CARRY.setdefault(NESTEROV, CARRY_REALS)
PD.CARRY.setdefault(QHM, CARRY_REALS)

# ------------------------------------------------------------------- arms --
NATIVE, TSS, GEN, ANCHOR = TC.NATIVE, TC.TSS, TC.GEN, TC.ANCHOR
#: the completed studies' learned two-tap operator (`ordinary_prospective`),
#: executed by EXACTLY their code paths: same rule, kappa start 0, projection,
#: validation, checkpoints, selection ordering and evaluation
OPERATOR = TC.OPERATOR
NAG_ARM, QHM_ARM = "literal_nesterov", "qhm"
#: the declared ladder, in order
LADDER = (NATIVE, OPERATOR, TSS, NAG_ARM, QHM_ARM, GEN)
LAW = {NATIVE: "momentum_delta", OPERATOR: OD.ORDINARY, TSS: FL.FILTERED,
       GEN: FL.FILTERED, NAG_ARM: NESTEROV, QHM_ARM: QHM,
       ANCHOR: "momentum_delta"}
NEW_ARMS = (NAG_ARM, QHM_ARM)
#: SCIENTIFIC names, the ONLY method names used in user-facing outputs. The
#: arm keys above are internal repository identifiers.
#:   Zucchet matched  = Zucchet et al. Eqs. (5) and (17), tau' = tau (= T,
#:                      learned);
#:   Zucchet mismatch = Zucchet et al. Eq. (10), tau u' = -u + f + tau' f',
#:                      discretized as Eq. (17), membrane tau fixed at one
#:                      step h and tau' = kappa h learned;
#:   both are the audited recurrence at M = 0, gamma = tau - tau', T = tau'.
SCIENTIFIC_NAME = {
    NATIVE: "Native Momentum DeltaNet",
    TSS: "Zucchet prospective dynamics, matched time constants \u03c4\u2032=\u03c4",
    OPERATOR: ("Zucchet prospective dynamics, learned time-constant mismatch "
               "\u03c4\u2032\u2260\u03c4"),
    GEN: "Generalized prospective dynamics (M,\u03b3,T)",
    NAG_ARM: "Literal Nesterov Momentum DeltaNet",
    QHM_ARM: "QHM Momentum DeltaNet",
    ANCHOR: "Native Momentum DeltaNet"}
#: the user-facing display order
DISPLAY_ORDER = (NATIVE, TSS, OPERATOR, GEN, NAG_ARM, QHM_ARM)
#: scope of the generalized verdict (declared before execution)
GENERALIZED_SCOPE = (
    "Generalized prospective dynamics (M,γ,T) is tested on its declared "
    "domain γ≥0 (M,T≥0), with the completed protocol's feasibility "
    "repair. The verdict is scoped to that declared domain; this is not the "
    "unrestricted generalized family. Endpoints of Zucchet prospective "
    "dynamics, learned time-constant mismatch τ′≠τ with τ′>τ "
    "require γ=τ−τ′<0 and are not contained in the tested arm.")
REALIZATION_CLASSIFICATION = "EXACTLY_EQUIVALENT_REALIZATIONS"
#: the controls generalized prospectivity must beat INDIVIDUALLY
CONTROLS = (NATIVE, OPERATOR, TSS, NAG_ARM, QHM_ARM)
#: user-facing display strings are the scientific names
DISPLAY = {a: SCIENTIFIC_NAME[a] for a in LAW}
EXTRA_PARAMETERS = {NATIVE: 0, OPERATOR: 1, TSS: 1, NAG_ARM: 0, QHM_ARM: 1,
                    GEN: 3}
CARRY_EXECUTED = {NATIVE: 2 * NM.D_V * NM.D_K, OPERATOR: OD.CARRY_REALS,
                  TSS: FL.CARRY_EXECUTED, NAG_ARM: CARRY_REALS,
                  QHM_ARM: CARRY_REALS, GEN: FL.CARRY_EXECUTED}
#: extra per-token work over native, in multiply-adds on the d_v x d_k state
EXTRA_WORK = {NATIVE: "none",
              OPERATOR: "prospective input: 1 matrix axpy; previous residual "
                        "carried",
              TSS: "processing filter: 4 matrix axpys",
              NAG_ARM: "lookahead L = Wbar - (beta mu) U: 1 matrix axpy",
              QHM_ARM: "mixed step nu U' + (1-nu) eta R: 2 matrix axpys",
              GEN: "processing filter: 4 matrix axpys"}

# the completed protocol, reused unchanged
UPDATES, LRS, VAL_AT = TR.UPDATES, TR.LRS, TR.VAL_AT
BATCH_PER_FAMILY = TR.BATCH_PER_FAMILY
VAL_PER_FAMILY, HELDOUT_PER_FAMILY = TR.VAL_PER_FAMILY, TR.HELDOUT_PER_FAMILY
SOURCE_RUN, SOURCE_DEV, SOURCE_FINAL = TR.SOURCE_RUN, TR.SOURCE_DEV, \
    TR.SOURCE_FINAL
STREAM = TR.STREAM                     # the SAME streams as the completed run
COMPLETED_RUN = ("/Users/durso/s5-runs/prospective-temporal-response/"
                 "20260917-163003")

# ------------------------------------------------ pre-registered analysis --
#: episode groups for the paired jackknife: blocks of 4 consecutive episodes
#: (one of every condition x order-flip), so every group is cell-balanced
N_GROUPS = 32
#: decision margin in accuracy, the protocol's existing +1 pp
MARGIN = 0.01
Z95 = 1.959963984540054
METRICS = {
    "revision": lambda m: m["primary"],
    "retention": lambda m: m["retention_revision_untouched"],
    "recall": lambda m: m["recall_overall"],
    "immediate_revised": lambda m: m["immediate_revision"],
    "later_revised": lambda m: m["later_parts"]["revised_later"],
    "later": lambda m: m["later"],
    "untouched_keys": lambda m: m["revision"]["parts"]["untouched_probe"],
    "revised_idle_gap": lambda m: m["revision"]["by_condition"][
        "revised_probe"]["idle_gap"],
    "revised_intervening_writes": lambda m: m["revision"]["by_condition"][
        "revised_probe"]["intervening_writes"],
    "untouched_idle_gap": lambda m: m["revision"]["by_condition"][
        "untouched_probe"]["idle_gap"],
    "untouched_intervening_writes": lambda m: m["revision"]["by_condition"][
        "untouched_probe"]["intervening_writes"],
}
#: EVERY pair of the ladder, oriented later-versus-earlier in ladder order
COMPARISONS = tuple((LADDER[j], LADDER[i]) for j in range(len(LADDER))
                    for i in range(j))
#: The FIVE PLANNED PRIMARY CONTRASTS: Generalized prospective dynamics versus
#: native, Zucchet matched, Zucchet learned mismatch, literal Nesterov and
#: QHM. The other ten pairs are DESCRIPTIVE.
PRIMARY_CONTRASTS = tuple(f"{GEN}_vs_{c}" for c in
                          (NATIVE, TSS, OPERATOR, NAG_ARM, QHM_ARM))
#: the metrics reported for every planned primary contrast
PRIMARY_METRICS = ("revision", "retention", "recall", "immediate_revised",
                   "later_revised")
FLIP = {"BETTER": "WORSE", "WORSE": "BETTER",
        "BETTER_BELOW_MARGIN": "WORSE_BELOW_MARGIN",
        "WORSE_BELOW_MARGIN": "BETTER_BELOW_MARGIN",
        "EQUIVALENT_WITHIN_MARGIN": "EQUIVALENT_WITHIN_MARGIN",
        "INDETERMINATE": "INDETERMINATE"}
#: the frozen-token condition of literal Nesterov (audit s2), unit key
NAG_CONDITION = "(beta eta - 1)(alpha + mu + alpha mu) < 1"
RECOMMENDATIONS = ("REDUNDANT_CONTROL_CONFIRMED",
                   "RUN_LITERAL_NESTEROV_AT_SCALE",
                   "GENERALIZED_GP_RETAINS_DISTINCT_ADVANTAGE",
                   "GENERALIZED_GP_REDUCES_TO_KNOWN_OPTIMIZER", "NO_GO")


# ------------------------------------------------------------------ steps --
def nesterov_step(carry, k, v, m, alpha, beta, mu, eta, scalar=None):
    """Literal Nesterov: the residual at L_t = Wbar_t - beta_t mu_t U_(t-1).
    `scalar` is unused (the signature of the production steps)."""
    W, U = carry
    Wb = alpha * W
    L = Wb - (beta * mu) * U
    R = m * jnp.outer(L @ k - v, k)
    U_new = mu * U + eta * R
    return (Wb - beta * U_new, U_new)


def qhm_step(carry, k, v, m, alpha, beta, mu, eta, nu):
    """QHM: W' = Wbar - beta [nu U' + (1 - nu) eta R]; nu = 1 is native."""
    W, U = carry
    Wb = alpha * W
    R = m * jnp.outer(Wb @ k - v, k)
    U_new = mu * U + eta * R
    return (Wb - beta * (nu * U_new + (1.0 - nu) * (eta * R)), U_new)


def native_shell_step(carry, k, v, m, alpha, beta, mu, eta, scalar=None):
    return NMD.momentum_delta_step(carry, k, v, m, alpha, beta, mu, eta)


STEPS = {NESTEROV: nesterov_step, QHM: qhm_step,
         NATIVE_SHELL: native_shell_step}


def rollout(rule, p, ep, dtype=None, trace=False):
    """One episode in the production shell (lines of `model.rollout`, same
    helpers and order); only the per-token step differs."""
    if rule not in STEPS:
        raise ValueError(f"{rule!r} is not a rule of this module")
    if dtype is None:
        dtype = p["key_raw"].dtype
    ep = AM.sanitize_episode(ep)
    key_id, val_id, event = ep["key_id"], ep["val_id"], ep["event"]
    k_all, k_valid = NMD.safe_normalize(p["key_raw"])
    keys = k_all[key_id].astype(dtype)
    valid = k_valid[key_id]
    has_v = (val_id >= 0).astype(dtype)
    vals = (p["value_table"][jnp.maximum(val_id, 0)]
            * has_v[:, None]).astype(dtype)
    mask = ((event == WRITE).astype(dtype)) * valid.astype(dtype)
    gx = NM.gate_features(key_id, val_id, event).astype(dtype)
    alpha, beta, mu, eta = NM._momentum_gates(p, gx)
    step_fn = STEPS[rule]
    scalar = (p[QHM_LEAF][0] if rule == QHM
              else jnp.zeros((), dtype=dtype))

    def step(carry, t):
        carry = step_fn(carry, keys[t], vals[t], mask[t], alpha[t], beta[t],
                        mu[t], eta[t], scalar)
        W = carry[0]
        logits = p["readout_W"] @ (W @ keys[t]) + p["readout_b"]
        outs = (logits, jnp.sqrt(jnp.sum(W ** 2)),
                jnp.sqrt(jnp.sum(carry[1] ** 2)))
        if trace:
            outs = outs + (carry[0], carry[1])
        return carry, outs

    z = jnp.zeros((NM.D_V, NM.D_K), dtype=dtype)
    carry, outs = jax.lax.scan(step, (z, z), jnp.arange(key_id.shape[0]))
    ret = dict(logits=outs[0], w_norm=outs[1], aux_norm=outs[2],
               gates=(alpha, beta, mu, eta), final_carry=carry, dtype=dtype,
               coeff=({QHM_LEAF: scalar} if rule == QHM else {}))
    if trace:
        ret["W_trace"], ret["U_trace"] = outs[3], outs[4]
    return ret


# --------------------------------------------------- compiled train / eval --
def _episode_loss(rule, p, ep):
    """The unchanged query cross-entropy (`study._episode_loss`)."""
    out = rollout(rule, p, ep)
    q = (ep["event"] == TK.QUERY)
    lab = jnp.maximum(ep["label"], 0)
    ce = optax.softmax_cross_entropy(
        out["logits"], jax.nn.one_hot(lab, TK.N_VALUES)) * q
    correct = (jnp.argmax(out["logits"], -1) == lab) * q
    aux = dict(ce=ce, correct=correct.astype(jnp.float32),
               q=q.astype(jnp.float32), w_norm=out["w_norm"],
               aux_norm=out["aux_norm"])
    if rule == NESTEROV:
        # the gates RETURNED by this rollout, for the episode-level
        # applicability record of the declared frozen-token condition
        aux["gates"] = jnp.stack(out["gates"], axis=-1)
        # a non-finite logit is a Nesterov FAILURE on that query: it counts
        # as incorrect (argmax of a NaN row is never taken as an answer) and
        # the episode stays in the denominator
        fin = jnp.all(jnp.isfinite(out["logits"]), axis=-1)
        aux["finite_tokens"] = fin
        aux["correct"] = aux["correct"] * fin.astype(jnp.float32)
    return jnp.sum(ce) / jnp.maximum(jnp.sum(q), 1.0), aux


def batch_loss(rule, p, eps):
    losses, aux = jax.vmap(lambda e: _episode_loss(rule, p, e))(eps)
    return jnp.mean(losses), aux


def project(p):
    """Post-update projection of nu onto [0, 1] from the UPDATED value; the
    optimizer state is untouched. A no-op (NaN telemetry) without nu."""
    if QHM_LEAF in p:
        pre = p[QHM_LEAF]
        post = jnp.clip(pre, QHM_DOMAIN[0], QHM_DOMAIN[1])
        one = jnp.ones((), dtype=pre.dtype)
        tel = dict(n_projected=jnp.sum(post != pre), pre=pre[0], post=post[0],
                   cap=one, bound=one,
                   overshoot=jnp.max(jnp.abs(pre - post)))
        return dict(p, **{QHM_LEAF: post}), tel
    nan = jnp.full((), jnp.nan, dtype=p["key_raw"].dtype)
    return p, dict(n_projected=jnp.asarray(0), pre=nan, post=nan, cap=nan,
                   bound=nan, overshoot=nan)


@partial(jax.jit, static_argnums=(0,))
def train_step(rule, p, opt, eps, lr):
    """`study.train_step` line for line, with this module's loss and
    projection. Same optimizer (`ST.TX`), same update, same scalars."""
    (loss, aux), g = jax.value_and_grad(batch_loss, argnums=1, has_aux=True)(
        rule, p, eps)
    raw, opt = ST.TX.update(g, opt, p)
    upd = jax.tree_util.tree_map(lambda u: -lr * u, raw)
    p = optax.apply_updates(p, upd)
    p, tel = project(p)
    acc = jnp.sum(aux["correct"]) / jnp.maximum(jnp.sum(aux["q"]), 1.0)
    extra_g = (g[QHM_LEAF][0] if rule == QHM
               else jnp.zeros((), dtype=loss.dtype))
    return (p, opt, loss, acc, optax.global_norm(g), optax.global_norm(upd),
            jnp.mean(aux["w_norm"]), jnp.mean(aux["aux_norm"]), tel, extra_g)


@partial(jax.jit, static_argnums=(0,))
def eval_batch(rule, p, eps):
    _, aux = batch_loss(rule, p, eps)
    return aux


# ------------------------------------------------------------- start trees --
def start_tree(arm, native_p):
    """Nesterov: the native tree, unchanged (no parameter). QHM: the native
    tree plus nu = 1 (the native function). Other arms: the completed
    study's own map."""
    law = LAW[arm]
    if law in NEW_RULES:
        missing = [k for k in PM.MOMENTUM_LEAVES if k not in native_p]
        extra = [k for k in native_p if k not in PM.MOMENTUM_LEAVES]
        if missing or extra:
            raise ValueError(f"not a native Momentum tree: missing {missing}, "
                             f"unexpected {extra}")
        if law == NESTEROV:
            return dict(native_p)
        dt = native_p["A_log"].dtype
        return dict(native_p, **{QHM_LEAF: jnp.full((1,), QHM_NU0, dtype=dt)})
    return TC.start_tree(arm, native_p)


# -------------------------------------------------- frozen-token diagnostic --
def table_report(arm, p):
    """TABLE coverage: exact classification of the rounded executed
    key-aligned 2x2 transitions of the PRODUCTION step over the write table,
    and the float64 closed-form Nesterov condition
    (q - 1)(alpha + mu + alpha mu) < 1. A frozen-token diagnostic only."""
    dtype = p["key_raw"].dtype
    gates = PD.table_gates(p, PD.write_table())
    law = LAW[arm]
    scalar = (p[QHM_LEAF][0] if law == QHM else jnp.zeros((), dtype=dtype))
    A = onp.asarray(PD.executed_transitions(STEPS[law], gates, scalar, dtype))
    labels, worst = {}, None
    for i in range(A.shape[0]):
        lab, J = PD.classify(A[i])
        labels[lab] = labels.get(lab, 0) + 1
        if J is not None and (worst is None or min(J) < worst):
            worst = min(J)
    a, b, mu, eta = (onp.asarray(x, onp.float64) for x in gates)
    rep = dict(rule=law, n_write_settings=int(A.shape[0]),
               classification=labels, min_jury_expression=worst,
               stable_everywhere=bool(set(labels) == {"stable"}))
    if law == NESTEROV:
        lhs = (b * eta - 1.0) * (a + mu + a * mu)
        rep["closed_form_condition"] = dict(
            statement="(q - 1)(alpha + mu + alpha mu) < 1, q = beta eta",
            max_lhs=float(lhs.max()),
            fraction_violating=float(onp.mean(lhs >= 1.0)))
    else:
        rep["nu"] = float(onp.asarray(p[QHM_LEAF], onp.float64).ravel()[0])
    return rep


def validate_new(arm, p):
    """None if acceptable. Nesterov's frozen-token instability is NOT a
    failure (declared: reported, and the checkpoint is ineligible for
    selection); non-finite values always fail. QHM is frozen-token stable on
    its whole domain by derivation, so an unstable QHM transition fails as a
    contradiction requiring diagnosis."""
    if not ST.all_finite(p):
        return "non-finite parameters"
    gr = PD.gate_range_report(p)
    if not gr["gates_valid"]:
        return f"gate range/finiteness failed: {gr}"
    if LAW[arm] == QHM:
        nu = float(onp.asarray(p[QHM_LEAF], onp.float64).ravel()[0])
        if not (onp.isfinite(nu) and QHM_DOMAIN[0] <= nu <= QHM_DOMAIN[1]):
            return f"nu {nu} outside {QHM_DOMAIN}"
    rep = table_report(arm, p)
    if rep["classification"].get("nonfinite"):
        return f"non-finite executed transitions {rep['classification']}"
    if LAW[arm] == QHM and rep["classification"].get("unstable"):
        return (f"QHM frozen-token transition unstable {rep['classification']}"
                " (contradicts the derivation; diagnosis required)")
    return None


# ------------------------------------------------------------- evaluation --
def evaluate(arm, p, eps_np, chunk=128, keep_arrays=False):
    """The declared metrics (`temporal_task.aggregate`) with the SAME compiled
    batch functions the completed study used for its arms, and this module's
    for the new ones. Optionally returns the per-token correct/CE arrays for
    the paired analysis."""
    law = LAW[arm]
    if law in NEW_RULES:
        batch_fn = eval_batch
    else:
        batch_fn = TC.eval_batch_f if law == FL.FILTERED else ST.eval_batch
    n = eps_np["event"].shape[0]
    cs, ces, wn, an, procs, sink, gts, fins = ([], [], [], [], [], [], [],
                                               [])
    for i in range(0, n, chunk):
        sl = {k: jnp.asarray(eps_np[k][i:i + chunk]) for k in TT.MODEL_INPUTS}
        aux = batch_fn(law, p, sl)
        if law == NESTEROV:
            gts.append(onp.asarray(aux["gates"], onp.float64))
            fins.append(onp.asarray(aux["finite_tokens"]))
        cs.append(onp.asarray(aux["correct"]))
        ces.append(onp.asarray(aux["ce"]))
        wn.append(onp.asarray(aux["w_norm"]))
        an.append(onp.asarray(aux["aux_norm"]))
        if law == FL.FILTERED:
            procs.append(onp.asarray(aux["proc_max_abs"]).ravel())
            sink.append(aux)
    correct, ce = onp.concatenate(cs), onp.concatenate(ces)
    m = TT.aggregate(correct, ce, eps_np)
    wn, an = onp.concatenate(wn), onp.concatenate(an)
    m["state_norms"] = dict(W_frobenius_mean=float(wn.mean()),
                            W_frobenius_max=float(wn.max()),
                            aux_frobenius_mean=float(an.mean()),
                            aux_frobenius_max=float(an.max()))
    sets = None
    if law == FL.FILTERED:
        proc = onp.concatenate(procs)
        m["processing_state"] = dict(
            max_abs_y_yprev_Rprev=float(proc.max()),
            finite=bool(proc.size and onp.all(onp.isfinite(proc))))
        sets = TC.executed_sets_from(sink)
        m["executed_coefficient_sets"] = sets
    if law == NESTEROV:
        m["nesterov_episode_applicability"] = episode_applicability(
            onp.concatenate(gts), eps_np["event"])
        fin = onp.concatenate(fins)
        q = onp.asarray(eps_np["event"]) == TK.QUERY
        ep_fail = ~fin.all(axis=1)
        m["nesterov_failures"] = dict(
            definition=("a token whose logits are non-finite; its query "
                        "counts as INCORRECT and the episode stays in the "
                        "denominator"),
            episodes=int(fin.shape[0]),
            episodes_failed=int(ep_fail.sum()),
            queries=int(q.sum()),
            queries_failed=int((~fin & q).sum()),
            state_norm_nonfinite_episodes=int(
                (~onp.isfinite(wn).reshape(fin.shape[0], -1).all(axis=1)
                 | ~onp.isfinite(an).reshape(fin.shape[0], -1).all(axis=1)
                 ).sum()))
    if keep_arrays:
        return m, sets, (correct, ce,
                         onp.concatenate(gts) if gts else None)
    return m, sets


def episode_violations(gates, event):
    """Per episode: the number of WRITE tokens whose executed gates violate
    literal Nesterov's frozen-token condition, evaluated in float64 on the
    gates the rollout returned. `gates` is (episodes, tokens, 4)."""
    a, b, mu, eta = (gates[..., i] for i in range(4))
    lhs = (b * eta - 1.0) * (a + mu + a * mu)
    bad = (lhs >= 1.0) & (onp.asarray(event) == WRITE)
    return bad.sum(axis=1), lhs


def episode_applicability(gates, event):
    n_bad, lhs = episode_violations(gates, event)
    writes = onp.asarray(event) == WRITE
    return dict(condition=NAG_CONDITION,
                episodes=int(n_bad.size),
                episodes_with_a_violating_write=int(onp.sum(n_bad > 0)),
                write_tokens=int(writes.sum()),
                violating_write_tokens=int(n_bad.sum()),
                max_lhs_on_writes=(float(lhs[writes].max())
                                   if writes.any() else None))


def checkpoint_failure(arm, p, opt, source_p, metrics, executed_sets):
    if arm not in NEW_ARMS:
        return TR.checkpoint_failure(arm, p, opt, source_p, metrics,
                                     executed_sets)
    if not ST.all_finite(p):
        return "non-finite parameters"
    if opt is not None and not ST.all_finite(opt):
        return "non-finite optimizer state"
    if not TR.metrics_finite(metrics):
        return "non-finite or empty evaluation cell, aggregate or state norm"
    return validate_new(arm, p)


# ------------------------------------------------------------ host steps --
def host_step(arm, p, opt, seed, u, lr, hist):
    """Completed arms: the completed study's host step, unchanged. New arms:
    the same stream, batch and optimizer, with this module's train step."""
    if arm not in NEW_ARMS:
        return TR.host_step(arm, p, opt, seed, u, lr, hist)
    eps = ST.to_jax(TT.generate_batch(TR.continuation_stream(seed, u),
                                      BATCH_PER_FAMILY))
    o = train_step(LAW[arm], p, opt, eps, lr)
    p, opt, tel = o[0], o[1], o[8]
    rec = dict(update=u, n_projected=int(tel["n_projected"]),
               pre=float(tel["pre"]), post=float(tel["post"]),
               cap=float(tel["cap"]), bound=float(tel["bound"]),
               overshoot=float(tel["overshoot"]), grad=float(o[9]),
               executed_filter_ok=True)
    scalars = dict(zip(ST.SCALAR_NAMES, (float(x) for x in o[2:8])))
    hist.append(rec)
    return p, opt, scalars, rec


def step_failure(arm, scalars, rec):
    if arm not in NEW_ARMS:
        return TC.step_failure(arm, scalars, rec)
    bad = MS.measured_scalar_failures(scalars)
    if bad:
        return f"non-finite step scalars {bad} at update {rec['update']}"
    if LAW[arm] == QHM:
        if not (onp.isfinite(rec["post"]) and onp.isfinite(rec["pre"])
                and QHM_DOMAIN[0] <= rec["post"] <= QHM_DOMAIN[1]):
            return f"nu telemetry {rec['pre']}->{rec['post']} at {rec['update']}"
    return None


def parameter_counts(arm, p):
    if arm not in NEW_ARMS:
        c = TC.parameter_counts(arm, p)
    else:
        c = PM.parameter_counts(LAW[arm], p)
        c.update(stored=c["total"], trainable=c["total"], frozen_constants=0,
                 regime="full")
    c["extra_parameters_declared"] = EXTRA_PARAMETERS.get(arm, 0)
    c["carry_real_numbers_executed"] = CARRY_EXECUTED.get(arm)
    c["extra_work_per_token"] = EXTRA_WORK.get(arm)
    return c


def frozen_leaf_differences(p, source_p, arm):
    return ({} if arm in NEW_ARMS
            else TC.frozen_leaf_differences(p, source_p, arm))


# ---------------------------------------------------------------- one run --
def run_one(arm, tag, lr_value, seed, source_p, val_np, updates, out,
            deadline, reserve_s, status, stage, source_label, save,
            keep_at=()):
    """`temporal_response.run_one`, dispatching the step, evaluation and
    acceptance per arm. Every checkpoint is evaluated, accepted and persisted
    before the next update. New arms also record the frozen-token table and a
    selection-eligibility flag (Nesterov's instability makes a checkpoint
    INELIGIBLE, not a failure)."""
    regime = "frozen" if arm == ANCHOR else "full"
    t0 = time.time()
    p = dict(source_p)
    ckpts = tuple(u for u in VAL_AT if u <= updates)
    if updates not in ckpts:
        ckpts = ckpts + (updates,)
    stem = f"{stage}_{arm}_{tag}_seed{seed}"
    log = status.setdefault("checkpoint_log", [])
    opt = None if regime == "frozen" else ST.TX.init(p)
    val_hist, hist, curve, kept, bad = [], [], [], {}, None
    for u in range(updates + 1):
        if u in ckpts:
            m, sets = evaluate(arm, p, val_np)
            fail = checkpoint_failure(arm, p, opt, source_p, m, sets)
            table = table_report(arm, p) if arm in NEW_ARMS else None
            eligible = bool(fail is None and (
                table is None or LAW[arm] != NESTEROV
                or not table["classification"].get("unstable")))
            pfile = os.path.join(out, "params", f"{stem}_u{u}.msgpack")
            ST.save_tree(pfile, p)
            ofile = None
            if opt is not None:
                ofile = os.path.join(out, "params",
                                     f"{stem}_u{u}_opt.msgpack")
                ST.save_tree(ofile, opt)
            entry = dict(update=u, **TR._summary(m),
                         state_norms=m["state_norms"],
                         processing_state=m.get("processing_state"),
                         executed_filter=([FL.filter_report(ex)
                                           for ex in sets] if sets else None),
                         frozen_token_table=table, eligible=eligible,
                         accepted=fail is None, failure=fail,
                         params_file=pfile, opt_file=ofile)
            val_hist.append(dict(entry, full=m))
            log.append(dict({k: v for k, v in entry.items()
                             if k not in ("parts", "by_delay")},
                            stage=stage, arm=arm, config=tag, lr=lr_value,
                            seed=seed))
            save()
            if fail:
                bad = f"checkpoint at update {u}: {fail}"
                break
            if u in keep_at:
                kept[u] = dict(p)
        if u == updates:
            break
        if time.time() > deadline - reserve_s:
            status["incomplete"].append(
                f"{stage}:{arm}/{tag}/seed{seed} stopped at update {u} of "
                f"{updates}")
            return None
        lr = jnp.asarray(lr_value, dtype=jnp.float32)
        p, opt, scalars, rec = host_step(arm, p, opt, seed, u, lr, hist)
        bad = step_failure(arm, scalars, rec)
        if bad:
            break
        if u % 25 == 0 or u == updates - 1:
            curve.append(dict(update=u, **scalars))
    last = val_hist[-1] if val_hist else None
    if bad is None and not (last and last["update"] == updates):
        bad = "endpoint checkpoint missing"
    rec = dict(
        tag=stage, rule=arm, law=LAW[arm], regime=regime,
        display=DISPLAY[arm], config=tag, lr=lr_value, seed=seed,
        source=source_label, wall_s=time.time() - t0, curve=curve,
        updates=updates,
        validation=[{k: v for k, v in h.items() if k != "full"}
                    for h in val_hist],
        final_validation=(last["full"] if last and bad is None else None),
        params=parameter_counts(arm, p),
        coefficient_history=hist,
        frozen_leaf_differences=frozen_leaf_differences(p, source_p, arm),
        invalid=bad)
    return rec, p, kept


# --------------------------------------------------------------- preflight --
def _cache(arm):
    law = LAW[arm]
    if law in NEW_RULES:
        return train_step
    return TC.train_step_filtered if law == FL.FILTERED else ST.train_step


def preflight(sources, val_np, out, status):
    """Measured on disposable state along the ACTUAL paths: 7 steps (all
    checked), one evaluation compile, one timed evaluation and one full
    checkpoint per arm. The projection covers every declared trajectory at
    the worst case (200 updates, all checkpoints), every held-out endpoint
    evaluation, the anchors and the paired analysis."""
    rows, failures, retraced_any = [], [], False
    p0 = sources[SOURCE_DEV]
    lr = jnp.asarray(LRS[0][1], dtype=jnp.float32)
    step_s, ckpt_s, eval_s = {}, {}, {}
    scratch = os.path.join(out, "preflight_disposable")
    for arm in LADDER:
        p = start_tree(arm, p0)
        opt = ST.TX.init(p)
        hist, acc_bad = [], None
        t0 = time.time()
        p2, opt2, sc, rec = host_step(arm, p, opt, SOURCE_DEV, 0, lr, hist)
        compile_s = time.time() - t0
        acc_bad = acc_bad or step_failure(arm, sc, rec)
        p2, opt2, sc, rec = host_step(arm, p2, opt2, SOURCE_DEV, 1, lr, hist)
        acc_bad = acc_bad or step_failure(arm, sc, rec)
        cache = _cache(arm)
        n0 = cache._cache_size()
        t1 = time.time()
        for u in range(2, 7):
            p2, opt2, sc, rec = host_step(arm, p2, opt2, SOURCE_DEV, u, lr,
                                          hist)
            acc_bad = acc_bad or step_failure(arm, sc, rec)
        step_s[arm] = (time.time() - t1) / 5.0
        retraced_any |= cache._cache_size() != n0
        t2 = time.time()
        evaluate(arm, p2, val_np)
        eval_compile_s = time.time() - t2
        t3 = time.time()
        m, sets = evaluate(arm, p2, val_np)
        eval_s[arm] = time.time() - t3
        fail = checkpoint_failure(arm, p2, opt2, p, m, sets)
        if arm in NEW_ARMS:
            table_report(arm, p2)
        ST.save_tree(os.path.join(scratch, f"{arm}.msgpack"), p2)
        ST.save_tree(os.path.join(scratch, f"{arm}_opt.msgpack"), opt2)
        ckpt_s[arm] = time.time() - t3
        acc_bad = acc_bad or fail
        if acc_bad:
            failures.append(f"{arm}: {acc_bad}")
        rows.append(dict(arm=arm, law=LAW[arm], compile_s_incurred=compile_s,
                         eval_compile_s_incurred=eval_compile_s,
                         step_s=step_s[arm], checkpoint_s=ckpt_s[arm],
                         evaluation_s=eval_s[arm],
                         measured_steps_checked=len(hist),
                         acceptance_failure=acc_bad,
                         params=parameter_counts(arm, p2)))
        print(f"[preflight] {arm:<24} step {step_s[arm] * 1e3:6.2f}ms "
              f"checkpoint {ckpt_s[arm]:5.2f}s eval {eval_s[arm]:5.2f}s "
              f"compile {compile_s:4.1f}s failure {acc_bad}")
    hf = HELDOUT_PER_FAMILY / VAL_PER_FAMILY
    total = 0.0
    for arm in LADDER:
        per_traj = UPDATES * step_s[arm] + len(VAL_AT) * ckpt_s[arm]
        total += (len(LRS) + len(SOURCE_FINAL)) * per_traj
        total += len(SOURCE_FINAL) * hf * eval_s[arm]
    total += (1 + len(SOURCE_FINAL)) * max(ckpt_s.values()) \
        + len(SOURCE_FINAL) * hf * max(eval_s.values())
    host_s = 60.0                   # host bookkeeping and the paired analysis
    total += host_s
    timing = dict(projected_remaining_s=total, host_allowance_s=host_s,
                  heldout_evaluation_factor=hf)
    if not all(onp.isfinite(v) and v >= 0 for v in timing.values()):
        failures.append(f"non-finite or negative timing {timing}")
    status["preflight"] = dict(rows=rows, retraced_any=bool(retraced_any),
                               failures=failures, **timing)
    print(f"PREFLIGHT_PROJECTED_TOTAL_S={total:.1f}")
    return total, bool(retraced_any), failures


# --------------------------------------------------------------- selection --
def candidates(dev_rows, arm):
    return [c for c in TR.checkpoints(dev_rows, arm)]


def _eligible_flags(dev_rows, arm):
    flags = {}
    for r in dev_rows:
        if r["rule"] != arm:
            continue
        for v in r["validation"]:
            flags[(r["config"], v["update"])] = v.get("eligible", True)
    return flags


def select(dev_rows, status):
    """Every family by the SAME development ordering (revision macro
    accuracy, lower revision CE, fewer updates, lower learning rate) over
    accepted checkpoints. Declared exception: a Nesterov checkpoint whose
    frozen-token table has an unstable transition is INELIGIBLE (excluded,
    recorded); if none is eligible Nesterov is UNAVAILABLE. Any other missing
    or unaccepted checkpoint fails the selection, as in the completed study."""
    sel, table, unavailable = {}, [], []
    need = len(VAL_AT) * len(LRS) - 1
    for arm in LADDER:
        cand = candidates(dev_rows, arm)
        if len(cand) != need:
            return None, dict(reason=f"{arm}: {len(cand)} of {need} "
                              "development checkpoints")
        bad = TR.ineligible(cand)
        if bad:
            return None, dict(reason=f"{arm}: unaccepted or non-finite "
                              f"checkpoints {bad}")
        flags = _eligible_flags(dev_rows, arm)
        ok = [c for c in cand if flags.get((c["config"], c["update"]), True)]
        excluded = [c for c in cand if c not in ok]
        if not ok:
            unavailable.append(arm)
            table.append(dict(arm=arm, chosen=None, n_candidates=len(cand),
                              excluded_frozen_token_unstable=len(excluded)))
            continue
        best = sorted(ok, key=TC.order_key)[0]
        sel[arm] = dict(best, selected_under=(
            "unconstrained development ordering, identical for every family"))
        table.append(dict(arm=arm, chosen=best, n_candidates=len(cand),
                          excluded_frozen_token_unstable=len(excluded)))
    status["selection"] = dict(
        selected=sel, table=table, unavailable=unavailable,
        rule=("each family: highest development revision macro accuracy, "
              "then lower revision cross-entropy, then fewer updates, then "
              "lower learning rate, over accepted checkpoints; Nesterov "
              "checkpoints with an unstable frozen-token transition are "
              "excluded (declared before execution)"))
    return sel, None


def final_plan(sel):
    return [dict(arm=arm, config=sel[arm]["config"], lr=sel[arm]["lr"],
                 updates=sel[arm]["update"], keep_at=[sel[arm]["update"]])
            for arm in LADDER if arm in sel]


# -------------------------------------------------------- paired analysis --
def groups_of(n):
    """Group of each episode: blocks of 4 consecutive episodes (one of every
    condition x order flip inside a family), dealt round-robin to N_GROUPS."""
    return (onp.arange(n) // 4) % N_GROUPS


def blocks_of(n):
    return onp.arange(n) // 4


def grouped_aggregates(correct, ce, batch, keep=None):
    """Aggregate on the kept episodes and the leave-one-group-out aggregates
    over the groups still present, all by `temporal_task.aggregate` itself.
    `keep` must consist of whole 4-episode blocks, so cells stay balanced."""
    n = batch["event"].shape[0]
    keep = onp.ones(n, dtype=bool) if keep is None else onp.asarray(keep)
    g = groups_of(n)

    def agg(mask):
        sub = {k: v[mask] for k, v in batch.items()}
        return TT.aggregate(correct[mask], ce[mask], sub)
    full = agg(keep)
    loo = [agg(keep & (g != j)) for j in sorted(set(g[keep].tolist()))]
    return full, loo


def paired_seed(fa, la, fb, lb):
    """Per metric: the paired difference on the SAME episodes and its
    grouped-jackknife standard error."""
    out = {}
    G = len(la)
    for name, f in METRICS.items():
        d = float(f(fa) - f(fb))
        loo = onp.array([f(a) - f(b) for a, b in zip(la, lb)], onp.float64)
        se = float(math.sqrt((G - 1) / G * onp.sum((loo - loo.mean()) ** 2)))
        out[name] = dict(difference=d, se=se, a=float(f(fa)), b=float(f(fb)))
    return out


def verdict(D, se_w, per_seed, margin=MARGIN):
    """PRE-REGISTERED rule on the PAIRED difference and the SE OF THAT
    DIFFERENCE (never a metric's own SE against the margin).

    BETTER      lower 95% bound > 0, every seed's difference > 0, D >= margin
    WORSE       upper 95% bound < 0, every seed's difference < 0, D <= -margin
    EQUIVALENT_WITHIN_MARGIN  the whole 95% interval inside (-margin, margin)
    BETTER_BELOW_MARGIN / WORSE_BELOW_MARGIN
                significant with consistent sign but |D| < margin, interval
                not inside the margin
    INDETERMINATE  anything else"""
    lo, hi = D - Z95 * se_w, D + Z95 * se_w
    pos = all(x > 0 for x in per_seed)
    neg = all(x < 0 for x in per_seed)
    if lo > 0 and pos and D >= margin:
        return "BETTER"
    if hi < 0 and neg and D <= -margin:
        return "WORSE"
    if lo > -margin and hi < margin:
        return "EQUIVALENT_WITHIN_MARGIN"
    if lo > 0 and pos:
        return "BETTER_BELOW_MARGIN"
    if hi < 0 and neg:
        return "WORSE_BELOW_MARGIN"
    return "INDETERMINATE"


def sign(x):
    return "+" if x > 0 else ("-" if x < 0 else "0")


def combine(per_seed_stats):
    """Across final seeds: D = mean of seed differences; SE_within =
    sqrt(sum se_s^2)/n (episode sampling, trained models fixed); SE_between =
    SD of seed differences / sqrt(n) (training-seed variation, reported);
    the sign of every seed's difference, reported."""
    out = {}
    seeds = sorted(per_seed_stats)
    for name in METRICS:
        d = [per_seed_stats[s][name]["difference"] for s in seeds]
        se = [per_seed_stats[s][name]["se"] for s in seeds]
        n = len(d)
        D = float(onp.mean(d))
        se_w = float(math.sqrt(sum(x * x for x in se)) / n)
        se_b = float(onp.std(d, ddof=1) / math.sqrt(n)) if n > 1 else None
        out[name] = dict(
            D=D, se_within=se_w, ci95=[D - Z95 * se_w, D + Z95 * se_w],
            se_between_seeds=se_b, per_seed=dict(zip(map(str, seeds), d)),
            per_seed_se=dict(zip(map(str, seeds), se)),
            per_seed_sign=dict(zip(map(str, seeds), [sign(x) for x in d])),
            label=verdict(D, se_w, d))
    return out


def label(comp, a, b, metric):
    """The label of `a` against `b`, from whichever orientation was run."""
    c = comp.get(f"{a}_vs_{b}")
    if c is not None:
        return c[metric]["label"]
    c = comp.get(f"{b}_vs_{a}")
    return None if c is None else FLIP[c[metric]["label"]]


def recommend(comp, unavailable):
    """PRE-REGISTERED mapping to exactly one recommendation. No comparator is
    chosen from results: generalized prospectivity is compared with EACH
    applicable control individually. REDUNDANT_CONTROL_CONFIRMED is reserved
    for an exact algebraic identity, which the audit ruled out."""
    controls = [a for a in CONTROLS if a not in unavailable]
    known = [a for a in (OPERATOR, NAG_ARM, QHM_ARM) if a not in unavailable]
    why = dict(applicable_controls=controls,
               applicable_known_optimizers=known)
    per = {c: {m: label(comp, GEN, c, m)
               for m in ("revision", "retention", "recall",
                         "immediate_revised", "later_revised", "later")}
           for c in controls}
    why["generalized_vs_each_control"] = per
    if controls and all(
            per[c]["revision"] == "BETTER"
            and per[c]["retention"] not in ("WORSE",)
            and per[c]["recall"] not in ("WORSE",) for c in controls):
        return "GENERALIZED_GP_RETAINS_DISTINCT_ADVANTAGE", why
    if NAG_ARM not in unavailable:
        others = [a for a in (OPERATOR, TSS, QHM_ARM) if a not in unavailable]
        conds = dict(
            beats_native=label(comp, NAG_ARM, NATIVE, "revision") == "BETTER",
            not_worse_than_each_other_control=all(
                label(comp, NAG_ARM, o, "revision") not in (
                    "WORSE", "WORSE_BELOW_MARGIN") for o in others),
            generalized_not_better=label(comp, GEN, NAG_ARM, "revision")
            not in ("BETTER", "BETTER_BELOW_MARGIN"),
            retention_not_worse_than_native=label(
                comp, NAG_ARM, NATIVE, "retention") != "WORSE",
            recall_not_worse_than_native=label(
                comp, NAG_ARM, NATIVE, "recall") != "WORSE")
        why["literal_nesterov_conditions"] = conds
        if all(conds.values()):
            return "RUN_LITERAL_NESTEROV_AT_SCALE", why
    ok = ("EQUIVALENT_WITHIN_MARGIN", "WORSE", "WORSE_BELOW_MARGIN")
    matched = [k for k in known
               if per[k]["revision"] in ok and per[k]["immediate_revised"] in ok]
    why["known_optimizers_generalized_does_not_beat"] = matched
    if matched:
        return "GENERALIZED_GP_REDUCES_TO_KNOWN_OPTIMIZER", why
    return "NO_GO", why


def immediate_claim(comp, unavailable):
    """PRE-REGISTERED: the immediate-revision claim holds only if generalized
    prospectivity is BETTER than EACH applicable control on immediate revised
    accuracy (all three seeds positive, paired CI above zero, >= 1 pp)."""
    controls = [a for a in CONTROLS if a not in unavailable]
    labels = {c: label(comp, GEN, c, "immediate_revised") for c in controls}
    later = {c: dict(later_revised=label(comp, GEN, c, "later_revised"),
                     later=label(comp, GEN, c, "later")) for c in controls}
    return dict(claim_holds=bool(controls) and all(
        v == "BETTER" for v in labels.values()),
        immediate_revised=labels, later_versus_each_control=later,
        controls=controls)


def stable_subset(n, violations):
    """Whole 4-episode blocks in which NO episode has a write token outside
    literal Nesterov's frozen-token condition (for that seed's endpoint).
    `violations` is per-episode counts or None (Nesterov unavailable)."""
    if violations is None:
        return onp.ones(n, dtype=bool)
    blk = blocks_of(n)
    bad_blocks = set(blk[onp.asarray(violations) > 0].tolist())
    return ~onp.isin(blk, sorted(bad_blocks))


def exclusion_record(keep, violations, batch):
    n = int(keep.size)
    fam, cond = batch["family"], batch["condition"]
    by = {f"family{int(f)}/condition{int(c)}": dict(
        episodes=int(onp.sum((fam == f) & (cond == c))),
        excluded=int(onp.sum(~keep & (fam == f) & (cond == c))))
        for f in onp.unique(fam) for c in onp.unique(cond)}
    return dict(episodes=n, excluded_episodes=int(n - keep.sum()),
                excluded_fraction=float((n - keep.sum()) / n),
                episodes_with_a_violating_write=(
                    None if violations is None
                    else int(onp.sum(onp.asarray(violations) > 0))),
                excluded_blocks=int(len(set(blocks_of(n)[~keep].tolist()))),
                blocks=int(n // 4), by_family_and_condition=by,
                rule=("whole balanced 4-episode blocks containing an episode "
                      "with a write token outside " + NAG_CONDITION))


def _comparisons(grouped):
    comp = {}
    for a, b in COMPARISONS:
        seeds = [s for s in SOURCE_FINAL
                 if (a, s) in grouped and (b, s) in grouped]
        if len(seeds) != len(SOURCE_FINAL):
            continue
        per = {s: paired_seed(grouped[(a, s)][0], grouped[(a, s)][1],
                              grouped[(b, s)][0], grouped[(b, s)][1])
               for s in seeds}
        comp[f"{a}_vs_{b}"] = combine(per)
    return comp


def paired_analysis(arrays, held_np, unavailable):
    """Episode-paired comparisons on the ONE common held-out set.

    PRIMARY: the COMPLETE original held-out set, every episode, for every
    arm. Nesterov failures (non-finite logits) are counted as incorrect and
    retained in the denominator. The planned primary contrasts are
    generalized versus each of the five controls; the other ten pairs are
    descriptive. The grouped full-sample aggregate must reproduce the
    evaluation's.

    SECONDARY MECHANISM DIAGNOSTIC ONLY: the common Nesterov-stable subset
    (whole balanced blocks with no write token outside the frozen-token
    condition, from the realized held-out gates). Its excluded fraction is
    reported; it never enters the headline recommendation."""
    t0 = time.time()
    n = held_np["event"].shape[0]
    keep, excl = {}, {}
    for seed in SOURCE_FINAL:
        viol = None
        if NAG_ARM not in unavailable and (NAG_ARM, seed) in arrays:
            viol = episode_violations(arrays[(NAG_ARM, seed)][2],
                                      held_np["event"])[0]
        keep[seed] = stable_subset(n, viol)
        excl[str(seed)] = exclusion_record(keep[seed], viol, held_np)
    out = {}
    for tag in ("full", "stable_subset"):
        grouped, bad = {}, []
        for (arm, seed), (correct, ce, _g, m) in arrays.items():
            if arm not in LADDER:
                continue               # the frozen anchor is not compared
            k = keep[seed] if tag == "stable_subset" else None
            if k is not None and not k.any():
                bad.append(f"{arm}/{seed}: empty stable subset")
                continue
            full, loo = grouped_aggregates(correct, ce, held_np, k)
            if tag == "full":
                for name, f in METRICS.items():
                    if f(full) != f(m):
                        raise RuntimeError(f"grouped aggregate differs for "
                                           f"{arm}/{seed}/{name}")
            if not all(onp.isfinite(f(full)) for f in METRICS.values()):
                bad.append(f"{arm}/{seed}: non-finite aggregate on {tag}")
                continue
            grouped[(arm, seed)] = (full, loo)
        comp = _comparisons(grouped)
        for name, c in comp.items():
            c["role"] = ("planned_primary_contrast" if name in
                         PRIMARY_CONTRASTS else "descriptive")
        out[tag] = dict(comparisons=comp, not_computable=bad)
    out["full"].update(recommendation=None, immediate_claim=immediate_claim(
        out["full"]["comparisons"], unavailable))
    out["stable_subset"].update(
        role=("SECONDARY MECHANISM DIAGNOSTIC: excludes held-out episodes "
              "by realized Nesterov gates; never used for the headline "
              "recommendation"),
        diagnostic_recommendation=recommend(
            out["stable_subset"]["comparisons"], unavailable)[0],
        immediate_claim=immediate_claim(out["stable_subset"]["comparisons"],
                                        unavailable))
    rec, why = recommend(out["full"]["comparisons"], unavailable)
    out["full"]["recommendation"] = rec
    return dict(
        primary="full", full=out["full"],
        mechanism_diagnostic_stable_subset=out["stable_subset"],
        stable_subset_exclusions=excl,
        planned_primary_contrasts=list(PRIMARY_CONTRASTS),
        scientific_names={a: SCIENTIFIC_NAME[a] for a in DISPLAY_ORDER},
        display_order=list(DISPLAY_ORDER),
        primary_metrics=list(PRIMARY_METRICS),
        generalized_scope=GENERALIZED_SCOPE,
        recommendation=rec, recommendation_basis=why,
        immediate_claim=out["full"]["immediate_claim"],
        rule=("paired difference on the same held-out episodes; SE from a "
              "grouped jackknife over balanced 4-episode blocks, within each "
              "seed, combined across the three final seeds (SE_within); "
              "SE_between_seeds and every seed's sign reported; labels by "
              "`verdict`. PRIMARY: the complete original held-out set; "
              "planned primary contrasts are generalized versus each of the "
              "five controls; the other ten pairs are descriptive; the "
              "Nesterov-stable subset is a secondary diagnostic only"),
        margin=MARGIN, n_groups=N_GROUPS, wall_s=time.time() - t0)


def nesterov_applicability(status):
    """EVERY literal-Nesterov checkpoint that was evaluated, with its
    identity, and every one whose executed frozen-token table contains an
    unstable transition, with the violated condition. Instability is a
    result against Nesterov's applicability, not missing data."""
    rows = [e for e in status.get("checkpoint_log", [])
            if e.get("arm") == NAG_ARM]
    unstable = []
    for e in rows:
        t = e.get("frozen_token_table") or {}
        if (t.get("classification") or {}).get("unstable"):
            unstable.append(dict(
                stage=e["stage"], config=e["config"], lr=e["lr"],
                seed=e["seed"], update=e["update"],
                params_file=e.get("params_file"),
                classification=t["classification"],
                n_write_settings=t.get("n_write_settings"),
                min_jury_expression=t.get("min_jury_expression"),
                violated_condition=NAG_CONDITION,
                closed_form=t.get("closed_form_condition")))
    return dict(condition=NAG_CONDITION,
                checkpoints_evaluated=len(rows),
                checkpoints_unstable=len(unstable),
                unstable_fraction=(len(unstable) / len(rows) if rows
                                   else None),
                unstable=unstable,
                note=("an unstable checkpoint is ineligible for selection "
                      "(declared); counted here as evidence about Nesterov's "
                      "applicability at the trained gates"))


# -------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root",
                    default="/Users/durso/s5-runs/prospective-nesterov-ladder")
    ap.add_argument("--source_run", default=SOURCE_RUN)
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--budget_s", type=float, default=600.0)
    ap.add_argument("--reserve_s", type=float, default=30.0)
    args = ap.parse_args()
    t0 = time.time()
    deadline = args.deadline if args.deadline else t0 + args.budget_s
    backend = jax.default_backend()
    if backend != "gpu":
        raise SystemExit(f"REFUSING: backend is {backend!r}, not 'gpu'.")
    if jax.config.read("jax_enable_x64") or jnp.zeros(1).dtype != jnp.float32:
        raise SystemExit("REFUSING: x64 is enabled; production is float32.")
    if TR.stream_overlaps():
        raise SystemExit(f"REFUSING: stream ranges overlap: "
                         f"{TR.stream_overlaps()}")
    if os.path.realpath(args.out_root).startswith(
            os.path.realpath(args.source_run)):
        raise SystemExit("REFUSING: output inside the read-only source run")
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(args.out_root, run_id)
    os.makedirs(out, exist_ok=True)
    status = dict(
        run_id=run_id, out=out, backend=backend,
        study=("Six-arm Momentum DeltaNet comparison: Native Momentum "
               "DeltaNet; Zucchet prospective dynamics, matched time "
               "constants; Zucchet prospective dynamics, learned "
               "time-constant mismatch; Generalized prospective dynamics "
               "(M,γ,T); Literal Nesterov Momentum DeltaNet; QHM "
               "Momentum DeltaNet - on the completed temporal-response "
               "protocol"),
        audit="docs/MDN_NESTEROV_QHM_AUDIT.md",
        protocol_reused="docs/PROSPECTIVE_TEMPORAL_RESPONSE_PROTOCOL.md",
        completed_run_same_streams=COMPLETED_RUN,
        source_run=args.source_run,
        source_seeds=dict(development=SOURCE_DEV, final=list(SOURCE_FINAL)),
        ladder=list(LADDER), arm_law=dict(LAW),
        scientific_names={a: SCIENTIFIC_NAME[a] for a in
                          (*DISPLAY_ORDER, ANCHOR)},
        display_order=list(DISPLAY_ORDER),
        generalized_scope=GENERALIZED_SCOPE,
        realization_classification=REALIZATION_CLASSIFICATION,
        display=dict(DISPLAY),
        extra_parameters=dict(EXTRA_PARAMETERS),
        carry_executed=dict(CARRY_EXECUTED), extra_work=dict(EXTRA_WORK),
        updates=UPDATES, checkpoints_at=list(VAL_AT),
        learning_rates=dict(LRS), streams=STREAM,
        heldout_policy=("the completed study's held-out stream, generated "
                        "only after every selection and endpoint is frozen; "
                        "never used for selection"),
        declared_departures=[
            "arms: Literal Nesterov Momentum DeltaNet and QHM Momentum "
            "DeltaNet are added; "
            "Zucchet prospective dynamics with a learned mismatch "
            "is kept with its exact completed implementation and is NOT "
            "substituted by QHM Momentum DeltaNet (production gates are token "
            "dependent)",
            "the controlled mechanism diagnostic and the matched-operating-"
            "point analysis of the completed study are not repeated",
            "the completed study's float64/float32 law checks are not "
            "re-run inside the cap: those modules are byte-identical; this "
            "study's checks include an unchanged-arms test"],
        analysis=dict(margin=MARGIN, n_groups=N_GROUPS,
                      comparisons=[f"{a}_vs_{b}" for a, b in COMPARISONS],
                      metrics=list(METRICS),
                      recommendations=list(RECOMMENDATIONS)),
        production_dtype=dict(x64=False, float_dtype="float32"),
        source=dict(hashes_at_restore=None), checkpoint_log=[],
        development=[], final_trajectories=[], final=[], incomplete=[])
    status_path = os.path.join(out, "status.json")

    def persist(st):
        st["wall_s"] = time.time() - t0
        ST.write(status_path, st)

    def rehash():
        keys = (status.get("source") or {}).get("hashes_at_restore")
        if not keys:
            return {}
        d = RS.sources_dir(args.source_run)
        return {k: (RS.sha256(os.path.join(d, k))
                    if os.path.isfile(os.path.join(d, k)) else None)
                for k in keys}

    signal.signal(signal.SIGTERM, ST._on_sigterm)
    code, _ = ST.guarded(lambda: run_study(args, status, out, deadline,
                                           lambda: persist(status)),
                         status, rehash, persist)
    return code


def run_study(args, status, out, deadline, save):
    val_np = TT.generate_batch(STREAM["dev_validation"], VAL_PER_FAMILY)
    eval_val_np = TT.generate_batch(STREAM["eval_validation"], VAL_PER_FAMILY)
    status["task_checks"] = dict(
        dev_validation=TT.structure_check(val_np),
        eval_validation=TT.structure_check(eval_val_np),
        dev_validation_digest=TT.episode_digest(val_np),
        eval_validation_digest=TT.episode_digest(eval_val_np))
    save()
    try:
        sources = TC.load_sources(args.source_run, status)
    except RS.SourceRefusal as e:
        status["failed"] = f"source refusal: {e}"
        return 4, "FAILED"
    print(f"[source] reused {len(sources)} read-only Momentum checkpoints")
    save()

    # start points: Nesterov is the native tree; QHM at nu = 1 is the native
    # function; TSS/generalized as in the completed study
    starts, bad = {}, []
    for seed, pn in sources.items():
        nag_p, qhm_p = start_tree(NAG_ARM, pn), start_tree(QHM_ARM, pn)
        same_tree = all(onp.array_equal(onp.asarray(nag_p[k]),
                                        onp.asarray(pn[k])) for k in pn) \
            and set(nag_p) == set(pn)
        # recovery at the completed study's own tolerance (TRAJ32), on its
        # representative episodes: this file's shell executing the native
        # step, and QHM at nu = 1, both against the production native rollout
        rec_rows, rec_fail = [], []
        for i in TC.recovery_episodes(val_np):
            ep = {k: jnp.asarray(val_np[k][i]) for k in TT.MODEL_INPUTS}
            ref = PM.rollout("momentum_delta", pn, ep)["logits"]
            errs = dict(
                native_shell=TC._rel(rollout(NATIVE_SHELL, pn, ep)["logits"],
                                     ref),
                qhm_nu_one=TC._rel(rollout(QHM, qhm_p, ep)["logits"], ref))
            rec_rows.append(dict(episode=int(i), relative_errors=errs))
            rec_fail += [f"episode {i} {k}: {e}" for k, e in errs.items()
                         if not (onp.isfinite(e) and e <= TC.TRAJ32)]
        tss_p = start_tree(TSS, pn)
        _, sets = evaluate(TSS, tss_p, val_np)
        tss_fail = TC.validate(TSS, tss_p, sets)
        starts[str(seed)] = dict(
            nesterov_tree_is_native_tree=bool(same_tree),
            shell_and_qhm_recovery=rec_rows,
            recovery_failures=rec_fail, tolerance=TC.TRAJ32,
            nesterov_start_table=table_report(NAG_ARM, nag_p),
            tss_start_acceptance_failure=tss_fail)
        if not same_tree or rec_fail or tss_fail:
            bad.append(seed)
    status["start_points"] = starts
    save()
    if bad:
        status["failed"] = f"start-point checks failed for seeds {bad}"
        return 4, "FAILED"

    proj, retraced, failures = preflight(sources, val_np, out, status)
    save()
    d = ST.decide_after_preflight(proj, retraced, failures,
                                  deadline - time.time() - args.reserve_s)
    if d is not None:
        code, label, why = d
        (status.__setitem__("failed", why) if code == 4
         else status["incomplete"].append(why))
        print(f"[!] {why}")
        return code, label

    def tree_of(arm, seed):
        e = status["source"]["entries"][seed]
        p = (dict(sources[seed]) if arm == ANCHOR
             else start_tree(arm, sources[seed]))
        return p, f"{e['file']} sha256={e['sha256']}"

    dev_rows = []
    for arm in LADDER:
        for tag, lr in LRS:
            p, label = tree_of(arm, SOURCE_DEV)
            r = run_one(arm, tag, lr, SOURCE_DEV, p, val_np, UPDATES, out,
                        deadline, args.reserve_s, status, "dev", label, save)
            if r is None:
                return 3, "INCOMPLETE"
            dev_rows.append(r[0])
            status["development"] = dev_rows
            save()
            if r[0]["invalid"]:
                status["failed"] = f"dev {arm}/{tag}: {r[0]['invalid']}"
                print(f"[!] {status['failed']}")
                return 4, "FAILED"

    sel, err = select(dev_rows, status)
    if sel is None:
        status["failed"] = f"selection could not be formed: {err}"
        return 4, "FAILED"
    unavailable = status["selection"]["unavailable"]
    fplan = final_plan(sel)
    frozen = copy.deepcopy(dict(selection=sel, final_plan=fplan,
                                unavailable=unavailable))
    status["frozen_before_finals"] = copy.deepcopy(frozen)
    status["final_plan"] = fplan
    ST.write(os.path.join(out, "selection.json"), frozen)
    for arm in DISPLAY_ORDER:
        s = sel.get(arm)
        print(f"[selection] {SCIENTIFIC_NAME[arm]} " + (
            "UNAVAILABLE" if s is None else
            f"lr {s['lr']} update {s['update']:>3} revision "
            f"{s['primary']:.4f} immediate {s['immediate_revision']:.4f} "
            f"later {s['later']:.4f}"))
    save()

    final_rows, traj_rows, endpoints = [], [], {}
    for tr in fplan:
        for seed in SOURCE_FINAL:
            p, label = tree_of(tr["arm"], seed)
            r = run_one(tr["arm"], tr["config"], tr["lr"], seed, p,
                        eval_val_np, tr["updates"], out, deadline,
                        args.reserve_s, status, "final", label, save,
                        keep_at=tuple(tr["keep_at"]))
            if r is None:
                status["heldout_opened"] = False
                return 3, "INCOMPLETE"
            rec, _, kept = r
            traj_rows.append(rec)
            status["final_trajectories"] = traj_rows
            save()
            if rec["invalid"]:
                status["failed"] = (f"final {tr['arm']}/{tr['config']}/"
                                    f"{seed}: {rec['invalid']}")
                return 4, "FAILED"
            u = tr["updates"]
            v = {x["update"]: x for x in rec["validation"]}[u]
            final_rows.append(dict(
                rule=tr["arm"], law=LAW[tr["arm"]], seed=seed,
                config=tr["config"], lr=tr["lr"], update=u,
                endpoint_params_file=v["params_file"],
                final_validation_summary={
                    k: v[k] for k in ("primary", "retention", "recall",
                                      "immediate_revision", "later")},
                frozen_token_table=v.get("frozen_token_table"),
                source=label))
            endpoints[(tr["arm"], seed)] = kept[u]
    for seed in SOURCE_FINAL:
        p, label = tree_of(ANCHOR, seed)
        final_rows.append(dict(rule=ANCHOR, law="momentum_delta", seed=seed,
                               config="-", lr=0.0, update=0, source=label,
                               final_validation_summary=None))
        endpoints[(ANCHOR, seed)] = p
    status["final"] = final_rows
    if copy.deepcopy(dict(selection=sel, final_plan=fplan,
                          unavailable=unavailable)) != frozen:
        status["failed"] = "selection or final plan changed during finals"
        return 4, "FAILED"

    status["heldout_opened"] = True
    status["heldout_opened_at"] = time.time()
    save()
    held_np = TT.generate_batch(STREAM["heldout"], HELDOUT_PER_FAMILY)
    status["task_checks"]["heldout"] = TT.structure_check(held_np)
    status["task_checks"]["heldout_digest"] = TT.episode_digest(held_np)
    arrays = {}
    for row in final_rows:
        arm = row["rule"]
        m, sets, (correct, ce, gts) = evaluate(
            arm, endpoints[(arm, row["seed"])], held_np, keep_arrays=True)
        row["heldout"] = m
        if LAW[arm] == NESTEROV:
            # Nesterov failures are RESULTS: counted incorrect, retained in
            # the denominator (m["nesterov_failures"]); the run fails only if
            # an accuracy aggregate itself is not finite
            if not all(onp.isfinite(f(m)) for f in METRICS.values()):
                status["failed"] = (f"non-finite accuracy aggregate "
                                    f"{arm}/{row['seed']}")
                return 4, "FAILED"
        elif not TR.metrics_finite(m) or (
                LAW[arm] == FL.FILTERED and (
                    not m["processing_state"]["finite"]
                    or any(FL.filter_failure(FL.filter_report(ex))
                           for ex in sets))):
            status["failed"] = (f"non-finite or unaccepted held-out "
                                f"evaluation {arm}/{row['seed']}")
            return 4, "FAILED"
        arrays[(arm, row["seed"])] = (correct, ce, gts, m)
    status["heldout_evaluation_complete"] = True
    save()
    analysis = paired_analysis(arrays, held_np, unavailable)
    status["paired_analysis"] = analysis
    applicability = nesterov_applicability(status)
    applicability["heldout_endpoints"] = {
        str(r["seed"]): dict(
            endpoint_table=r.get("frozen_token_table"),
            episodes=r["heldout"].get("nesterov_episode_applicability"),
            failures_retained_in_denominator=r["heldout"].get(
                "nesterov_failures"))
        for r in final_rows if r["rule"] == NAG_ARM}
    applicability["eligibility_basis"] = (
        "parameter-domain only: the executed frozen-token table over every "
        "input at which a write can occur; never held-out gates")
    applicability["unavailable"] = NAG_ARM in unavailable
    status["nesterov_applicability"] = applicability
    status["work_completed"] = dict(
        development_runs=len(dev_rows), final_trajectories=len(traj_rows),
        checkpoints_validated_and_persisted=len(status["checkpoint_log"]),
        total_updates=sum(r["updates"] for r in dev_rows + traj_rows))
    results = dict(run_id=status["run_id"], ladder=list(LADDER),
                   display_order=list(DISPLAY_ORDER),
                   generalized_scope=GENERALIZED_SCOPE,
                   scientific_names={a: SCIENTIFIC_NAME[a] for a in
                                     (*DISPLAY_ORDER, ANCHOR)},
                   unavailable=unavailable,
                   selection={a: {k: s[k] for k in ("config", "lr", "update",
                                                    "primary",
                                                    "immediate_revision",
                                                    "later")}
                              for a, s in sel.items()},
                   heldout={f"{r['rule']}/{r['seed']}": {
                       name: float(f(r["heldout"]))
                       for name, f in METRICS.items()} for r in final_rows},
                   paired_analysis=analysis,
                   nesterov_applicability=applicability)
    ST.write(os.path.join(out, "results.json"), results)
    print(f"[nesterov] checkpoints evaluated "
          f"{applicability['checkpoints_evaluated']}, unstable "
          f"{applicability['checkpoints_unstable']}; held-out failures "
          + str({s: (e.get('failures_retained_in_denominator') or {}).get(
              'episodes_failed')
              for s, e in applicability['heldout_endpoints'].items()})
          + "; diagnostic stable-subset exclusions "
          + str({s: e['excluded_fraction'] for s, e in
                 analysis['stable_subset_exclusions'].items()}))
    print(f"[recommendation] {analysis['recommendation']} (primary: full "
          f"held-out set; diagnostic stable subset: "
          f"{analysis['mechanism_diagnostic_stable_subset']['diagnostic_recommendation']})")
    print(f"[immediate claim] holds="
          f"{analysis['immediate_claim']['claim_holds']} "
          f"{analysis['immediate_claim']['immediate_revised']}")
    print(f"[scope] {GENERALIZED_SCOPE}")
    for name in PRIMARY_CONTRASTS:
        c = analysis["full"]["comparisons"].get(name)
        if c is None:
            other = name.split("_vs_", 1)[1]
            print(f"[primary] {SCIENTIFIC_NAME[GEN]} vs "
                  f"{SCIENTIFIC_NAME[other]}: not computable")
            continue
        other = name.split("_vs_", 1)[1]
        for mt in PRIMARY_METRICS:
            x = c[mt]
            print(f"[primary] {SCIENTIFIC_NAME[GEN]} vs "
                  f"{SCIENTIFIC_NAME[other]} | {mt}: "
                  f"D {x['D']:+.4f} SE {x['se_within']:.4f} CI95 "
                  f"[{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] signs "
                  f"{''.join(x['per_seed_sign'].values())} -> {x['label']}")
    for name, c in analysis["full"]["comparisons"].items():
        first, second = name.split("_vs_", 1)
        comparison = f"{SCIENTIFIC_NAME[first]} vs {SCIENTIFIC_NAME[second]}"
        print(f"[paired:{c['role'][:7]}] {comparison} | "
              f"revision {c['revision']['D']:+.4f} "
              f"({c['revision']['label']}) immediate "
              f"{c['immediate_revised']['D']:+.4f} "
              f"{''.join(c['immediate_revised']['per_seed_sign'].values())} "
              f"({c['immediate_revised']['label']}) later_revised "
              f"{c['later_revised']['D']:+.4f} "
              f"{''.join(c['later_revised']['per_seed_sign'].values())} "
              f"({c['later_revised']['label']})")
    status["complete"] = True
    return 0, "PASS"


if __name__ == "__main__":
    sys.exit(main())
