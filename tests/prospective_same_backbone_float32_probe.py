"""Production float32 probe for the same-backbone study. x64 OFF.

Review R2: the focused-check module runs in float64, while the study refuses
to start unless x64 is disabled. This probe exercises the NEW paths in the
production precision: the ordinary operator's rollout and streaming cache, and
the coefficient-only optimizer path.

Declared constants, unchanged from the completed studies:
  TRAJ32 = 2e-5  relative: logits, carries and streaming;
  FD32 = (1e-2, 3e-3), REL32 = 2e-2, resolvability
                 |jvp| >= 100 eps32 max(|f|, 1) / h_min, with the
                 finite-before-resolution ordering; a FINITE unresolvable
                 derivative is a reported limitation, never a fabricated dead
                 parameter;
  REF32 = REL32  relative: the float32 kappa JVP against an INDEPENDENT
                 float64 sequential sensitivity of the same rounded inputs
                 (a cross-precision comparison, so the float32 smoke
                 tolerance applies, not a float64 identity tolerance).

Requires PM_SOURCE_RUN (the read-only replication run).
"""

import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "0")
import jax                                                        # noqa: E402
import jax.numpy as jnp                                           # noqa: E402
import numpy as onp                                               # noqa: E402

assert not jax.config.read("jax_enable_x64")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json                                                        # noqa: E402
import optax                                                       # noqa: E402
from experiments.nested_memory import dynamics as NMD              # noqa: E402
from experiments.nested_memory import model as NM                  # noqa: E402
from experiments.nested_memory import task as TK                   # noqa: E402
from experiments.prospective_momentum import model as PM           # noqa: E402
from experiments.prospective_momentum import ordinary as OD        # noqa: E402
from experiments.prospective_momentum import replication_sources as RS  # noqa
from experiments.prospective_momentum import same_backbone as SB   # noqa: E402
from experiments.prospective_momentum import study as ST           # noqa: E402

assert not jax.config.read("jax_enable_x64"), "an import enabled x64"
fails = []
eps32 = float(onp.finfo(onp.float32).eps)
F32 = onp.float32
TRAJ32, FD32, REL32 = 2e-5, (1e-2, 3e-3), 2e-2
REF32 = REL32
KAPPA_TEST = 0.8


def all_finite(*arrs):
    return all(bool(onp.all(onp.isfinite(onp.asarray(a, onp.float64))))
               for a in arrs)


def rel(a, b):
    a, b = onp.asarray(a, onp.float64), onp.asarray(b, onp.float64)
    if not all_finite(a, b):
        return float("inf")
    n = float(onp.linalg.norm(b))
    return float(onp.linalg.norm(a - b)) / (n if n > 0 else 1.0)


def check_dtype(label, *arrs):
    for a in arrs:
        if onp.asarray(a).dtype != F32:
            fails.append(f"dtype {label}: {onp.asarray(a).dtype}")


def tangent_decision(label, jvp, fval, perturbed):
    """Finite-before-resolution, then the declared relative tolerance."""
    fds = {h: (fp - fm) / (2 * h) for h, (fp, fm) in perturbed.items()}
    errs = {h: abs(jvp - fd) / max(abs(fd), 1e-30) for h, fd in fds.items()}
    if not all_finite(jvp, fval, [v for pair in perturbed.values()
                                  for v in pair], list(fds.values()),
                      list(errs.values())):
        return [f"{label}: non-finite loss, JVP, perturbed loss, FD or "
                f"error (jvp={jvp}, f={fval})"], False
    thr = 100 * eps32 * max(abs(fval), 1.0) / min(perturbed)
    for h, e in errs.items():
        print(f"  {label} jvp {jvp:.6e} fd(h={h}) {fds[h]:.6e} rel {e:.2e} "
              f"(threshold {thr:.2e})")
    if abs(jvp) < thr:
        return [], True
    return [f"{label} h={h}: rel {e:.2e}" for h, e in errs.items()
            if not e < REL32], False


# ---- the read-only source, in production precision --------------------------
run = os.environ.get("PM_SOURCE_RUN", SB.SOURCE_RUN)
if not os.path.isdir(RS.sources_dir(run)):
    print(f"FAIL: read-only source run {run} is required")
    sys.exit(1)
with open(os.path.join(RS.sources_dir(run), "manifest.json")) as fh:
    manifest = json.load(fh)
pn = RS.restore_source(run, RS.find_entry(manifest, SB.SOURCE_DEV,
                                          SB.SOURCE_FAMILY))
check_dtype("restored source leaves", *pn.values())
p_op = PM.add_extension(pn, OD.ORDINARY)
check_dtype("operator leaves", *p_op.values())

b = TK.generate_batch(9900, 1)
ep = {k: jnp.asarray(b[k][0]) for k in ("key_id", "val_id", "event", "label")}
q = ep["event"] == TK.QUERY


def loss_fn(rule, ep_):
    qq = ep_["event"] == TK.QUERY

    def f(p, c0=None):
        out = PM.rollout(rule, p, ep_, carry0=c0)
        ce = optax.softmax_cross_entropy(
            out["logits"], jax.nn.one_hot(jnp.maximum(ep_["label"], 0),
                                          TK.N_VALUES)) * qq
        return jnp.sum(ce) / jnp.sum(qq)
    return f


# ---- 1. native nesting at kappa = 0, in float32 -----------------------------
on = PM.rollout("momentum_delta", pn, ep)
oo = PM.rollout(OD.ORDINARY, p_op, ep)
check_dtype("operator outputs", oo["logits"], *oo["final_carry"], *oo["gates"])
worst = rel(oo["logits"], on["logits"])
for x, y in zip(oo["final_carry"][:2], on["final_carry"]):
    worst = max(worst, rel(x, y))
print(f"  kappa=0 native nesting (float32): worst relative {worst:.2e} "
      f"(TRAJ32 {TRAJ32})")
if not worst <= TRAJ32:
    fails.append(f"float32 native nesting {worst:.2e}")
if len(oo["final_carry"]) != 3:
    fails.append("the operator must carry three matrices")

# ---- 2. nonzero kappa: streaming across a chunk boundary --------------------
pk = dict(p_op, kappa=jnp.asarray([KAPPA_TEST], jnp.float32))
full = PM.rollout(OD.ORDINARY, pk, ep)
a1 = PM.rollout(OD.ORDINARY, pk, {k: v[:29] for k, v in ep.items()})
a2 = PM.rollout(OD.ORDINARY, pk, {k: v[29:] for k, v in ep.items()},
                carry0=a1["final_carry"])
stream = rel(jnp.concatenate([a1["logits"], a2["logits"]]), full["logits"])
for x, y in zip(a2["final_carry"], full["final_carry"]):
    stream = max(stream, rel(x, y))
print(f"  kappa={KAPPA_TEST} streaming across a boundary: worst relative "
      f"{stream:.2e} (TRAJ32 {TRAJ32})")
if not stream <= TRAJ32:
    fails.append(f"float32 streaming {stream:.2e}")
check_dtype("streamed carries", *a2["final_carry"])


# ---- 3. the kappa derivative: float32 FD and an INDEPENDENT float64 reference
def sensitivity_reference(p, kappa, ep_):
    """Sequential NumPy float64 primal + d/dkappa recursion for the OPERATOR.
    No JAX differentiation, no production update, no finite differences. Only
    kappa-independent inputs (preprocessing, gates, readout) are shared."""
    from experiments.adaptive_memory import model as AM
    e = AM.sanitize_episode(ep_)
    key_id, val_id, event = e["key_id"], e["val_id"], e["event"]
    k_all, k_valid = NMD.safe_normalize(p["key_raw"])
    keys = onp.asarray(k_all[key_id], onp.float64)
    has_v = (val_id >= 0).astype(jnp.float32)
    vals = onp.asarray(p["value_table"][jnp.maximum(val_id, 0)]
                       * has_v[:, None], onp.float64)
    mask = onp.asarray((event == TK.WRITE).astype(jnp.float32)
                       * k_valid[key_id], onp.float64)
    gx = NM.gate_features(key_id, val_id, event).astype(jnp.float32)
    al, be, mu, eta = (onp.asarray(x, onp.float64)
                       for x in NM._momentum_gates(p, gx))
    H = onp.asarray(p["readout_W"], onp.float64)
    bias = onp.asarray(p["readout_b"], onp.float64)
    lab = onp.maximum(onp.asarray(e["label"]), 0)
    qq = (onp.asarray(e["event"]) == TK.QUERY).astype(onp.float64)
    d_v, d_k = vals.shape[1], keys.shape[1]
    W = onp.zeros((d_v, d_k)); U = onp.zeros((d_v, d_k))
    Rp = onp.zeros((d_v, d_k))
    dW = onp.zeros((d_v, d_k)); dU = onp.zeros((d_v, d_k))
    dRp = onp.zeros((d_v, d_k))
    L = keys.shape[0]
    logits = onp.zeros((L, H.shape[0])); dlogits = onp.zeros_like(logits)
    for t in range(L):
        k, v, m = keys[t], vals[t], mask[t]
        Wb, dWb = al[t] * W, al[t] * dW
        R = m * onp.outer(Wb @ k - v, k)
        dR = m * onp.outer(dWb @ k, k)
        Rpros = (1.0 + kappa) * R - kappa * Rp
        dRpros = (R - Rp) + (1.0 + kappa) * dR - kappa * dRp
        U_new = mu[t] * U + eta[t] * Rpros
        dU_new = mu[t] * dU + eta[t] * dRpros
        W_new = Wb - be[t] * U_new
        dW_new = dWb - be[t] * dU_new
        W, U, Rp, dW, dU, dRp = W_new, U_new, R, dW_new, dU_new, dR
        logits[t] = H @ (W @ k) + bias
        dlogits[t] = H @ (dW @ k)
    z = logits - logits.max(axis=1, keepdims=True)
    logp = z - onp.log(onp.exp(z).sum(axis=1))[:, None]
    sm = onp.exp(logp)
    y = onp.eye(logits.shape[1])[lab]
    nq = max(qq.sum(), 1.0)
    return dict(logits=logits, W=W, U=U,
                loss=float(-(qq * (y * logp).sum(axis=1)).sum() / nq),
                dL_dkappa=float((qq * ((sm - y) * dlogits).sum(axis=1)).sum()
                                / nq))


f_op = loss_fn(OD.ORDINARY, ep)
fk = lambda kk: f_op(dict(p_op, kappa=jnp.asarray([kk], jnp.float32)))  # noqa
jvp = float(jax.jvp(fk, (jnp.float32(KAPPA_TEST),), (jnp.float32(1.0),))[1])
fval = float(fk(jnp.float32(KAPPA_TEST)))
ref = sensitivity_reference(p_op, KAPPA_TEST, ep)
if not all_finite(jvp, fval, ref["loss"], ref["dL_dkappa"], ref["logits"],
                  ref["W"], ref["U"], full["logits"], *full["final_carry"]):
    fails.append("non-finite loss, JVP, reference or production state")
else:
    e_log = rel(ref["logits"], full["logits"])
    print(f"  reference vs production (float64 vs float32): logits {e_log:.2e}"
          f" loss {ref['loss']!r} vs {fval!r}")
    if not e_log <= TRAJ32:
        fails.append(f"reference logits disagree: {e_log:.2e}")
    err = abs(jvp - ref["dL_dkappa"]) / max(abs(ref["dL_dkappa"]), 1e-30)
    print(f"  kappa={KAPPA_TEST} jvp {jvp:.6e} independent float64 reference "
          f"{ref['dL_dkappa']:.6e} rel {err:.2e} (REF32 {REF32})")
    if not err <= REF32:
        fails.append(f"kappa derivative vs float64 reference: {err:.2e}")
    pert = {h: (float(fk(jnp.float32(KAPPA_TEST + h))),
                float(fk(jnp.float32(KAPPA_TEST - h)))) for h in FD32}
    f_, lim_ = tangent_decision("OPERATOR kappa", jvp, fval, pert)
    fails.extend(f_)
    if lim_:
        print("  LIMITATION: the float32 kappa derivative is FINITE and below "
              "float32 finite-difference resolvability; the independent "
              "float64 reference above is the evidence. Not a dead parameter.")

# ---- 4. the coefficient-only optimizer path, in float32 ---------------------
bt = {k: jnp.asarray(v) for k, v in TK.generate_batch(9901, 2).items()
      if k in ("key_id", "val_id", "event", "label")}
lr = jnp.asarray(0.01, jnp.float32)
for law in ("prospective_momentum", OD.ORDINARY):
    p = dict(PM.add_extension(pn, law), kappa=jnp.asarray([0.3], jnp.float32))
    opt = ST.TX.init(p)
    full_step = ST.train_step(law, p, opt, bt, lr)
    froz = SB.train_step_coefficient_only(law, p, opt, bt, lr)
    check_dtype(f"{law} frozen-step leaves", *froz[0].values())
    changed = SB.frozen_leaf_differences(froz[0], p)
    same_loss = float(full_step[2]) == float(froz[2])
    same_grad = float(full_step[9]) == float(froz[9])
    moved = float(froz[0]["kappa"][0]) != float(p["kappa"][0])
    print(f"  {law}: coefficient-only loss {float(froz[2]):.6f} kappa grad "
          f"{float(froz[9]):.3e} frozen leaves changed {len(changed)} kappa "
          f"moved {moved}")
    if changed:
        fails.append(f"{law}: frozen leaves changed {changed}")
    if not (same_loss and same_grad):
        fails.append(f"{law}: freezing changed the loss or the kappa gradient")
    if not (moved and all_finite(float(froz[9])) and float(froz[9]) != 0.0):
        fails.append(f"{law}: kappa did not move or its gradient vanished")
    if not SB.frozen_leaf_differences(full_step[0], p):
        fails.append(f"{law}: the full regime did not move the backbone")

print("FLOAT32 PROBE:", "PASS" if not fails else f"FAIL {fails}")
sys.exit(0 if not fails else 1)
