"""Production-dtype probe for the stable generalized continuation. x64 OFF.

Runs in its own process: x64 is process-global. Covers what only float32 can
show - the declared numerical interior against rounding, and the baseline
identities at the study's OWN declared gate tolerances.

  1. Projection in float32 across nearly real modes, extreme z, simultaneous
     updates to poles/clock/T, and P != H; every projected point must pass the
     float64 evaluation of its EXECUTED float32 coefficients.
  2. Full-network B(q=0) vs A and C(r=0, q, t) vs B(q): logits, shared
     gradients within the mixed absolute/relative tolerance, and the
     copied-gradient optimizer-routing identity, at the study's declared gates.
  3. The T derivative is available away from rho = 1.

AMENDED BEFORE EXECUTION (review of 97cedfa):
  R1  the rounding witness T = 1, omega = 1, a = 3 * 2^-26 through the
      PRODUCTION float32 bound and exp: rho must be exactly 1 and stable;
  R2  an r-ONLY directional derivative at the actual start q = r = t = 0
      against central differences at BOTH declared float32 steps, a
      vanishing t derivative there, the mixed absolute/relative gradient
      identity on the PAIR-SPECIFIC shared leaves (q included for C vs B) and
      on INPUT gradients, and the optimizer-routing identity with copied
      gradients;
  R0  recurrent reference horizon 10, input reference 5.

Declared float32 derivative constants: steps (1e-2, 3e-3); relative
tolerance 2e-2 at EACH step; resolvability |jvp| >= 100 eps32 |f| / h_min;
t-derivative at the start <= 1e-3 max|dL/dr|.
"""

import math
import os
import sys

os.environ.setdefault("JAX_ENABLE_X64", "0")

import jax                                                        # noqa: E402
import jax.numpy as jnp                                           # noqa: E402
import numpy as onp                                               # noqa: E402

assert not jax.config.read("jax_enable_x64"), "x64 must be OFF in this probe"
jax.config.update("jax_default_matmul_precision", "highest")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flax.traverse_util import flatten_dict, unflatten_dict        # noqa: E402
from s5 import stable_gp as SG                                     # noqa: E402
from s5.rawat_model import RawatClassifier                         # noqa: E402
from s5.rawat_s5 import init_substrate_ssm                         # noqa: E402
from tests.response_reference import ssm_kwargs as _ssm_kwargs     # noqa: E402
from experiments.gp import stable_gp_study as ST                   # noqa: E402

assert not jax.config.read("jax_enable_x64"), \
    "x64 was switched back ON by an import; this probe would measure float64"
assert jnp.zeros(1).dtype == onp.float32

fails = []
print(f"  backend={jax.default_backend()} x64={jax.config.read('jax_enable_x64')}"
      f" eps_num={SG.eps_num(onp.float32):.3e}")


# ---------------------------------------------------------------- 1 ---------
def layer(rs, P, lam_im_scale):
    lam_re = -onp.exp(rs.uniform(-6, 2, P)).astype(onp.float32)
    # nearly real modes down to |Im lambda| = 1.2e-4 against |Re lambda| up to
    # 7.4, and Delta down to 9e-4: z reaches ~6e12 (rho_max ~ 6e12), far beyond
    # any HiPPO-initialized or trained mode, plus two EXACTLY real modes.
    # Stated so the covered range is explicit rather than implied.
    lam_im = (rs.choice([-1, 1], P) * onp.exp(rs.uniform(-9, 3, P))
              * lam_im_scale).astype(onp.float32)
    lam_im[:2] = 0.0                                  # exactly real modes
    log_step = onp.log(onp.exp(rs.uniform(-7, 0, P))).astype(onp.float32)[:, None]
    return {"Lambda_re": jnp.asarray(lam_re), "Lambda_im": jnp.asarray(lam_im),
            "log_step": jnp.asarray(log_step),
            SG.LEAF_T: jnp.asarray(rs.uniform(-4, 4, P).astype(onp.float32)),
            SG.LEAF_RHO: jnp.asarray(onp.zeros(P, onp.float32)),
            SG.LEAF_T_IN: jnp.asarray(onp.zeros(P, onp.float32))}


project = jax.jit(SG.project_stable_domain)
worst_rel_margin, n_checked = float("inf"), 0
for trial in range(40):
    rs = onp.random.RandomState(1000 + trial)
    P = 16
    tree = {"encoder": {f"layers_{i}": {"seq": layer(rs, P, 1.0)}
                        for i in range(2)}}
    # propose OUTWARD r everywhere, far and near
    for i in range(2):
        seq = tree["encoder"][f"layers_{i}"]["seq"]
        a, w = SG.modal_j(seq["Lambda_re"], seq["Lambda_im"], seq["log_step"])
        b = SG.log_rho_upper(a, w, SG.RECURRENT_T_REFERENCE
                             * jnp.exp(seq[SG.LEAF_T]), onp.float32)
        bump = rs.choice([1e-6, 1e-3, 1.0, 50.0], P).astype(onp.float32)
        seq[SG.LEAF_RHO] = jnp.where(jnp.isfinite(b), b + bump, 3.0)
    out, tel = project(tree)
    rep = SG.executed_domain_report(out)
    for lay in rep["layers"]:
        n_checked += lay["n_modes"]
        m = lay["min_relative_margin_complex"]
        if m is not None:
            worst_rel_margin = min(worst_rel_margin, m)
    if not rep["passed"]:
        fails.append(f"projection trial {trial}: executed float32 point left "
                     f"the domain: {rep}")
    # a simultaneous update of poles, clock and T changes the bound
    for i in range(2):
        seq = out["encoder"][f"layers_{i}"]["seq"]
        seq["Lambda_im"] = seq["Lambda_im"] * 1.5
        seq["log_step"] = seq["log_step"] + 0.3
        seq[SG.LEAF_T] = seq[SG.LEAF_T] - 0.7
    out2, _ = project(out)
    if not SG.executed_domain_report(out2)["passed"]:
        fails.append(f"trial {trial}: re-projection after a coupled update "
                     f"did not restore the domain")
    for leaf in (SG.LEAF_RHO, SG.LEAF_T, "Lambda_re"):
        for i in range(2):
            if out2["encoder"][f"layers_{i}"]["seq"][leaf].dtype != onp.float32:
                fails.append(f"{leaf} dtype drifted")
print(f"  [1] projection: {n_checked} executed float32 modes checked in "
      f"float64, worst relative margin to rho_max {worst_rel_margin:.3e}")


# ---------------------------------------------------------------- 2 ---------
def production_kwargs(P, H, **overrides):
    """Initializer arrays EXPLICITLY in production dtypes (dispatch-1 fix).

    `tests/response_reference.ssm_kwargs` supplies NumPy float64 pole arrays
    and complex128 V/Vinv. S5SSM stores the pole initializers unchanged as
    parameters, so the first dispatch's probe ran with float64 Lambda leaves
    even with x64 off. The shared neutral helper is left untouched; the probe
    casts here: poles float32, eigenvector matrices complex64.
    """
    kw = dict(_ssm_kwargs(P, H), **overrides)
    kw["Lambda_re_init"] = onp.asarray(kw["Lambda_re_init"], onp.float32)
    kw["Lambda_im_init"] = onp.asarray(kw["Lambda_im_init"], onp.float32)
    kw["V"] = onp.asarray(kw["V"], onp.complex64)
    kw["Vinv"] = onp.asarray(kw["Vinv"], onp.complex64)
    return kw


def assert_float32_leaves(tag, variables):
    """Dtype assertion for EVERY arm's params and batch statistics."""
    for coll in ("params", "batch_stats"):
        for k, v in flatten_dict(variables.get(coll, {})).items():
            if onp.asarray(v).dtype != onp.float32:
                fails.append(f"{tag} {coll} leaf {'/'.join(k)} is "
                             f"{onp.asarray(v).dtype}, not float32")


def net(arm, P=4, H=6, L=24):
    m = RawatClassifier(ssm=init_substrate_ssm(arm, **production_kwargs(P, H)),
                        d_model=H, n_layers=2, d_output=3, readout_width=5,
                        mlp_hidden=7, training=False)
    x = jnp.asarray(onp.random.RandomState(1).randn(L, 2).astype(onp.float32))
    v = m.init(jax.random.PRNGKey(0), x, jnp.ones(L))
    return m, v, x


def setleaf(params, name, vals):
    flat = flatten_dict(params)
    for k in flat:
        if k[-1] == name:
            flat[k] = jnp.asarray(vals, dtype=flat[k].dtype)
    return unflatten_dict(flat)


def rel(a, b):
    a = onp.asarray(a, onp.float64); b = onp.asarray(b, onp.float64)
    n = float(onp.sqrt(onp.sum(b ** 2)))
    return float(onp.sqrt(onp.sum((a - b) ** 2))) / (n if n > 0 else 1.0)


rs = onp.random.RandomState(7)
ma, va, x = net("alpha_p_s5")
mb, vb, _ = net("rawat_learned_input")
mc, vc, _ = net("sgp_learned_input")
for tag, vv in (("A", va), ("B", vb), ("C", vc)):
    assert_float32_leaves(tag, vv)
w = jnp.asarray(rs.randn(3).astype(onp.float32))
q = rs.uniform(-1, 1, 4).astype(onp.float32)
t = rs.uniform(-1, 1, 4).astype(onp.float32)
pb_q = setleaf(vb["params"], SG.LEAF_T_IN, q)
pc_q = setleaf(setleaf(vc["params"], SG.LEAF_T_IN, q), SG.LEAF_T, t)
tx, _ = ST.make_optimizer(10, 10)


def loss(m, bs):
    return lambda p, xx: jnp.sum(w * m.apply({"params": p, "batch_stats": bs},
                                             xx, jnp.ones(xx.shape[0])))


for name, mx, px, vx, mr, pr, vr in (
        ("B(q=0) vs A", mb, vb["params"], vb, ma, va["params"], va),
        ("C(r=0,q,t) vs B(q)", mc, pc_q, vc, mb, pb_q, vb)):
    yx = mx.apply({"params": px, "batch_stats": vx["batch_stats"]}, x,
                  jnp.ones(x.shape[0]))
    yr = mr.apply({"params": pr, "batch_stats": vr["batch_stats"]}, x,
                  jnp.ones(x.shape[0]))
    lr_ = rel(yx, yr)
    gx, gxx = jax.grad(loss(mx, vx["batch_stats"]), argnums=(0, 1))(px, x)
    gr, grx = jax.grad(loss(mr, vr["batch_stats"]), argnums=(0, 1))(pr, x)
    gp = ST.compare_shared(gx, gr, onp.float32)
    gi = ST.compare_shared(gxx, grx, onp.float32)
    route = ST.routing_identity(tx, px, pr, gr)
    shared_names = sorted(gp["leaves"])
    q_in = any(n.endswith(SG.LEAF_T_IN) for n in shared_names)
    print(f"  [2] {name}: logits {lr_:.2e} (tol {ST.LOGIT_REL_TOL:.0e})  "
          f"param grads {'PASS' if gp['passed'] else 'FAIL'} over "
          f"{gp['n_leaves']} shared leaves (q included: {q_in}; worst "
          f"{gp['worst_leaf']})  input grads "
          f"{'PASS' if gi['passed'] else 'FAIL'}  routing "
          f"{route['worst']:.2e} (tol {ST.UPDATE_MAX_TOL:.0e})")
    if name.startswith("C") and not q_in:
        fails.append("C vs B comparison omitted the shared q leaves")
    if not (lr_ <= ST.LOGIT_REL_TOL and gp["passed"] and gi["passed"]
            and route["passed"]):
        fails.append(f"{name}: production identity outside declared gate")


# ---------------------------------------------------------------- 3 ---------
pc_r = setleaf(pc_q, SG.LEAF_RHO, rs.uniform(-0.6, 0.2, 4).astype(onp.float32))
pc_r = SG.project_stable_domain(pc_r)[0]
g = flatten_dict(jax.grad(loss(mc, vc["batch_stats"]))(pc_r, x))
tg = max(float(onp.max(onp.abs(onp.asarray(v))))
         for k, v in g.items() if k[-1] == SG.LEAF_T)
print(f"  [3] |dL/dt| away from rho = 1: {tg:.3e}")
if not (onp.isfinite(tg) and tg > 0.0):
    fails.append("T derivative unavailable away from rho = 1")


# ---------------------------------------------------------------- R1 --------
a_w = jnp.asarray(3 * 2.0 ** -26, jnp.float32)
one = jnp.asarray(1.0, jnp.float32)
bound_w = SG.log_rho_upper(a_w[None], one[None], one[None], onp.float32)
rho_w = float(jnp.exp(bound_w)[0])
rmax_w = float(SG.rho_max_float64(onp.float32(3 * 2.0 ** -26), 1.0, 1.0))
print(f"  [R1] witness: bound {float(bound_w[0]):.3e}  executed rho "
      f"{rho_w!r}  rho_max {rmax_w!r}")
if not (rho_w == 1.0 and rho_w < rmax_w
        and SG.stability_S(3 * 2.0 ** -26, 1.0, 1.0, rho_w) > 0):
    fails.append("R1 witness: production bound did not return the rho = 1 "
                 "fallback inside the domain")

# ---------------------------------------------------------------- R2 --------
FD32_STEPS, FD32_REL = (1e-2, 3e-3), 2e-2
kw = production_kwargs(4, 6, dt_min=0.1, dt_max=1.0)
m0 = RawatClassifier(ssm=init_substrate_ssm("sgp_learned_input", **kw),
                     d_model=6, n_layers=2, d_output=3, readout_width=5,
                     mlp_hidden=7, training=False)
x0 = jnp.asarray(onp.random.RandomState(8).randn(24, 2).astype(onp.float32))
v0 = m0.init(jax.random.PRNGKey(3), x0, jnp.ones(24))
assert_float32_leaves("R2 r-only fixture (C)", v0)
rs0 = onp.random.RandomState(27)
w0 = jnp.asarray(rs0.randn(3).astype(onp.float32))
f0 = lambda p: jnp.sum(w0 * m0.apply(                              # noqa: E731
    {"params": p, "batch_stats": v0["batch_stats"]}, x0, jnp.ones(24)))
p0 = v0["params"]
flat0 = flatten_dict(p0)
d0 = unflatten_dict({k: (jnp.asarray(rs0.randn(*onp.shape(val))
                                     .astype(onp.float32))
                         if k[-1] == SG.LEAF_RHO else jnp.zeros_like(val))
                     for k, val in flat0.items()})
jvp0 = float(jax.jvp(f0, (p0,), (d0,))[1])
fval = float(f0(p0))
eps32 = float(onp.finfo(onp.float32).eps)
resolvable = abs(jvp0) >= 100 * eps32 * max(abs(fval), 1.0) / min(FD32_STEPS)
print(f"  [R2] r-only jvp at start {jvp0:.5e}  f {fval:.4e}  resolvable "
      f"{resolvable}")
if not resolvable:
    fails.append("R2 r-only fixture is not resolvable in float32")
for h in FD32_STEPS:
    hp = jax.tree_util.tree_map(lambda a_, b_: a_ + onp.float32(h) * b_, p0, d0)
    hm = jax.tree_util.tree_map(lambda a_, b_: a_ - onp.float32(h) * b_, p0, d0)
    fd = (float(f0(hp)) - float(f0(hm))) / (2 * h)
    rel_e = abs(jvp0 - fd) / max(abs(fd), 1e-30)
    print(f"  [R2] h={h:g}  fd {fd:.5e}  rel {rel_e:.2e} (tol {FD32_REL:.0e})")
    if not rel_e < FD32_REL:
        fails.append(f"R2 r-only derivative at h={h}: rel {rel_e:.2e}")
g0 = flatten_dict(jax.grad(f0)(p0))
gr0 = max(float(onp.max(onp.abs(onp.asarray(v)))) for k, v in g0.items()
          if k[-1] == SG.LEAF_RHO)
gt0 = max(float(onp.max(onp.abs(onp.asarray(v)))) for k, v in g0.items()
          if k[-1] == SG.LEAF_T)
print(f"  [R2] start: max|dL/dr| {gr0:.3e}  max|dL/dt| {gt0:.3e}")
if not (gr0 > 0 and gt0 <= 1e-3 * gr0):
    fails.append("R2 start: r derivative absent or t derivative not ~0")

if fails:
    print("FAILURES:")
    for f in fails:
        print("  -", f)
    sys.exit(1)
print("  float32 probe: all checks passed")
