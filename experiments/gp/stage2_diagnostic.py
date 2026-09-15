"""Stage 2 diagnostic: explain the failed fixed-GP screen from SAVED models.

READ-ONLY on Stage 2 artifacts. No training, no optimizer update, no resume,
no test scoring, no coefficient or architecture change. Derivative evaluation
without a parameter update is authorized and used.

Protocol: docs/GP_STAGE2_DIAGNOSTIC_PROTOCOL.md (frozen before execution).

The Stage 2 verdict stands. Nothing here revises it.
"""

import argparse
import hashlib
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

import dataloaders.speech_commands10 as SC                        # noqa: E402
from s5 import substrate_diagnostics as SD                        # noqa: E402
from s5.checkpointing import provenance                           # noqa: E402
from s5.physical_coefficients import SYMMETRIC_REFERENCE          # noqa: E402
from s5.rawat_model import RawatClassifier                        # noqa: E402
from s5.rawat_s5 import init_substrate_ssm                        # noqa: E402

from experiments.gp import rawat_benchmark as RB                  # noqa: E402

BANDS = ((0, 0), (1, 4), (5, 16), (17, 64), (65, 160), (161, 511))
N_LAGS = 512
N_FREQ = 129
SUBSET_SEED = 20260915
DIRECTION_SEED = 20260916
ANCHORS = (80, 160)
N_DIRECTIONS = 4
N_SUBSET = 32
CE_TOL = 1e-5
FD_REL_TOL = 1e-3
FD_STEPS = (1e-3, 1e-2)
FD_GATE_STEP = 1e-2
ARMS_1E3 = ("native_s5", "alpha_p_s5", "gain_clip_s5", "gp_fixed_m0",
            "gp_fixed_mass")

#: PASS -> 0, INCOMPLETE -> 3, FAILED -> 4. A failed required check or a
#: budget-incomplete run must NOT exit 0 (R3).
EXIT_CODES = {"PASS": 0, "INCOMPLETE": 3, "FAILED": 4}


# --------------------------------------------------------------- utilities
class Status:
    """Check registry that GATES analysis and determines the exit code.

    R3: previously every check appended a `passed` boolean that nothing read,
    so a failed restoration, a failed adapter comparison, a failed derivative
    check, nonzero future sensitivity or a CHANGED SOURCE FILE could still end
    with DIAGNOSTIC_EXIT=0. Now a required failure blocks dependent analysis
    for that arm and forces a nonzero exit, and an unavailable optional phase
    is distinguished from a failed required check.
    """

    def __init__(self, out):
        self.out = out
        self.checks = []
        self.failed_required = []
        self.unexecuted = []
        self.blocked_arms = set()

    def record(self, name, passed, required=True, arm=None, detail=None):
        rec = dict(check=name, arm=arm, passed=bool(passed),
                   required=bool(required), detail=detail,
                   time=time.time())
        self.checks.append(rec)
        if not passed and required:
            self.failed_required.append(rec)
            if arm:
                self.blocked_arms.add(arm)
            print(f"[FAIL] {name}" + (f" [{arm}]" if arm else ""))
        # persist immediately: a crash must not lose the record
        write_json(os.path.join(self.out, "checks.json"), self.checks)
        return bool(passed)

    def skip(self, name, reason, arm=None, required=False):
        rec = dict(check=name, arm=arm, status="not_executed", reason=reason,
                   required=bool(required))
        self.unexecuted.append(rec)
        self.checks.append(rec)
        write_json(os.path.join(self.out, "checks.json"), self.checks)
        print(f"[skip] {name}" + (f" [{arm}]" if arm else "") + f": {reason}")

    def blocked(self, arm):
        return arm in self.blocked_arms

    def final(self, hashes_ok, budget_incomplete):
        if self.failed_required or hashes_ok is False:
            return "FAILED"
        if budget_incomplete or hashes_ok is None or self.unexecuted:
            return "INCOMPLETE"
        return "PASS"


class Budget:
    """Wall-clock deadline with phase accounting. Never auto-extends.

    R4: the deadline is set by the LAUNCHER and passed in as an absolute epoch
    time, so backend verification and the focused tests run inside the same
    20-minute budget rather than before it starts.
    """

    def __init__(self, seconds, deadline=None):
        self.t0 = time.time()
        if deadline is not None:
            self.limit = max(0.0, float(deadline) - self.t0)
            self.absolute_deadline = float(deadline)
        else:
            self.limit = seconds
            self.absolute_deadline = self.t0 + seconds
        self.incomplete = False
        self.phases = []

    def left(self):
        return self.limit - (time.time() - self.t0)

    def ok(self, need=0.0):
        return self.left() > need

    def mark(self, name, status="complete", note=None):
        if status.startswith("skipped"):
            self.incomplete = True
        self.phases.append(dict(phase=name, status=status,
                                elapsed_s=time.time() - self.t0, note=note))
        print(f"[budget] {name}: {status}  elapsed {time.time()-self.t0:.1f}s  "
              f"left {self.left():.1f}s")


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(chunk), b""):
            h.update(blk)
    return h.hexdigest()


def hash_tree(root, names=("metrics.jsonl", "config.json", "summary.json",
                           "best.msgpack", "best.meta.json", "last.msgpack",
                           "last.meta.json")):
    out = {}
    for d in sorted(os.listdir(root)):
        rd = os.path.join(root, d)
        if not os.path.isdir(rd):
            if os.path.isfile(rd):
                out[d] = sha256_file(rd)
            continue
        for n in names:
            p = os.path.join(rd, n)
            if os.path.exists(p):
                out[f"{d}/{n}"] = sha256_file(p)
    return out


def jsonable(o):
    if isinstance(o, (onp.integer,)):
        return int(o)
    if isinstance(o, (onp.floating,)):
        return float(o)
    if isinstance(o, onp.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (bool, int, float, str)) or o is None:
        return o
    return str(o)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(jsonable(obj), fh, indent=2)
    return path


# ------------------------------------------------------- A: inventory/load
def read_run(run_dir):
    cfg_p = os.path.join(run_dir, "config.json")
    met_p = os.path.join(run_dir, "metrics.jsonl")
    out = dict(run_dir=run_dir, name=os.path.basename(run_dir))
    if os.path.exists(cfg_p):
        with open(cfg_p) as fh:
            out["config_json"] = json.load(fh)
    if os.path.exists(met_p):
        with open(met_p) as fh:
            out["metrics"] = [json.loads(l) for l in fh if l.strip()]
    for n in ("best", "last"):
        mp = os.path.join(run_dir, f"{n}.meta.json")
        if os.path.exists(mp):
            with open(mp) as fh:
                out[f"{n}_meta"] = json.load(fh)
    sp = os.path.join(run_dir, "summary.json")
    if os.path.exists(sp):
        with open(sp) as fh:
            out["summary"] = json.load(fh)
    return out


def build_template(cfg, training):
    """A structurally correct state for restoring, built from SAVED config."""
    class A:
        pass
    a = A()
    for k, v in cfg.items():
        setattr(a, k, v)
    model = RB.build_model(a.arm, a.d_model, a.ssm_size, a.n_layers,
                           training=training)
    return a, model


def restore_readonly(run_dir, cfg, name="best"):
    """Restore params AND batch statistics. Never applies an update."""
    from s5.checkpointing import restore_checkpoint
    a, model = build_template(cfg, training=True)
    _, eval_model = a, RB.build_model(a.arm, a.d_model, a.ssm_size,
                                      a.n_layers, training=False)
    dummy = jnp.zeros((2, SC.N_FRAMES, SC.N_MFCC))
    variables = model.init({"params": jax.random.PRNGKey(a.seed),
                            "dropout": jax.random.PRNGKey(a.seed + 1)},
                           dummy, jnp.ones((2, SC.N_FRAMES)), None)
    tx, _, _ = RB.make_optimizer(a, cfg.get("_steps_per_epoch", 843))
    state = RB.TrainState.create(apply_fn=model.apply,
                                 params=variables["params"], tx=tx,
                                 batch_stats=variables.get("batch_stats"))
    state, bs, meta = restore_checkpoint(run_dir, name, state,
                                         state.batch_stats)
    # R5: record the ACTUAL dtypes rather than leaving an empty placeholder.
    # Nothing is cast here; if a dtype differed from the saved one it would
    # show up as a mismatch in this record.
    def _dtypes(tree):
        from flax.traverse_util import flatten_dict
        if tree is None:
            return None
        return sorted({str(v.dtype) for v in flatten_dict(tree).values()})
    casts = dict(params_dtypes=_dtypes(state.params),
                 batch_stats_dtypes=_dtypes(bs),
                 cast_applied=False,
                 note="restored as saved; no dtype conversion performed")
    return dict(args=a, model=model, eval_model=eval_model, params=state.params,
                batch_stats=bs, step=int(state.step), meta=meta,
                init_params=variables["params"],
                init_batch_stats=variables.get("batch_stats"), casts=casts)


# ---------------------------------------------------------- C: core extract
def single_module(arm, d_model, ssm_size, n_layers, training=False):
    return RawatClassifier(
        ssm=init_substrate_ssm(arm, physical=SYMMETRIC_REFERENCE,
                               **RB.ssm_kwargs(d_model, ssm_size)),
        d_model=d_model, n_layers=n_layers, d_output=RB.D_OUTPUT,
        training=training)


def extract_cores(loaded, arm):
    a = loaded["args"]
    model = loaded["model"]
    variables = {"params": loaded["params"]}
    if loaded["batch_stats"] is not None:
        variables["batch_stats"] = loaded["batch_stats"]
    out = []
    for layer in range(a.n_layers):
        core = SD.read_core(model, variables, layer)
        K = SD.impulse_matrices(core, N_LAGS)
        rho = SD.spectral_radius(core)
        rec = dict(
            arm=arm, layer=layer, response=core["response"],
            input_gain=core["input_gain"], clip_eigs=bool(core["clip_eigs"]),
            physical=core["physical"],
            poles=SD.pole_summary(core),
            impulse_bands=SD.band_energy(K, BANDS),
            spectral_radius_descriptive_only=rho,
            window_lags=N_LAGS,
            window_remainder="unknown (NOT bounded); band energies are "
                             "explicitly windowed at 0..511",
            K0_norm=float(onp.sqrt(onp.sum(K[0] ** 2))),
            D_norm=float(onp.sqrt(onp.sum(onp.asarray(core["D"]) ** 2))),
            K0_minus_D_norm=float(onp.sqrt(onp.sum(
                (K[0] - onp.diag(onp.asarray(core["D"]))) ** 2))),
        )
        if core["response"] != "one_tap":
            Kc = SD.counterfactual_one_tap(core, N_LAGS)
            rec["counterfactual_one_tap"] = dict(
                bands=SD.band_energy(Kc, BANDS),
                K0_norm=float(onp.sqrt(onp.sum(Kc[0] ** 2))),
                rel_change_vs_executed=float(
                    onp.sqrt(onp.sum((K - Kc) ** 2))
                    / max(onp.sqrt(onp.sum(Kc ** 2)), 1e-30)))
        w, Hf = SD.frequency_response(core, N_FREQ)
        mag = onp.abs(Hf).reshape(N_FREQ, -1)
        rec["frequency"] = dict(
            w=w.tolist(),
            gain_fro=onp.sqrt((mag ** 2).sum(axis=1)).tolist(),
            dc_gain=float(onp.sqrt((onp.abs(Hf[0]) ** 2).sum())),
            nyquist_gain=float(onp.sqrt((onp.abs(Hf[-1]) ** 2).sum())))
        out.append(rec)
    return out


def verify_adapter_all_layers(loaded, arm, n=24, tol=1e-4):
    """Adapter vs executed CORE, for EVERY reported layer and EVERY input.

    R3: the previous version checked layer 0 and at most six input coordinates
    while four layers were reported. All H impulses are now batched through a
    vmap, so the checked scope equals the reported scope, and the scope is
    recorded in the result rather than implied.

    The target is the recurrent CORE (`layers[l].seq`), not the whole nonlinear
    residual SequenceLayer; the wording is kept consistent with that (R5).
    """
    a = loaded["args"]
    variables = {"params": loaded["params"]}
    if loaded["batch_stats"] is not None:
        variables["batch_stats"] = loaded["batch_stats"]
    single = single_module(a.arm, a.d_model, a.ssm_size, a.n_layers,
                           training=False)
    results = []
    for layer in range(a.n_layers):
        core = SD.read_core(loaded["model"], variables, layer)
        K = SD.impulse_matrices(core, n)                 # (n, H, H)
        H = int(core["H"])
        X = onp.zeros((H, n, H), dtype=onp.float32)
        for h in range(H):
            X[h, 0, h] = 1.0

        def one(xx, _layer=layer):
            return single.apply(
                variables, xx, jnp.ones(n), None,
                method=lambda m, u, t, l: m.encoder.layers[_layer].seq(u))

        Y = onp.asarray(jax.vmap(one)(jnp.asarray(X)))   # (H, n, H)
        pred = onp.transpose(K, (2, 0, 1))               # (H, n, H)
        den = max(float(onp.max(onp.abs(pred))), 1e-12)
        worst = float(onp.max(onp.abs(Y - pred))) / den
        results.append(dict(arm=arm, layer=layer, worst_rel=worst, tol=tol,
                            passed=bool(worst < tol), scope="recurrent core",
                            n_inputs_checked=H, n_lags_checked=int(n),
                            n_inputs_total=H))
    return results


# ------------------------------------------------------------- D: gradients
def fixed_subset(Xtr, Ytr):
    idx = onp.random.RandomState(SUBSET_SEED).permutation(
        Xtr.shape[0])[:N_SUBSET]
    idx = onp.sort(idx)
    return idx, jnp.asarray(onp.asarray(Xtr[idx])), jnp.asarray(Ytr[idx])


def block_activation_report(loaded, x, y):
    """Per-layer activation RMS and gradients at each recurrent-core INPUT and
    at the pre-pooling activation.

    R5: parameter-gradient norms alone cannot establish depth attenuation. The
    stack is re-run through the bound submodules with an additive zero offset
    at each layer input and before pooling; differentiating with respect to
    those offsets gives the exact activation gradients without modifying the
    model. Inference mode: saved batch statistics, dropout disabled.
    """
    import optax
    from flax import linen as nn
    a = loaded["args"]
    variables = {"params": loaded["params"]}
    if loaded["batch_stats"] is not None:
        variables["batch_stats"] = loaded["batch_stats"]
    single = single_module(a.arm, a.d_model, a.ssm_size, a.n_layers,
                           training=False)
    L = x.shape[1]
    n_sites = a.n_layers + 1                       # layer inputs + pre-pooling

    def fwd(m, xx, tt, ll, offs):
        h = m.encoder.encoder(xx)
        acts = []
        for i, lyr in enumerate(m.encoder.layers):
            h = h + offs[i]
            h = lyr(h)
            acts.append(h)
        h = h + offs[len(m.encoder.layers)]
        z = m.readout_proj(m.readout_norm(h))
        pooled = jnp.mean(z, axis=0)
        out = m.mlp_out(m.drop(nn.gelu(m.mlp_in(pooled))))
        return out, acts

    def loss_fn(offs):
        def one(xx):
            return single.apply(variables, xx, jnp.ones(L), None, offs,
                                method=fwd)
        logits, acts = jax.vmap(one)(x)
        onehot = jax.nn.one_hot(y, RB.D_OUTPUT)
        loss = optax.softmax_cross_entropy(
            logits, optax.smooth_labels(onehot, a.label_smoothing)).mean()
        return loss, acts

    offs = [jnp.zeros((L, a.d_model)) for _ in range(n_sites)]
    (loss, acts), g = jax.value_and_grad(loss_fn, has_aux=True)(offs)
    sites = []
    for i, gi in enumerate(g):
        name = (f"core_input_layer_{i}" if i < a.n_layers
                else "pre_pooling_activation")
        gi = onp.asarray(gi)
        sites.append(dict(site=name,
                          grad_norm=float(onp.sqrt(onp.sum(gi ** 2))),
                          grad_rms=float(onp.sqrt(onp.mean(gi ** 2)))))
    act = []
    for i, ai in enumerate(acts):
        ai = onp.asarray(ai)
        act.append(dict(layer=i, activation_rms=float(onp.sqrt(onp.mean(
            ai ** 2)))))
    return dict(loss=float(loss), sites=sites, activations=act,
                mode="inference")


def grad_report(loaded, x, y, training_mode=False):
    a = loaded["args"]
    model = (loaded["model"] if training_mode else loaded["eval_model"])
    params = loaded["params"]
    bs = loaded["batch_stats"]
    rng = jax.random.PRNGKey(0)

    def loss_fn(p):
        l, (logits, _) = RB.loss_and_logits(p, bs, model, x, y, rng,
                                            training_mode, a.label_smoothing)
        return l, logits

    (loss, logits), g = jax.value_and_grad(loss_fn, has_aux=True)(params)
    from flax.traverse_util import flatten_dict
    gf = {"/".join(k): onp.asarray(v) for k, v in flatten_dict(g).items()}
    pf = {"/".join(k): onp.asarray(v) for k, v in flatten_dict(params).items()}
    per_layer = {}
    for name, gv in gf.items():
        layer = None
        for tok in name.split("/"):
            if tok.startswith("layers_"):
                layer = tok
        key = layer or "readout_or_other"
        d = per_layer.setdefault(key, dict(g2=0.0, p2=0.0, n=0))
        d["g2"] += float(onp.sum(gv ** 2))
        d["p2"] += float(onp.sum(pf[name] ** 2))
        d["n"] += int(gv.size)
    for k, d in per_layer.items():
        d["grad_norm"] = float(onp.sqrt(d["g2"]))
        d["param_norm"] = float(onp.sqrt(d["p2"]))
        d["ratio"] = d["grad_norm"] / max(d["param_norm"], 1e-30)
        d.pop("g2"); d.pop("p2")
    total = float(onp.sqrt(sum(float(onp.sum(v ** 2)) for v in gf.values())))
    return dict(loss=float(loss), total_grad_norm=total, per_layer=per_layer,
                logits_rms=float(onp.sqrt(onp.mean(onp.asarray(logits) ** 2))),
                training_mode=bool(training_mode))


def sensitivity_probe(loaded, x):
    """VJP of the PRE-POOLING encoder output at fixed anchors, by lag band."""
    a = loaded["args"]
    variables = {"params": loaded["params"]}
    if loaded["batch_stats"] is not None:
        variables["batch_stats"] = loaded["batch_stats"]
    single = single_module(a.arm, a.d_model, a.ssm_size, a.n_layers,
                           training=False)
    L = x.shape[1]

    def enc(xx):
        return single.apply(variables, xx, jnp.ones(L), None,
                            method=lambda m, u, t, l: m.encoder(u, t))

    def batched(xx):
        return jax.vmap(enc)(xx)

    rs = onp.random.RandomState(DIRECTION_SEED)
    out = []
    for anchor in ANCHORS:
        def f(xx):
            return batched(xx)[:, anchor, :]
        y, vjp = jax.vjp(f, x)
        dm = y.shape[-1]
        per_dir = []
        for d in range(N_DIRECTIONS):
            v = rs.choice([-1.0, 1.0], size=(x.shape[0], dm))
            v = v / onp.sqrt(dm)
            g = onp.asarray(vjp(jnp.asarray(v, dtype=y.dtype))[0])   # (B,L,Hin)
            mag = onp.sqrt((g ** 2).sum(axis=(0, 2)))                # per frame
            future = float(onp.max(onp.abs(g[:, anchor + 1:, :]))) \
                if anchor + 1 < L else 0.0
            bands = []
            for lo, hi in BANDS:
                f0, f1 = anchor - hi, anchor - lo
                f0 = max(f0, 0)
                if f1 < 0:
                    bands.append(dict(lo=lo, hi=hi, energy=0.0,
                                      in_history=False))
                    continue
                e = float(onp.sum(mag[f0:f1 + 1] ** 2))
                bands.append(dict(lo=lo, hi=hi, energy=e, in_history=True))
            tot = sum(b["energy"] for b in bands)
            for b in bands:
                b["fraction"] = b["energy"] / tot if tot > 0 else float("nan")
            per_dir.append(dict(direction=d, bands=bands,
                                future_sensitivity_max=future,
                                total_energy=tot))
        out.append(dict(anchor=anchor, directions=per_dir))
    return out


def finite_difference_check(loaded, x):
    """One fixed direction: JVP against a central difference."""
    a = loaded["args"]
    variables = {"params": loaded["params"]}
    if loaded["batch_stats"] is not None:
        variables["batch_stats"] = loaded["batch_stats"]
    single = single_module(a.arm, a.d_model, a.ssm_size, a.n_layers,
                           training=False)
    L = x.shape[1]

    def enc_sum(xx):
        h = jax.vmap(lambda u: single.apply(
            variables, u, jnp.ones(L), None,
            method=lambda m, uu, t, l: m.encoder(uu, t)))(xx)
        return jnp.sum(h[:, ANCHORS[0], :])

    rs = onp.random.RandomState(DIRECTION_SEED + 1)
    d = rs.randn(*x.shape)
    # RMS-1, NOT unit L2. A direction normalized in L2 over ~10^5 elements
    # moves each element by only ~2.5e-6 at step 1e-3, roughly 20x float32
    # epsilon, so the central difference measures ROUNDING rather than the
    # derivative. That mis-specification is why the first version of this
    # check failed at rel 0.4-2.9; the tolerance was not the problem.
    d = d / d.std()
    d = jnp.asarray(d, dtype=x.dtype)
    _, jvp = jax.jvp(enc_sum, (x,), (d,))
    per_step = {}
    for h in FD_STEPS:
        plus = float(enc_sum(x + h * d))
        minus = float(enc_sum(x - h * d))
        fd = (plus - minus) / (2 * h)
        per_step[f"{h:.0e}"] = dict(
            finite_difference=fd,
            rel_error=abs(float(jvp) - fd) / max(abs(fd), 1e-8))
    gate = per_step[f"{FD_GATE_STEP:.0e}"]
    return dict(jvp=float(jvp), per_step=per_step,
                gate_step=FD_GATE_STEP, rel_error=gate["rel_error"],
                tol=FD_REL_TOL, passed=bool(gate["rel_error"] < FD_REL_TOL),
                direction_norm="rms1",
                note="float32 central difference; the direction is RMS-1 so "
                     "the per-element perturbation stays above float32 "
                     "resolution. Both steps are reported; the gate uses 1e-2, "
                     "where rounding no longer dominates.")


def make_plots(out, screen, cores, sens, ST):
    """Plots are OPTIONAL: a failure here never fails the diagnostic."""
    plots = dict(attempted=True, written=[], error=None)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        spe = None
        for sc in screen:
            if sc["lr"] != 1e-3:
                continue
            spe = sc["lr_schedule"].get("steps_per_epoch_FROM_SAVED_METADATA") \
                or spe
            ep = [c["epoch"] for c in sc["curve"]]
            ax[0].plot(ep, [c["val_accuracy"] for c in sc["curve"]],
                       marker="o", label=sc["arm"])
            ax[1].plot(ep, [c["train_loss_SMOOTHED"] for c in sc["curve"]],
                       marker="o", label=sc["arm"])
        # R5: steps/epoch comes from SAVED METADATA, not a hardcoded constant
        if spe:
            steps = onp.arange(0, 10 * spe)
            cos10 = 1e-6 + (1e-3 - 1e-6) * 0.5 * (
                1 + onp.cos(onp.pi * steps / (10 * spe)))
            cos300 = 1e-6 + (1e-3 - 1e-6) * 0.5 * (
                1 + onp.cos(onp.pi * steps / (300 * spe)))
            ax[2].plot(steps, cos10, label="Stage 2: TEN-EPOCH cosine")
            ax[2].plot(steps, cos300, "--",
                       label="first 10 epochs of a 300-epoch cosine")
            ax[2].set_title(f"schedule definitions (analytic, NOT run)\n"
                            f"steps/epoch = {spe} from saved metadata")
        ax[2].set_xlabel("update"); ax[2].set_ylabel("learning rate")
        ax[0].set_title("validation accuracy"); ax[0].set_xlabel("epoch")
        ax[1].set_title("train loss (label-smoothed)"); ax[1].set_xlabel("epoch")
        for a_ in ax:
            a_.legend(fontsize=7); a_.grid(alpha=.3)
        fig.tight_layout()
        f1 = os.path.join(out, "curves_and_schedule.png")
        fig.savefig(f1, dpi=110); plt.close(fig); plots["written"].append(f1)

        if cores:
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            for c in cores:
                if c["layer"] != 0:
                    continue
                ax[0].scatter(c["poles"]["continuous_pole_re"],
                              c["poles"]["continuous_pole_im"],
                              s=14, label=c["arm"], alpha=.7)
                ax[1].plot(range(len(BANDS)),
                           [b["energy"] for b in c["impulse_bands"]["bands"]],
                           marker="o", label=c["arm"])
            ax[0].set_title("CONTINUOUS poles, layer 0 (not log of discrete)")
            ax[0].set_xlabel("Re"); ax[0].set_ylabel("Im"); ax[0].grid(alpha=.3)
            ax[1].set_yscale("log")
            ax[1].set_title("core impulse energy by lag band (windowed 0..511)")
            ax[1].set_xticks(list(range(len(BANDS))))
            ax[1].set_xticklabels([f"{lo}-{hi}" for lo, hi in BANDS],
                                  fontsize=7)
            ax[1].grid(alpha=.3)
            for a_ in ax:
                a_.legend(fontsize=7)
            fig.tight_layout()
            f2 = os.path.join(out, "poles_and_impulse.png")
            fig.savefig(f2, dpi=110); plt.close(fig)
            plots["written"].append(f2)

        if sens:
            fig, ax = plt.subplots(1, len(ANCHORS), figsize=(11, 4))
            ax = onp.atleast_1d(ax)
            for ai, anchor in enumerate(ANCHORS):
                for srec in sens:
                    rec = next(a_ for a_ in srec["anchors"]
                               if a_["anchor"] == anchor)
                    fr = onp.mean([[b["fraction"] for b in d["bands"]]
                                   for d in rec["directions"]], axis=0)
                    ax[ai].plot(range(len(BANDS)), fr, marker="o",
                                label=srec["arm"])
                ax[ai].set_title(f"pre-pooling sensitivity, anchor {anchor}")
                ax[ai].set_yscale("log")
                ax[ai].set_xticks(list(range(len(BANDS))))
                ax[ai].set_xticklabels([f"{lo}-{hi}" for lo, hi in BANDS],
                                       fontsize=7)
                ax[ai].grid(alpha=.3); ax[ai].legend(fontsize=7)
            fig.tight_layout()
            f3 = os.path.join(out, "sensitivity_by_lag.png")
            fig.savefig(f3, dpi=110); plt.close(fig)
            plots["written"].append(f3)
    except Exception as exc:
        plots["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[!] plots skipped: {plots['error']}")
        ST.skip("plots", plots["error"])
    write_json(os.path.join(out, "plots.json"), plots)
    return plots


# ------------------------------------------------------------------- main
PHASES = ("A", "B", "B2", "C", "C_init", "C_last", "D1", "D2", "D3")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2_dir", default="/Users/durso/s5-runs/stage2")
    ap.add_argument("--out_root",
                    default="/Users/durso/s5-runs/stage2-diagnostics")
    ap.add_argument("--data_cache", default="/Users/durso/s5-runs/sc10_cache")
    ap.add_argument("--budget_s", type=float, default=1200.0)
    ap.add_argument("--deadline", type=float, default=None,
                    help="absolute epoch deadline from the launcher, so the "
                         "backend check and focused tests sit INSIDE the cap")
    ap.add_argument("--reserve_s", type=float, default=25.0,
                    help="cleanup reserve kept inside the cap")
    ap.add_argument("--only_phases", default=None,
                    help="comma-separated subset of "
                         + ",".join(PHASES) +
                         "; the documented way to run genuinely missing "
                         "phases. Each invocation is otherwise a FRESH RERUN, "
                         "not a resumption.")
    ap.add_argument("--matmul_precision", default="highest")
    ap.add_argument("--allow_cpu", action="store_true")
    args = ap.parse_args()

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(args.out_root, run_id)
    os.makedirs(out, exist_ok=True)
    ST = Status(out)
    B = Budget(args.budget_s, args.deadline)
    want = set(PHASES if not args.only_phases
               else [p.strip() for p in args.only_phases.split(",")])

    hashes_before = None
    hashes_after = None
    hashes_ok = None                      # None = UNKNOWN, never assumed True
    loaded_arms = {}
    backend = "unknown"
    interrupted = None

    def _sigterm(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            signal.signal(sig, _sigterm)
        except Exception:
            pass

    try:
        # R4: GPU is verified INSIDE the budget and before any numerical work
        backend = RB.require_gpu(args.allow_cpu)
        jax.config.update("jax_default_matmul_precision", args.matmul_precision)
        print(f"[*] output {out}")
        print(f"[*] backend={backend} budget={B.limit:.0f}s "
              f"reserve={args.reserve_s:.0f}s phases={sorted(want)}")

        # ---- A: inventory + hashes BEFORE
        hashes_before = hash_tree(args.stage2_dir)
        runs = [read_run(os.path.join(args.stage2_dir, d))
                for d in sorted(os.listdir(args.stage2_dir))
                if os.path.isdir(os.path.join(args.stage2_dir, d))]
        write_json(os.path.join(out, "manifest.json"), dict(
            run_id=run_id, stage2_dir=args.stage2_dir, n_runs=len(runs),
            backend=backend, provenance=provenance(),
            source_hashes_before=hashes_before,
            runs=[dict(name=r["name"],
                       arm=r.get("config_json", {}).get("config", {}).get("arm"),
                       lr=r.get("config_json", {}).get("config", {}).get("lr"),
                       seed=r.get("config_json", {}).get("config", {}).get("seed"),
                       epochs_recorded=len(r.get("metrics", [])),
                       steps_per_epoch=r.get("config_json", {}).get("extra", {})
                       .get("steps_per_epoch"),
                       has_best=os.path.exists(os.path.join(r["run_dir"],
                                                            "best.msgpack")),
                       has_last=os.path.exists(os.path.join(r["run_dir"],
                                                            "last.msgpack")),
                       original_provenance=r.get("config_json", {})
                       .get("provenance"))
                  for r in runs]))
        ST.record("A_inventory", len(runs) > 0, detail=f"{len(runs)} runs")
        B.mark("A_inventory")

        # ---- B: screen reconstruction from saved records only
        screen = []
        for r in runs:
            cfg = r.get("config_json", {}).get("config", {})
            extra = r.get("config_json", {}).get("extra", {})
            mets = r.get("metrics", [])
            if not mets:
                continue
            best = max(mets, key=lambda m: m.get("val_accuracy", -1))
            ties = [m for m in mets
                    if m.get("val_accuracy") == best.get("val_accuracy")]
            spe = extra.get("steps_per_epoch")
            screen.append(dict(
                run=r["name"], arm=cfg.get("arm"), lr=cfg.get("lr"),
                seed=cfg.get("seed"), epochs=len(mets),
                best_epoch=best.get("epoch"), last_epoch=mets[-1].get("epoch"),
                best_is_last=bool(best.get("epoch") == mets[-1].get("epoch")),
                n_ties_on_val_accuracy=len(ties),
                selection_rule_used_by_runner=
                "first strict improvement in val_accuracy; ties keep the "
                "EARLIER epoch",
                val_accuracy=best.get("val_accuracy"),
                val_cross_entropy=best.get("val_cross_entropy"),
                curve=[dict(epoch=m.get("epoch"),
                            train_loss_SMOOTHED=m.get("train_loss"),
                            train_acc=m.get("train_acc"),
                            val_accuracy=m.get("val_accuracy"),
                            val_cross_entropy_UNSMOOTHED=m.get(
                                "val_cross_entropy"),
                            grad_norm_LAST_MINIBATCH=m.get("grad_norm"),
                            epoch_s_EXCLUDES_CHECKPOINT_WRITE=m.get("epoch_s"))
                       for m in mets],
                lr_schedule=dict(
                    kind="cosine_decay over steps_per_epoch*epochs",
                    note="TEN-EPOCH COSINE SCHEDULE, not a prefix of a "
                         "300-epoch schedule",
                    init_value=cfg.get("lr"), final_value=cfg.get("lr_final"),
                    epochs=cfg.get("epochs"),
                    steps_per_epoch_FROM_SAVED_METADATA=spe,
                    total_decay_steps=(spe * cfg["epochs"]
                                       if spe and cfg.get("epochs") else None))))
        write_json(os.path.join(out, "screen_reconstruction.json"), screen)
        ST.record("B_screen_reconstruction", len(screen) > 0)
        B.mark("B_screen_reconstruction")

        # ---- B2: restoration corroboration (REQUIRED; gates everything after)
        data, dmani = SC.load_splits(args.data_cache, splits=("train", "val"))
        Xtr, Ytr = data["train"]
        Xva, Yva = data["val"]
        sel = {sc["arm"]: sc for sc in screen
               if sc["lr"] == 1e-3 and sc["arm"] in ARMS_1E3}
        restore_checks = []
        for arm in ARMS_1E3:
            if "B2" not in want:
                ST.skip("B2_restore", "phase not selected", arm=arm)
                continue
            if arm not in sel:
                ST.skip("B2_restore", "no lr=1e-3 run found", arm=arm,
                        required=True)
                continue
            if not B.ok(args.reserve_s + 60):
                ST.skip("B2_restore", "budget exhausted", arm=arm,
                        required=True)
                B.mark(f"B2_{arm}", "skipped_budget")
                continue
            r = next(x for x in runs if x["name"] == sel[arm]["run"])
            cfg = r["config_json"]["config"]
            L = restore_readonly(r["run_dir"], cfg, "best")
            res = RB.evaluate_split(L["params"], L["batch_stats"],
                                    L["eval_model"], Xva, Yva,
                                    cfg.get("batch", 32))
            n = res["n"]
            saved_correct = int(round(sel[arm]["val_accuracy"] * n))
            got_correct = int(round(res["accuracy"] * n))
            ce_diff = abs(res["cross_entropy"] - sel[arm]["val_cross_entropy"])
            rec = dict(arm=arm, run=sel[arm]["run"], n=n,
                       saved_accuracy=sel[arm]["val_accuracy"],
                       restored_accuracy=res["accuracy"],
                       saved_correct=saved_correct, restored_correct=got_correct,
                       counts_match=bool(saved_correct == got_correct),
                       saved_ce=sel[arm]["val_cross_entropy"],
                       restored_ce=res["cross_entropy"], ce_abs_diff=ce_diff,
                       ce_tol=CE_TOL, ce_within_tol=bool(ce_diff < CE_TOL),
                       checkpoint_epoch=L["meta"].get("epoch"),
                       dtypes=L["casts"])
            restore_checks.append(rec)
            write_json(os.path.join(out, "restore_checks.json"), restore_checks)
            ok = ST.record("B2_restore_counts", rec["counts_match"], arm=arm,
                           detail=f"saved {saved_correct} vs restored "
                                  f"{got_correct}")
            ST.record("B2_restore_ce", rec["ce_within_tol"], arm=arm,
                      detail=f"|diff| {ce_diff:.3e} vs tol {CE_TOL}")
            print(f"  restore {arm}: saved {saved_correct}, restored "
                  f"{got_correct}, ce diff {ce_diff:.2e}")
            if ok:
                loaded_arms[arm] = (L, r, sel[arm])
            else:
                print(f"  [FAIL] {arm}: restoration mismatch; dependent "
                      f"interpretation for this arm is STOPPED")
            B.mark(f"B2_{arm}")

        # ---- C: core extraction, gated by the adapter check per arm
        cores, adapter_checks = [], []
        for arm, (L, r, sc) in list(loaded_arms.items()):
            if "C" not in want or ST.blocked(arm):
                ST.skip("C_cores", "arm blocked or phase not selected", arm=arm)
                continue
            if not B.ok(args.reserve_s + 45):
                ST.skip("C_cores", "budget exhausted", arm=arm, required=True)
                B.mark(f"C_{arm}", "skipped_budget")
                continue
            checks = verify_adapter_all_layers(L, arm)
            adapter_checks.extend(checks)
            write_json(os.path.join(out, "adapter_checks.json"), adapter_checks)
            ok = all(c["passed"] for c in checks)
            ST.record("C_adapter_equals_executed_core", ok, arm=arm,
                      detail=f"worst rel "
                             f"{max(c['worst_rel'] for c in checks):.2e} over "
                             f"{len(checks)} layers")
            if not ok:
                continue
            cores.extend(extract_cores(L, arm))
            write_json(os.path.join(out, "cores.json"), cores)
            B.mark(f"C_{arm}")

        # ---- C_init: reconstructed seed-100 initialization (R5 D4, light)
        init_cores = []
        for arm, (L, r, sc) in list(loaded_arms.items()):
            if "C_init" not in want or ST.blocked(arm):
                continue
            if not B.ok(args.reserve_s + 30):
                ST.skip("C_init", "budget exhausted", arm=arm)
                B.mark(f"C_init_{arm}", "skipped_budget")
                continue
            iv = {"params": L["init_params"]}
            if L["init_batch_stats"] is not None:
                iv["batch_stats"] = L["init_batch_stats"]
            for layer in range(L["args"].n_layers):
                core = SD.read_core(L["model"], iv, layer)
                K = SD.impulse_matrices(core, N_LAGS)
                init_cores.append(dict(
                    arm=arm, layer=layer, label="RECONSTRUCTED INITIALIZATION "
                                                "(not a saved checkpoint)",
                    poles=SD.pole_summary(core),
                    impulse_bands=SD.band_energy(K, BANDS),
                    window_lags=N_LAGS,
                    window_remainder="unknown (not bounded)"))
            write_json(os.path.join(out, "init_cores.json"), init_cores)
            B.mark(f"C_init_{arm}")

        # ---- C_last: last vs best, lightweight pole summary only (R5)
        last_cmp = []
        for arm, (L, r, sc) in list(loaded_arms.items()):
            if "C_last" not in want or ST.blocked(arm):
                continue
            if sc["best_is_last"]:
                last_cmp.append(dict(arm=arm, compared=False,
                                     reason="best IS last"))
                continue
            if not B.ok(args.reserve_s + 30):
                ST.skip("C_last", "budget exhausted", arm=arm)
                B.mark(f"C_last_{arm}", "skipped_budget")
                continue
            cfg = r["config_json"]["config"]
            LL = restore_readonly(r["run_dir"], cfg, "last")
            lv = {"params": LL["params"]}
            if LL["batch_stats"] is not None:
                lv["batch_stats"] = LL["batch_stats"]
            for layer in range(LL["args"].n_layers):
                last_cmp.append(dict(arm=arm, layer=layer, compared=True,
                                     poles=SD.pole_summary(
                                         SD.read_core(LL["model"], lv, layer))))
            write_json(os.path.join(out, "last_vs_best.json"), last_cmp)
            B.mark(f"C_last_{arm}")

        # ---- D: gradients, activations and sensitivity
        idx, xs, ys = fixed_subset(Xtr, Ytr)
        write_json(os.path.join(out, "subset.json"),
                   dict(seed=SUBSET_SEED, n=int(len(idx)),
                        indices=idx.tolist(), split="train",
                        split_sha256=dmani["feature_sha256"]["train"],
                        note="post-screen diagnostic sample, NOT an "
                             "independent confirmation sample"))
        grads, sens, fdchecks, blocks = [], [], [], []
        for arm, (L, r, sc) in list(loaded_arms.items()):
            if "D1" not in want or ST.blocked(arm):
                continue
            if not B.ok(args.reserve_s + 45):
                ST.skip("D1_gradients", "budget exhausted", arm=arm)
                B.mark(f"D1_{arm}", "skipped_budget")
                continue
            g = grad_report(L, xs, ys, training_mode=False)
            g["arm"] = arm
            grads.append(g)
            blocks.append(dict(arm=arm, **block_activation_report(L, xs, ys)))
            write_json(os.path.join(out, "gradients.json"), grads)
            write_json(os.path.join(out, "block_activations.json"), blocks)
            B.mark(f"D1_{arm}")

        for arm, (L, r, sc) in list(loaded_arms.items()):
            if "D2" not in want or ST.blocked(arm):
                continue
            if not B.ok(args.reserve_s + 60):
                ST.skip("D2_sensitivity", "budget exhausted", arm=arm)
                B.mark(f"D2_{arm}", "skipped_budget")
                continue
            fd = dict(arm=arm, **finite_difference_check(L, xs))
            fdchecks.append(fd)
            write_json(os.path.join(out, "fd_checks.json"), fdchecks)
            ST.record("D2_jvp_vs_finite_difference", fd["passed"], arm=arm,
                      detail=f"rel {fd['rel_error']:.2e} at step "
                             f"{fd['gate_step']:.0e}")
            srec = dict(arm=arm, anchors=sensitivity_probe(L, xs))
            sens.append(srec)
            write_json(os.path.join(out, "sensitivity.json"), sens)
            worst_future = max(
                d["future_sensitivity_max"]
                for a_ in srec["anchors"] for d in a_["directions"])
            ST.record("D2_no_future_dependency", worst_future == 0.0, arm=arm,
                      detail=f"max future sensitivity {worst_future:.3e}")
            B.mark(f"D2_{arm}")

        for arm, (L, r, sc) in list(loaded_arms.items()):
            if "D3" not in want or ST.blocked(arm):
                continue
            if not B.ok(args.reserve_s + 40):
                ST.skip("D3_training_mode_gradients",
                        "budget exhausted (OPTIONAL phase)", arm=arm)
                B.mark(f"D3_{arm}", "skipped_budget")
                continue
            g = grad_report(L, xs, ys, training_mode=True)
            g["arm"] = arm
            g["label"] = ("TRAINING-MODE normalization: BatchNorm couples "
                          "samples and times; not a purely causal recurrence "
                          "diagnostic")
            grads.append(g)
            write_json(os.path.join(out, "gradients.json"), grads)
            B.mark(f"D3_{arm}")

        make_plots(out, screen, cores, sens, ST)

    except KeyboardInterrupt as exc:
        interrupted = str(exc)
        print(f"[!] interrupted: {exc}")
    except Exception as exc:                      # preserve what exists
        interrupted = f"{type(exc).__name__}: {exc}"
        print(f"[!] ERROR: {interrupted}")
        import traceback
        traceback.print_exc()
    finally:
        # R3.5: always TRY to close the read-only guarantee; if we cannot,
        # invariance is reported UNKNOWN, never assumed true.
        try:
            if hashes_before is not None:
                hashes_after = hash_tree(args.stage2_dir)
                hashes_ok = (hashes_before == hashes_after)
        except Exception as exc:
            print(f"[!] could not re-hash sources: {exc}")
            hashes_ok = None
        diff = ([k for k in set(hashes_before or {}) | set(hashes_after or {})
                 if (hashes_before or {}).get(k) != (hashes_after or {}).get(k)]
                if hashes_after is not None else None)
        if hashes_ok is True:
            ST.record("F_source_files_unchanged", True)
        elif hashes_ok is False:
            ST.record("F_source_files_unchanged", False, detail=str(diff))
        else:
            ST.skip("F_source_files_unchanged",
                    "could not be determined", required=True)
        budget_incomplete = bool(B.incomplete or interrupted)
        status = ST.final(hashes_ok, budget_incomplete)
        summary = dict(
            run_id=run_id, out=out, backend=backend, status=status,
            wall_s=time.time() - B.t0, budget_s=B.limit,
            deadline_from_launcher=bool(args.deadline is not None),
            phases=B.phases, interrupted=interrupted,
            failed_required_checks=ST.failed_required,
            unexecuted_checks=ST.unexecuted,
            source_hashes_unchanged=hashes_ok,
            source_hash_differences=diff,
            test_split_opened=False,
            n_arms_analysed=len(loaded_arms),
            not_implemented=[
                "initialization GRADIENT probes (D4): deferred by the "
                "protocol's priority order; reconstructed-initialization CORES "
                "are computed",
            ])
        write_json(os.path.join(out, "summary.json"), summary)
        print(f"[*] source hashes unchanged: {hashes_ok}")
        if diff:
            print(f"[!] CHANGED: {diff}")
        print(f"[*] wall {summary['wall_s']:.1f}s of {B.limit:.0f}s")
        print(f"DIAGNOSTIC_STATUS={status} out={out}")
        code = EXIT_CODES[status]
        sys.exit(code)


if __name__ == "__main__":
    main()
