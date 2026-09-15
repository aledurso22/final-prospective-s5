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


# --------------------------------------------------------------- utilities
class Budget:
    """Wall-clock deadline with phase accounting. Never auto-extends."""

    def __init__(self, seconds):
        self.t0 = time.time()
        self.limit = seconds
        self.phases = []

    def left(self):
        return self.limit - (time.time() - self.t0)

    def ok(self, need=0.0):
        return self.left() > need

    def mark(self, name, status="complete", note=None):
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
    casts = {}
    for k, v in jax.tree_util.tree_leaves_with_path(state.params):
        pass
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
            spectral_radius=rho,
            truncation_tail_bound=SD.truncation_tail_bound(K, rho),
            window_lags=N_LAGS,
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


def verify_adapter_vs_forward(loaded, arm, layer=0, n=24, tol=1e-4):
    """The adapter must equal the EXECUTED layer, on the restored checkpoint."""
    a = loaded["args"]
    model = loaded["model"]
    variables = {"params": loaded["params"]}
    if loaded["batch_stats"] is not None:
        variables["batch_stats"] = loaded["batch_stats"]
    core = SD.read_core(model, variables, layer)
    K = SD.impulse_matrices(core, n)
    H = int(core["H"])

    def run(x):
        def f(m, xx, tt, ll):
            return m.encoder.layers[layer].seq(xx)
        return model.apply(variables, x[None], jnp.ones((1, x.shape[0])), None,
                           method=f)
    worst = 0.0
    for h in range(min(H, 6)):
        x = onp.zeros((n, H), dtype=onp.float32)
        x[0, h] = 1.0
        try:
            y = onp.asarray(run(jnp.asarray(x)))[0]
        except Exception:
            single = single_module(a.arm, a.d_model, a.ssm_size, a.n_layers)
            y = onp.asarray(single.apply(
                variables, jnp.asarray(x), jnp.ones(n), None,
                method=lambda m, xx, tt, ll: m.encoder.layers[layer].seq(xx)))
        den = max(float(onp.max(onp.abs(K[:, :, h]))), 1e-12)
        worst = max(worst, float(onp.max(onp.abs(y - K[:, :, h]))) / den)
    return dict(arm=arm, layer=layer, worst_rel=worst, tol=tol,
                passed=bool(worst < tol))


# ------------------------------------------------------------- D: gradients
def fixed_subset(Xtr, Ytr):
    idx = onp.random.RandomState(SUBSET_SEED).permutation(
        Xtr.shape[0])[:N_SUBSET]
    idx = onp.sort(idx)
    return idx, jnp.asarray(onp.asarray(Xtr[idx])), jnp.asarray(Ytr[idx])


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


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2_dir", default="/Users/durso/s5-runs/stage2")
    ap.add_argument("--out_root",
                    default="/Users/durso/s5-runs/stage2-diagnostics")
    ap.add_argument("--data_cache", default="/Users/durso/s5-runs/sc10_cache")
    ap.add_argument("--budget_s", type=float, default=1200.0)
    ap.add_argument("--matmul_precision", default="highest")
    ap.add_argument("--allow_cpu", action="store_true")
    args = ap.parse_args()

    backend = RB.require_gpu(args.allow_cpu)
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)
    B = Budget(args.budget_s)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(args.out_root, run_id)
    os.makedirs(out, exist_ok=True)
    print(f"[*] diagnostic output {out}")
    print(f"[*] backend={backend} budget={args.budget_s:.0f}s")

    # ---- A: inventory + hashes BEFORE
    hashes_before = hash_tree(args.stage2_dir)
    runs = []
    for d in sorted(os.listdir(args.stage2_dir)):
        rd = os.path.join(args.stage2_dir, d)
        if os.path.isdir(rd):
            runs.append(read_run(rd))
    manifest = dict(run_id=run_id, stage2_dir=args.stage2_dir,
                    n_runs=len(runs), backend=backend,
                    provenance=provenance(),
                    source_hashes_before=hashes_before,
                    runs=[dict(name=r["name"],
                               arm=r.get("config_json", {}).get("config", {})
                               .get("arm"),
                               lr=r.get("config_json", {}).get("config", {})
                               .get("lr"),
                               seed=r.get("config_json", {}).get("config", {})
                               .get("seed"),
                               epochs_recorded=len(r.get("metrics", [])),
                               has_best=os.path.exists(
                                   os.path.join(r["run_dir"], "best.msgpack")),
                               has_last=os.path.exists(
                                   os.path.join(r["run_dir"], "last.msgpack")),
                               original_provenance=r.get("config_json", {})
                               .get("provenance"))
                          for r in runs])
    write_json(os.path.join(out, "manifest.json"), manifest)
    B.mark("A_inventory")

    # ---- B: reconstruct the screen from saved records only
    screen = []
    for r in runs:
        cfg = r.get("config_json", {}).get("config", {})
        mets = r.get("metrics", [])
        if not mets:
            continue
        best = max(mets, key=lambda m: m.get("val_accuracy", -1))
        ties = [m for m in mets
                if m.get("val_accuracy") == best.get("val_accuracy")]
        screen.append(dict(
            run=r["name"], arm=cfg.get("arm"), lr=cfg.get("lr"),
            seed=cfg.get("seed"), epochs=len(mets),
            best_epoch=best.get("epoch"), last_epoch=mets[-1].get("epoch"),
            best_is_last=bool(best.get("epoch") == mets[-1].get("epoch")),
            n_ties_on_val_accuracy=len(ties),
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
                note="TEN-EPOCH COSINE SCHEDULE, not a prefix of a 300-epoch "
                     "schedule",
                init_value=cfg.get("lr"), final_value=cfg.get("lr_final"),
                epochs=cfg.get("epochs"))))
    write_json(os.path.join(out, "screen_reconstruction.json"), screen)
    B.mark("B_screen_reconstruction")

    # ---- B2: corroborate restoration on validation (no reselection)
    data, dmani = SC.load_splits(args.data_cache, splits=("train", "val"))
    Xtr, Ytr = data["train"]
    Xva, Yva = data["val"]
    sel = {}
    for s in screen:
        if s["lr"] == 1e-3 and s["arm"] in ARMS_1E3:
            sel[s["arm"]] = s
    restore_checks = []
    loaded_arms = {}
    for arm in ARMS_1E3:
        if arm not in sel or not B.ok(60):
            B.mark(f"B2_{arm}", "skipped_budget")
            continue
        r = next(x for x in runs if x["name"] == sel[arm]["run"])
        cfg = r["config_json"]["config"]
        L = restore_readonly(r["run_dir"], cfg, "best")
        loaded_arms[arm] = (L, r, sel[arm])
        res = RB.evaluate_split(L["params"], L["batch_stats"], L["eval_model"],
                                Xva, Yva, cfg.get("batch", 32))
        n = res["n"]
        saved_acc = sel[arm]["val_accuracy"]
        saved_correct = int(round(saved_acc * n))
        got_correct = int(round(res["accuracy"] * n))
        restore_checks.append(dict(
            arm=arm, run=sel[arm]["run"], n=n,
            saved_accuracy=saved_acc, restored_accuracy=res["accuracy"],
            saved_correct=saved_correct, restored_correct=got_correct,
            counts_match=bool(saved_correct == got_correct),
            saved_ce=sel[arm]["val_cross_entropy"],
            restored_ce=res["cross_entropy"],
            ce_abs_diff=abs(res["cross_entropy"]
                            - sel[arm]["val_cross_entropy"]),
            ce_tol=CE_TOL,
            ce_within_tol=bool(abs(res["cross_entropy"]
                                   - sel[arm]["val_cross_entropy"]) < CE_TOL),
            checkpoint_epoch=L["meta"].get("epoch")))
        print(f"  restore {arm}: saved {saved_correct} correct, "
              f"restored {got_correct}, ce diff "
              f"{restore_checks[-1]['ce_abs_diff']:.2e}")
        B.mark(f"B2_{arm}")
    write_json(os.path.join(out, "restore_checks.json"), restore_checks)

    # ---- C: core extraction
    cores, adapter_checks = [], []
    for arm, (L, r, s) in loaded_arms.items():
        if not B.ok(45):
            B.mark(f"C_{arm}", "skipped_budget")
            continue
        adapter_checks.append(verify_adapter_vs_forward(L, arm))
        cores.extend(extract_cores(L, arm))
        B.mark(f"C_{arm}")
    write_json(os.path.join(out, "cores.json"), cores)
    write_json(os.path.join(out, "adapter_checks.json"), adapter_checks)

    # ---- D: gradients and sensitivity
    idx, xs, ys = fixed_subset(Xtr, Ytr)
    write_json(os.path.join(out, "subset.json"),
               dict(seed=SUBSET_SEED, n=int(len(idx)), indices=idx.tolist(),
                    split="train",
                    split_sha256=dmani["feature_sha256"]["train"],
                    note="post-screen diagnostic sample, NOT an independent "
                         "confirmation sample"))
    grads, sens, fdchecks = [], [], []
    for arm, (L, r, s) in loaded_arms.items():
        if not B.ok(45):
            B.mark(f"D1_{arm}", "skipped_budget")
            continue
        g = grad_report(L, xs, ys, training_mode=False)
        g["arm"] = arm
        grads.append(g)
        B.mark(f"D1_{arm}")
    for arm, (L, r, s) in loaded_arms.items():
        if not B.ok(60):
            B.mark(f"D2_{arm}", "skipped_budget")
            continue
        fdchecks.append(dict(arm=arm, **finite_difference_check(L, xs)))
        sens.append(dict(arm=arm, anchors=sensitivity_probe(L, xs)))
        B.mark(f"D2_{arm}")
    for arm, (L, r, s) in loaded_arms.items():
        if not B.ok(40):
            B.mark(f"D3_{arm}", "skipped_budget")
            continue
        g = grad_report(L, xs, ys, training_mode=True)
        g["arm"] = arm
        g["label"] = ("TRAINING-MODE normalization: BatchNorm couples samples "
                      "and times; not a purely causal recurrence diagnostic")
        grads.append(g)
        B.mark(f"D3_{arm}")
    write_json(os.path.join(out, "gradients.json"), grads)
    write_json(os.path.join(out, "sensitivity.json"), sens)
    write_json(os.path.join(out, "fd_checks.json"), fdchecks)

    # ---- plots (optional; never affects the read-only guarantee)
    plots = dict(attempted=True, written=[], error=None)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # 1. training curves and the ten-epoch cosine schedule
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for sc in screen:
            if sc["lr"] != 1e-3:
                continue
            ep = [c["epoch"] for c in sc["curve"]]
            ax[0].plot(ep, [c["val_accuracy"] for c in sc["curve"]],
                       marker="o", label=sc["arm"])
            ax[1].plot(ep, [c["train_loss_SMOOTHED"] for c in sc["curve"]],
                       marker="o", label=sc["arm"])
        steps = onp.arange(0, 10 * 843)
        cos10 = 1e-6 + (1e-3 - 1e-6) * 0.5 * (1 + onp.cos(onp.pi * steps
                                                          / (10 * 843)))
        cos300 = 1e-6 + (1e-3 - 1e-6) * 0.5 * (1 + onp.cos(onp.pi * steps
                                                           / (300 * 843)))
        ax[2].plot(steps, cos10, label="Stage 2: TEN-EPOCH cosine")
        ax[2].plot(steps, cos300, "--",
                   label="first 10 epochs of a 300-epoch cosine")
        ax[2].set_xlabel("update"); ax[2].set_ylabel("learning rate")
        ax[2].set_title("schedule definitions (analytic, NOT run)")
        ax[0].set_title("validation accuracy"); ax[0].set_xlabel("epoch")
        ax[1].set_title("train loss (label-smoothed)"); ax[1].set_xlabel("epoch")
        for a_ in ax:
            a_.legend(fontsize=7); a_.grid(alpha=.3)
        fig.tight_layout()
        f1 = os.path.join(out, "curves_and_schedule.png")
        fig.savefig(f1, dpi=110); plt.close(fig); plots["written"].append(f1)

        # 2. poles and impulse energy by lag band, layer 0
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        for c in cores:
            if c["layer"] != 0:
                continue
            ax[0].scatter(c["poles"]["eff_pole_re"], c["poles"]["eff_pole_im"],
                          s=14, label=c["arm"], alpha=.7)
            xs_ = range(len(BANDS))
            ax[1].plot(list(xs_), [b["energy"] for b in
                                   c["impulse_bands"]["bands"]],
                       marker="o", label=c["arm"])
        ax[0].set_title("effective poles, layer 0"); ax[0].grid(alpha=.3)
        ax[0].set_xlabel("Re"); ax[0].set_ylabel("Im")
        ax[1].set_yscale("log"); ax[1].set_title("impulse energy by lag band")
        ax[1].set_xticks(list(range(len(BANDS))))
        ax[1].set_xticklabels([f"{lo}-{hi}" for lo, hi in BANDS], fontsize=7)
        ax[1].grid(alpha=.3)
        for a_ in ax:
            a_.legend(fontsize=7)
        fig.tight_layout()
        f2 = os.path.join(out, "poles_and_impulse.png")
        fig.savefig(f2, dpi=110); plt.close(fig); plots["written"].append(f2)

        # 3. current vs history sensitivity, pre-pooling
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
            fig.savefig(f3, dpi=110); plt.close(fig); plots["written"].append(f3)
    except Exception as exc:                       # never fail the diagnostic
        plots["error"] = f"{type(exc).__name__}: {exc}"
        print(f"[!] plots skipped: {plots['error']}")
    write_json(os.path.join(out, "plots.json"), plots)

    # ---- F: hashes AFTER
    hashes_after = hash_tree(args.stage2_dir)
    unchanged = hashes_before == hashes_after
    diff = [k for k in set(hashes_before) | set(hashes_after)
            if hashes_before.get(k) != hashes_after.get(k)]
    summary = dict(run_id=run_id, out=out, backend=backend,
                   wall_s=time.time() - B.t0, budget_s=args.budget_s,
                   phases=B.phases,
                   source_hashes_unchanged=bool(unchanged),
                   source_hash_differences=diff,
                   test_split_opened=False,
                   data_manifest=dict(
                       train=dmani["feature_sha256"]["train"],
                       val=dmani["feature_sha256"]["val"]),
                   n_arms_analysed=len(loaded_arms),
                   plots=plots)
    write_json(os.path.join(out, "summary.json"), summary)
    print(f"[*] source hashes unchanged: {unchanged}")
    if diff:
        print(f"[!] CHANGED: {diff}")
    print(f"[*] wall {summary['wall_s']:.1f}s of {args.budget_s:.0f}s")
    print(f"DIAGNOSTIC_DONE out={out}")
    return summary


if __name__ == "__main__":
    main()
