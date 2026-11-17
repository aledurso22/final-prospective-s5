"""Does BatchNorm explain a validation collapse? Frozen-weight diagnostic.

Reads saved epoch checkpoints of ONE training task (default: A, seed 302, of
the timescale 2x2 run) and asks, with every weight frozen:

  1. REPRODUCTION. The epoch-`--reproduce-epoch` checkpoint, its saved
     weights and its saved BatchNorm running statistics, evaluated by the
     runner's own `evaluate` and eval model (dropout off, running averages,
     batch 16, file order) must give the validation record that training
     wrote to metrics.jsonl. Exact (bitwise) equality and the absolute
     difference are both reported.

  2. RECALIBRATION FROM TRAINING DATA ONLY. New population statistics for
     every BatchNorm layer are estimated from training examples (never the
     validation split) and then validation is evaluated again with the SAME
     frozen weights, dropout off, running averages. Two estimates:
       * recal_dropout_off: forward passes with dropout disabled -- the
         statistics the evaluation network actually sees;
       * recal_dropout_on:  forward passes with the training dropout (rate
         0.1, the training step's own rng per batch) -- the conditions under
         which the saved running averages were accumulated.
     Each layer's per-batch mean and variance are captured exactly (Flax
     BatchNorm with momentum 0), and pooled as a cumulative average:
         mean = avg_b mean_b,  var = avg_b (var_b + mean_b^2) - mean^2,
     i.e. the population moments, not an exponential moving average.
     Upstream layers are normalized by their batch statistics during these
     passes, as in training.

  Reading the three numbers per checkpoint:
     saved ~ recal_dropout_on << recal_dropout_off  -> dropout/BatchNorm
         variance shift (the stats are right for training, wrong for eval);
     saved << recal_dropout_on ~ recal_dropout_off  -> stale running averages
         (the weights moved faster than the 0.95 EMA tracked them);
     all three ~ equal and low                       -> not BatchNorm.

  3. CONTEXT, from training data only: a fixed training probe evaluated with
     saved and recalibrated statistics and with per-batch statistics
     (dropout off), plus the logged per-epoch training loss/accuracy and the
     pole_diagnostics record of each epoch.

PRESERVATION. Nothing in the task or run directory is opened for writing.
Every checkpoint is hashed before and after, and the metrics.jsonl records
of the examined epochs are re-read at the end and compared; any change
fails the run. Output goes to --out, which must be new (or empty) and may
not lie inside the run directory. Recalibrated statistics are saved there,
never next to the checkpoints.

    JAX_ENABLE_X64=0 python -m experiments.s5_predictor_corrector.bn_collapse_diagnostic \
        --task /Users/durso/s5-runs/s5-timescale-2x2/20261117-195818/generalized_prospective_s5/302 \
        --data-cache /Users/durso/s5-runs/sc10_official_cache \
        --out /Users/durso/s5-runs/bn_diagnostic_A302/<new dir>
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as onp

DROPOUT_TRAINING = 0.1


class DiagnosticError(RuntimeError):
    pass


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_metadata(run_root):
    path = os.path.join(run_root, "run_metadata.txt")
    if not os.path.exists(path):
        raise DiagnosticError(f"missing {path}")
    meta = {}
    with open(path) as fh:
        for line in fh:
            if "=" in line:
                k, v = line.rstrip("\n").split("=", 1)
                meta[k] = v
    return meta


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:   # a line still being written
                pass
    return rows


def validation_records(task):
    """epoch -> the validation record training wrote (first one per epoch)."""
    out = {}
    for row in read_jsonl(os.path.join(task, "metrics.jsonl")):
        out.setdefault(int(row["epoch"]), row["validation"])
    return out


def training_epoch_summary(task, epochs):
    sums = {}
    for row in read_jsonl(os.path.join(task, "step_metrics.jsonl")):
        e = int(row["epoch"])
        if e in epochs:
            s = sums.setdefault(e, {"n": 0, "loss": 0.0, "acc": 0.0, "nonfinite": 0,
                                    "max_radius": 0.0, "last_quarter_loss": []})
            s["n"] += 1
            s["loss"] += float(row["loss"])
            s["acc"] += float(row["accuracy"])
            s["nonfinite"] += int(bool(row.get("nonfinite")))
            s["max_radius"] = max(s["max_radius"], float(row["companion_spectral_radius"]))
            s["last_quarter_loss"].append(float(row["loss"]))
    out = {}
    for e, s in sorted(sums.items()):
        tail = s["last_quarter_loss"][-max(1, s["n"] // 4):]
        out[e] = {"steps": s["n"],
                  "mean_train_loss_dropout_on": s["loss"] / s["n"],
                  "mean_train_accuracy_dropout_on": s["acc"] / s["n"],
                  "mean_train_loss_last_quarter": float(onp.mean(tail)),
                  "nonfinite_steps": s["nonfinite"],
                  "max_companion_spectral_radius": s["max_radius"]}
    return out


def pole_context(task, epochs):
    out = {}
    for row in read_jsonl(os.path.join(task, "pole_diagnostics.jsonl")):
        e = int(row.get("epoch", -1))
        if e in epochs and row.get("at") != "numerical_failure":
            pp = row.get("production_precision", {})
            out[e] = {k: row.get(k) for k in ("stable", "unstable", "min_margin")}
            out[e]["production_precision"] = {
                k: pp.get(k) for k in ("stable", "numerical_boundary", "unstable",
                                       "max_radius_production")}
    return out


def _quant(v):
    v = onp.asarray(v, onp.float64).ravel()
    finite = v[onp.isfinite(v)]
    q = ({k: float(x) for k, x in zip(("min", "p50", "p95", "max"),
                                       onp.percentile(finite, (0, 50, 95, 100)))}
         if finite.size else {})
    q["nonfinite"] = int(v.size - finite.size)
    return q


def compare_stats(ref, other, flatten_dict):
    """Per BatchNorm layer: other/ref variance ratio and |mean shift| in ref std."""
    fr, fo = flatten_dict(ref), flatten_dict(other)
    layers = sorted({k[:-1] for k in fr})
    out = {}
    for layer in layers:
        m_r, v_r = (onp.asarray(fr[layer + (n,)], onp.float64) for n in ("mean", "var"))
        m_o, v_o = (onp.asarray(fo[layer + (n,)], onp.float64) for n in ("mean", "var"))
        with onp.errstate(divide="ignore", invalid="ignore"):
            out["/".join(layer)] = {
                "var_ratio": _quant(v_o / v_r),
                "abs_mean_shift_in_ref_std": _quant(onp.abs(m_o - m_r) / onp.sqrt(v_r + 1e-5))}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--task", required=True,
                    help="<run root>/<arm>/<seed> of the training task")
    ap.add_argument("--epochs", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--reproduce-epoch", type=int, default=2)
    ap.add_argument("--data-cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--recal-examples", type=int, default=0,
                    help="training examples for recalibration; 0 = the whole "
                         "training split (default)")
    ap.add_argument("--probe-examples", type=int, default=2048,
                    help="fixed training-split probe for train-side accuracy")
    ap.add_argument("--reproduce-atol", type=float, default=1e-5,
                    help="tolerance reported (not enforced) when the original "
                         "validation record is not reproduced bitwise")
    args = ap.parse_args(argv)
    t_start = time.perf_counter()

    task = os.path.realpath(args.task)
    arm, seed = os.path.basename(os.path.dirname(task)), int(os.path.basename(task))
    run_root = os.path.dirname(os.path.dirname(task))
    out = os.path.realpath(args.out)
    if out == run_root or out.startswith(run_root + os.sep):
        raise DiagnosticError(f"--out {out} lies inside the run directory {run_root}")
    if os.path.exists(out) and os.listdir(out):
        raise DiagnosticError(f"refusing to write into non-empty {out}")
    if args.reproduce_epoch not in args.epochs:
        raise DiagnosticError("--reproduce-epoch must be one of --epochs")
    epochs = sorted(set(args.epochs))
    ckpts = {e: os.path.join(task, f"checkpoint_epoch_{e:02d}.msgpack") for e in epochs}
    for e, p in ckpts.items():
        if not os.path.exists(p):
            raise DiagnosticError(f"missing checkpoint {p}")
    meta = read_metadata(run_root)
    records = validation_records(task)
    for e in epochs:
        if e not in records:
            raise DiagnosticError(f"metrics.jsonl has no validation record for epoch {e}")
    hashes_before = {e: sha256(p) for e, p in ckpts.items()}
    records_before = {e: records[e] for e in epochs}

    import jax
    import jax.numpy as jnp
    from flax import serialization
    from flax.traverse_util import flatten_dict, unflatten_dict

    if jax.config.read("jax_enable_x64"):
        raise DiagnosticError("JAX_ENABLE_X64 is on; training ran in float32")
    from dataloaders import speech_commands10 as SC
    from experiments.s5_three_arm_full import data as EXPERIMENT_DATA
    from experiments.s5_three_arm_full import runner as R
    from s5.train_helpers import compute_accuracy, cross_entropy_loss, eval_step

    if arm not in R.RUNNABLE_ARMS:
        raise DiagnosticError(f"parent directory {arm!r} is not a runner arm")
    impl = meta.get("scan_implementation")
    if impl:
        R.GENERALIZED_IMPLEMENTATION = impl
    if meta.get("batch_size") not in (None, str(R.BATCH_SIZE)):
        raise DiagnosticError(f"run batch_size {meta.get('batch_size')} != {R.BATCH_SIZE}")
    if meta.get("data_cache") and os.path.realpath(meta["data_cache"]) != os.path.realpath(args.data_cache):
        raise DiagnosticError(f"--data-cache {args.data_cache} is not the run's "
                              f"{meta['data_cache']}")
    epochs_requested = int(meta.get("epochs_requested", R.EPOCHS))

    os.makedirs(os.path.join(out, "recalibrated_stats"), exist_ok=True)
    progress = open(os.path.join(out, "progress.jsonl"), "a")

    def log(record):
        record = {"t": round(time.perf_counter() - t_start, 1), **record}
        progress.write(json.dumps(record) + "\n")
        progress.flush()
        print(json.dumps(record), flush=True)

    cache, manifest = EXPERIMENT_DATA.load_official_raw(args.data_cache, ("train", "val"))
    x_tr, y_tr = cache["train"]
    x_val, y_val = cache["val"]
    n_train = len(y_tr)
    steps_per_epoch = n_train // R.BATCH_SIZE
    B = R.BATCH_SIZE
    ones = jnp.ones((B, R.SEQ_LEN))

    template = R.init_state(arm, seed)
    eval_model = R.model_for(arm, False)
    cls = R.model_cls_for(arm)
    # BatchNorm on batch statistics with momentum 0: the "running" stats it
    # returns are exactly this batch's moments
    moments_off = cls(training=True, dropout=0.0, bn_momentum=0.0)
    moments_on = cls(training=True, dropout=DROPOUT_TRAINING, bn_momentum=0.0)
    batchstat_model = cls(training=True, dropout=0.0, bn_momentum=0.0)
    if R.SHARED_CONFIG.dropout != DROPOUT_TRAINING:
        raise DiagnosticError("training dropout is not 0.1")

    def make_moments(model):
        @jax.jit
        def f(params, neutral, xb, rng):
            _, mv = model.apply({"params": params, "batch_stats": neutral}, xb,
                                ones, rngs={"dropout": rng}, mutable=["batch_stats"])
            return mv["batch_stats"]
        return f

    moments_fn = {"recal_dropout_off": make_moments(moments_off),
                  "recal_dropout_on": make_moments(moments_on)}

    @jax.jit
    def batchstat_eval(params, neutral, xb, yb):
        logits, _ = batchstat_model.apply({"params": params, "batch_stats": neutral}, xb,
                                          ones[:xb.shape[0]], mutable=["batch_stats"])
        return cross_entropy_loss(logits, yb), compute_accuracy(logits, yb), jnp.argmax(logits, -1)

    def eval_with_predictions(state, x, y):
        """`runner.evaluate`, plus the predicted-class histogram; same
        eval_step, batch size, order and host reduction."""
        losses, correct, preds, count = [], [], [], 0
        for start in range(0, len(y), B):
            xb, yb = jnp.asarray(x[start:start + B]), jnp.asarray(y[start:start + B])
            bl, ba, logits = eval_step(xb, yb, jnp.ones((xb.shape[0], R.SEQ_LEN)),
                                       state, eval_model, True)
            losses.append(jnp.sum(bl)); correct.append(jnp.sum(ba))
            preds.append(jnp.argmax(logits, -1)); count += len(yb)
        hl, hc, hp = jax.device_get((losses, correct, preds))
        c = 0
        for v in hc:
            c += int(v)
        hist = onp.bincount(onp.concatenate(hp), minlength=len(SC.WORDS))
        return {"cross_entropy": float(sum(float(v) for v in hl) / count),
                "accuracy": float(c / count), "n": count,
                "predicted_class_histogram": hist.tolist(),
                "majority_prediction_share": float(hist.max() / count)}

    def batchstat_on(params, neutral, x, y):
        losses, correct, preds, count = [], [], [], 0
        for start in range(0, len(y) - len(y) % B, B):
            bl, ba, p = batchstat_eval(params, neutral, jnp.asarray(x[start:start + B]),
                                       jnp.asarray(y[start:start + B]))
            losses.append(jnp.sum(bl)); correct.append(jnp.sum(ba)); preds.append(p)
            count += B
        hl, hc, hp = jax.device_get((losses, correct, preds))
        hist = onp.bincount(onp.concatenate(hp), minlength=len(SC.WORDS))
        return {"cross_entropy": float(sum(float(v) for v in hl) / count),
                "accuracy": float(sum(int(v) for v in hc) / count), "n": count,
                "majority_prediction_share": float(hist.max() / count)}

    def recalibrate(name, params, neutral, epoch):
        """Population moments over the training batches of `epoch` (1-based),
        in that epoch's order; dropout rng = the training step's own key."""
        fn = moments_fn[name]
        batches = list(SC.epoch_batches(n_train, B, seed, epoch - 1, drop_last=True))
        if args.recal_examples:
            batches = batches[:max(1, args.recal_examples // B)]
        s1 = s2 = None
        for i, idx in enumerate(batches):
            step = (epoch - 1) * steps_per_epoch + i
            xb, _ = R.batch_arrays(x_tr, y_tr, idx)     # exactly as training
            bs = jax.device_get(fn(params, neutral, xb,
                                   jax.random.PRNGKey(seed * 1000 + step)))
            f = flatten_dict(bs)
            if s1 is None:
                s1 = {k[:-1]: 0.0 for k in f}
                s2 = {k[:-1]: 0.0 for k in f}
            for layer in s1:
                m = onp.asarray(f[layer + ("mean",)], onp.float64)
                v = onp.asarray(f[layer + ("var",)], onp.float64)
                s1[layer] = s1[layer] + m
                s2[layer] = s2[layer] + v + m * m
        n = len(batches)
        flat = {}
        for layer in s1:
            mean = s1[layer] / n
            var = onp.maximum(s2[layer] / n - mean * mean, 0.0)
            flat[layer + ("mean",)] = mean.astype(onp.float32)
            flat[layer + ("var",)] = var.astype(onp.float32)
        return unflatten_dict(flat), n * B

    # fixed training probe, independent of the epoch: sorted for mmap locality
    probe_idx = onp.sort(onp.random.RandomState(20261117).permutation(n_train)[:args.probe_examples])
    x_probe, y_probe = onp.asarray(x_tr[probe_idx]), onp.asarray(y_tr[probe_idx])

    log({"event": "start", "task": task, "arm": arm, "seed": seed, "epochs": epochs,
         "n_train": n_train, "n_val": len(y_val), "devices": [str(d) for d in jax.devices()],
         "scan_implementation": R.GENERALIZED_IMPLEMENTATION})

    results = {}
    for e in epochs:
        with open(ckpts[e], "rb") as fh:
            raw = fh.read()
        if hashlib.sha256(raw).hexdigest() != hashes_before[e]:
            raise DiagnosticError(f"checkpoint {ckpts[e]} changed while reading")
        state = serialization.from_bytes(template, raw)
        if int(state.step) != e * steps_per_epoch:
            raise DiagnosticError(f"epoch {e} checkpoint has step {int(state.step)}, "
                                  f"expected {e * steps_per_epoch}")
        saved_stats = state.batch_stats
        neutral = jax.tree_util.tree_map(
            lambda v: jnp.zeros_like(v), saved_stats)  # ignored at momentum 0
        flat_saved = flatten_dict(jax.device_get(saved_stats))
        r = {"checkpoint": ckpts[e], "checkpoint_sha256": hashes_before[e],
             "step": int(state.step), "recorded_validation": records[e],
             "saved_stats_nonfinite": int(sum(int((~onp.isfinite(onp.asarray(v))).sum())
                                              for v in flat_saved.values()))}

        # 1. the original evaluation, verbatim
        original = R.evaluate(state, eval_model, x_val, y_val)
        r["validation"] = {"saved_stats_runner_evaluate": original}
        rec = records[e]
        r["reproduction"] = {
            "bitwise_identical": original == {k: rec[k] for k in ("cross_entropy", "accuracy", "n")},
            "abs_diff_cross_entropy": abs(original["cross_entropy"] - rec["cross_entropy"]),
            "abs_diff_accuracy": abs(original["accuracy"] - rec["accuracy"]),
            "within_atol": (abs(original["cross_entropy"] - rec["cross_entropy"]) <= args.reproduce_atol
                            and original["accuracy"] == rec["accuracy"])}
        log({"event": "reproduction", "epoch": e, **r["reproduction"],
             "recorded": rec, "reproduced": original})
        detail = eval_with_predictions(state, x_val, y_val)
        if {k: detail[k] for k in ("cross_entropy", "accuracy", "n")} != original:
            raise DiagnosticError("detailed evaluation disagrees with runner.evaluate")
        r["validation"]["saved_stats"] = detail

        # 2. recalibrated statistics, training data only, weights frozen
        recal = {}
        for name in ("recal_dropout_off", "recal_dropout_on"):
            stats, n_used = recalibrate(name, state.params, neutral, e)
            recal[name] = stats
            with open(os.path.join(out, "recalibrated_stats",
                                   f"epoch_{e:02d}_{name}.msgpack"), "wb") as fh:
                fh.write(serialization.msgpack_serialize(jax.device_get(stats)))
            v = eval_with_predictions(state.replace(batch_stats=stats), x_val, y_val)
            v["recalibration_examples"] = n_used
            r["validation"][name] = v
            log({"event": "validation", "epoch": e, "stats": name,
                 **{k: v[k] for k in ("accuracy", "cross_entropy", "majority_prediction_share")}})
        r["stats_comparison"] = {
            "saved_vs_recal_dropout_off": compare_stats(recal["recal_dropout_off"], saved_stats, flatten_dict),
            "recal_dropout_on_vs_recal_dropout_off": compare_stats(
                recal["recal_dropout_off"], recal["recal_dropout_on"], flatten_dict)}

        # 3. training-side context (training split only)
        r["train_probe"] = {"examples": int(len(y_probe))}
        for name, stats in (("saved_stats", saved_stats), *recal.items()):
            p = eval_with_predictions(state.replace(batch_stats=stats), x_probe, y_probe)
            r["train_probe"][name] = {k: p[k] for k in ("accuracy", "cross_entropy",
                                                         "majority_prediction_share")}
        r["train_probe"]["per_batch_stats_dropout_off"] = batchstat_on(
            state.params, neutral, x_probe, y_probe)
        log({"event": "train_probe", "epoch": e, **{k: v["accuracy"] for k, v in
                                                     r["train_probe"].items() if isinstance(v, dict)}})
        results[e] = r
        del state

    # preservation: nothing the run owns changed
    hashes_after = {e: sha256(p) for e, p in ckpts.items()}
    records_after = validation_records(task)
    preserved = {"checkpoints_unchanged": hashes_after == hashes_before,
                 "validation_records_unchanged": all(records_after.get(e) == records_before[e]
                                                     for e in epochs)}
    report = {
        "purpose": "frozen-weight BatchNorm diagnostic; training data only for statistics",
        "task": task, "arm": arm, "seed": seed, "run_root": run_root,
        "run_metadata": meta, "epochs_requested_by_run": epochs_requested,
        "reproduce_epoch": args.reproduce_epoch,
        "data_cache": os.path.realpath(args.data_cache),
        "data_manifest_counts": manifest.get("counts"),
        "protocol": {"batch_size": B, "eval": "runner.evaluate / eval_step, dropout off, running averages",
                     "recalibration": "population moments from per-batch BatchNorm moments "
                                      "(momentum 0), the checkpoint epoch's own training batches",
                     "dropout_rate_training": DROPOUT_TRAINING,
                     "validation_used_for_statistics": False,
                     "test_split_read": False},
        "environment": {"jax": jax.__version__, "devices": [str(d) for d in jax.devices()],
                        "xla_flags": os.environ.get("XLA_FLAGS"),
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                        "hostname": os.uname().nodename},
        "training_log_context": {"per_epoch": training_epoch_summary(task, set(epochs)),
                                 "poles": pole_context(task, set(epochs))},
        "epochs": {str(e): results[e] for e in epochs},
        "summary": {str(e): {
            "recorded_val_accuracy": results[e]["recorded_validation"]["accuracy"],
            **{f"val_accuracy_{k}": v["accuracy"] for k, v in results[e]["validation"].items()
               if k != "saved_stats_runner_evaluate"},
            **{f"val_majority_share_{k}": v["majority_prediction_share"]
               for k, v in results[e]["validation"].items() if k != "saved_stats_runner_evaluate"},
            **{f"train_probe_accuracy_{k}": v["accuracy"] for k, v in results[e]["train_probe"].items()
               if isinstance(v, dict)}} for e in epochs},
        "preservation": preserved,
        "wall_seconds": round(time.perf_counter() - t_start, 1)}
    path = os.path.join(out, "bn_diagnostic.json")
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)
    progress.close()
    print(json.dumps({"summary": report["summary"], "preservation": preserved,
                      "reproduction": {e: results[e]["reproduction"] for e in epochs}}, indent=2))
    if not all(preserved.values()):
        raise DiagnosticError(f"run artifacts changed during the diagnostic: {preserved}")
    return report


if __name__ == "__main__":
    try:
        main()
    except DiagnosticError as err:
        print(f"FAIL: {err}", file=sys.stderr)
        sys.exit(2)
