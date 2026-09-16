"""Part A: fast tracking and delayed recall, four temporal laws, full BPTT.

One process, one deadline. This part tests whether a temporal FORWARD model is
useful under a common accurate optimizer. It is not a credit-assignment result:
every arm is trained with exact BPTT through its own discrete updates, so a
better score here says something about the forward law and nothing about any
local learning rule.
"""

import argparse
import json
import os
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as onp
import optax

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

import tasks.dual_recall as TK                                     # noqa: E402
from s5 import tss_models as TM                                    # noqa: E402
from s5.checkpointing import provenance                            # noqa: E402
from s5.tss_cells import (DT, IDEAL_ITERS, T_HORIZON,             # noqa: E402
                          assert_numeric_pytree, equivalent_adaptation,
                          refinement_error, retained_vector_field,
                          rollout_ode, symmetric_reference,
                          tss_gamma_equivalent, tss_vector_field)

SEEDS = (100, 101, 102)
UPDATES = 300
BATCH = 16
#: One learning rate, declared before execution and identical for every arm.
#: No development budget was spent on any arm, so the budget is equal at zero;
#: choosing a rate for the new model alone is exactly what this avoids.
LR = 3e-3
GRAD_CLIP = 1.0

#: R8, predeclared CORRECTNESS tolerances. These gate execution validity, not
#: the scientific verdict: a scientific loss is a result, a NaN or a failed
#: reference is not valid completed evidence.
MAX_FIXED_POINT_RESIDUAL = 1e-3      # on the TRAINED parameters
MAX_REFINEMENT_ERROR = 1e-4          # RK4 factor-4 refinement, relative
MAX_GRAD_ITER_DRIFT = 1e-3           # gradient at 40 vs 80 fixed-point iters

#: R9: ONE optimizer transform, built once and reused by every arm and seed.
#: `tx` is a static jit argument, so a fresh `optax.chain(...)` per arm/seed
#: would have a new object identity and force a retrace each time - which is
#: exactly what a "one compilation per arm" projection assumes does not happen.
_TX = None


def get_tx():
    global _TX
    if _TX is None:
        _TX = optax.chain(optax.clip_by_global_norm(GRAD_CLIP), optax.adam(LR))
    return _TX


def all_finite(tree):
    return bool(jax.tree_util.tree_all(jax.tree_util.tree_map(
        lambda v: bool(onp.all(onp.isfinite(onp.asarray(v)))), tree)))


def loss_terms(arm, p, xs, y_sig, y_cls, q_idx, cfg):
    """Mean current-signal MSE plus mean recall cross entropy at query steps.

    The two terms are averaged SEPARATELY and then summed, so the 63 non-query
    steps cannot drown out the single recall decision.
    """
    pred_sig, logits, diag = TM.batched_forward(arm, p, xs, cfg)
    mse = jnp.mean((pred_sig - y_sig) ** 2)
    rows = jnp.arange(xs.shape[0])
    q_logits = logits[rows, q_idx]
    ce = jnp.mean(optax.softmax_cross_entropy(
        q_logits, jax.nn.one_hot(y_cls, TK.N_CLASSES)))
    acc = jnp.mean(jnp.argmax(q_logits, -1) == y_cls)
    return mse + ce, dict(mse=mse, ce=ce, acc=acc,
                          residual=diag["fixed_point_residual"])


@partial(jax.jit, static_argnums=(0, 1))
def train_step(arm, tx, p, opt, xs, y_sig, y_cls, q_idx, cfg):
    (loss, aux), g = jax.value_and_grad(loss_terms, argnums=1, has_aux=True)(
        arm, p, xs, y_sig, y_cls, q_idx, cfg)
    upd, opt = tx.update(g, opt, p)
    p = optax.apply_updates(p, upd)
    p = TM.project_params(arm, p)
    return p, opt, loss, aux, optax.global_norm(g)


@partial(jax.jit, static_argnums=(0,))
def eval_all(arm, p, xs, y_sig, y_cls, q_idx, cfg):
    pred_sig, logits, diag = TM.batched_forward(arm, p, xs, cfg)
    rows = jnp.arange(xs.shape[0])
    q_logits = logits[rows, q_idx]
    return dict(mse=jnp.mean((pred_sig - y_sig) ** 2),
                ce=jnp.mean(optax.softmax_cross_entropy(
                    q_logits, jax.nn.one_hot(y_cls, TK.N_CLASSES))),
                correct=(jnp.argmax(q_logits, -1) == y_cls),
                logits=q_logits,
                residual=diag["fixed_point_residual"])


def evaluate(arm, p, batch, cfg):
    xs, y_sig, y_cls, q_idx, meta = batch
    out = eval_all(arm, p, jnp.asarray(xs), jnp.asarray(y_sig),
                   jnp.asarray(y_cls), jnp.asarray(q_idx), cfg)
    correct = onp.asarray(out["correct"])
    by_delay = {int(d): float(onp.mean(correct[meta["delay"] == d]))
                for d in TK.DELAYS}
    return dict(signal_mse=float(out["mse"]), recall_ce=float(out["ce"]),
                recall_accuracy=float(onp.mean(correct)),
                recall_by_delay=by_delay,
                combined=float(out["mse"]) + float(out["ce"]),
                fixed_point_residual=float(out["residual"]),
                logits=onp.asarray(out["logits"]))


def interventions(arm, p, base, cfg, seed):
    """Cue shuffle and distractor replacement, on matched regenerated batches."""
    shuffled = TK.shuffled_cue(base, seed)
    replaced = TK.replaced_distractors(base, seed + 1)
    b = evaluate(arm, p, base, cfg)
    s = evaluate(arm, p, shuffled, cfg)
    r = evaluate(arm, p, replaced, cfg)
    return dict(
        base_recall=b["recall_accuracy"],
        follows_shuffled_cue=s["recall_accuracy"],
        recall_after_distractor_replacement=r["recall_accuracy"],
        cue_logit_l2=float(onp.sqrt(onp.mean(
            onp.sum((s["logits"] - b["logits"]) ** 2, axis=-1)))),
        distractor_logit_l2=float(onp.sqrt(onp.mean(
            onp.sum((r["logits"] - b["logits"]) ** 2, axis=-1)))),
        note=("follows_shuffled_cue scores the NEW cue class: a model that "
              "actually reads the cue should stay high, one that memorized "
              "the distractor stream should not"))


def trained_correctness(arm, p, cfg, es, seed):
    """R8: verify the integrator and the fixed-point solve on the TRAINED
    parameters actually used for the scores, not only at initialization.

    Three things are checked and all three are gated:

    * the fixed-point residual at the learned weights, for the arms that solve
      one (a converged residual at seed 100's initialization says nothing about
      the learned nonlinear map);
    * the RK4 step-refinement error at the learned vector field;
    * the drift of the GRADIENT when the fixed-point iteration count is
      doubled. A small residual does not establish that derivatives through 40
      unrolled iterations have converged, which is a different question.
    """
    out = dict(arm=arm, seed=seed)
    if arm in ("ideal_prospective", "tss_memory_then_prospective"):
        _, diag = TM.temporal(arm, p, es, cfg)
        out["fixed_point_residual_trained"] = float(
            diag["fixed_point_residual"])
        out["fixed_point_ok"] = bool(out["fixed_point_residual_trained"]
                                     < MAX_FIXED_POINT_RESIDUAL)

        def head(pp, iters):
            s_out, _ = TM.temporal(arm, pp, es[:16], cfg, iters=iters)
            return jnp.sum(s_out ** 2)

        g40 = jax.grad(head)(p, IDEAL_ITERS)
        g80 = jax.grad(head)(p, 2 * IDEAL_ITERS)
        num = float(optax.global_norm(
            jax.tree_util.tree_map(lambda a, b: a - b, g40, g80)))
        den = float(optax.global_norm(g40)) + 1e-12
        out["gradient_iteration_drift"] = num / den
        out["gradient_iteration_ok"] = bool(num / den < MAX_GRAD_ITER_DRIFT)
    else:
        out["fixed_point_ok"] = True
        out["gradient_iteration_ok"] = True

    if arm in ("tss_finite_adaptation", "retained_compartment"):
        n = TM.UNITS[arm]
        if arm == "tss_finite_adaptation":
            c = cfg["tss"]
            vf = tss_vector_field(p["cell"], c["tau_m"], c["eps"], c["tau_p"])
        else:
            c = cfg["retained"]
            vf = retained_vector_field(p["cell"], c["gamma"], c["T"], c["M"])
        err = float(refinement_error(vf, jnp.zeros((2, n)), es, TM.N_SUB))
        out["refinement_error_trained"] = err
        out["refinement_ok"] = bool(err < MAX_REFINEMENT_ERROR)
    else:
        out["refinement_ok"] = True
        out["refinement_error_trained"] = None
        if arm == "tss_memory_then_prospective":
            out["refinement_note"] = ("the complex linear memory uses EXACT "
                                      "zero-order hold; there is no "
                                      "integration step to refine")
    out["passed"] = bool(out["fixed_point_ok"] and out["refinement_ok"]
                         and out["gradient_iteration_ok"])
    return out


def save_tree(path, tree):
    from flax import serialization
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(serialization.to_bytes(
            jax.tree_util.tree_map(lambda v: onp.asarray(v), tree)))
    return path


def jsonable(o):
    if isinstance(o, onp.integer):
        return int(o)
    if isinstance(o, onp.floating):
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


def write(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(jsonable(obj), fh, indent=2)


def _cache_size(fn):
    try:
        return int(fn._cache_size())
    except Exception:
        return None


def _time(fn, *a, n=3):
    """Compile, WARM UP AGAIN, then time `n` calls. Returns (compile, steady).

    The second warm-up is not redundant. A jitted step that consumes its own
    output retraces when the output's weak-typing differs from the input's, and
    that retrace lands inside the timing loop and is silently amortized over
    it - which is exactly how a 3 ms step read as 2.5 s.
    """
    t0 = time.time()
    r = fn(*a)
    jax.block_until_ready(r)
    compile_s = time.time() - t0
    jax.block_until_ready(fn(*a))          # absorb any second trace
    before = _cache_size(fn)
    t1 = time.time()
    for _ in range(n):
        r = fn(*a)
    jax.block_until_ready(r)
    steady = (time.time() - t1) / n
    after = _cache_size(fn)
    if before is not None and after is not None and after != before:
        print(f"[!] RETRACE during timing: jit cache {before} -> {after}; "
              f"the reported steady cost is contaminated")
    return compile_s, steady


def component_timing(arm, p, cfg, es, args=None, tx=None, opt=None):
    """Bisect the training step, stage by stage.

    The first breakdown showed the temporal layer was fast while the step was
    not, which ruled out the recurrence but did not locate the cost. This
    version times EVERY stage the step performs - encode, temporal forward and
    gradient, readout, the full loss gradient, the optimizer update and the
    projection - so the remainder is not somewhere unmeasured.
    """
    out = {}
    # `es` arrives BATCHED and time-first, exactly as production supplies it,
    # so these numbers are comparable with the full step rather than with a
    # single sequence. Measuring one sequence is what previously made the
    # temporal layer look fast while the step took three seconds.
    fwd = jax.jit(lambda pp: TM.temporal(arm, pp, es, cfg)[0])
    c, t = _time(fwd, p)
    out["temporal_forward_ms"] = 1e3 * t
    gsum = jax.jit(jax.grad(lambda pp: jnp.sum(
        TM.temporal(arm, pp, es, cfg)[0] ** 2)))
    c2, t2 = _time(gsum, p)
    out["temporal_grad_ms"] = 1e3 * t2
    if arm == "tss_memory_then_prospective":
        from s5.tss_cells import complex_memory_rollout
        mem = jax.jit(lambda pp: complex_memory_rollout(pp["mem"], es))
        _, tm = _time(mem, p)
        out["memory_forward_ms"] = 1e3 * tm
        sm = complex_memory_rollout(p["mem"], es)
        pro = jax.jit(lambda pp: TM.processing_fixed_point(pp["pros"], sm)[0])
        _, tp = _time(pro, p)
        out["processing_forward_ms"] = 1e3 * tp

    if args is not None:
        enc = jax.jit(lambda pp: TM.encode(pp, args[0]))
        _, te = _time(enc, p)
        out["encode_ms"] = 1e3 * te
        s_out = TM.temporal(arm, p, es, cfg)[0]
        s_bt = jnp.swapaxes(s_out, 0, 1)
        ro = jax.jit(lambda pp: TM.readout(pp, s_bt)[1])
        _, tr = _time(ro, p)
        out["readout_ms"] = 1e3 * tr
        lg = jax.jit(jax.value_and_grad(
            lambda pp: loss_terms(arm, pp, *args, cfg)[0]))
        _, tl = _time(lg, p)
        out["loss_value_and_grad_ms"] = 1e3 * tl
        proj = jax.jit(lambda pp: TM.project_params(arm, pp))
        _, tj = _time(proj, p)
        out["project_params_ms"] = 1e3 * tj
        if tx is not None and opt is not None:
            g = jax.grad(lambda pp: loss_terms(arm, pp, *args, cfg)[0])(p)

            def upd(pp, gg, oo):
                u, oo = tx.update(gg, oo, pp)
                return optax.apply_updates(pp, u)

            jupd = jax.jit(upd)
            _, tu = _time(jupd, p, g, opt)
            out["optimizer_update_ms"] = 1e3 * tu
    return out


def preflight(cfg, status, profile=False):
    """Measure compile and steady step cost for EVERY arm, then project.

    Compile and steady cost are measured separately and only the steady part is
    scaled by the update count. Preflight state is discarded; the comparative
    runs re-initialize.

    `profile` adds the stage-by-stage breakdown. It is DIAGNOSTIC and costs
    about seven extra jit compilations per arm - roughly 84 s of a 600 s cap,
    which is a sixth of the budget spent measuring rather than running. It is
    therefore off by default and turned on automatically when the projection
    does NOT fit, which is exactly when it is worth having.
    """
    rng = onp.random.RandomState(0)
    xs, y_sig, y_cls, q_idx, _ = TK.generate(rng, BATCH)
    args = (jnp.asarray(xs), jnp.asarray(y_sig), jnp.asarray(y_cls),
            jnp.asarray(q_idx))
    rows, total = [], 0.0
    for arm in TM.ARMS:
        p = TM.init_params(arm, SEEDS[0])
        tx = get_tx()
        opt = tx.init(p)
        t0 = time.time()
        p2, opt2, loss, aux, gn = train_step(arm, tx, p, opt, *args, cfg)
        loss.block_until_ready()
        compile_s = time.time() - t0
        # Feed the output back ONCE before timing: a step that consumes its own
        # output can retrace when the output's weak-typing differs from the
        # input's, and that retrace would otherwise be averaged into the step.
        p2, opt2, loss, aux, gn = train_step(arm, tx, p2, opt2, *args, cfg)
        loss.block_until_ready()
        n_before = _cache_size(train_step)
        t1 = time.time()
        for _ in range(3):
            p2, opt2, loss, aux, gn = train_step(arm, tx, p2, opt2, *args, cfg)
        loss.block_until_ready()
        step_s = (time.time() - t1) / 3.0
        n_after = _cache_size(train_step)
        retraced = (n_before is not None and n_after is not None
                    and n_after != n_before)
        if retraced:
            print(f"[!] {arm}: RETRACE during step timing "
                  f"({n_before} -> {n_after}); projection is unreliable")
        arm_s = compile_s + len(SEEDS) * UPDATES * step_s
        total += arm_s
        rows.append(dict(arm=arm, compile_s=compile_s, step_s=step_s,
                         arm_total_s=arm_s, retraced_during_timing=retraced,
                         jit_cache=[n_before, n_after],
                         params=TM.parameter_count(p),
                         state=TM.temporal_state_count(arm)))
        print(f"[preflight] {arm:24s} compile {compile_s:6.1f}s  "
              f"step {step_s * 1e3:7.2f}ms  arm {arm_s:6.1f}s  "
              f"params {TM.parameter_count(p):5d}  "
              f"states {TM.temporal_state_count(arm)['total']:3d}")
        if profile:
            es = jnp.swapaxes(TM.encode(p, args[0]), 0, 1)   # (L, B, D_ENC)
            comp = component_timing(arm, p, cfg, es, args=args, tx=tx, opt=opt)
            rows[-1]["components_ms"] = comp
            print("            components " + "  ".join(
                f"{k.replace('_ms', '')}={v:.2f}ms" for k, v in comp.items()))
        del p2, opt2
    # Host-side work outside the timed training step: per seed and arm, the
    # 192-sequence evaluation, three intervention evaluations, the checkpoint
    # write, and the trained-parameter correctness checks -- which compile a
    # fresh gradient at TWO fixed-point iteration counts for the two
    # fixed-point arms and a refinement rollout for the two ODE arms, roughly
    # 18 additional compilations across the batch.
    #
    # It is an ALLOWANCE, not a measurement, and deliberately generous:
    # under-allowing it would let the inner deadline stop training mid-batch
    # and record arms as incomplete, which is a worse failure than refusing up
    # front. Raised from 60 s for that reason.
    host_s = 120.0
    total += host_s
    status["preflight"] = dict(
        rows=rows, host_allowance_s=host_s, projected_total_s=total,
        optimizer_transform_reused=True,
        scope=("compile once per arm - the optimizer transform is a single "
               "cached object, so a new closure identity cannot silently "
               "force a retrace - plus seeds x updates, evaluation, "
               "interventions, checkpoints and the post-training checks"))
    print(f"PREFLIGHT_A_PROJECTED_TOTAL_S={total:.1f}")
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="/Users/durso/s5-runs/tss_pilot")
    ap.add_argument("--run_id", default=None)
    ap.add_argument("--updates", type=int, default=UPDATES)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--budget_s", type=float, default=600.0)
    ap.add_argument("--reserve_s", type=float, default=30.0)
    ap.add_argument("--matmul_precision", default="highest")
    ap.add_argument("--preflight_only", action="store_true")
    ap.add_argument("--part_b_run", default=None,
                    help="artifact directory of the COMPLETED Part B, recorded "
                         "so the two halves stay linked without re-running it")
    ap.add_argument("--part_b_commit", default=None)
    ap.add_argument("--profile", action="store_true",
                    help="always measure the stage-by-stage breakdown; it is "
                         "otherwise measured only when the projection fails")
    ap.add_argument("--allow_cpu", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    deadline = args.deadline if args.deadline else t0 + args.budget_s
    left = lambda: deadline - time.time() - args.reserve_s        # noqa: E731

    backend = jax.default_backend()
    if backend != "gpu" and not args.allow_cpu:
        raise SystemExit(f"REFUSING: backend is {backend!r}, not 'gpu'.")
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)

    cfg = assert_numeric_pytree(TM.coefficients(), "Part A cfg")
    meta = TM.coefficient_metadata()
    seeds = [int(s) for s in args.seeds.split(",")]
    run_id = args.run_id or time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(args.out_root, run_id, "part_a")
    os.makedirs(out, exist_ok=True)

    eval_batch = TK.balanced_eval_set()
    struct = TK.structure_check(eval_batch)
    leak = TK.leakage_probe(eval_batch)
    ref = symmetric_reference()
    tss_eq = tss_gamma_equivalent(cfg["tss"]["tau_m"], cfg["tss"]["eps"],
                                  cfg["tss"]["tau_p"])

    status = dict(
        part="A", run_id=run_id, out=out, backend=backend, seeds=seeds,
        arms=list(TM.ARMS), updates=args.updates, batch=BATCH, lr=LR,
        grad_clip=GRAD_CLIP, dt=DT, clock="one input step = one unit of time",
        coefficients=cfg, coefficient_metadata=meta,
        closed_loop=TM.closed_loop_report(cfg), symmetric_reference=ref,
        state_budget=TM.STATE_BUDGET,
        state_counts={a: TM.temporal_state_count(a) for a in TM.ARMS},
        task=dict(structure=struct, leakage_probe=leak,
                  seq_len=TK.SEQ_LEN, delays=list(TK.DELAYS),
                  cue_steps=list(TK.CUE_STEPS), signal_tau=TK.SIGNAL_TAU),
        sector_audit=dict(
            tss_matched_point_maps_to=tss_eq,
            claim=("TSS's matching prescription tau_p = tau_m maps to gamma = "
                   "0 under (MAP), so M <= gamma T fails for any M > 0: arms 2 "
                   "and 4 are two coefficient sectors of ONE family, and the "
                   "circuit inequality excludes the TSS-matched sector. They "
                   "are not equivalent realizations and not different model "
                   "classes."),
            matched_open_loop_denominator=(
                "arms 2 and 4 share the OPEN-LOOP denominator Q, so they "
                "differ only in the prospective zero: f + T f' against "
                "f + 2T f'. R7: this does NOT equalize closed-loop poles once "
                "f = W tanh(s) + Ux + b; see `closed_loop` for the "
                "characteristic polynomials at frozen Jacobian eigenvalues."),
            memory_comparator=(
                "arm 3's memory is a bank of INDEPENDENT COMPLEX LINEAR leaky "
                "units - the small version of TSS Section 3.3's structure - "
                "with an exact zero-order hold and NO direct encoded-input "
                "path into its processing stage. Its stateless processing "
                "stage remains a declared reduction.")),
        correctness_tolerances=dict(
            fixed_point_residual=MAX_FIXED_POINT_RESIDUAL,
            refinement_error=MAX_REFINEMENT_ERROR,
            gradient_iteration_drift=MAX_GRAD_ITER_DRIFT,
            note=("R8: these gate EXECUTION VALIDITY. A scientific loss is a "
                  "result and leaves the status PASS; a NaN or a failed "
                  "reference is not valid completed evidence and sets FAILED.")),
        interpretation=dict(
            what_this_tests="usefulness of the temporal FORWARD model under a "
                            "common accurate optimizer (exact BPTT)",
            what_this_does_not_test="any credit-assignment claim; Part B is "
                                    "the experiment that can support one",
            parameter_vs_state=("the state budget is matched, parameter counts "
                                "are NOT; both are reported and no "
                                "capacity-efficiency claim follows from this "
                                "batch"),
            seeds="three seeds identify a development signal, not a "
                  "superiority theorem"),
        part_b_reference=dict(
            run=args.part_b_run, commit=args.part_b_commit,
            note=("Part B completed separately and is NOT re-run here. It is a "
                  "frozen-parameter probe with fixed inputs, seeds and filters "
                  "that reproduced identically on five dispatches; there is no "
                  "scientific requirement that both parts share one "
                  "invocation.")),
        results=[], incomplete=[])
    print(f"[*] out={out} backend={backend} seeds={seeds}")
    print(f"[*] task structure: {struct}")
    print(f"[*] leakage probe: held-out {leak['held_out_accuracy']:.4f} vs "
          f"permutation null {leak['null_threshold']:.4f} "
          f"(chance {leak['chance']:.4f}) -> "
          f"{'PASS' if leak['passed'] else 'FAIL'}")
    if not leak["passed"]:
        # R6/R8: a leaking task would make every recall number mean something
        # else, so this invalidates the run rather than being noted in passing.
        status["failed"] = (f"leakage probe FAILED: held-out "
                            f"{leak['held_out_accuracy']:.4f} exceeds the "
                            f"permutation null threshold "
                            f"{leak['null_threshold']:.4f}")
        write(os.path.join(out, "status.json"), status)
        print(f"[FAIL] {status['failed']}")
        print(f"TSS_A_STATUS=FAILED out={out}")
        return 4
    print(f"[*] TSS matched point maps to gamma={tss_eq['gamma']:.6g}, "
          f"admissible under M<=gamma*T: {tss_eq['admissible']}")
    write(os.path.join(out, "status.json"), status)

    proj = preflight(cfg, status, profile=args.profile)
    write(os.path.join(out, "status.json"), status)
    if args.preflight_only:
        print(f"TSS_A_STATUS=PREFLIGHT_ONLY out={out}")
        return 0
    if proj > left() and not args.profile:
        # It does not fit: NOW the stage breakdown is worth its cost, because
        # it is the thing that explains the projection. It costs about 60 s, so
        # it runs only if that much is genuinely spare -- a diagnostic must not
        # become the thing that overruns the deadline it is diagnosing.
        if left() > 90:
            print("[*] projection does not fit; measuring the stage breakdown")
            preflight(cfg, status, profile=True)
            write(os.path.join(out, "status.json"), status)
        else:
            print("[*] projection does not fit and there is no room to measure "
                  "the stage breakdown; re-run with --profile to obtain it")
    if proj > left():
        status["incomplete"].append(
            f"projected {proj:.0f}s > remaining {left():.0f}s; Part A NOT "
            f"started. No arm, seed or update count was reduced.")
        write(os.path.join(out, "status.json"), status)
        print(f"[!] {status['incomplete'][-1]}")
        print(f"TSS_A_STATUS=INCOMPLETE out={out}")
        return 3

    rows, checks = [], []
    init_by_arm = {}
    for seed in seeds:
        if left() < 20:
            status["incomplete"].append(f"seed {seed} not started (budget)")
            break
        rng = onp.random.RandomState(seed)
        # ONE ordered stream of training batches, shared by every arm
        stream = [TK.generate(rng, BATCH) for _ in range(args.updates)]
        for arm in TM.ARMS:
            if left() < 10:
                status["incomplete"].append(f"seed {seed} arm {arm}: budget")
                continue
            p = TM.init_params(arm, seed)
            init_by_arm.setdefault(seed, {})[arm] = p
            tx = get_tx()
            opt = tx.init(p)
            curve, t_a, stopped = [], time.time(), None
            for i, (xs, ysig, ycls, q, _m) in enumerate(stream):
                # R9: the INNER loop honours this part's own sub-deadline.
                # Relying on the outer watchdog would let Part A eat the whole
                # budget and leave Part B a partial comparison.
                if i % 25 == 0 and left() < 5:
                    stopped = i
                    break
                p, opt, loss, aux, gn = train_step(
                    arm, tx, p, opt, jnp.asarray(xs), jnp.asarray(ysig),
                    jnp.asarray(ycls), jnp.asarray(q), cfg)
                if i % 25 == 0 or i == args.updates - 1:
                    rec = dict(update=i, loss=float(loss),
                               mse=float(aux["mse"]), ce=float(aux["ce"]),
                               train_acc=float(aux["acc"]),
                               grad_norm=float(gn))
                    if not all(onp.isfinite(v) for v in rec.values()):
                        status["failed"] = (f"seed {seed} arm {arm}: "
                                            f"non-finite training output at "
                                            f"update {i}: {rec}")
                        write(os.path.join(out, "status.json"), status)
                        print(f"[FAIL] {status['failed']}")
                        print(f"TSS_A_STATUS=FAILED out={out}")
                        return 4
                    curve.append(rec)
            wall = time.time() - t_a
            if stopped is not None:
                status["incomplete"].append(
                    f"seed {seed} arm {arm}: stopped at update {stopped} "
                    f"(budget); NOT scored")
                write(os.path.join(out, "status.json"), status)
                continue
            ev = evaluate(arm, p, eval_batch, cfg)
            iv = interventions(arm, p, eval_batch, cfg, seed)
            ev.pop("logits", None)
            # R8: correctness on the TRAINED parameters, and finite metrics
            es_probe = TM.encode(p, jnp.asarray(eval_batch[0][0]))
            chk = trained_correctness(arm, p, cfg, es_probe, seed)
            checks.append(chk)
            if not chk["passed"]:
                status["failed"] = (f"seed {seed} arm {arm}: trained-parameter "
                                    f"correctness check FAILED: {chk}")
                status["checks"] = checks
                write(os.path.join(out, "status.json"), status)
                print(f"[FAIL] {status['failed']}")
                print(f"TSS_A_STATUS=FAILED out={out}")
                return 4
            numeric = {k: v for k, v in ev.items()
                       if isinstance(v, (int, float))}
            if not all(onp.isfinite(v) for v in numeric.values()):
                status["failed"] = (f"seed {seed} arm {arm}: non-finite "
                                    f"evaluation metric: {numeric}")
                write(os.path.join(out, "status.json"), status)
                print(f"TSS_A_STATUS=FAILED out={out}")
                return 4
            save_tree(os.path.join(out, "params",
                                   f"final_seed{seed}_{arm}.msgpack"), p)
            row = dict(seed=seed, arm=arm, wall_s=wall, curve=curve,
                       params=TM.parameter_count(p),
                       state_counts=TM.temporal_state_count(arm),
                       local_jacobian=TM.local_jacobian_report(arm, p),
                       trained_checks=chk,
                       **ev, interventions=iv)
            rows.append(row)
            status["results"] = rows
            write(os.path.join(out, "results.json"), rows)
            write(os.path.join(out, "status.json"), status)
            print(f"[seed {seed}] {arm:24s} mse={ev['signal_mse']:.4f}  "
                  f"recall={ev['recall_accuracy']:.4f}  "
                  f"({'/'.join('%.2f' % ev['recall_by_delay'][d] for d in TK.DELAYS)})"
                  f"  combined={ev['combined']:.4f}  {wall:.0f}s")

    if init_by_arm:
        s0 = sorted(init_by_arm)[0]
        status["shared_tensors_at_seed_%d" % s0] = TM.shared_tensor_report(
            init_by_arm[s0])
    status["checks"] = checks
    status["provenance"] = provenance()

    # paired differences, per seed, against every other arm
    by = {(r["seed"], r["arm"]): r for r in rows}
    pairs = {}
    for a in TM.ARMS:
        for b in TM.ARMS:
            if a >= b:
                continue
            d = [dict(seed=s,
                      recall_pp=100.0 * (by[(s, a)]["recall_accuracy"]
                                         - by[(s, b)]["recall_accuracy"]),
                      signal_mse=(by[(s, a)]["signal_mse"]
                                  - by[(s, b)]["signal_mse"]),
                      combined=(by[(s, a)]["combined"] - by[(s, b)]["combined"]))
                 for s in seeds if (s, a) in by and (s, b) in by]
            if d:
                pairs[f"{a} - {b}"] = dict(
                    per_seed=d,
                    mean_recall_pp=float(onp.mean([x["recall_pp"] for x in d])),
                    mean_signal_mse=float(onp.mean([x["signal_mse"]
                                                    for x in d])),
                    mean_combined=float(onp.mean([x["combined"] for x in d])))
    status["paired_differences"] = pairs
    status["wall_s"] = time.time() - t0
    status["complete"] = (len(rows) == len(seeds) * len(TM.ARMS)
                          and not status["incomplete"])
    write(os.path.join(out, "status.json"), status)

    print("\n  arm                       recall   d8/d16/d32        signal MSE")
    for a in TM.ARMS:
        rs = [r for r in rows if r["arm"] == a]
        if not rs:
            print(f"  {a:24s} NOT COMPLETED")
            continue
        print(f"  {a:24s} {onp.mean([r['recall_accuracy'] for r in rs]):.4f}   "
              + "/".join("%.2f" % onp.mean([r["recall_by_delay"][d] for r in rs])
                         for d in TK.DELAYS)
              + f"      {onp.mean([r['signal_mse'] for r in rs]):.4f}")
    print(f"[*] wall {status['wall_s']:.0f}s  rows {len(rows)}/"
          f"{len(seeds) * len(TM.ARMS)}")
    if status["incomplete"]:
        for m in status["incomplete"]:
            print(f"[INCOMPLETE] {m}")
        print(f"TSS_A_STATUS=INCOMPLETE out={out}")
        return 3
    print(f"TSS_A_STATUS=COMPLETE out={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
