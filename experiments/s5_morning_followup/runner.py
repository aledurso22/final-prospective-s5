"""Fresh, smoke-gated N/D/P replication without changing old runner defaults."""

import argparse
from contextlib import ExitStack, contextmanager
import json
import math
from pathlib import Path

import jax
import numpy as np
from flax.traverse_util import flatten_dict

from experiments.s5_three_arm_full import runner as R
from experiments.s5_gradient_history import runner as H
from experiments.s5_proximal_history import runner as P

NATIVE = "native_matched_s5"
CORRECTOR = "calibrated_prediction_s5"
ARMS = (NATIVE, CORRECTOR, P.ARM)
SEEDS = (302, 303, 304, 305)
EPOCHS = 15


@contextmanager
def configured(arm, alpha=1.0, proximal_ratio=.5):
    if arm not in ARMS:
        raise ValueError("unsupported replication arm")
    P.validate_config(alpha, proximal_ratio)
    if proximal_ratio != .5 or (arm != P.ARM and alpha != 1.0):
        raise ValueError("replication fixes proximal ratio=.5; alpha applies only to P")
    old_seeds, old_scan = R.SEEDS, R.GENERALIZED_IMPLEMENTATION
    try:
        R.SEEDS = tuple(dict.fromkeys(old_seeds + SEEDS))
        R.GENERALIZED_IMPLEMENTATION = "factored"
        with ExitStack() as stack:
            if arm == P.ARM:
                stack.enter_context(P.configured(alpha, proximal_ratio))
                stack.enter_context(H.registered_history())
            yield
    finally:
        R.SEEDS, R.GENERALIZED_IMPLEMENTATION = old_seeds, old_scan


def protocol_identity(data_cache, seed, arm, alpha=1.0, proximal_ratio=.5):
    """Call inside configured(): bind sources/data, not just hyperparameters."""
    identity = H.protocol_identity(data_cache, seed)
    identity.update(arm=arm, epochs=EPOCHS, seed=seed,
                    scan_implementation=("history_stationary_block" if arm == P.ARM
                                         else "factored" if arm == CORRECTOR else None),
                    replication="morning-paired-seeds", outer_optimizer="unchanged")
    if arm == P.ARM:
        identity.update(alpha=alpha, proximal_ratio=proximal_ratio)
    else:
        identity.pop("alpha", None)
        identity["scan"] = identity["scan_implementation"]
    return identity


def matched_initialization(seed):
    """Float32 N/P parameters and optimizer; N/D shared parameter leaves."""
    if jax.config.read("jax_enable_x64"):
        raise ValueError("matched initialization requires float32")
    with configured(NATIVE):
        native = R.init_state(NATIVE, seed)
    with configured(CORRECTOR):
        corrector = R.init_state(CORRECTOR, seed)
    with configured(P.ARM):
        proximal = R.init_state(P.ARM, seed)
    pn, pd, pp = [flatten_dict(s.params) for s in (native, corrector, proximal)]
    if set(pn) != set(pp) or not set(pn).issubset(pd):
        raise ValueError("replication backbone parameter names do not match")
    for key in pn:
        if not (np.array_equal(np.asarray(pn[key]), np.asarray(pd[key])) and
                np.array_equal(np.asarray(pn[key]), np.asarray(pp[key]))):
            raise ValueError(f"replication backbone differs: {key}")
    tn, tp = [jax.tree_util.tree_flatten(s.opt_state) for s in (native, proximal)]
    if tn[1] != tp[1] or not all(np.array_equal(np.asarray(a), np.asarray(b))
                                for a, b in zip(tn[0], tp[0])):
        raise ValueError("P initial optimizer differs from native")
    return {"seed":seed, "native_proximal_parameters_identical":True,
            "native_proximal_optimizer_identical":True,
            "native_corrector_shared_parameters_identical":True}


def check_smoke_gate(report, production, identity):
    if report.get("protocol_identity") != identity:
        raise ValueError("smoke protocol, source, data or seed differs")
    if report.get("backend") != "gpu":
        raise ValueError("training requires a GPU smoke")
    if (report.get("status") != "SMOKE_PASS" or not report.get("checkpoint_restored")
            or report.get("training_steps") != 2):
        raise ValueError("smoke did not complete two finite updates and restoration")
    for item in (report, production):
        if item.get("code_identifier") != identity["arm"] or item.get("seed") != identity["seed"]:
            raise ValueError("smoke arm or seed mismatch")
        if item.get("scan_implementation") != identity["scan_implementation"]:
            raise ValueError("smoke scan mismatch")
        if item.get("epochs_requested") != EPOCHS:
            raise ValueError("smoke schedule mismatch")
    if production.get("x64_enabled") is not False:
        raise ValueError("smoke did not use production float32")
    if not production.get("gradients_finite") or not production.get("state_finite"):
        raise ValueError("production smoke is nonfinite")
    for key in ("loss", "accuracy", "gradient_norm"):
        value = production.get(key)
        if not isinstance(value, (float, int)) or not math.isfinite(value):
            raise ValueError("production smoke has nonfinite observables")
    for key, low, high in (("validation_accuracy", 0, 1),
                           ("validation_cross_entropy", 0, math.inf)):
        value = report.get(key)
        if not isinstance(value, (float, int)) or not math.isfinite(value) or not low <= value <= high:
            raise ValueError("invalid smoke validation metric")
    if not report.get("telemetry_check", {}).get("values_agree"):
        raise ValueError("smoke telemetry disagrees")
    for poles in (production.get("initial_poles", {}), report.get("poles_after_smoke_steps", {})):
        precision = poles.get("production_precision", {})
        if precision.get("unstable", 1) or precision.get("nonfinite", poles.get("nonfinite_modes", 0)):
            raise ValueError("smoke contains unstable or nonfinite modes")
        for key in ("max_radius_production", "max_radius_float64"):
            value = precision.get(key)
            if not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
                raise ValueError("smoke pole radius is invalid")
    if identity["arm"] == P.ARM:
        H.check_smoke_gate(report, identity)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--proximal-ratio", type=float, default=.5)
    parser.add_argument("--data-cache")
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke-report")
    mode = parser.add_mutually_exclusive_group(required=True)
    for flag in ("smoke", "train", "profile"):
        mode.add_argument(f"--{flag}", action="store_true")
    args = parser.parse_args(argv)
    if jax.config.read("jax_enable_x64"):
        parser.error("production replication requires JAX_ENABLE_X64=0")
    out = Path(args.out)
    if out.exists():
        parser.error("output must be new; existing results are never reused")
    if args.profile:
        if args.arm != P.ARM or args.proximal_ratio != .5:
            parser.error("profile mode is for proximal history with ratio=.5 only")
        P.memory_profile(args.alpha, args.proximal_ratio, args.seed, out)
        return
    with configured(args.arm, args.alpha, args.proximal_ratio):
        if not args.data_cache:
            parser.error("--data-cache is required for smoke/train")
        identity = protocol_identity(args.data_cache, args.seed, args.arm,
                                     args.alpha, args.proximal_ratio)
        if args.train:
            if not args.smoke_report:
                parser.error("--train requires --smoke-report")
            smoke_path = Path(args.smoke_report)
            report = json.loads(smoke_path.read_text())
            production = json.loads((smoke_path.parent / "production_check.json").read_text())
            check_smoke_gate(report, production, identity)
            if jax.default_backend() != "gpu":
                parser.error("training requires the verified GPU backend")
        out.mkdir(parents=True)
        (out / "replication_protocol.json").write_text(json.dumps(identity, indent=2)+"\n")
        args.epochs = EPOCHS
        try:
            R.run_task(args)
            result_path = out / ("smoke_result.json" if args.smoke else "task_result.json")
            result = json.loads(result_path.read_text())
            result.update(protocol_identity=identity, backend=jax.default_backend(),
                          scan_implementation=identity["scan_implementation"],
                          outer_optimizer="unchanged")
            if args.arm == P.ARM:
                result.update(history_alpha=args.alpha, proximal_ratio=args.proximal_ratio)
            result_path.write_text(json.dumps(result, indent=2)+"\n")
            if args.smoke and result["backend"] == "gpu":
                production = json.loads((out / "production_check.json").read_text())
                check_smoke_gate(result, production, identity)
        except Exception as error:
            (out / "failure.json").write_text(json.dumps({"arm":args.arm,"seed":args.seed,
                "error_type":type(error).__name__,"failure":str(error)}, indent=2)+"\n")
            raise


if __name__ == "__main__":
    main()
