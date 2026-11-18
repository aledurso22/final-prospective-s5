"""Isolated proximal-residual history S5, preserving the training protocol.

This changes the SSM state recurrence, not the external training optimizer.
The tested history runner supplies batching, float32 initialization, optimizer,
validation selection, telemetry and checkpoint behavior. All registration is
local to this process and is restored after success or failure.
"""
import argparse
from contextlib import contextmanager
from functools import partial
import hashlib
import json
import math
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.linalg import expm

from experiments.s5_gradient_history import runner as H
from experiments.s5_three_arm_full import runner as R
from experiments.s5_three_arm_full.pole_diagnostics import classify, BOUNDARY_TOL
from s5.proximal_history import ProximalGradientHistoryS5SSM, proximal_history_zoh

ARM = "proximal_history_s5"
PROXIMAL_RATIO = 0.5


def validate_config(alpha, proximal_ratio):
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be positive and finite")
    if not math.isfinite(proximal_ratio) or proximal_ratio < 0:
        raise ValueError("proximal_ratio must be finite and nonnegative")


def transitions(params, alpha, proximal_ratio=PROXIMAL_RATIO):
    return jnp.concatenate([proximal_history_zoh(
        k, alpha=alpha, proximal_ratio=proximal_ratio)[0]
        for _, k in H._kappas(params)], axis=0)


def pole_summary(params, alpha, proximal_ratio=PROXIMAL_RATIO):
    host = jax.device_get(params)
    host64 = jax.tree_util.tree_map(lambda x: np.asarray(x, np.float64), host)
    kappas = np.concatenate([k for _, k in H._kappas(host64, np)])
    matrices = np.asarray(jax.device_get(transitions(params, alpha, proximal_ratio)))
    radius32 = np.max(np.abs(np.linalg.eigvals(matrices.astype(np.complex128))), axis=-1)
    radius_solver = np.max(np.abs(np.asarray(jax.device_get(
        jnp.linalg.eigvals(jnp.asarray(matrices))))), axis=-1)
    horizon = alpha / np.abs(kappas)
    radius64, generator_radius = [], []
    effective = []
    for k, ell in zip(kappas, horizon):
        divisor = 1 + proximal_ratio*ell*k
        keff = k/divisor
        effective.append(keff)
        gen = np.array([[0, 1/ell, 0], [-2*keff, -1/ell, 1/ell],
                        [keff, 0, -1/ell]], np.complex128)
        aug = np.zeros((4,4), np.complex128)
        aug[:3,:3] = gen
        aug[:3,3] = np.array([0,2,-1])/divisor
        radius64.append(np.max(np.abs(np.linalg.eigvals(expm(aug)[:3,:3]))))
        generator_radius.append(np.exp(np.max(np.linalg.eigvals(gen).real)))
    radius64 = np.asarray(radius64)
    ok, edge, bad = classify(radius32, radius64)
    finite = np.isfinite(radius32) & np.isfinite(radius64)
    ok &= finite; edge &= finite; bad &= finite
    return {"arm":ARM, "n_modes":len(kappas), "history_alpha":alpha,
        "proximal_ratio":proximal_ratio, "horizon_normalization":"magnitude",
        "scan_implementation":"history_stationary_block",
        "radius_semantics":"actual proximal-residual exact-ZOH 3x3 transition",
        "stable":int(np.sum(finite & (radius64<1))),
        "unstable":int(np.sum(finite & (radius64>=1))),
        "nonfinite_modes":int(np.sum(~finite)), "min_margin":float(1-radius64.max()),
        "horizon_tokens":H._quant(horizon), "radius_reference":H._quant(radius64),
        "effective_kappa_real":H._quant(np.real(effective)),
        "max_expm_vs_generator_radius_difference":float(np.max(np.abs(radius64-generator_radius))),
        "production_precision":{"dtype":"complex64", "boundary_tol":BOUNDARY_TOL,
            "stable":int(ok.sum()), "numerical_boundary":int(edge.sum()),
            "unstable":int(bad.sum()), "nonfinite":int((~finite).sum()),
            "max_radius_production":float(radius32.max()),
            "max_radius_production_eigensolver":float(radius_solver.max()),
            "max_radius_float64":float(radius64.max()),
            "max_abs_radius_difference_production_vs_float64":float(np.max(abs(radius32-radius64))),
            "max_eigensolver_radius_difference":float(np.max(abs(radius_solver-radius32))),
            "min_margin_production":float(1-radius32.max())}}


@contextmanager
def configured(alpha, proximal_ratio=PROXIMAL_RATIO):
    """Scope every patch to this process; old arms and defaults are restored."""
    validate_config(alpha, proximal_ratio)
    old = {key:getattr(H,key) for key in (
        "ARM", "ALPHA", "NormalizedGradientHistoryS5SSM", "production_transitions",
        "history_pole_summary", "protocol_identity", "registered_history")}

    def identity(data_cache, seed):
        result = old["protocol_identity"](data_cache, seed)
        result.update(proximal_ratio=proximal_ratio, horizon_normalization="magnitude",
                      recurrence="proximal-residual gradient history", outer_optimizer="unchanged")
        return result

    @contextmanager
    def registered():
        with old["registered_history"]():
            R.SCIENTIFIC_NAMES[ARM] = (f"proximal-residual gradient-history S5; alpha={alpha}, "
                f"proximal_ratio={proximal_ratio}; exact ZOH")
            R.ARM_CONFIGS[ARM] = {"shared":R.SHARED_CONFIG,
                "recurrence_constructor":"ProximalGradientHistoryS5SSM",
                "extra_parameters":(), "state_dimension":3*R.SSM_SIZE_BASE,
                "history_alpha":alpha, "proximal_ratio":proximal_ratio}
            yield

    H.ARM, H.ALPHA = ARM, alpha
    H.NormalizedGradientHistoryS5SSM = partial(ProximalGradientHistoryS5SSM,
                                              proximal_ratio=proximal_ratio)
    H.production_transitions = lambda params: transitions(params, alpha, proximal_ratio)
    H.history_pole_summary = lambda params: pole_summary(params, alpha, proximal_ratio)
    H.protocol_identity = identity
    H.registered_history = registered
    try:
        yield ARM
    finally:
        for key,value in old.items(): setattr(H,key,value)


def memory_profile(alpha, proximal_ratio, seed, out):
    from experiments.s5_gradient_history import memory_profile as MP
    out = Path(out)
    if out.exists(): raise ValueError(f"refusing existing output: {out}")
    with configured(alpha, proximal_ratio) as arm:
        old_arms = MP.ARMS
        MP.ARMS = tuple(dict.fromkeys(old_arms+(arm,)))
        try:
            result = MP.compile_profile(arm, seed=seed, history_scan="parallel")
        finally:
            MP.ARMS = old_arms
    root = Path(__file__).resolve().parents[2]
    result.update(history_alpha=alpha, proximal_ratio=proximal_ratio)
    for rel in ("s5/proximal_history.py", "experiments/s5_proximal_history/runner.py"):
        result["source_sha256"][rel] = hashlib.sha256((root/rel).read_bytes()).hexdigest()
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+"\n")
    for row in result["updates"]:
        print(row["kind"],row["memory"],flush=True)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--alpha",type=float,required=True)
    p.add_argument("--proximal-ratio",type=float,default=PROXIMAL_RATIO)
    p.add_argument("--seed",type=int,choices=(301,302,303),required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--data-cache")
    p.add_argument("--smoke-report")
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke",action="store_true")
    mode.add_argument("--train",action="store_true")
    mode.add_argument("--profile",action="store_true")
    args=p.parse_args(argv)
    try:validate_config(args.alpha,args.proximal_ratio)
    except ValueError as e:p.error(str(e))
    if args.profile:
        memory_profile(args.alpha,args.proximal_ratio,args.seed,args.out)
        return
    if not args.data_cache:p.error("--data-cache is required for smoke/train")
    forwarded=["--data-cache",args.data_cache,"--out",args.out,"--seed",str(args.seed),
               "--smoke" if args.smoke else "--train"]
    if args.smoke_report:forwarded += ["--smoke-report",args.smoke_report]
    with configured(args.alpha,args.proximal_ratio):
        H.main(forwarded)
    for filename in ("production_check.json","smoke_result.json","task_result.json"):
        path=Path(args.out)/filename
        if path.exists():
            result=json.loads(path.read_text())
            result.update(history_alpha=args.alpha,proximal_ratio=args.proximal_ratio,
                          recurrence="proximal-residual gradient history",outer_optimizer="unchanged")
            path.write_text(json.dumps(result,indent=2)+"\n")


if __name__=="__main__":main()
