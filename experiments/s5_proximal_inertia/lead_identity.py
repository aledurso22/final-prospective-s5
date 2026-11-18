"""sigma=1 at this commit vs the ORIGINAL proximal_inertial_s5 module, on this backend.

Loads s5/proximal_inertia.py as it was at --legacy-rev (default acb4d85, the
commit that trained the sigma=1 baseline) straight from git, builds one
production-width S5 layer with identical parameters, and compares forward
outputs, all parameter gradients and the (Ad, bd) blocks with this commit's
sigma=1 path. Also reports how far sigma=0 moves Ad (mathematically not at all;
expm scaling sees the input column, so rounding may differ) and bd.

Exact equality is checked and reported; nothing is assumed. Any discrepancy
is printed with its size.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict

from experiments.s5_three_arm_full import runner as R
from s5 import proximal_inertia as NEW

LEGACY_REV = "acb4d850b7f868ddcfd85da498792c6ff5987749"
ROOT = Path(__file__).resolve().parents[2]


def legacy_module(rev=LEGACY_REV):
    src = subprocess.run(["git", "show", f"{rev}:s5/proximal_inertia.py"], cwd=ROOT,
                         check=True, capture_output=True, text=True).stdout
    src = src.replace("from .ssm import S5SSM", "from s5.ssm import S5SSM")
    path = Path(tempfile.mkdtemp()) / "proximal_inertia_legacy.py"
    path.write_text(src)
    spec = importlib.util.spec_from_file_location("proximal_inertia_legacy", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _diff(a, b):
    a, b = np.asarray(a), np.asarray(b)
    scale = float(np.max(np.abs(b))) if b.size else 0.0
    err = float(np.max(np.abs(a - b))) if a.size else 0.0
    return {"bitwise_identical": bool(a.dtype == b.dtype and a.shape == b.shape
                                      and np.array_equal(a.view(np.uint8), b.view(np.uint8))),
            "max_abs_difference": err, "relative_to_max": err / scale if scale else err}


def compare(length=16000, seed=0, rev=LEGACY_REV):
    old = legacy_module(rev)
    kw = {**R.ssm_kwargs("native_matched_s5"), "step_rescale": 1.0}
    x = jnp.asarray(np.random.default_rng(seed).normal(size=(length, kw["H"])), jnp.float32)
    layer_old = old.ProximalInertialS5SSM(**kw)
    layer_new = NEW.ProximalInertialS5SSM(**kw, input_lead=1.0)
    params = layer_old.init(jax.random.PRNGKey(seed), x)

    def run(layer):
        f = jax.jit(lambda p: layer.apply(p, x))
        g = jax.jit(jax.grad(lambda p: jnp.sum(layer.apply(p, x) ** 2)))
        y, grads = f(params), g(params)
        blocks = layer.apply(params, method=lambda m: (m.inertia_Ad, m.inertia_bd))
        return jax.device_get((y, grads, blocks))

    y0, g0, (A0, b0) = run(layer_old)
    y1, g1, (A1, b1) = run(layer_new)
    grads = {"/".join(k): _diff(v, flatten_dict(g0)[k]) for k, v in flatten_dict(g1).items()}
    _, _, (A2, b2) = run(NEW.ProximalInertialS5SSM(**kw, input_lead=0.0))
    out = {"backend": jax.default_backend(), "device": str(jax.devices()[0]),
           "legacy_rev": rev, "sequence_length": length, "x64": jax.config.read("jax_enable_x64"),
           "sigma1_vs_legacy": {"output": _diff(y1, y0), "Ad": _diff(A1, A0), "bd": _diff(b1, b0),
                                "gradients": grads,
                                "all_bitwise_identical": all(d["bitwise_identical"] for d in
                                    [_diff(y1, y0), _diff(A1, A0), _diff(b1, b0), *grads.values()])},
           "sigma0_vs_sigma1": {"Ad": _diff(A2, A1), "bd": _diff(b2, b1)}}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--length", type=int, default=16000)
    ap.add_argument("--legacy-rev", default=LEGACY_REV)
    ap.add_argument("--require-gpu", action="store_true")
    ap.add_argument("--max-relative", type=float, default=0.0,
                    help="fail if any sigma=1 vs legacy relative difference exceeds this (0 = exact)")
    args = ap.parse_args(argv)
    if Path(args.out).exists():
        raise SystemExit(f"FAIL: refusing existing output {args.out}")
    if args.require_gpu and jax.default_backend() != "gpu":
        raise SystemExit("FAIL: GPU backend required")
    rep = compare(args.length, rev=args.legacy_rev)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rep, indent=2) + "\n")
    s1 = rep["sigma1_vs_legacy"]
    worst = max([s1["output"]["relative_to_max"], s1["Ad"]["relative_to_max"], s1["bd"]["relative_to_max"],
                 *(d["relative_to_max"] for d in s1["gradients"].values())])
    print(json.dumps({"backend": rep["backend"], "sigma1_bitwise_identical_to_legacy": s1["all_bitwise_identical"],
                      "sigma1_worst_relative_difference": worst,
                      "sigma0_vs_sigma1_Ad_max_abs": rep["sigma0_vs_sigma1"]["Ad"]["max_abs_difference"]}),
          flush=True)
    if worst > args.max_relative:
        raise SystemExit(f"FAIL: sigma=1 differs from {args.legacy_rev[:7]} (relative {worst:.3g})")
    return rep


if __name__ == "__main__":
    main()
