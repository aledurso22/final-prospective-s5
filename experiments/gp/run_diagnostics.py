"""Milestone C: linear-core diagnostics for each mechanism.

SCOPE: one layer's linear core. NOT the transfer of the nonlinear network.
Native Lambda is not the effective generalized pole.

    python -m experiments.gp.run_diagnostics [--outdir DIR] [--init-scale S]
"""
import argparse
import json
import os
import sys

import jax
import numpy as onp

jax.config.update("jax_enable_x64", True)
import jax.numpy as np                                            # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))

from s5.checkpointing import provenance                            # noqa: E402
from s5.gp_diagnostics import core_from_module, summarize           # noqa: E402
from s5.gp_ssm import init_gp_ssm                                   # noqa: E402
from tests.test_gp_prospective import ssm_kwargs                    # noqa: E402

MECHANISMS = ("plain", "gp_scalar", "gp_diagonal", "prospective_input",
              "full_state_pc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/gp/diagnostics")
    ap.add_argument("--init-scale", type=float, default=0.3)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    u = jax.random.normal(jax.random.PRNGKey(0), (32, 4))
    out = {"provenance": provenance(), "init_scale": args.init_scale,
           "scope": "single layer linear core; NOT the nonlinear network",
           "mechanisms": {}}

    print(f"{'mechanism':<19}{'max|a_bar|':>11}{'margin':>9}{'eff.efold':>11}"
          f"{'hist.frac':>11}{'imp.supp':>10}{'hankel_top':>12}{'DC':>10}")
    for m in MECHANISMS:
        scale = args.init_scale if m != "plain" else None
        kw = dict(mechanism=m, **ssm_kwargs())
        if m != "plain":
            kw["gp_init_scale"] = args.init_scale
        mod = init_gp_ssm(**kw)(step_rescale=1.0)
        v = jax.tree_util.tree_map(lambda x: x.astype(np.float64),
                                   mod.init(jax.random.PRNGKey(0), u))
        core, a, a_eff = core_from_module(mod, v)
        s = summarize(core, a=a, a_eff=a_eff, n_impulse=512, n_hankel=48)
        finite = [e for e in s["effective_pole_real"] if e < 0]
        efold = float(onp.max([-1.0 / e for e in finite])) if finite else float("inf")
        s["max_effective_efolding_steps"] = efold
        s["init_scale"] = scale
        out["mechanisms"][m] = s
        print(f"{m:<19}{s['max_discrete_pole_abs']:>11.6f}"
              f"{s['margin_discrete']:>9.4f}{efold:>11.2f}"
              f"{s['history_fraction']:>11.4f}{s['impulse_support']:>10d}"
              f"{s['hankel_top']:>12.4e}{s['dc_gain_norm']:>10.4f}")

    path = os.path.join(args.outdir, "diagnostics.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    print(f"\nwrote {path}")
    print("\nReading: 'plain' and 'gp_*' share the same native Lambda; the")
    print("generalized poles differ because a_eff = a/(1 - t a). full_state_pc")
    print("has impulse support 1 and zero Hankel content: its driven history")
    print("cancels exactly, which is the predicted negative control.")


if __name__ == "__main__":
    main()
