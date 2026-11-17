"""The ACTUAL production initialization, drawn in float32, whatever the caller.

`runner.init_state(arm, seed)` draws different random values when
JAX_ENABLE_X64 is on: the float64 path of jax.random produces different
log_step and B (Lambda is deterministic and unaffected). Training ran in
float32, so the matched initialization MUST be drawn with x64 off. This
helper does that in a clean subprocess and returns float numpy arrays, so
float64 analyses can import it safely.
"""

import os
import subprocess
import sys
import tempfile

import numpy as onp

_CHILD = r"""
import sys, numpy as onp
import jax
assert not jax.config.read("jax_enable_x64")
from flax.traverse_util import flatten_dict
from experiments.s5_three_arm_full import runner as R
flat = flatten_dict(R.init_state(sys.argv[1], int(sys.argv[2])).params)
onp.savez(sys.argv[3], **{"/".join(k): onp.asarray(v) for k, v in flat.items()})
"""

_CACHE = {}


def production_init_flat(arm, seed):
    """{tuple path: array} of runner.init_state(arm, seed).params, float32."""
    key = (arm, int(seed))
    if key not in _CACHE:
        root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(__file__))))
        env = dict(os.environ, JAX_ENABLE_X64="0",
                   JAX_PLATFORMS=os.environ.get("JAX_PLATFORMS", "cpu"),
                   PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""))
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "init.npz")
            subprocess.run([sys.executable, "-c", _CHILD, arm, str(seed), out],
                           cwd=root, env=env, check=True,
                           stdout=subprocess.DEVNULL)
            with onp.load(out) as z:
                _CACHE[key] = {tuple(k.split("/")): z[k] for k in z.files}
    return _CACHE[key]
