"""Fused per-step training telemetry: the same fields, one device round trip.

The original loop computed, after every optimizer step and EAGERLY:
  * `_all_finite(state)`: one `bool(jnp.all(jnp.isfinite(leaf)))` -- one
    blocking device->host sync -- PER LEAF of the TrainState (params, both
    Adam moments, batch stats, step);
  * the state norm, one eager reduction per leaf;
  * the companion spectral radius, dozens of eager ops per layer;
  * `float(loss)`, `float(accuracy)`, `float(grad_norm)`, `bool(finite)`:
    four more syncs;
and then opened, appended to and closed `step_metrics.jsonl`.

Here the three state quantities are ONE jitted function, all seven scalars
come back in ONE `jax.device_get`, and the JSONL handle stays open and is
flushed every `FLUSH_EVERY` steps, at every epoch end, and before a failure
is raised. The logged fields, their meaning and the every-step cadence are
unchanged; nothing here feeds back into training (`state` is only read).
"""

import json
import math
import os
import time

import jax
import jax.numpy as jnp

from experiments.s5_three_arm_full.pole_diagnostics import radius_array

FLUSH_EVERY = 50


def _array_leaves(tree):
    return [v for v in jax.tree_util.tree_leaves(tree) if hasattr(v, "dtype")]


def make_state_telemetry(arm):
    """jit(state) -> (all finite, state norm, max companion radius)."""
    @jax.jit
    def telemetry(state):
        leaves = _array_leaves(state)
        finite = jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in leaves]))
        norm = jnp.sqrt(sum(jnp.sum(jnp.abs(v) ** 2) for v in leaves))
        return finite, norm, radius_array(state.params, arm)
    return telemetry


def legacy_state_telemetry(state, arm):
    """The ORIGINAL eager computation, kept for verification and timing."""
    finite = bool(all(bool(jnp.all(jnp.isfinite(v))) for v in _array_leaves(state)))
    norm = float(jnp.sqrt(sum(jnp.sum(jnp.abs(v) ** 2) for v in _array_leaves(state))))
    radius = float(radius_array(state.params, arm))
    return finite, norm, radius


def host_scalars(loss, accuracy, grad_norm, gradients_finite, telemetry):
    """ONE device->host transfer for every per-step scalar."""
    (loss, accuracy, grad_norm, gradients_finite,
     (finite, norm, radius)) = jax.device_get(
        (loss, accuracy, grad_norm, gradients_finite, telemetry))
    return dict(loss=float(loss), accuracy=float(accuracy),
                gradient_norm=float(grad_norm),
                gradients_finite=bool(gradients_finite),
                state_finite=bool(finite), state_norm=float(norm),
                companion_spectral_radius=float(radius))


def is_nonfinite(s):
    return not (s["gradients_finite"] and s["state_finite"]
                and math.isfinite(s["loss"]) and math.isfinite(s["accuracy"])
                and math.isfinite(s["gradient_norm"]))


class JsonlWriter:
    """Append-only JSONL with an open handle and periodic flushes."""

    def __init__(self, path, flush_every=FLUSH_EVERY):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._fh = open(path, "a")
        self._flush_every, self._pending = flush_every, 0

    def write(self, record):
        self._fh.write(json.dumps(record) + "\n")
        self._pending += 1
        if self._pending >= self._flush_every:
            self.flush()

    def flush(self):
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._pending = 0

    def close(self):
        if not self._fh.closed:
            self.flush()
            self._fh.close()


def measure_overhead(state, arm, repeats=5):
    """Legacy vs fused state telemetry on THIS state and device: agreement and
    per-call wall time (median of `repeats`, after one warm-up each)."""
    fused = make_state_telemetry(arm)
    jax.block_until_ready(fused(state))
    legacy_state_telemetry(state, arm)
    times = {"legacy": [], "fused": []}
    for _ in range(repeats):
        t0 = time.perf_counter(); old = legacy_state_telemetry(state, arm)
        times["legacy"].append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        new = jax.device_get(jax.block_until_ready(fused(state)))
        times["fused"].append(time.perf_counter() - t0)
    new = (bool(new[0]), float(new[1]), float(new[2]))
    med = {k: sorted(v)[len(v) // 2] for k, v in times.items()}
    rel = lambda a, b: abs(a - b) / max(abs(b), 1e-30)
    agree = (old[0] == new[0] and rel(new[1], old[1]) < 1e-5
             and rel(new[2], old[2]) < 1e-6)
    return {"arm": arm, "legacy_seconds": med["legacy"],
            "fused_seconds": med["fused"],
            "speedup": med["legacy"] / max(med["fused"], 1e-12),
            "legacy_values": old, "fused_values": new,
            "values_agree": bool(agree),
            "n_array_leaves": len(_array_leaves(state)),
            "backend": jax.default_backend()}
