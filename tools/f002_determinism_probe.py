"""F-002: locate the EARLIEST point at which two identical runs diverge.

Runs the real S5 training pipeline twice and compares, in order:

    1. data order / first batches
    2. parameter initialization
    3. first forward pass (logits)
    4. first loss
    5. first gradients
    6. first optimizer update (params and optimizer state)
    7. several subsequent steps

and reports the first stage that differs, with its magnitude. It does NOT
change any architecture; it only observes.

By default both repeats run inside ONE process, which isolates GPU/XLA
nondeterminism from process-level variation (fresh PRNG, fresh allocator,
different autotuning). Use --write to dump a JSON digest instead, and run the
script twice, to test ACROSS processes as well - that is the condition the
G2b runs actually met.

    python tools/f002_determinism_probe.py
    python tools/f002_determinism_probe.py --deterministic
    python tools/f002_determinism_probe.py --write runA.json
"""

import argparse
import json
import os
import sys

# XLA flags must be set before JAX initializes its backend.
_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--deterministic", action="store_true")
_known, _ = _ap.parse_known_args()
if _known.deterministic:
    existing = os.environ.get("XLA_FLAGS", "")
    os.environ["XLA_FLAGS"] = (existing + " --xla_gpu_deterministic_ops=true").strip()

import jax                                                    # noqa: E402
import jax.numpy as jnp                                       # noqa: E402
import numpy as np                                            # noqa: E402
import optax                                                  # noqa: E402
from functools import partial                                 # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from s5.dataloading import Datasets                           # noqa: E402
from s5.seq_model import BatchClassificationModel             # noqa: E402
from s5.ssm import init_S5SSM                                 # noqa: E402
from s5.ssm_init import make_DPLR_HiPPO                       # noqa: E402
from s5.train_helpers import (create_train_state, prep_batch, # noqa: E402
                              train_step)
from jax.scipy.linalg import block_diag                       # noqa: E402

SEED = 1919
BSZ = 64
N_LAYERS, D_MODEL, SSM_SIZE, BLOCKS = 2, 64, 64, 2
N_EXTRA_STEPS = 5


def build_model_cls(n_classes, seq_len):
    ssm_size, block_size = SSM_SIZE, SSM_SIZE // BLOCKS
    Lambda, _, _, V, _ = make_DPLR_HiPPO(block_size)
    block_size //= 2
    ssm_size //= 2
    Lambda, V = Lambda[:block_size], V[:, :block_size]
    Vc = V.conj().T
    Lambda = (Lambda * jnp.ones((BLOCKS, block_size))).ravel()
    V = block_diag(*([V] * BLOCKS))
    Vinv = block_diag(*([Vc] * BLOCKS))
    ssm_init_fn = init_S5SSM(
        H=D_MODEL, P=ssm_size, Lambda_re_init=Lambda.real,
        Lambda_im_init=Lambda.imag, V=V, Vinv=Vinv,
        C_init="trunc_standard_normal", discretization="zoh",
        dt_min=0.001, dt_max=0.1, conj_sym=True, clip_eigs=False,
        bidirectional=False)
    return partial(
        BatchClassificationModel, ssm=ssm_init_fn, d_output=n_classes,
        d_model=D_MODEL, n_layers=N_LAYERS, padded=False,
        activation="half_glu1", dropout=0.0, mode="pool", prenorm=True,
        batchnorm=False)


def one_run(dir_name):
    """Execute the staged pipeline once and return per-stage digests."""
    create_fn = Datasets["mnist-classification"]
    trainloader, _, _, _, n_classes, seq_len, in_dim, _ = create_fn(
        dir_name, seed=SEED, bsz=BSZ)

    stages = {}

    # --- 1. data order / first batches
    batches = []
    for i, batch in enumerate(trainloader):
        inputs, labels, times = prep_batch(batch, seq_len, in_dim)
        batches.append((np.asarray(inputs), np.asarray(labels),
                        np.asarray(times)))
        if i == 2:
            break
    stages["1_data_order"] = np.concatenate(
        [b[1].ravel() for b in batches]).astype(np.int64)
    stages["1_data_values"] = np.concatenate(
        [b[0].ravel()[:512] for b in batches]).astype(np.float64)

    # --- 2. parameter initialization
    model_cls = build_model_cls(n_classes, seq_len)
    state = create_train_state(model_cls, jax.random.PRNGKey(SEED),
                               padded=False, retrieval=False, in_dim=in_dim,
                               bsz=BSZ, seq_len=seq_len, batchnorm=False)
    flat0 = np.concatenate(
        [np.asarray(p).ravel() for p in jax.tree_util.tree_leaves(state.params)])
    stages["2_init_params"] = flat0.astype(np.float64)

    inputs, labels, times = batches[0]
    model = model_cls(training=True)

    # --- 3. first forward pass
    logits = model.apply({"params": state.params}, jnp.asarray(inputs),
                         jnp.asarray(times))
    stages["3_forward_logits"] = np.asarray(logits).astype(np.float64).ravel()

    # --- 4. first loss
    def loss_fn(params):
        lg = model.apply({"params": params}, jnp.asarray(inputs),
                         jnp.asarray(times))
        idx = jnp.asarray(labels).astype(jnp.int32)
        return -jnp.mean(lg[jnp.arange(lg.shape[0]), idx])
    loss0, grads = jax.value_and_grad(loss_fn)(state.params)
    stages["4_first_loss"] = np.array([float(loss0)], dtype=np.float64)

    # --- 5. first gradients
    stages["5_first_grads"] = np.concatenate(
        [np.asarray(g).ravel() for g in jax.tree_util.tree_leaves(grads)]
    ).astype(np.float64)

    # --- 6. first optimizer update (through the repo's own train_step)
    new_state, step_loss = train_step(state, jax.random.PRNGKey(0),
                                      jnp.asarray(inputs), jnp.asarray(labels),
                                      jnp.asarray(times), model, False)
    stages["6_after_update_params"] = np.concatenate(
        [np.asarray(p).ravel()
         for p in jax.tree_util.tree_leaves(new_state.params)]).astype(np.float64)
    stages["6_after_update_optstate"] = np.concatenate(
        [np.asarray(o).ravel() for o in jax.tree_util.tree_leaves(new_state.opt_state)
         if np.asarray(o).dtype.kind == "f"]).astype(np.float64)

    # --- 7. several subsequent steps
    losses = [float(step_loss)]
    st = new_state
    for i in range(1, min(N_EXTRA_STEPS, len(batches))):
        inp, lab, tim = batches[i]
        st, l = train_step(st, jax.random.PRNGKey(i), jnp.asarray(inp),
                           jnp.asarray(lab), jnp.asarray(tim), model, False)
        losses.append(float(l))
    stages["7_step_losses"] = np.array(losses, dtype=np.float64)
    return stages


ORDER = ["1_data_order", "1_data_values", "2_init_params", "3_forward_logits",
         "4_first_loss", "5_first_grads", "6_after_update_params",
         "6_after_update_optstate", "7_step_losses"]


def compare(a, b):
    print(f"{'stage':<28}{'identical':>11}{'max|diff|':>14}{'n':>10}")
    first = None
    rows = {}
    for key in ORDER:
        if key not in a or key not in b:
            continue
        u, v = np.asarray(a[key], float), np.asarray(b[key], float)
        n = min(u.size, v.size)
        d = float(np.max(np.abs(u[:n] - v[:n]))) if n else 0.0
        same = d == 0.0
        rows[key] = dict(identical=bool(same), max_abs_diff=d, n=int(n))
        print(f"{key:<28}{str(same):>11}{d:>14.3e}{n:>10d}")
        if not same and first is None:
            first = key
    print()
    if first is None:
        print("RESULT: fully identical through every stage -> reproducible.")
    else:
        print(f"RESULT: first divergence at stage '{first}' "
              f"(max|diff| = {rows[first]['max_abs_diff']:.3e})")
    return first, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deterministic", action="store_true",
                    help="set XLA_FLAGS=--xla_gpu_deterministic_ops=true")
    ap.add_argument("--dir_name", default="./cache_dir")
    ap.add_argument("--write", default=None,
                    help="write this run's digest to JSON instead of "
                         "comparing in-process (run twice, then --compare)")
    ap.add_argument("--compare", nargs=2, default=None,
                    help="compare two JSON digests written with --write")
    args = ap.parse_args()

    print(f"backend={jax.default_backend()} devices={jax.devices()}")
    print(f"XLA_FLAGS={os.environ.get('XLA_FLAGS','<unset>')}")
    print()

    if args.compare:
        with open(args.compare[0]) as f: a = json.load(f)
        with open(args.compare[1]) as f: b = json.load(f)
        compare({k: np.array(v) for k, v in a.items()},
                {k: np.array(v) for k, v in b.items()})
        return

    if args.write:
        stages = one_run(args.dir_name)
        with open(args.write, "w") as f:
            json.dump({k: np.asarray(v).tolist() for k, v in stages.items()}, f)
        print(f"wrote {args.write}")
        return

    print("=== repeat A ===")
    a = one_run(args.dir_name)
    print("=== repeat B (same process) ===")
    b = one_run(args.dir_name)
    print()
    compare(a, b)


if __name__ == "__main__":
    main()
