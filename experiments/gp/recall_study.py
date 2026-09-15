"""Bounded memory-recall comparison. GPU only, one process, one deadline.

Paired continuation: per seed, train the matched ordinary S5 for a shared
warm-up, clone its COMMON parameters into every arm, then continue each arm for
the same number of updates on IDENTICAL ordered minibatches with a FRESH
optimizer. Full BPTT throughout.

Nothing here selects checkpoints or seeds by the comparison, and the held-out
delay is never trained or selected on.
"""

import argparse
import json
import math
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

import tasks.recall as T                                          # noqa: E402
from s5.checkpointing import provenance                           # noqa: E402
from s5.rawat_s5 import RHO_INIT_RECALL, init_substrate_ssm       # noqa: E402
from s5.recall_model import BatchRecallModel, parameter_count     # noqa: E402
from s5.ssm_init import make_DPLR_HiPPO                           # noqa: E402
from jax.scipy.linalg import block_diag                           # noqa: E402

D_MODEL = 32
SSM_SIZE = 32
N_LAYERS = 2
BLOCKS = 4
BATCH = 32
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
EVAL_PER_DELAY = 1024
EVAL_BATCH = 256
SEEDS = (100, 101, 102)

#: response-change gate, declared before execution
INIT_RESPONSE_TOL = 0.01          # 1% relative, signal-only core response
IMPULSE_LAGS = 128
N_FREQ = 65

ARMS = ("ordinary", "rawat", "professor", "gp_rho", "gp_rho_frozen",
        "ordinary_2x")
#: which substrate arm each study arm uses
ARM_SUBSTRATE = dict(ordinary="gain_clip_s5", rawat="alpha_p_s5",
                     professor="prospective_recurrence", gp_rho="gp_rho",
                     gp_rho_frozen="gp_rho_frozen", ordinary_2x="gain_clip_s5")


def ssm_kwargs(ssm_size=SSM_SIZE, d_model=D_MODEL, blocks=BLOCKS):
    blk = ssm_size // blocks
    Lam, _, _, V, _ = make_DPLR_HiPPO(blk)
    blk //= 2
    P = ssm_size // 2
    Lam, V = Lam[:blk], V[:, :blk]
    Vc = V.conj().T
    Lam = (Lam * jnp.ones((blocks, blk))).ravel()
    return dict(H=d_model, P=P, Lambda_re_init=Lam.real,
                Lambda_im_init=Lam.imag, V=block_diag(*([V] * blocks)),
                Vinv=block_diag(*([Vc] * blocks)),
                C_init="trunc_standard_normal", discretization="zoh",
                dt_min=0.001, dt_max=0.1, conj_sym=True, bidirectional=False)


#: Models are built ONCE per arm and reused across seeds. Two separately
#: constructed modules are not equal (their `ssm` field is a functools.partial,
#: compared by identity), so rebuilding per seed would miss the jit cache and
#: recompile every arm for every seed.
_MODEL_CACHE = {}


def build(arm, ssm_size=SSM_SIZE):
    key = (arm, ssm_size)
    if key not in _MODEL_CACHE:
        ssm = init_substrate_ssm(ARM_SUBSTRATE[arm], **ssm_kwargs(ssm_size))
        _MODEL_CACHE[key] = BatchRecallModel(
            ssm=ssm, d_model=D_MODEL, n_layers=N_LAYERS,
            d_output=T.N_SYMBOLS, query_index=T.QUERY_INDEX)
    return _MODEL_CACHE[key]


def build_single(arm, ssm_size=SSM_SIZE):
    """UNBATCHED view of the same module, for `method=` reads.

    Parameters are shared (`variable_axes={"params": None}`), so the same
    parameter tree applies to both. Reads go through this view because
    `nn.vmap` would try to map non-array outputs such as state counts and
    parameter-derived coefficients.
    """
    from s5.recall_model import RecallModel
    ssm = init_substrate_ssm(ARM_SUBSTRATE[arm], **ssm_kwargs(ssm_size))
    return RecallModel(ssm=ssm, d_model=D_MODEL, n_layers=N_LAYERS,
                       d_output=T.N_SYMBOLS, query_index=T.QUERY_INDEX)


def read_state_counts(arm, params, ssm_size=SSM_SIZE):
    m = build_single(arm, ssm_size)
    x = jnp.zeros((T.SEQ_LEN, T.N_CHANNELS))
    ts = jnp.ones((T.SEQ_LEN,))
    return m.apply({"params": params}, x, ts,
                   method=lambda mm, xx, tt: [
                       mm.encoder.layers[i].seq.state_counts()
                       for i in range(N_LAYERS)])


def init_params(arm, seed, ssm_size=SSM_SIZE):
    m = build(arm, ssm_size)
    x = jnp.zeros((2, T.SEQ_LEN, T.N_CHANNELS))
    ts = jnp.ones((2, T.SEQ_LEN))
    return m, m.init(jax.random.PRNGKey(seed), x, ts)["params"]


def make_tx(freeze_rho=False):
    tx = optax.chain(optax.clip_by_global_norm(GRAD_CLIP),
                     optax.adamw(LR, weight_decay=WEIGHT_DECAY))
    if not freeze_rho:
        return tx
    # frozen arm: rho receives no update. Implemented by zeroing its updates,
    # NOT by removing the leaf, so the two arms keep identical structure.
    def label(path, _):
        name = path[-1].key if hasattr(path[-1], "key") else str(path[-1])
        return "frozen" if name == "log_response_rho_only" else "train"
    return optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP),
        optax.multi_transform(
            {"train": optax.adamw(LR, weight_decay=WEIGHT_DECAY),
             "frozen": optax.set_to_zero()},
            lambda p: jax.tree_util.tree_map_with_path(label, p)))


def loss_fn(params, model, x, y):
    logits = model.apply({"params": params}, x, jnp.ones(x.shape[:2]))
    onehot = jax.nn.one_hot(y, T.N_SYMBOLS)
    loss = optax.softmax_cross_entropy(logits, onehot).mean()
    return loss, logits


# Both jitted ONCE at module level with the module and optimizer as STATIC
# arguments, so the compilation is reused across seeds. Defining a jitted
# closure inside a helper would retrace on every call.
@partial(jax.jit, static_argnums=(0, 1))
def train_step(model, tx, params, opt_state, x, y):
    (loss, logits), g = jax.value_and_grad(loss_fn, has_aux=True)(
        params, model, x, y)
    upd, opt_state = tx.update(g, opt_state, params)
    params = optax.apply_updates(params, upd)
    acc = jnp.mean(jnp.argmax(logits, -1) == y)
    return params, opt_state, loss, acc, optax.global_norm(g)


@partial(jax.jit, static_argnums=(0,))
def eval_batch(model, params, xb, yb):
    logits = model.apply({"params": params}, xb, jnp.ones(xb.shape[:2]))
    return (jnp.sum(jnp.argmax(logits, -1) == yb),
            jnp.sum(optax.softmax_cross_entropy(
                logits, jax.nn.one_hot(yb, T.N_SYMBOLS))))


def evaluate(model, params, x, y, batch=EVAL_BATCH):
    correct, ce, n = 0, 0.0, x.shape[0]
    for i in range(0, n, batch):
        c, l = eval_batch(model, params, jnp.asarray(x[i:i + batch]),
                          jnp.asarray(y[i:i + batch]))
        correct += int(c); ce += float(l)
    return dict(accuracy=correct / n, cross_entropy=ce / n, n=int(n))


# ------------------------------------------------------------ cloning ------
def clone_common(src, dst):
    """Copy every parameter the two trees SHARE, leave the rest as initialized.

    Shapes must match where names match; a mismatch is an error, not a silent
    skip, because a silently unshared parameter would break the pairing.
    """
    from flax.traverse_util import flatten_dict, unflatten_dict
    a = flatten_dict(src)
    b = flatten_dict(dst)
    out, copied, kept = {}, [], []
    for k, v in b.items():
        if k in a:
            if a[k].shape != v.shape:
                raise ValueError(f"shape mismatch for {'/'.join(k)}: "
                                 f"{a[k].shape} vs {v.shape}")
            out[k] = a[k]; copied.append("/".join(k))
        else:
            out[k] = v; kept.append("/".join(k))
    return unflatten_dict(out), copied, kept


def embed_double_state(src, dst):
    """Function-preserving embedding of the warm-up model into 2x modes.

    The first P modes are the warm-up's; the added P modes get stable poles and
    NONZERO input coupling but ZERO readout coupling, so the initial function is
    unchanged while the added modes keep a trainable readout path.
    """
    from flax.traverse_util import flatten_dict, unflatten_dict
    a = flatten_dict(src)
    b = flatten_dict(dst)
    out, notes = {}, []
    for k, v in b.items():
        name = k[-1]
        if k not in a:
            out[k] = v
            continue
        w = a[k]
        if w.shape == v.shape:
            out[k] = w
            continue
        if name in ("Lambda_re", "Lambda_im"):
            P = w.shape[0]
            out[k] = v.at[:P].set(w)
            notes.append(f"{'/'.join(k)}: first {P} copied, rest kept")
        elif name == "log_step":
            P = w.shape[0]
            out[k] = v.at[:P].set(w)
            notes.append(f"{'/'.join(k)}: first {P} copied")
        elif name == "B":                       # (P, H, 2) -> (2P, H, 2)
            P = w.shape[0]
            out[k] = v.at[:P].set(w)            # added modes keep NONZERO B
            notes.append(f"{'/'.join(k)}: first {P} copied, added rows nonzero")
        elif name == "C":                       # (H, P, 2) -> (H, 2P, 2)
            P = w.shape[1]
            nv = v.at[:, :P, :].set(w)
            nv = nv.at[:, P:, :].set(0.0)       # added modes: ZERO readout
            out[k] = nv
            notes.append(f"{'/'.join(k)}: first {P} copied, added cols ZEROED")
        else:
            raise ValueError(f"unexpected shape change for {'/'.join(k)}: "
                             f"{w.shape} -> {v.shape}")
    return unflatten_dict(out), notes


# --------------------------------------------- initialization response gate
def core_response_change(arm_a, params_a, arm_b, params_b, layer,
                         ssm_size=SSM_SIZE):
    """Signal-only core impulse and frequency difference, EXCLUDING native D.

    Predeclared norm: relative Frobenius over lags 0..IMPULSE_LAGS-1 of the
    per-layer recurrent CORE impulse with the native D contribution removed,

        rel = ||K_b - K_a||_F / ||K_a||_F

    with the zero-reference case handled explicitly: if ||K_a||_F is below
    1e-12 the relative figure is reported as None and the absolute difference
    is used instead.
    """
    from s5 import substrate_diagnostics as SD

    def read(m, xx, tt):
        seq = m.encoder.layers[layer].seq
        Lam, B_c, Delta = seq._native()
        B_tilde = seq.B[..., 0] + 1j * seq.B[..., 1]
        return dict(response=seq.response, input_gain=seq.input_gain,
                    clip_eigs=seq.clip_eigs, conj_sym=seq.conj_sym,
                    P=seq.P, H=seq.H, Lambda_clipped=Lam,
                    Lambda_raw_param=seq.Lambda_re + 1j * seq.Lambda_im,
                    Lambda_initializer_field=(seq.Lambda_re_init
                                              + 1j * seq.Lambda_im_init),
                    B_tilde=B_tilde, B_c=B_c, Delta=Delta,
                    a=Lam * Delta, b=Delta[:, None] * B_c,
                    C_tilde=seq.C_tilde, D=seq.D,
                    coefficients=seq.coefficients(),
                    physical=dict(T=seq.physical.T, gamma=seq.physical.gamma,
                                  rho=seq.physical.rho,
                                  mass=seq.physical.mass))

    x = jnp.zeros((T.SEQ_LEN, T.N_CHANNELS))
    ts = jnp.ones((T.SEQ_LEN,))
    ca = build_single(arm_a, ssm_size).apply({"params": params_a}, x, ts,
                                             method=read)
    cb = build_single(arm_b, ssm_size).apply({"params": params_b}, x, ts,
                                             method=read)
    Ka = SD.impulse_matrices(ca, IMPULSE_LAGS)
    Kb = SD.impulse_matrices(cb, IMPULSE_LAGS)
    Da = onp.diag(onp.asarray(ca["D"]))
    Db = onp.diag(onp.asarray(cb["D"]))
    Ka = Ka.copy(); Kb = Kb.copy()
    Ka[0] -= Da; Kb[0] -= Db              # SIGNAL ONLY: remove native D
    na = float(onp.sqrt(onp.sum(Ka ** 2)))
    diff = float(onp.sqrt(onp.sum((Kb - Ka) ** 2)))
    wa, Ha = SD.frequency_response(ca, N_FREQ)
    _, Hb = SD.frequency_response(cb, N_FREQ)
    Ha = Ha - Da[None]; Hb = Hb - Db[None]
    nfa = float(onp.sqrt(onp.sum(onp.abs(Ha) ** 2)))
    fdiff = float(onp.sqrt(onp.sum(onp.abs(Hb - Ha) ** 2)))
    return dict(layer=layer, impulse_ref_norm=na, impulse_abs_diff=diff,
                impulse_rel=(diff / na if na > 1e-12 else None),
                freq_ref_norm=nfa, freq_abs_diff=fdiff,
                freq_rel=(fdiff / nfa if nfa > 1e-12 else None),
                lags=IMPULSE_LAGS, n_freq=N_FREQ,
                zero_reference=bool(na <= 1e-12))


def query_logit_change(model_a, params_a, model_b, params_b, x):
    ts = jnp.ones(x.shape[:2])
    ya = onp.asarray(model_a.apply({"params": params_a}, x, ts))
    yb = onp.asarray(model_b.apply({"params": params_b}, x, ts))
    na = float(onp.sqrt(onp.sum(ya ** 2)))
    d = float(onp.sqrt(onp.sum((yb - ya) ** 2)))
    return dict(ref_norm=na, abs_diff=d,
                rel=(d / na if na > 1e-12 else None),
                zero_reference=bool(na <= 1e-12))


# ------------------------------------------------------------------- main --
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


def write(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(jsonable(obj), fh, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default="/Users/durso/s5-runs/recall")
    ap.add_argument("--warmup", type=int, default=1024)
    ap.add_argument("--updates", type=int, default=1024)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--deadline", type=float, default=None)
    ap.add_argument("--budget_s", type=float, default=1200.0)
    ap.add_argument("--reserve_s", type=float, default=40.0)
    ap.add_argument("--matmul_precision", default="highest")
    ap.add_argument("--allow_cpu", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    deadline = args.deadline if args.deadline else t0 + args.budget_s

    def left():
        return deadline - time.time() - args.reserve_s

    backend = jax.default_backend()
    if backend != "gpu" and not args.allow_cpu:
        raise SystemExit(f"REFUSING: backend is {backend!r}, not 'gpu'.")
    jax.config.update("jax_default_matmul_precision", args.matmul_precision)

    run_id = time.strftime("%Y%m%d-%H%M%S")
    out = os.path.join(args.out_root, run_id)
    os.makedirs(out, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",")]
    print(f"[*] out={out} backend={backend} seeds={seeds} "
          f"warmup={args.warmup} updates={args.updates}")

    status = dict(run_id=run_id, out=out, backend=backend,
                  provenance=provenance(), seeds=seeds,
                  warmup=args.warmup, updates=args.updates,
                  arms=list(ARMS), budget_s=args.budget_s,
                  init_response_tol=INIT_RESPONSE_TOL,
                  rho_init=RHO_INIT_RECALL,
                  stages=[], results=[], gate=None, incomplete=[])

    # ---- fixed evaluation and probe data, shared across arms and seeds
    ev_rng = onp.random.RandomState(999)
    eval_sets = {}
    for d in list(T.TRAIN_DELAYS) + [T.HELDOUT_DELAY]:
        eval_sets[d] = T.generate_fixed_delay(ev_rng, EVAL_PER_DELAY, d)
    probe = T.probe_interventions(onp.random.RandomState(1234), 64)
    struct = T.check_pairing(*eval_sets[32])
    status["task_structure"] = struct
    print(f"[*] task check: {struct}")
    write(os.path.join(out, "status.json"), status)

    all_rows = []
    for seed in seeds:
        if left() < 60:
            status["incomplete"].append(f"seed {seed} not started (budget)")
            print(f"[!] budget: seed {seed} NOT STARTED")
            break
        tr_rng = onp.random.RandomState(seed)
        # one fixed ordered stream of minibatches, shared by every arm
        stream = [T.generate(tr_rng, BATCH // 2)
                  for _ in range(args.warmup + args.updates)]

        # ---- warm-up on the matched ordinary arm
        model_o, p0 = init_params("ordinary", seed)
        tx = make_tx()
        opt = tx.init(p0)
        t_w = time.time()
        p = p0
        for i in range(args.warmup):
            x, y = stream[i]
            p, opt, loss, acc, gn = train_step(model_o, tx, p, opt,
                                               jnp.asarray(x), jnp.asarray(y))
        loss.block_until_ready()
        warm_s = time.time() - t_w
        print(f"[seed {seed}] warm-up {args.warmup} updates in {warm_s:.1f}s "
              f"loss={float(loss):.4f} acc={float(acc):.3f}")
        status["stages"].append(dict(seed=seed, stage="warmup",
                                     wall_s=warm_s, loss=float(loss),
                                     acc=float(acc)))
        write(os.path.join(out, "status.json"), status)

        # ---- initialization response gate, before ANY continuation
        if status["gate"] is None:
            gate = dict(seed=seed, arms={})
            xprobe = jnp.asarray(eval_sets[32][0][:32])
            for arm in ("gp_rho", "gp_rho_frozen", "rawat", "professor"):
                m_a, p_a = init_params(arm, seed)
                cloned, copied, kept = clone_common(p, p_a)
                layers = [core_response_change("ordinary", p, arm, cloned, l)
                          for l in range(N_LAYERS)]
                ql = query_logit_change(model_o, p, m_a, cloned, xprobe)
                rels = [l["impulse_rel"] for l in layers
                        if l["impulse_rel"] is not None]
                gate["arms"][arm] = dict(layers=layers, query_logits=ql,
                                         worst_impulse_rel=(max(rels)
                                                            if rels else None),
                                         n_cloned=len(copied),
                                         n_arm_specific=len(kept))
                w = gate["arms"][arm]["worst_impulse_rel"]
                print(f"[gate] {arm:16s} worst signal-only core impulse rel = "
                      f"{('%.4e' % w) if w is not None else 'n/a'}"
                      f"   query-logit rel = "
                      f"{('%.4e' % ql['rel']) if ql['rel'] is not None else 'n/a'}")
            gp_worst = max(
                v for k in ("gp_rho", "gp_rho_frozen")
                for v in [gate["arms"][k]["worst_impulse_rel"]]
                if v is not None)
            gate["gp_worst_impulse_rel"] = gp_worst
            gate["tolerance"] = INIT_RESPONSE_TOL
            gate["passed"] = bool(gp_worst <= INIT_RESPONSE_TOL)
            status["gate"] = gate
            write(os.path.join(out, "status.json"), status)
            if not gate["passed"]:
                print(f"[STOP] generalized arms' signal-only core response "
                      f"changes by {gp_worst:.4e} at initialization, above the "
                      f"predeclared {INIT_RESPONSE_TOL:.0e}. Reporting and "
                      f"stopping this initialization route WITHOUT reading "
                      f"comparative scores, as the protocol requires.")
                status["stopped_by_gate"] = True
                write(os.path.join(out, "status.json"), status)
                print(f"RECALL_STATUS=GATE_STOP out={out}")
                return 3
            print(f"[gate] PASSED: worst {gp_worst:.4e} <= "
                  f"{INIT_RESPONSE_TOL:.0e}")

        # ---- projection before comparative training
        per_update = warm_s / max(args.warmup, 1)
        proj = len(ARMS) * args.updates * per_update * 1.6   # 2-state arms cost
        print(f"[*] projection for this seed's continuations ~{proj:.0f}s, "
              f"remaining {left():.0f}s")
        if proj > left():
            status["incomplete"].append(
                f"seed {seed}: continuations projected {proj:.0f}s > remaining "
                f"{left():.0f}s; NOT started")
            print("[!] projection does not fit; not starting continuations")
            break

        # ---- continuations
        for arm in ARMS:
            if left() < 30:
                status["incomplete"].append(f"seed {seed} arm {arm}: budget")
                print(f"[!] budget: seed {seed} arm {arm} NOT STARTED")
                continue
            ssm_size = SSM_SIZE * 2 if arm == "ordinary_2x" else SSM_SIZE
            m_a, p_init = init_params(arm, seed, ssm_size)
            if arm == "ordinary_2x":
                p_a, notes = embed_double_state(p, p_init)
                extra = dict(embedding_notes=notes)
            else:
                p_a, copied, kept = clone_common(p, p_init)
                extra = dict(n_cloned=len(copied), n_arm_specific=len(kept))
            tx_a = make_tx(freeze_rho=(arm == "gp_rho_frozen"))
            opt_a = tx_a.init(p_a)               # FRESH optimizer for every arm
            rho0 = None
            if arm in ("gp_rho", "gp_rho_frozen"):
                from flax.traverse_util import flatten_dict
                rho0 = {"/".join(k): onp.exp(onp.clip(onp.asarray(v),
                                                      -9.21, -1e-4)).tolist()
                        for k, v in flatten_dict(p_a).items()
                        if k[-1] == "log_response_rho_only"}
            t_a = time.time()
            q = p_a
            for i in range(args.updates):
                x, y = stream[args.warmup + i]
                q, opt_a, loss, acc, gn = train_step(
                    m_a, tx_a, q, opt_a, jnp.asarray(x), jnp.asarray(y))
            loss.block_until_ready()
            wall = time.time() - t_a
            per_delay = {}
            for d, (xe, ye) in eval_sets.items():
                per_delay[str(d)] = evaluate(m_a, q, xe, ye)
            trained = [per_delay[str(d)]["accuracy"] for d in (32, 64)]
            primary = float(sum(trained) / 2)
            # paired interventions on the fixed probe batch
            bx, by, cx, cy, dx = probe
            ts = jnp.ones(bx.shape[:2])
            lb = onp.asarray(m_a.apply({"params": q}, jnp.asarray(bx), ts))
            lc = onp.asarray(m_a.apply({"params": q}, jnp.asarray(cx), ts))
            ld = onp.asarray(m_a.apply({"params": q}, jnp.asarray(dx), ts))
            interv = dict(
                base_accuracy=float(onp.mean(onp.argmax(lb, -1) == by)),
                cue_changed_accuracy=float(onp.mean(onp.argmax(lc, -1) == cy)),
                cue_change_logit_l2=float(onp.sqrt(onp.mean(
                    onp.sum((lc - lb) ** 2, axis=-1)))),
                distractor_change_logit_l2=float(onp.sqrt(onp.mean(
                    onp.sum((ld - lb) ** 2, axis=-1)))),
                distractor_change_pred_flip=float(onp.mean(
                    onp.argmax(ld, -1) != onp.argmax(lb, -1))))
            rho_final = None
            if arm in ("gp_rho", "gp_rho_frozen"):
                from flax.traverse_util import flatten_dict
                rho_final = {"/".join(k): onp.exp(onp.clip(onp.asarray(v),
                                                           -9.21, -1e-4)).tolist()
                             for k, v in flatten_dict(q).items()
                             if k[-1] == "log_response_rho_only"}
            from flax.traverse_util import flatten_dict
            clock = {"/".join(k): onp.exp(onp.asarray(v)).ravel().tolist()
                     for k, v in flatten_dict(q).items() if k[-1] == "log_step"}
            counts = read_state_counts(arm, q, ssm_size)
            row = dict(seed=seed, arm=arm, primary_mean_acc_32_64=primary,
                       per_delay=per_delay, wall_s=wall,
                       params=parameter_count(q), state_counts=counts,
                       final_train_loss=float(loss), final_train_acc=float(acc),
                       interventions=interv, rho_init=rho0, rho_final=rho_final,
                       effective_clock=clock, **extra)
            all_rows.append(row)
            status["results"] = all_rows
            write(os.path.join(out, "results.json"), all_rows)
            write(os.path.join(out, "status.json"), status)
            print(f"[seed {seed}] {arm:16s} primary(32,64)={primary:.4f}  "
                  + "  ".join(f"d{d}={per_delay[str(d)]['accuracy']:.3f}"
                              for d in (8, 32, 64, 96))
                  + f"  {wall:.0f}s")

    status["wall_s"] = time.time() - t0
    status["complete"] = (len(all_rows) == len(seeds) * len(ARMS)
                          and not status["incomplete"])
    write(os.path.join(out, "status.json"), status)
    print(f"[*] wall {status['wall_s']:.0f}s  rows {len(all_rows)}/"
          f"{len(seeds) * len(ARMS)}")
    if status["incomplete"]:
        for m in status["incomplete"]:
            print(f"[INCOMPLETE] {m}")
        print(f"RECALL_STATUS=INCOMPLETE out={out}")
        return 3
    print(f"RECALL_STATUS=COMPLETE out={out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
