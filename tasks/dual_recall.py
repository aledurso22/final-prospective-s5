"""Fast tracking plus delayed cued recall in one 64-step sequence.

Two supervised outputs share ONE temporal representation:

* reconstruct the current scalar signal `u_t` at every step (fast, needs the
  present);
* report which of eight cues was written earlier, when a query marker arrives
  (slow, needs history across a distractor stream).

The point is that neither output can be served by the other's timescale, and
nothing tells the network which modes should carry which. There is no pooling,
no flattening, no cue buffer, no absolute-time feature and no future input: the
readout sees only the temporal state at the current step.

Input channels, 11 in total and identical for every arm:

    [0]     u_t        current signal, unit variance
    [1:9]   content    one-hot: the CUE at the write step, an independent
                       distractor at every other step
    [9]     write      1 at the write step
    [10]    query      1 at the query step

Timing: the cue is written at a step drawn from 4..8 and the query arrives
`delay` steps later, `delay` in {8, 16, 32}, so the query lands at 12..40 and
always inside the 64-step window. Cue class, current signal and distractors are
drawn independently, so the query-step input carries NO information about the
cue class - which the leakage probe measures rather than assumes.
"""

import numpy as onp

SEQ_LEN = 64
N_CLASSES = 8
N_CHANNELS = 11
CUE_STEPS = (4, 5, 6, 7, 8)
DELAYS = (8, 16, 32)
#: correlation time of the current signal, in steps. Declared: short enough
#: that tracking it needs the present, long enough to be learnable at all.
SIGNAL_TAU = 4.0
SIGNAL_RHO = float(onp.exp(-1.0 / SIGNAL_TAU))

IDX_SIGNAL = 0
IDX_CONTENT = slice(1, 1 + N_CLASSES)
IDX_WRITE = 1 + N_CLASSES
IDX_QUERY = 2 + N_CLASSES


def _signal(rng, n, length=SEQ_LEN):
    """AR(1) with unit stationary variance: u_t = rho u_{t-1} + sqrt(1-rho^2) xi."""
    u = onp.zeros((n, length), dtype=onp.float32)
    u[:, 0] = rng.randn(n)
    sd = float(onp.sqrt(1.0 - SIGNAL_RHO ** 2))
    for t in range(1, length):
        u[:, t] = SIGNAL_RHO * u[:, t - 1] + sd * rng.randn(n)
    return u


def generate(rng, n, cue_steps=CUE_STEPS, delays=DELAYS, classes=None,
             delay_choice=None, cue_step_choice=None, length=SEQ_LEN):
    """A batch of sequences. Returns (x, y_signal, y_class, query_index, meta).

    `classes`, `delay_choice` and `cue_step_choice` override the random draws,
    which is how the evaluation set is built with exactly equal counts and how
    the interventions re-generate a matched batch.
    """
    cls = (rng.randint(0, N_CLASSES, n) if classes is None
           else onp.asarray(classes, dtype=int))
    dly = (rng.choice(delays, n) if delay_choice is None
           else onp.asarray(delay_choice, dtype=int))
    t_cue = (rng.choice(cue_steps, n) if cue_step_choice is None
             else onp.asarray(cue_step_choice, dtype=int))
    t_qry = t_cue + dly
    if onp.any(t_qry >= length):
        raise ValueError("a query falls outside the window; check the delays")

    u = _signal(rng, n, length)
    # independent distractors EVERYWHERE, then the cue overwrites its own step
    content = rng.randint(0, N_CLASSES, (n, length))
    x = onp.zeros((n, length, N_CHANNELS), dtype=onp.float32)
    x[:, :, IDX_SIGNAL] = u
    rows = onp.arange(n)
    content[rows, t_cue] = cls
    x[rows[:, None], onp.arange(length)[None, :],
      1 + content] = 1.0
    x[rows, t_cue, IDX_WRITE] = 1.0
    x[rows, t_qry, IDX_QUERY] = 1.0
    meta = dict(cue_step=t_cue, delay=dly, query_index=t_qry,
                content=content, cue_class=cls)
    return x, u, cls.astype(onp.int32), t_qry.astype(onp.int32), meta


def balanced_eval_set(seed=999, reps=8):
    """Exactly equal counts for all 8 classes x 3 delays. Fresh, never trained."""
    rng = onp.random.RandomState(seed)
    cls, dly = [], []
    for c in range(N_CLASSES):
        for d in DELAYS:
            cls += [c] * reps
            dly += [d] * reps
    cls = onp.asarray(cls); dly = onp.asarray(dly)
    order = rng.permutation(cls.size)
    return generate(rng, cls.size, classes=cls[order], delay_choice=dly[order])


def shuffled_cue(batch, seed):
    """Same sequences with a DIFFERENT cue class written at the same step.

    Everything else - signal, distractors, markers, timing - is reproduced from
    the same draw, so a change in the recall output is attributable to the cue.
    """
    x, u, cls, tq, meta = batch
    rng = onp.random.RandomState(seed)
    shift = rng.randint(1, N_CLASSES, cls.size)     # never zero: always changes
    new_cls = (cls + shift) % N_CLASSES
    x2 = x.copy()
    rows = onp.arange(cls.size)
    x2[rows, meta["cue_step"], 1 + cls] = 0.0
    x2[rows, meta["cue_step"], 1 + new_cls] = 1.0
    return x2, u, new_cls.astype(onp.int32), tq, dict(meta, cue_class=new_cls)


def replaced_distractors(batch, seed):
    """Same cue, same signal, same timing; every DISTRACTOR resampled."""
    x, u, cls, tq, meta = batch
    rng = onp.random.RandomState(seed)
    n, length = x.shape[0], x.shape[1]
    new_content = rng.randint(0, N_CLASSES, (n, length))
    rows = onp.arange(n)
    new_content[rows, meta["cue_step"]] = cls        # the cue is preserved
    x2 = x.copy()
    x2[:, :, IDX_CONTENT] = 0.0
    x2[rows[:, None], onp.arange(length)[None, :], 1 + new_content] = 1.0
    return x2, u, cls, tq, dict(meta, content=new_content)


def query_step_inputs(batch):
    """The input vector AT the query step only, for the leakage probe."""
    x, _, cls, tq, _ = batch
    rows = onp.arange(x.shape[0])
    return x[rows, tq], cls


def _ridge_fit(xq, cls, ridge):
    n, d = xq.shape
    Phi = onp.concatenate([xq, onp.ones((n, 1), dtype=xq.dtype)], axis=1)
    Y = onp.eye(N_CLASSES, dtype=xq.dtype)[cls]
    A = Phi.T @ Phi + ridge * onp.eye(d + 1, dtype=xq.dtype)
    return onp.linalg.solve(A, Phi.T @ Y)


def _ridge_score(W, xq, cls):
    n = xq.shape[0]
    Phi = onp.concatenate([xq, onp.ones((n, 1), dtype=xq.dtype)], axis=1)
    return float(onp.mean(onp.argmax(Phi @ W, axis=1) == cls))


def leakage_probe(batch, fit_batch=None, ridge=1e-3, n_permutations=64,
                  seed=4242, quantile=0.99):
    """Can a linear reader recover the cue from the QUERY-STEP input alone?

    R6. The previous version fitted and scored on the same 192 examples and
    compared the result with `chance + 0.06`. In-sample ridge can exploit
    accidental label associations even when the generator is independent, and a
    ridge solution does not maximize accuracy, so it was neither a fair
    estimate nor the "upper bound on linear leakage" it was described as. Both
    claims are withdrawn.

    What is measured now:

    * the classifier is fitted on a SEPARATE generated set and scored on the
      evaluation set, so the number is held out;
    * a PERMUTATION NULL is built by refitting on shuffled labels and scoring
      the same held-out set, which calibrates what this estimator returns when
      there is by construction nothing to find;
    * the declared criterion is that the held-out score does not exceed the
      null's upper quantile.

    This measures what THIS linear reader extracts. It is not a bound over all
    classifiers and not a bound on information.
    """
    rng = onp.random.RandomState(seed)
    if fit_batch is None:
        fit_batch = generate(rng, 4 * batch[0].shape[0])
    xq_fit, cls_fit = query_step_inputs(fit_batch)
    xq_ev, cls_ev = query_step_inputs(batch)
    acc = _ridge_score(_ridge_fit(xq_fit, cls_fit, ridge), xq_ev, cls_ev)
    null = []
    for _ in range(n_permutations):
        perm = rng.permutation(cls_fit.size)
        null.append(_ridge_score(_ridge_fit(xq_fit, cls_fit[perm], ridge),
                                 xq_ev, cls_ev))
    null = onp.asarray(null)
    thresh = float(onp.quantile(null, quantile))
    return dict(held_out_accuracy=acc, chance=1.0 / N_CLASSES,
                n_fit=int(cls_fit.size), n_eval=int(cls_ev.size),
                null_mean=float(null.mean()), null_max=float(null.max()),
                null_quantile=quantile, null_threshold=thresh,
                passed=bool(acc <= thresh),
                note=("held-out score of a ridge reader on the query-step "
                      "input, calibrated against a label-permutation null. "
                      "Not a bound over all classifiers and not a bound on "
                      "information."))


def make_leaking_batch(batch, strength=1.0):
    """A deliberately LEAKING fixture: the cue is written into the query step.

    R6 asks for a positive control, so that a leakage probe which always passes
    is distinguishable from one that works. `leakage_probe` must fail on this.
    """
    x, u, cls, tq, meta = batch
    x2 = x.copy()
    rows = onp.arange(cls.size)
    x2[rows, tq, IDX_CONTENT] = 0.0
    x2[rows, tq, 1 + cls] = strength
    return x2, u, cls, tq, meta


def structure_check(batch):
    """Verify at run time what the task is asserted to be."""
    x, u, cls, tq, meta = batch
    n, length, _ = x.shape
    rows = onp.arange(n)
    content_hot = x[:, :, IDX_CONTENT].sum(axis=2)
    return dict(
        n=int(n), length=int(length),
        one_content_symbol_per_step=bool(onp.all(content_hot == 1.0)),
        exactly_one_write=bool(onp.all(x[:, :, IDX_WRITE].sum(1) == 1.0)),
        exactly_one_query=bool(onp.all(x[:, :, IDX_QUERY].sum(1) == 1.0)),
        cue_present_at_write=bool(onp.all(
            x[rows, meta["cue_step"], 1 + cls] == 1.0)),
        query_after_cue=bool(onp.all(tq > meta["cue_step"])),
        query_inside_window=bool(onp.all(tq < length)),
        no_cue_marker_at_query=bool(onp.all(x[rows, tq, IDX_WRITE] == 0.0)),
        signal_variance=float(onp.var(u)),
        delay_counts={int(d): int(onp.sum(meta["delay"] == d))
                      for d in DELAYS},
        class_counts={int(c): int(onp.sum(cls == c)) for c in range(N_CLASSES)})
