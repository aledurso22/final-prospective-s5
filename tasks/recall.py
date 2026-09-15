"""Recall the latest marked symbol through distractors. Causal, query-endpoint.

Every token carries a one-hot symbol, a cue marker and a query marker. The
final token (index 127) is the query and has NO symbol payload, so a memoryless
causal model receives an identical final input regardless of the answer and its
output cannot depend on the target. That is the point of the professor control.

Generator, fixed for this study:

* 8 symbol classes, sequence length 128, query at index 127.
* Exactly TWO earlier tokens are marked cues; every other earlier token is an
  unmarked distractor drawn from the same alphabet.
* The target is the symbol of the LATEST marked cue, regardless of the
  distractors in between.
* Query-to-latest-cue delay is 8, 32 or 64 in training, equally represented.
  The gap between the two cues is drawn independently from 8, 16, 24. Delay 96
  is HELD OUT for extrapolation and never trained or selected on.
* Sequences are generated in PAIRS with identical distractors and identical
  marker positions, differing only by swapping the two distinct marked symbols.
  The pair therefore has the same token multiset and different targets, so a
  bag-of-symbols statistic cannot separate them.

Channels: [0:8] symbol one-hot, [8] cue marker, [9] query marker. 10 total.
"""

import numpy as onp

N_SYMBOLS = 8
SEQ_LEN = 128
QUERY_INDEX = SEQ_LEN - 1
N_CHANNELS = N_SYMBOLS + 2
CUE_CHANNEL = N_SYMBOLS
QUERY_CHANNEL = N_SYMBOLS + 1
TRAIN_DELAYS = (8, 32, 64)
HELDOUT_DELAY = 96
CUE_GAPS = (8, 16, 24)


def _encode(symbols, cue_positions, n=SEQ_LEN):
    x = onp.zeros((n, N_CHANNELS), dtype=onp.float32)
    x[onp.arange(n - 1), symbols[:n - 1]] = 1.0
    x[list(cue_positions), CUE_CHANNEL] = 1.0
    x[QUERY_INDEX, QUERY_CHANNEL] = 1.0          # query carries no symbol
    return x


def generate(rng, n_pairs, delays=TRAIN_DELAYS, gaps=CUE_GAPS):
    """`2 * n_pairs` sequences: each pair shares everything but swapped cues.

    Returns (x, y) with x float32 (2*n_pairs, SEQ_LEN, N_CHANNELS) and y int32.
    Target classes and pair order are balanced by construction.
    """
    xs, ys = [], []
    for i in range(n_pairs):
        delay = delays[i % len(delays)]
        gap = gaps[(i // len(delays)) % len(gaps)]
        late = QUERY_INDEX - delay
        early = late - gap
        assert 0 <= early < late < QUERY_INDEX, (early, late)
        symbols = rng.randint(0, N_SYMBOLS, size=SEQ_LEN)
        s_late = rng.randint(0, N_SYMBOLS)
        s_early = (s_late + 1 + rng.randint(0, N_SYMBOLS - 1)) % N_SYMBOLS
        assert s_late != s_early
        cues = (early, late)
        for order, (a, b) in enumerate(((s_early, s_late), (s_late, s_early))):
            sym = symbols.copy()
            sym[early], sym[late] = a, b
            xs.append(_encode(sym, cues))
            ys.append(b)                          # LATEST cue is the target
    x = onp.stack(xs)
    y = onp.asarray(ys, dtype=onp.int32)
    perm = rng.permutation(x.shape[0])
    return x[perm], y[perm]


def generate_fixed_delay(rng, n, delay, gaps=CUE_GAPS):
    """Evaluation set at ONE delay. `n` is rounded up to an even number."""
    n_pairs = (n + 1) // 2
    return generate(rng, n_pairs, delays=(delay,), gaps=gaps)


def probe_interventions(rng, n_pairs, delay=32, gap=16):
    """Paired interventions on a fixed batch, for causal attribution.

    Returns base, cue_changed and distractor_changed, all identical except for
    the single intended edit, plus the base and post-edit targets.
    """
    base_x, base_y, cue_x, cue_y, dis_x = [], [], [], [], []
    late = QUERY_INDEX - delay
    early = late - gap
    for _ in range(n_pairs):
        symbols = rng.randint(0, N_SYMBOLS, size=SEQ_LEN)
        s_late = rng.randint(0, N_SYMBOLS)
        s_early = (s_late + 1 + rng.randint(0, N_SYMBOLS - 1)) % N_SYMBOLS
        sym = symbols.copy()
        sym[early], sym[late] = s_early, s_late
        cues = (early, late)
        base_x.append(_encode(sym, cues)); base_y.append(s_late)
        # intervention 1: change the RELEVANT marked cue
        new_late = (s_late + 1 + rng.randint(0, N_SYMBOLS - 1)) % N_SYMBOLS
        sym_c = sym.copy(); sym_c[late] = new_late
        cue_x.append(_encode(sym_c, cues)); cue_y.append(new_late)
        # intervention 2: change one UNMARKED distractor; target unchanged
        pos = rng.randint(0, QUERY_INDEX)
        while pos in cues:
            pos = rng.randint(0, QUERY_INDEX)
        sym_d = sym.copy()
        sym_d[pos] = (sym[pos] + 1 + rng.randint(0, N_SYMBOLS - 1)) % N_SYMBOLS
        dis_x.append(_encode(sym_d, cues))
    return (onp.stack(base_x), onp.asarray(base_y, dtype=onp.int32),
            onp.stack(cue_x), onp.asarray(cue_y, dtype=onp.int32),
            onp.stack(dis_x))


def check_pairing(x, y):
    """Structural facts the task must satisfy, for the focused checks."""
    n = x.shape[0]
    out = dict(n=int(n), seq_len=int(x.shape[1]), channels=int(x.shape[2]))
    out["query_marker_only_at_end"] = bool(
        onp.all(x[:, QUERY_INDEX, QUERY_CHANNEL] == 1.0)
        and onp.all(x[:, :QUERY_INDEX, QUERY_CHANNEL] == 0.0))
    out["query_has_no_symbol"] = bool(
        onp.all(x[:, QUERY_INDEX, :N_SYMBOLS] == 0.0))
    out["exactly_two_cues"] = bool(
        onp.all(x[:, :, CUE_CHANNEL].sum(axis=1) == 2))
    counts = onp.bincount(y, minlength=N_SYMBOLS)
    out["label_counts"] = counts.tolist()
    out["label_balance_max_dev"] = float(
        onp.max(onp.abs(counts - counts.mean())) / max(counts.mean(), 1e-9))
    return out
