"""Temporal-response task: revision and untouched probes at declared delays.

Protocol: docs/PROSPECTIVE_TEMPORAL_RESPONSE_PROTOCOL.md. Same vocabulary,
event types, input contract and latest-write oracle as
`experiments.nested_memory.task`; that module and every completed study are
unchanged.

Layout of one 64-token episode, zero-based:

    0-9     write the ten target associations, random order
    10-12   three writes on distinct NON-target keys, independent values
    13-53   five probe BLOCKS, one per delay d in {1, 2, 4, 8, 16}, in a random
            block order. A block starting at token r is
              r              rewrite one SELECTED target
              r+1 .. r+d-1   d-1 fill tokens (the episode's condition)
              r+d, r+d+1     a probe PAIR: query the revised key and query one
                             UNTOUCHED target, in a balanced order
            so a block occupies d + 2 tokens and the five occupy 41
    54-63   query every target once, random order (late probes)

Five selected and five untouched targets; each block pairs one selected with
one distinct untouched target.

Conditions, one per episode, exactly balanced within every batch:
    idle_gap            every fill token is IDLE (key id 0, no value: the
                        generator's existing idle default)
    intervening_writes  every fill token writes a uniformly drawn NON-target
                        key with an independent value

Families, as before: in `revision` the selected target is rewritten with a
value DIFFERENT from its original; in `recall` the original value is
repeated. The model receives no family, condition, delay, kind or oracle
field: training and evaluation pass only key_id, val_id, event (and the label
to the loss).

Two different times are recorded for every query and never conflated:
    since_revision  tokens since the block's revision write (block probes; for
                    late probes of a selected key, since its own revision)
    age             tokens since the QUERIED key's own last write (for an
                    untouched key, since its initial write)

Balance per batch (n_per_family a multiple of 4): in each family, every
(kind in {revised, untouched}) x delay x condition cell has exactly n/2
queries, and each cell's probe offsets (0 = first in the pair, 1 = second) are
split exactly in half; late probes give 5 selected and 5 untouched per
episode. Counts per episode: 20 queries in both conditions; 18 non-fill writes
in both conditions; the 26 fill tokens are idle or writes BY DESIGN (the
manipulated variable), so total writes are 18 (idle) or 44 (intervening).
"""

import hashlib

import numpy as onp

from experiments.nested_memory.task import (FAMILIES, IDLE, N_KEYS,
                                            N_VALUES, QUERY, WRITE)

SEQ_LEN = 64
N_TARGETS = 10
N_SELECTED = 5
N_FILLER = 3
DELAYS = (1, 2, 4, 8, 16)
CONDITIONS = ("idle_gap", "intervening_writes")
#: per-query kinds
KINDS = ("revised_probe", "untouched_probe", "late_selected", "late_untouched")
BLOCK_START = N_TARGETS + N_FILLER
LATE_START = SEQ_LEN - N_TARGETS
N_QUERIES = 2 * len(DELAYS) + N_TARGETS
N_FILL = sum(d - 1 for d in DELAYS)
assert BLOCK_START + sum(d + 2 for d in DELAYS) == LATE_START
#: every array the generator emits; only MODEL_INPUTS may reach the model
MODEL_INPUTS = ("key_id", "val_id", "event", "label")
TOKEN_FIELDS = ("key_id", "val_id", "event", "label", "kind", "delay",
                "offset", "since_revision", "age")


def generate_one(rng, family, condition, order_flip):
    """One episode. `condition` indexes CONDITIONS; `order_flip` in {0, 1}
    sets the balanced probe order (revised first iff (order_flip + delay
    index) is even)."""
    targets = rng.choice(N_KEYS, size=N_TARGETS, replace=False)
    non_targets = onp.setdiff1d(onp.arange(N_KEYS), targets)
    initial = rng.randint(0, N_VALUES, size=N_TARGETS)

    key_id = onp.zeros(SEQ_LEN, dtype=onp.int32)
    val_id = onp.full(SEQ_LEN, -1, dtype=onp.int32)
    event = onp.full(SEQ_LEN, IDLE, dtype=onp.int32)
    kind = onp.full(SEQ_LEN, -1, dtype=onp.int32)
    delay = onp.full(SEQ_LEN, -1, dtype=onp.int32)
    offset = onp.full(SEQ_LEN, -1, dtype=onp.int32)
    since = onp.full(SEQ_LEN, -1, dtype=onp.int32)

    for i, t in enumerate(rng.permutation(N_TARGETS)):
        key_id[i], val_id[i], event[i] = targets[t], initial[t], WRITE
    for i, k in enumerate(rng.choice(non_targets, size=N_FILLER,
                                     replace=False)):
        key_id[N_TARGETS + i] = k
        val_id[N_TARGETS + i] = rng.randint(0, N_VALUES)
        event[N_TARGETS + i] = WRITE

    sel = rng.choice(N_TARGETS, size=N_SELECTED, replace=False)
    unt = onp.setdiff1d(onp.arange(N_TARGETS), sel)
    unt = unt[rng.permutation(unt.size)]
    block_delays = [DELAYS[j] for j in rng.permutation(len(DELAYS))]
    revised_at = {}
    new_value = onp.full(N_TARGETS, -1, dtype=onp.int32)
    pos = BLOCK_START
    for j, d in enumerate(block_delays):
        s, u = int(sel[j]), int(unt[j])
        if family == "revision":
            nv = (initial[s] + rng.randint(1, N_VALUES)) % N_VALUES
        else:
            nv = initial[s]
        new_value[s] = nv
        r = pos
        key_id[r], val_id[r], event[r] = targets[s], nv, WRITE
        revised_at[s] = r
        for f in range(1, d):
            t = r + f
            if CONDITIONS[condition] == "intervening_writes":
                key_id[t] = non_targets[rng.randint(0, non_targets.size)]
                val_id[t] = rng.randint(0, N_VALUES)
                event[t] = WRITE
        revised_first = (order_flip + DELAYS.index(d)) % 2 == 0
        pair = (s, u) if revised_first else (u, s)
        for off, tk in enumerate(pair):
            t = r + d + off
            key_id[t], event[t] = targets[tk], QUERY
            kind[t] = 0 if tk == s else 1
            delay[t], offset[t], since[t] = d, off, t - r
        pos = r + d + 2
    assert pos == LATE_START

    for i, tk in enumerate(rng.permutation(N_TARGETS)):
        t = LATE_START + i
        key_id[t], event[t] = targets[tk], QUERY
        if int(tk) in revised_at:
            kind[t], since[t] = 2, t - revised_at[int(tk)]
        else:
            kind[t] = 3

    label = onp.full(SEQ_LEN, -1, dtype=onp.int32)
    age = onp.full(SEQ_LEN, -1, dtype=onp.int32)
    latest, when = {}, {}
    for t in range(SEQ_LEN):
        if event[t] == WRITE:
            latest[int(key_id[t])] = int(val_id[t])
            when[int(key_id[t])] = t
        elif event[t] == QUERY:
            label[t] = latest[int(key_id[t])]
            age[t] = t - when[int(key_id[t])]
    return dict(key_id=key_id, val_id=val_id, event=event, label=label,
                kind=kind, delay=delay, offset=offset, since_revision=since,
                age=age, targets=targets, new_value=new_value)


def generate_batch(seed, n_per_family, families=FAMILIES):
    """A batch from one named stream. Each family's episodes alternate the two
    conditions, and the probe order flips every second episode of a
    condition, so every cell is exactly balanced when n_per_family is a
    multiple of 4 (required)."""
    if n_per_family % 4:
        raise ValueError(f"n_per_family={n_per_family} must be a multiple of "
                         "4 for the declared exact balance")
    rng = onp.random.RandomState(seed)
    eps, fam_idx, cond = [], [], []
    for fam in families:
        for i in range(n_per_family):
            c, flip = i % 2, (i // 2) % 2
            eps.append(generate_one(rng, fam, c, flip))
            fam_idx.append(FAMILIES.index(fam))
            cond.append(c)
    out = {k: onp.stack([e[k] for e in eps]) for k in TOKEN_FIELDS}
    out["family"] = onp.asarray(fam_idx, dtype=onp.int32)
    out["condition"] = onp.asarray(cond, dtype=onp.int32)
    return out


def oracle_is_exact(batch):
    """Independently recompute latest-write retrieval and compare."""
    ok = True
    for b in range(batch["event"].shape[0]):
        latest = {}
        for t in range(batch["event"].shape[1]):
            if batch["event"][b, t] == WRITE:
                latest[int(batch["key_id"][b, t])] = int(batch["val_id"][b, t])
            elif batch["event"][b, t] == QUERY:
                ok &= latest[int(batch["key_id"][b, t])] == int(
                    batch["label"][b, t])
    return bool(ok)


def cell_counts(batch):
    """Query counts per family x kind x delay x condition x offset."""
    out = {}
    ev, kd = batch["event"], batch["kind"]
    for fi, fam in enumerate(FAMILIES):
        fm = (batch["family"] == fi)[:, None]
        for ci, cn in enumerate(CONDITIONS):
            cm = fm & (batch["condition"] == ci)[:, None] & (ev == QUERY)
            for k in (0, 1):
                for d in DELAYS:
                    for off in (0, 1):
                        sel = cm & (kd == k) & (batch["delay"] == d) & (
                            batch["offset"] == off)
                        out[f"{fam}/{KINDS[k]}/d{d}/{cn}/offset{off}"] = int(
                            sel.sum())
            for k in (2, 3):
                out[f"{fam}/{KINDS[k]}/{cn}"] = int((cm & (kd == k)).sum())
    return out


def structure_check(batch):
    """What the task is asserted to be, verified on generated data."""
    ev, vid, lab, kd = (batch["event"], batch["val_id"], batch["label"],
                        batch["kind"])
    q, w, idle = ev == QUERY, ev == WRITE, ev == IDLE
    counts = cell_counts(batch)
    block = {k: v for k, v in counts.items() if "/d" in k}
    ages = {}
    for k in range(4):
        m = q & (kd == k)
        ages[KINDS[k]] = dict(
            since_revision=dict(min=int(batch["since_revision"][m].min()),
                                max=int(batch["since_revision"][m].max()))
            if k < 3 else None,
            age=dict(min=int(batch["age"][m].min()),
                     median=float(onp.median(batch["age"][m])),
                     max=int(batch["age"][m].max())))
    writes_by_condition = {
        cn: sorted(set(w[batch["condition"] == ci].sum(1).tolist()))
        for ci, cn in enumerate(CONDITIONS)}
    idle_by_condition = {
        cn: sorted(set(idle[batch["condition"] == ci].sum(1).tolist()))
        for ci, cn in enumerate(CONDITIONS)}
    return dict(
        n=int(ev.shape[0]), length=int(ev.shape[1]),
        queries_per_sequence=sorted(set(q.sum(1).tolist())),
        writes_per_sequence_by_condition=writes_by_condition,
        idle_per_sequence_by_condition=idle_by_condition,
        query_value_field_is_absent=bool(onp.all(vid[q] == -1)),
        idle_value_field_is_absent=bool(onp.all(vid[idle] == -1)),
        every_query_has_a_label=bool(onp.all(lab[q] >= 0)),
        no_label_outside_queries=bool(onp.all(lab[~q] == -1)),
        every_query_has_a_kind=bool(onp.all(kd[q] >= 0)),
        block_cells_balanced=bool(len(set(block.values())) == 1),
        block_cell_size=(sorted(set(block.values()))),
        late_counts={k: v for k, v in counts.items() if "/d" not in k},
        times=ages,
        oracle_exact=oracle_is_exact(batch),
        note=("since_revision and age are distinct: an untouched key's age "
              "counts from its own initial write"))


def episode_digest(batch, n=4):
    h = hashlib.sha256()
    for k in TOKEN_FIELDS:
        h.update(onp.asarray(batch[k][:n]).tobytes())
    h.update(onp.asarray(batch["condition"][:n]).tobytes())
    return h.hexdigest()


# ------------------------------------------------------------- metrics -----
#: delays whose probes count as LATER behaviour in the timing analysis
LATER_DELAYS = (4, 8, 16)
#: the immediate-revision operating coordinate
IMMEDIATE_DELAY = 1


def _cell(correct, ce, mask):
    n = int(mask.sum())
    if n == 0:
        return dict(accuracy=None, cross_entropy=None, n=0)
    return dict(accuracy=float(correct[mask].mean()),
                cross_entropy=float(ce[mask].mean()), n=n)


def _mean(vals):
    vals = list(vals)
    if not vals or any(v is None for v in vals):
        return None
    return float(onp.mean(vals))


def aggregate(correct, ce, batch):
    """Declared weighting (protocol s3). `correct`, `ce`: per-token arrays
    (episodes x tokens), meaningful at queries only.

    Per family:
      cells       accuracy/CE for every (kind, delay, condition) block cell and
                  every (late kind, condition) cell; offsets pooled (they are
                  exactly balanced inside every cell)
      revised_probe, untouched_probe
                  EQUAL-WEIGHT mean over the ten (delay x condition) cells
      late_selected, late_untouched
                  equal-weight mean over the two conditions
      macro       equal-weight mean of those four
      by_delay    per kind and delay, equal-weight mean over conditions
      by_condition per kind and condition (block kinds: mean over delays)

    Study aggregates:
      primary            revision-family macro accuracy
      revision_ce        revision-family macro cross-entropy (same weights)
      retention_revision_untouched
                         mean(revision untouched_probe, revision late_untouched)
      recall_overall     recall-family macro accuracy
      immediate_revision revision-family revised_probe at delay 1
      later              mean of three declared parts: revision-family
                         revised_probe at delays 4, 8, 16; retention; and the
                         recall family's later accuracy = mean(block probes of
                         both kinds at delays 4, 8, 16, late macro)
    """
    ev = batch["event"]
    q = ev == QUERY
    kd, dl = batch["kind"], batch["delay"]
    out = {}
    for fi, fam in enumerate(FAMILIES):
        fm = (batch["family"] == fi)[:, None] & q
        cells, ce_cells = {}, {}
        for ci, cn in enumerate(CONDITIONS):
            cm = fm & (batch["condition"] == ci)[:, None]
            for k in (0, 1):
                for d in DELAYS:
                    cells[(k, d, cn)] = _cell(correct, ce,
                                              cm & (kd == k) & (dl == d))
            for k in (2, 3):
                cells[(k, None, cn)] = _cell(correct, ce, cm & (kd == k))

        def acc(k, d, cn):
            return cells[(k, d, cn)]["accuracy"]

        def cex(k, d, cn):
            return cells[(k, d, cn)]["cross_entropy"]
        parts = {}
        ce_parts = {}
        for k in (0, 1):
            parts[KINDS[k]] = _mean(acc(k, d, cn) for d in DELAYS
                                    for cn in CONDITIONS)
            ce_parts[KINDS[k]] = _mean(cex(k, d, cn) for d in DELAYS
                                       for cn in CONDITIONS)
        for k in (2, 3):
            parts[KINDS[k]] = _mean(acc(k, None, cn) for cn in CONDITIONS)
            ce_parts[KINDS[k]] = _mean(cex(k, None, cn) for cn in CONDITIONS)
        by_delay = {KINDS[k]: {str(d): _mean(acc(k, d, cn)
                                             for cn in CONDITIONS)
                               for d in DELAYS} for k in (0, 1)}
        by_condition = {KINDS[k]: {cn: (_mean(acc(k, d, cn) for d in DELAYS)
                                        if k < 2 else acc(k, None, cn))
                                   for cn in CONDITIONS} for k in range(4)}
        out[fam] = dict(
            cells={f"{KINDS[k]}/{'d%d' % d if d else 'late'}/{cn}": v
                   for (k, d, cn), v in cells.items()},
            parts=parts, ce_parts=ce_parts,
            macro_accuracy=_mean(parts.values()),
            macro_cross_entropy=_mean(ce_parts.values()),
            by_delay=by_delay, by_condition=by_condition,
            later_block_accuracy=_mean(acc(k, d, cn) for k in (0, 1)
                                       for d in LATER_DELAYS
                                       for cn in CONDITIONS))
    rev, rec = out["revision"], out["recall"]
    out["primary"] = rev["macro_accuracy"]
    out["revision_ce"] = rev["macro_cross_entropy"]
    out["retention_revision_untouched"] = _mean(
        [rev["parts"]["untouched_probe"], rev["parts"]["late_untouched"]])
    out["recall_overall"] = rec["macro_accuracy"]
    out["immediate_revision"] = rev["by_delay"]["revised_probe"][
        str(IMMEDIATE_DELAY)]
    revised_later = _mean(rev["by_delay"]["revised_probe"][str(d)]
                          for d in LATER_DELAYS)
    recall_later = _mean([rec["later_block_accuracy"],
                          _mean([rec["parts"]["late_selected"],
                                 rec["parts"]["late_untouched"]])])
    out["later_parts"] = dict(revised_later=revised_later,
                              retention=out["retention_revision_untouched"],
                              recall_later=recall_later)
    out["later"] = _mean(out["later_parts"].values())
    return out


#: every scalar the study requires finite
REQUIRED_SCALARS = ("primary", "revision_ce", "retention_revision_untouched",
                    "recall_overall", "immediate_revision", "later")


def metrics_finite(m):
    """Every declared cell non-empty and finite, and every study aggregate
    finite."""
    for fam in FAMILIES:
        for name, c in m[fam]["cells"].items():
            if c["n"] <= 0 or c["accuracy"] is None or not (
                    onp.isfinite(c["accuracy"])
                    and onp.isfinite(c["cross_entropy"])):
                return False
    vals = [m[k] for k in REQUIRED_SCALARS] + list(m["later_parts"].values())
    return all(v is not None and onp.isfinite(v) for v in vals)
