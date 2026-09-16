"""Two deterministic, seed-generated task families over 64 event intervals.

Both families share the SAME event structure and the model receives no family
identifier. Paired key-value events occupy one interval, which is a deliberate
simplification: these are called **MQAR-inspired recall** and **association
revision**, and neither is the published MQAR dataset or protocol.

Layout, zero-based:

    0-7     write the eight target associations in random order
    8-15    eight writes on distinct NON-target keys, independent values
    16-27   four groups of three: write one selected target key; query that key
            on the next interval; then query one of the four untouched targets
    28-55   twenty-eight distracting writes on non-target keys
    56-63   query every target once, in random order

In **association revision** each of the four selected targets is rewritten with
a uniformly sampled value DIFFERENT from its original; in **recall** the
original value is repeated.

Four equally populated query categories per family: immediate selected,
middle untouched, late selected, late untouched. "Immediate" means exactly one
interval after the corresponding middle write. The other categories' ages are
NOT matched and their actual distributions are reported rather than assumed.

The model API never receives the evolving key-value dictionary. Queries carry
no value field, and labels are computed by an oracle that is used for losses
and metrics only.
"""

import hashlib

import numpy as onp

N_KEYS = 32
N_VALUES = 8
N_TARGETS = 8
N_SELECTED = 4
SEQ_LEN = 64
N_QUERIES = 16
WRITE, QUERY, IDLE = 0, 1, 2
FAMILIES = ("recall", "revision")
#: query categories, in a fixed order used by every report
CATEGORIES = ("immediate_selected", "middle_untouched",
              "late_selected", "late_untouched")


def generate_one(rng, family):
    """One episode. Returns arrays plus the oracle labels and categories."""
    targets = rng.choice(N_KEYS, size=N_TARGETS, replace=False)
    non_targets = onp.setdiff1d(onp.arange(N_KEYS), targets)
    initial = rng.randint(0, N_VALUES, size=N_TARGETS)

    key_id = onp.zeros(SEQ_LEN, dtype=onp.int32)
    val_id = onp.full(SEQ_LEN, -1, dtype=onp.int32)
    event = onp.full(SEQ_LEN, IDLE, dtype=onp.int32)

    # 0-7: the eight target associations, random order
    order = rng.permutation(N_TARGETS)
    for i, t in enumerate(order):
        key_id[i], val_id[i], event[i] = targets[t], initial[t], WRITE

    # 8-15: eight distinct non-target writes
    filler = rng.choice(non_targets, size=8, replace=False)
    for i in range(8):
        key_id[8 + i] = filler[i]
        val_id[8 + i] = rng.randint(0, N_VALUES)
        event[8 + i] = WRITE

    # 16-27: four groups of (write selected, query it, query an untouched)
    sel = rng.choice(N_TARGETS, size=N_SELECTED, replace=False)
    untouched = onp.setdiff1d(onp.arange(N_TARGETS), sel)
    sel = sel[rng.permutation(N_SELECTED)]
    unt_order = untouched[rng.permutation(N_SELECTED)]
    new_value = onp.zeros(N_TARGETS, dtype=onp.int32) - 1
    cat = onp.full(SEQ_LEN, -1, dtype=onp.int32)
    for g in range(N_SELECTED):
        s, u = sel[g], unt_order[g]
        if family == "revision":
            # a NEW value, uniformly sampled from the other seven
            shift = rng.randint(1, N_VALUES)
            nv = (initial[s] + shift) % N_VALUES
        else:
            nv = initial[s]
        new_value[s] = nv
        b = 16 + 3 * g
        key_id[b], val_id[b], event[b] = targets[s], nv, WRITE
        key_id[b + 1], event[b + 1], cat[b + 1] = targets[s], QUERY, 0
        key_id[b + 2], event[b + 2], cat[b + 2] = targets[u], QUERY, 1

    # 28-55: twenty-eight distracting non-target writes
    for i in range(28):
        key_id[28 + i] = non_targets[rng.randint(0, non_targets.size)]
        val_id[28 + i] = rng.randint(0, N_VALUES)
        event[28 + i] = WRITE

    # 56-63: query every target once, random order
    late = rng.permutation(N_TARGETS)
    for i, t in enumerate(late):
        key_id[56 + i], event[56 + i] = targets[t], QUERY
        cat[56 + i] = 2 if t in sel else 3

    # oracle: the most recent observed write to the queried key
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
                category=cat, age=age, targets=targets, selected=targets[sel],
                initial=initial, new_value=new_value)


def generate_batch(seed, n_per_family, families=FAMILIES):
    """A batch with equal numbers from both families, from a named stream."""
    rng = onp.random.RandomState(seed)
    eps = []
    for fam in families:
        for _ in range(n_per_family):
            eps.append((fam, generate_one(rng, fam)))
    out = {k: onp.stack([e[k] for _, e in eps]) for k in
           ("key_id", "val_id", "event", "label", "category", "age")}
    out["family"] = onp.array([FAMILIES.index(f) for f, _ in eps],
                              dtype=onp.int32)
    return out


def structure_check(batch):
    """Verify at run time what the task is asserted to be, including that a
    query carries no value field and no write target."""
    ev, vid, lab = batch["event"], batch["val_id"], batch["label"]
    q = ev == QUERY
    w = ev == WRITE
    return dict(
        n=int(ev.shape[0]), length=int(ev.shape[1]),
        queries_per_sequence=sorted(set(q.sum(1).tolist())),
        writes_per_sequence=sorted(set(w.sum(1).tolist())),
        query_value_field_is_absent=bool(onp.all(vid[q] == -1)),
        query_has_no_write_target=bool(onp.all(~(q & w))),
        every_query_has_a_label=bool(onp.all(lab[q] >= 0)),
        no_label_outside_queries=bool(onp.all(lab[~q] == -1)),
        labels_in_range=bool(onp.all((lab[q] >= 0) & (lab[q] < N_VALUES))),
        category_counts={CATEGORIES[c]: int((batch["category"] == c).sum())
                         for c in range(4)},
        age_by_category={CATEGORIES[c]: dict(
            min=int(batch["age"][batch["category"] == c].min()),
            median=float(onp.median(batch["age"][batch["category"] == c])),
            max=int(batch["age"][batch["category"] == c].max()))
            for c in range(4)},
        label_histogram=[int((lab[q] == v).sum()) for v in range(N_VALUES)],
        note=("ages are reported, not matched: only the immediate-selected "
              "category has a fixed age of one interval"))


def episode_digest(batch, n=4):
    """Hash a few generated episodes, for the pre-training record."""
    h = hashlib.sha256()
    for k in ("key_id", "val_id", "event", "label", "category"):
        h.update(onp.asarray(batch[k][:n]).tobytes())
    return h.hexdigest()


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
