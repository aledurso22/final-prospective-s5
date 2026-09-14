"""Controlled cue / distractor / recall task with a separate current-input head.

Purpose: isolate DELAYED RECALL from CURRENT PROCESSING in one sequence, so a
mechanism that changes temporal response can be measured on both at once.
sMNIST is a sequential task and may require history; it simply does not
separate these two demands the way this one does.

Specification (frozen; see docs/GP_EXPERIMENT_PROTOCOL.md)
----------------------------------------------------------
* length 256 (train / dev / test); extrapolation split length 512
* 12 input channels:
      [0:8]  payload bits, iid in {-1,+1}, FRESH AT EVERY TOKEN
      [8:10] current bits, iid in {-1,+1}
      [10]   cue flag
      [11]   query flag
* cue position ~ U{8..31}
* delay bucket ~ U{[16,31], [32,63], [64,95], [96,128]}, then delay ~ U{bucket}
* query position = cue + delay                      (all endpoints inclusive)
* recall target  = the 8 payload bits AT THE CUE, scored ONLY at the query token
* current target = XOR of the two current bits, scored at EVERY token
* labels and masks are never network inputs; delay is independent of payload

Because every token carries a fresh payload drawn from the same distribution,
later payloads are genuine distractors and the query token's own payload
carries no information about the answer.

Loss is the sum of two SEPARATELY averaged binary cross-entropies,

    L = L_recall + L_current

so 256 current targets cannot drown out the single recall event.

RNG domains
-----------
`Split.TRAIN/DEV/TEST/EXTRAP` fold disjoint domain constants into the root key,
so no example can appear in two splits. Evaluation sets are materialized once
and frozen before any model selection.
"""

from dataclasses import dataclass
from enum import IntEnum

import jax
import jax.numpy as jnp

N_PAYLOAD = 8
N_CURRENT = 2
N_CHANNELS = N_PAYLOAD + N_CURRENT + 2          # + cue flag + query flag
CUE_FLAG = N_PAYLOAD + N_CURRENT                # channel 10
QUERY_FLAG = CUE_FLAG + 1                       # channel 11

DELAY_BUCKETS = ((16, 31), (32, 63), (64, 95), (96, 128))
EXTRAP_DELAYS = (192, 256, 384)


class Split(IntEnum):
    """Disjoint RNG domains. Folding these in keeps the splits independent."""
    TRAIN = 0
    DEV = 1
    TEST = 2
    EXTRAP = 3


@dataclass(frozen=True)
class TaskConfig:
    seq_len: int = 256
    cue_lo: int = 8
    cue_hi: int = 31                 # inclusive
    buckets: tuple = DELAY_BUCKETS
    extrap_seq_len: int = 512
    extrap_delays: tuple = EXTRAP_DELAYS

    def max_query(self):
        return self.cue_hi + max(hi for _, hi in self.buckets)

    def validate(self):
        assert self.max_query() < self.seq_len, (
            f"query can reach {self.max_query()} >= seq_len {self.seq_len}")
        assert self.cue_hi + max(self.extrap_delays) < self.extrap_seq_len
        return True


def split_key(seed, split, index=0):
    """Root key for a split. Disjoint domains by construction."""
    return jax.random.fold_in(
        jax.random.fold_in(jax.random.PRNGKey(seed), int(split)), int(index))


def generate(key, batch_size, cfg=TaskConfig(), seq_len=None, delays=None):
    """One batch.

    Args:
        delays: if given, sample delays uniformly from this explicit tuple
            (used by the extrapolation split); otherwise use the buckets.
    Returns dict with
        x               (B, L, 12) float32 inputs
        recall_target   (B, 8)     in {0,1}, the cue payload
        query_idx       (B,)       int32
        current_target  (B, L)     in {0,1}, XOR of the two current bits
        cue_idx, delay, bucket     bookkeeping, NOT inputs
    """
    L = seq_len or cfg.seq_len
    k_pay, k_cur, k_cue, k_buck, k_del = jax.random.split(key, 5)

    payload = jnp.where(
        jax.random.bernoulli(k_pay, 0.5, (batch_size, L, N_PAYLOAD)), 1.0, -1.0)
    current = jnp.where(
        jax.random.bernoulli(k_cur, 0.5, (batch_size, L, N_CURRENT)), 1.0, -1.0)

    cue_idx = jax.random.randint(k_cue, (batch_size,), cfg.cue_lo,
                                 cfg.cue_hi + 1)

    if delays is None:
        b = jax.random.randint(k_buck, (batch_size,), 0, len(cfg.buckets))
        los = jnp.asarray([lo for lo, _ in cfg.buckets])
        his = jnp.asarray([hi for _, hi in cfg.buckets])
        u = jax.random.uniform(k_del, (batch_size,))
        span = (his[b] - los[b] + 1).astype(jnp.float32)
        delay = los[b] + jnp.floor(u * span).astype(jnp.int32)
        bucket = b
    else:
        choices = jnp.asarray(delays)
        b = jax.random.randint(k_del, (batch_size,), 0, len(delays))
        delay = choices[b]
        bucket = b

    query_idx = cue_idx + delay
    t = jnp.arange(L)[None, :]
    cue_flag = (t == cue_idx[:, None]).astype(jnp.float32)
    query_flag = (t == query_idx[:, None]).astype(jnp.float32)

    x = jnp.concatenate([payload, current,
                         cue_flag[..., None], query_flag[..., None]], axis=-1)

    # recall target: the payload AT THE CUE, as bits in {0,1}
    cue_payload = jnp.take_along_axis(
        payload, cue_idx[:, None, None].repeat(N_PAYLOAD, axis=2), axis=1)[:, 0]
    recall_target = (cue_payload + 1.0) * 0.5

    # current target: XOR of the two +-1 bits <=> product == -1
    current_target = (1.0 - current[..., 0] * current[..., 1]) * 0.5

    return dict(x=x.astype(jnp.float32),
                recall_target=recall_target.astype(jnp.float32),
                query_idx=query_idx.astype(jnp.int32),
                current_target=current_target.astype(jnp.float32),
                cue_idx=cue_idx.astype(jnp.int32),
                delay=delay.astype(jnp.int32),
                bucket=bucket.astype(jnp.int32))


def oracle_recall(batch):
    """The exact answer, from the bookkeeping fields. For testing only."""
    payload = batch["x"][..., :N_PAYLOAD]
    cue = batch["cue_idx"]
    got = jnp.take_along_axis(
        payload, cue[:, None, None].repeat(N_PAYLOAD, axis=2), axis=1)[:, 0]
    return (got + 1.0) * 0.5


def oracle_current(batch):
    c = batch["x"][..., N_PAYLOAD:N_PAYLOAD + N_CURRENT]
    return (1.0 - c[..., 0] * c[..., 1]) * 0.5


def _bce(logits, targets):
    """Numerically stable elementwise binary cross-entropy."""
    return jnp.maximum(logits, 0) - logits * targets + jnp.log1p(
        jnp.exp(-jnp.abs(logits)))


def losses(recall_logits, current_logits, batch):
    """Separately averaged BCEs. Recall is read ONLY at the query token.

    recall_logits  (B, L, 8)
    current_logits (B, L)
    """
    q = batch["query_idx"]
    at_query = jnp.take_along_axis(
        recall_logits, q[:, None, None].repeat(recall_logits.shape[-1], axis=2),
        axis=1)[:, 0]                                        # (B, 8)
    l_recall = jnp.mean(_bce(at_query, batch["recall_target"]))
    l_current = jnp.mean(_bce(current_logits, batch["current_target"]))
    return dict(loss=l_recall + l_current, recall_bce=l_recall,
                current_bce=l_current, recall_logits_at_query=at_query)


def metrics(recall_logits, current_logits, batch, n_buckets=len(DELAY_BUCKETS)):
    """Every declared metric, not only the flattering ones."""
    out = losses(recall_logits, current_logits, batch)
    at_query = out.pop("recall_logits_at_query")
    pred = (at_query > 0).astype(jnp.float32)
    correct = (pred == batch["recall_target"]).astype(jnp.float32)
    res = dict(joint_bce=out["loss"], recall_bce=out["recall_bce"],
               current_bce=out["current_bce"],
               recall_bit_acc=jnp.mean(correct),
               recall_exact8_acc=jnp.mean(jnp.all(correct > 0.5, axis=-1)),
               current_acc=jnp.mean(((current_logits > 0).astype(jnp.float32)
                                     == batch["current_target"])))
    per_bit = jnp.mean(correct, axis=-1)
    for k in range(n_buckets):
        m = (batch["bucket"] == k).astype(jnp.float32)
        res[f"recall_bit_acc_bucket{k}"] = jnp.sum(per_bit * m) / jnp.maximum(
            jnp.sum(m), 1.0)
    return res
