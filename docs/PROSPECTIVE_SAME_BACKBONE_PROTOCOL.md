# Protocol: same backbone (ordinary operator) and the retention intervention

**Status: frozen for static review. NOT AUTHORIZED TO RUN.** Nothing in this
study has executed, locally or on the cluster.

- Brief: `PROSPECTIVE_SAME_BACKBONE_AND_RETENTION_BRIEF_2026_09_17.md`.
- Audit (read first): `docs/PROSPECTIVE_SAME_BACKBONE_AUDIT.md`.
- Reused, unchanged: `docs/PROSPECTIVE_MOMENTUM_PROTOCOL.md` (equations,
  projection, frozen-token scope), `docs/PROSPECTIVE_MOMENTUM_PROOF_AUDIT.md`.
- Completed and preserved: `docs/PROSPECTIVE_MOMENTUM_REPORT.md` and
  `docs/PROSPECTIVE_MOMENTUM_REPLICATION_REPORT.md`, including their failed
  retention safeguards.
- Branch: `prospective-same-backbone`, from `prospective-momentum-replication`
  at 2f05a2d.

## 1. The two questions

1. **Same backbone.** Does the prospective correction provide anything beyond
   ordinary first-order prospective residual processing applied to the same
   Momentum memory? The audit proves the two are the *same law* whenever
   `eta/mu` is constant between consecutive tokens, so any measured difference
   is attributable to token-varying gates and nothing else.
2. **Retention.** Can the revision gain survive when the pretrained backbone
   is held fixed and only the prospective coefficient trains? This is an
   experimental hypothesis, not a promised retention guarantee. It does not
   prevent interference from writes along non-orthogonal keys.

No new loss, no retention reweighting, no checkpoint selection, no new
forward equation beyond the declared comparator, and no threshold change.

## 2. Comparator resolution: literal TSS Eq. (17)

Reported explicitly before the arm list is frozen, as the brief requires.

- The residual operator is **ordinary first-order prospective processing**,
  not the full TSS Eq. (17). Labels keep that distinction.
- As a processing stage, literal Eq. (17) at `tau = h` equals the operator at
  `kappa = 1`, under the declared same-token output timing (audit §3). For
  `tau != h` it has an extra state and pole and is a different mechanism.
- **Decision: no literal TSS processing-stage arm in this batch.** At
  `tau = h` it is an algebraically identical special case of the trained-kappa
  operator; for other `tau` it would need its own recurrence, initialization,
  placement, coefficient policy, carry and cost declaration, which neither
  question requires. A **free algebraic check** (no arm, no training) verifies
  the `tau = h` identity and that `tau != h` differs.
- The completed direct-fast-weight Eq. (17) arm keeps its stated applicability
  limits and is **not** relabelled; it does not settle the same-backbone
  question. If the coordinator wants the `tau`-learning stage as a trained
  comparator, that is a separate specification and a separate batch.

## 3. Arms (frozen list)

Three laws × training regime; the frozen native needs no training.

| Arm | Law | Regime | Stored params | Trainable | Carry (reals) |
|---|---|---|---:|---:|---:|
| `prospective_full` | candidate | backbone + kappa | 570 | 570 | 128 |
| `prospective_coeff` | candidate | **kappa only** | 570 | **1** | 128 |
| `ordinary_full` | ordinary operator | backbone + kappa | 570 | 570 | **192** |
| `ordinary_coeff` | ordinary operator | **kappa only** | 570 | **1** | **192** |
| `native_full` | native Momentum | backbone | 569 | 569 | 128 |
| `native_frozen` | native Momentum | **no training**, anchor | 569 | **0** | 128 |

Stored and trainable parameter counts are reported separately in every
record and in the digest.

The operator's third carry matrix is the previous residual: genuine streaming
state, carried across chunk boundaries and zero at episode start. No claim of
equal carry is made.

**Executed laws.** Candidate unchanged. Operator:

    Rpros_t = R_t + kappa (R_t - R_(t-1)),  U_t = mu_t U_(t-1) + eta_t Rpros_t,
    W_t = Wbar_t - beta_t U_t

with the write mask applied to R before the difference, `Rpros` never
re-masked (so the first idle step after a write carries `-kappa eta R_(t-1)`),
old carries on every right-hand side, and `h = 1`. All three laws are the
**same function** at `kappa = 0`, checked per source seed before training.

## 4. Sources: reused read-only

The independently pretrained Momentum checkpoints of the completed
replication run `20260917-011842`: seed **500** for development, **501–503**
for finals. Each is restored through the manifest with its sha256 verified,
nothing is written into that run, and the same files are re-verified at
finalization by both the launcher and the study finalizer.

All arms at a seed share that seed's Momentum source and paired continuation
batches. **This is an intervention study on independently pretrained sources,
not another independent-source replication.**

## 5. Training regimes

- **Full continuation:** unchanged — the backbone and kappa train together.
- **Coefficient-only:** every pretrained leaf (keys, value embeddings,
  readout, all gate parameters) stays exactly fixed; only `kappa` trains,
  starting at 0.
  - Freezing **excludes leaves from the optimizer**; it does not detach
    recurrent states or truncate temporal differentiation. `dL/dkappa` flows
    through the complete recurrence by full BPTT, and no `stop_gradient`
    appears anywhere.
  - Implementation: the gradient of every frozen leaf is zeroed **before** the
    transformation, so the global-norm clip and Adam see only the trainable
    scalar; the update is masked again defensively. **Declared consequence:**
    the clip therefore acts on the trainable scalar alone, whereas in the full
    regime it acts on the whole gradient. This is stated in advance, not
    chosen after seeing results.
  - Bitwise invariance of every frozen leaf is asserted after training (and in
    preflight); any change fails the run.

Everything else is unchanged: the unweighted query-mean cross-entropy, the
optimizer, batch sizes, 200 updates, the two learning-rate slots (0.003 and
0.01), the update-200 selection rule (development only), evaluation
conventions and the projection.

No query label, category label, future key, age or evaluation statistic
enters any recurrence or gate. No protected-key cache and no oracle
projection.

## 6. Streams (frozen, verified disjoint)

| Purpose | Seed | Range |
|---|---|---|
| Continuation training | 320,000,000 + seed·10,000 + update | 325,000,000–325,030,199 |
| Development validation | 350,000,000 | single |
| Final validation (diagnostic) | 351,000,000 | single |
| Held-out | 360,000,000 | single |

Asserted disjoint at start from every earlier study, including the
replication's source-training and continuation ranges. One common held-out
set, generated after all final runs, opening persisted first, never used for
selection.

## 7. Stability and projection

The audit derives the operator's frozen-token transition (a 3×3) and shows its
characteristic polynomial is `z` times the candidate's quadratic: the same
Jury conditions, the same kappa bound, plus a zero eigenvalue. The existing
projection and its declared numerical margin therefore apply unchanged to the
operator — derived, not copied. Validation classifies the operator's rounded
executed 3×3 transitions over the write table in exact rational arithmetic
(cubic Jury), and unstable or non-finite classifications fail. Frozen-token
checks remain distinct from switching stability.

## 8. Focused checks (cluster, inside the cap)

`tests/test_prospective_same_backbone.py`, tolerances unchanged
(ID64 1e-9, GRAD64 1e-8 with the 1e3·eps64·G floor, EXACT64 1e-12):

1. constant `eta`, `mu`: candidate and operator coincide in W and in the
   mapped state, closed loop, with arbitrary alpha, beta, masks, keys and
   values, at kappa ∈ {0, 0.7, 3};
2. equality holds when `eta/mu` is constant and fails when it varies;
3. the one-step difference equals `kappa[eta_t - mu_t eta_(t-1)/mu_(t-1)]R_(t-1)`;
4. the gate-transported reference reproduces the candidate on arbitrary token
   schedules;
5. first idle-step timing and mask ordering;
6. streaming across a chunk boundary, and the dormant-but-stored cache at
   kappa = 0, with the 192-value carry asserted;
7. regression: at kappa = 0 (and g = 1) the completed laws are unchanged by
   the additive dispatch, on the restored source, with nonzero incoming
   carries;
8. shared parameter gradients agree at kappa = 0 for both extensions, and the
   kappa gradient is finite;
9. literal TSS Eq. (17) at `tau = h` equals the operator at kappa = 1, and
   `tau != h` differs;
10. the operator's executed 3×3 matches its analytic form; its characteristic
    polynomial is `z` times the candidate's quadratic; classification agrees
    with the candidate's; above the bound it is unstable;
11. coefficient-only: identical loss and identical kappa gradient to the full
    regime (so freezing does not truncate BPTT), backbone bitwise unchanged,
    kappa moved, and the full regime does move the backbone;
11b. **production precision (amendment R2):** a dedicated x64-DISABLED probe
    process, `tests/prospective_same_backbone_float32_probe.py`, inside the
    same check budget. It asserts parameter, carry, gate and output dtypes and
    finiteness; native nesting at kappa = 0; nonzero-kappa streaming across a
    chunk boundary (TRAJ32 2e-5); the kappa JVP against an INDEPENDENT float64
    sequential sensitivity of the same rounded inputs (REL32 2e-2, a
    cross-precision comparison) and against float32 central differences under
    the declared FD32/REL32 resolvability and limitation policy, finiteness
    first; and the coefficient-only step preserving every pretrained leaf
    exactly while keeping a finite, nonzero kappa gradient through the
    recurrence. The completed studies are not re-run;
12. the arm plan, parameter/carry counts, planned work (25 trained runs,
    5,000 updates, 4 frozen evaluations), stream freshness, per-arm
    development-only selection, every declared comparison reported separately
    with both conditions, and the read-only source verification.

The completed studies' suites are not re-run; the only shared code touched is
an additive dispatch, covered by check 7.

## 9. Budget

One 600-second cap covers the backend probe, focused checks, preflight,
25 trained continuations (5,000 updates), 4 frozen-source evaluations,
held-out evaluation, kill grace, source verification, the digest and the
terminal verdict. Preflight measures both regimes for every trained arm on
disposable state and projects all remaining work plus a recorded 40-second
host allowance; if it does not fit, the run stops INCOMPLETE with the
measured shortfall. Nothing is trimmed and there are no retries. The
supervisor, terminal verdict merging, finalizer and digest are the reused,
passing infrastructure.

## 10. Reporting (digest written before the run)

Reported **separately**, each with its own verdict:

1. candidate versus ordinary operator (fully continued, and again under
   coefficient-only training);
2. candidate versus continued native, and versus the frozen-source anchor;
3. coefficient-only versus full continuation for each law (marked
   `descriptive_only`: these describe the training intervention, and freezing
   is **not** required to raise revision for them to be informative), and
   coefficient-only versus the frozen source;
4. the operator versus continued native;
5. **amendment R3:** coefficient-only versus the **continued native**
   baseline, for the candidate and for the operator. Freezing could beat the
   untrained source while still falling behind the further-trained native
   baseline whose retention the intervention is meant to preserve. These use
   already planned runs: no extra training, sources or data.

The revision difference is printed beside retention and recall in every
comparison: no-measured-decrease alone is never a win if revision fails.

For every comparison the digest prints the mean and per-seed primary,
retention and recall differences, the existing ±1 pp safeguard verdict, **and
the stronger descriptive condition** (no measured decrease: mean retention and
recall both ≥ 0), with the direction of each change stated. A safeguard pass
is not zero loss and not proof of noninferiority. Query categories are printed
with full names: immediate selected, middle untouched, late selected, late
untouched. Also printed: learned kappa with projection activity and bound
ratios, frozen-leaf invariance, gate tables, state norms, parameters, carries
and cost.

## 11. Not claimed

- **No joint literature win can be claimed from this batch.** The active
  native comparator is the continued native Momentum rule plus its frozen
  anchor; `gated_delta` is absent and is named as absent. The completed Gated
  DeltaNet results are historical, not fresh matched evaluations.
- For constant `eta/mu` the two laws are identical, so this is not a test of
  "generalized versus ordinary prospectivity" in general.
- Three seeds on a small associative task: not significance, not a benchmark,
  not SOTA. The QHM relationship and the frozen-token scope limits stand.
- Reusing sources means this is an intervention study, not a replication.

## 12. Amendments after the static review of the worktree

Fixture, orchestration and reporting only; equations, loss, sources, schedule
and the 600-second cap are unchanged, and the added comparisons need no extra
training.

- **R1.** `ST.compare` resolves `PD.DISPLAY[other]`, and this study's rows are
  keyed by ARM id, so the arm ids are registered additively (historical law
  entries untouched); `prospective_full` and `prospective_coeff` remain
  distinct populations. The screen fixture now exercises every declared pair,
  and its semantic assertion on the scope note is case-insensitive.
- **R2.** A dedicated x64-disabled probe covers the new operator rollout,
  streaming cache and coefficient-only optimizer path in production
  precision (§8, item 11b).
- **R3.** Two comparisons added: coefficient-only versus continued native for
  the candidate and for the operator. The coefficient-only-versus-full pairs
  are marked descriptive.
- **Reporting.** Stored and trainable parameters are reported separately; the
  digest prints full category names for per-seed rows as well as means; the
  audit's determinant order, the matched/reachable-state qualification of the
  state map, and the "nonzero in general; magnitude to be measured" wording
  are corrected; table checks are never described as observed-rollout
  coverage.

## 13. Open items for the reviewer

1. The TSS decision in §2: no literal processing-stage arm. Confirm, or
   specify one.
2. The declared clipping consequence in §5 for coefficient-only training.
3. The operator carries 192 real numbers versus the candidate's 128; the
   comparison is not carry-matched, by construction.
4. `ordinary_prospective` is registered by additive dictionary entries
   (`DISPLAY`, `CARRY`, `EXTRA_GRAD`) and an additive rollout branch; no
   completed study's arm list, projection or report changes, and check 7
   covers it.
5. Observed-rollout gate statistics are attached to the candidate and native
   arms by the reused evaluation; for the operator arm only TABLE coverage is
   reported (same gate block, same parameters).
6. Budget is unmeasured until preflight; the checks are a new, smaller suite.
