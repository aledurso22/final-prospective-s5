# Frozen protocol: Stage B, the predictive readout diagnostic

17 September 2026. **Frozen for static review. NOT AUTHORIZED TO RUN. No
training in this batch.** Nothing here re-runs, reinterprets or re-evaluates
a completed study; the containment, temporal-response and retention-aware
verdicts, including the infeasible and unavailable ones, stand as recorded.

**What this measures.** On saved native checkpoints, with the backbone
frozen, the readout's job has a ground truth: estimate `W_(t+k)` from the
trajectory up to `t`. If no member of the generalized interior predicts the
near-future fast weight better than the literal-TSS line, the added freedom
does not help on the read path and this line of work stops (s6).

## 1. Placement and the one-way coupling

The backbone is exactly native: residuals keep using `W`, never `X`
(`docs/PROSPECTIVE_READOUT_STAGE_A.md` s1). The readout is

    X_t = a X_(t-1) - b X_(t-2) + c W_t - d W_(t-1),

with `a, b, c, d` the existing functions of `(M, gamma, T, h)` over the
**corrected** domain (`docs/PROSPECTIVE_COEFFICIENT_DOMAIN.md`). Because `X`
enters no backbone line, **one rollout per checkpoint serves every arm**, and
`W` and `U` are bitwise identical across arms. A check asserts this.

## 2. Frozen checkpoints, data and verification

- **Checkpoints:** the declared, read-only native endpoints of the completed
  temporal-response run `20260917-163003` (seed 500 development, 501–503
  final; slot B, update 200), reusing the existing loader: mapping re-derived
  from that run's saved status, restored, hashed, and required to reproduce
  its recorded metrics. Re-verified at finish by the finalizer and by the
  launcher's extra verification.
- **Data:** one fresh probe stream, 720,000,000, 32 episodes per family (64
  total) of the temporal task, structure-checked (balance and oracle) before
  use.
- **Trajectory:** taken from the processing law at its exact native point,
  whose logits are verified against the native rule at TRAJ32 = 2e-5 in the
  same run; every returned array is checked finite and in the production
  dtype, and its executed coefficients must pass the corrected gate.

## 3. Primary measurement

**Primary target: the counterfactual roll-forward.** From `(W_t, U_t)`, the
native recurrence is rolled forward `k` **idle** steps: on an idle token the
residual is zero, so `U <- mu_idle U` and `W <- alpha_idle W - beta_idle U`,
with the gates of an IDLE token (key 0, no value). That is the transient
already in flight — exactly what a prospective readout is entitled to
anticipate. Predicting writes that have not happened is not the readout's
job, and under the intervening-write condition the actual future partly is
unpredictable noise.

**Primary projection: the query key.** What the model needs is the value read
at the query key, so the primary error is measured **at the tokens where a
query is answered**, projected on that query's own key `q`:

    ||(X_t - target_k(t)) q|| / ||(W_t - target_k(t)) q||

**1.0 means "no better than reading the native fast weight."** A pair is
scored only if the denominator exceeds 1e-6; exclusions are counted.

**Secondary:** the same key-projected error against the **actual**
`W_(t+k)`, and the full-matrix (Frobenius) errors against both targets. A
Frobenius win can come from directions nobody queries, so it never decides.

**The primary target is closed form, and how predictable it is gets measured
and can switch the decision — declared here, before the run.** With idle
gates the roll-forward is exactly

    target_k = alpha^k W_t - c_k U_t,
    c_k = beta mu (alpha^k - mu^k)/(alpha - mu)        (verified exactly),

so the `U`-lookahead family contains the `U` part exactly and differs from
the target only by the decay term `(1 - alpha^k) W_t` — the very term the
filtered arms can extrapolate and the lookahead cannot. If `alpha^k` is close
to one, the best `lambda` will match the target almost exactly and every
filtered arm loses by construction. That is an honest answer, not a metric
flaw, but it would make the verdict close to foreknown.

**Declared rule.** Each checkpoint reports the closed-form coefficients
`alpha^k` and `c_k`, how much the per-token gates vary, and the best
lookahead arm's error on the **selection** half. If that error is at most
**0.05** at **at least 3 of the 5** offsets, the primary target counts as
near-trivially predictable and **the interior-versus-baseline decision moves
to the secondary target** (the actual `W_(t+k)`), where real writes and
varying gates make the trend estimate non-trivial. The rule is evaluated on
both targets and both are reported; only which one decides changes, and the
switch is decided on selection-half data, never on the confirmation half.

**Power.** Every reported cell carries its count `n` and the standard error
of its mean, and is flagged **underpowered** when `n < 100` or that standard
error exceeds half the 0.01 decision margin. The rule itself is evaluated on
the pooled `all`-kind, `both`-condition cell — about 640 answered queries per
offset on the confirmation half — while the split by query kind and condition
is reporting. Underpowered cells are listed in the run record and named in
the report instead of being read as results.

Reported separately, never as one aggregate: query kind (revised keys,
i.e. revised probes and late selected, versus untouched keys), fill condition
(both, idle gap, intervening writes), offset `k` in {1, 2, 4, 8, 16}, half
(s4), and checkpoint.

**Component split.** For every token the run reports the norms of the write
component `-beta_t U_t` and the decay component `(alpha_t - 1) W_(t-1)` of
`dW_t`, per condition and for write versus non-write tokens, with a check
that the two sum to `dW_t`. With the two-tap versus `U`-lookahead contrast
(s5) this is what shows whether a winner extrapolates the write while not
extrapolating the decay.

**Secondary label curves.** Revised-label and untouched-label probabilities
through the existing learned readout map, at offsets 0–17 after each
delay-16 revision. They decide nothing.

## 4. Selection and confirmation halves (declared before the run)

Inside every (family, condition) stratum of the 64 probe episodes, the first
half is the **selection** half and the second the **confirmation** half.
Arms are chosen on selection; the winner's **confirmation** number is what is
reported and what the stopping rule uses. With about 80 arms, an argmin on a
single set would find grid-search noise; this project has already been burned
once by a single seed carrying a mean.

## 5. Arms, at matched search budget

Every family that may enter the stopping rule has **exactly seven** members,
so the interior cannot win the argmin by having more grid points:

| Family | Members | Extra carry |
|---|---|---|
| Two-tap readout `(1+kappa) W_t - kappa W_(t-1)` | `kappa` in {0.25, 0.5, 1, 1.5, 2, 2.5, 3} (includes `kappa > 1`, i.e. `gamma < 0`) | none |
| Literal TSS | `T` in {0.75, 1, 1.5, 2, 3, 4, 8} | one matrix |
| `U`-lookahead `X_t = W_t - lambda beta_t U_t` | `lambda` in {0.1, 0.25, 0.5, 1, 1.5, 2, 3} | none |
| Generalized interior | seven declared points, below | two matrices |

**The seven interior points, and why each is there.** They were chosen
without data, around the centre point `(M, gamma, T) = (0.25, 0, 1)`:

| Point | Why |
|---|---|
| (0.25, 0, 1) | centre: a moderate second-order smoother at the TSS-like horizon |
| (0.1, 0, 1) | mass an order below the centre: near the first-order (literal TSS) limit |
| (0.5, 0, 1) | mass doubled: a slower, heavier velocity smoother |
| (0.25, −0.25, 1) | negative damping at fixed mass: total lead `(h-gamma)/h = 1.25`, the region the old passive domain excluded |
| (0.25, +0.25, 1) | positive damping at fixed mass: lead `0.75`, the interior of the old domain |
| (0.25, 0, 0.5) | shorter horizon at fixed mass and lead |
| (0.25, 0, 2) | longer horizon at fixed mass and lead |

This is a one-at-a-time design around one centre, not a search. **A negative
verdict is therefore a statement about these seven points**, on this task,
these checkpoints and these offsets — **not about the interior of the family
as a whole.** The report and the saved verdict both say so. The wider
descriptive grid is reported beside them precisely so that a miss in the
declared seven would be visible.

Plus the **native identity** (reference, 1 arm) and a **descriptive extended
interior** (`M` in {0.1,0.25,0.5,1} x `gamma` in {−0.5,−0.25,0,0.25,0.5} x
`T` in {0.5,1,2}, admissible members only) that is reported but **never
enters the stopping rule**.

Every filter arm's executed coefficients must pass the corrected gate before
any measurement; an inadmissible or unstable member fails the run rather than
being dropped quietly. Grids are not widened and categories are not re-cut
after seeing results.

## 6. Predeclared stopping rule

Declared here, before execution:

> **Baseline:** the better of the **literal-TSS line** and the
> **`U`-lookahead** — the strongest controls, since the lookahead carries no
> extra state and does not extrapolate the decay term at all. Beating the
> TSS line alone is not enough.
>
> **Procedure, per checkpoint:** on the **selection** half, take the argmin
> of the generalized interior and the argmin of the baseline families, at
> matched budget, for the primary key at each offset. Read both arms'
> **confirmation**-half numbers. The interior wins that offset only if its
> confirmation number is at least **0.01** below the baseline's. The
> checkpoint passes if the interior wins **at least 3 of the 5** offsets.
>
> **Power, read before the verdict:** the observed standard error of the
> **pooled** primary metric (all query kinds, both fill conditions,
> confirmation half) is reported **per offset** for both compared arms,
> before any offset is called a win or a loss. If that standard error
> exceeds the decision margin of **0.01** at an offset — or is unavailable —
> that offset cannot resolve a 0.01 difference in either direction. Its
> contribution to the 3-of-5 count is reported as **indeterminate**: it is
> not counted as a pass and not counted as a fail. Indeterminate offsets
> never enter the won count. If, after removing them, the count can no
> longer be reached in either direction, the **checkpoint's** outcome is
> itself `indeterminate`, and an overall verdict that rests on such
> checkpoints is reported as a failure of resolution — not as a STOP. (The
> separate `underpowered` flag, standard error above half the margin, stays
> as a reporting flag and does not change any count.)
>
> **Target:** whichever the declared closed-form test of s3 selects **for
> that checkpoint** — the branch is evaluated per checkpoint, on that
> checkpoint's own idle gates and its own selection-half numbers, never
> globally, because ᾱ differs across the four. The branch **reorders the
> headline; it never discards a result.** Both rules are computed and
> printed for every checkpoint, the decisive one labelled as such and the
> other labelled "reported beside it, not discarded", with the full primary
> numbers still shown. When the branch fires, the report states: *primary
> was near-exactly predictable (best lookahead error X at these offsets), so
> the verdict is taken on the secondary target.*
>
> **Overall:** the interior helps only if it passes on **at least 3 of the 4
> checkpoints** — not on a pooled mean. Otherwise: **STOP.** Given that we
> hold the ground truth, the added freedom does not help on the read path for
> **the seven declared interior points**, and no trained comparison follows.
> A STOP is reported only when the 3-of-4 count is actually out of reach; if
> indeterminate checkpoints leave it reachable, the overall verdict is
> `INDETERMINATE`.

Passing authorizes **nothing** by itself: a trained comparison remains a
separate decision after Stage B reports.

## 7. Budget and constraints

One 600-second cap covers GPU startup, the existing law checks, this stage's
checks, source and checkpoint verification, the diagnostic and finalization.
There is **no training and no optimizer**: the run's update count is zero. A
measured projection from the first arm's cost decides whether the remaining
arms fit; if not, the run stops as INCOMPLETE with nothing reduced. No
retries, no trimming, one launch. Sources and checkpoints are read-only and
verified before and after.

## 8. Scope

The prediction target is the native fast weight, not task accuracy: a lower
normalized error is evidence about the read path only. Prior art
(`docs/PROSPECTIVE_PRIOR_ART.md`) applies unchanged: the filters are known
optimizers, the placement is what is being examined, and no result here is
described as expected to be positive.

## 9. Amendment record (coordinator review, 17 September 2026)

1. **Baseline corrected.** The rule now measures the interior against the
   best of {literal-TSS line, `U`-lookahead}, not the TSS line alone.
2. **Primary target corrected** to the counterfactual roll-forward over idle
   tokens; the actual `W_(t+k)` is secondary.
3. **Primary error is key-projected** at answered queries; the Frobenius norm
   is secondary.
4. **Selection/confirmation halves** declared, and the rule now requires the
   margin on at least 3 of the 4 checkpoints instead of a pooled mean.
5. **Search budget matched**: every family in the rule has seven members; the
   wider interior grid is descriptive only.
6. **Cost projection** from one arm of every structural class.

These change the decision procedure, not the placement or the equations. With
1, 2 and 4 applied the stage is decisive either way.

## 10. Second amendment (coordinator review, 17 September 2026)

1. **Closed-form predictability of the primary target** is measured per
   checkpoint (closed-form coefficients, gate variation, the best lookahead's
   selection-half error) and, by a rule declared **before** the run, can move
   the decision to the secondary target.
2. **The seven interior points are justified one by one** (s5), and both the
   protocol and the saved verdict state that a negative result is about those
   seven points, not about the interior.
3. **Power is reported per cell** (`n`, standard error, underpowered flag),
   and the rule is evaluated on the pooled cell rather than on the sparse
   per-kind, per-offset ones.
4. **Proposed and repaired coordinates are logged separately**, and a grid
   point the corrected repair would move **fails** rather than being reported
   at its nominal location, because `T` also sets the smoother's window.
5. **The gate is exercised on the production float32 path for `gamma < 0`**,
   and the rounded classification is compared with the float64 one.


## 11. Third amendment (coordinator review, 17 September 2026)

Two reporting changes, declared before the single launch; no change to the
model, arms, grids, budget, margin, counts or the 600-second cap.

1. **The 0.05 branch reports both targets.** The threshold and the
   `>= 3 of 5` near-exact condition are unchanged, and the branch is
   evaluated **per checkpoint** (confirmed in code: `closed_form_diagnostic`
   is called inside the per-checkpoint loop, on that checkpoint's own gates
   and selection half; `overall_verdict` records the decisive target per
   checkpoint). It no longer replaces one target's rule with the other's:
   both rules are computed and reported for every checkpoint, and when the
   switch fires the report says "primary was near-exactly predictable (best
   lookahead error X at offsets …), so the verdict is taken on the secondary
   target", with the primary numbers still printed. A branch that discards a
   result is one a future reader cannot check; this one only reorders which
   target carries the verdict.

2. **Indeterminate offsets.** The observed standard error of the pooled
   primary metric is reported per offset before the verdict is interpreted.
   An offset whose standard error exceeds the 0.01 margin contributes
   `indeterminate` to the 3-of-5 count rather than a pass or a fail, and a
   checkpoint whose count becomes unreachable either way is itself
   `indeterminate` rather than a STOP. This changes what a noisy offset is
   allowed to claim; it does not loosen the margin, the counts or any
   criterion.
