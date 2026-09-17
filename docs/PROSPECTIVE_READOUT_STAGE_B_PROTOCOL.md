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

For each arm, checkpoint and offset `k` in {1, 2, 4, 8, 16}:

    ||X_t - W_(t+k)||_F / ||W_t - W_(t+k)||_F

**1.0 means "no better than reading the native fast weight."** A pair
`(t, t+k)` is scored only if `||W_t - W_(t+k)||_F > 1e-6`; excluded pairs are
counted and reported.

Reported **separately**, never as one aggregate:

| Split | Values |
|---|---|
| Direction | overall (Frobenius); projected on the revised key (the most recent revision at `t`); projected on the untouched target keys (mean over the five) |
| Fill condition | both; idle gap; intervening writes |
| Offset | each `k` separately |
| Checkpoint | per seed, and pooled as the mean of the four |

**Component split.** For every token the run reports the norms of the write
component `-beta_t U_t` and the decay component `(alpha_t - 1) W_(t-1)` of
`dW_t`, separately for write and non-write tokens and per condition, with a
check that the two sum to `dW_t`. Together with the two-tap/`U`-lookahead
contrast (s4), this is what shows whether a filter that beats literal TSS
does it **by extrapolating the write while not extrapolating the decay** —
the specific hypothesis for the mass term and the specific mechanism that
could protect retention.

**Secondary.** Revised-label and untouched-label probabilities through the
existing learned readout map, at offsets 0–17 after each delay-16 revision,
reusing the existing mechanism-diagnostic machinery. Secondary results do not
decide anything.

## 4. Arms (frozen before execution)

| Family | Grid | Members |
|---|---|---|
| Native identity | — | 1 |
| Two-tap readout `(1+kappa) W_t - kappa W_(t-1)` | `kappa` in {0.25, 0.5, 1, 1.5, 2, 2.5, 3} | 7 (includes `kappa > 1`, i.e. `gamma < 0`) |
| Literal TSS | `T` in {0.75, 1, 1.5, 2, 3, 4, 8} | 7 |
| Generalized interior | `M` in {0.1, 0.25, 0.5, 1}, `gamma` in {−0.5, −0.25, 0, 0.25, 0.5}, `T` in {0.5, 1, 2}, kept only if admissible | ≤ 60 |
| `U`-lookahead `X_t = W_t - lambda beta_t U_t` | `lambda` in {0.25, 0.5, 1, 1.5, 2} | 5 |

Every filter arm's executed coefficients must pass the corrected gate before
any measurement; an inadmissible or unstable member fails the run rather than
being dropped quietly. **Grids are not widened, and categories are not re-cut,
after seeing results.**

Carry cost, from Stage A: the identity, two-tap and `U`-lookahead arms need
**no extra carried state**; literal TSS needs one matrix; the generalized
interior needs two.

## 5. Reporting

The digest prints, per category and condition, a table of arms × offsets with
the best arm marked, then the **argmin per category, condition and offset** —
not a single aggregate — then the component split, then the secondary curves.

## 6. Predeclared stopping rule

Declared here, before execution:

> The generalized interior helps on the read path only if its best member
> beats the best literal-TSS member on the `overall/both` category by at
> least **0.01** in normalized error, at **at least 3 of the 5** offsets.
> Otherwise: **STOP.** Given that we hold the ground truth, the added freedom
> does not help on the read path, and no trained comparison follows.

The rule is evaluated per checkpoint and pooled. Meeting it authorizes
**nothing** by itself: a trained comparison remains a separate decision after
Stage B reports. Failing it ends this line of work. Grids, categories,
offsets and the margin are frozen; they will not be adjusted afterwards.

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
