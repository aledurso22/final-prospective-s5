# MDN recurrent-kernel bridge: plan

Status: PLAN ONLY. No code written, no repository cloned, no job submitted.

This document records the decisions taken on 2026-09-22 for the step between the
finished S5 study and a paper-scale Momentum DeltaNet run. It is a decision
record, not a result. Every number quoted from an earlier study is marked with
its provenance, and section 9 flags the one number set that is not yet in this
repository.

---

## 1. What this bridge is for

Two completed studies point in opposite directions, and they differ in more than
one variable at once:

| study | architecture | placement of the prospective law | headline |
|---|---|---|---|
| S5 three-arm | diagonal SSM, Speech Commands-10, from scratch | on the **recurrence** | generalized costs 1.00 pp |
| MDN six-arm ladder | 8x8 fast weight, synthetic nested memory, continuation | on the **drive** (write residual) | generalized gains 7.53 pp immediate revision |

Neither result transfers to the other, because the law is in a different place.
The bridge holds the architecture fixed at the real MDN and the placement fixed
at the drive, so that the only remaining question is whether the effect measured
on a synthetic 8x8 memory survives in a real language model.

**Scope.** The bridge answers "is the effect real in a real architecture". It
does NOT answer "is this practical at scale". Those are separable and the first
is answered first. See section 6.

---

## 2. The law, and exactly where it goes

Native Momentum DeltaNet, four lines:

    Wbar_t = alpha_t W_(t-1)
    R_t    = m_t (Wbar_t k_t - v_t) k_t^T
    U_t    = mu_t U_(t-1) + eta_t R_t
    W_t    = Wbar_t - beta_t U_t

Every arm changes exactly one of these four lines:

| arm | line changed | change |
|---|---|---|
| Gated DeltaNet | (different layer) | architecture baseline, not a rule variant |
| MDN native | none | baseline |
| QHM | 4 | `W_t = Wbar_t - beta_t [nu U_t + (1 - nu) eta_t R_t]` |
| TSS (M=gamma=0) | 3 | `R_t -> y_t` |
| Generalized (M,gamma,T) | 3 | `R_t -> y_t` |
| Literal Nesterov | 2 | `R_t = m_t (L_t k_t - v_t) k_t^T`, `L_t = Wbar_t - beta_t mu_t U_(t-1)` |

with the filter state `y` driven by, `h` = 1 token, `A = M + h(gamma + T)`:

    A (y_t - y_(t-1)) = M (y_(t-1) - y_(t-2)) + h^2 (R_t - y_(t-1)) + h T (R_t - R_(t-1))

executed in its explicit coefficient form

    y_t = a y_(t-1) - b y_(t-2) + c R_t - d R_(t-1)
    a = [2M + h(gamma+T) - h^2] / A      b = M / A
    c = [h^2 + h T] / A                  d = h T / A

The momentum update and the weight update are the native ones, unchanged.

**The coefficients are gate-independent.** `a, b, c, d` depend only on
`(M, gamma, T)`. The token-varying gates `alpha_t, beta_t, mu_t, eta_t` never
enter them, so the filter is a constant-coefficient LTI system on the residual
stream. This is the property that makes the Jury conditions certifiable at a
checkpoint, and it is why the generalized arm is comparable in difficulty to the
two-tap arm rather than harder.

Source of these equations: `experiments/prospective_momentum/filtered.py` and
`ordinary.py` at `bcdd09447e7ef789cb7a18b3d7b3d00197c87386`.

---

## 3. The arms, and why each one is present

| # | arm | role | parameters | implementation |
|---|---|---|---|---|
| 1 | Gated DeltaNet | **calibration** against the published MDN-vs-GDN gap | 0 | none, exists in fla |
| 2 | MDN native | baseline | 0 | none, their code |
| 3 | QHM (`nu`) | optimizer control; `nu = 1` is native exactly | 1 | smallest |
| 4 | TSS (`M = gamma = 0`, `T`) | the zero-mass boundary face | 1 | shares the arm-5 kernel |
| 5 | Generalized (`M, gamma, T`) | the finite-mass law | 3 | one filter path |
| 6 | Literal Nesterov | optimizer control | 0 | changes the evaluation point |

Arm 1 runs FIRST and ALONE. If the harness does not reproduce the paper's own
MDN-versus-GDN gap at the chosen small configuration, the harness is wrong and
nothing downstream is interpretable.

### 3.1 Why TSS and not the two-tap operator

The two-tap operator (`R + kappa (R - R_prev)`, the `M = 0, gamma + T = h` face)
was the WORST arm of the six-arm ladder, below native. TSS was second best. TSS
is therefore the informative control.

Both are exact coefficient points of the same generalized filter, so including
the two-tap arm costs a GPU and no implementation. Include it only if GPUs are
spare; it is the arm that demonstrates that not every prospective term helps.

### 3.2 Why TSS is the scientifically interesting control

Verified in this repository on 2026-09-22, to machine precision:

    zucchet_coefficients(lambda_bar, b_bar, T)
      == generalized_coefficients(lambda_bar, b_bar, T, mass=0, gamma=0)

    a1 max|diff| = 5.55e-16      c1 max|diff| = 8.88e-16
    a2 max|diff| = 2.22e-16      c2 max|diff| = 0.0

So **the S5 arm that failed is the zero-mass boundary face of the S5 arm that
worked.** At the production initialization `T = 0.05` the companion roots of the
zero-mass arm have magnitude 1.16 to 1.63, which is the rho = 1.10 to 1.71 that
produced NaN at epoch 0, step 0 on all three seeds. Finite `M` and `gamma` pull
the roots inside the unit disc. That is the S5 result stated structurally: mass
and damping are what make the prospective law trainable ON A RECURRENCE.

### 3.3 The falsifiable prediction this sets up

The stability of TSS depends entirely on placement:

* **on the drive**: `b = M/A = 0`, so the filter has the single pole
  `a = 1 - h/T`, stable iff `T > h/2`. This is why TSS did not diverge in the
  six-arm ladder.
* **on the recurrence**: `lambda_bar` enters `a1, a2`, the roots leave the unit
  disc, and training dies immediately. This is what S5 observed.

**Prediction: TSS-on-drive trains; TSS-on-recurrence reproduces the S5 blow-up
in a completely different architecture.** A single cheap probe tests it. If it
holds, the S5 failure is attributable to placement rather than to architecture,
which is the strongest available conclusion from this line of work.

The main bridge runs every arm on the **drive**. Recurrence placement for the
full ladder is a follow-up, not part of this bridge.

---

## 4. Where this lands in the official codebase

Official implementation: `github.com/HuuYuLong/MomentumDeltaNet` ("MDN:
Parallelizing Stepwise Momentum for Delta Linear Attention", ICML 2026). It
vendors flash-linear-attention; upstreaming is in flight as fla PR #1208.
Requires PyTorch >= 2.5, Triton >= 3.0.

| path | what it is | do we touch it |
|---|---|---|
| `fla/ops/momentum_delta_rule/naive.py` | pure-PyTorch reference: `recurrent_momentum_delta_rule_ref()`, `chunk_momentum_delta_rule_ref()` | yes, first |
| `fla/ops/momentum_delta_rule/fused_recurrent.py` | recurrent Triton kernel, `fused_recurrent_mode_rule()` | **yes, this is the bridge** |
| `fla/ops/momentum_delta_rule/chunk.py` | chunkwise Triton fwd/bwd, `chunk_mode_rule()` | **no, not in this bridge** |
| `fla/layers/momentum_deltanet.py` | HF-integrated layer | yes, to expose the new leaves |
| `flame/configs/`, `flame/training_scripts/` | training harness, incl. `training_mdn_400M.sh` | yes, a smaller config |

`naive.py` shipping a recurrent reference beside the fast kernel is the same
oracle discipline this repository used on the S5 two-compartment branch
(sequential scan as default oracle, fast path certified against it to 1.83e-13
in float64). It is reused rather than rebuilt.

---

## 5. The decision that makes the ladder implementable now

**Modify `fused_recurrent` only. Do not touch `chunk.py` in this bridge.**

The six-arm ladder that produced the existing results never had a parallel form
at all: `rollout` is a sequential scan over 64 tokens. So the equations port
one-for-one, and the implementation does not. The recurrent Triton kernel is the
path where the equations port one-for-one AND the implementation is real.

Consequences, all of them good for a go/no-go:

* No chunkwise derivation is required **for any arm**, including literal
  Nesterov, whose only blocker was precisely that derivation. The full six-arm
  ladder becomes implementable on day one.
* The expensive derivation work moves AFTER the go/no-go instead of before it.
* The cost is throughput, and therefore scale. The bridge runs at a small
  configuration with a modest context, and cannot speak to practicality at 400M.

---

## 6. The endpoint, and why the obvious one is wrong

From the six-arm ladder, prospective cluster minus optimizer cluster:

| metric | delta (pp) |
|---|---|
| immediate revision | **+7.53** |
| later revised | +2.67 |
| revision | +1.28 |
| retention | -0.39 |
| recall | -0.37 |

The effect is narrow and specific: overwriting something recently written. It
costs a little retention and recall.

**A bridge judged on language-model validation loss will return a false NO-GO.**
LM loss aggregates precisely the regimes where the rule is neutral to slightly
negative. The endpoint must be revision-shaped: write a key, overwrite it, query
it. MQAR- and NIAH-style retrieval probes are necessary but not sufficient, as
they test retrieval rather than revision.

Report LM loss as a guardrail (it must not deteriorate materially), not as the
decision variable.

---

## 7. Order of work

Nothing before step 5 requires a GPU beyond a small development device.

1. Clone at a **pinned commit**; record the SHA and enforce it with an
   `EXPECTED_COMMIT` guard in the launcher. See section 9 for why.
2. Baseline gate, before touching anything: verify the repository's own
   `naive.py` references agree with `chunk_mode_rule()` and
   `fused_recurrent_mode_rule()`, forward and backward. If this does not hold out
   of the box, stop and find out why.
3. Implement each arm in the **reference path only** (`naive.py` style),
   certified against an independently written float64 reference. Not against
   their code: an independent reference, as on the S5 branch.
4. Port each arm into `fused_recurrent`, gated against step 3.
5. Calibration run: Gated DeltaNet versus native MDN at the chosen small
   configuration, checked against the published gap.
6. Six-arm ladder, one arm per GPU, common pretrained native checkpoint.
7. Go/no-go on section 6's endpoint. Only on a pass does `chunk.py` work begin.

### 7.1 Initialization

Every modified arm starts at its exact native point: `nu = 1`, `M = gamma = 0`
with `T = h`, `kappa = 0`. The first forward pass must reproduce native outputs
exactly. Each arm then continues from a COMMON pretrained native checkpoint, so
that five separate pretraining runs are not needed to isolate the effect of the
rule.

This is not a new idea to validate. It is the protocol the six-arm ladder
already ran, with sha256-verified sources and a common per-seed source across
all momentum-family arms; see `experiments/prospective_momentum/replication_sources.py`.
Port it rather than rederive it.

A short from-scratch run should still follow a positive continuation result,
because successful continuation does not establish stable pretraining from
random initialization.

---

## 8. What does not transfer from the S5 branch

The S5 two-compartment work (`s5/factored_recurrence.py`, 80.5x over the
sequential scan, exact to 1.83e-13 in float64 across all 1152 production modes)
transfers as **mathematics and as an independent cross-check**, not as code:

* transfers: the second-order companion algebra; numerically stable complex
  quadratic roots (constructive sum, then `minor = -a2 / major`); the Jury /
  Schur-Cohn admissibility conditions; the float64 certification pattern; the
  discipline of keeping a sequential oracle as the default implementation.
* does not transfer: any line of it. That repository is JAX; this one is PyTorch
  and Triton.

Keep the MDN work in its own repository or worktree. Vendoring PyTorch and
Triton into the JAX repository would drag its test suite and CI assumptions
somewhere they do not fit.

---

## 9. Open prerequisite: the six-arm results are not in this repository

Every number in sections 3.1 and 6 comes from a run at
`bcdd09447e7ef789cb7a18b3d7b3d00197c87386` (tip of `mdn-nesterov-qhm-audit`,
seeds 501-503) whose results were **never committed**.
`docs/MDN_NESTEROV_QHM_RESULTS.md` does not exist, and section 9 of
`docs/MDN_NESTEROV_QHM_AUDIT.md` still reads "Not run".

Two provenance weaknesses found at that commit:

* `bin/run_experiments/cluster_prospective_nesterov_ladder.sh` has **no
  `EXPECTED_COMMIT` guard**. It only echoes `git rev-parse HEAD` into the log, so
  the tie between code and numbers lives in the run's stdout rather than in a
  gate that would have refused to launch from a different tree. This is why
  section 7 step 1 requires such a guard for the bridge.
* The run replays episode sources from `SOURCE_RUN`
  (`prospective-momentum-replication/20260917-011842`), so the code alone does
  not determine the results.

The bridge is justified by the strength of those numbers. They should be in the
repository before compute is spent on them, taken from the run's own
`results.json` rather than from retyped means. The deciding statistics under the
preregistration are the per-seed signs and the paired jackknife over 32
cell-balanced episode groups at a 1 pp margin, NOT the means quoted above.

Recover the commit the run recorded with:

    grep -m1 '^commit' $PROSPECTIVE_RUNS/prospective-nesterov-ladder/logs/*/*.out

---

## 10. Things deliberately not decided here

* The model size for the bridge. 30M to 100M is the intended range; the exact
  configuration depends on the hardware, which is not recorded anywhere in this
  repository's launchers (they query `nvidia-smi` at run time and write
  `gpu_telemetry.json`). Read it from a past run directory before sizing.
* Wall-clock. No projection is offered. A wall-clock estimate on the S5 branch
  was wrong by roughly 9x because a steps-per-epoch figure was guessed rather
  than measured; no figure appears here without a measurement.
* Whether the bridge trains from scratch as well as from a checkpoint; see 7.1.
* Recurrence placement for the full ladder. Out of scope; only the single TSS
  stability probe of 3.3 is in scope.
