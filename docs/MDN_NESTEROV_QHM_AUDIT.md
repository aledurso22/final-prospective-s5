# MDN Nesterov/QHM equivalence audit and controlled ladder

18 September 2026. Branch `mdn-nesterov-qhm-audit`, from `53e89f3`. Every
completed prospective and generalized-prospective result is frozen: nothing
here overwrites, relabels or reinterprets it. Algebra first; the experiment
(§8) is pre-registered below, before any new result exists.

Evidence classes. **Exact** = checked in exact rational arithmetic
(`tests/test_mdn_nesterov_qhm_identities.py`, 20 tests, pure Python
`Fraction`, passing locally). **Production** = checked in float64 against the
executed JAX code (`tests/test_mdn_nesterov_qhm_production.py`, cluster).
**Cited** = from the MDN paper.

## 0. Answers

| # | Question | Answer |
|---|---|---|
| 1 | Is our ordinary prospective MDN literally Nesterov? | **No.** Neither the ladder's literal-TSS arm nor the two-tap operator (`ordinary_prospective`) is literal Nesterov, not even with fixed gates (§3–§5) |
| 2 | What breaks it? | For the two-tap: **(a)** it stores and reads Nesterov's *lookahead*, not its iterate, even at fixed gates; **(b)** `alpha != 1` needs a step rescale the arm does not have; **(c)** token-varying `beta_t`, `mu_t` make the bridge non-causal, because it would need the *next* token's gates; **(d)** token-varying `eta_t` breaks the momentum map. The residual's key dependence (`k_t k_t^T`), masks and the linear delta-rule loss do **not** break it. For literal TSS: the processing pole `1 - h/T` for `T != h`, and at `T = h` a correction weight 1 where Nesterov needs `alpha mu/(alpha + mu - alpha mu) < 1` |
| 3 | QHM, two-tap prediction correction, or both? | The two-tap is **both**: literally a two-tap correction of the residual and, when `mu`, `eta` are constant, **exactly QHM** for any `alpha_t`, `beta_t`, keys and masks. Literal TSS is a two-tap with an **EMA-smoothed** difference (PID with averaged D, `K_d = 1`), and QHM only at `T = h` |
| 4 | Does generalized prospectivity beat the strongest of TSS, Nesterov and QHM? | **Pending the run** (§8), decided by the pre-registered paired rule |
| 5 | Is the +5.2–5.3 pp immediate-revision advantage still present against literal Nesterov? | **Pending the run** |
| 6 | Does any advantage persist later? | **Pending the run**. The completed runs show immediate +5.34 / +5.21 with later −0.12 / −0.26 against literal TSS |
| 7 | Distinct degree of freedom, or a standard optimizer reparameterized? | **Algebraically distinct in one direction.** The `(M, gamma, T)` family contains the two-tap for `kappa <= 1`, hence QHM points (fixed `mu`, `eta`) and a Nesterov-read-ahead point (`kappa = mu`, `alpha = 1`, fixed gates). `M > 0` adds a second-order smoother of the difference that none of these optimizers has. It does **not** contain literal Nesterov, whose correction follows the token gate `beta_t mu_t` and which reads the iterate. Whether the extra freedom helps is question 4 |

**Algebraic gate: `DISTINCT_UPDATE`** for the arm the ladder uses (literal TSS),
and `QHM_EQUIVALENT_NOT_LITERAL_NAG` for the two-tap operator. Neither is
`EXACT_EQUIVALENCE_FULL_MDN` or `EXACT_EQUIVALENCE_FIXED_GATES_ONLY` with
respect to literal Nesterov, so the literal-Nesterov control is
**non-redundant**. It is implemented and pre-registered in §8, not yet run.

## 1. The recurrence, in the repository's orientation

MDN v1 (arXiv:2605.05838v1, Eqs. 3–5, **cited**), with `S, M in R^(d_k x d_v)`:

    M_t = mu_t M_(t-1) + eta_t grad L(S~_(t-1)),   S_t = S~_(t-1) - beta_t M_t,
    S~_(t-1) = alpha_t S_(t-1),   L(S~) = 1/2 ||v_t - S~^T k_t||^2,
    grad L(S~) = -k_t (v_t - S~^T k_t)^T.

The repository stores `W = S^T` (`d_v x d_k`, `W k` predicts `v`) and
`U = M^T`, and executes (`experiments/nested_memory/dynamics.py:
momentum_delta_step`):

    Wbar_t = alpha_t W_(t-1),   R_t = m_t (Wbar_t k_t - v_t) k_t^T = m_t (grad L)^T,
    U_t = mu_t U_(t-1) + eta_t R_t,   W_t = Wbar_t - beta_t U_t.

Same signs and ordering as MDN; `m_t` masks the residual to write tokens. The
gates are token functions (`NM._momentum_gates` on the token's own features),
so every gate at token `t` is available causally at `t`.

**MDN's output correction is not Nesterov.** §3.3 of the paper: "output
correction with `q_t = q_t - d k_t` before L2Norm" (after Hu et al. 2025). It
modifies the *query* on the read path and never touches `S`, `M` or the loss.
Nesterov changes where the *write* residual is evaluated. The shared shell
omits the output correction for every arm
(`experiments/nested_memory/model.py`).

**One record flagged, not resolved.** `docs/PROSPECTIVE_PRIOR_ART.md` records a
17 September external verification that MDN's Appendix C names Nesterov
momentum, Adam and Muon as future work. An 18 September fetch of the v1 HTML
through a summarizing tool did not surface that sentence. Nothing below
depends on it. The earlier record stands and should be re-checked by hand.

## 2. Literal Nesterov on the MDN recurrence

**Convention.** Sutskever et al. write
`v_(k+1) = mu v_k - eps grad f(theta_k + mu v_k)`: the gradient is taken at the
point the *momentum part of the current step* reaches. MDN's step is
`-beta_t (mu_t U_(t-1) + eta_t R_t)` after the decay, and its momentum part is
`-beta_t mu_t U_(t-1)`. So

    L_t = alpha_t W_(t-1) - c_t U_(t-1),    c_t = beta_t mu_t          (derived)

with the lookahead taken after the native decay, as MDN applies it. The
alternative reading `c'_t = mu_t beta_(t-1)` (the previous step's velocity)
coincides only for constant `beta`. It is not declared, because `L_t` would
then not lie on the executed path.

**Recurrent update** (`nesterov_step`; zero extra parameters):

    Wbar_t = alpha_t W_(t-1),   L_t = Wbar_t - beta_t mu_t U_(t-1),
    R_t^NAG = m_t (L_t k_t - v_t) k_t^T,
    U_t = mu_t U_(t-1) + eta_t R_t^NAG,   W_t = Wbar_t - beta_t U_t = L_t - beta_t eta_t R_t^NAG.

**Effective two-step recurrence** (exact, token gates, `t >= 2`):

    L_t = alpha_t X_(t-1) + (beta_t mu_t / beta_(t-1)) (X_(t-1) - alpha_(t-1) X_(t-2)),
    Nesterov: X_t = L_t - beta_t eta_t G_t(L_t),
    native:   X_t = L_t - beta_t eta_t G_t(alpha_t X_(t-1)),

where `G_t(P) = m_t (P k_t - v_t) k_t^T`. Native and Nesterov share the same
extrapolation and differ only in where the gradient is evaluated
(`test_nesterov_two_step_form_shares_the_extrapolation_with_native`).

**Momentum form.** Because the delta-rule residual is affine in `P`,

    U_t^NAG = mu_t U_(t-1) + eta_t R_t(Wbar_t) - eta_t m_t beta_t mu_t (U_(t-1) k_t) k_t^T.

That is, native momentum whose component along the current key is damped by
the loss before it is carried. Nesterov equals native exactly on non-write
tokens (`m_t = 0`) and whenever `mu_t = 0` (exact).

**Cost.**
- State: `(W, U)`, 128 reals, the same as native, with no extra carry.
- Compute: one extra `d_v x d_k` axpy per token to form `L_t`; the residual
  matvec is the same size.

**Chunkwise form.** The per-token transition on the rows `[W, U]` is

    [[alpha(I - q K), alpha eta K], [-beta mu (I - q K), mu (I - q K)]],   K = m k k^T,  q = beta eta.

It is linear in the state, with identity-plus-rank-one blocks. But the
momentum row now carries `(I - q K)`, which native's `mu I` does not.
Equivalently, the transition is a gate shear followed by a delta-rule factor,
in the opposite order to native. MDN's intra-chunk expressions
(`V~ = U - Y S + Z M`, the UT transform of `prod (I - beta k k^T)`) are derived
for the native order, so **their exactness for Nesterov is not established**. A
new chunkwise derivation would be required. It is plausible (the factors are
still identity plus rank one), but it is not the MDN one. The controlled
comparison below uses the recurrent oracle, which is exact.

**Frozen-token stability** (unit key, `m = 1`; exact):
`tr = (alpha + mu)(1 - q)`, `det = alpha mu (1 - q)`.
- Two Jury expressions are always positive in the pinned gate ranges.
- The third, `1 + tr + det = 1 + (1 - q)(alpha + mu + alpha mu)`, gives:
  **literal Nesterov is frozen-token stable iff
  `(q - 1)(alpha + mu + alpha mu) < 1`**.
- Native is stable throughout the pinned ranges. At
  `alpha = mu = 9/10, beta = 1, eta = 19/10`, Nesterov is unstable where
  native is stable (exact). This is the classical fact that Nesterov needs a
  smaller effective step than heavy ball.
- QHM with `nu in [0, 1]` is stable throughout (exact).

## 3. The existing corrections, exactly

| Arm (code) | Update into the native momentum |
|---|---|
| `native_full` | `U_t = mu U + eta R_t` |
| `operator_full` = `ordinary_prospective` (`ordinary.py`) | `U_t = mu U + eta [(1 + kappa) R_t - kappa R_(t-1)]` |
| `tss_processing` (`filtered.py`, `M = gamma = 0`, `T` trains) | `y_t = (1 - h/T) y_(t-1) + (1 + h/T) R_t - R_(t-1)`, `U_t = mu U + eta y_t`; equivalently `y = R + (1/T)(1 - z^-1)/(1 - (1 - h/T) z^-1) R` |
| `generalized_processing` (`M, gamma, T` train) | `y_t = a y - b y_prev + c R_t - d R_(t-1)`, `U_t = mu U + eta y_t` |
| `prospective_momentum` (first candidate) | maps onto the two-tap exactly for constant `eta`, `mu` (`docs/PROSPECTIVE_SAME_BACKBONE_AUDIT.md`), so it inherits the two-tap's classification |

## 4. The coefficient map, with every assumption

| Claim | Exact under | Fails when | Test |
|---|---|---|---|
| two-tap(`kappa`) = **QHM**, `nu = 1 - kappa/(mu(1 + kappa))`, step `(1 + kappa) beta` | constant `mu` and `eta`; **any** `alpha_t`, `beta_t`, keys, values, masks; closed loop; no condition on decay placement | `eta_t` or `mu_t` varies. The exact condition is that `eta_s(1 + kappa) - kappa eta_(s+1)/mu_(s+1)` is constant over the history | `test_two_tap_is_qhm_exactly_when_mu_and_eta_are_constant`, `test_token_varying_eta_breaks_the_qhm_bridge` |
| QHM `nu` in `[0, 1]` | `0 <= kappa <= mu/(1 - mu)`; `kappa = mu` gives `nu = mu/(1 + mu)`; `kappa = mu/(1 - mu)` gives `nu = 0` (first-order delta rule, the PM boundary of the prior-art record) | — | `test_qhm_weights_of_the_two_tap` |
| two-tap(`kappa = mu`) and literal Nesterov share the gradient sequence, with the two-tap's stored `W_t` equal to Nesterov's next lookahead `L_(t+1)` | fixed `beta`, `mu`, `eta` and **`alpha = 1`**; any keys, values, masks | **always read at different points**: already after the first write, `W_1 = -beta eta (1 + mu) g_1` versus `X_1 = -beta eta g_1` | `test_fixed_gates_alpha_one_two_tap_at_kappa_mu_is_nesterov_read_ahead` |
| Same, for fixed `alpha != 1` | `kappa* = alpha mu/(alpha + mu - alpha mu)` **and** a step `beta' = beta (alpha + mu - alpha mu)/alpha`, which the executed arm cannot express | with the executed `beta`, no `kappa` works unless `alpha = 1` | `test_fixed_gates_general_alpha_needs_a_rescaled_step` |
| Any causal two-tap or processing state equal to Nesterov's lookahead | never, with token gates: `L_2` depends on token 2's `(beta_2, mu_2)` | two sequences equal through token 1 have equal causal states but different `L_2` | `test_token_gates_make_the_two_tap_bridge_noncausal` (the smallest counterexample, two tokens) |
| literal TSS = two-tap(`kappa = 1`) | `T = h`, closed loop, any gates | `T != h`: an extra pole `1 - h/T`, not cancelled (zero `T/(T + h)`) | `test_tss_at_T_equal_h_is_the_two_tap_at_kappa_one`, `test_literal_tss_is_never_nesterov` |
| literal TSS = literal Nesterov | **never** in the pinned gates: at `T = h` it would need `alpha mu/(alpha + mu - alpha mu) = 1`, i.e. `alpha = mu = 1`; for `T != h` the pole | — | `test_literal_tss_is_never_nesterov` |
| generalized family contains two-tap(`kappa`) | `(M, gamma, T) = (0, h(1 - kappa), kappa h)` with `kappa <= 1` under the passive domain (`kappa > 1` needs `gamma < 0`) | — | `test_generalized_family_contains_the_two_tap_up_to_kappa_one` |
| zero-momentum and boundary reductions | `mu = 0`: Nesterov = native = gated delta. `nu = 1`: QHM = native. `nu = 0`: delta rule on `Wbar`. `kappa = 0`: native. `m = 0`: Nesterov = native | — | `test_zero_momentum_and_nonwrite_boundaries`, `test_qhm_boundaries` |

**No containment hierarchy is claimed.** The only exact inclusions are the
ones in this table, under the assumptions stated in its rows.

## 5. Classification

- **Literal-TSS processing (the ladder's arm 2): `DISTINCT_UPDATE`.** It is not
  literal Nesterov under any admissible gates. It is QHM only at `T = h` with
  constant `mu >= 1/2` and constant `eta`.
- **Two-tap operator (`ordinary_prospective`): `QHM_EQUIVALENT_NOT_LITERAL_NAG`.**
  It is exact QHM with constant `mu`, `eta`. It matches Nesterov only as a
  read-ahead of Nesterov's trajectory at fixed gates and `alpha = 1`, never
  in the stored coordinates.

Full equivalence fails, so, per the brief, the distinct literal-Nesterov
control is implemented (`experiments/prospective_momentum/nesterov_ladder.py`)
and QHM is included as a separate arm. With token gates, QHM is distinct from
both literal TSS and literal Nesterov (§4).

## 6. Implementation

Nothing completed is modified. Every existing module is byte-identical; the
additions are:

- `experiments/prospective_momentum/optimizer_forms.py`: pure-Python reference
  forms (native, Nesterov, QHM, two-tap, the executed processing filter),
  usable with `Fraction`;
- `experiments/prospective_momentum/nesterov_ladder.py`: the two rules in the
  production shell (the same helpers and order as `model.rollout`), their
  training step (`study.train_step` line for line, with the `nu` projection
  onto `[0, 1]`), the ladder runner and the paired analysis;
- `experiments/prospective_momentum/nesterov_ladder_summary.py`: a read-only
  digest, **with** a `__main__` entry point;
- `bin/run_experiments/cluster_prospective_nesterov_ladder.sh`;
- the two test files.

The native, literal-TSS and generalized arms run through the completed
study's own code (`temporal_response.host_step`, `tss_containment` validation
and repair), not a re-implementation. `PD.DISPLAY`/`PD.CARRY` gain entries by
`setdefault` only.

## 7. Tests

| Required | Where |
|---|---|
| hand-computed 1-, 2-, 3-token updates | exact `test_hand_computed_one_two_three_token_updates`; production bitwise `test_hand_values_are_reproduced_bitwise` (Nesterov `W = 1/2, 3/4, 13/16`; QHM `nu = 1/2`: `1/2, 3/4, 27/32`) |
| causality | exact `test_every_form_is_causal`; production `test_no_future_token_or_gate_enters_an_update` (future keys, values and events perturbed; earlier logits bitwise equal) |
| fixed-coefficient identities | §4 rows |
| variable-gate counterexamples | `test_token_gates_make_the_two_tap_bridge_noncausal`, `test_token_varying_eta_breaks_the_qhm_bridge` |
| zero-momentum / boundary reductions | `test_zero_momentum_and_nonwrite_boundaries`, `test_qhm_boundaries` |
| implemented recurrence = direct unrolled reference | `test_rollout_equals_a_direct_unrolled_reference` (float64, `1e-12`) |
| finite outputs in the declared stable region | `test_outputs_stay_finite_in_the_declared_stable_region`, plus the exact Jury closed forms |
| existing ordinary and generalized arms unchanged | `test_completed_arms_still_execute_their_documented_laws` (native, two-tap, literal-TSS and interior processing against the exact forms, float64); `test_the_ladder_shell_is_the_production_shell` |

## 8. Pre-registered protocol (written before any new result)

**Reused unchanged from `docs/PROSPECTIVE_TEMPORAL_RESPONSE_PROTOCOL.md`:**
- the task (`temporal_task`), episode construction, cell weighting and every
  metric definition;
- the read-only sources (seed 500 development; 501, 502, 503 final);
- the **same stream seeds** (`temporal_response.STREAM`);
- 200 updates; learning rates A = 0.003 and B = 0.01; checkpoints at 0, 25,
  50, 100 and 200;
- the train/validation/held-out separation;
- the single development ordering (revision macro accuracy, then lower
  revision CE, then fewer updates, then lower learning rate), identical for
  every arm.

**Ladder** (the order is declared):

| Arm | Law | Extra params | Executed carry | Start |
|---|---|---|---|---|
| native | MDN | 0 | 128 | source |
| literal TSS | processing, `M = gamma = 0`, `T` trains | 1 | 320 | `T0 = h` |
| literal Nesterov | `c_t = beta_t mu_t` | 0 | 128 | source (same tree as native) |
| QHM | `nu in [0, 1]`, projected after each update | 1 | 128 | `nu = 1` (native function) |
| generalized | `M, gamma, T` train | 3 | 320 | `T0 = h` |

Update budgets are identical. Parameters and state are matched where the laws
allow, and the unavoidable differences are the table's. The two-tap operator
arm of the completed study is not in the ladder; its result is frozen. The
mechanism diagnostic and the matched-operating-point analysis are not
repeated. The completed study's own checks run inside the cap; its law
modules are byte-identical and are covered by the unchanged-arms test.

**New coefficient.** QHM's `nu` is learned by BPTT on the training stream,
and its endpoint is chosen on the development validation stream. **No new
coefficient touches the held-out stream.** Nesterov has no coefficient.

**Declared start checks.**
- Nesterov's tree equals the native tree.
- On the completed study's representative episodes, and at its tolerance
  `TRAJ32`, this file's shell running the native step and QHM at `nu = 1`
  both reproduce the production native logits.
- The literal-TSS start is accepted as before.

**Declared stability policy.** A literal-Nesterov checkpoint whose executed
frozen-token table contains an unstable transition is **ineligible for
selection**; it is recorded, not a run failure. If no Nesterov checkpoint is
eligible, Nesterov is reported **UNAVAILABLE**. It is not damped, clipped or
rescaled. Non-finite values fail the run for every arm. A QHM instability
fails the run, because it contradicts the derivation.

**Analysis: paired, on the one common held-out set.**

1. For each final seed and each pair of arms, compute the difference of each
   metric on **the same held-out episodes**.
2. Its standard error comes from a **grouped jackknife** over 32 groups of
   balanced 4-episode blocks. Each block holds one episode of every condition
   and order flip, so every group is cell-balanced.
3. Across the three seeds:
   - `D` = mean of the seed differences;
   - `SE_within = sqrt(sum SE_s^2)/3`;
   - `CI95 = D ± 1.96 SE_within`;
   - `SE_between` (the spread across training seeds) is reported alongside.

The uncertainty is always that of the **paired difference**. No metric's own SE
is ever compared with the margin, which is the flaw recorded in
`docs/PROSPECTIVE_READOUT_REPORT.md` §5.

**Metrics:**
- revision (primary), retention, recall;
- immediate revised-key accuracy (nominal delay 1: offsets 1 and 2);
- later revised-key accuracy (delays 4, 8, 16) and the composite `later`;
- untouched keys;
- revised and untouched accuracy in the idle-gap and intervening-writes
  conditions.

**Labels** (margin 0.01, the protocol's +1 pp):

| Label | Condition |
|---|---|
| `BETTER` | lower bound > 0, every seed > 0, `D >= 0.01` |
| `WORSE` | the mirror of `BETTER` |
| `EQUIVALENT_WITHIN_MARGIN` | the whole interval inside `(-0.01, 0.01)` |
| `BETTER_BELOW_MARGIN` / `WORSE_BELOW_MARGIN` | significant, consistent in sign, not inside the margin, `abs(D) < 0.01` |
| `INDETERMINATE` | anything else |

**Comparisons:**

| Candidate | Against |
|---|---|
| generalized | TSS, Nesterov, QHM, native |
| Nesterov | TSS, QHM, native |
| QHM | TSS, native |
| TSS | native |

"Strongest comparator" means the arm with the highest mean **final-validation**
revision among TSS, Nesterov and QHM, taken on the `eval_validation` stream
and never on held-out data. "Strongest known optimizer" is the same over
Nesterov and QHM.

**Recommendation, mechanically, in this order:**

1. `GENERALIZED_GP_RETAINS_DISTINCT_ADVANTAGE`: generalized vs the strongest
   comparator is `BETTER` on revision, and not `WORSE` on retention or
   recall.
2. `RUN_LITERAL_NESTEROV_AT_SCALE`: all of the following hold.
   - Nesterov is `BETTER` than native on revision.
   - It is not worse (at or below the margin) than TSS or QHM.
   - Generalized is not better than it (at or below the margin).
   - Its retention and recall are not `WORSE` than native.
3. `GENERALIZED_GP_REDUCES_TO_KNOWN_OPTIMIZER`: generalized vs the strongest
   known optimizer is `EQUIVALENT_WITHIN_MARGIN` or worse on **both** revision
   and immediate revision.
4. `NO_GO`: otherwise, or on an incomplete or failed run.

`REDUNDANT_CONTROL_CONFIRMED` is reserved for an exact algebraic identity. §5
ruled that out, so the experiment cannot return it.

Question 5 is read from `generalized_vs_literal_nesterov / immediate_revised`
and question 6 from the `later_revised` and `later` rows. Both are secondary
to the recommendation. The completed runs' per-seed immediate differences
against TSS were +7.62 / +7.23 / +0.78 (retention-aware λ = 0), so the
every-seed sign condition is informative, not a formality.

**Budget.** One 600-second cap with a 30-second reserve. Measured preflight
refuses rather than trims. At most 5,000 updates. No retries.

**Command** (cluster, after clearance):

    cd "$PROSPECTIVE_REPO" && git fetch origin mdn-nesterov-qhm-audit \
      && git checkout mdn-nesterov-qhm-audit && git reset --keep origin/mdn-nesterov-qhm-audit \
      && git rev-parse HEAD \
      && bash bin/run_experiments/cluster_prospective_nesterov_ladder.sh 2>&1 \
         | tee ~/nesterov_ladder_$(date +%Y%m%d-%H%M%S).log

## 9. Results

Not run. The results report (`docs/MDN_NESTEROV_QHM_RESULTS.md`) and the
machine-readable `results.json` extract will be added after the single
dispatch, linked from here, and will not change §0–§8.
