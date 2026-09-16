# Stable generalized prospective continuation — report

**Status: implemented, awaiting coordinator review. Not executed.**

No check, restore, calibration or training for this study has run anywhere.
Local work was limited to editing, `ast` syntax checks, and static audits of
cross-module symbols and call signatures. Every result section below is empty
until a cluster run fills it, favourable or not.

## Provenance

| | |
|---|---|
| Brief | `NEXT_PROSPECTIVE_S5_CODING_BRIEF_2026_09_16.md` |
| Protocol | `docs/STABLE_GP_CONTINUATION_PROTOCOL.md` (frozen before execution) |
| Branch | `stable-generalized-prospective-s5` |
| Parent | `89a05ca` (`learned-response-timescale`) |
| Worktree | `/private/tmp/wt/sgp` |
| Implementation commit | *(recorded at commit)* |
| Executed commit | *(recorded from the launcher)* |
| Source checkpoint | *(derived and hashed by the runner from `$PROSPECTIVE_RUNS/stage2`)* |
| Command | `bash bin/run_experiments/cluster_stable_gp.sh` |
| Artifacts | `$PROSPECTIVE_RUNS/stable-gp/<stamp>/`, logs in `$PROSPECTIVE_RUNS/stable-gp/logs/<stamp>/` |

## What was implemented

| Piece | Location |
|---|---|
| Modal `j` from raw leaves with the forward clip and clock; `S`; log-arithmetic projection bound with masked division; coupled post-update projection of the complete layer; float64 validation of executed coefficients; transfer function; **TSS Eq. (17)** and its generalized discrete descendant | `s5/stable_gp.py` (new) |
| `rawat_learned_input` (B) and `sgp_learned_input` (C): leaves declared after the common tree, coefficients, two-tap block and diagonal realizations, carries, joint resets, state counts, arms | `s5/rawat_s5.py` (additions only) |
| Diagonal and block two-tap diagnostics for B and C; the one-tap block remainder refuses the two-tap block law instead of returning a wrong tail | `s5/substrate_diagnostics.py` (additions only) |
| Source selection, restore and reproduction, whitelisted warm start, optimizer groups, identity gate, epoch-0 measurement, preflight, nine paired runs, per-epoch domain validation, telemetry, response change, screen | `experiments/gp/stable_gp_study.py` (new) |
| Focused float64 checks and float32 probe | `tests/test_stable_gp.py`, `tests/stable_gp_float32_probe.py` (new) |
| Capped launcher | `bin/run_experiments/cluster_stable_gp.sh` (new) |

No historical arm, runner, protocol or report was modified. `gp_rho_prospin`,
its `rho <= 0.9999` bound and its deferred runner are unchanged and not run.

## Deliberate choices a reviewer should check

1. **B is implemented diagonally, and C as a block.** B uses Rawat's own
   coefficient helper with a per-mode horizon, so `B(q=0) = A` holds with
   identical arithmetic. C uses the Padé block exponential. The **exact**
   `C(r=0) = B` identity is therefore checked in float64 (1e-9 logits, 1e-8
   gradients). In float32 it is gated at 5e-4 logits, 2e-3 gradients and 2e-3
   of `lr` for first updates. Those production-gate tolerances are a judgement
   call and are stated in the protocol with their justification.
2. **Update comparison metric.** It is the maximum entrywise difference in units
   of `lr`, not a per-leaf relative norm. Adam's first step normalizes each
   entry to about `±lr`, so a relative norm would turn float noise in near-zero
   gradients into a spurious failure.
3. **Projection scope.** Only `r` is projected, and its bound is recomputed from
   the updated poles, clock and `T`. `q` and `t` are unconstrained, and their
   non-finiteness is caught by the per-epoch executed-domain validation and the
   finiteness checks.
4. **float32 probe coverage.** Nearly real modes go down to `|Im lambda| = 1.2e-4`,
   with `z` up to about 6e12, plus exactly real modes. The float32 rounding of
   the bound shrinks the `32 eps` interior as `z` grows, so the covered range is
   stated explicitly. Trained checkpoints are validated per epoch regardless of
   whether they fall in that range.

## Known risks, before execution

* **Budget.** Timescale-study measurements on the same GPU put a 10-epoch Rawat
  run at about 70 s and a block run at about 130 s. That projects roughly
  810 s of training, plus checks, restore, the gate and compilation, against
  1160 s usable. It may not fit. Preflight will then refuse and report the
  measured obstruction, without trimming anything.
* **First execution.** This is the first execution of all new code, inside the
  cap. Static audits do not substitute for the cluster checks.

## Results

*(empty — to be filled from cluster output)*

### Checks
### Source checkpoint and reproduction
### Identity gate and epoch 0
### Preflight
### Per-stream endpoints (epoch 10)
### Screen: C vs A, C vs B (paired values), B vs A (descriptive)
### T_in / rho / T movement, stability margins, passive occupancy, projection telemetry
### Current-input and history response changes
### Cost: parameters, carry, time

## Guarantees: checked versus analytical

See protocol §12. Numerical PASS is distinct from development success, and
neither by itself establishes superiority.
