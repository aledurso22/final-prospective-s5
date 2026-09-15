# Combined prospective input + prospective recurrence: status

Protocol: `docs/COMBINED_INPUT_RECURRENCE_PROTOCOL.md`.

> **No result is reported here, because nothing has been executed.**
> The implementation is complete and committed. Its comparative training batch
> was **deferred before dispatch** by the priority handoff of 15 September
> 2026, which reorders the experiments so that the standalone recurrence with
> learned response timescales is tested first
> (`docs/LEARNED_RESPONSE_TIMESCALE_PROTOCOL.md`).

## Execution status

| item | value |
|---|---|
| implementation parent | `6a5bc7fadc324cef72cb2eaab60e6aa2dcfe76e6` |
| branch | `combined-input-recurrence` |
| cluster runs dispatched | **none** |
| cluster runs completed | **none** |
| focused checks executed | **none** (written, cluster-only, not yet run) |
| budget consumed | **0 s** |
| artifacts | none created |

Nothing was started, so there is nothing to preserve or reconcile. No earlier
run directory was touched and no earlier report changed.

**What "implemented" means here and what it does not.** The composition's
algebra, its exact interval law, its streaming/reset semantics, its diagnostics
extraction and its projection policy are written, and a focused check is
declared for each with predeclared dtype-appropriate tolerances. **None of
those checks has been run**, because all numerical work belongs on the cluster
and this batch was deferred before it got there. No claim of numerical
validation is made, and the status will stay this way until a run happens.

## What was implemented

| piece | location |
|---|---|
| exact two-tap block coefficients `J_in = T_in A_bar B`, `B_+`, `B_-` | `s5/gp_fixed.py::mass_block_two_tap` |
| two-tap block scan, parallel and independent sequential, with `q` and `x_{k-1}` carries and joint reset | `s5/gp_fixed.py::mass_scan_two_tap*` |
| response `gp_rho_prospin`, arm of the same name, state counts including the input buffer | `s5/rawat_s5.py` |
| two-tap block impulse/frequency extraction, kept distinct from the one-tap mass rules | `s5/substrate_diagnostics.py` |
| the single shared post-update projector and its telemetry | `s5/response_projection.py` |
| Speech Commands trainer: imports that projector; per-arm response-leaf policy | `experiments/gp/rawat_benchmark.py` |
| the deferred six-run paired study | `experiments/gp/combined_study.py` |
| focused checks and the production-dtype probe | `tests/test_combined_input_recurrence.py`, `tests/combined_float32_probe.py` |
| launcher, refusing to run unless re-authorized | `bin/run_experiments/cluster_combined.sh` |

## Reusability, which is the point of finishing it

The recurrent core and the input correction are kept **separate** by
construction. `mass_block_two_tap` takes the recurrent horizon `T` and the
input horizon `T_in` as independent arguments, and `T` may be a scalar or a
per-mode `(P,)` array. The composition therefore accepts either the
fixed-timescale core or the learned-timescale core of the priority study
without a second derivation; `T_in` is never silently replaced by a learned
`T_i`, and no additional `Delta` is inserted. Setting `T_in = 0` recovers the
plain generalized recurrence, which is what the priority batch runs.

## When this is re-authorized

The batch in `docs/COMBINED_INPUT_RECURRENCE_PROTOCOL.md` s6 stands as
declared. Before it runs, its runner must be re-checked against the
then-current recurrent core, and it must not share a budget with another
study. A completed failed performance comparison would be a result, not a
software failure; that distinction is kept in the status contract the launcher
prints.
