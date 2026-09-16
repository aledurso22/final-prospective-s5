# Report: prospective correction of a trained Momentum DeltaNet memory

Protocol: `docs/PROSPECTIVE_MOMENTUM_PROTOCOL.md`.
Audit: `docs/PROSPECTIVE_MOMENTUM_PROOF_AUDIT.md`.

## Dispatch 1 — `81b6461`: FAILED/4 at focused checks

**No study stage ran.** Source reproduction and update-zero identity (study
stage), preflight, training and performance evaluation did not run. There is
no performance result. The restored-source numerical checks inside the
focused suite did run.

| | |
|---|---|
| started | 2026-09-16T21:43:52Z, `pgi15-gpu3`, SLURM 66104, GPU, JAX 0.11.0 |
| commit | `81b64618c752817d8f4f51eaf550fd86bdd62b55` |
| pytest | **48 passed, 2 failed**, 128.16 s |
| launcher | 234 s of 600. This is a different measurement from pytest's time, and the transcript does not account for the whole difference. |
| source integrity | all six sha256 checks OK (`integrity=0`) |
| terminal | `FAILED 4`, reason `checks exited with 1`; `digest=omitted:no-status-json` |
| logs | `/Users/durso/s5-runs/prospective-momentum/logs/20260916-234352/` (preserved) |

### Failure 1 — `test_kappa_tangent_at_the_restored_start_and_an_interior_point`

```
tests/test_prospective_momentum.py:304:  _kappa_fd(p, 0.5 * kmax, ep)
tests/test_prospective_momentum.py:291:
>   assert abs(jvp - fd) <= FD64 * max(abs(fd), 1e-12), (h, jvp, fd)
E   AssertionError: (1e-06, 5.1932157753886346e-05, 5.1932097511198094e-05)
E   assert 6.024268825216735e-11 <= (1e-06 * 5.1932097511198094e-05)
Captured stdout:
  kappa=0 jvp -8.066587e-04 fd(h=1e-05) -8.066586e-04
  kappa=0 jvp -8.066587e-04 fd(h=1e-06) -8.066588e-04
  kappa=66.58 jvp 5.193216e-05 fd(h=1e-05) 5.193216e-05
  kappa=66.58 jvp 5.193216e-05 fd(h=1e-06) 5.193210e-05
```

At kappa = 66.57722942307834 the h = 1e-6 central difference differs from the
JVP by 6.02e-11 absolute, about 1.16e-6 relative, against the declared 1e-6.
Agreement is better at h = 1e-5. That is consistent with cancellation or
rounding in the finite difference, but it does not by itself establish either
that diagnosis or the correctness of the derivative. The float64 interior
derivative check **failed** in this dispatch. It is not reported as passing.

### Failure 2 — `test_supervisor_cleans_members_left_by_a_completed_leader`

```
tests/test_prospective_momentum.py:911:
>   assert rec["orphans_cleaned"] and rec["kill_sent"]
E   assert (True and False)
Captured stdout:
{'started': True, 'outcome': 'completed', 'rc': 0, 'leader_rc': 0,
 'term_sent': True, 'kill_sent': False, 'orphans_cleaned': True,
 'subreaper': True, 'error': None} 0.0792393684387207
```

The fixture let the leader exit before the descendant had installed its
TERM-ignore disposition. A correctly delivered TERM could then empty the group
without KILL. The outcome is consistent with that race, but the exact signal
ordering was not measured. The prior static review had not identified this
fixture race.

### Passing measurements, scoped

- Restored checkpoint, float32: native nesting at kappa = 0 and g = 1, worst
  relative error 0.00.
- Frozen-token bounds from the source table gates: kappa < 133.1545 and
  g < 267.3090. The projected float32 caps were 133.0214 and 267.0417, and
  the effective relative margin was 9.998e-4.
- Executed float32 table transitions: 288 of 288 stable for native,
  candidate-at-cap and gain-at-cap. Worst entry errors: 2.48e-8, 8.05e-8 and
  1.04e-7.
- Executed-gain underflow (`log_g = -110` executes as 0) was rejected. The
  ordinary gain's relative discrepancy was 1.26e-8.
- Float32 training step: the step cap equalled the cap of the updated gates
  (125.1691), with all 288 transitions stable.
- **Limitations, not passes.** The float32 START derivative (jvp −4.97e-4)
  and INTERIOR derivative (jvp −2.94e-5; FD relative errors 0.23 and 0.48)
  were finite but below the declared float32 resolvability threshold of
  3.97e-3.
- These are scoped numerical checks. They are not training results and not a
  switching certificate.

### Response (before any relaunch)

A test- and documentation-only post-dispatch amendment
(protocol, "Post-dispatch amendment after dispatch 1"):

1. An independent analytic float64 sensitivity recursion is now the decisive
   kappa-derivative reference, at the unchanged 1e-6 relative tolerance, on
   the same checkpoint, episode and two kappa values. Both finite differences
   are kept as printed diagnostics.
2. A bounded readiness handshake is added to the supervisor fixtures.

The model, training protocol, performance criteria and cap are unchanged.
Relaunch requires static confirmation.
