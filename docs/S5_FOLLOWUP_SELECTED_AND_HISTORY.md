# Follow-up: fixed selections, seed 303, and finite-response history

This work starts from `940459bca69cccf3470768faaf4b41f0b66a914f` in a separate
checkout. Existing model, optimizer, training runner, data, original launchers
and evaluation implementations are unchanged. No cluster work was launched
from the development machine.

## What is being checked first

The completed N/A/C/D study at
`/Users/durso/s5-runs/s5-timescale-2x2/20261117-195818` selected these validation
scores (rounded values supplied from the cluster):

| Arm | Seed 301 | Seed 302 | Mean |
|---|---:|---:|---:|
| N, native | 96.84% | 96.44% | 96.64% |
| A, original generalized | 96.19% | 96.46% | 96.33% |
| C, calibrated old corrector | 96.73% | 96.95% | 96.84% |
| D, calibrated prediction corrector | 96.98% | 96.79% | 96.88% |

Calibration improves both seeds relative to A. D has a small positive paired
lead over N; D versus C remains unresolved. These two seeds do not establish
statistical equivalence or a reliable accuracy improvement. The audit reads
the exact files rather than reconstructing results from this rounded table.

The earlier A302 epoch-2 intervention demonstrated that replacing saved BN
statistics could recover evaluation at fixed weights. It does not establish
whether selected-epoch rankings are sensitive to BN. That is the purpose of
the uniform follow-up.

### Selected-checkpoint audit

`experiments.s5_predictor_corrector.selected_checkpoint_audit` requires complete
15-epoch histories and verifies the original selection rule (maximum validation
accuracy, then minimum cross-entropy, then earliest epoch). It freezes checkpoint,
metrics, result and metadata hashes. It reports paired results and the actual
selected N/A/C/D modes using each arm's own equation. Mode analysis is float64
equation analysis, separately labeled from production float32 replay.

`experiments.s5_predictor_corrector.selected_bn_audit` consumes that manifest.
It applies both existing training-only BN recalibrations to every selected
checkpoint, using the original evaluation code and identical procedures.
There is no epoch reselection, choice of recalibration by validation score, or
replacement of primary results. Aggregation requires successful saved-score
reproduction and all eight tasks. Outputs are outside the original run.

Limitations: pooled moments use upstream batch normalization during collection;
they are not a recursive fixed point of evaluation-mode population moments.
The inherited diagnostic's sorted per-batch training probe is excluded from
interpretation because the cache can be class grouped. Raw variance ratios do
not alone establish which BN features cause performance changes.

### Seed 303

`allocation_s5_seed303_four_gpu.sh` runs N/A/C/D × 303, one task per GPU, using
the unchanged production runner. Batch 16, 15 epochs, schedule, initialization,
selection rule and factored scans remain the same. Each task first passes the
existing full-path smoke. It shares its own compilation cache with training.
Outputs are new under `s5-timescale-seed303/`.

### Submit the two jobs

From a checkout pinned to the follow-up commit on `pgi14-head`:

```bash
EXPECTED_COMMIT=<full-follow-up-sha> bash bin/run_experiments/submit_s5_followup.sh
```

The helper submits two independent jobs on `pgi14-gpu1`, partition `pgi14`:
four RTX4090 GPUs, 16 CPUs, 100 GB and a four-hour limit each. The jobs can run
alongside each other when all eight GPUs are available. Each creates a separate
node-local checkout and uses the existing environment read-only. It does not
submit history training. `DRY_RUN=1` prints the allocation plan without submitting.

Reports:

- Seed 303: `/Users/durso/s5-runs/s5-timescale-seed303/<timestamp>/`.
- Selected audit: `/Users/durso/s5-runs/s5-selected-audit/job-<id>/`.
- Exact results/modes: `checkpoints/selected_checkpoint_audit.json`.
- Uniform BN sensitivity: `bn/selected_bn_summary.json`.
- Job logs: `/Users/durso/s5-runs/s5-followup-<job-name>-<id>.log`.

No held-out test split is read by either job.

## History prototype: principle and S5 mapping

Replace instantaneous gradient differentiation with a finite-response history:

\[
\ell\dot q=g-q,\qquad
m\ddot x+c\dot x=-g-\frac{\tau}{\ell}(g-q).
\]

For a static differentiable strongly convex objective, an energy certificate
can use the Fenchel gap:

\[
V=f(x)-f_*+\tfrac m2\|\dot x\|^2
 +\frac{\tau}{\ell}\bigl[f(x)+f^*(q)-q^\top x\bigr].
\]

Writing \(y=\nabla f^*(q)\), differentiation gives

\[
\dot V=-c\|\dot x\|^2-
\frac{\tau}{\ell^2}\langle x-y,\nabla f(x)-\nabla f(y)\rangle\le0.
\]

The conjugate is only part of the proof; the update does not compute it. This
does not prove acceleration, novelty, stochastic robustness, or stability for
S5's rotating complex modes. The memory idea is related to filtered derivative
feedback; no new-optimizer discovery is claimed from this implementation alone.

For S5 use token units and the continuous generator, not `1-lambda_bar`:
\(\kappa=-\Lambda\Delta\), \(b=\Delta\widetilde B u\), \(r=\kappa s-b\).
Choose \(c=1,m=\tau\). Then

\[
\tau\ddot s+\dot s=-r-\frac{\tau}{\ell}(r-q),\qquad
\ell\dot q=r-q.
\]

The continuous transfer is

\[
H(p)=\frac{1+(\tau+\ell)p}
 {\tau\ell p^3+(\tau+\ell)p^2+[1+\kappa(\tau+\ell)]p+\kappa}.
\]

Its zero-history limit as \(\ell\to0\) recovers native continuous S5.
Continuous DC gain and low-frequency delay match native; sampled delay is not
asserted to match exactly. Exchanging the two horizons leaves this transfer
unchanged, so their physical roles are not separately identifiable from it.

The initial research constructor fixes
\(\tau_i=\ell_i=0.10/|\kappa_i|\), recomputed differentiably as the native
parameters learn. Equal normalized horizons are an additional modeling choice,
not a consequence of the energy certificate. There are no extra learned
parameters, but there are three complex states per native mode. A future claim
about memory mechanisms requires an appropriate extra-state control.

`s5/gradient_history.py` computes the exact ZOH realization via a 4×4 augmented
matrix exponential in production complex64, using scaled states
\((s,\tau\dot s,\tau q)\). A stationary divide-and-conquer scan reuses powers
of the constant transition, avoiding per-token matrix-matrix products. Sequential
and generic associative implementations remain independent references. No
eigendecomposition or pole separation occurs in the differentiable scan.

`history_alpha=0` is a separately implemented native control, not numerical
evaluation of the singular three-state formula. Positive-alpha stability must
be monitored during training; the constructor does not silently clip its poles.

### Verification and remaining gate

Local tests cover independent SciPy exponentials, sequential/parallel outputs
and gradients, repeated poles, odd sequence lengths, reverse scans, finite
differences, the native limit, exact backbone equality and 16,000-token float32
execution. Integration tests cover native optimizer initialization, scoped
registration, synthetic finite training, checkpoint restoration and actual
three-state pole telemetry.

An independent true-complex64 check of all 1,152 previously diagnosed initialized
modes found all stable (maximum radius 0.9995210), with maximum transition-matrix
absolute error 1.58e-6 versus independent float64 exponentials. These are
initializations, not trained checkpoints.

A local CPU benchmark (one bidirectional production-width layer, 16,000 tokens,
batch 1, forward plus all-parameter backward, seven timed repetitions) measured
15.06 ms for native, 35.34 ms for C/factored and 43.31 ms for history/stationary.
This includes history's true complex64 exponential. It is a single-layer CPU
measurement, not a GPU training-time estimate.

The separate `experiments.s5_gradient_history.runner` reuses the existing
training protocol in a scoped registration. Full training requires a successful
GPU smoke report matching source hashes, commit, data manifest, seed and settings.
CPU checks cannot pass that gate. Full six-layer batch-16 GPU memory, runtime,
learned stability and accuracy are unverified. The follow-up submission helper
deliberately leaves this pilot unsubmitted while the current comparison closes.

For a later GPU smoke in its own allocation, from the pinned checkout:

```bash
JAX_ENABLE_X64=0 JAX_PLATFORMS=cuda \
  "$PROSPECTIVE_VENV/bin/python" -u -m experiments.s5_gradient_history.runner \
  --data-cache "$S5_THREE_ARM_DATA" --seed 301 --smoke --out <new-smoke-directory>
```

The eventual pilot uses seeds 301/302 and compares against those same seeds,
separately from the unchanged seed-303 replication. No accuracy gain is assumed.

## Queued history pilot

`submit_s5_history_pilot.sh` adds only the history pilot; it does not resubmit
the two follow-up jobs. Run it from the pinned follow-up checkout on the head
node with `EXPECTED_COMMIT` and `HISTORY_AFTER_JOB` set. The latter is the
selected-checkpoint audit job, currently 76860. The helper requests two RTX4090
GPUs, 16 CPUs, 100 GB and six hours on pgi14-gpu1. It waits for that audit job
to complete successfully. If its final eight-task audit summary already exists,
the helper verifies that report and its manifest hash instead of relying on a
possibly expired Slurm job id.

The pilot uses its own node-local checkout. Its output is
`/Users/durso/s5-runs/s5-history-pilot/job-<id>/`, with external task logs under
`logs/`, smoke artifacts under `smoke/gradient_history_s5/<seed>/`, and fresh
training outputs under `gradient_history_s5/<seed>/`. Both seeds share only
their own smoke/training compilation cache. Preflight checks the full native
parameter tree and optimizer initialization, in production float32, for both
seeds. Both full-size GPU smokes must pass before either training begins.

Smoke gates check finite production loss/gradients/state, the exact two smoke
updates, finite validation metrics, checkpoint restoration, telemetry agreement,
the correct scan and protocol identity, and initial/post-smoke pole diagnostics.
A failure preserves the diagnostic outputs and blocks training; batch size,
precision, dynamics and stabilizers are not changed to make a failed gate pass.
Finite pole instability arising later during training remains logged under the
existing protocol, while a nonfinite training step fails the task.

### Selected C301 stability finding

The completed audit reports C301 layer 3, mode 28 at radius
1.0000356246235906 in float64 and 1.0000356435775757 in production float32.
This exceeds the recorded 1e-5 boundary tolerance; it is a genuine learned
instability, not merely a disagreement between precisions. The isolated pole's
homogeneous amplitude grows by about 1.768 over 16,000 tokens (doubling time
about 19,457 tokens). This is not the full network's output gain: input/readout
coupling, other modes, nonlinearities and sequence resets matter. The result
can remain finite and accurate on the benchmark while lacking asymptotic
stability. Both selected D checkpoints passed the strict mode check, which
does not prove that D or the history family is stable throughout training.
