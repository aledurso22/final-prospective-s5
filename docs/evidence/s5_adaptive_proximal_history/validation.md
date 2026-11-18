# Local validation, 2026-11-18

Base commit: acb4d850b7f868ddcfd85da498792c6ff5987749.
Backend: CPU. Python executable used:
/Users/alessandrodurso/Documents/final-prospective-s5/.venv/bin/python.

## Combined production-precision regression

Environment: JAX_PLATFORMS=cpu, JAX_ENABLE_X64=0, PYTHONPATH=.

Command: python -m pytest -q
tests/test_adaptive_proximal_history.py
tests/test_adaptive_proximal_history_runner.py
tests/test_adaptive_proximal_history_audit.py
tests/test_adaptive_proximal_history_launcher.py
tests/test_proximal_history.py
tests/test_proximal_history_runner.py

Result: **66 passed in 40.17s**. This includes 40 new tests and 26 existing P
tests. Synthetic two-update CPU smoke executions verify real optimizer updates
and checkpoint restoration for both arms; they are not full-size cluster runs.

## Mathematical checks in float64

Environment: JAX_PLATFORMS=cpu, JAX_ENABLE_X64=1, PYTHONPATH=.
Command: python -m pytest -q tests/test_adaptive_proximal_history.py
Result: **12 passed in 9.48s**.

These cover exact zero-multiplier block identity with P, initial layer outputs
and shared gradients, independent physical-coordinate ZOH and equilibrium,
finite-difference scale gradients, a complex stability screen, and frozen-scale
gradients. Production training explicitly refuses x64.

## Integration, audit and submission coverage

- Candidate xi updates; frozen-control xi remains exactly zero after actual
  optimizer updates. Common parameter initialization, optimizer grouping and
  synthetic shared updates agree with P.
- Poles and per-epoch diagnostics use actual saved xi, not the original alpha.
- Combined auxiliary kernel energy agrees with dense real B/C kernels in tests
  including cross-mode and cross-pole cancellation. An unresolved decomposition
  withholds the result instead of dropping modes.
- Audit checks selection, source hashes and output separation, and labels its
  xi=0 same-weight counterfactual separately from the trained control.
- Shell orchestration tests execute disposable launchers with fake workers and
  a capturing sbatch stub. They verify node/resource arguments, global smoke
  gating, one assigned GPU per task, fresh output refusal and preservation of
  completed training if only the audit fails. They submit no real job.

## Not verified locally

Full-size CUDA compilation, float32 GPU arithmetic, memory use, throughput,
real-data training and accuracy remain unverified. The cluster launcher runs
four full-size GPU smokes and checks candidate/control behavior before training.
Compiler buffer accounting is not a measured peak-memory benchmark.

No cluster job was launched and no held-out test data was read during this work.
