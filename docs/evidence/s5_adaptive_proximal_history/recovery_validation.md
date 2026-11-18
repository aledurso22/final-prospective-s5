# Adaptive P recovery validation — 2026-11-18

Scope: recover the four interrupted adaptive/frozen P tasks from source commit
4372acd76c9f53e8d83d4f929d5bbd4745ace3ad, without editing that implementation.

Local runtime: Python 3.14.4, JAX 0.11.1, CPU, JAX_ENABLE_X64=0.

Combined command:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=0 \
  /Users/alessandrodurso/Documents/final-prospective-s5/.venv/bin/python -m pytest -q \
  tests/test_adaptive_proximal_history_resume.py \
  tests/test_adaptive_history_resume_launcher.py \
  tests/test_history_resume.py \
  tests/test_adaptive_proximal_history.py \
  tests/test_adaptive_proximal_history_runner.py \
  tests/test_adaptive_proximal_history_audit.py \
  tests/test_adaptive_proximal_history_launcher.py
```

Result: **85 passed in 75.58 seconds**. Of these, 10 tests cover the new recovery
module and 27 cover the new launcher's actual shell orchestration.

The real tiny training tests cover both adaptive and frozen-control arms.
After a simulated interruption during a partial epoch, recovered checkpoint
bytes, training records and validation records match uninterrupted training.
This includes optimizer state, BatchNorm and learned horizon parameters. An
earlier best checkpoint remains eligible; completed output is accepted by the
existing adaptive audit. The preflight-only path executes reproduction and
discarded updates without starting training or copying completed checkpoints.
Bad reproduction, code/protocol bindings, incomplete source state, inappropriate
output paths and CPU-only production execution are refused.

The launcher tests run the shell with fake scheduler/GPU-worker processes to
check stage order, device assignment, exclusivity evidence, reservation of all
eight GPUs, source preservation, and global preflight gating. A failed audit
preserves successful TRAINED states and output files. These are orchestration
tests, not real Slurm allocations or GPU execution.

Both new scripts also pass `bash -n`. Original model, optimizer, training-loop,
scan, and audit source files remain unchanged. Recovery validates their hashes
against the source commit at runtime.

The actual cluster checkpoints are not available locally. Their integrity and
GPU reproduction are not claimed here. The submitted job requires exact saved
validation accuracy and cross-entropy agreement within 1e-6, plus two finite
scheduled training updates, for all four restored states before continuation.
It reports whether reproduction is also bitwise identical. Source hashes are
checked throughout, and disposable preflight updates never become the resumed
state.
