# Compile-only memory comparison

Both reports use the same source hashes, float32 production shapes (batch 16,
16,000 tokens, six layers, width 96), and CPU backend. They compile abstract
TrainState and input arguments; they execute no update and read no dataset.

| Implementation | Observable compiler buffer total | Telemetry compiler buffer total |
|---|---:|---:|
| `parallel_reference`: original stationary-scan autodiff | 22.376423 GiB | 22.376422 GiB |
| `parallel`: exact custom adjoint | 13.041095 GiB | 13.041094 GiB |

The total is argument bytes + output bytes + temporary bytes − aliased bytes.
The observed reduction is about 41.7%. This is compiler accounting on CPU,
not GPU peak memory, an allocator measurement, or a guarantee that the 4090
smoke will fit. Live initialization buffers, libraries, cached executables,
allocation policy, and GPU-specific lowering are outside this comparison.

The recorded git commit is the base commit at the time of profiling; the
per-file source hashes identify the tested working implementation, including
the new adjoint. Both reports have identical source hashes. The original
autodiff path remains available explicitly for reproduction.

From the repository root, with the existing environment:

```bash
JAX_ENABLE_X64=0 JAX_PLATFORMS=cpu "$PROSPECTIVE_VENV/bin/python" \
  -m experiments.s5_gradient_history.memory_profile \
  --history-scan parallel_reference --out <new-reference-report.json>

JAX_ENABLE_X64=0 JAX_PLATFORMS=cpu "$PROSPECTIVE_VENV/bin/python" \
  -m experiments.s5_gradient_history.memory_profile \
  --history-scan parallel --out <new-adjoint-report.json>
```

Use `JAX_PLATFORMS=cuda` in the intended single-GPU allocation to obtain its
own compiler report before the actual full-size smoke. Existing output files
are refused.
