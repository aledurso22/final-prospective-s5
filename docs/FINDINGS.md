# Findings

Defects and hardening items discovered during gate work. A finding is recorded
here whether or not it blocks the gate it was found under.

| ID | Date | Severity | Area | Status |
|---|---|---|---|---|
| `F-002` | 2026-09-03 | investigation | training loop, full GPU runs | OPEN - one-epoch training is not bit-reproducible on RTX 3090 |
| `F-001` | 2026-09-02 | hardening | `s5/prospective.py`, fixed `alpha=0` | **CLOSED** — GPU-VERIFIED (`E1-002`, `0316e3c`, SLURM 63311, 87/87) |

---

## `F-001` - fixed `alpha=0` is not identity-preserving for Inf/NaN

**Status: CLOSED — GPU-VERIFIED.** Fixed by `d60252f`, verified on the RTX 3090
as `E1-002` (commit `0316e3c`, SLURM 63311, 87 passed, `exit=0`). Recorded as a
real implementation defect / hardening finding; the original behaviour was
explicitly **not** a passed identity case.

### Scope

Theory scoped the mathematical identity statement
`F_PC-S5,alpha=0(u; theta) == F_S5(u; theta)` to **finite** values. This
finding is therefore separated from the identity gate:

- **G2a-core** - exact identity for finite valid model tensors. Zero tolerance.
  Unaffected by this finding.
- **G2a-robustness** - this finding. Non-finite inputs. Does not gate G2a-core.

### Defect

The operator computes `x + alpha * (x - previous)`. At `alpha = 0` the
correction term is `0.0 * (x - previous)`, which is exactly `0.0` only when
`x - previous` is finite. For non-finite values `0.0 * (+/-inf) = NaN` and
`0.0 * NaN = NaN`.

Two consequences, both measured on the current implementation:

1. the non-finite position becomes `NaN` instead of propagating `+/-inf`; and
2. **the following timestep is also corrupted**, because `previous[t+1]` is the
   non-finite value, so `0.0 * (finite - inf) = NaN`. Plain S5 leaves that
   position clean.

### Evidence

Contaminating a single element at `t=2`, `alpha=0`, remaining values finite:

| Input at t=2 | plain S5 t=2 | `alpha=0` PC t=2 | plain S5 **t=3** | `alpha=0` PC **t=3** |
|---|---|---|---|---|
| `inf` | `inf` | `NaN` | `10.0` | **`NaN`** |
| `-inf` | `-inf` | `NaN` | `10.0` | **`NaN`** |
| `NaN` | `NaN` | `NaN` | `10.0` | **`NaN`** |

Asserted by `tests/test_g2a_identity.py::
test_alpha_zero_identity_holds_only_for_finite_inputs`, so any change in
behaviour is deliberate rather than silent.

### Practical impact

Low for healthy training, where preactivations stay finite - which is why
G2a-core is unaffected. It matters when a run diverges: plain S5 and
`alpha=0` PC-S5 would then differ in *where* corruption appears, and the
prospective path corrupts one timestep more than the baseline. That would make
a diverged paired comparison misleading rather than merely useless.

### Disposition - PATCHED

A static-zero bypass was approved and implemented in `ProspectiveLead`
(`prospective-lead` `d60252f`):

- `apply_parallel` returns the input sequence directly at fixed `alpha == 0`;
- `step` returns `(token, cache)`, so the cache is neither read nor written;
- the module and all wiring still execute, preserving the end-to-end code-path
  control;
- `alpha > 0` behaviour, the S5 recurrence, the scan, the layers and the CLI
  are untouched.

The guard is Python-level by necessity - a traced `alpha` cannot be branched on
inside `jit` - so it lives on `ProspectiveLead`, where `alpha` is a static
dataclass field, and not on the module-level functions. `bool` is excluded so
`False` cannot masquerade as a fixed zero.

**Deliberate scope limit.** The module-level `apply_parallel`/`step` functions
are *not* bypassed and still compute `x + alpha*(x - previous)`. They remain
the tested definition of the operator, and
`test_f001_hardening.py::test_2_free_function_still_shows_the_arithmetic_hazard`
asserts the un-bypassed arithmetic still exhibits the Inf/NaN behaviour. The
split between the two layers is therefore explicit and tested, not accidental.

**Known consequence.** After this patch, `alpha=0` through the module no longer
exercises the correction arithmetic, so G2a-core is trivially satisfied at the
module level and is weaker as an arithmetic control than it was. The arithmetic
control is retained via the free functions in `tests/test_g2a_identity.py`.

24 new tests; full suite 87 passed on CPU and **87 passed on the RTX 3090**
(`E1-002`). F-001 is closed.


---

## `F-002` - one-epoch training is not bit-reproducible on the RTX 3090

**Status: OPEN.** Discovered during G2b (`E2-003`).

Two runs of **plain S5** at the same commit (`0316e3c`), same flags
(`--prospective_mode=off`), same seed (1919), same node and same environment
produced different results:

| Metric | `C2` | `C2R` | difference |
|---|---|---|---|
| Training Loss | 1.4051856994628906 | 1.405187726020813 | 2.027e-06 |
| Val loss | 0.36359065771102905 | 0.36366006731987 | 6.941e-05 |
| Test Loss | 0.33723869919776917 | 0.3372980058193207 | 5.931e-05 |

No prospective code executes in either run, so this is a property of the
training loop and not of the prospective operator.

Data ordering is controlled (`torch.Generator` seeded from `--jax_seed`,
`num_workers=0`) and dropout is disabled (`--p_dropout=0.0`), so the remaining
candidates are non-deterministic GPU reductions/atomics and autotuned kernel
selection under XLA.

**Consequence.** Until this is characterized or removed, no paired full-run
training comparison - including every future `alpha > 0` comparison - can be
interpreted at a resolution finer than this noise. It is therefore a
prerequisite for the sweep (G4), not merely a curiosity.

**Next step.** Test `XLA_FLAGS=--xla_gpu_deterministic_ops=true`; if runs
become bit-reproducible, paired comparisons become exact. Otherwise the noise
distribution must be characterized with repeats and every comparison judged
against it.
