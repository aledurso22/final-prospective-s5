# Experimental gates

A gate is a pass/fail check with its criterion fixed **before** the run. If a
criterion is not written down in advance it is not a gate, it is a
rationalization.

| Gate | Status |
|---|---|
| G0 - plain-S5 baseline tests on target GPU | PASSED (13/13, `E2-002`) |
| G1 - plain-S5 one-epoch GPU smoke | PASSED (`E2-002`) |
| G2a-core - `alpha=0` exact identity, finite tensors | **PASSED** on RTX 3090 (`E1-001` 63/63; re-passed post-hardening `E1-002` 87/87) |
| G2a-robustness - `alpha=0` under Inf/NaN | **CLOSED** - `F-001` patched and GPU-verified (`E1-002`) |
| G2b - paired one-epoch training diagnostic | **PROTOCOL FROZEN, NOT RUN** |
| G3 - lag metrics instrumentation | not started |
| G4 - sMNIST alpha/placement sweep | not started |

---

# G2 - prospective `alpha=0` identity control

## What is being claimed

With `--prospective_mode=lead --prospective_alpha=0` and
`--prospective_alpha_learned=False`, the prospective branch must compute
**the same function** as plain S5.

This is an architectural claim, provable from the code:

- the operator computes `x + alpha * (x - previous)`; at `alpha=0` the second
  term is `0.0 * finite = 0.0` exactly, and `x + 0.0 == x` exactly in IEEE 754
  for all finite `x`;
- with `prospective_alpha_learned=False` the module calls no `self.param`, so
  the parameter tree and the PRNG consumption are unchanged;
- `--p_dropout=0.0` removes the dropout RNG stream.

The claim is therefore **exact identity of the computed function**, not
"statistically indistinguishable training".

## Scope split (frozen per the theory specification)

**G2a-core** - the authoritative identity gate. The claim
`F_PC-S5,alpha=0(u; theta) == F_S5(u; theta)` is scoped by theory to **finite
valid model tensors**, under identical parameters, input, masks, RNG keys,
optimizer state and precision. Exact, zero tolerance.

**G2a-robustness** - non-finite behaviour. The current `0*(x-prev)`
implementation is **not** identity-preserving for Inf/NaN and contaminates the
following timestep. Recorded as `F-001` in [FINDINGS.md](FINDINGS.md), an open
implementation defect. It is **not** a passed identity case and it does **not**
gate G2a-core.

## G2a-core - direct tensor/model identity (primary, authoritative)

This is the gate. It does not involve training, a dataloader, or an optimizer,
so no run-to-run nondeterminism can weaken it.

### Coverage required, and current gaps

The architectural identity claim is carried by fixed-input / fixed-parameter
functional and gradient tests, **not** by the training comparison. Required
coverage, against what the suite has today:

| # | Required assertion | Status | Test |
|---|---|---|---|
| 1 | exact parameter-tree structure, leaf shapes and total count | **PARTIAL - must extend** | `test_off_mode_reproduces_baseline_layer` compares key paths only (`sorted(flatten_dict(...))`), at layer level, not on the real-S5 model, and never compares shapes or a parameter count |
| 2 | exact fixed-batch **loss** | **MISSING** | logits are compared; no scalar loss is |
| 3 | exact **gradients** w.r.t. the ordinary S5/model parameters | **MISSING** | `test_real_s5_prospective_forward_jit_and_grad` only checks finiteness/presence at `alpha != 0`; no off-vs-`alpha=0` gradient equality exists |
| 4 | exact result of **one optimizer update** from identical params and batch | **MISSING** | no such comparison exists on either branch |
| 5 | parallel-path identity | COVERED | `test_alpha_zero_is_exact_identity` (`atol=0`), `test_off_mode_stack_reproduces_baseline_stack`, `test_real_s5_prospective_alpha_zero_matches_off` |
| 6 | streaming `step` identity and reset/boundary behaviour | **PARTIAL - must extend** | `test_parallel_matches_streaming_with_resets` and `test_module_parallel_matches_streaming_steps` prove parallel == streaming for arbitrary alpha and zero correction at resets, but nothing asserts that at `alpha=0` the streaming path equals the plain-S5 preactivation |

Four new tests and two strengthenings are therefore required before G2a can be
executed as specified. They are **not yet written** - see "Tests to be added".

### Tests to be added (not yet implemented)

- `test_alpha_zero_param_tree_identical`: full `jax.tree_util` comparison of
  the real-S5 model's parameter tree against `mode='off'` - identical key
  paths, identical leaf shapes and dtypes, identical total parameter count
  (26,058 in the smoke configuration).
- `test_alpha_zero_fixed_batch_loss_identical`: same fixed batch and fixed
  parameters, `assert_array_equal` on the scalar loss.
- `test_alpha_zero_gradients_identical`: `jax.grad` of that loss w.r.t. every
  ordinary S5/model parameter, compared leaf-by-leaf with `assert_array_equal`.
  Ordinary parameters only; at `alpha=0` fixed there is no alpha parameter.
- `test_alpha_zero_optimizer_step_identical`: from identical initial
  parameters and one identical batch, one `optax` update, then
  `assert_array_equal` on every updated leaf.
- extend #1 above to the stack and the real-S5 model, not just `SequenceLayer`.
- `test_alpha_zero_streaming_matches_plain_preactivation`: the streaming
  `step` path at `alpha=0`, run token by token, equals the plain-S5 block
  preactivation exactly, including across a reset boundary.

All use `assert_array_equal`. All run on CPU and on the RTX 3090.

### Existing tests, to be executed on the RTX 3090:

| Test | Asserts |
|---|---|
| `test_real_s5_prospective_alpha_zero_matches_off` | on the **real S5SSM**, `BatchClassificationModel` logits at `alpha=0, layers=all` are `assert_array_equal` to `prospective_mode='off'` |
| `test_off_mode_reproduces_baseline_layer` | `off` mode reproduces `main`'s `SequenceLayer` output *and* parameter tree, over 4 activations x prenorm/postnorm |
| `test_off_mode_stack_reproduces_baseline_stack` | whole `StackedEncoderModel`, `alpha=0, layers=all`, equals `off` |
| `test_alpha_zero_is_exact_identity` | operator level, `atol=0` |
| `test_ssm_source_digest_is_unchanged` | `s5/ssm.py` identical to `main` |
| `test_ssm_and_scan_match_baseline_branch` | `git diff main -- s5/ssm.py` is empty |

Command:

```bash
git switch prospective-lead
pytest -q
pytest -q tests/test_prospective_integration.py -k "alpha_zero or baseline or ssm" -v
```

### Pass criterion

**Exact equality is the only PASS.** Every assertion in G2a is `atol=0,
rtol=0` (`assert_array_equal`). There is no tolerance band that counts as a
pass.

### Predefined outcome classification

Written before the run so the outcome cannot be reinterpreted afterwards.
Note there are only three verdicts and only one of them is PASS.

| Observation | Verdict | Action |
|---|---|---|
| exact equality on every G2a assertion | **PASS** | proceed to G2b |
| any non-zero discrepancy, however small | **INVESTIGATE / BLOCKED** | G2 does not pass. Demonstrate the backend/compiler mechanism, then bring it back for an explicit decision. |
| parameter count, shapes or tree structure differ | **FAIL** | stop; the `alpha=0` path is allocating or restructuring something |

A small discrepancy is **not** a pass and must never be relabelled as one.
It is a blocking finding pending explanation.

If the INVESTIGATE branch is taken, the mechanism must be *demonstrated*, not
asserted. A sufficient demonstration inserts an algebraically-null operation
(for example `x + 0.0 * x`) into **plain S5 alone**, on the same hardware, and
shows it reproduces a discrepancy of the same character and magnitude. That
isolates compiler/backend behaviour from the prospective operator. Absent such
a demonstration, the discrepancy is treated as a defect in the operator.

Any decision to accept a bounded tolerance on some backend is a decision for
the project owner, recorded with its rationale - not a judgement this gate is
permitted to make on its own.

## G2b - paired one-epoch alpha=0 training diagnostic (FROZEN, not run)

Secondary and diagnostic. It does **not** carry the architectural identity
claim and **cannot rescue a failed G2a**. G2a-core has already PASSED
(`E1-001`, `E1-002`); G2b characterizes end-to-end training behaviour only.

Evidence label: **E2 - smoke/diagnostic**. One epoch at one seed characterizes
nothing beyond "it runs and behaves the same".

### Required runs - four, ~1 min each

All four in a **single SLURM allocation**, back to back, on the same node, with
provenance captured in the same invocation.

| Run | Branch | Flags added to the baseline script | Purpose |
|---|---|---|---|
| `C1` | `main` | none | branch-level plain-S5 control |
| `C2` | `prospective-lead` | `--prospective_mode=off` | **same-commit** plain-S5 control |
| `C2R` | `prospective-lead` | `--prospective_mode=off` (repeat) | run-to-run noise floor `d_noise` |
| `T`  | `prospective-lead` | `--prospective_mode=lead --prospective_alpha=0.0 --prospective_alpha_learned=False --prospective_layers=all` | the alpha=0 treatment |

Rationale for four rather than two: `C2` vs `T` is the decisive comparison
because it is the **same commit and same binary**, differing only by the flag,
so a difference cannot be attributed to the branch. `C1` vs `C2` separately
confirms the branch itself does not perturb plain S5. `C2R` supplies the noise
floor without which `C2` vs `T` is uninterpretable.

`C1` is *not* a re-use of `E2-002`: it is re-run inside this allocation so all
four runs share hardware state and environment.

### Held identical across all four

Seed `1919`; `bsz=64`, `n_layers=2`, `d_model=64`, `ssm_size_base=64`,
`blocks=2`, `batchnorm=False`, `bidirectional=False`, `p_dropout=0.0`,
`epochs=1`; node `pgi15-gpu3`, one RTX 3090, `CUDA_VISIBLE_DEVICES=0`; the
same venv and `requirements-frozen.txt`; `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
`WANDB_MODE=offline`; the same `cache_dir`.

**Data ordering is controlled, not assumed.** `make_data_loader` builds a
`torch.Generator` seeded with `--jax_seed` and `DataLoader` is constructed with
`num_workers` defaulting to 0 (single process). With seed 1919 fixed, shuffle
order is identical across all four runs.

`p_dropout=0.0` removes the dropout RNG stream, so no stochastic path differs.

### Metrics and artifacts compared

Extracted from each run's log:

- trainable parameter count
- train loss, val loss, val accuracy, test loss, test accuracy
- best val loss / accuracy, best test loss / accuracy

Descriptive only, never part of the criterion: wall clock, peak GPU memory.

Artifacts per run: full stdout log, plus one shared `provenance.txt` and
`requirements-frozen.txt` for the allocation.

### Predeclared pass / stop criterion

Fixed before execution.

**Hard precondition.** Trainable parameters must be exactly **26,058 in all
four runs**. Any deviation is an immediate **STOP** - it would mean the
`alpha=0` path allocates or restructures parameters, contradicting G2a-core.

Define, per metric, over the exact printed values:

```
d_noise    = |metric(C2) - metric(C2R)|      # run-to-run noise floor
d_identity = |metric(C2) - metric(T)|        # the paired comparison
d_branch   = |metric(C1) - metric(C2)|       # branch control
```

| Condition (all metrics) | Verdict |
|---|---|
| `d_identity == 0` | **PASS** |
| `d_noise > 0` and `d_identity <= d_noise` | **PASS** - within the measured noise floor |
| `d_noise == 0` and `d_identity > 0` | **INVESTIGATE** - the training loop is deterministic, so alpha=0 should reproduce exactly given G2a-core passed |
| `d_identity > d_noise` | **INVESTIGATE** |
| parameter count != 26,058 anywhere | **STOP** |

`d_branch` is reported separately. `d_branch > d_noise` is its own finding
about the branch, not about the alpha=0 path, and does not by itself fail G2b.

`d_noise` is **descriptive only**. It characterizes the training loop; it never
redefines what alpha=0 identity means, and a large `d_noise` cannot be used to
excuse a large `d_identity` beyond the explicit rule above.

### Interpretation limits, fixed in advance

- G2b passing adds **no** support for any `alpha > 0` claim.
- G2b failing does **not** retract G2a-core, which is the architectural claim
  and is already established by exact tests. It would be a finding to
  investigate before any paired `alpha > 0` comparison is trusted.
- Full-run nondeterminism, if observed, is recorded as its own separate finding
  about the training loop.

### Out of scope

`alpha > 0`; lag, prefix accuracy or time-to-correct (no instrumentation
exists - G3); throughput or memory comparison; the sweep (G4).

## Open questions - awaiting the theory specification

Left deliberately unresolved. **No implementation change is to be made around
either point until reconciled.** Current behaviour is stated as fact so the
theory response has something concrete to rule on.

1. **Should `alpha=0` traverse the prospective code path, or bypass it?**
   *Current implementation:* it traverses. `prospective_mode='lead',
   alpha=0` constructs the `ProspectiveLead` submodule and executes
   `x + 0.0 * (x - previous)`. Only `prospective_mode='off'` skips the module
   entirely. So `alpha=0` is presently an end-to-end **code-path control**,
   which is the stronger control but also the one exposed to compiler/fusion
   differences. An explicit bypass would guarantee bit-identity trivially, at
   the cost of no longer testing the code path.

2. **What does identity mean for the prospective streaming cache?**
   *Current implementation:* `step(token, cache, alpha, reset)` returns
   `(corrected_token, token)` - the cache always stores the **uncorrected**
   token, and at `alpha=0` the returned token equals the input exactly.
   Plain S5 has no such cache, so "identity" for the cache is undefined
   until theory specifies whether the requirement is (a) output identity only,
   (b) cache contents equal to the plain-S5 block preactivation, or
   (c) the cache being unobservable at `alpha=0`.

## Explicitly out of scope for G2

- any claim about `alpha > 0` improving anything
- lag, prefix accuracy or time-to-correct (no instrumentation exists - G3)
- throughput or memory comparisons
- the sweep (G4)

---

# Appendix - testing the CPU/GPU numerical difference

`E2-001` (CPU) and `E2-002` (GPU) differ in the third decimal place. The cause
is **not established** and no attribution is recorded. If it becomes worth
resolving, a direct test is:

```bash
JAX_DEFAULT_MATMUL_PRECISION=highest ./bin/run_experiments/run_baseline_mnist_smoke.sh
```

If the GPU metrics move toward the CPU values, default matmul precision was a
contributing factor; if they do not, it was not, and kernel/reduction order or
fusion are the remaining candidates. Until such a run exists, the difference
stays unattributed.

This is a curiosity, not a blocker: it does not affect G2, which is a
same-hardware comparison throughout.
