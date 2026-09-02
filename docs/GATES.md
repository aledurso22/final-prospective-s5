# Experimental gates

A gate is a pass/fail check with its criterion fixed **before** the run. If a
criterion is not written down in advance it is not a gate, it is a
rationalization.

| Gate | Status |
|---|---|
| G0 - plain-S5 baseline tests on target GPU | PASSED (13/13, `E2-002`) |
| G1 - plain-S5 one-epoch GPU smoke | PASSED (`E2-002`) |
| G2 - prospective `alpha=0` identity control | **DESIGNED, NOT RUN** |
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

## G2a - direct tensor/model identity (primary, authoritative)

This is the gate. It does not involve training, a dataloader, or an optimizer,
so no run-to-run nondeterminism can weaken it.

Existing tests, to be executed on the RTX 3090:

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

`test_real_s5_prospective_alpha_zero_matches_off` passes with **exact**
equality, and the full suite is 44/44.

### Predefined failure interpretation

Written before the run so the outcome cannot be reinterpreted afterwards.

| Observation | Interpretation | Action |
|---|---|---|
| exact equality | G2a PASSED | proceed to G2b |
| differences at or below ~1e-6 relative | **not** an identity failure in the mathematics; adding `x + 0.0*(x-prev)` can change XLA fusion, and different fusion can change reduction order and therefore rounding | investigate and confirm the mechanism, then decide explicitly whether to assert exactness on CPU and a bounded tolerance on GPU. Record the decision. |
| differences above ~1e-6 relative | real defect | stop, do not proceed to G2b |
| differing parameter counts or parameter trees | real defect - the `alpha=0` path is allocating something | stop |

The middle row is a known possibility, not an excuse prepared in advance: if it
occurs, the mechanism must be *demonstrated* (e.g. by showing the same
discrepancy appears when an algebraically-null operation is inserted into plain
S5 alone), not assumed.

## G2b - paired one-epoch training behaviour (secondary, diagnostic)

Same GPU, same host, same commit-pair, same seed, same configuration. This
characterizes end-to-end behaviour; it does **not** define the identity claim.

Prerequisite - establish the run-to-run noise floor on plain S5 first,
otherwise a paired difference cannot be interpreted:

```bash
# on main, twice, same seed
./bin/run_experiments/run_baseline_mnist_smoke.sh   # -> baseline_A.log
./bin/run_experiments/run_baseline_mnist_smoke.sh   # -> baseline_B.log
```

Then the paired run on `prospective-lead`:

```bash
./bin/run_experiments/run_baseline_mnist_smoke.sh \
  --prospective_mode=lead \
  --prospective_alpha=0.0 \
  --prospective_alpha_learned=False \
  --prospective_layers=all
```

Held identical across all runs: GPU and host, `CUDA_VISIBLE_DEVICES`, seed
1919, `bsz=64`, `n_layers=2`, `d_model=64`, `ssm_size_base=64`, `blocks=2`,
`batchnorm=False`, `bidirectional=False`, `p_dropout=0.0`, dataset and data
order, environment variables, `requirements-frozen.txt`.

### Pass criterion

Let `d_AB` be the difference between the two plain-S5 repeats (the noise
floor) and `d_0` the difference between plain S5 and `alpha=0`, on train loss,
val loss, val accuracy, test loss, test accuracy and parameter count.

- Parameter count must be **26,058 in every run**. Any deviation fails outright.
- If `d_AB == 0` (training is bit-reproducible): require `d_0 == 0`. Anything
  else is a finding to investigate, given that G2a asserts exact identity.
- If `d_AB > 0` (training is not bit-reproducible): require `d_0 <= d_AB`,
  i.e. the prospective `alpha=0` run differs from plain S5 by no more than
  plain S5 differs from itself. `d_0` materially larger than `d_AB` is a
  finding, not a pass.

Reporting `d_0` without `d_AB` is meaningless and is not acceptable evidence.

### Evidence label

G2b is **E2 - smoke evidence only**, like `E2-001` and `E2-002`. A single epoch
at one seed characterizes nothing beyond "it runs and behaves the same".

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
