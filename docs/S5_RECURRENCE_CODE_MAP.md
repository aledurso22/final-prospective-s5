# S5 recurrence code map

Native S5 is the reference implementation. The native transition, scan and
output path remain in `s5/ssm.py`; the two prospective mathematical changes are
centralized in `s5/three_arm_recurrences.py` and are reused by the prospective
modules.

The pinned upstream native reference is `lindermanlab/S5` commit
`3c18fdb6b06414da35e77b94b9cd855f6a95ef17` (`upstream/main` in this clone).
The upstream-to-local native diff is intentionally limited to modern JAX type
annotations in `s5/ssm.py`; the native discretization, scan, and model path are
numerically audited by `tests/test_upstream_native_identity.py`. The historical
project lineage also contains old generalized CLI/wrapper additions in
`run_train.py`; the three-arm raw-audio harness does not call those paths.

| Scientific method | Mathematical recurrence | State carried between tokens | Trainable parameters | Exact source file and function | Exact lines changed relative to native S5 | Reduction/boundary identities | Expected overhead |
|---|---|---|---|---|---|---|---|
| Native matched S5 | `s_next = Lambda_bar*s + B_bar*x` | One complex diagonal S5 state per mode | Native `Lambda`, `B`, `C`, `D`, `log_step` | `s5/ssm.py:apply_ssm`, `s5/ssm.py:S5SSM.__call__` | Reference: no recurrence modification; native source lines 50-81 and 232-250 | `s5/three_arm_recurrences.py:13-15` is the one-step equivalent | Baseline scan and readout cost |
| Zucchet prospective S5 recurrence | `(I-T a)s_dot = a s + b x + T b x_dot`; exact ZOH after derivative-free realization | One ordinary S5 history state `u`; observed `s=u+d_x x` | Native parameters plus positive prospective `T` (`gp_response_raw`) | `s5/three_arm_recurrences.py:18-36`; adapter change `s5/gp_coefficients.py:58,171-173` | New recurrence lines `18-36`; adapter replacement relative to native generalized coefficient code at `s5/gp_coefficients.py:171-175` | Exact generalized boundary `M=0, gamma=1`; `T=0` reduces to native coefficients | One extra coefficient vector and one extra tied input feedthrough; same state order |
| Generalized prospective S5 recurrence `(M,gamma,T)` | `M s_ddot + (gamma-Ta) s_dot - a s = b x + T b x_dot`; exact augmented-matrix ZOH | Two-component state `(s,w)`, `w=M s_dot-T(a s+b x)` | Native parameters plus positive `T` and positive `M`/mass ratio (`so_response_raw`, `so_mu_ratio_raw`); `gamma=1` is fixed by protocol | `s5/three_arm_recurrences.py:39-76`; adapter change `s5/gp_second_order.py:50,85-96` | New recurrence lines `39-76`; adapter replacement relative to native second-order code at `s5/gp_second_order.py:85-102` | Mathematical `M=0` boundary is the Zucchet recurrence; production tests verify convergence as `M` tends to zero | Two state components, dense 2x2 block exponential per mode, and two learned recurrence leaves per mode |

The line references above are intentionally function-level rather than a claim
that the native file was edited. To inspect the actual patch:

```bash
git diff origin/stable-generalized-prospective-s5...HEAD -- \
  s5/three_arm_recurrences.py s5/gp_coefficients.py s5/gp_second_order.py \
  experiments/s5_three_arm_full/runner.py
```

Native project base commit:

```text
origin/stable-generalized-prospective-s5 @ 727ba34a3795dfd897de6b12bf0cf3d3feeda785
```

Upstream S5 to local-native audit:

```bash
git diff 3c18fdb6b06414da35e77b94b9cd855f6a95ef17...HEAD -- \
  s5/ssm.py run_train.py s5/train.py s5/train_helpers.py
```

Local-native to recurrence audit:

```bash
git diff origin/stable-generalized-prospective-s5...HEAD -- \
  s5/three_arm_recurrences.py s5/gp_coefficients.py s5/gp_second_order.py \
  s5/gp_ssm.py experiments/s5_three_arm_full
```

For this experiment, intentional runtime differences from the pinned upstream
Speech Commands path are the ten-class decoder, the official-list raw-audio
cache, and the two prospective recurrence selections. The native arm retains
the upstream recurrence equations. No other scientific arm is present.

The array and finalizer are inspectable without invoking the training runner:

```bash
sed -n '1,220p' bin/slurm/s5_three_arm_full_array.sbatch
sed -n '1,220p' bin/slurm/s5_three_arm_full_finalize.sbatch
git diff --no-ext-diff 2ee2fa8040611ca8dd552b9cd9c45d077386d3c1..HEAD -- \
  s5/three_arm_recurrences.py s5/gp_coefficients.py s5/gp_second_order.py \
  s5/gp_ssm.py experiments/s5_three_arm_full
```

Compact transition summary:

```text
native:      (Lambda_bar, B_bar, s, x) -> Lambda_bar*s + B_bar*x
prospective: (a, b, T, u, x) -> exp(a/(1-Ta))*u + phi1(...)*b/(1-Ta)^2*x
             observed state = u + T*b/(1-Ta)*x
generalized: (a, b, M, gamma, T, (s,w), x)
             -> exp(A(M,gamma,T))*[s,w] + ZOH(B(M,gamma,T))*x
             output = first component of the new two-state carry
```
