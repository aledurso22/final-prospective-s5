# S5 recurrence code map

Native S5 is the reference implementation. The native transition, scan and
output path remain in `s5/ssm.py`; the two production prospective mathematical
changes are isolated in `s5/prospective_ssm.py` and
`s5/generalized_prospective_ssm.py`. `s5/three_arm_factory.py` is the only
production constructor map. Historical `gp_*` modules remain available for
legacy tests and reports but are not imported by the production runner.

The pinned upstream native reference is `lindermanlab/S5` commit
`3c18fdb6b06414da35e77b94b9cd855f6a95ef17` (`upstream/main` in this clone).
The upstream-to-local native diff is intentionally limited to modern JAX type
annotations in `s5/ssm.py`; the native discretization, scan, and model path are
numerically audited by `tests/test_upstream_native_identity.py`. The historical
project lineage also contains old generalized CLI/wrapper additions in
`run_train.py`; the three-arm raw-audio harness does not call those paths.

| Scientific method | Mathematical recurrence | State carried between tokens | Trainable parameters | Exact source file and function | Exact lines changed relative to native S5 | Reduction/boundary identities | Expected overhead |
|---|---|---|---|---|---|---|---|
| Native S5 recurrence under the shared stability constraint | `s_next = Lambda_bar*s + B_bar*x` | One complex diagonal S5 state per mode | Native `Lambda`, `B`, `C`, `D`, `log_step` | `s5/ssm.py:apply_ssm`, `s5/ssm.py:S5SSM.__call__` | Reference: no recurrence modification | Pinned-upstream identity tests | Baseline scan and readout cost |
| Zucchet prospective S5 recurrence | `(I-T a)s_dot = a s + b x + T b x_dot`; exact ZOH after derivative-free realization | One ordinary S5 history state `u`; observed `s=u+d_x x` | Native parameters plus positive `T` (`prospective_T_raw`) | `s5/prospective_ssm.py:_clocked_coefficients`, `ProspectiveS5SSM.__call__` | Isolated coefficient and scan implementation | `T→0` recovers native coefficients; `M=0,γ=1` is the generalized boundary | One extra parameter vector and tied input feedthrough; same state order |
| Generalized prospective S5 recurrence `(M,gamma,T)` | `M s_ddot + (gamma-Ta) s_dot - a s = b x + T b x_dot`; exact 4x4 augmented ZOH | Two-component state `(s,w)`, `w=M s_dot-T(as+bx)` | Native parameters plus positive `T` and bounded ratio `rho` (`generalized_T_raw`, `generalized_rho_raw`), `M=rho*T`; `gamma=1` fixed | `s5/generalized_prospective_ssm.py:generalized_zoh_coefficients`, `GeneralizedProspectiveS5SSM.__call__` | Isolated 2x2 coefficient and block-scan implementation | `M→0` recovers Zucchet dynamics in the tested boundary regime | Two state components, exact 4x4 exponential, and two recurrence parameter vectors |

The line references above are intentionally function-level rather than a claim
that the native file was edited.

Native project base commit:

```text
origin/stable-generalized-prospective-s5 @ 727ba34a3795dfd897de6b12bf0cf3d3feeda785
```

Upstream S5 to local-native audit:

```bash
git diff 3c18fdb6b06414da35e77b94b9cd855f6a95ef17...HEAD -- \
  s5/ssm.py run_train.py s5/train.py s5/train_helpers.py
```

Authoritative three-arm diffs:

```bash
git diff 3c18fdb6b06414da35e77b94b9cd855f6a95ef17...HEAD -- \
  s5/ssm.py s5/prospective_ssm.py s5/generalized_prospective_ssm.py \
  s5/three_arm_factory.py experiments/s5_three_arm_full/runner.py
git diff 2ee2fa8040611ca8dd552b9cd9c45d077386d3c1..HEAD -- \
  s5/ssm.py s5/prospective_ssm.py s5/generalized_prospective_ssm.py \
  s5/three_arm_factory.py
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
  s5/ssm.py s5/prospective_ssm.py s5/generalized_prospective_ssm.py \
  s5/three_arm_factory.py experiments/s5_three_arm_full/runner.py
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
