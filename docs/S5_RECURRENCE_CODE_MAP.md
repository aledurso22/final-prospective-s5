# S5 recurrence code map

Native S5 is the reference implementation. The native transition, scan and
output path remain in `s5/ssm.py`; the two prospective mathematical changes are
centralized in `s5/three_arm_recurrences.py` and are reused by the prospective
modules.

| Scientific method | Mathematical recurrence | State carried between tokens | Trainable parameters | Exact source file and function | Exact lines changed relative to native S5 | Reduction/boundary identities | Expected overhead |
|---|---|---|---|---|---|---|---|
| Native matched S5 | `s_next = Lambda_bar*s + B_bar*x` | One complex diagonal S5 state per mode | Native `Lambda`, `B`, `C`, `D`, `log_step` | `s5/ssm.py:apply_ssm`, `s5/ssm.py:S5SSM.__call__` | Reference: no recurrence modification | `s5/three_arm_recurrences.py:native_matched_s5_transition` is the one-step equivalent | Baseline scan and readout cost |
| Zucchet prospective S5 recurrence | `(I-T a)s_dot = a s + b x + T b x_dot`; exact ZOH after derivative-free realization | One ordinary S5 history state `u`; observed `s=u+d_x x` | Native parameters plus positive prospective `T` (`gp_response_raw`) | `s5/three_arm_recurrences.py:zucchet_prospective_s5_coefficients`, `zucchet_prospective_s5_transition`; production adapter `s5/gp_ssm.py:GPSSM` | Only coefficient construction/readout feedthrough differs; scan/output remain shared in `gp_ssm.py` | Exact generalized boundary `M=0, gamma=1`; `T=0` reduces to native coefficients | One extra coefficient vector and one extra tied input feedthrough; same state order |
| Generalized prospective S5 recurrence `(M,gamma,T)` | `M s_ddot + (gamma-Ta) s_dot - a s = b x + T b x_dot`; exact augmented-matrix ZOH | Two-component state `(s,w)`, `w=M s_dot-T(a s+b x)` | Native parameters plus positive `T` and positive `M`/mass ratio (`so_response_raw`, `so_mu_ratio_raw`); `gamma=1` is fixed by protocol | `s5/three_arm_recurrences.py:generalized_prospective_s5_generator`, `generalized_prospective_s5_zoh`, `generalized_prospective_s5_transition`; production adapter `s5/gp_second_order.py:SecondOrderGPSSM` | Only the recurrence generator, block ZOH and two-state scan differ; native S5 projection and readout conventions remain unchanged | Mathematical `M=0` boundary is the Zucchet recurrence; production tests verify convergence as `M` tends to zero | Two state components, dense 2x2 block exponential per mode, and two learned recurrence leaves per mode |

The line references above are intentionally function-level rather than a claim
that the native file was edited. To inspect the actual patch:

```bash
git diff origin/stable-generalized-prospective-s5...HEAD -- \
  s5/three_arm_recurrences.py s5/gp_coefficients.py s5/gp_second_order.py \
  experiments/s5_three_arm_full/runner.py
```

Native upstream/base commit:

```text
origin/stable-generalized-prospective-s5 @ 727ba34a3795dfd897de6b12bf0cf3d3feeda785
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
