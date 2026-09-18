# S5 recurrence code map

Native `s5/ssm.py` is the unchanged reference path. The two finite-difference
implementations share only coefficient/scan helpers; the production factory is
`s5/three_arm_factory.py`.

| Scientific method | Equation function | State | Extra parameters | Source |
|---|---|---|---|---|
| Native S5 | `s5.ssm.apply_ssm` | one complex S5 state per mode | none | `s5/ssm.py` |
| Zucchet prospective dynamics — finite-difference realization | `s5.discrete_recurrence.zucchet_coefficients` | `[s_t,s_{t-1}]`, plus causal previous input in full-sequence preprocessing | `prospective_T_raw` | `s5/prospective_ssm.py` |
| generalized prospective dynamics `(M,gamma,T)` — finite-difference realization | `s5.discrete_recurrence.generalized_coefficients` | `[s_t,s_{t-1}]`, plus causal previous input in full-sequence preprocessing | `generalized_T_raw`, `generalized_rho_raw`, `generalized_gamma_raw` | `s5/generalized_prospective_ssm.py` |

Both prospective modules obtain \((\bar A,\bar B)\) through the native ZOH
discretization, construct the declared target map, and call
`scan_companion`. The only recurrence change is the companion coefficient and
two-tap input contribution. Bidirectional reverse scans reverse token time
before applying the same delayed-input convention.

The generalized parameterization is \(T=softplus(T_raw)\),
\(\gamma=softplus(\gamma_raw)\), and \(M=\rho\gamma T\), with bounded
\(\rho\). It is not an independently projected mass. The companion carries
twice the native recurrent state; the full-sequence scan does not carry an
additional recurrent input state.

Useful inspection commands:

```bash
git diff <native-commit>...HEAD -- s5/ssm.py s5/prospective_ssm.py \
  s5/generalized_prospective_ssm.py s5/discrete_recurrence.py \
  s5/three_arm_factory.py
git diff --stat <native-commit>...HEAD
```

The training runner imports only the three factory constructors and records the
scientific name separately from each code identifier. No historical `gp_*`
module is imported by production.
