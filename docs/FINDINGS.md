# Findings

Defects and hardening items discovered during gate work. A finding is recorded
here whether or not it blocks the gate it was found under.

| ID | Date | Severity | Area | Status |
|---|---|---|---|---|
| `F-001` | 2026-09-02 | hardening | `s5/prospective.py`, fixed `alpha=0` | OPEN - patch proposed, not implemented |

---

## `F-001` - fixed `alpha=0` is not identity-preserving for Inf/NaN

**Status: OPEN.** Recorded as a real implementation defect / hardening finding.
It is explicitly **not** a passed identity case.

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

### Disposition

A hardening patch introducing a static-zero bypass has been **proposed and not
implemented**, pending review. Theory prefers bypass on exactly these grounds
(`0*NaN`, stale cache, RNG/state hazards). No architecture change has been made.
