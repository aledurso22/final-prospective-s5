# `mechanism/` - three DIFFERENT prospective mechanisms, kept apart on purpose

If you are opening this repository for the first time, these are **not** the
same idea, and they live in separate files so they cannot be confused.

| File | Mechanism | Where the prospective term acts | Branch that owns it |
|---|---|---|---|
| `readout_lead_control.py` | **readout PC** - `b_pc[k] = b[k] + alpha(b[k]-b[k-1])` | AFTER the readout, on the block output. The state recurrence is untouched. | `prospective-lead` (validated: G2a PASSED) |
| `full_state_pc.py` | **professor Idea 1** - full state PC | INSIDE the state dynamics, on **every** mode | `professor-state-pc` |
| `projected_state_pc.py` | **professor Idea 2** - projected/modal state PC | INSIDE the state dynamics, on the **tracking modes only** (`P_T`) | `professor-state-pc` |
| `static_equilibrium_bypass.py` | hostile control - no dynamics at all on tracking modes | n/a | `professor-state-pc` |
| `plain_ssm.py` | the ordinary baseline | nowhere | `main` |

Nothing in this package touches S5. It is a standalone synthetic study on a
2-mode diagonal linear system, integrated exactly, so every claim is checked
against a closed-form prediction rather than against a training curve.

Run it:

```bash
python experiments/run_professor_mechanism.py
python -m pytest tests/test_mechanism.py -q
```

Findings are written up in [`docs/PROFESSOR_MECHANISM.md`](../docs/PROFESSOR_MECHANISM.md).
