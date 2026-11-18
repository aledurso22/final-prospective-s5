# Proximal inertial S5: candidate, exact first-order control, trained-P audit

Date: 2026-11-18. Implementation and local verification. **No training has run.**
Theory and motivation: `docs/S5_PROXIMAL_INERTIA_HANDOFF.md`.

## The two new arms

Per complex mode, in token time, with kappa = -Lambda*Delta and b = Delta*B~*u:

    m s'' + s' = -r_mu - tau d/dt r_mu,     r_mu = (kappa s - b)/(1 + mu kappa)
    tau = 0.1/|kappa|,   mu = tau/2,   m = 1.5 tau   (candidate, proximal_inertial_s5)
                                       m = tau       (control,   proximal_inertial_control_s5)

The ratios are fixed. tau, mu and m follow Lambda and Delta differentiably, and
no parameter is added: both arms have exactly the native parameter tree, native
initialization and native optimizer groups (asserted in tests and again in the
cluster pre-flight).

**Implementation** (`s5/proximal_inertia.py`):
- Derivative-free coordinates w = m s' + tau r_mu:
  `d/dt[s,w] = [[-tau k/m, 1/m], [(tau/m-1)k, -1/m]][s,w] + (1/d)[tau/m, 1-tau/m] b`,
  with d = 1+mu kappa and k = kappa/d. w and s are continuous across input jumps, so
  the input-derivative impulse is kept exactly.
- Exact physical-input ZOH from one augmented 3x3 exponential per mode.
- No eigendecomposition in the differentiable path. The old `gp_second_order`
  constructor (which restricts m/tau < 1) is not used.
- Width-2 stationary block scan with an explicit recomputing adjoint. This is a width-generic copy of the
  P/H scan; `s5/gradient_history.py` is untouched, so P and H are unchanged.
- Native position readout, conjugate factor, bidirectional zero starts and D skip are kept.
- Training goes through the existing smoke-gated history runner, with registration scoped to the process
  (`experiments/s5_proximal_inertia/runner.py`). Batch 16, 15 epochs, split,
  selection rule and outer optimizer are inherited unchanged.

## Verification (local CPU)

`tests/test_proximal_inertia.py`: 17 tests pass in float64 and 12 in float32. The other 5 are float64-only.
- **Control is exactly first-order proximal flow.** The w row decouples bit-exactly
  (A_ws = 0, b_w = 0), and w stays exactly zero over a scan. s matches the closed form
  s' = -r_mu ZOH recurrence.
- **Native limit.** mass_ratio=1, proximal_ratio=0 gives exp(-kappa), (1-e^-kappa)/kappa for any
  alpha. A full S5 layer then matches the native `S5SSM` output on identical parameters.
  The candidate converges to native as alpha -> 0, at first order.
- **Independent reference.** The exponential agrees with a separate construction in
  physical (s, s') coordinates with explicit velocity jumps at token boundaries, to 1e-9.
  Dropping the jump gives a materially different response.
- **Transfer and cancellation.** H(p) = (1+tau p)/[(1+mu kappa)(m p^2+p) + kappa(1+tau p)]
  holds to 1e-9. The pole at -1/tau cancels exactly at m = tau and not at m = 1.5 tau.
- **Sharp stability bound.** All continuous poles are in the left half-plane over 4,001 angles x 4
  magnitudes of kappa, and the ZOH radius is < 1. The handoff counterexample
  (mu = 0.4 < m - tau) is unstable, and the same kappa at mu = 0.5 is stable.
- **Gradients.** The custom adjoint equals ordinary autodiff on all parameters and is
  finite. It matches central finite differences in float64.
- **Scan agreement.** parallel vs parallel_reference vs associative vs sequential, forward and
  reverse, at 16,000 tokens.

`tests/test_proximal_inertia_runner.py` (8 + launcher dry-run, float32) checks:
- scoped registration and its restoration;
- that the control's protocol identity differs from the candidate's only in arm and mass_ratio;
- identical native parameters and optimizer state;
- real CPU smokes through the CLI;
- that training refuses a CPU smoke;
- compile and measured memory profiles;
- the launcher dry run, and its refusal of the wrong GPU count.

`tests/test_block_audit.py` (5) checks:
- that P (3x3) and both new arms (2x2) audit end to end;
- the control's auxiliary residue < 1e-9 (exact cancellation), while the candidate's is released;
- that audit impulses equal the production scan;
- the selection-rule and configuration refusals.

The existing P, H, memory-audit and N/A/C/D integration suites still pass (127
float32 + 31 float64 tests together).

**Not verified locally:** full-size GPU memory, step time and GPU float32 arithmetic.
Each cluster task measures them on its own RTX 4090 before its smoke, and trains only
if that smoke passes.

## Initialization evidence (production float32 constructor + float64 reference)

`docs/evidence/s5_proximal_inertia/init_audit_*`: reconstructed native float32
initialization, seeds 301/302, 384 modes each, 16,000-lag B/C-coupled kernels.

| Arm | float32 stable / boundary / unstable | max abs(z) | dominant decay / native (median) | auxiliary decay, median of per-layer median (p95) |
|---|---|---|---|---|
| candidate, m=1.5 tau | 384 / 0 / 0 per seed | 0.99950 | **1.0087** | 4.0–4.2 (78–91) samples |
| control, m=tau | 384 / 0 / 0 per seed | 0.99945 | **0.7563** | unexcited (residue < 1e-9) |

The candidate reproduces the handoff screen (1.0087). **Caveat for interpretation:**
the control keeps the same tau and mu, but its dominant pole is the first-order proximal one,
about 24% faster-decaying than native at initialization. The candidate's boundary choice
m = tau + mu cancels that shift to second order. A candidate-vs-control difference therefore
combines (a) the released inertial mode and (b) a dominant-timescale difference. The
trained audit reports both (`dominant_decay_over_native_counterfactual`, auxiliary
residues and energy fractions), so they can be read separately. Native and P at the same
seeds remain the practical comparators.

## Trained-P audit and trained-pole monitoring

`experiments/s5_proximal_inertia/block_audit.py` is read-only and audits each task in three stages:
reconstructed init, the selected checkpoint (re-derived from the full validation history and
required to equal `task_result.json`), and the final epoch.

Per layer and mode it reports:
- the actual exact-ZOH block poles, in float64 (SciPy) and in production float32 (the arm's own JAX
  constructor);
- counts of growing, boundary and nonfinite poles, with nothing clipped;
- the native pole at the same weights, labeled as an untrained counterfactual;
- the 16,000-lag kernel 2 Re(C h_t Delta B~) per direction;
- per-pole residues and component energy fractions, which are non-additive and withheld when ill-conditioned.

Checkpoints are hashed before and after the audit.

The cluster job runs it on CPU, alongside the GPU tasks, over P alpha=1 seeds 301–305 and
alpha=0.1 seeds 301/302. Its status (`status/p-audit`) is explicit:
- `PASS audited=N unavailable=0`;
- `PARTIAL audited=N unavailable=M`: unfinished P tasks are listed in `p-audit/summary.txt`, and this is not fatal;
- `FAIL ...`: the audit crashed, its report is unreadable, nothing was auditable, or no status was written.

A FAIL marks the whole Slurm job failed (exit 1), even when all training succeeded. After training it audits the four new tasks. During training, every epoch writes
the arm's own 2x2 pole summary (float32 vs float64, stable/boundary/unstable) to
`pole_diagnostics.jsonl`, and every step logs the block spectral radius.

## Cluster run

One allocation of four RTX 4090s on pgi14-gpu1. Each GPU runs one task:
compile profile, then measured full-shape memory and step time, then the full-path GPU smoke,
then training only if that smoke passed. The four tasks run in parallel.

Before any GPU work, a CPU pre-flight checks:
- the data cache;
- native parameter and optimizer identity for both arms and both seeds;
- stable initial poles.

The job clones the pinned commit into `/Local/durso/s5-proximal-inertia-<job>`.
It writes only to `/Users/durso/s5-runs/s5-proximal-inertia/job-<job>/`:
- `<arm>/<seed>/`: training outputs;
- `smoke/`: GPU smokes;
- `profiles/`: compile and measured memory, step time;
- `p-audit/`: the trained-P audit;
- `inertia-audit/`: trained audit of the new tasks;
- `status/` and `logs/`.

The log is `/Users/durso/s5-runs/s5-proximal-inertia-<job>.log`. It ends with a summary from
`experiments/s5_proximal_inertia/job_report.py`. That summary is reporting only: missing or null
telemetry (e.g. `peak_gib=None` when the allocator exposes no statistics) prints as `n/a`,
and a formatting error cannot change the job's outcome. The test split is never read.

Comparisons to make afterwards, all paired by seed and all using the selected-validation rule:
- candidate vs control, which isolates the inertial mode, subject to the caveat above;
- candidate vs native N and P at seeds 301/302, for practical value.

A gain over P would not by itself show a FOOF mechanism, since the history construction differs too.
