# Input-lead ablation of proximal inertial S5 (sigma = 0)

Date: 2026-11-18. Implementation, CPU checks and prepared cluster jobs. **Nothing has been
trained or launched.** This is one ablation of `proximal_inertial_s5` (m = 1.5 tau). It is not
a new architecture and makes no accuracy claim.

## The change

The existing model expands to

    m s'' + s' = -r_mu - tau kappa/d s' + sigma tau/d b',   d = 1 + mu kappa,  r_mu = (kappa s - b)/d.

- sigma = 1 is `proximal_inertial_s5`, unchanged.
- sigma = 0 is the new arm `proximal_inertial_nolead_s5`.

Everything else stays the same: tau = 0.1/|kappa|, mu = tau/2, m = 1.5 tau, the native
parameters and initialization, the outer optimizer, backbone, batch 16, 15 epochs, split and
selection rule. No learned parameter is added. The matched baseline is `proximal_inertial_s5` at
m = 1.5 tau, **not** the m = tau control.

Implementation: `s5/proximal_inertia.py` gains a static `input_lead`.

    w_sigma = m s' + tau kappa s/d - sigma tau b/d
    A       = [[-tau kappa/(m d), 1/m], [(tau/m - 1) kappa/d, -1/m]]    (unchanged)
    g_sigma = [sigma tau/m, 1 - sigma tau/m] / d

Exact physical-input ZOH comes from the same augmented 3x3 exponential, with zero prehistory
and the position readout. sigma = 1 runs the original expression literally.

Identities, verified in tests:
- H_sigma(p) = (1 + sigma tau p)/[(1+mu kappa)(m p^2 + p) + kappa(1 + tau p)].
- Poles and the DC gain 1/kappa are unchanged; residues change.
- The continuous DC delay is 1/kappa + mu + (1 - sigma) tau, so sigma = 0 adds tau.
- Input jumps: m Delta(s') = sigma tau Delta(b)/d.

## CPU checks (all pass)

`tests/test_proximal_inertia_lead.py` (9 in float64; float32 runs the 4 that are not float64-only):
- the transfer identity, DC gain and DC delay, for both sigmas;
- A is bitwise identical between sigmas, the poles are equal and bd changes;
- an independent physical-coordinate simulation with the sigma jump agrees to 1e-9 for
  sigma = 0 and 1, and each policy disagrees with the other's jump (so the old jump is
  not retained at sigma = 0);
- sigma = 0's explicit adjoint equals autodiff, and it matches finite differences in float64;
- **sigma = 1 is bitwise identical to commit acb4d85**, the code that trained the baseline, in
  output, every gradient, Ad and bd. On CPU this was checked at 16,000 tokens in float32,
  and in tests at 2,048 tokens in both precisions.

Integration and job tests: `test_proximal_inertia_lead_integration.py` (4) and
`test_proximal_inertia_lead_jobs.py` (9). They check:
- that the ablation's protocol identity differs from the baseline's only in arm and input_lead;
- identical native parameters, and a CPU smoke;
- that the audit's counterfactual leaves Ad unchanged (< 1e-12) and changes only bd;
- the baseline-reuse gate, including its refusals;
- the paired report;
- script syntax, dry runs and GPU-count refusals.

The earlier proximal inertial suites still pass.

## Both input policies at identical initialized weights

`docs/evidence/s5_proximal_inertia/init_input_lead_policy/`: reconstructed float32 init, seeds
301/302, 6 layers, 16,000-lag B/C-coupled kernels, sigma = 0 as a counterfactual on sigma = 1 weights.

| Quantity (per-layer medians) | sigma = 1 | sigma = 0 |
|---|---|---|
| Kernel difference energy, relative to sigma = 1 | — | 0.022–0.029 per layer |
| Total kernel energy, sigma0/sigma1 | 1 | 0.97 |
| Auxiliary-pole relative residue, p50 (p95) | 0.23 (0.28) | 0.47 (0.50) |
| Auxiliary component, fraction of mode energy, p50 | 0.003 | 0.026 |
| Cross-pole interference, fraction of mode energy, p50 | -0.031 | -0.10 |
| Forward L90 lag (samples) | 70–180 | 74–194 |
| max abs(Ad difference) | — | 2e-16 |

Removing the lead roughly doubles the auxiliary residue share and its energy, increases
destructive interference between the two poles, and slightly lengthens the energy profile,
consistent with the extra tau delay. Larger auxiliary energy is not evidence of better accuracy.

## Checkpoint audit of the trained sigma = 1 runs: pending (cluster)

The selected `proximal_inertial_s5` checkpoints of `/Users/durso/s5-runs/s5-proximal-inertia/job-76984`,
seeds 301/302, live on the cluster. The probe job audits them read-only, with the selection rule
re-derived and checkpoints hashed before and after. It covers reconstructed init, the
selected epoch and the final epoch, with sigma = 0 as the diagnostic policy at the same frozen
weights. For each, it reports kernel difference energies, auxiliary residues, component
energies and interference.

## Cluster jobs (prepared, not launched)

**Probe** (`cluster_s5_proximal_inertia_lead_probe.sh`, 1 x RTX 4090, trains nothing):
- the sigma = 1 vs acb4d85 identity on GPU;
- the full-shape forward/backward probe for sigma = 0 and sigma = 1, measuring peak allocator
  memory and synchronized step time;
- alongside, the checkpoint audit above.

Output goes to `/Users/durso/s5-runs/s5-proximal-inertia-lead/probe-job-<id>/`.

**Training** (`cluster_s5_proximal_inertia_lead_train.sh`, 2 x RTX 4090): fresh sigma = 0 at seeds
301/302, one per GPU, each with its own profile and GPU smoke before training. Output goes to
`/Users/durso/s5-runs/s5-proximal-inertia-lead/train-job-<id>/`. The test split is never read.

**Why no baseline rerun.** The completed sigma = 1 runs are reused, because the only remaining
differences can be checked, and the job checks them before spending anything. Training refuses to
start unless both hold:
1. `baseline_check.py` passes. The baseline tasks completed 15 epochs and their selected epoch
   equals the selection rule. The configuration is matched (m = 1.5 tau, mu = tau/2,
   alpha = 0.1, sigma = 1). The recorded protocol matches: batch, epochs, scan, ratios,
   data-manifest digest, JAX version, float32, and training at acb4d85.
2. sigma = 1 at the new commit is bitwise identical to acb4d85 on that GPU.

If either fails, the reason is written to `baseline_check.json` (or the identity report), and
the baseline should be rerun alongside the ablation instead of comparing across protocols.
Afterwards the job audits the trained sigma = 0 checkpoints, with sigma = 1 as the counterfactual,
and prints a paired-by-seed comparison under the selected-validation rule. Two seeds do not
support an accuracy claim either way.
