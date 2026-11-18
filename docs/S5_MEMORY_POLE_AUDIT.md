# N/A/C checkpoint memory audit

This analysis tests whether target-scale calibration changed the recurrent decay spectrum and whether the extra recurrent component is linearly excited and read out. It does not train a model, read a dataset, reevaluate accuracy, or modify source checkpoints. No real cluster checkpoint result is included until this tool is run there.

## Inputs and selection

The fixed comparison uses native N (`native_matched_s5`), original A (`generalized_prospective_s5`), and target-scale-corrected C (`calibrated_old_s5`), seeds 301–303. The coefficient rule for C is `k=gamma`, retaining the original corrector; A uses `k=T`.

- Seeds 301/302: `/Users/durso/s5-runs/s5-timescale-2x2/20261117-195818`.
- Seed 303: `/Users/durso/s5-runs/s5-timescale-seed303/20261117-230628` (the completed retry, not the interrupted earlier wave).

Main figures compare reconstructed initialization with each task's recorded selected-validation checkpoint. Selection is verified against the complete logged validation history: maximum accuracy, then minimum cross-entropy, then earliest epoch. `--all-epochs` additionally analyses every saved epoch 1–15 in these nine tasks. The historical selection is not changed.

Initialization is reconstructed with the production runner in an isolated CPU subprocess with `JAX_ENABLE_X64=0`. The resulting arrays are analysed in NumPy float64. This avoids the previously discovered different-random-draw problem from initializing with x64 enabled. Reconstruction is not proof that a historical process, JAX version or backend used bit-identical initial parameters. Agreement reported across reconstructed arms refers to their SSM parameters, not historical saved epoch-zero checkpoints.

## Pole measurements

For native, the only recurrent pole is `a`. For A/C the roots solve `z²-a1*z-a2=0`. The implementation uses a stable product-form quadratic evaluation rather than subtracting nearly equal numbers for the smaller root. Poles are ordered by magnitude within each channel, without claiming a uniquely tracked eigenmode across training.

For each pole, the e-folding decay length is `-1/log(abs(z))` samples only when `0<abs(z)<1`. A/C secondary poles remain included when they are stable even if the dominant pole in that channel is unstable. Zero, exactly-unit, growing and nonfinite roots are counted separately. None is clipped into a finite decay distribution. A repeated pole can introduce a polynomial transient, so its single exponential timescale is not a complete description of a response.

Counts distinguish complex channels from poles: six layers × 64 channels, one pole per native channel and two per A/C channel. Shared bidirectional/conjugate copies are not counted twice. Strict `abs(z)>=1` counts are separate from the `1e-5` numerical-boundary band and the count above `1+1e-5`. Float32 calculations use the actual coefficient/root helpers on CPU; they are not claimed to replay GPU arithmetic bit for bit.

The log-scale empirical cumulative distributions show each layer and seed separately at initialization and after training. The 16,000-sample sequence horizon is marked. The JSON and CSV retain individual poles and denominator counts; unstable or undefined decay lengths are not hidden in maxima or medians. Paired layer summaries compare dominant finite decay lengths with native, using matched seed/layer/initialized-channel index. Channels are not independent statistical replicates.

## Impulse responses and observability

Finite-horizon impulse responses use the actual recurrence and learned input/readout matrices over 16,000 samples. They distinguish state-recurrence components from responses accessible through `B` and `C`. Pole residues and component-energy measures must retain cancellation and interference qualifications, especially near repeated roots. A zero or negligible residue can make a recurrent pole absent or weak in an input-output response.

The measured object is the linear SSM branch. It excludes the direct skip term and nonlinear inter-layer processing. Forward and backward branches must be identified separately; their summed energies are not a complete bidirectional network response. A linearly observable slow component is evidence about dynamics, not proof of useful task memory or an accuracy improvement. Finite-horizon responses remain meaningful for mildly growing modes; they do not imply infinite-horizon stability.

The JSON additionally includes an interior bilateral recurrent-kernel energy: the zero-lag kernels from both directions are added before squaring, and the positive/negative lag tails occupy separate supports. Its signed lag window is `[-15999,15999]`, not a claim that a centered impulse has that much context in a finite 16,000-token example. The plotted direction-energy sum remains explicitly labeled. Per-component energy fractions can exceed one because cross-root interference is recorded separately; they are not probabilities or additive shares of full-layer energy.

## Cluster submission

Use a clean checkout containing this published revision and pin its full SHA:

```bash
EXPECTED_COMMIT="$(git rev-parse HEAD)" DRY_RUN=1 \
  bash bin/run_experiments/submit_s5_memory_audit.sh

EXPECTED_COMMIT="$(git rev-parse HEAD)" \
  bash bin/run_experiments/submit_s5_memory_audit.sh
```

The helper submits a CPU-only job on `pgi14-gpu1`, partition `pgi14`, requesting 4 CPUs, 24 GB RAM and 2 hours. It requests no GPUs and hides CUDA devices from the analysis. The node restriction is necessary because the Python environment is `/Local/durso/final-prospective-s5/.venv`; `/Local` is node-local. The job may wait for CPU/RAM resources even when GPUs are unused.

Each job creates its own checkout `/Local/durso/s5-memory-audit-<jobid>` and writes a new `/Users/durso/s5-runs/s5-memory-audit/job-<jobid>` directory. The main log is `/Users/durso/s5-runs/s5-memory-audit-<jobid>.log`. Neither the active training checkout nor its output directory is used for writes. Source hashes are checked by the analysis. An existing output directory is refused. The submission helper does not push code or launch any training.

Primary products are `memory_poles.json`, `poles.csv`, `summary.txt`, and PNG/PDF per-seed layer figures, plus the all-epoch analysis when requested. The result is ready for interpretation only after successful completion; smoke tests and synthetic fixtures cannot establish the trained models' memory spectra.

`--all-epochs` writes `checkpoint_analyses/<arm>/<seed>/epoch_NN.json` for all 135 source checkpoints, plus `epoch_summaries.json` and `epoch_summaries.csv`. Every source checkpoint is hashed before and after the analysis. `memory_poles.json` is written only after the complete audit succeeds. A partial output directory does not establish successful completion.

## Local validation

50 focused tests pass: known complex poles, zero/unit/growing/repeated roots, per-pole exclusions, actual N/A/C coefficient routing, explicit dense-kernel energy oracles, input/readout cancellations, component interference, synthetic complete-history selection/provenance, source mutation detection, plotting and CPU-only submission. A separate local check analysed all six production-width layers from the actual float32 initializer for N/A/C seed 301 over 16,000 lags. These checks validate the implementation; trained-checkpoint measurements still require the cluster run.
