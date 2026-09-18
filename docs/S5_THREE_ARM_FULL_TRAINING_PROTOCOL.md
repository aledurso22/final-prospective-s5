# Three-arm full-training raw-audio S5 protocol

This preregistered experiment contains exactly three scientific arms:

| Scientific name | Code identifier | Recurrence |
|---|---|---|
| Native S5 | `native_matched_s5` | native ZOH S5 |
| Zucchet prospective dynamics — finite-difference realization | `zucchet_prospective_s5` | literal finite-difference negative control |
| generalized prospective dynamics \((M,\gamma,T)\) — finite-difference realization | `generalized_prospective_s5` | generalized finite-difference recurrence |

## Equations

The native discrete S5 coefficients are \(\bar A=e^{\Delta\Lambda}\) and
\(\bar B=\phi_1(\Delta\Lambda)\Delta B_c\), where
\(\phi_1(a)=(e^a-1)/a\), continuously \(\phi_1(0)=1\). The physical sample
interval is \(h=1/16000\) seconds; code uses the equivalent sample clock
\(h=1\). The raw sample is held constant within an interval.

For both prospective arms, \(f_t=F_Ts_t+G_Tx_t\), with
\[
F_T=I+\frac{T}{h}(\bar A-I),\qquad G_T=\frac{T}{h}\bar B.
\]

**Zucchet prospective dynamics — finite-difference realization.** With
\(k=T/h\),
\[
s_{t+1}=\bar A s_t+\bar Bx_t+f_t-f_{t-1},
\]
or
\[
s_{t+1}=((1-k)I+(1+k)\bar A)s_t+((k-1)I-k\bar A)s_{t-1}
+(1+k)\bar Bx_t-k\bar Bx_{t-1}.
\]

**generalized prospective dynamics \((M,\gamma,T)\) — finite-difference realization.**
With \(q=M+h(\gamma+T)\), \(\alpha=M/q\), \(\beta=h^2/q\), and
\(\delta=hT/q\),
\[
s_{t+1}=A_1s_t+A_2s_{t-1}+C_1x_t+C_2x_{t-1},
\]
where \(A_1=(1+\alpha-\beta)I+(\beta+\delta)F_T\),
\(A_2=-\alpha I-\delta F_T\), \(C_1=(\beta+\delta)G_T\), and
\(C_2=-\delta G_T\). The scan carries \(z_t=[s_t,s_{t-1}]\), with
\(s_{-1}=s_{-2}=0\) and \(x_{-1}=0\). Parameters use
\(T=\operatorname{softplus}(T_{raw})\), \(\gamma=\operatorname{softplus}(\gamma_{raw})\),
and \(M=\rho\gamma T\), with bounded \(\rho\). The exact \(M=\gamma=0\)
boundary is tested algebraically; the literal ordinary arm is not damped.

## Training contract

All arms use Speech Commands v0.02, the ten selected keywords, raw waveforms of
length 16,000, official validation/testing lists, training-derived channel
normalization, bidirectional S5, depth 6, feature width 96, latent size 128,
16 HiPPO blocks, dropout 0.1, batch normalization, mean pooling, and a ten-class
decoder. Only the recurrence and its necessary parameters differ.

Each arm/seed is from scratch with seeds 301, 302, 303; 40 epochs; batch size
16; global LR 0.008; SSM LR 0.002; weight decay 0.04 with upstream
`noBCdecay` exceptions; one-epoch linear warmup over `steps_per_epoch`, then
cosine annealing; checkpoint each epoch and validation-select the checkpoint.
The test split is opened once by the finalizer after all nine jobs succeed.

Every update records scientific arm, seed, epoch, step, loss, accuracy, state
norm, gradient norm, finite-gradient/state flags, nonfinite status, and maximum
companion spectral radius. A nonfinite update writes `failure.json` with exact
task identity and step and terminates only that task; the finalizer rejects the
incomplete ladder. Formal stability analysis is a preregistered mechanistic
prediction, not a launch gate.

## Execution

Each array task first performs one exact-shape finite update using the same
model, optimizer, batch size and sequence length. It fails closed on nonfinite
loss, gradients, or complete state, then automatically reinitializes and starts
the 40-epoch training. There is no separate manual preflight cycle.

The array is `0-8%4`, one RTX 3090 per task, 10 CPUs, 50 GB RAM. Dispatch order
is fixed scientific order and is not based on historical timing. The finalizer
is submitted with `afterok`, verifies all nine task artifacts, and opens test
exactly once.

Launch only after reviewing the committed checkout:

```bash
EXPECTED_COMMIT=<authoritative-commit> \
  bash bin/run_experiments/cluster_s5_three_arm_full.sh
```
