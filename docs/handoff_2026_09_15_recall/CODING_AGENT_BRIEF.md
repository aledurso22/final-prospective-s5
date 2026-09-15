# Bounded memory-recall comparison using the existing derived recurrence

15 September 2026. Coordinator handoff to Claude Code. Preparation only: the coordinator has run no numerical checks or training for this study. Preserve the completed Speech Commands studies and their unsuccessful screens. This is a new mechanistic experiment, not a replacement benchmark result.

## Purpose and permitted claim

Test whether the existing physically constrained generalized prospective recurrence improves useful recall over ordinary S5 and Rawat's prospective-input S5 when the answer requires information from an earlier input. No new physical term, activity-dependent gate, auxiliary learning objective, local error rule, or meta-learner is requested. Train by full BPTT.

A positive result could support: "In a controlled recall task, an S5 using the physically derived generalized prospective recurrence improves recall under matched training conditions."

It would not establish a general advantage on practical benchmarks, a literal biological implementation of arbitrary complex S5 feedback, or that a physical interpretation itself creates computational capacity. A fixed linear realization remains an augmented SSM. The controls below distinguish response learning and added state from the whole method's performance.

The observed 84.85% from prospective coding in the recurrence was obtained on a mean-pooled speech classifier. It does not decompose accuracy into static and memory contributions. The new task removes pooling and makes the query input independent of the answer.

## 1. Keep the equation; remove only the redundant coordinate

Read the coordinator's `outputs/constrained_response_handoff_2026_09_15/POST_RUN_ALGEBRA_AND_REPORT_REVIEW.md` in the handoff workspace first. Its report corrections still apply; do not rewrite historical execution provenance.

For constant coefficients within each sequence, the implemented family is

\[
\gamma_n\rho T\ddot s+\gamma_n\dot s+r_n+T\dot r_n=0,
\qquad r_n=-\Delta\lambda s-\Delta B_c x.
\]

Define \(\widehat\Delta=\Delta/\gamma_n\). Exactly the same dynamics become

\[
\boxed{\rho T\ddot s+\dot s+r+T\dot r=0,
\qquad r=-\widehat\Delta\lambda s-\widehat\Delta B_c x.}
\]

Use gamma_n=1, learned native log_step representing log(hat_Delta), and one learned rho per stored complex mode, shared by its conjugate. Keep T=5 input intervals and the existing numerical rho bounds [0.01, 0.9999]. Mass is derived: mu=T*rho, not an independent parameter. Keep the established pole clipping and common alpha-scaled input map for the matched arms.

The existing exact realization is unchanged:

\[
\rho\dot s=-r-(1-\rho)v,\qquad T\dot v=\dot s-v.
\]

Use the existing block ZOH, associative scan, physical-state readout and native D. No delayed-input approximation to the total residual derivative. Fixed coefficients during a sequence are essential; ordinary parameter updates occur between independent sequence batches.

The existing positive component map remains available with gamma_n=1: kappa_0=1.5, c_s=c_d=7.5, G_s=G_d=kappa_0/rho, h=G_s*sqrt(1-rho), g_L=kappa_0/(1+sqrt(1-rho)). It retains the earlier constant-conductance/additive-current and prospective-source assumptions. Applying this response to S5 feedback remains the declared computational extension; it is not an independent microscopic derivation of every learned S5 matrix.

Removing gamma_n preserves representable forward responses, not AdamW optimizer trajectories. A historical checkpoint conversion must absorb its **executed, clipped** gamma_n into log_step; do not claim an optimizer-state conversion. This experiment need not reuse Speech Commands weights.

## 2. One task: recall the latest marked symbol through distractors

Use a small causal S5 stack and a tokenwise query readout. Suggested fixed configuration for this bounded study: two layers, model width 32, base SSM size 32, four HiPPO blocks, conjugate symmetry, ZOH, no dropout. Check actual state counts instead of inferring them from configuration labels.

Replace sequence/batch normalization with the same tokenwise LayerNorm in every arm. No temporal pooling, bidirectionality, temporal normalization, attention, or readout access to earlier activations. Retain the shared ordinary nonlinear blocks and residual connections. All models receive the same input encoding and query readout.

Concrete generator:

- Eight symbol classes. Inputs encode a symbol, a cue marker, and a query marker. The final query has no symbol payload.
- Sequence length 128. The query is at index 127. Exactly two earlier tokens are marked cues; all other earlier tokens are unmarked distractors from the same symbol alphabet.
- The target is the symbol of the **latest marked cue**, regardless of intervening distractors. Query-to-latest-cue delays during training are 8, 32, and 64, equally represented. The gap between the two cues is independently chosen from 8, 16, and 24. These choices keep all cue positions valid.
- Generate paired sequences with identical distractors and marker positions, but swap the two distinct marked symbols. Their token multisets are identical and their targets differ. Balance target classes and paired order by construction.
- Use independent training and evaluation random streams, shared across arms. Evaluate 1,024 examples per delay at delays 8, 32, 64 and the held-out longer delay 96. The 96-delay result is extrapolation, not part of training or model selection.

The training objective is only cross entropy at the final query. No additional short-delay head, hand-assigned memory neurons, memory-preservation penalty, or auxiliary teacher. Short and long delays are conditions of the same task.

A truly memoryless causal model receives identical final query inputs, so its output cannot depend on the target. The professor recurrence must implement that property exactly. This is the purpose of the control; failure to optimize ordinary static weights is not required.

## 3. Paired continuation, with no response-initialization search on scores

For each seed 100, 101 and 102, train the small matched ordinary S5 for 1,024 updates. Clone its common parameters into the comparison arms below and train each for 1,024 further updates on identical ordered minibatches. Batch size 32, full BPTT, fixed learning rate 1e-3. Use the existing benchmark optimizer policy consistently, with no response-specific learning-rate tuning. Start a fresh optimizer for **every** continuation arm, including ordinary S5. Record decay conventions for log parameters.

This is a paired continuation study. Warm-up cost is shared and must be reported. It does not answer which model trains best from scratch, and Rawat's model is a literature mechanism transplanted to this task, not a reproduction of its published benchmark schedule.

Initialize the generalized recurrence with fixed rho_0=0.9998, inside the existing bounds. This is a declared initialization choice, not a physiological measurement. At rho=1 its transfer reduces to the ordinary memory-bearing SSM:

\[
G(p)=\frac{b(1+Tp)}{\rho Tp^2+(1+Tj)p+j},\quad
G_{\rho=1}(p)=\frac{b}{p+j},\quad j=-\widehat\Delta\lambda.
\]

Near-one rho does not guarantee functional closeness for high-Q complex modes. Before continuation, measure the actual **signal-only** discrete core response change (exclude native D), and the query-logit change on unlabeled initialization probes. Record per-layer finite-window impulse differences and frequency-grid differences, with their scope. Do not describe these arms as function-matched just from rho's numerical value. If core relative changes exceed 1% under the protocol's predeclared response norm, stop this initialization route and report it before changing rho or reading comparative scores. Define the norm and probes in the committed protocol; handle zero reference response explicitly.

Clone continuation arms:

1. **Ordinary S5, matched substrate:** original carry, learned poles/clock/input/output.
2. **Rawat prospective-input S5:** the existing implemented input-side two-tap law, fixed horizon 5, same common parameters and substrate. Report its own initial function change; do not label it function-matched.
3. **Prospective coding in the recurrence:** exact professor zero-prehistory map, same substrate and readout. Compute -B_c/lambda directly; its exact data-loss gradient through log_step is zero. No artificial recurrent buffer.
4. **Generalized prospective coding in the recurrence, learned rho:** the equation above, only rho added to native learned quantities.
5. **The same generalized recurrence, frozen rho:** identical initial function and carry to arm 4, rho frozen at rho_0. This tests whether learning the response helps within the same dynamical family.
6. **Ordinary S5 with matched recurrent-state capacity:** twice the stored complex modes, same layer width/readout architecture. Embed the warm-up model by retaining its modes and adding stable modes with nonzero input coupling and initially zero readout coupling. Verify the initial ordinary function is preserved and added modes have a trainable readout path. Report extra parameters and actual cost; this matches recurrent state, not every resource.

No learned gamma_n, M=0 rerun, activity-dependent coefficients, new controller, truncated temporal gradients or meta-learning in this study.

## 4. Measurements and interpretation

Report fixed-endpoint accuracy and cross entropy by delay for every arm/seed. The primary comparison is the equally weighted mean accuracy at the two trained longer delays, 32 and 64. Report paired differences against BOTH ordinary S5 and Rawat's prospective-input S5, with each seed shown; three seeds do not establish a reliable population variance or a broad benchmark claim. Report delay 8 separately to expose a retention/responsiveness tradeoff, and delay 96 as held-out extrapolation. Do not select checkpoints or seeds by the best comparison.

Also report parameter counts, physical/auxiliary carry, Rawat's input buffer, wall time, learned rho and effective clock distributions. Saved coefficients alone are not a memory-importance score.

Use one small, fixed held-out probe batch for paired input interventions: change the relevant marked cue versus change an unmarked distractor, keeping other inputs fixed. Measure prediction changes and recall, not just gradient magnitude. Report both interventions even if they do not support selectivity. Mode ablations can be deferred; do not multiply the experiment to obtain a desirable mechanism story.

What the comparisons permit:

- Arm 4 versus arms 1 and 2: whether the full construction improves this recall experiment over the literature mechanisms.
- Arm 4 versus arm 5: whether learning rho helps relative to the same initial fixed response. This is task-trained response shaping, not automatically meta-learning or within-sequence adaptation.
- Arm 4 versus arm 6: whether a larger ordinary recurrent state is an adequate alternative under the declared training conditions. This alone does not isolate every possible benefit of the exact coefficient ties.
- Arm 3: the fully matched prospective recurrence loses driven linear memory. This is not a claim that TSS's actual memory architecture cannot learn memory; TSS explicitly retains non-prospective memory neurons.

To assert that the **particular physical constraints** outperform more general filters would need a further suitably matched response-parameterization control. Do not make that stronger assertion from this small batch.

## 5. Implementation and execution discipline

Work on a new feature branch from the current verified checkout; record checkout path and parent commit. Preserve concurrent and historical work. Keep changes scoped to the new parameterization, task, runner and necessary correctness checks.

No local numerical runs by either agent. Static inspection and code preparation locally are allowed. Commit/push code and a concrete protocol before cluster numerical execution. Give the user one short launcher; the coordinator cannot send commands directly to Claude Code or operate the cluster.

Cluster total cap: 1,200 seconds including backend startup, focused checks, compilation, warm-ups, all continuations, evaluation and cleanup. Use a deadline and time projection before comparative training. Reuse compilations across seeds where valid. If the declared batch does not fit, report INCOMPLETE and the projection; do not silently reduce seeds, omit controls, or train beyond the cap. This study does not trigger a larger run automatically.

Required focused checks, reusing existing tested paths:

- Constant-coefficient gamma/clock reparameterization preserves the executed blocks, input drive and forward response with identical carry.
- The rho=1 algebraic limit agrees with ordinary S5 under consistent zero initialization; use this only as a mathematical boundary check, not as the physical training configuration.
- Query causality, tokenwise normalization, resets, paired generator symmetry and balanced labels; professor query output is invariant to the earlier cue.
- Actual production dtype, finite full-BPTT gradients and an update to rho; frozen arm excludes rho updates. Professor log_step is structurally unused by the data loss.
- Common-parameter cloning, function-preserving state-capacity embedding, and reported initialization response differences are checked on the actually executed modules.

Use fresh artifact directories. Do not modify old Speech Commands data, checkpoints, scores or protocols. Report actual checks, completed runs and any incomplete stage in `docs/PROSPECTIVE_MEMORY_RECALL_REPORT.md`, including commit, environment, commands, seeds, timing, all metrics and limitations. Push code/report to the public repository and provide commit-specific links.

## Literature scope

- TSS, Section 3.3: https://arxiv.org/html/2511.14917v2 — its memory experiment separates memory-retaining neurons from prospective processing; our study tests a different recurrent response under BPTT.
- Rawat et al., S5 construction and Appendix E: https://arxiv.org/html/2609.04134v1 — preserve its input-side correction as a comparator. This toy experiment is not its Speech Commands benchmark.
- Existing derivation contract: `outputs/cluster_gp_handoff_2026_09_15/DERIVATION_CONTRACT.md` in the coordinator workspace. Its fixed-reference training policy was superseded by the later constrained-learning brief; its circuit, source and normalization qualifications still apply.
