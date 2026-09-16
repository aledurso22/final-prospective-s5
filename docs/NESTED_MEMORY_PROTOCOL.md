# Nested associative-memory study: frozen protocol

Committed **before** any numerical execution. Coordinator sources:
`nested_prospective_memory_handoff_2026_09_16/` — `CODING_AGENT_BRIEF.md`,
`EQUATION_CONTRACT.md`, `PILOT_PROTOCOL.md`, and the `ANALYTICAL_AUDIT.md`
supplement of 16 September 2026.

Branch `nested-prospective-memory`, worktree, parent
`62c076739a9afa1624faec68961e7e500d6f1ed8` (`tss-pilot`). The completed TSS
pilot and every earlier study are untouched; this adds files and changes none
of theirs.

**Nothing here has been executed.** No result is recorded until after a run.

## 1. Question

Does the physically constrained generalized prospective rule improve the
learning and revision of associative memory compared with ordinary delta writes
and existing momentum-based memory updates?

The memory is an **inner learner over sequence time**; outer full BPTT trains
the feature tables, the readout and the literature gates. This is not a fixed
linear recurrence substitution, an eligibility trace, or an optimizer acting on
slow weights. **Physical coefficients are fixed throughout.**

## 2. The law, and the coordinates that make the step exact

```
E_t(W) = (m_t/2)||W k_t - v_t||^2 ,   R_t = m_t (W k_t - v_t) k_t^T
M Wddot + gamma Wdot + R + T Rdot = 0 ,   M, gamma, T > 0 ,  M <= gamma T
```

carried as `P = M Wdot + T R`:

```
Wdot = (P - T R)/M ,   Pdot = -gamma P/M + (gamma T/M - 1) R
```

**Declared first reference, fixed for this batch: `gamma = T = 1`, `M = 3/4`,
`h = 1`.** Identifying one token interval with this time unit is an explicit
computational convention, not a measured constant. No source-strength or
horizon sweep is permitted here.

Carrying `P` rather than `Wdot` is essential: at a key, value or mask jump the
velocity jumps by `-T dR/M`, and `P` is continuous there. `P` is never reset at
a query, a token boundary or an episode-internal key change; only a new episode
resets `W = P = 0`. **`W` moves during a query interval** because its velocity
has not vanished; freezing it would define a different method and could hide an
interference cost.

### Exact token step, verified

For a held unit key, with `e = Wk - v`, `p = Pk`,
`A_1 = [[-T/M, 1/M], [gamma T/M - 1, -gamma/M]]`, `F = exp(h A_1)`,
`a0 = exp(-gamma h/M)`, `b0 = (1-a0)/gamma`:

```
(e', p') = F (e, p)
W' = W + b0 P + [(e' - e) - b0 p] k^T
P' = a0 P + [p' - a0 p] k^T          and at m = 0:  W' = W + b0 P,  P' = a0 P
```

`F, a0, b0` are computed **once on the host in float64** and typed explicitly;
no matrix exponential runs inside the scan, and `-expm1` is used for `1 - a0`.
Verified values at the declared reference: eigenvalues of `A_1` are `-2/3` and
`-2`; `F11 = 0.3243762011346`, `F21 = 0.0945204589490`,
`a0 = 0.2635971381157`, `b0 = 0.7364028618843`.

A raw key below the norm floor `1e-6` is **not** treated as unit: the key is
zeroed and the write suppressed, with the gradient taken through the safe
branch.

## 3. Five arms

| Identifier | Display name | Carry | Trainable | Policy |
|---|---|---:|---:|---|
| `prospective_memory` | Generalized prospective memory | 128 | 392 | exact held-input law, `gamma=T=1`, `M=3/4`, fixed |
| `inertial_memory` | Inertial memory without prospective correction | 128 | 392 | exact `M Wddot + gamma Wdot + R = 0`, same `M, gamma` |
| `delta_matched_write` | Delta memory with matched first write | 64 | 392 | fixed `beta = 1 - F11` |
| `gated_delta` | Gated DeltaNet rule | 64 | 480 | published rule and gate family in the common shell |
| `momentum_delta` | Momentum DeltaNet rule | 128 | 569 | published rule and gate family in the common shell |

Carries match the declared table. **Parameter budgets are not equal**: the
gates add 88 and 177 trainable values respectively, and the report says so. The
inertial arm is the exact same-state ablation of the prospective term; it is a
passive heavy-ball comparison and, with `T = 0`, lies **outside** the
`M <= gamma T` sector, so it is never described as another admissible point of
the same family. Momentum DeltaNet is the principal literature comparison at
the same recurrent-state size.

## 4. Source audit, pinned

Gate parameterizations are taken from the **official implementation**, not
paraphrased from the papers.

| item | source | pinned |
|---|---|---|
| Momentum DeltaNet rule and gates | `github.com/HuuYuLong/MomentumDeltaNet`, `flash-linear-attention/fla/layers/momentum_deltanet.py` | commit **`c6e77fa261fb0c002fae1a14b6209a5b28d2edc9`** |
| Momentum reference recurrence | same repo, `fla/ops/momentum_delta_rule/naive.py` | same commit |
| Gated DeltaNet rule and gates | same repo, `fla/layers/gated_deltanet.py` | same commit |
| Gated DeltaNet paper | `arxiv.org/html/2412.06464v2`, s2.2-3.1 | defers `alpha` to Mamba2's parameterization |
| Momentum DeltaNet paper | `arxiv.org/html/2605.05838v1`, Eqs. 4-5, s3.3 | — |

Executed gate maps, verbatim from the pinned layer:

```
log_alpha = -exp(A_log) * softplus(a_proj(x) + dt_bias)
log_mu    = -exp(Mu_log) * softplus(m_proj(x) + mu_bias) ,  clamp_min(min_log_mu)
eta       = tanh(e_proj(x)/tau) + 1                      # range (0, 2)
beta      = sigmoid(b_proj(x))
theta     = arctan(eta * exp(log_factor))
beta      = sin(theta)^2 * beta ;  log_alpha += log(cos(theta)^2)
```

Initializers, verbatim: `A ~ U(0,16)`, `A_log = log A`; `Mu ~ U(0,16)`,
`Mu_log = log Mu`; `dt ~ exp(U(log 1e-3, log 1e-1))` clamped at `1e-4`,
`dt_bias = mu_bias = dt + log(-expm1(-dt))`;
`log_factor = log(U(0, 4))`; `tau = sqrt(hidden/tau_factor)`, `tau_factor = 1`.
Projections use PyTorch's `nn.Linear(bias=False)` default,
`U(-1/sqrt(in), 1/sqrt(in))`.

**The `min_log_mu` ambiguity is resolved from the pinned configuration, not by
preference: the official constructor default is `min_log_mu: float = -2.`**,
frozen here before any training. The brief noted `-1` appears in a paper
passage; the official default governs, and no output-dependent choice was made.

**Declared small-shell departures, for every arm alike:** no short convolution,
no output correction, no output gating or RMS normalization, one head, no value
expansion, no large-model block; the gate input is the token's own event
description (key one-hot 32, observed-value one-hot 8 zeroed at a query, and
WRITE/QUERY/IDLE flags) rather than a backbone hidden state. **An
implementation of a published rule inside this shell is not a reproduction of
that paper's model**, and these omissions are an adaptation, not a claim that
the extras are unnecessary.

## 5. Common shell

One memory block, one head, `d_k = d_v = 8`; 32 key identities, 8 value
identities. A 32x8 raw key table initialized standard normal from the
initialization stream and normalized in use; an 8x8 value table initialized to
the one-hot basis; an 8x8 readout initialized to the identity with zero bias.
**These are shared exactly across arms at a given seed**; gate tensors are
drawn from a separate stream so they cannot change the common tables or the
data order.

A query supplies only its key and event type — **no value field and no write
target**. Non-query readouts are unsupervised. There is no short convolution,
attention, pooling, position feature, residual route to logits, external
dictionary, auxiliary-state readout, or hidden state outside the documented
carry. **The readout sees `W q` only, never `P`.** Each query is read **after**
advancing its interval.

Full BPTT differentiates through every memory step and every auxiliary state;
**there is no `stop_gradient` anywhere in the memory path**, verified
statically. Fixed physical coefficients are constants and appear in no
optimizer leaf.

## 6. Task

Two deterministic families over 64 intervals with 16 queries and identical
event structure; the model receives no family identifier. Called
**MQAR-inspired recall** and **association revision**; neither is the published
MQAR dataset or protocol, and paired key-value events occupy one interval as a
deliberate simplification.

Layout: 0-7 write eight target associations; 8-15 eight distinct non-target
writes; 16-27 four groups of (write a selected target, query it, query an
untouched target); 28-55 twenty-eight distracting non-target writes; 56-63
query every target once. In **revision** each of the four selected targets is
rewritten with a uniformly sampled *different* value; in **recall** the
original value is repeated.

Four equally populated query categories: immediate selected, middle untouched,
late selected, late untouched. **Only "immediate" has a fixed age** (exactly
one interval); the others' age distributions are reported, not matched.
Verified on generation: 16 queries and 48 writes per sequence, equal category
counts, query value field absent, no query carries a write target, every query
has an oracle label and none outside queries, and latest-write retrieval is
exact. Oracle arrays are used for labels and metrics only; the model API never
receives the key-value dictionary. A few episodes are hashed before training.

## 7. Training and evaluation

Seeds **100, 101, 102**, paired across arms. **200 optimizer updates**, batch
**16** (8 per family), sequence length 64. Adam, `lr = 3e-3`, `b1 = 0.9`,
`b2 = 0.999`, `eps = 1e-8`, no weight decay, global gradient-norm clip
**1.0**, no dropout, **no learning-rate sweep**. Production dtype float32, no
mixed precision. RNG streams are **named** — initialization 0, train 1000000,
validation 7000000, held-out 9000000 — not derived from registration order.

Validation on **256 sequences per family**, fixed and shared, at updates 0, 100
and 200; these are learning curves, not selection opportunities. Held-out
evaluation on **512 sequences per family** from a separate stream, at the fixed
200-update checkpoint, and **opened only after every declared configuration
finishes** — an incomplete batch has no comparative verdict. Final parameters
and all scalar metrics are saved, including update-0 results so a lack of
learning is visible.

## 8. Outcomes and the predeclared rule

Primary: **revision-family macro accuracy**, equal weight per category. Also
recorded per seed and as paired differences: per-family and per-category
accuracy and cross entropy; early revision versus late retrieval; untouched
retention; accuracy at updates 0/100/200; state, auxiliary and gradient norms;
gate ranges and boundary occupancy; state and parameter counts; timings.

A **promising development signal** is predeclared as: at least **+1 percentage
point** mean primary accuracy over **each** literature arm, positive paired
primary differences for **all three seeds** against each, and no more than a
1-point mean regression in either revision-family untouched retention or
overall recall-family accuracy against either literature arm. This is a
screening rule, not a significance claim.

Attribution requires the **same-state inertial ablation**: if it matches or
wins, the prospective derivative is not credited. The matched-first-write delta
tests whether immediate write strength alone explains a result; it does not
equalize capacity.

**If all arms sit near the 12.5 % chance level, or the strong controls show no
useful learning, the study is inconclusive** and no superiority is described
from a chance-level difference. Seed-level uncertainty is reported; the
thousands of correlated queries are not treated as independent replicates. No
configuration may be dropped.

## 9. Analytical supplement of 16 September 2026, recorded

`ANALYTICAL_AUDIT.md`. These are analytical findings — **not evidence of better
retrieval** — and they introduce no architecture, coefficient, training
configuration or success criterion. The declared study is unchanged.

1. **Direction of the added term.** On a held write the law is
   `M Wddot + gamma Wdot + T Wdot k k^T + (Wk - v)k^T = 0`: velocity along `k`
   is damped by `gamma + T`, orthogonal components by `gamma`. This is
   curvature-dependent damping, the Hessian being `X -> X k k^T`. It does
   **not** mean the damping knows which associations a future query needs, and
   delta writes are already key-selective.
2. **Exact persistent/transient coordinates.** With `delta = gamma T - M`,
   `C = W + P/gamma` and `Z = -P/gamma`, `C` accumulates residual updates and
   `Z` is an exponentially filtered correction driven by the same residual
   evaluated at `W = C + Z`. On a no-write interval `C` is constant and `Z`
   decays with time constant `M/gamma`. "Persistent" refers **only** to
   no-write intervals, not immunity to later interfering writes. The readout
   stays `W q`; reading `C q` would be a different architecture and is not
   used.
3. **Incremental stability.** For `0 < M < gamma T` and two trajectories on
   **identical** inputs,
   `V = gamma/2 ||X + Y/gamma||^2 + M/(2 gamma delta) ||Y||^2` is positive
   definite and `Vdot = -m||Xk||^2 - ||Y||^2/delta <= 0`. Initial-state
   differences do not grow in this weighted norm. It does **not** compare
   different inputs or parameters, guarantee retrieval, establish
   bounded-input-bounded-state behaviour, or imply well-conditioned outer
   gradients; the nullspace permits non-decaying differences. Reported as a
   check before and after exact steps, adding no configuration.
4. **The comparator matches the write, not the query.** With
   `x = exp(-2/3)`, `y = exp(-2)`: `F11 = (x+y)/2`, `F21 = (x-y)/4 > 0`, so
   `beta_write = 1 - F11 = 0.6756238`,
   `beta_query = 1 - F11 - b0 F21 = 0.6060187`,
   `beta_settled = 1 - F11 - F21/gamma = 0.5811033`. The candidate's
   first-query effect is **smaller** than its write-end effect. The comparator
   keeps `beta = beta_write` exactly as declared: it is **not** relabelled
   "query-matched" and `beta` is not changed after seeing outcomes. A
   first-write magnitude match was never a full transfer-function match.
5. **Prior work that must be acknowledged.** For a fixed objective
   `Rdot = H(W) Wdot`, so the proposed continuous flow lies in the established
   family of **inertial optimization with Hessian-driven damping** —
   Alvarez, Attouch, Bolte and Redont (2002); after division by `M` the
   potential is `E/M`, the viscous coefficient `gamma/M` and the Hessian
   coefficient `T`. Attouch, Chbani, Fadili and Riahi (arXiv:1907.10536)
   implement Hessian damping through gradient changes without building a
   Hessian, so **the absence of a numerical Hessian is not itself a novelty
   claim**. Momentum DeltaNet already combines associative residual writes with
   an auxiliary momentum state, so **novelty cannot be claimed from having two
   memory states**, and equivalence or advantage cannot be settled by counting
   them.

The defensible question is unchanged: is this constrained residual-derivative
response a useful inner memory update compared with published delta and
momentum rules? This study can supply task evidence for that. It cannot
establish a new optimization principle, a unique biological derivation of fast
weights, or any improvement over S5 or Rawat — neither is trained here.

## 10. Cap and checks

One hard **600-second** cluster budget with a 30-second serialization reserve,
covering startup, checks, compilation, preflight, training, evaluation and
cleanup. The preflight measures synchronized steady-state cost for **all five
arms** plus evaluation, warms a step on its own optimizer output before timing,
records compilation separately, and reports a retrace rather than reading it as
slow physics. If the batch does not fit, it is **not started**, the measured
obstruction is reported and nothing is silently reduced.

Predeclared check tolerances: `1e-9` for the rank-one step against an
independent dense augmented-ODE matrix exponential in float64; `2e-5` over a
64-token float32 trajectory; `1e-5` relative for JVP against central
differences in float64 with an absolute near-zero branch; `2e-2` in production
float32 at **both** declared perturbations `1e-2` and `3e-3`, both recorded;
`1e-10` for the analytic limits and the literature reductions; `2e-5` for
literature parity against separate literal references; `2e-5` for chunked
streaming against an unsplit sequence.

## 11. Prepared, not launched

An S5 bridge — native S5, Rawat prospective-input S5, and the same S5 encoder
augmented with delta, momentum-delta or generalized prospective memory — is a
separate later batch. It is **not** launched here, this shell is not replaced
by a large backbone, and no claim about Rawat's published accuracy follows from
this study.
