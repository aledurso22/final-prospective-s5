# Combined prospective input + prospective recurrence: protocol

Committed **before** any numerical execution. Brief:
coordinator `outputs/combined_input_recurrence_handoff_2026_09_15/CODING_AGENT_BRIEF.md`.
Implementation parent `6a5bc7fadc324cef72cb2eaab60e6aa2dcfe76e6`, branch
`combined-input-recurrence`.

> **STATUS: implementation complete and committed; the comparative training
> batch is DEFERRED.** The priority handoff of 15 September 2026
> (`outputs/learned_timescale_priority_handoff_2026_09_15/CODING_AGENT_BRIEF.md`)
> reorders the experiments: the standalone recurrence with learned response
> timescales is tested first. **No combined run has been dispatched or
> executed**, and no cluster budget has been spent on it. This protocol and the
> code remain the specification for the later composition, which must be able
> to use either the fixed-timescale or the learned-timescale recurrent core.

## 1. Question

Does adding the physically constrained recurrent response to Rawat's
prospective-input S5 improve the short Speech Commands validation screen over
Rawat's prospective-input S5 alone? The hypothesis is **complementarity**: the
input correction changes current-input coupling, the generalized recurrence
changes the temporal response. This is an experiment, not an implication that
the combination must improve accuracy.

## 2. The equation, and where the clock is absorbed

Per stored complex mode, with the clock absorbed **exactly once**
(`Delta = exp(log_step)`, `B_c = diag(-Re lambda) B_tilde`, real poles clipped
at most to `-1e-4`):

```
rho T s'' + s' + r + T r' = 0 ,   r = j s - b x ,   j = -Delta lambda ,
b = Delta B_c
```

with `gamma_n = 1`, `T = 5`, `rho_0 = 0.9998`, `rho` bounds `[0.01, 0.9999]`,
one `rho` per stored mode shared with its conjugate, and mass **derived**,
`mu = T rho`. `gamma_n` is not reintroduced: `(Delta, gamma_n, rho)` is
input-output equivalent to `(Delta/gamma_n, 1, rho)`, so it only
re-parameterizes the already-learned clock.

Block realization, `q = (s, v)`, unchanged from `s5/gp_fixed.py`:

```
A_rho = [[ -j/rho,       -(1-rho)/rho ],      B_rho = [ b/rho ,
         [ -j/(rho T),   -1/(rho T)   ]]                b/(rho T) ]
```

The positive component map is unchanged for this fixed-`T` composition:
`kappa_0 = 1.5`, `c_s = c_d = 7.5`, `G_s = G_d = kappa_0/rho`,
`h = G_s sqrt(1-rho)`, `g_L = kappa_0/(1 + sqrt(1-rho))`. The
constant-conductance / additive-current and prescribed-prospective-source
qualifications remain in force. Composing this derived family with an
additional literature input filter is an **architectural choice**; the whole
composition is not uniquely forced by the circuit derivation.

## 3. The input correction

```
q' = A_rho q + B_rho (x + T_in x') ,   T_in = 5, FIXED, never learned
```

Continuous state transfer, zero prehistory:

```
G(p) = b (1 + T p)(1 + T_in p) / [ rho T p^2 + (1 + T j) p + j ]
```

a **product** of two prospective factors, not a single one. Held-token timing
as in Rawat: at the start of interval `k` the input jumps `x_{k-1} -> x_k`,
then one interval of held-input evolution, then the state is read.
`q_{-1} = 0`, `x_{-1} = 0`. Exact interval law:

```
A_bar = exp(A_rho) ,  B_bar = int_0^1 exp(A_rho u) B_rho du
J_in  = T_in * A_bar @ B_rho           <- the CONTINUOUS B, not B_bar
B_+   = B_bar + J_in ,   B_- = -J_in
q_k   = A_bar q_{k-1} + B_+ x_k + B_- x_{k-1}
```

Implemented in `s5/gp_fixed.py::mass_block_two_tap` /
`mass_scan_two_tap`, response `gp_rho_prospin`, arm `gp_rho_prospin`.

Declared consequences, each covered by a check:

* `B_+ + B_- = B_bar`, so the **DC response is unchanged**;
* the autonomous block poles are **untouched** - only input coupling and output
  residues change;
* at `T_in = 0` the law reduces to the plain generalized recurrence;
* at the mathematical boundary `rho = 1` the physical output reduces to
  **Rawat's alpha-P-S5**, because `v` decouples from `s` there. `rho = 1` is
  used as an algebraic limit; the trained parameter's numerical margin is
  `0.9999`.

Only `s` is read, through the existing `C` and conjugate convention. Native
learned `D x_k` and the residual path are unchanged: `D` is **not** prospected,
the `M = 0` arm's `d_x x` feedthrough is **not** added, and the drive and the
readout are **not** both prospected.

Streaming carries are `q` **and** the previous token. Both are reset at a new
independent sequence, including internal reset masks: clearing only `q` leaks
the previous sequence's last token through the second tap. Full BPTT reaches
the delayed-input path, including its dependence on earlier lower-layer
activations; there is no `stop_gradient` on `q` or on the delayed input.

## 4. Integration hazards addressed

1. **Projection whitelist.** `s5/response_projection.py` is now the single
   post-update projector, covering `log_response_gamma`, `log_response_rho` and
   `log_response_rho_only`. `experiments/gp/rawat_benchmark.py` imports it
   rather than keeping a second list, which previously did not recognize the
   rho-only leaf. Optimizer state is untouched by projection.
2. **Parameter policy.** `ARM_RESPONSE_LEAVES` is keyed by arm, so the
   rho-only arms are admitted with exactly one `(P,)` leaf per layer while a
   stray response leaf on a historical fixed arm is refused.
3. **Trainer arity.** The Speech Commands `train_step` signature is
   **unchanged**; the combined study uses its own runner with its own step
   function, so no existing caller breaks.
4. **Diagnostics.** `gp_rho_prospin` is extracted by its own two-tap block
   rules in `s5/substrate_diagnostics.py`, not by the one-tap mass rules.
5. **Metadata.** The runner writes the **executed** response policy and derived
   coefficients, not the static symmetric-reference table.
6. **Data access.** `SC.load_splits(..., splits=("train","val"))`; the test
   arrays are never opened. Fresh artifact directories.

## 5. Predeclared checks and tolerances

`tests/test_combined_input_recurrence.py` (float64) and
`tests/combined_float32_probe.py` (production dtypes, own process).

| check | tolerance |
|---|---|
| coefficients vs an independent dense real-pair reference (scipy `expm` + Gauss-Legendre, several `rho`, `P != H`) | `1e-10` rel, float64 |
| jump law vs an independently integrated trajectory (`solve_ivp`) | `1e-8`, float64 |
| `T_in = 0` and `rho = 1` reductions, forward | `1e-10`, float64 |
| gradients of shared parameters/inputs in those limits | `1e-9`, float64 |
| parallel scan vs an independent sequential recurrence, with carries, chunks and resets | `1e-10`, float64 |
| production `float32`/`complex64` against the same reference | `2e-4` rel |
| diagnostics vs an actual forward impulse / its DFT | `1e-5` / `1e-4` |

The float64 figures check the algebra; the `2e-4` figure is what production
dtypes achieve and is a resolution statement, not a loosened criterion. The
three wrong ways to form `J_in` (`T_in B_bar`, `T_in A_bar B_bar`, an extra
`Delta`) are checked to **disagree**, so the passing case is evidence.

## 6. The deferred training batch

Recorded as declared, and **not executed**: seeds 100/101/102, two arms, six
runs from scratch, ten epochs, batch 32, Stage 2 architecture (4 layers,
`d_model = 32`, SSM size 32, 8 HiPPO blocks, conjugate symmetry, ZOH), AdamW
`lr = 1e-3` cosine to `1e-6`, weight decay `1e-4` on all trainable parameters,
global clip 1, label smoothing 0.1, projection after the optimizer update.
Primary outcome: **validation accuracy after the fixed tenth epoch**, paired
combined-minus-Rawat per seed and the mean; best-validation scores reported
separately and never substituted for the endpoint. Expected counts, to be
verified rather than assumed: Rawat 35,050 parameters and 256 carried real
values across four layers; combined 35,114 parameters (+64 `rho` scalars) and
384 carried real values (+128 auxiliary). No imported `+0.3` pp threshold.

`experiments/gp/combined_study.py` implements this batch and
`bin/run_experiments/cluster_combined.sh` would launch it inside one 1,200 s
cap. Neither has been run. Whenever it is authorized, the runner must be
re-checked against the then-current recurrent core.
