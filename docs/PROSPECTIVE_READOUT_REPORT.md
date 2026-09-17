# The prospective readout: Stage A's redundancy conclusion and Stage B's verdict

17 September 2026. Branch `prospective-readout`. This document holds the two
results together, because neither is readable alone: Stage A says which parts
of the proposed readout family are new, and Stage B measures whether the part
that is new helps on the read path.

## 1. Provenance

| | |
|---|---|
| Stage B dispatch | `20260917-233715`, commit `3e0fcd6`, operational **PASS**, 104 s of 600 s |
| Failed first dispatch | `20260917-232635`, commit `b926b6f`, FAILED in the check suite, preserved (protocol s12) |
| Checkpoints (read only) | four native endpoints of `prospective-temporal-response/20260917-163003`, hashes identical before and after, `SOURCE_UNCHANGED=True` |
| Training | none. The backbone is frozen; nothing was optimized |
| Arms | 77, of which 23 exercise `gamma < 0` on the production path; 0 arms needed feasibility repair |
| Data | 32 episodes per family on a fresh stream, split into declared selection and confirmation halves |

The digest for this run was initially empty: `readout_probe_summary` defined
`main(run_dir)` with no `__main__` block, so the launcher recorded
"digest: complete" over nothing. Fixed in `2820072` and regenerated read-only
from the saved JSON; no result was recomputed.

## 2. Stage A: the redundancy conclusion

Verified exactly in rational arithmetic, re-verified in float64 on the
cluster. Carry means state held between tokens beyond what the native step
already has; `W_(t-1)` is free, since the step holds it while forming `W_t`.

| Readout | Family position | Extra carry |
|---|---|---|
| Native identity `X = W` | `M = 0, gamma = h, T = 0` | 0 |
| Two-tap `(1+kappa)W_t - kappa W_(t-1)` | `M = 0`, `gamma + T = h` | 0 |
| `U`-lookahead | — | 0 |
| Literal TSS, `T != h` | `M = gamma = 0` | 1 matrix |
| Generalized interior | `M > 0` | 2 matrices |

**Partly redundant, and it has to be said plainly.** The FIR line of the
family — `M = 0`, `gamma + T = h`, any `kappa`, including `kappa > 1` — is
computable from what the native step already holds. So is the `U`-lookahead.
Neither adds carried state; they are cheap controls, not new mechanisms. The
only thing not computable from `(W, U)` at a single token is the **smoothed
velocity**: first order (literal TSS, one extra matrix) and second order
(`M > 0`, two extra matrices). Stage B therefore tests exactly one thing
worth testing: whether paying for that carried state buys prediction.

## 3. Stage B: the declared verdict

**INDETERMINATE on all four checkpoints, on both targets.** Not a pass, and
not a STOP.

The declared rule requires the interior to beat the best of the literal-TSS
line and the `U`-lookahead by 0.01 on at least 3 of 5 offsets, at matched
search budget, chosen on the selection half and read on the confirmation
half. The declared power rule (protocol s11) says an offset whose pooled
standard error exceeds the 0.01 margin contributes `indeterminate` rather
than a pass or a fail. The observed pooled standard error was **0.014 to
0.021 at every offset on every checkpoint** on the primary target, and 0.016
to 2.17 on the secondary. Every offset was therefore indeterminate, every
checkpoint's count became unreachable in either direction, and the overall
verdict is a failure of resolution.

Under the rule as it stood before the second reporting change, all twenty
offsets would have counted as "not a win" and the run would have printed a
**STOP** — a negative verdict on data whose declared power test it fails.
That change earned its place.

## 4. What the numbers show, as observations

The verdict above is what the declared rule returns, and it stands. But the
recorded numbers say something the rule cannot express, and suppressing it
would be its own kind of dishonesty.

Best interior arm versus best baseline on the primary target, confirmation
half (1.0 = no better than reading the native fast weight):

| Checkpoint | offset | interior | baseline | difference |
|---|---|---|---|---|
| 500 | k1 | 0.8538 | 0.3569 (lookahead 0.1) | **+0.4969** |
| 500 | k16 | 0.7817 | 0.4904 (lookahead 0.25) | **+0.2914** |
| 501 | k1 | 0.8996 | 0.3342 (lookahead 0.1) | **+0.5654** |
| 501 | k16 | 0.8057 | 0.4824 (lookahead 0.1) | **+0.3233** |
| 502 | k1 | 0.8254 | 0.2480 (lookahead 0.25) | **+0.5774** |
| 502 | k16 | 0.8412 | 0.2702 (lookahead 0.5) | **+0.5710** |
| 503 | k1 | 0.8199 | 0.4845 (lookahead 0.25) | **+0.3354** |
| 503 | k16 | 0.8037 | 0.3706 (lookahead 0.25) | **+0.4331** |

Positive means the interior is **worse**. It is worse at every offset on
every checkpoint, by 0.29 to 0.58 — twenty-nine to fifty-eight times the
decision margin, and more than ten times the two arms' combined standard
error. The interior also loses to the literal-TSS line (best 0.69 to 0.78)
everywhere, not only to the lookahead. The best predictor of the rolled-
forward fast weight, on every checkpoint and nearly every cell, is the
`U`-lookahead at small lambda — **a carry-free control that Stage A already
showed adds nothing to the native step**.

The direction is unambiguous; what is unresolved is only whether the
declared 0.01-scale comparison could have been made at all, and it could not.

## 5. A methodological flaw in our own rule, recorded not repaired

The declared power rule compares the **standard error of the metric** with
the decision margin. The scientifically apt comparison is between the
**observed difference and the standard error of that difference**. A gap of
0.50 with arm standard errors near 0.015 is resolved beyond any reasonable
doubt, yet our rule labels it indeterminate, because the rule was written to
protect a 0.01-scale claim and cannot recognize a 0.50-scale one.

This is recorded as a flaw in the rule, not repaired after the fact. No
re-evaluation of this run is authorized or performed; the INDETERMINATE
verdict stands as the declared rule returns it, and the magnitudes above are
reported as observations rather than promoted to a verdict. Turning an
observation into a verdict requires a rule declared before a run, which this
one was not.

## 6. Why the carried state did not pay, mechanically

The idle roll-forward is exactly `target_k = alpha^k W_t - c_k U_t`. The
measured idle `alpha` is **0.26 to 0.40** across the four checkpoints, so
`alpha^k` collapses fast: the decay gap `1 - alpha^k` is already 0.60 to 0.74
at k=1 and above 0.99 by k=4. The target is therefore dominated by the
momentum term `-c_k U_t`, which the `U`-lookahead reproduces exactly and for
free, while the part the filters can extrapolate — the decaying `W` term — is
nearly gone by the second step. Paying two matrices of carry to smooth a
velocity is paying for the term that vanishes.

The declared 0.05 branch **did not fire on any checkpoint**: the best
lookahead arm was within 0.05 of the primary target at 0 of 5 offsets
everywhere, so the primary carried the verdict throughout and the secondary
rule was reported beside it. The idle roll-forward was not trivially
predictable. The reason is visible in the gate statistics: `beta` varies
enormously within a checkpoint (mean 0.27 to 0.35, min 0.011, max 0.71), so
no fixed lambda matches it token by token.

On the secondary target — the actual `W_(t+k)`, with real writes — nearly
every arm scores at or above 1.0, and the standard errors reach 2.17. Nothing
in this study predicts the actual future fast weight better than reading the
present one.

## 7. Scope

A negative reading here is a statement about the **seven declared interior
points**, on this task, these four checkpoints and these five offsets — not
about the interior of the coefficient family as a whole, and not about
architectures beyond this backbone. Nothing here authorizes a trained
comparison; passing would not have, and this outcome certainly does not. The
Stage A redundancy conclusion, by contrast, is algebraic and holds wherever
the family is defined.

## 8. The two together

Stage A: the cheap part of the family is redundant with the native step, and
the only genuinely new part is carried smoothed velocity. Stage B: on the
read path, at the declared margin, that carried state could not be shown to
help, and the recorded numbers run clearly against it, with a carry-free
control the best predictor everywhere. The prospective-readout line has not
produced a positive result, and the honest assets from this branch are the
prior-art identification (the two-tap at `kappa = mu` is Nesterov gradient
correction, verified exactly), the corrected coefficient domain, and this
pair of negative and unresolved findings.
