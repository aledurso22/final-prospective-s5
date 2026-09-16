# Generalized prospective memory around the delta boundary — report

**Status: prepared, amended per reviews of `535fb02` (protocol §12) and
`f13295c` (§13); authorized for one cluster launch; not yet executed.**

Check scope, stated precisely: the actual initialized tree is checked for
value/shared-gradient nesting AND tangents in float64; in the float32 probe it
is checked for tangents only, while float32 value/shared-gradient nesting uses
the nonzero-gate stress fixture. No local numerical run, no cluster launch.
Every result section will be filled only from cluster output.

Proof audit: `docs/META_DELTA_PROOF_AUDIT.md` (no algebraic discrepancy found;
two executed-precision points handled). Protocol: `docs/META_DELTA_PROTOCOL.md`
(proposed, for static review). Unresolved issues are listed in protocol §11.

The completed adaptive-memory and S5 results are unchanged and are not
re-interpreted here: the earlier restricted family (`rho < 1`) remains a
negative result for that family, and its trained rho values (~0.66) were not
pressing against the bound.

## Verdict structure, fixed before execution

The literature verdict (Momentum and Gated DeltaNet) and the matched-delta
verdict are reported together. Two further verdicts are reported separately:
the TSS Eq. (17) comparison, "applied directly to the fast weight;
applicability-limited", and the heavy-ball comparison, "separately trained
family; gamma and M matched at initialization only".

None substitutes for another, and none is causal attribution to `T Rdot`.

## Results

*(empty)*
