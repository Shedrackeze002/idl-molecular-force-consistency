# Shape-recovery diagnostics: shape-recovery-fixture-diagnostics

Evaluated branches: 4 (held out: 4)

Category totals over held-out branches: matched_reference 2, unmatched_converged 1, ambiguous 0, unresolved 0, nonconverged 1

| molecule | connection | denominator | converged | matched_reference | unmatched_converged | ambiguous | unresolved | nonconverged | rates | availability |
|---|---|---|---|---|---|---|---|---|---|---|
| analytic | recover | 2 | 2 | 2 | 0 | 0 | 0 | 0 | matched_reference 1.0, unmatched_converged 0.0, ambiguous 0.0 | available |
| analytic | wrongshape | 2 | 2 | 0 | 1 | 0 | 0 | 1 | matched_reference 0.0, unmatched_converged 0.5, ambiguous 0.0 | available |

Disclaimer: the outcome taxonomy has five categories (matched_reference, unmatched_converged, ambiguous, unresolved, nonconverged); the seven-shape shorthand is not used. A converged endpoint is a force-field fixed point, not a minimum: it is never described as a minimum. A nonconverged run is never counted as a wrong shape. These numbers come from an analytic fixture arm, not from a trained model.
