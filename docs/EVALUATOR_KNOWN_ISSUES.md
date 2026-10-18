# Evaluator known issues

## Frozen relaxation v1: scale cap and displacement, 18 October 2026

The numerical procedure uses candidate = positions + scale × force. Its growth cap limits the scale to 1.0 Å²/eV after growth, not coordinate displacement to 1 Å. The generic validator also permits a larger initial scale, although the study freezes the initial scale at 0.2 Å²/eV. Docstrings claiming an absolute per-trial displacement cap or guaranteed landing on a minimum are inaccurate. The protected relaxer is retained unchanged for reproducibility.

Trial convergence is checked before force-direction agreement. A trial below the force threshold is accepted immediately, even if its geometry is wrong. Geometry matching correctly remains a separate gate. No energy decrease is required. For a purely local model in a force-free separated regime, force-only convergence can be vacuous; this does not imply reference recovery.

The runner retains endpoint coordinates, force/step counts and stop facts, but drops the relaxer's returned accepted-step list, force history and final maximum component. Historical endpoint records cannot reconstruct the trajectory. A separately reviewed observational trace is proposed; it has not been implemented or run.

Reference-start controls can measure what this procedure reaches from a reference within its cap. Failure cannot establish that no nearby stationary point exists. Any future solver change needs a separate protocol and must retain the original negative outcomes. See [the dated diagnostic addendum](../reports/E3X_RECOVERY_DIAGNOSIS_REVIEW_ADDENDUM_2026-10-18.md).
