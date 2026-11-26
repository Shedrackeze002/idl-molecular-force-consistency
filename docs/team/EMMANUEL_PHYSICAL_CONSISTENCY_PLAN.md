# Emmanuel physical-consistency evaluator plan

Owner: Emmanuel Aaron. Scope: the first bounded physical-consistency and cost deliverable from `PROJECT_HANDOFF.md`.

## Inputs and contract

The evaluator consumes the shared `evaluation.adapter` arm contract. Positions are angstrom, forces are eV/angstrom, scalar molecular energy is eV, atom order is preserved, and unavailable energy remains unavailable. This change does not select checkpoints, open test data, or alter the validated supervised scorer.

## Metrics

1. At named geometries, compare supplied forces with a central-difference negative energy gradient. Report MAE, RMSE, maximum absolute disagreement and MSE in explicitly labelled units.
2. On explicitly closed, frozen polygonal loops, integrate `F dot dr` with the trapezoidal rule and report signed and absolute work in eV. Open paths are refused rather than silently closed. A frozen uniform subdivision count supports quadrature-refinement controls without changing the accepted integration rule; coarse nonzero work alone is not evidence of nonconservativity.
3. Measure force-inference wall time after separately reported warmup, with a synchronization hook for accelerator backends. Comparisons are valid only on identical hardware, precision, batch, warmup, synchronization and software.

## Acceptance checks

- A separable harmonic potential has near-zero energy/force disagreement and loop work.
- A rotational nonconservative field has nonzero loop work.
- A force-only arm records energy-gradient comparison as unavailable while retaining valid loop work.
- Nonfinite supplier output is an explicit failed result with its adapter refusal.
- Results are JSON-safe and record configuration, units and evaluation counts.
- Nonfinite derived arithmetic is routed to an explicit failed result, and empty timing batches are refused.

## Claim limits and dependencies

Sampled loops and geometries can expose local inconsistency but cannot prove global conservativity. The included timing is a smoke measurement, not a cross-model benchmark. Trained-model comparison remains blocked on reviewed checkpoint adapters, accessible checkpoints, frozen validation geometries/loops and a matched-hardware protocol. Main-test evaluation is out of scope until model selection and protocols are frozen.

Before production evaluation, this plan and any material changes require the two technical reviews specified by `TEAM.md`.
