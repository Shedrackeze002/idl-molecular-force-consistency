# Scientific follow-up: late-rate amendment A

Date: 2026-10-01. Reviewed changes only in the design, implementation plan, and `docs/reviews/2026-10-01-late-rate-author-rulings.json`. No code execution or training performed.

**PASS for implementation.** Remaining material scientific issues: none. Amendment A preserves the parent, seed, support, loss weights, scalar rates, optimizer-state inheritance, update budget, acceptance baselines/gates, order, scoring grid and caps. There is no scientific protocol drift.

- S1 is resolved: the two-stage schedule is explicitly one bundled intervention; phase-specific causal or optimality conclusions are excluded.
- S2 is resolved: endpoint contrasts have explicit orientation and units, with unavailable zero-denominator ratios carrying a reason.
- S3 is resolved: saved-only analysis cannot amend the protocol without a new prospective reviewed amendment.
- Engineering safeguards strengthen the intended comparison: both arms independently check both boundaries, exact score grids and common starts are required, and partially persisted/failed trials cannot yield complete contrasts. New executable provenance is correctly separate from historical ancestry rather than requiring different code revisions to have equal fingerprints.

The review gate remains limited to implementing this one-seed fitting diagnostic. Final implementation/verification review, pre-registration and all scientific limitations in the initial review remain applicable; no full-pilot or main-hypothesis clearance is implied.
