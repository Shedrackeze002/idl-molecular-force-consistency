# Offset-profile scientific plan follow-up

Decision: **PASS** for the final document hashes below. Engineering clarification A introduces no blocking scientific issue and preserves the previously approved experiment. The earlier review remains retained at `docs/reviews/2026-10-02-offset-profile-scientific-plan.md`; its mathematical assessment, verified source baselines/grid, limitations and inspection-error record still apply.

## Final reviewed identities

- Design: `docs/superpowers/specs/2026-10-02-offset-profile-design.md`; SHA-256 `b7c2d5ca7fb0fa86a5fac99e2fd5db5b37301c35b290ac4e5d7a0d55369e8e63`.
- Plan: `docs/superpowers/plans/2026-10-02-offset-profile.md`; SHA-256 `bfad4a08f3f056313750a1286b9df4136cf08f9b88f0cf489b05b1303c518baf`.
- Source ancestry reviewed in the original review: `5e1c81f5322ecc7e92cfcc3238d285ffc1949387`.

Both final files were read and their supplied hashes independently matched. No implementation, numerical imports, model calls, training, tests, dependencies or historical raw/source modifications occurred. Only this follow-up review record was added; there were no new inspection errors.

## Amendment assessment

The explicit `profile_step` return of actual masked production gradient makes the full-support oracle comparison concrete. The original requirement to check the **unmasked** accumulated bias derivative before masking remains present and necessary; an already masked zero is not evidence for that check. Complete nonbias gradient comparison and independent NumPy Adam/outgoing calibration remain required. Cached kernel calls inside explicit microbatch loops preserve global incoming-state evaluation and make real call/exposure accounting visible; they do not authorize local microbatch centering.

All-field source-zero/projected-zero identities and invariant model RNG, sampler RNG and normalization fields strengthen the paired-start control. Ordered fitting IDs, input identity, ordinal/phase/update, frozen config/gate, state and NPZ hashes now tie each projection to its accepted state and score commit. This reduces the chance that correct-looking projection statistics from a different support or update could receive scientific endorsement.

Retaining binary ready outputs before finite validation and strict JSON appropriately preserves NaN/Inf or other failed-return evidence. The added ID/receipt/root/gate/child failure regressions make the original invalid-execution rules reviewable without turning partial measurements into successful trials.

The exact source-path count reconciles: original source-map file + 347 members + 247 spike raw files + 6 declared extras = **601**. The separately config-pinned spike inventory supplies membership authority rather than becoming an extra payload member. Implementation must independently verify the actual exact set, its bytes and the inventory binding before numerical entry and at closure/reporting; this follow-up validates the specified contract, not a future manifest.

The intervention still consists of one 6000-update width32 trial with global 21-atom bias profiling, unchanged rate/weights/supports, original unprojected raw gates, 63 scored observations and the same total cap/reserve/once-only child. Historical-reference, fitted-anchor, construction-effect and sparse-observation limitations are preserved. No additional scientific arm or favorable selection is introduced.

No further scientific plan repair is required. This PASS is specific to the two final hashes above; both final independent reviews and subsequent implementation/prelaunch gates are still required before execution.
