# Energy-offset profiling: engineering clarification A follow-up

Verdict: **PASS** for the final plan/design below. Clarification A makes the initial engineering review's concrete implementation checks explicit and introduces no material engineering contradiction. It preserves the one-intervention protocol and its existing execution/reporting gates. Implementation and numerical validity are not established by this plan verdict.

Exact hashes independently read and confirmed:

- `docs/superpowers/plans/2026-10-02-offset-profile.md`: `bfad4a08f3f056313750a1286b9df4136cf08f9b88f0cf489b05b1303c518baf`.
- `docs/superpowers/specs/2026-10-02-offset-profile-design.md`: `b7c2d5ca7fb0fa86a5fac99e2fd5db5b37301c35b290ac4e5d7a0d55369e8e63`.

Initial engineering review remains retained in `docs/reviews/2026-10-02-offset-profile-engineering-plan.md`, SHA-256 `1034f6eec22e414eb09e25fb42d03e1178b2724cd24903403e4e596282d0f901`.

## Amendment verification

`profile_step` now explicitly returns `(candidate, actual_gradient, receipt)`. The oracle receives the actual masked production derivative without requiring another production proposal. Script-local per-microbatch callbacks wrap raw prediction, weighted-gradient and energy-only kernels, making ready/returned/finite/unknown work and partial artifacts observable rather than hiding multiple calls in an aggregate frozen helper. The original unmasked-bias tolerance remains a separate mandatory check before masking.

Source unprojected0 and intervention projected0 have separate full-field identities and local initial hashes. The source checkpoint envelope/payload and original parameter/optimizer hashes are retained; the independent source blueprint binds schema. Model RNG, sampler RNG, normalization and update fields participate in audits, with unchanged RNG/normalization fields required in subsequent states. This avoids applying original source initial-hash assumptions to a projected0 checkpoint. Child and saved audits explicitly check every state field.

Projection and accepted-step receipts now bind ordinal, phase, incoming/outgoing update, ordered fitting IDs/input hash, gate/config identity, state hashes and raw NPZ hash. Score/history/immutable commit receipts link back to accepted update/projection receipts. Initial/training/warmup/oracle-reference phases remain distinct. Binary ready outputs precede finite validation and strict JSON, so NaN/Inf failure artifacts remain representable and retained.

The declared injected regressions cover NaN output, swapped IDs/receipts, orphan commits, root identity/symlink substitution, child nonzero exit/missing completion and substituted gates. These extend the existing late-call, unknown interruption, duplicate root/probe, source/provenance drift, original-baseline and conservative incomplete-report requirements. They must exercise real file/control boundaries and inspect artifacts/work/endorsement, not only error messages.

The source-set arithmetic is consistent:347 original source-map members plus the map file itself plus247 spike raw members plus6 declared extras gives601 paths. The spike raw inventory is separately pinned by configuration SHA and validates membership; it is explicitly excluded from the payload-member count. Implementation review must confirm exact sets and reject overlaps, omissions, extras and changed bytes rather than rely on counts alone.

The180-second reserve remains part of the3600-second budget; the child receives actual remaining parent time minus closure reserve, and the independent3660-second supervisor/TERM10-second/KILL/no-retry protocol is unchanged. Original raw baselines,63 scored states,6000 accepted updates,6001 accepted calibrations, actual oracles1/3001 and one child remain unchanged. Complete execution endorsement still requires saved state/receipt/source/gate/oracle/child/cap closure; raw threshold arithmetic is separate from that validity gate.

## Remaining gates and inspection record

No blocking engineering issue remains at plan stage. Before launch, independently review the implementation against the amended requirements, retain focused/full-suite evidence for the exact bytes, freeze source/launch/review/log identities and bind closure to the original gate. After any numerical attempt, retain its successful/failed/interrupted evidence and perform saved-only audits/independent result review without retry.

This follow-up read the final documents and their hashes and wrote only this review record. No source/config/runtime/raw edits, tests, model construction/calls, gradients, Adam proposals, training, checkpoint restoration or installation occurred. No new inspection error or truncation occurred; the initial review's inspection-output truncations remain preserved there.
