# Scientific follow-up: amended team handoff

3 October 2026. Verdict: **PASS. No material unresolved scientific or allocation finding.** This is a bounded follow-up to `docs/team/reviews/2026-10-03-handoff-scientific.md`, reviewing `reports/execution/team-handoff-2026-10-03/amendment-A.json` and the exact handoff diff against the preserved initial draft. It clears the amended prospective handoff only, not downstream implementation or production launch.

## Exact final reviewed files

- `docs/team/PROJECT_HANDOFF.md`: SHA-256 `12f23e73c807b8f422cdfbdcca822b49794f0ce66e3af72138a0c8f69689d379`.
- `docs/team/CONTRIBUTIONS.md`: SHA-256 `37974f5c2348e85919d2395dd1e9cb680c1662d1c1d2cdc325d932a008c9c7da`; unchanged from the initial review.

## Resolution and assessment

All three initial scientific suggestions are addressed: original force-MSE fractions and the strict below-10% gate are explicit; held-out connections retain seen-reference-minimum flags with angstrom RMSD; and physical metrics specify eV loop work, eV/angstrom disagreement, labeled squared errors, and separate seed variability from connection-level uncertainty. These refinements retain the original scientific question, failed numerical outcomes and independent connection units.

The additional engineering clarifications support scientific validity: Shedrack integrates one adapter agreement with atom order, species encoding, units, energy gauge/reference and differentiability/force correspondence; missing energy remains unavailable through a force-only metric wrapper; early MACE fixtures use the training-only loader and verify record/ordering/unit/reference round trips. Launch-checkout provenance and source/dependency immutability are explicit. Onboarding completion requires verified repository/artifact access and persistence, with synthetic fixture work allowed beforehand. No owner assignment, numerical setting, model comparison, leakage control or measured-result status changed.

The four substantive technical streams still give Shedrack the largest training/integration share. The contribution ledger still makes no advance attribution or fabricated success claim. Remaining metric/evaluator/adapter details belong in each workstream's specifically reviewed implementation plan, as stated by the document and `AGENTS.md`.

## Limits and preservation

No broader artifact audit, model import, test, training, prediction, implementation, frozen-source/log/Obsidian change or external communication occurred. I hashed only the two final drafts and inspected the preserved-draft diff and amendment record. The initial review and drafts remain unchanged. The diff command returned its ordinary difference exit code, so the chained hash display did not execute; this follow-up separately computed and asserted both hashes against the supplied final candidates before writing this record. No discrepancy was found.

Any later material amendment requires a separate bounded follow-up at its new exact hashes. Downstream implementation/experiment plans require their own two independent reviews.
