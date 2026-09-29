# Fresh whole-branch review and verified fixes

A fresh gpt-6-astra reviewer reviewed a832e47..2a7af87 against the design, initial plan and two-review-approved learning amendment. The reviewer independently reran53tests and confirmed the user-owned proposal changes were preserved. No Critical or deferred Minor issue was found. Initial readiness: with fixes.

## Important findings

1. Resuming fitting restored optimizer/sampling state but reset historical best score/update and validation history. A parent synthetic end-to-end regression selected update3 instead of the superior earlier update1. The checkpoint now preserves integrity-checked selection controls, the corresponding fullbest state and historical validation results. Resume restores them; exactties retain the earlier update. Older checkpoints remain readable for inference, but fitting resume without selection history is rejected.
2. For symmetric methane, an O(3) symmetry permutation with slightly lower numeric RMSD could hide exact stored-coordinate equality and skip hardlabel rejection. The parent reproduced the failure. Exact species/coordinate/graph identity now uses its own deterministic coordinate permutation; rawforce conflicts use that permutation. Near-geometry force differences retain the existing diagnostic-only policy.

Both regressions were observed failing (2failures in14.47s), then passing (2passed in15.96s). Final fullsuite:55passed in38.59s. Corrected duplicate audit on8874selected archive records stayed at zero accepted pairs, unresolved identities and conflicts. Fresh aspirin preparation reproduced canonical identity4b9f220201d6483e2b2a8da74c10330c3762942ff4c6fabe6389ee27249c43aa. No extra research training was performed. Parent fixes were verified by regressions and the fullsuite; no redundant re-review was requested.

## Reviewer-declined scope and parent rulings

Future direct/consistency/MACE arms and test evaluation are separate study milestones. Actual GPU execution/cross-platform reproducibility is unavailable and remains unverified. Both learning-gate failures are honestly retained outcomes under the approved stop condition. These scope decisions and their costs are recorded in `reports/implementation-execution-record.md`; none is claimed complete.

The two review defects did not cause the two observed fresh-run learning failures. The exact-conflict case is a synthetic implementation diagnostic, not evidence of conflicting Landscape17 labels. Recorded run metrics/cost belong to their archived code/config fingerprints, before these fixes. Future measured costs must use a new run identity.
