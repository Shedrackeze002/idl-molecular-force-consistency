# Team handoff engineering follow-up review

Date: 3 October 2026. Bounded review of Amendment A against the preserved initial draft.

## Verdict

**PASS for the final amended handoff and unchanged empty ledger.** No material issues remain in the reviewed diff. This approves presentation as proposed ownership/onboarding guidance only. It does not approve implementation, assignments, a numerical launch, external publication or claimed contributions.

## Exact identities

| Artifact | SHA256 |
|---|---|
| Final `docs/team/PROJECT_HANDOFF.md` | `12f23e73c807b8f422cdfbdcca822b49794f0ce66e3af72138a0c8f69689d379` |
| Unchanged `docs/team/CONTRIBUTIONS.md` | `37974f5c2348e85919d2395dd1e9cb680c1662d1c1d2cdc325d932a008c9c7da` |
| Preserved initial `reports/execution/team-handoff-2026-10-03/drafts/PROJECT_HANDOFF.md` | `14f3f74b140b87717b54d8bfd37291dd1efc6a7ac9c347ebea37e4fa154bf81f` |
| Initial `docs/team/reviews/2026-10-03-handoff-engineering.md` | `c80f47d2f66bc4937c1d53685bade6622149215248a2292f8bc872e560461154` |

Hashes were read from actual file bytes and match `reports/execution/team-handoff-2026-10-03/amendment-A.json`. The preserved and current contribution ledgers compare equal.

## Diff assessment

- The shared adapter now has Shedrack as agreement/integration owner and explicitly records species encoding, atom order, molecular/batched shape, scalar energy, energy gauge/reference, coordinate differentiability, energy/force correspondence and deterministic failures. These resolve the initial engineering review's interface clarifications without pretending a concrete implementation exists.
- A force-only metric wrapper is expressly required because the existing paired scorer needs finite energy. Missing energy remains unavailable; count/pooling conventions remain consistent with the validated scorer. This avoids a zero-energy substitution or an inappropriate call through the current paired-energy API.
- The concrete training-only loader and round-trip ID/order/unit/energy checks are now specified for early MACE export fixtures. The distinction from the validation-opening bundle loader is accurate against the source read in the initial review. Separate MACE dependencies and training-only preprocessing controls remain intact.
- The launch checkout now explicitly prohibits HEAD-changing operations and source/dependency writes during a bound trial, including documentation-only commits. Separate working checkouts preserve parallel development. This addresses both code-file fingerprint and revision identity, rather than protecting merges alone.
- Repository/storage access, manifests, checksum instructions and persistent output locations must now be verified and recorded before onboarding is marked complete. Synthetic work may begin earlier. No actual access or reproduction result is invented.
- Scientific clarifications label the force fractions as MSE, the gate as strictly below 10%, structural RMSD in angstrom, loop work in eV and force disagreement in eV/angstrom. Seen-reference-minimum status is preserved, and seed variability is distinguished from whole-connection uncertainty. These additions are coherent with the previously reviewed engineering boundaries and defer exact analysis policy to a later plan.

The diff changes only explanatory progress wording and boundary/dependency details. The four responsibility rows, roles, model comparison, scientific controls, contribution-authorship guidance, review requirements and proposed milestone sequence are unchanged. The ledger remains Proposed with placeholders and no credited work. No amendment undermines the initial PASS or introduces new implementation authorization.

## Review scope

Read only Amendment A, the exact handoff diff against the preserved draft, final/preserved hashes and ledger byte equality. Relied on the initial review for source semantics already examined there; did not repeat a broader source/numerical audit or fresh web lookup. No lookup failed in this follow-up. No model, test, training, installation, implementation, other-document/evidence-log/Obsidian edit or external action occurred. Only this separate reviewer-owned follow-up was created; the initial review remains preserved.
