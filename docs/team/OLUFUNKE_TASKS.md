# Olufunke Abodunrin: shape recovery

Updated 15 October 2026. PR #4 evaluator is merged with authored commits preserved. PR #6 follow-up input validation/backtracking/fingerprint/documentation repairs are also merged; do not repeat those fixes. Accepted implementation/analytic controls are not trained-model recovery results.

1. Agree the shared checkpoint adapter with Idris and integrate MACE and later E3x: fixed atom order, Z/symbol boundary,Å, (N,3)forces in eV/Å,optional energy,precision, artifact identities.
2. Implement and measure the validation-only no-relaxation control and symmetry-aware nearest-reference comparator under the frozen starts/stopping policy. Keep test evaluation separate.
3. Evaluate the exact A1-selected E3x checkpoint for A2 when available. Recovery must not select models/configurations/strengths.
4. Write recovery methods, outcomes and limitations; retain unavailable/failed rows. A force-field fixed point is not automatically an energy minimum.

MACE Run A is now a verified same-split reference. Obtain original model/source artifacts and verify access before integration claims. Two plan reviews precede new work. [Status](PROJECT_STATUS.md) · [access](../reproduce-oct15-results.md).

---

## Stage 2 report generator: assignment, 2 November 2026

Shedrack assigned the reporter side of the Stage 2 movement-cap study to Olufunke: `stage2_capped/report.py` and its tests, which turn saved result rows and the study's label records into labelled tables, and reproduce the accepted 50-row recovery table from the accepted records. Read [the handover](OLUFUNKE_STAGE2_REPORT_HANDOVER_2026-11-02.md) for access, the required printing rules, what not to touch and the review process. This is an assigned task, not a completed contribution: record it in the [contribution ledger](CONTRIBUTIONS.md) only after two independent reviews accept it. Plan acceptance does not establish a model result; no real-model run exists yet.
