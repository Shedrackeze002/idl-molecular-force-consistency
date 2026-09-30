# Research findings to carry into the paper

These are navigation notes, not a separate numerical evidence source. Read the exact values, source/run identities and limits in [the repository paper evidence log](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/reports/PAPER_EVIDENCE_LOG.md).

- DATA-002: chemical graph mapping and an oxygen permutation can resolve an endpoint that looks far apart under index-fixed alignment. This motivates explicit atom-permutation handling.
- DATA-005: unit provenance needs source/calculator evidence; XYZ headers alone are insufficient. Hessian conventions are a separate issue.
- DATA-006/006A: a paracetamol terminal remains outside the frozen endpoint threshold; it is reported and gated rather than dropped or accommodated by a looser threshold.
- DATA-007: two aspirin filename endpoint pairs differ from frozen geometric matching. Preserve named and resolved identities; this is a conditional archived-data observation, not a claim of corrupted data or new physical connectivity.
- DATA-008: whole-connection splitting does not imply unseen minima or independent observations. All aspirin reference minima are training-seen in the frozen first split.
- IMPLEMENTATION-001: untrained symmetry and derivative checks can establish implementation behavior before any learned-accuracy claim. A zero initialization needs deliberately nontrivial parameters for these checks.
- ENGINEERING-001/001A/002: energy loss can fall while force fitting remains weak. Compare absolute force error with zero-force predictions and preserve failed learning gates; random-initialization reductions can overstate learning. The one reviewed initialization adjustment did not resolve acceptance.

All are pre-comparison findings. None establishes that consistency training improves molecular shape recovery. New findings should be logged in the repository first, then linked here when useful.

## Diagnostic follow-up, 30 September 2026

- DIAGNOSTIC-002: exact linear readout fitting on two fixed backbones leaves substantial real-force error. This isolates a conditional feature limitation, not total network capacity.
- DIAGNOSTIC-003: an easy synthetic teacher sharing the student initial backbone is learned under the fixed pipeline; synthetic success does not prove DFT fit or generalization.
- DIAGNOSTIC-004: using all16 frames per update did not close the real-force gap. Preserve this failure alongside both earlier failed attempts; full pilot stays pending.
- IMPLEMENTATION-004/004A: the original run lacks intermediate full-subset histories; figure generation first failed on that missing file. Plot only verified endpoints, never interpolate unseen metrics or substitute minibatch loss.

[Diagnostic report and exact evidence](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/reports/force-learning-diagnostics.md) explains the controls, identities and claim limits.


## Coverage comparison, 30 September 2026

- IMPL-COVERAGE-001/002/003: label-blind frozen subset selection, full-batch-equivalent sequential accumulated gradients, fixed-cap controls and input-only exhaustive checkpoint probes were verified before production. Failed engineering/fixture checks are preserved alongside passing checks.
- COVERAGE-001/002/003 PRE/RESULT and COVERAGE-MATRIX-RESULT: original16 replay, diverse16 and nested diverse128 completed all fixed budgets but failed their joint force/energy acceptance gates because force error remained high. Energy fitting passed its individual threshold. This negative result is part of the paper evidence, not a discarded trial.
- Larger coverage did not close the force-fitting gap under this model/loss/budget. One seed, changed difficulty and 8x frame exposures per update for128 prevent attributing differences solely to sample count or making generalization/capacity claims. B fits only2original anchors; its14other anchors are still global training records.

[Coverage report, complete metrics, timing and checksums](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/reports/data-coverage-diagnostics.md). Repository evidence remains authoritative. Next is a separately saved, twice-reviewed architecture/loss-gradient plan; full pilot remains pending.


- IMPL-COVERAGE-004: independent review verified the measured artifacts and identified a handled-exception retention bug. Three failed regressions were repaired and independently cleared; all82tests pass. Numerical trials were preserved without rerunning training. Review records remain retained.


## 30 September 2026 — Frozen representation and loss gradients

FINDING-GRADIENT-001: symmetry-related necessary force floors are far below the failed gates; cutoff graphs are connected with complete two-hop reach, which is descriptive rather than capacity proof.
FINDING-GRADIENT-002: initial scaled energy-gradient dominance and strong final fitting gradient opposition; local evidence only, not Adam trajectory or causal attribution. The final C anchor has positive alignment, preserving mixed evidence.
FINDING-GRADIENT-003: every case/objective passes the predeclared directional criterion, with one failed larger step explicitly retained beside its passing smaller step. Parameter/optimizer/source identities unchanged.
GRADIENT-REPORT-001: [Authoritative report and exact numbers](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/reports/representation-gradient-diagnostics.md); [append-only evidence log](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/reports/PAPER_EVIDENCE_LOG.md).114engineering tests passed; final independent implementation/evidence review pending. No new training/weights/model/labels or full pilot. Next proposed investigation: reviewed loss/optimization control.



## Frozen diagnostic completion — final review and balanced repairs

IMPL-GRADIENT-004/005: independent review completed after a usage-limit interruption. It supported all saved research observations and found two Important failure-handling/reporting bugs. Both were reproduced with failed engineering fixtures and repaired; all 117 tests pass. The reviewer did not re-review repairs; author regressions/full suite verify them. No remaining Critical/Important findings or minors. Raw diagnostic execution/figures remain unchanged, with no research rerun or learning acceptance. Refreshed report hashes and every failed check are in the append-only repository log. Next: a separately planned/twice-reviewed loss/optimization control.
