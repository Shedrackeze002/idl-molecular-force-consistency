# Implementation execution and research-gate status

29 September2026. The amended code/audit/diagnostic delivery is verified. The baseline-learning milestone remains incomplete: both learning checks failed; fullpilot pending; paracetamol preparation blocked.

The record below preserves task verification, decisions and trade-offs from the execution ledger. Earlier pending/proposed lines are historical; later approval/completion lines give the final status. It records tests and experimental stop conditions rather than claiming the full study is finished.

# SDD ledger — plan: docs/superpowers/plans/2026-09-29-data-baseline-implementation.md

Execution approved by Shedrack on 29 September 2026 after two independent plan reviews. Native implementation; retain pre-existing proposal deletion and Group9.pdf.

Pre-flight: Tasks 1→2→3 share immutable FrameRecord and inventory; consistent. Tasks 3→4 share alignment, duplicate audit and endpoint statuses; explicit preparation gates. Tasks 4→5→6 share physical units, training-only Normalization and PreparedBundle; scalar residual derivative is consistent. Task 6→7 shares run/checkpoint/validation contracts; consistent.

Ruling: use a dedicated local branch in the approved existing checkout rather than create another checkout — all implementation is additive and user proposal changes are preserved; cost if wrong is checkout isolation during concurrent editing. No shared-branch push or merge is authorized.

Ruling: locate shared data dataclasses in data/types.py and trainer dataclasses in train.py — the plan names interfaces but leaves type ownership open; cost if wrong is a local import refactor.

Tasks 1–7: pending. No baseline tests existed before implementation.
Task 1: complete (commits a832e47..36debad, tests: .venv/bin/python -m pytest tests/test_source_xyz.py -q → 15 passed in 0.06s)
Task 2: complete (commits 36debad..1f7f429, tests: .venv/bin/python -m pytest tests/test_source_xyz.py tests/test_connections_units.py -q → 25 passed in 0.07s)

Task 3: full audit: 91/92 endpoints matched; paracetamol/01_2/plus nearest min1 RMSD 0.06431211546693098 exceeds frozen 0.05. No duplicate pairs <=1e-5 A. Preparation remains gated for paracetamol; do not adjust tolerances.
Ruling proposed for review: gate preparation per selected molecule so certified aspirin can reach its first pilot while paracetamol remains explicitly blocked — consistent with the first milestone's primary molecule and separately trained replication; cost if wrong is scope/manifest confusion, addressed by explicit per-molecule statuses and tests. Two independent reviewers must clear this clarification before Task 4 execution.
Task 3: complete (commits 1f7f429..aaa9aec, tests: .venv/bin/python -m pytest tests/test_source_xyz.py tests/test_connections_units.py tests/test_geometry_matching.py -q → 32 passed in 0.20s)

Ruling approved by both reviewers: preparation gates are per selected molecule; aspirin may proceed, paracetamol remains blocked. Isolation tests required. Cost if wrong: scope/manifest confusion, controlled by explicit molecule identities.
Task 4: complete (commits aaa9aec..4abb572, tests: .venv/bin/python -m pytest tests/test_source_xyz.py tests/test_connections_units.py tests/test_geometry_matching.py tests/test_splits_prepare.py -q → 38 passed in 0.24s)
Task 5: complete (commits 4abb572..c9a4468, tests: .venv/bin/python -m pytest -q → 44 passed in 6.19s)
Task 6: complete (commits c9a4468..3e6a495, tests: .venv/bin/python -m pytest -q → 49 passed in 15.60s)

Task 7: first seed0 learning gate failed at2000 updates; force fraction remaining0.22444164. Preserved ENGINEERING-001. Full pilot not started.
Ruling: add exactly one prospective zero-readout check after both independent reviewers approved the saved learning-gate amendment — official E3x initialization and measured objective gradients motivate it, without proving causality — cost if wrong: one extra engineering attempt and initialization-conditioned acceptance; own-baseline and zero-force reporting prevent stronger claims.
Ruling: do not create a paracetamol split while its gate fails — a frozen assignment would imply readiness prematurely — cost if wrong: later split generation must remain deterministic.

Task 7 amended delivery: second zero-readout check failed at2000 updates, force fractionremaining0.7472377. ENGINEERING-002 preserves results. Approved amendment's stop condition applies: no full pilot or further tuning. Code/audit/diagnostic handoff is implemented; baseline-learning milestone remains incomplete.
Task 7: complete (commits 3e6a495..2a7af87, tests: .venv/bin/python -m pytest -q → 53 passed in 27.29s)

Final review: fresh gpt-6-astra reviewer, range a832e47..2a7af87, independently53/53 tests. Two Important findings, no Critical or deferred Minor findings.
Final: fixed resume selection/history loss — test_resumed_selection_keeps_earlier_best_and_exact_tie RED→GREEN; resumed best state preserved atupdate1 against worse state and later exacttie; suite55/55.
Final: fixed symmetry hiding exact-coordinate label conflicts — test_symmetric_exact_positions_do_not_hide_conflicting_labels RED→GREEN, including row reordering/force conflict; suite55/55.
Final: verification .venv/bin/python -m pytest -q →55passed in38.59s; full8874-record duplicate re-audit unchanged; fresh aspirin preparation canonical identity4b9f220201d6483e2b2a8da74c10330c3762942ff4c6fabe6389ee27249c43aa unchanged.
Final: Ruling: checkpoints without historical selection controls remain readable for inference but cannot silently resume fitting — reject absent controls rather than reset best selection — cost if wrong: legacy runs require original code or explicit migration for resume.
Final: Ruling: future direct/consistency/MACE/test evaluation remains outside this amended delivery — preserve the separate reviewed study milestones — cost if wrong: the broader empirical project is still unfinished.
Final: Ruling: actual GPU and cross-platform reproducibility remain unverified — CPU evidence plus locked GPU launch instructions cannot certify unavailable hardware — cost if wrong: a platform-specific issue may still need resolution.
Final: Ruling: preserve both failed learning gates and leave fullpilot pending under the approved stop condition — failed acceptance is a research outcome, not permission to tune until success — cost if wrong: pilot progress waits for a new reviewed diagnostic plan.
Final branch is retained locally; no push, merge or publication. User proposal changes preserved. Task7 complete means amended implementation/diagnostic delivery, not baseline-learning milestone success.
