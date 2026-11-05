# Representation-study closeout implementation plan

26 October 2026. Status: plan and saved results accepted by both independent technical reviews. Implements the coordinator's approved saved-output closeout, alongside the existing MACE workstream. Plan acceptance does not accept the scientific results.

**Goal:** Turn the six completed representation fits into a verified descriptive analysis and research report using their retained outputs.

**Architecture:** One independent saved-data auditor reads authenticated metadata, opaque checkpoint envelopes/control histories and cached validation predictions. A separate plotting script consumes its machine-readable output. Historical fitting/scoring sources are unchanged.

**Stack:** existing Python 3.12 environment; NumPy, msgpack, Matplotlib and standard library. No model framework import or package installation.

**Specification:** [saved-output closeout](../../research/REPRESENTATION_CLOSEOUT_SPEC_2026-10-26.md), with the original [study specification](../../research/REPRESENTATION_COMPARISON_SPEC_2026-10-20.md).

## Fixed constraints and review focus

- Preserve all six candidates and both seeds, SP primary/S64 secondary, and all original recipe/selection settings.
- Audit validation-only saved arrays: 974 records, 21 atoms, five literal groups including distinct `18_2`.
- Use 401 scheduled events and the fixed 20-event window `190500–200000`; BEST uses the first minimum.
- Compare independent endpoint reductions with `atol=1e-14, rtol=1e-12`; no recentering or prediction transformation.
- Read model payloads as opaque bytes only; no model restoration, initialization, inference, fitting, test payload or GPU use.
- Preserve private paths in the local inventory; public results use relative artifact identities.
- Check especially self-consistent metric edits, shuffled/missing IDs, stale or future history, unequal-group pooling, hash failures and incomplete timing. No invalid input can produce an accepted interpretation.

### R01: inventory and authenticate retained inputs

Create a local inventory and explicit path map for the sources named in the specification. Record original/raw/export hash distinctions and source/config/data/seed identity checks. Public input records contain relative paths and hashes, without copying numerical payloads.

- [x] Save rationale and this plan/specification before analysis, obtain both technical reviews and resolve material findings.
- [x] Create the inventory of six exported/raw scores, twelve prediction files, six run config/provenance/summary/history sets, twelve checkpoints and both historical reload proofs per fit, validation manifest/arrays, plus the original reference and launch receipts. Record the old-to-current root mapping explicitly. LAST has the complete history; BEST has the authenticated selected-update prefix.
- [x] Verify raw score/checkpoint/prediction bindings and all six fixed recipes. Retain any failure rather than substituting another run.

### R02: implement the independent saved-data audit

**Create:** `scripts/close_representation_study.py` and `tests/test_representation_closeout.py`.

**Interfaces:** `history_summary(history: list) -> dict`; `reduce_predictions(ids: list, groups: list, energy, forces, labels: dict) -> dict`; `paired_summaries(rows: list) -> dict`; `audit_study(repository: Path, retained_root: Path, inventory: dict) -> dict`; `write_artifacts(result: dict, output: Path) -> None`. CLI accepts explicit `--repository`, `--retained-root`, `--inventory`, `--output`.

- [x] Write analytic tests proving exact-ID reordering, unequal-group equal-weight versus pooled MAE, Cartesian RMSE, finite checks, missing/duplicate IDs and literal connection IDs. Run them and retain the missing-module/function failures.
- [x] Add tests for complete 401-event schedules, first tied minimum, exact late20 bounds, paired sign changes, unknown timing, corrupted bindings and existing-output refusal.
- [x] Implement the smallest pure saved-data auditor. Authenticate msgpack envelopes/control without decoding model payloads; use control history as authority and compare standalone history. Independently reduce endpoint predictions and verify all exported metrics and identities.
- [x] Run the new tests plus existing representation-analysis/protocol tests. Use a separate-process import guard to confirm no JAX/Torch/Flax or model/training imports. Retain failures and fixes. Commit the new analysis source/tests before the saved-data closeout run.

Verification: `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src:scripts python -m pytest -q tests/test_representation_closeout.py tests/test_representation_analysis.py tests/test_representation_protocol.py -p no:cacheprovider`. Expected: all selected tests pass; no real model execution.

### R03: run the fixed audit and create reproducible figures

**Create:** `scripts/plot_representation_closeout.py`; result directory `reports/representation-closeout-2026-10-26/` with JSON/CSV reductions and figures. Plotting interfaces consume only the closeout JSON and CSV, not the original arrays or model files.

- [x] Run the auditor once against the reviewed inventory in a fresh directory. Retain a complete failure record if any input or reduction fails. Do not relax acceptance from observed discrepancies without recorded review.
- [x] Generate family/seed BEST/LAST/late20 tables, paired differences, connection-level endpoint tables, complete validation trajectories and scoped training timing tables.
- [x] Create and visually inspect PNG/PDF figures with both seeds visible, consistent units, unsmoothed data and descriptive captions. Check figure values against the machine-readable reductions.
- [x] Rehash all source inputs and freeze output hashes. Record actual run command, code commit, verification counts, elapsed analysis time and limits.

### R04: write the report and complete independent result reviews

- [x] Write `reports/REPRESENTATION_STUDY_CLOSEOUT_2026-10-26.md` from verified outputs, with the hypothesis, original protocol, all results, cost scope, unsuccessful operational history and bounded interpretation. Preserve E64 recipe-selection and validation-selection limitations.
- [x] Obtain independent scientific and engineering saved-result reviews. Provide exact source/input/output identities, original plan/spec and the report. Reviewers must independently reconstruct arithmetic and claim limits from saved evidence; they must not query models.
- [x] Resolve material findings, re-review material revisions, and save acceptance only when required checks pass. Preserve all original outcomes and failures.
- [ ] Publish the reviewed result after approval of the concrete payload; verify remote identities. Evidence log, stable P31 tracker item and local publication package are prepared; original teammate ownership is preserved.

Completion means reviewed saved-output evidence, not proof of equivariance causality, recovery benefit, cross-molecule superiority or held-out generalization. No promise of a positive result is part of this plan.
