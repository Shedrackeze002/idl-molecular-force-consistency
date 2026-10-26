# MACE loader and validation parity implementation plan

26 October 2026. Status: accepted by independent scientific and engineering plan reviews. Implements master tasks P06–P09. Implementation remains Idris's workstream; the coordinator prepares the plan and review support. Execute the checkboxes in order and retain evidence beside each completed step. Plan acceptance does not establish implementation or numerical launch readiness.

**Goal:** Admit the verified MACE final model and test the existing evaluator bridge against all 974 saved validation predictions.

**Architecture:** A loader authenticates the packet, environment and model snapshot before constructing the calculator. A worker validates and joins records, queries the shared bridge and saves parity evidence. A small supervisor bounds the attempt; a saved-output audit never loads the model.

**Tech stack:** Python 3.12.13, the retained MACE/Torch environment, NumPy, ASE, pytest and standard-library subprocess/hash/JSON utilities. No package installation or frozen-source changes.

**Spec:** [MACE loader/parity specification](../../research/MACE_LOADER_PARITY_SPEC_2026-10-26.md). Read it together with [the existing contract](../../mace-evaluator-contract.md).

## Global constraints

- Preserve the primary final EMA model and all pinned identities in the specification; one calculator, CPU/one Torch thread/float64, no acceleration or compilation.
- Validate exactly 974 records and use absolute parity tolerances `1e-10 eV` and `1e-10 eV/Å`, with zero relative tolerance.
- Use at most 974 force and 974 energy accessor requests; no relaxation, training, alternative model or held-out test access.
- Use one fresh private attempt; shared 1,800-second cap, 256 MiB output cap and 16 MiB per log; failure cannot become a pass.
- Keep implementation outside frozen `src/`, `evaluation/` and the existing bridge. Preserve author ownership and user changes.
- Each task starts with a failing behavior test, then the smallest implementation, passing checks and a bounded authored commit. Do not test by loading the real model before the launch gate.

## Review focus

1. Self-consistent tampering with a packet/manifest must still fail the independently pinned manifest/model identities before construction.
2. Valid rows in different storage orders must join by exact IDs; duplicates, missing IDs, species changes and malformed arrays must fail before model construction.
3. A correct model tag with the wrong constructor path must fail; the verified private snapshot must be the actual calculator input.
4. Timeout, subprocess failure, output overflow and partial predictions must survive in evidence and never produce `PARITY_PASS`.
5. Near-zero energies/forces must use absolute-only comparisons; default relative tolerance, post-hoc re-centering and omitted rows must not pass.

## Files and interfaces

Create only these product files, plus small result summaries after execution:

- `scripts/mace_loader.py`: packet/environment/model-snapshot admission and bridge construction.
- `scripts/check_mace_parity.py`: fixed input validation, joined prediction worker, comparison and CLI.
- `scripts/supervise_mace_parity.py`: fresh-directory allocation and bounded process-group execution.
- `scripts/audit_mace_parity.py`: independent saved-output verification without MACE/Torch model import.
- `tests/test_mace_loader.py`, `tests/test_mace_parity.py`, `tests/test_mace_parity_supervisor.py`, `tests/test_mace_parity_audit.py`: synthetic refusal and acceptance controls.
- `docs/run-mace-parity.md`: commands, artifact access, identities, budgets and limits.

Use the existing `scripts/mace_reference_adapter.py:calculator_arm` unchanged. Do not import the old scorer CLI to inherit its obsolete absolute paths; reuse its exact documented calculator recipe and independently verified source identities.

### Task 1: authentication and admitted calculator construction

**Consumes:** private `packet_root: Path`, fresh `attempt_root: Path`, authenticated specification constants. **Produces:** `verify_packet(packet_root: Path) -> dict`; `load_admitted_arm(packet_root: Path, attempt_root: Path) -> tuple[dict, dict]`, returning the shared arm and separate admission receipt.

- [ ] Write `test_rejects_tampered_manifest_before_constructor`, `test_rejects_missing_symlink_or_wrong_model_before_constructor` and `test_rejects_environment_source_mismatch_before_constructor`. Assert zero constructor calls and preserved refusal reason.
- [ ] Write `test_constructs_from_verified_snapshot_path` with a fake pinned-recipe constructor and analytic ASE calculator. Assert copied bytes, pre/post hashes, constructor CPU/thread/dtype flags and unchanged bridge verification tags. Keep factory injection confined to private unit-test seams.
- [ ] Run the loader tests and retain the expected missing-module or missing-function failures.
- [ ] Implement packet checks, metadata verification, one snapshot copy and the pinned production constructor. Save admission separately; no original file writes.
- [ ] Run loader tests and existing bridge tests; retain output. Commit only the new loader/tests and relevant documentation.

Verification from the isolated source root, using the retained synthetic-test environment and `PYTHONPATH=.:src:scripts`: `python -m pytest -q tests/test_mace_loader.py tests/test_mace_reference_adapter.py -p no:cacheprovider`. Expected: all selected synthetic tests pass; no checkpoint is loaded.

### Task 2: input validation and deterministic parity worker

**Consumes:** the authenticated packet and, after full input validation, the arm/receipt from Task 1. **Produces:** `validate_inputs(packet_root: Path) -> dict`; `compare_predictions(actual: dict, expected: dict) -> dict`; `run_parity(packet_root: Path, attempt_root: Path) -> dict`. Production order is `verify_packet → validate_inputs → load_admitted_arm → predict`; the saved-output audit follows execution.

- [ ] Write `test_permuted_storage_order_joins_exact_ids`, asserting the same per-record comparisons after row-order changes. Write duplicate/missing-ID, membership, species/order, dtype/shape, nonfinite and pickle-refusal cases; each must cause zero model constructions.
- [ ] Write absolute-bound tests just below, at and above `1e-10`, including near-zero values and a case that would pass a nonzero relative tolerance. Assert mismatch remains `PARITY_FAIL`; no re-centering occurs.
- [ ] Write `test_analytic_worker_records_exact_accessor_counts` and `test_worker_failure_retains_completed_rows`. Verify exactly one fake construction, one force and one energy request per fixture record, and an incomplete result after an injected failure. Production counts stay fixed at 974/974.
- [ ] Run these tests to confirm missing behavior fails, then implement the ID join, complete prevalidation, shared-bridge calls and atomic prediction/status persistence. Keep input validation before Task 1 model construction in the production control flow.
- [ ] Run Tasks 1–2 tests and existing bridge regressions. Commit the new worker/tests and document exact fields/statuses.

Verification: `python -m pytest -q tests/test_mace_loader.py tests/test_mace_parity.py tests/test_mace_reference_adapter.py -p no:cacheprovider`. Expected: all selected synthetic cases pass without real model construction.

### Task 3: bounded execution and independent saved audit

**Consumes:** worker CLI `python scripts/check_mace_parity.py --packet-root PATH --attempt-root PATH`. **Produces:** supervisor CLI `python scripts/supervise_mace_parity.py --packet-root PATH --output PATH`; `audit_saved_parity(packet_root: Path, attempt_root: Path) -> dict`; audit CLI `python scripts/audit_mace_parity.py --packet-root PATH --attempt-root PATH`.

- [ ] Write supervisor tests for existing-directory refusal, child nonzero exit, timeout/group cleanup, log/total-output overflow and persistence crossing the shared cap. Use controlled tiny test workers and a clock seam; no long sleeps or learned calculations.
- [ ] Write saved-audit tests that independently join IDs and recompute maxima/counts. Reject altered predictions/status, partial rows, input/snapshot/source hash mismatch and invented success flags. Assert no MACE/Torch model import.
- [ ] Run these tests and retain the expected failures. Implement the supervisor and an independent NumPy/standard-library audit; do not make the audit trust worker comparison fields.
- [ ] Run all four new test files plus bridge, runner, outcome and reporter regressions. Expected: zero failures; retain expected archive skips and warning scope. Commit only reviewed source/tests/docs; preserve source identity through execution.

Verification: `python -m pytest -q tests/test_mace_loader.py tests/test_mace_parity.py tests/test_mace_parity_supervisor.py tests/test_mace_parity_audit.py tests/test_mace_reference_adapter.py tests/test_shape_recovery_runner.py tests/test_shape_recovery_outcomes.py tests/test_shape_recovery_report.py -p no:cacheprovider`.

### Task 4: implementation review, bounded launch and saved-result closeout

- [ ] Obtain two independent implementation reviews of the exact committed loader, worker, supervisor, auditor, tests and unchanged bridge. Resolve material findings and rerun affected checks.
- [ ] Save the launch receipt with final code commit/hashes, packet and environment checks, exact existing interpreter, approved plan/reviews, output path and fixed budgets. Confirm private packet access and the coordinator's bounded execution authorization. A metadata-only PASS is not admission/parity.
- [ ] Run the supervisor once in the retained MACE environment with the fixed packet and a new output directory. Retain every outcome and partial file; no automatic retry. The exact command is filled into the reviewed launch receipt rather than guessed from historical absolute paths.
- [ ] Run the independent saved audit in the same source identity without loading the model. Obtain two saved-result reviews; keep execution validity separate from numerical agreement.
- [ ] Append the evidence log and update master P07–P09 according to actual completion. Publish the bounded code and reviewed outcome, including failure/incomplete limits; verify remote file identities. Exclude model, raw packet, private snapshot and operating records.

Acceptance: `PARITY_PASS` requires all fixed inputs/counts, both absolute maxima within bounds, valid admission and intact final evidence within the shared cap. A valid negative result is recorded rather than retried. No recovery or held-out test authorization follows automatically.

## Current readiness and ownership

P04 source readiness and synthetic baseline are established; P05 remote handover presence is verified under the coordinator's receipt convention. The public handover and contract match retained bytes. A public task summary uses different prose from the local historical version while preserving ownership.

No published Idris loader contribution was visible in the inspected branches and PRs on 26 October. This does not establish that he has done no private work. Agree the implementation handoff before starting P07; the coordination plan must not become duplicate code or misattributed contribution.
