# Verification-Only Width/Rate Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for native implementation after concrete human artifact review and two independent scientific/engineering plan reviews. No implementation delegation. Preserve all review workspaces.

**Goal:** Verify one separately identified provenance-normalized copy of the saved final endpoint with zero additional training and preserve the failed original trial.
**Architecture:** frozen stdlib recovery contract → metadata-only staging helper and one bounded subprocess → saved-only recovery report. Existing width/rate/trainer/checkpoint reader and original reports stay unchanged.
**Tech Stack:** locked msgpack, existing CPU/float64 JAX verifier in a fresh child, standard library reporting; no new dependency.
**Spec:** `docs/superpowers/specs/2026-10-01-width-rate-recovery-design.md`.

## Global Constraints

- Config `configs/aspirin-width-rate-recovery.json` canonical identity `55c63996e555fad2c65edb38525dcd27f022188650ee255353b24e20a83c487c`; new 163-file pin manifest SHA `f7fff5b24a9e2eae365a97924d720d996d9c60321c39fc00711d307a03e5a4d4`, plus original unchanged326source checks.
- Protected new root `runs/aspirin-width-rate-recovery-2026-10-01`, child `verification-copy`; fail on existing/symlink/unsafe destination. Preserve original failed outputs and proposal.pdf/Group9.pdf user changes.
- Original6000 endpoint, feature16/LR.001,128fit/16fullyfittedanchor, original normalization/state/input/reference identities; atol1e-10/rtol0, CPU/float64. Only envelope provenance text and its copied receipt/contract checkpoint SHA may change.
- Exactly one numerical subprocess, total180s boundary cap; no training, gradient/Adam proposals, warmup or blocked-cell execution. No retry/fallback. A passed recovery is not original matrix validity, contrasts or pilot clearance.
- Before action and after outcomes, append rationale/settings/status/hashes/limits to evidence log; preserve every failure and unknown-work receipt. Full suite prior to one production verifier; no duplicate suite without a relevant change/failure.
- Existing source/trainer/reader/width-rate reporter/CLI unchanged; create new isolated modules only. Record new code hashes/revision. Native method and retained feature checkout follow established context; no push/merge/cleanup.

## Review Focus

1. Semantically equal provenance with reordered integer/string keys: exact allowed metadata delta, payload/control bytes unchanged; test Task1.
2. Metadata semantic substitution, corrupt envelope/reference/history, unsafe symlink/path or coherently rehashed local copies: pinned source rejection before staging/numerics; test Task1.
3. Interruptions between copy/checkpoint/history/receipt publication: partial preparation retained, no complete receipt or numerical launch; test Task1.
4. Child failure/timeout/source change after a valid receipt: one attempt, honest known versus unknown work, no zero-time implication or missing reference count; test Task2.
5. Coherently edited report claiming valid original matrix, old timing backfill or invented child work: saved-only strict binding, independent statuses; test Task3.

### Task 1: Frozen recovery contract and metadata-only transformation

**Files:** Create `src/force_consistency/width_rate_recovery_contract.py`, `src/force_consistency/width_rate_recovery.py`; tests `tests/test_width_rate_recovery_contract.py`, `tests/test_width_rate_recovery.py`.
**Interfaces:** `verify_recovery_sources(config: dict) -> dict` validates exact reviewed config/pin identities and original contract/sources, returns source receipt/provenance/envelope metadata and file digests. `stage_recovery_copy(config: dict, root: Path, check_budget: Callable[[], None]) -> dict` consumes those guarded source bytes, returns transformation receipt. Neither imports numerics, loads arrays nor restores state. Task2 consumes its staged child/receipt; Task3 rechecks it against originals.

- [ ] Write meaningful failing tests: `test_recovery_contract_rejects_rebased_controls_before_io` mutates source path, cap, count, tolerance, state/hash or extra key; sentinel staging/child not reached. `test_pin_and_destination_guard` rejects substituted/missing source bytes, replacement pin map, existing roots and symlink escapes.
- [ ] Write `test_only_provenance_and_checkpoint_binding_change`: tiny serialized checkpoint with keys200/1000 fails the unmodified strict reader before transform; after stage, original source SHA unchanged, copied payload/control bytes exactly equal, copied provenance JSON equals original, original/copy checkpoint SHA differ, only allowed receipt/contract fields differ. A real tiny engineering template can restore the copy; no production state is touched. Add semantic-difference, corrupt-payload/control/reference/history and unsafe-receipt regressions.
- [ ] Run `.venv/bin/python -m pytest -q tests/test_width_rate_recovery_contract.py tests/test_width_rate_recovery.py`; expect the specifically missing implementation/guard failures, retain logs. Implement smallest frozen guard and ten-file transformation from spec, using msgpack only for envelope bytes. Preserve source and staged immutable versions; commit receipt last. Add failure injections before/after checkpoint/history/receipt writes; require preparation incomplete and no child launch.
- [ ] Re-run focused tests, expect all pass, save exact logs/counts. Verify importing the pure contract and transformation in a numerical-import-denying subprocess and denying np.load does not fail. Commit new helpers/tests only, preserve user changes.

### Task 2: One verifier with honest deadline and work accounting

**Files:** Extend new `src/force_consistency/width_rate_recovery.py`; extend `tests/test_width_rate_recovery.py`. No existing module changes.
**Interfaces:** `recover_width_rate(config: dict) -> dict`, module entry point `python -m force_consistency.width_rate_recovery --config PATH`. Runtime-only clock/deadline and injected engineering probe runner belong to private helpers, never mutate frozen config. Consumes Task1 receipt; returns separately labeled summary with source failed status, source/copy/code/config/pin identities, attempted/completed work, child result, elapsed and validity.

- [ ] Write failing tests `test_recovery_one_probe_never_trains` (training/weighted-step/warmup/preflight symbols replaced with fail sentinels), `test_tiny_fresh_process_verifies_copied_state` (real tiny engineering state and subprocess using existing private saved verifier), `test_recovery_failed_probe_retains_unknown_work`, `test_no_launch_after_cap`, `test_source_change_after_pass_preserves_known_child_work`, `test_final_persistence_failure_retained` and `test_existing_root_prevents_second_attempt`.
- [ ] Run focused tests and retain RED. Implement root-protected controller/prospective receipt, exact single child module `--probe verification-copy`, inherited executable and CPU env, remaining-positive timeout and boundary checks. Record subprocess-started before launch; failed unreceipted work unknown, attempted duration measured. Capture code/dependency versions via metadata without importing numerics in parent. Emit actual statuses and terminal failure receipt even after budget/output failure; no re-probe.
- [ ] Require bound successful child receipt: count/update6000, exact restored state hashes, normalization/case/config/checkpoint/history identities, float64, maximum prediction error≤1e-10, initializer1/16frames, prediction144frames. Only then record completed child counts; retain known counts if later guard fails. New updates/optimizer/warmup0. Histories/gates old, timing old and original unknown work not backfilled.
- [ ] Re-run focused tests, expect pass; preserve all logs and commit isolated controller/tests. No production verifier yet.

### Task 3: Saved-only report, once-only verification and durable records

**Files:** Create `scripts/report_width_rate_recovery.py`, `tests/test_width_rate_recovery_report.py`; later generate `reports/width-rate-recovery-verification.json/md`, `reports/width-rate-recovery-execution-record.md`, `docs/reviews/2026-10-01-width-rate-recovery-final-review.md`, durable `reports/execution/width-rate-recovery-2026-10-01`; append evidence log/update README/navigation only after actual outcomes.
**Interfaces:** `generate(root: Path, output: Path) -> dict` consumes only guarded saved JSON/bytes plus pure Task1 contract, validates deltas/receipt/code/time/work, and writes a distinct recovery report without any numerical imports/array loading. Failed/partial statuses remain reportable; exit0 only for valid copied verification, exit1 for failed/invalid input, never implies matrix or pilot success.

- [ ] Write failing saved-record tests for exact valid copy/receipt, missing/corrupt inputs or changed payload/control/provenance semantics, changed published receipt/child identities, impossible duration/counts, fake original-matrix success, hybrid blocked-cell claims and old-probe timing backfill. Add interrupted-before/after-child cases and numerical-import/array-load-denying report subprocess.
- [ ] Run RED, implement the separate reporter with immutable source/delta binding and source failed/blocked/gate fields. Retain engineering fixtures as synthetic. Run focused GREEN then `.venv/bin/python -m pytest -q` with writable plotting caches if needed; read full output, capture count/time/hash. Full prior baseline327is contextual; actual final count must come from this run.
- [ ] Complete prospective evidence with concrete accepted config/pins/code revision and source/copy hashes. On approval, launch exactly once: `JAX_PLATFORMS=cpu .venv/bin/python -m force_consistency.width_rate_recovery --config configs/aspirin-width-rate-recovery.json`. Expect either bounded passed endpoint verification or a truthful failed/incomplete terminal record; neither clears learning gates or original matrix. Capture exit/stdout/stderr and all work receipts; do not retry any failed numerical action.
- [ ] Generate saved-only report with `--root runs/aspirin-width-rate-recovery-2026-10-01 --output reports`; verify expected exit/status and independent original/recovery identities. Do not replace old width-rate reports or populate their unavailable contrasts/terminal summaries.
- [ ] One fresh independent final reviewer reconciles code/spec/saved results; author verifies material fixes with RED→GREEN/full suite, no production rerun. Save every finding/author ruling/declined scope. Archive raw recovery files, all workspace/review/failure logs and original-source hash map; verify CRC/entry sets/SHA bytes. Preserve originals.
- [ ] Append actual outcomes/limits/next check, update Obsidian via CLI backups/readback (record unavailability and pending navigation if needed), scoped local commit. Full pilot remains pending. Next numerical matrix completion/replacement is a separate reviewed prospective milestone.

## Draft self-review and authorization boundary

Each specification section maps to Tasks1–3; exact config/source identities, ten-file allowed transformation, strict existing child verifier, actual/unknown counts, global cap and preserved original status are covered. No placeholders or new scientific settings. This is a planning artifact, not executable clearance. Obtain two independent scientific/engineering reviews and re-review material revisions; then present the concrete plan to the user before native implementation or the newly amended production verifier.
