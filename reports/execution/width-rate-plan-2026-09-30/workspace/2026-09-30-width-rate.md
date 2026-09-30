# Crossed Width and Fixed Rate Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans for native implementation after the human plan review and two independent scientific/engineering reviews. No implementation delegation. Preserve all review workspaces.

**Goal:** Build and execute one frozen fresh4cell width/rate diagnostic with full historical replay and balanced outcome/provenance reporting.
**Architecture:** pure frozen contract and hash/control preflight → isolated fresh runner/controller using existing weighted-gradient/Adam math → saved-JSON-only report. Shared trainer/model/loss infrastructure stays unchanged.
**Tech Stack:** locked JAX/Flax/Optax/E3x float64CPU, standard JSON/checkpoints, existing Matplotlib; no new dependency.
**Spec:** `docs/superpowers/specs/2026-09-30-width-rate-design.md`.

## Global Constraints

- Config `configs/aspirin-width-rate-diagnostic.json`; reviewed canonical identity113d98ebdc068cc5fadd24bacab53c73f73a65f804ef4b5f9cfddc1358be816a and source-pin manifest identity bound in pure contract. All301source files verified before arrays and after cells/end; original baselines bound to historical records.
- Exactly width16/.001, width16/.0003, width32/.001, width32/.0003 sequential; fresh seed0/count0/zero moments, one initialization per width cloned to rates. Same128/16fully-fittedanchor, normalizations, balanced E.014584365670390719/F1.9854156343296092 and other model knobs/Adam settings from spec.
- 6000updates/768000committed exposures per cell;24000/3072000new total. One discarded128-exposure warmup per cell separately counted; independent first-step reference also discarded, no source mutation.
- Exactly64scores:0,1,250,2001,100…6000. Replay cell matches63savedhistoricalpoints (no invented source1), strict1e-10/1e-8; duplicate2000mustmatch before merge.
- Original joint<.1E/FrawMSE gates both roles; no baseline reset/recenter/held-out input/schedule/best checkpoint/learningearlystop or automatic pilot. Terminal5500→6000 abschange<.01 descriptive only.
- Arm caps1800sfor16/3600sfor32; global11400s includes preflight/output. Unknown pending versus synchronized/persisted work separate, failed verification blocks later cells; failed learning alone does not. No numerical retry/backend change.

## Review Focus

1. Coherent edits to copied controls/pin maps/config must fail before the train-array loader: task1 pins canonical identity and direct raw ancestry.
2. Equal seed is not equal cross-width parameters; task1 tests within-width hashes, cross-width zero-head predictions/raw baselines, and count0/zero moments.
3. Historical schedule contains250, not1; task1 tests63point join and exact2000duplicate; task2 adds first-step1 for64newpoints without inventing old evidence.
4. Persistence/interrupt/timeouts can leave attempted work unconfirmed; task2 tests checkpoint/history write failure, KeyboardInterrupt, caps/fresh subprocess and later-arm blocking.
5. Copied report receipts/config can claim complete counts or cleared pilots; task3 tests strict source/config/history/state/receipt/counter/budget guards and four-cell-only contrasts.

### Task 1: Frozen contract and genuine fresh initialization

**Files:** Create `src/force_consistency/width_rate_contract.py`, `src/force_consistency/width_rate_runs.py`, `tests/test_width_rate_preflight.py`. Existing configs/spec accompany these files; no shared numerical edit.
**Interfaces:**
- `validate_contract(config: dict) -> None` and `expected_updates() -> list[int]` in the pure contract.
- `verify_width_rate_sources(config: dict) -> dict` returns pinned historical controls and63point replay record, using only metadata/bytes.
- `width_rate_preflight(config: dict, check_budget: callable, save_partial: callable) -> dict` returns fitting, anchor, normalization, per-width model/initial state, immutable original baselines, replay record and provenance. Models/states are never embedded in saved JSON.

- [ ] Write RED `test_contract_coherent_rebase_before_arrays`, `test_pin_replacement_and_missing_file_before_arrays`, `test_historical_duplicate_2000_must_match`, `test_historical_schedule_contains_250_not_1`: loader sentinel must remain uncalled on corrupted config/controls/pins; source join has exactly63points; altered duplicate raw metrics and any omission fail.
- [ ] Run `.venv/bin/pytest -q tests/test_width_rate_preflight.py`; expected missing-feature RED. Save exact result and evidence, distinguishing absent-feature collection failure from behavior failure.
- [ ] Implement pure exact canonical config/pin identities and safe source path/hash allowlist; verify existing convergence source guards with unmodified nested historical config and the pinned compact/raw control copies. Extract balanced histories22+42, require exactduplicate2000, join63points. Only then invoke training-only `balance_preflight` and frozen subset selection.
- [ ] Write RED `test_pair_initial_hashes_and_zero_moments`, `test_width_zero_head_baselines_match_original`, `test_wrong_normalization_or_count_blocks`, `test_only_features_change`: actual16/32model initialization with historical anchor batch; one seed0state per width; cloned same-width states equal, count/update0 and all mu/nu0; cross-width predictions/raw original baselines match tolerance; other config knobs unchanged. Record parameter counts, not theoretical capacity.
- [ ] Implement fresh initializer/templates and explicit initial-state checkpoint/metadata, raw baseline comparison to pinned original record, within-width state hashes, train-only/no-held-out guards. Numerical initialization/prediction work has synchronized counts and budget checks.
- [ ] Run focused preflight checks GREEN; append every failed trial/correction and commit only scoped artifacts. Preserve user `proposal.pdf` deletion and untracked `Group9.pdf`.

### Task 2: Verified fresh4cell runner and failure-retaining controller

**Files:** Extend `src/force_consistency/width_rate_runs.py`; modify `src/force_consistency/cli.py`; create `tests/test_width_rate_runs.py`.
**Interfaces:**
- `run_width_rate_arm(config: dict, case: dict, initial: dict, fitting, anchor, model, normalization, original_baselines: dict, historical: dict, root: Path, provenance: dict) -> dict` produces actual64row history, metrics/gates/counts/time/state/oracle/receipt/failure data. The production wrapper validates the frozen cell; tiny test helpers do not expose a CLI/config bypass.
- `execute_width_rate_matrix(root: Path, config: dict, runner: callable, verify_sources: callable) -> dict` orders the four cases, checks sources and records every executed/partial/blocked outcome.
- `width_rate_probe(root: Path) -> dict` restores/validates exact fresh checkpoint from a feature-specific template in a subprocess.
- `diagnose_width_rate(config: dict) -> dict` creates protected root, runs preflight/ordered controller and appends evidence. CLI subcommand `diagnose-width-rate --config …`.

- [ ] Write RED independent tiny mixed-energy/force multi-step fullbatch reference versus accumulated update, with16-frame splitting replaced by tiny uneven fixture batches and nonzero force derivatives. Verify params/moments/count and exact exposures; a reset or warmup advancement must fail. Test fresh firststep from both approved widths/rates against fullbatch gradient plus separate NumPy Adam formula, not the production gradient builder.
- [ ] Write RED ordered/controller tests: replay mismatch at a common point blocks later cells; joint learning failure still runs them; source mutation blocks; existing nonempty root refused; scheduled64history includes1/250/2001 and bothroles gates; attempted/pending versus actual/saved counts; no fictitious6000 after interruption.
- [ ] Implement existing `build_weighted_step` at each frozen fixed LR, discarded warmup and independent first-step reference. Full128reference gradient uses direct mixed derivative and separate NumPy bias correction; retain discrepancy/reference identities and actual work counts. Compare replay as each of63savedcommonpoints arrives. No weight recalibration, trainer-resume change or historical controller execution.
- [ ] Write RED checkpoint corruption with predictions unchanged but wrongcount/norm/feature template; exactcheckpointSHA mismatch, badfloat64, subprocess timeout; KeyboardInterrupt in unsynchronized step, atomicJSON/checkpoint-save/controller-summary failure, wall-before/after scoring/probe. Last finite versioned checkpoint/history remains readable, attempted unknown flagged, remaining arms blocked and independent failure receipt retained if summary persistence fails.
- [ ] Implement atomic JSON/versioned checkpoints, explicit boundary caps, synchronized finite state/counts, fresh subprocess exhaustive predictions/count/update/norm/canonical-state/exactcheckpointSHA verification with remaining-time timeout. Shared per-width initialization is measured once in preflight. Charged arm elapsed = that width initialization seconds + owncell elapsed (starting before state copy), strictly below1800/3600; subtract initialization charge from remaining arm deadline. The same shared duration is a conservative charge in both paired caps, while global actual elapsed/work counts include it only once. Save shared initialization, owncell elapsed and charged elapsed separately; report does not double-count real compute.
- [ ] Add CLI routing/config-safe JSON outcome; `.venv/bin/pytest -q tests/test_width_rate_preflight.py tests/test_width_rate_runs.py` GREEN, save RED/GREEN/failed attempts and scoped commit.

### Task 3: Saved-only report, one fixed matrix and final research records

**Files:** Create `scripts/report_width_rate.py`, `tests/test_width_rate_report.py`, `reports/width-rate-diagnostics.md/json`, four-cell curves in `reports/figures/`, execution record and final review; append evidence and update current navigation.
**Interfaces:**
- `verified_arm(arm: dict, config: dict, sources: dict) -> bool` verifies saved binding provenance/control/history/gate/count/state/receipt/budget.
- `factorial_contrasts(arms: dict) -> dict` computes for eachrole/objective four endpoint ratios and ratio-of-rate-ratios interaction only on complete valid positive-denominator cells.
- `generate(root: Path, output: Path) -> dict` uses saved JSON/bytes only, sharing pure contract and binding report to exact raw artifacts/history/config/source-pin identities.

- [ ] Write RED missing/duplicate/off-schedule64history, fabricated source1, localbaseline×1000coherentgate rebase, omittedsourcepin/receipt, wrongfeature/rate/support or6010updates, swappedcellinitial hash, incompletecaps, nonfinite/zero denominator, partial/blockedcontrasts and learning-failure-versus-execution-valid cases. Report cannot infer a cleared pilot or a capacity/convergence certificate.
- [ ] Implement all raw/pergroup/equalgroup/overlap histories, originalfractions/weightedloss, actual24000updates/3072000exposures with partial truth, measuredparams/compile/steady/eval/probe/total timing, discarded reference/warmup counts and strictfrozen/source/receipt/ancestry guards. All points/thresholds plotted; label one-seed descriptive ratios and actual widths, not replicated PaiNN. Terminal label remains descriptive; no selected checkpoint/candidate.
- [ ] Run focusedreport GREEN and `.venv/bin/pytest -q` fullsuite GREEN before production. Save code/lock/config/plan/review hashes; commit implementation. A proposed matrix is not marked executed until its CLI starts.
- [ ] After human plan review/authorization and both independent reviewers clear material amendments, invoke exact reviewed configuration **once**: `.venv/bin/python -m force_consistency.cli diagnose-width-rate --config configs/aspirin-width-rate-diagnostic.json`. Log prospective rationale/controls/gates immediately before; append each partial/failure/blocked/completed outcome as it occurs. No retry or adaptive intervention.
- [ ] Generate report from saved records and visually inspect every panel/threshold/prescribed fluctuation; reconcile fresh16/.001 all63historicalcommonmeasurements and full matrix receipts/source hashes. Keep all scientific failures and limitations; fullpilotpending even if a cell passes.
- [ ] One fresh whole-branch final review plus saved-result reconciliation; reproduce/repair Critical/Important findings with meaningful regressions and fullsuite, no research rerun unless separately authorized. Record all rulings and limitations. Refresh Obsidian through CLI with livebackup/readback; verified durablearchive supplements preserved workspace/rawruns. Scoped local commit; no push/merge/cleanup of review records.

## Self-review and status

Draft self-check: each spec control maps to tasks1–3; interfaces agree; five listed failure classes have explicit tests; pure contract/report contains no JAX numerical work. Shared init-cap charges versus actual once-only work are explicit and require reviewer scrutiny. No code or research executed for this plan. Two independent plan reviews and the human's concrete artifact review remain pending.
