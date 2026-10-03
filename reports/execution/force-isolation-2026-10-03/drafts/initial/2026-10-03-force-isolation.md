# Force-gradient isolation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Primary implements inline; subagents review only. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and verify one controlled energy-gradient-removal ablation, retaining the completed offset-profiled joint run as a fixed comparison.

**Architecture:** New pure contract, worker/child, supervisor and saved-only reporter reuse unchanged numerical/ownership/audit helpers. The worker owns orchestration rather than bypassing the old canonical guard. All old sources and user artifacts stay pinned.

**Tech Stack:** Existing Python/JAX/Flax/Optax/E3x/NumPy environment, CPU,float64. No installation or runtime change.

**Spec:** `docs/superpowers/specs/2026-10-03-force-isolation-design.md`.

## Global Constraints

- Scope is the exact spec and proposed-settings.json. Model32/13569params/21atoms, fitting128/anchors16(group01), seed0,CPU,float64,micro16,Adam0.0003; energy0.0/force1.9854156343296092 without renormalization.
- Restore original capacity width32 update0, never an endpoint or random initializer; global energy-gauge projection retained. Initial full projected state must EXACTLY match control `b22c5108b3e08c088f20f2bc862e01ddb6df9fd3961ce949f951e41f1512e4bc` before warmup.
- One invocation/one child/6000updates/6001accepted calibrations;63scores/62postzero INCLUDING3001. Oracles1/3001 with combined1e-10/1e-8; child144role entries absolute1e-10.
-3600s charged wall/180s reserve/3660s supervisor/TERM10s thenKILL; no early stopping/retry/resume/search/adaptation or new data access.
- Original raw MSE strict0.1 force-only BOTHroles and separate unchanged joint BOTHmetrics/BOTHroles. Gauge/energy-free-supervision/held-out/convergence/mechanism claims prohibited.
-18,963 exact source pins with control numeric manifest; old artifacts immutable; incidentalDS_Store excluded explicitly. Entire prior evidence prefix preserved. UserPDFs/review folders untouched.
- Execution is not authorized by a plan PASS alone: independent implementation followup and final frozen-suite/source/runtime gate also precede production. Current milestone is preparation and dual plan review; proposed experiment remains UNEXECUTED in planning records.

## Review Focus

1. Drifted energy coefficient/force renormalization/normalization/IDs — reject publicly before numerical imports; Task1 guards/tests.
2. Source/control identity mismatch or hash-complete but member-incomplete map — reject missing/extra/symlink and full-state mismatch, retain incidentals; Task1 membership and Task2 setup tests.
3. Hidden use of energy labels in gradient or masking shared parameters — prove pure-force gradient independence, label influence confined to bias gauge, and trainable kernel/body; Task2 independent numeric tests.
4. Child uses old worker/old guard, repeated entry, or late final write accepted — own new routing/guards, exclusive attempt and invalidation through last durable write; Task2/3 actual control-boundary regressions.
5. Reporter confuses force-only with joint success, projected baseline,3001 grid,invalid completion or adverse group energy — raw recomputation/failclosed/saved-source comparison and negative fixtures; Task3 tests.

---

### Task1: Pure prospective source and review contract

**Files:** Create `scripts/force_isolation_contract.py`, `tests/test_force_isolation_contract.py`, `configs/aspirin-force-isolation.json`; planning records in `reports/execution/force-isolation-2026-10-03/`. Do not modify old source pins/configs/tests/scripts.

**Interfaces:** `verify_inputs(config:dict)->dict` returns `root`,original `provenance`,`zero`,`zero_receipt`,`pins`,`control_root`,`control_provenance`,`control_audit`,`control_initial_full_hash`; `verify_gate(config:dict,expected:dict|None=None)->dict`; `verify_invocation(config:dict,gate:dict)->dict`; `CONFIG_ID:str`, `CONTEXT_PATHS:tuple[str,...]`. Returned old-source fields remain compatible with unchanged saved-array audit/helpers.

- [ ] Write failing tests: `test_force_coefficient_is_not_renormalized` asserts energy0 and force1.9854156343296092 accepted while force2 rejected; `test_public_guard_precedes_numerical_import` rejects canonical drift without JAX/model import; `test_source_membership_exact` rejects missing/extra/source-map disagreement; `test_sources_control_roles_and_projected_zero_bound` rejects changed ordered IDs/model/norm/source0/controlinitial0; `test_incidental_control_metadata_is_explicit` accepts only retained declaredDS_Store and rejects extra numeric file; `test_review_and_invocation_bindings` rejects wrong ordinal/config/root/runtime/revision/source/hash and absent two-final-PASS plan/implementation gates.
- [ ] Run `.venv/bin/pytest -q tests/test_force_isolation_contract.py`; preserve actual RED failure and settings/log paths.
- [ ] Implement signatures above with pure existingcc/msgpack/NumPy audits; reconstruct exact18,963-key union from recorded old map/control inventory/old gate and context15path tuple. Full saved-only old control audit must pass, irrespective of its failed learning gates. Validate source/control source-zero identities, labels/norm/model/weights/grids/child and original raw baselines. Freeze canonical generated config from proposed settings + derived plan/source hash fields, no numerical option overrides or test-mode public flag.
- [ ] Validate all tests GREEN and source-only preservation/prefix; record config canonical/file and plan/source identities; save prospective work/budget counts. Commit this task's config/contract/tests/evidence explicitly, excluding user PDFs.

### Task2: New guarded worker and own once-only child

**Files:** Create `scripts/train_force_isolation.py`, `tests/test_force_isolation_runs.py`; import unchanged `scripts/offset_profile_core.py` and pure/helper definitions from old worker. Reuse source environment. No numerical-core/source modification.

**Interfaces:** Consumes Task1 contract. Produces `execute(config:dict)->dict`, `run_owned(config:dict,source:dict,gate:dict,code:dict,owner,start:float,*,probe_runner)->dict` private orchestration; `probe(root:Path,token:str)->dict`, `launch_probe(root:Path,timeout:float,token:str)->dict`, `probe_owned(root:Path,source:dict,gate:dict,attempt:dict,entry:dict)->dict`; CLI `--config` defaults to new fixed config and `--probe --token` goes through new pure guards. Reporter imports only pure dispatch/helper paths until saved-state decoding.

- [ ] Write failing tests: `test_force_gradient_independent_of_energy_labels` compares every leaf against independently differentiated pure-force loss and finite altered energy labels; `test_gauge_projection_leaves_force_and_body_gradient_unchanged` compares force arrays,body/readout-kernel gradients and moments before/after scalar bias calibration; `test_private_joint_engine_matches_frozen_control` uses tiny unequal-chunk real-model fixture and joint coefficient pair to match old engine full-state/history/projection/NumPy oracle outcomes; `test_initial_full_state_exact_before_any_updates` rejects control0 mismatch with zero accepted updates; `test_nonzero_body_moment_oracle` uses actual nonzero incoming Adam state and checks combined tolerance; `test_real_engine_late_durable_summary_invalidates` forces deadline crossing at actual final write; `test_no_public_initializer_retry_or_endpoint_resume` rejects production overrides.
- [ ] Add boundary tests `test_child_routes_new_worker_and_contract`, `test_child_claimed_entry_and_token_once`, `test_missing_child_completion_suppresses_validity`, `test_successful_private_child_restore_no_initializer`, and directory replacement/symlink/unknown/nonfinite-return retention. Tiny engineering fixtures access private engine directly and never a production bypass/root.
- [ ] Run `.venv/bin/pytest -q tests/test_force_isolation_runs.py`; save RED evidence before implementing.
- [ ] Implement own orchestration/child, preserving old operation order, charge clock immediately after exclusive root reservation, pure source/gate/invocation/runtime checks before JAX/model/initializer, restore original source0, replay original baselines, global projection, exact control0 equality, scored0 then discarded warmup, actual accepted profile steps once, full128 pure-force/NumPy witnesses1/3001, immutable commits/receipts/projections/histories/predictions, attempted/returned/valid counts, all RNG/norm/body-bias protections. Do not call old6000engine or monkeypatch its old guards. New child process command points to new worker, source/gate/endpoint/root/token bound before numerical import, completion rechecked before validity. Compute distinct force-only and joint outcomes from raw endpoint before final charged summary writes. Late/unknown/nonfinite ownership failures invalidate and retain artifacts; force learning failure alone does not invalidate execution.
- [ ] Run focused GREEN plus old offset/force-contract regressions at final byte identities; inspect exact source preservation. Commit task2 and failure evidence with explicit scoped files.

### Task3: Supervisor, saved-only comparison and complete trial gate

**Files:** Create `scripts/run_force_isolation.py`, `scripts/report_force_isolation.py`, `tests/test_force_isolation_report.py`; generated later `reports/force-isolation-diagnostics.json/.md`, `reports/force-isolation-paper-findings.md`, execution archive/figure/records.

**Interfaces:** `run_force_isolation.main()->int` executes only new canonical config/new worker, exclusive invocation1; reporter `audit_saved(root:Path,*,complete:bool=True,source:dict|None=None,gate:dict|None=None)->dict`, `build_report(root:Path)->dict`, `render(root:Path,outdir:Path)->dict`. Audit wrapper verifies Task1 sources/gate/control then passes explicit source/gate to unchanged tensor audit. Report schemas include `execution_valid`, `force_only_gate_passed`, `joint_learning_gate_passed`, per-role raw force/joint gates, exact62-grid contrasts against offset control, all raw/decomposition/group/limits/work provenance; no sole trust in summary flags.

- [ ] Write failing tests for forged force/joint gates, force-only-pass/joint-fail distinction, immutable-original baselines despite projection, invalid/missing/orphan summary or source/projection/oracle/child suppression, source membership drift, same ordered IDs/grid incl3001, negative group and anchor-centered outcomes, per-role force-component pooling and populationvariance decomposition, dense-vs-sparse limits, source0 exact equality, canonical report regeneration. Supervisor tests assert new command/timeout/TERM-grace/KILL/oneordinal/no retry/exclusive root and frozen invocation binding.
- [ ] Run `.venv/bin/pytest -q tests/test_force_isolation_report.py`; retain RED before code.
- [ ] Implement supervisor and reporter signatures. Both report comparisons and force/joint gates come from saved predictions/targets, with saved tensor/receipt/oracle/child/ownership audit first. Incomplete/invalid status gives `resolve_verification_or_operational_failure; no research retry`, never numerical endorsement. Existing source/final-closeout/archival identities remain fixed.
- [ ] Run all focused new tests and the full suite ONCE at final participant bytes; retain failed runs if any, repair verified issues and rerun only affected/final required checks. Obtain independent implementation review, resolve material feedback and hash-bound followup before public launch. Include exact source runtime, config, plan/reviews, new four scripts/three tests, imported oldcore/worker/reporter/helpers, test logs and source maps in immutable prelaunch gate. The old656-suite evidence is source context, not evidence for new implementation. Reuse dedicated checkout; no installation/worktree cleanup/GPU/fullpilot.
- [ ] Append final prospective accepted/attempt/reference/score budget/settings/source/gate checks and gate PASS; commit launch source before generating hash-bound launch revision. Execute `.venv/bin/python scripts/run_force_isolation.py` exactly once when implementation/launch gates are complete. Preserve process IDs/logs/witnesses/failures, monitor with≤60s user updates, no tuning/restart.
- [ ] After numerical closure, perform separately timed saved-only JSON/Markdown rebuild equality, raw numerical inventory/ZIP exactunique members/allmemberhashes/CRC, matched62 rawE/F figure/caption/visualQA, independent endpoint/group/variance arithmetic, fresh independent saved-result review, preserve incidentals/failed/interrupted attempts, final source/runtime/oldraw/protectedprefix checks, balanced paper findings and append-only log. Update Obsidian via explicit vault-first CLI, real-note backups and exact readbacks; preserve diagnostic responses instead of treating them as note content. Record actual outcome and next decision, even if both gates fail; fullpilotpending. Scoped result commit leaves user PDFs and all prior review folders intact.

## Self-review and handoff

All spec sections map to Task1(source/control/config), Task2(numerics/state/child/budget) or Task3(supervisor/audit/metrics/closure). Interfaces use one consistent source dict and explicitly owned new public boundaries. Scientific hypothesis versus mathematical gauge construction, original/joint/force gates and compared grids remain distinct. No source path is guessed during execution: exact15context paths are retained in prospective planning context and frozen by the source-map membership contract. Two independent plan reviews must cover this final exact design/settings/plan/source context before any implementation. Material revisions go back to both reviewers; primary weighs feedback against evidence. This plan does not report an unexecuted trial as a result.
