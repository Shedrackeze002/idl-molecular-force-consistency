# Offset-profile implementation review — 3 October 2026

Verdict: **CHANGES REQUIRED before the once-only production launch.** This is an independent read-only prelaunch implementation review, not a measured result or permission to execute production numerics. The mathematical implementation appears consistent with the reviewed intervention. Four completion/audit defects below must be repaired and independently re-reviewed against fresh targeted and full-suite evidence.

## Scope and evidence

Reviewed `AGENTS.md`, the final design/plan and both final plan followups, all four offset scripts, all three new test files, the configuration, saved engineering logs and relevant unchanged state/source/Adam helpers. Reviewed base `f06c404` through HEAD `ac13a199b36c5dcbc924e9c3453c16a68de5a1de`, including Task 1, which already existed at the supplied base. No source, test, configuration, runtime, historical raw data or PDF was changed. Only this review report is written. No tests or numerical jobs were launched by this reviewer; the primary's full suite was running concurrently.

The focused evidence file records **32 passed in 12.06s**, SHA256 `da324d6d35d7baee5710b68d566bf66f97ca57966c586d66db04b457d994ed05`. Earlier failed engineering trials are retained and described in the evidence log; they are not treated as current failures. The full-suite log was still partial when inspected and provides no final passing verdict for this review. The prospective production root does not exist. A diff of `src/`, `pyproject.toml`, `uv.lock`, and `runs/` across the supplied commits is empty. The final plan gate's declared artifact hashes match the current files. User PDF changes remain outside this review.

## Required corrections

### R1 [P2] Recompute gate decisions during the saved audit

**Locations:** `scripts/report_offset_profile.py:101-106`, `:210-212`; producer `scripts/train_offset_profile.py:278`.

The audit independently recomputes each saved prediction's raw E/F metrics and compares them with the history, but never recomputes `fitting_gate` or `anchor_gate`. The final report reads these stored gate objects and derives `learning_gate_passed` from their `accepted` flags. A self-consistently republished history/commit with altered gate flags or fractions can therefore pass the metric audit and make the report declare a learning success that its independently recomputed metrics do not establish. The normal producer computes the right gates; the defect is the independent saved-verification boundary.

For every scored row, compute the gate from the recomputed metrics, the pinned original **unprojected** zero-state baseline, and the fixed fraction, and require complete equality with the stored gate. Derive the final reported gates from those independently computed values. Add a real saved-fixture mutation that changes only a gate decision/fraction and refreshes superficial history/commit receipts so rejection is specifically attributable to the gate recomputation, rather than an earlier hash mismatch.

### R2 [P2] Bind new launch-code provenance during saved verification

**Locations:** `scripts/report_offset_profile.py:62-75`; child `scripts/train_offset_profile.py:191`; live launch `:369-371`.

`audit_saved` verifies many source/gate fields but omits `p['code'] == code.json`, verification of that local launch code record, and binding its code fingerprint/revision to the frozen prelaunch gate. The child verifies `code.json` independently but uses checkpoint provenance `p` without requiring the two records to agree. Thus inconsistent new launch-code attribution can survive saved verification even though the historical source checks verify unchanged source bytes and runtime. This is a provenance defect, not evidence that current source/runtime actually changed.

On the canonical production route, require the local code record and checkpoint provenance to agree; verify its live source/runtime participants; bind fingerprint and launch revision to the frozen gate. Preserve the private tiny fixture only within the existing private API, with no public CLI guard bypass. Add a saved mutation that changes the new provenance/code binding and updates superficial checkpoint/commit hashes, then prove saved reporting rejects it. Also test mismatched gate fingerprint/revision. Historical-code verification does not replace the new attribution check.

### R3 [P2] Charge final durable writes to the total wall cap

**Location:** `scripts/train_offset_profile.py:351-359`.

The final `elapsed_seconds` value and cap decision are computed before `j.save()` and the durable `summary.json` write. Those operations serialize, flush and fsync; their time is omitted from the persisted elapsed measurement. A run can be marked completed with elapsed below 3600 seconds even when finalization itself crosses the strict 3600-second limit. The earlier closure checks charge source/gate verification, but do not close this final gap.

Measure/check after the final durable-write boundary, persist a corrected elapsed value and conservative invalid status if the boundary was late, and make the return value agree. Account for any corrective write without restoring validity. Add an injected clock/write regression where the final journal or summary write crosses the deadline after training/child/guards succeed. The 3660-second supervisor is a containment cap, not a substitute for the scientific 3600-second validity rule.

### R4 [P2] Bind the worker to the supervisor's original invocation gate

**Locations:** `scripts/run_offset_profile.py:13-17`; `scripts/train_offset_profile.py:365-371`.

The supervisor saves its frozen gate in `production-invocation.json`, but the worker independently reloads the current gate and never compares it with that invocation. Replacement with a different still-cleared gate in the supervisor-to-worker interval can therefore become the worker's accepted baseline, and equality with that later baseline at closure will not detect the substitution. The final source fingerprint check alone does not bind the complete review/log/config/gate object to the original invocation.

Before worker numerical entry, verify the exclusive invocation's ordinal, configuration identity, declared production root and complete gate against the object frozen by the supervisor, including code fingerprint and launch revision. Preserve an explicit fail-closed path for an invocation/gate mismatch and add a real-file substituted-gate regression. Direct execution must not create an alternate unbound research route. This is an engineering identity clarification of the existing frozen-gate requirement, not a scientific intervention change.

## Positive findings

- `offset_profile_core.py:25-40,73-125` projects one scalar per-atom bias from all fitting residuals, evaluates every raw microbatch derivative at the same incoming state, weights unequal chunks by their structure counts, checks the unmasked aggregate bias derivative, and masks only that bias leaf. It does not center individual chunks. Constant atom counts make structure-weighted raw force gradients consistent across chunks.
- `:136-165` independently differentiates the full-support centered energy objective plus raw force loss, compares every masked gradient leaf, applies the separate NumPy bias-corrected Adam implementation, calibrates the independent proposal from full-batch predictions, and compares parameters, moments and E/F predictions. The real 3/3/1 fixture with nonzero readout and incoming counts 0/3 exercises a meaningful global-versus-local distinction, nonzero body moments, force invariance and input immutability. Production witnesses use the actual candidate at updates 1/3001; they do not accept a second proposal.
- Restoration decodes the known Optax tuple without parameter initialization. Source-zero parameter/optimizer identities, canonical full-field identities, and unchanged sampler/model RNG/normalization fields are separately represented; saved audits verify serialized state schema, float64 params/moments, count, zero bias moments and all-field hashes. The original envelope bytes remain preserved.
- Exact source membership is 601 payload paths plus the separately pinned spike inventory. Pure guards precede public numerical imports. Configuration fixes 6000 updates, rate/weights/model/supports and 63 score updates; both raw role gates retain original source-zero baselines.
- Root and probe reservation are exclusive. Directory leases reject inode and symlink replacement. The child is token bound to state/history/input/gate/root identities, starts with remaining parent budget, restores without initialization, and compares all 128+16 endpoint role predictions at absolute 1e-10. Nonzero exit or absent completion cannot validate execution.
- Journal states distinguish attempted/unknown, ready returned, and finite-valid work. Returned binary data precedes finite validation/strict JSON; late returned calls fail closed. Accepted updates, accepted calibrations, scored persistence and discarded warmup/reference work remain distinct. Latest work slots are intentionally overwritten after successful steps; accepted projection vectors/receipts and scored artifacts are immutable.
- Saved reporting avoids model evaluation, independently recomputes metric decompositions, matches 62 postzero historical score observations, separates dense correction statistics, and states construction, historical-reference, fitted-support, sparse-observation and pending-full-pilot limits. Missing/invalid completion suppresses endorsement.

## Specification deviations and remaining coverage

1. **Oracle-reference projection representation needs reconciliation before claiming full receipt conformance.** `train_offset_profile.py:320-326` saves oracle tensors and a phase/input binding, while `offset_profile_core.py:147-159` records the reference correction and raw mean. It does not emit the same dedicated raw-error NPZ/full projection receipt used by initial/training/warmup projection, with signed mean/MSE/variance, before/after bias and state links. The mathematical oracle itself is checked; this is an artifact-schema deviation from clarification A's “each projection vector/receipt” requirement. Either add the missing complete record/audit or explicitly record a justified equivalence to the retained raw energies/targets/params with an independent saved check. Do not silently claim every phase has the stated receipt schema.
2. **Saved oracle incoming-state binding is incompletely checked.** `report_offset_profile.py:147-161` checks tensor comparisons and the candidate full hash, but does not require the oracle's incoming full hash/update pair to match the actual preceding accepted state (0/3000) and outgoing update (1/3001). These states are scored and available. Add this binding and a swapped incoming-identity mutation while repairing R2.
3. **Child coverage is negative-path coverage, not a successful production-shaped reproduction witness.** `test_offset_profile_runs.py:95-122` tests token/duplicate/substitution and failure/missing completion, but its real child exits on the deliberately noncanonical private configuration. A successful fresh process through restoration, saved audit, role predictions and completion is not demonstrated by these tests. A private real-model child success regression below the real control/file boundary would strengthen confidence; it must not weaken the canonical public guard or consume the once-only production root.
4. The requested full-suite gate remains pending. Reuse a completed matching suite log only for exactly the reviewed bytes; fixes above require new focused/full-suite evidence and followup review before gate freeze.

These are distinguished from polish: formatting/style compactness, cleanup of intentionally retained logs, and alternate figure presentation are not launch blockers. The existing old GREEN logs with failures are correctly retained historical engineering attempts.

## Declined-to-judge behaviors

I do not judge learning success, endpoint force improvement, excursion suppression, mechanism, convergence, held-out generalization, or causal superiority from engineering fixtures. I do not certify that full-support width32 oracle compilation, dense fsync/serialization or the 6001-receipt child audit will fit the remaining real wall budget; no production-scale performance witness was run. A cap-triggered failure must remain a preserved failed/incomplete attempt without retry. I do not certify archive/CRC/member-byte checks, figure visual QA, report byte regeneration, independent saved-result review, final Obsidian readbacks or production source closure because those closeout artifacts do not yet exist. No alternate arm, production rerun or GPU job is authorized by this report.

## Reviewed participating identities

SHA256 values at the frozen reviewed HEAD:

| File | SHA256 |
|---|---|
| scripts/offset_profile_core.py | 64d19703dbd219cde7971a1af649c81c3fd7dd31ad2288e68dabdb7c78e3d684 |
| scripts/train_offset_profile.py | ad230e442780203248542de4578460e74ea0d525a23afb883b07e32b900282cf |
| scripts/run_offset_profile.py | 4de2553efb7d61a8856985418e60f59b19cc57395b9c2c8adbbc993c96515dc6 |
| scripts/report_offset_profile.py | c96899a2dd27da469c6d1858fae111ad146f2bb13e8e9dd66d80cabb2a38878c |
| tests/test_offset_profile_core.py | 3b0883babff0a315805af7215fc1cc394e40cda6b14b7af14c171aa5889cd277 |
| tests/test_offset_profile_runs.py | f5c3bd71269e51c68152a8c95ce2d69375bd3a898291d2fb7b60225f11a78e6b |
| tests/test_offset_profile_report.py | 05dc41be6a2ce2ff57bf7b45c723f101c770f5af277bef26f711676fc6845297 |
| configs/aspirin-offset-profile.json | 03048bf92954cfe7d62d509af792595690d98b53f8a10f05ea9f55ffd2b3ec4a |
| docs/superpowers/specs/2026-10-02-offset-profile-design.md | b7c2d5ca7fb0fa86a5fac99e2fd5db5b37301c35b290ac4e5d7a0d55369e8e63 |
| docs/superpowers/plans/2026-10-02-offset-profile.md | bfad4a08f3f056313750a1286b9df4136cf08f9b88f0cf489b05b1303c518baf |
| docs/reviews/2026-10-02-offset-profile-scientific-plan-followup.md | dba63b697df0e0966e20869f5455b5e5f8e9b4b1c41f036977cbf02154daa01f |
| docs/reviews/2026-10-02-offset-profile-engineering-plan-followup.md | a211f38091387b21de9aa23487d1158ab1d486da1626c58c8ebc0b6be88ff7bf |
| reports/execution/offset-profile-2026-10-02/plan-gate.json | 1fd75bfc6152b654d25952fa014d21a9d29f9fe81bf43e8a253d6f00bc96586f |
| reports/execution/offset-profile-2026-10-02/source-pins.json | c527a7b19c4ebe4aa681add2e2a649306a4d3f44a9e029eacb5a7b9b3d876882 |
| reports/execution/offset-profile-2026-10-02/implementation-focused-v1.txt | da324d6d35d7baee5710b68d566bf66f97ca57966c586d66db04b457d994ed05 |
