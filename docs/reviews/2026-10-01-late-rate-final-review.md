# Late-rate final implementation review

Date: 2026-10-01. Scope: bounded continuation only, before production. Reviewed AGENTS.md, the current design including amendment A, implementation plan, frozen configuration, source pin manifest, new late_rate.py, all new tests, and the directly used checkpoint/Adam/gradient/metric APIs. Read the relevant evidence-log tail and retained engineering records. No production training, historical full-branch review, or duplicated full-suite run. The primary agent's full regression was still running when inspected.

**Production gate: NOT CLEARED. No Critical finding; four Important findings must be repaired and reviewed before launch.** Existing tests cover useful local guards but do not exercise the full controller/report lifecycle for these cases.

Reviewed identities:

| File | SHA256 |
|---|---|
| src/force_consistency/late_rate.py | f744f77b8ee8f4c14add7f7b103924d3aad3eb0484ce3b1a10e7cedeb7d856d2 |
| tests/test_late_rate.py | 50682d14bc425188990847ce66e0dc2953fd8f62ba58ff4f96c0e5b955467a91 |
| docs/superpowers/specs/2026-10-01-late-rate-design.md | b29226945ee5aed91b69166b2a8e3b699aefcacec11d5c6a03ffb8fccb9068ad |
| docs/superpowers/plans/2026-10-01-late-rate.md | 1deffd24d55b0f6a5817ee292db639cffb5182adc6e9e188bb280dd94b540c1e |
| configs/aspirin-late-rate.json | 690173c68507499204ac67fcb005428014ce69915b58621563d01c9d483008dc |

## Important findings and focused fixes

### I1 — Performed work is dropped from partial report totals and prediction work is undercounted

Locations: late_rate.py:174-175, 240, 322-325, 439, 465-487, 556.

When _validated_arm fails for an interrupted or incomplete arm, build_report retains raw_summary but creates no top-level arm counters. The aggregate additional_updates/additional_frame_exposures therefore silently defaults that arm to zero. A no-training reproduction with two incomplete summaries each recording two computed updates, one persisted update, three attempted updates, and one unknown pending operation reported zero aggregate updates/exposures, although four computed updates/512 fitting exposures were saved. Contrasts were correctly unavailable. Reproduction retained in 2026-10-01-late-rate-final-review-reproduction.txt.

Separately, load_parent evaluates both source prediction reproduction and raw endpoint coverage metrics over 144 frames; parent_load_prediction_frames records only 144 rather than 288 per successful parent load. Score evaluations, final reference production, and child verification are not assembled into a complete separate prediction-work account. Reporting a warmup frame count and oracle proposal records alone does not fulfill the amended separate-work accounting requirement.

Fix: distinguish performed/returned computed work, attempted work, receipt-verified persisted work, complete-arm comparison work, and unknown intervals in the saved report, preserving known partial counters without promoting an invalid arm to valid execution. Reconcile partial persisted counters against the durable commit/history when available. Do not replace unverifiable work with an exact zero. Record actual parent source-reproduction and endpoint-scoring frames separately; explicitly count scoring, reference generation, child verification, template initialization, warmup and discarded boundary proposals, with partial/unknown status for interrupted operations. Avoid counting inherited 6000 updates as new training.

Focused verification: one failed arm after scored6001 and unscored6002 plus an unknown6003; one publication orphan; one blocked unstarted arm. Assert performed totals, persisted totals, unknown status and suppressed contrasts. Confirm successful parent load reports two144-frame prediction passes, and all other prediction/proposal categories match actual calls.

### I2 — The shared wall cap closes before substantive final reporting/persistence

Locations: late_rate.py:540-563.

execute freezes completion elapsed_seconds and checks the6000s deadline after the raw inventory is written. It then performs build_report, which reads and hashes the complete raw tree and checkpoints, writes both final reports, and appends the final evidence entry. There is no subsequent time check. These operations belong to setup/verification/output persistence under the amended shared cap. Crossing the deadline during this finalization still returns execution_valid=true and complete endpoint contrasts. The120s dispatch reserve does not prove finalization stayed within that reserve.

Fix: close the controller only after all substantive report/evidence persistence, with a final elapsed/cap check and durable closure identity. Preserve the immutable raw snapshot/inventory separately from the final closure/report identities to avoid self-hash cycles. If finalization overruns, mark execution invalid, remove complete contrasts from final deliverables, retain actual elapsed/stop reason and all computed work, and persist the corrected closure. Arm post-summary cap rejection is already present and should remain.

Focused verification: deterministic fake clock, with numerical arms finishing below caps and report/evidence writing advancing beyond6000s. The returned report and durable final report/closure must be invalid and contrast-free; no additional arm or training occurs. Also cover a valid close below the cap.

### I3 — Saved reports do not bind an arm to the expected schedule name

Locations: late_rate.py:381-383, 408, 420, 459-463; child contract checks at348-353.

_validated_arm derives arm from local provenance and uses that same local value to check saved rates and boundary oracles. build_report stores that result under its requested directory name without requiring returned/provenance/contract arm to equal the requested name. A complete constant arm copied under late_decay therefore passes the same local schedule checks; both results can be exposed as a constant-versus-decay contrast even though both contain the constant schedule. An inventory assembled from those files hashes the duplicated artifacts successfully; this is distinct from simple later tampering with an already frozen inventory, which is rejected. The checkpoint-contract arm/update/normalization fields are also not all reconciled against their corresponding local values.

Fix: accept an expected_arm in the saved validator (or explicitly enforce it in build_report), require expected name == provenance arm == summary arm == contract arm == child binding arm, and reconcile contract update/normalization with config/provenance/checkpoint. Enforce the same local contract consistency in the child probe. Never use an unverified local arm value to select the comparison label.

Focused verification: a constant arm duplicated under late_decay with an honestly regenerated raw inventory must fail complete comparison. A changed contract arm must fail probe/report. Genuine constant and late_decay arms remain valid.

### I4 — Local saved lineage is self-consistent but is not reconciled to the frozen parent

Locations: late_rate.py:253-258, 336-342, 377-401, 410, 455, 472-473.

The production loader checks the true source well, but the saved-only completion validator never compares saved parent_checkpoint_sha256/parent_provenance_sha256 to the pinned original receipt/provenance. It also does not compare original_baselines, model, normalization, support/array identities, source_pins map, initial state hashes or initial raw metrics to the frozen parent's corresponding values. It validates local input arrays against local metadata and acceptance gates against local baselines, then compares the two arms' initial records only to each other. Both arms can share a different self-consistent start or baseline and still satisfy these local report checks. Verifying the original global pin manifest does not establish that these arm-local values descend from it. The child binding identifies its local provenance/config but does not independently reconcile that provenance to the expected parent lineage.

Fix: add a saved-only expected-parent contract reader after verifying the frozen pin manifest; use the pinned parent receipt/provenance/history/config/contract to compare the relevant model, normalization, support, arrays, original baselines, source map, parent hashes, start hashes and metrics. Require exact144-frame support and absolute source-reproduction tolerance for complete production arms. Reuse that comparator in production child/report routes while retaining an explicitly isolated internal tiny-fixture route if needed. This needs no model evaluation and does not require old and current code fingerprints to match.

Focused verification: wrong parent hash, jointly shifted initial hashes/metrics, altered local original baseline with recomputed local gates, local support/model/norm substitution, and empty/substituted local source map must fail even when the local receipts/summary and regenerated inventory agree. Unchanged learning failure remains valid execution once all original lineage checks pass.

## Five requested failure modes traced

| Focus | Finding |
|---|---|
| Nonzero Adam6000/moments and8001 boundary | load_parent uses exact original JSON provenance; validate_state checks count/update/norm/dtype; nonzero incoming moments are required. _boundary uses full128-frame gradient with independent NumPy Adam and compares it to accumulated128-frame proposal and actual committed update at6001/8001. New scalar-rate transforms consume the existing optimizer state. Tiny multistep tests include7999->8001 and actual fresh child. No reset defect found. |
| Reference/arrays/provenance disagreement before committed work | Frozen config and source pins are checked before loading; parent receipts/contracts/config/history/child identities reconcile; _parts verifies bytes and array identities; original prediction reproduction and raw metric checks precede continuation. Parent paths are pinned and source checks repeat. Local launch behavior is sound; saved-lineage gap is I4. |
| Failed/partial child or receipt cannot create complete contrast | Probe process return/timeout and exact child binding are checked; failed arm blocks later arm; summary failure gates saved validation; raw receipt/history/reference/input/child corruption blocks contrast. I2/I3/I4 are remaining complete-report validity holes. |
| Computed/persisted/unknown/discarded work | Phase precedes training blocks, warmup/oracles/probe, bounded block interval is durable, attempted/computed/persisted counters are distinct, and orphan snapshots are retained without receipt credit. KeyboardInterrupt and publication-failure tests are meaningful. Report/work issues are I1; shared finalization cap issue is I2. |
| Raw energy errors versus residual decomposition/baseline | saved_observations reads saved reference/labels without model evaluation, reconciles per-record/group aggregate and signed mean-squared plus centered variance to raw MSE. Endpoint report reconstructs raw residual/group metrics; gates use original_baselines and never the centered residual variance. Frozen update0 acceptance remains unchanged. No recentering defect found. |

## Minor findings

M1: verify_code checks recorded file bytes, dependency versions and Python, but not self-consistency of code_fingerprint/lock_sha256 against code_files or current full revision. The current provenance creator writes these correctly; add consistency checks and explicit revision policy if full revision identity is intended to be enforced rather than merely recorded. No old/new code equality requirement should be introduced.

M2: _prediction_error does not explicitly assert finite actual/reference predictions before maximum-error accumulation. NaNs can evade a max comparison; downstream report residual validation currently rejects nonfinite references. Add finite checks in the source/fresh child proof itself so the proof cannot emit a misleading passed receipt.

## Evidence and next gate

The retained focused GREEN artifact records33passing tests, including real tiny subprocess/count restoration and an actual144-frame parent reproduction. These are engineering checks, not the continuation result. The fresh no-training report reproduction independently confirms I1. I2-I4 are code-traced findings; no invented executed corruption trial is claimed. The full suite was not rerun by this reviewer, and production remains unexecuted.

Next gate: implement focused repairs, retain RED/GREEN attempts and every review record, run appropriate targeted/full verification, then obtain bounded follow-up review of the material changes before the once-only production launch. No numerical protocol change is required by these fixes.
