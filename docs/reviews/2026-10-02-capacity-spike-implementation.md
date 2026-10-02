# Capacity spike diagnostic implementation review, 2 October 2026

Verdict: **NEEDS_REPAIR**. Numerical launch remains gated. This is an independent prelaunch implementation review against the final amended design/plan and both plan followups. No research diagnostic or training was executed.

Base HEAD: `b823ac2c912f625d965792c7081f4c5a9bbe6397`. New script/config/tests were unstaged. Exact inspected bytes are listed below; this verdict does not cover subsequent revisions.

## Material findings

### E1 — Preserve every returned prediction microbatch before a later call or verification

`scripts/diagnose_capacity_spikes.py:307–327` accumulates a complete role's prediction arrays in memory, then checks historical scores before writing predictions.npz. A timeout or interruption in the second prediction call loses the first completed microbatch's predictions. A score mismatch loses all returned predictions for that role. Known-work counters survive, but the promised partial numerical observations do not. This matters because execution is once-only and these saved results cannot be recreated with an authorized retry.

Independently reproduced with deterministic stubs: a second-call TimeoutError and a historical-score schema mismatch both produced `predictions_retained: false`; both did preserve `state-audit.json`. No real model or derivative was invoked. Repair by saving exclusive per-microbatch prediction/error records, exact ordered record IDs and partial/completion metadata before the next call or score check. Keep the whole-role convenience artifact, but retain the microbatch evidence on every failure. Add regression tests for both fault paths.

### E2 — Support saved-only reporting after operational termination

`render` at lines477–479 unconditionally reads root summary.json. That file is written only near the end of `execute` at line471. The prescribed 1860-second supervisor kill, or another abrupt interruption, can leave config/partial/work artifacts but no root summary. Report-only then raises FileNotFoundError rather than retaining the attempted case, unknown in-flight work and downstream blocked cases in a reviewable report.

Independently reproduced with a minimal interrupted-run directory. Repair by synthesizing a conservative interrupted summary in a saved-only recovery path, retaining existing case statuses/partial pointers and identifying missing downstream cases. Recheck source integrity in saved-only mode or mark it unverified/false. Never infer returned or valid work from an attempted call, never execute numerical work or retry, and do not endorse a case missing completion evidence. Add a terminated-run report test.

### E3 — Reconfirm the separately pinned analysis artifacts at closure

`verify_gate` hashes the script/config/reviews/test logs before launch, but the success and failure closure paths call only `verify_inputs` at lines452 and465. Those source guards cover historical raw files and frozen src/runtime, not the analysis script/config/review manifest. Thus analysis participant bytes can drift during execution without invalidating the global scientific endorsement, despite the reviewed before/after identity requirement.

Repair by checking the exact frozen launch artifact map again on both closure paths and routing any mismatch to global integrity failure, suppressing all endorsements. Preserve the original launch manifest rather than accepting a replacement manifest. Add a fault-injection test where a separately pinned analysis artifact changes after the initial gate; historical raw inputs may remain unchanged.

## Checks that passed inspection

- Pure byte audit found all347 source pins correct, all343 raw inventory members exact, and no diff in src/, pyproject.toml or uv.lock. The report/config/source allowlist and historical canonical-report gate are bound to fixed hashes.
- The eight cases and ordering match the reviewed design. Exclusive root creation prevents a second ordinary invocation. The monotonic budget starts after exclusive reservation and before guards/imports; each instrumented kernel call checks the 1740-second numerical cutoff, and final elapsed time gates total1800 seconds. Actual supervisor launch is still an execution prerequisite.
- Snapshot loading uses the frozen serialized envelope/schema/count/normalization/history-hash audit followed by msgpack_restore; no model parameter initializer or optimizer step is called. Parameter dictionaries alone feed model evaluation. Both widths use original prepared fitting/anchor inputs and normalization.
- Readiness precedes known-returned counters, finite validation precedes valid counters, and a call without a ready return stays unknown. Expected loop totals reconcile to72/256/512 calls and1152/4096/8192 exposures. These are separate invocation categories, with zero training updates/proposals.
- Energy and force derivatives use the frozen separate kernels; independent unweighted and production-weighted joint derivatives and losses are checked. Geometry relabeling recursively reaches leaves/blocks. Atomic head bias uses prediction-minus-target residuals, atom count21, squared energy scale, and production energy weight. Its force derivative is checked against zero.
- Finite differences use unit gradient or seeded Rademacher fallback, both fixed steps, and at-least-one-step agreement per objective. The fallback draws directions, not new model parameters. Partial gradient/FD receipts persist during accumulation.
- The full decoded state (including optimizer, counters, RNG and normalization) is hashed before and after analyze_case, including caught numerical failure via finally; the independent stub faults confirmed state-audit retention. Source checkpoint hashes are also checked in finally.
- Saved-case endorsement requires completed summary and verified observations plus global source integrity. Complete contrasts require valid overall execution. Scientific labels retain fitted-support, posthoc/sparse-selection, directional-FD and non-Adam/causal limitations. Signed component increment shares remain unclipped and undefined for nonpositive MSE increase.

## Evidence and limitations

Reviewed existing focused evidence reports **83 passed in25.97s**. I did not rerun those tests or run the full suite concurrently. The full suite and revised focused tests must cover the repaired exact bytes before launch. Existing tests do not yet explicitly cover prediction retention, abrupt report recovery or final analysis-byte drift. Additional small coverage of FD first-step-fail/second-step-pass would better protect the exact criterion, though the current any-step code is correct on inspection.

Independent evidence lives in `reports/execution/spike-diagnostic-2026-10-02/implementation-review/`: inspected-hashes.json, pure-source-verification.json, failure-retention-check.py, failure-retention-results.json, failure-retention-output.txt and the preserved stub output folders. These engineering checks constructed no EnergyModel, initialized no model parameters, performed no real prediction/gradient, did not restore production states and changed no source/raw artifact. The review creates only review evidence.

One harmless inspection failure is preserved here: a read-only search referenced nonexistent `tests/conftest.py` and returned an rg missing-file diagnostic; relevant float64 enablement was then located in `src/force_consistency/models/backbone.py`. This did not affect the findings. The injected timeout, score mismatch and missing-summary exceptions are intentional engineering evidence, retained verbatim in the files above.

## Exact inspected SHA-256 values

- `scripts/diagnose_capacity_spikes.py`: `565d7dfe54dcba59658aa3491972a8a039585388ee474a5023453333ddb5edd4`
- `tests/test_capacity_spikes.py`: `58a5040ebe1d217cffab2a033d121542dd1d9f87a5c86b97b817981ffdcdf7be`
- `configs/aspirin-capacity-spike.json`: `73463e015fd8d9282a039cd68a1a11fa26f1febc6dbfa1877fb093af84a1248e`
- `docs/superpowers/specs/2026-10-02-capacity-spike-design.md`: `d9c5b564ba0771c17f7c94cbf41a7d553e8838ec39b55c657cb932a607feb96b`
- `docs/superpowers/plans/2026-10-02-capacity-spike.md`: `eb4bf915c5ca7a1efc1a7b90246a40ef8b59bd98afbf4a695cbc073ea7377e1e`
- `docs/reviews/2026-10-02-capacity-spike-engineering-plan-followup.md`: `a7ba890f33664e5da14294e9d6c52829f019941a3d1791f7d9c13b8e9c4eef71`
- `docs/reviews/2026-10-02-capacity-spike-scientific-plan-followup.md`: `bec2486cd44e7e18f4cadcf04486aaeb0eb52d503b335a85ad5888ad94ae6708`
- `reports/execution/spike-diagnostic-2026-10-02/source-pins.json`: `26f6914685dc38709652408b015356c7e4fd38bd5d1b392c9550c2cb2719c130`
- `reports/execution/spike-diagnostic-2026-10-02/progress.md`: `9ec7fd384dd2694f4b5fd6760e418e75b5d7684d63d080ded1a2a2f02a75e478`
- `reports/execution/spike-diagnostic-2026-10-02/focused-verification.txt`: `6f66cf76e0ce23b79bea36025307313747b421835be7df6f501631f905d35c2c`


## Repair followup 1 — E1/E2 resolved, E3 manifest identity still needs repair

Verdict: **NEEDS_REPAIR** for script SHA-256 `9ff93f8ea5742de614a08eabcbee61601095f4b4face7dab84e10b8a4b886210`, tests `92dcb13805b94ab064bc05da7ae86c8d2e6e433e9fb89d6fe845445b8ea0db84`, unchanged config `73463e015fd8d9282a039cd68a1a11fa26f1febc6dbfa1877fb093af84a1248e`. The initial findings and independent reproductions above remain historical evidence.

E1 is resolved: Journal invokes on_returned after readiness/known-work persistence and before finite validation; each prediction microbatch is saved with exclusive NPZ creation, exact ordered IDs, predictions/targets, state hash, NPZ hash and separate returned/finite-valid status. Both the later-call-failure and history-mismatch regressions now pass. Full-role score rejection cannot erase earlier microbatch observations.

E2 is resolved: render reconstructs an interrupted summary when root completion is absent, performs fresh/conservative integrity checks, includes retained per-case work with unknown in-flight state, marks missing attempted cases interrupted and remaining cases blocked, and permits no complete-run contrast. Report reconstruction invokes no model kernel or retry.

E3 is partly resolved: closure now invokes both historical source and analysis gate verification on successful and failed paths. However, verify_execution_integrity rereads the mutable external gate without comparing it to the original launch gate saved in the run. A changed artifact plus a regenerated external manifest can therefore pass numerical closure, although render later compares saved/external gates and suppresses endorsement. Complete the originally requested preservation of the launch manifest by binding closure to its original object/map and rejecting substitution. Add an injected replacement-manifest test. This is the remaining material issue.

Inspected retained repair RED:4 failed/26 passed in8.74s, GREEN:30 passed in12.37s; updated focused evidence:87 passed in30.11s. I did not rerun numerical fixtures or the full suite. The primary records full-suite attempt1 intentionally interrupted at179 passed in145.09s before source repair; this is not a completed-suite claim. No new inspection failures occurred in this followup.

- Followup evidence `reports/execution/spike-diagnostic-2026-10-02/review-repairs-red.txt` SHA-256: `3ed775594d0326b4e2d2d3c9c5df624aaf49eb773bfe273ebaff3de7d470b91a`.

- Followup evidence `reports/execution/spike-diagnostic-2026-10-02/review-repairs-green.txt` SHA-256: `77c7942233e5479bd0fef2fea27029c84ff05eb24d04d8816a7a8e4edb677fb6`.

- Followup evidence `reports/execution/spike-diagnostic-2026-10-02/focused-verification.txt` SHA-256: `2f3f9ff6491f5ed363d4261ef574f7c8b4c717aa4aa7b0ca3747363655610bd2`.


## Final implementation followup — PASS

Verdict: **PASS** for the final exact implementation hashes below. All three material findings E1/E2/E3 are resolved on inspected code and retained focused evidence. This supersedes the earlier implementation verdicts without deleting their history or reproductions. It clears the independent implementation-review gate; required full-suite completion, final prospective evidence, scoped checkpoint and frozen launch manifests remain prelaunch execution prerequisites. This is not a numerical result or a claim that the diagnostic has run.

E3 now binds both numerical closure paths to the gate object captured at entry. verify_execution_integrity checks historical sources and current analysis artifact hashes, then rejects any manifest unequal to that original gate. Report-only loads the run-frozen gate and uses the same comparison; a missing frozen gate conservatively prevents endorsement. The amended regression changes an analysis file, confirms rejection, regenerates the external gate to match the changed file, then confirms that substitution is also rejected. Retained targeted GREEN:1 passed in7.16s. Final focused evidence:87 passed in34.18s. E1/E2 changes and their tests remain as approved in followup1.

I additionally inspected `scripts/run_capacity_spikes.py`. It verifies the analysis gate and source guard, checks the separate launch guard's exact launcher/config/prelaunch hashes, rejects an existing numerical root, exclusively reserves ordinal1 in retained execution records, opens stdout/stderr exclusively, and spawns the child in a new process group with the same Python interpreter. It waits1860seconds, sends SIGTERM to the process group, permits10seconds for termination and escalates to SIGKILL if necessary; it never retries. Completion stores actual exit code, stop reason, elapsed time and stdout/stderr hashes. Child-side root reservation remains the numerical once-only barrier and the internal1800second/60second-reserve timer remains distinct from operational supervision. Source revision is recorded by the supervisor and independently recorded from the actual repository by the child. A zero process exit alone is not scientific validity; saved case/run verification remains authoritative.

I performed only read-only source/log/hash inspection and syntax parsing for this followup, then appended this review. No research prediction, model construction, parameter initialization, derivative evaluation, optimizer work, production restoration, supervisor launch or test-suite rerun was performed by the reviewer. No additional inspection failures occurred. Full-suite completion is pending at this review and must be bound in the final prelaunch map before execution. Freeze this review after that map is produced; place later result reviews in separate files so the pinned implementation review remains immutable.

### Final inspected SHA-256 values

- `scripts/diagnose_capacity_spikes.py`: `8bfa6ff4bae33e2a4175f5dadc7cd237eabb607260fe03951d22357870447924`
- `scripts/run_capacity_spikes.py`: `fe93ff7040ce59738b96038daffca5f135df53fb25b54e0aeac6ca29fd56bc37`
- `tests/test_capacity_spikes.py`: `3690c2c506e60df9575fa67534b8ab84c0fab25263c519e89c0d9be924fa74e1`
- `configs/aspirin-capacity-spike.json`: `73463e015fd8d9282a039cd68a1a11fa26f1febc6dbfa1877fb093af84a1248e`
- `reports/execution/spike-diagnostic-2026-10-02/gate-substitution-red.txt`: `1a7352f71a5e39d560678700a0fcf8ff9acb032bccb8646dc2a442a154fe111c`
- `reports/execution/spike-diagnostic-2026-10-02/gate-substitution-green.txt`: `e6ce24532f0ff4df9a085bad48206e2f6f8e3db793f63851e02979637a1659f7`
- `reports/execution/spike-diagnostic-2026-10-02/focused-verification.txt`: `f665c2d10f1a84b8e2469e02a0634c83d12b2d823142e5c77dea2b0c6b5e0f53`
