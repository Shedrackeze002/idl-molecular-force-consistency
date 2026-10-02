# Capacity implementation repair follow-up

Date: 2026-10-02. Original independent implementation reviewer. **Verdict: PASS for material implementation findings at the identities below.** All three initial P2 findings and the partial-child-phase advisory are resolved in the inspected implementation. No new material defect was found in the repair or the once-only launcher. Production still requires the author's full-suite verification and matching preproduction gate; this review does not assert those pending checks have passed.

## Independent checks and fixed identities

Read the repaired contract/report modules, current runner, new `scripts/run_capacity.py`, added tests and retained RED/GREEN logs. Independently repeated the original omitted-participant code-guard reproduction: it now raises `ValueError: Incomplete participant/runtime identity`. Rechecked the frozen configuration and all20 source pins with the pure source validator. The reviewer process imported neither JAX nor NumPy. No suite, training, model initialization/prediction or production was run by the reviewer.

Configuration remains SHA256 `655cfd44c0eeacd2e2cb62210b2a78eb5562a417dc01e09a707dfba931f6e960`; source-pin map remains `9611a194fc9c30906f6b9e6a4b66eb4590d279b0799b8299ca0133dc7b0e6fd7`. Thus the repairs change enforcement/reporting, not the reviewed numerical intervention.

Complete reviewed-source/test/log identities and the pure reproduction result are saved in `reports/execution/capacity-2026-10-02/implementation-review-followup-verification.json`. Principal source hashes:

- `capacity_contract.py`: `b98fc5e584d7bbe1e2b14723e6456b1f52091fe59cc7978e0ffe065e4ecc23ef`.
- `capacity_runs.py`: `e336840e110c12b005ccde6630b68c42a49ffe25191970a1dcd404a3536ab70c`.
- `capacity_report.py`: `fa81adc7fcde9280fa9cb64e4adefb2e040b8e8e95d881ab97f218a351aae27a`.
- `scripts/run_capacity.py`: `5fa3fd7865a157599ce86b70aab4647c6cb2863d92cbf06db21fc0a118d1df0f`.

## Finding dispositions

**CAPACITY-IMPL-P2-001 resolved.** `capacity_report.py:205-212` separates invalid closure/hash evidence from an honestly recorded wall-cap overrun. `_build_report:258-261` uses the same cap error string inserted by `finalize:313-317`, preserving canonical status and verification errors. The repaired post-marker overrun test now asserts exact report reconstruction and canonical saved JSON/Markdown equality as well as absent contrasts. `task3-review-repair-red.txt` retains the failing equality assertion; `task3-green-final.txt` records the repaired pass. A corrupted report/evidence binding still produces a distinct verification error rather than being accepted as a cap failure.

**CAPACITY-IMPL-P2-002 resolved.** `_build_report:221-251` loads and validates fixed controller case assignments, binds each complete case record exactly to its local summary and uses the controller-returned record for partial reported work when local summary is absent. `_partial_case` independently verifies available persisted checkpoint/history receipts, while `reported_child_work`, `local_raw_summary` and `reported_work_source` keep reported work distinct. The added missing-local-summary test preserves computed3/attempted3/exposures21 and reported child14 predictions. Its initial failure is retained in `task3-review-repair-red.txt`. The controller failure test separately verifies only the first case runs, the later case is blocked with zero known work, no later child attempt exists and root reuse is rejected.

**CAPACITY-IMPL-P2-003 resolved.** `capacity_contract.py:38-55` independently enumerates the exact current participating source/config files, requires all seven captured dependency keys, verifies file/fingerprint/lock/runtime values and requires the recorded SHA to resolve to a repository commit. It deliberately does not require saved launch revision to equal current HEAD, allowing later documentation commits. Added file-set, dependency-key and nonexistent-commit tests all failed before the repair and pass in the focused GREEN suite. Independent pure reproduction confirms the original sparse record is rejected before numerical imports.

**Partial-child-phase advisory resolved.** `_partial_case:194-199` preserves `probe-phase.json`, reported successful template initialization/frame counts and unknown prediction count without promoting phase records to full completion verification. The meaningful phase-only test failed before implementation and passed afterward. Fully bound child completion remains separately verified by `_child_work`.

## Once-only launcher and additional repair

The launcher validates the exact config/source, current code and saved preproduction gate; binds launcher bytes and all named verification artifacts; exclusively persists ordinal1/no-retry invocation before spawning the numerical process; rejects an existing production root; retains stdout/stderr, PID and exit/timeout completion; and terminates the child process group on external timeout without retry. Its operational timeout allowance is distinct from the controller's scientific wall-cap acceptance, which remains mandatory for complete contrasts. The launcher itself was inspected and hashed, not executed.

The controller source-snapshot repair captures repository CWD before entering the anchored snapshot directory (`capacity_report.py:337,348-353`) instead of inferring a repository root from the production path shape. The retained controller RED and subsequent single-test GREEN demonstrate the earlier assumption and repaired behavior for an out-of-layout fixture. This does not alter the production layout or source identities.

## Verification limits and preservation

Inspected focused GREEN evidence: `task3-green-final.txt` reports **64 passed in48.40s**. Retained review-repair RED reports2 failures; code/phase RED reports4 failures. These are engineering trials, not capacity scientific outcomes. The full regression suite was still pending when this follow-up was issued and must be captured by the author before launch. No production result or full-pilot acceptance follows from this PASS.

The initial review, unexpected sparse-guard acceptance trial, repaired rejection trial, all RED/GREEN outputs and review records remain preserved. No numerical implementation/source was edited by the reviewer and no user file was changed.
