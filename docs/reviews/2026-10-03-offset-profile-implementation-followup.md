# Offset-profile implementation followup — 3 October 2026

Verdict: **PASS for the implementation repairs at HEAD `06d35b7746e4fd9fe492f8393ec3c4a65fba5d44`.** R1–R4 and the material oracle/child coverage concerns in the prior implementation review are resolved. No remaining material implementation issue was identified within this read-only review. This verdict is not a completed full-suite gate, a production result, or a substitute for the final frozen prelaunch gate. Production must remain unlaunched until the final suite passes for these bytes and the gate pins that evidence and this review.

## Evidence and scope

Reviewed the exact repair diff from `ac13a199b36c5dcbc924e9c3453c16a68de5a1de`, current scripts/tests, the preserved prior review, resolution record, meaningful old-byte RED witnesses, latest focused log, and the saved-closeout and launch-gate utilities. No tests were run alongside the primary's active full suite. No participating source/configuration/test/history/raw file was changed; only this separate followup review is written. No production numerics, model evaluation, initialization, or research root was created by this reviewer.

The resolution participant hashes independently match the current files. Latest focused evidence is **41 passed in 15.57s**, SHA256 `f06d189f272ddf5cacdc369fef5ebfe7f7138d1d502b0a2917912fe1d1fb4755`. The preserved frozen-old-byte final RED witness records **9 failed, 18 passed in 10.05s**, SHA256 `9ac4512164120ccb1dd4e3ea498b73823c6205c3b89727e7179e6925fb5d235a`. Its failed cases include altered saved gates/code/oracle incoming identity, final durable-write checks, original invocation binding, successful-child restoration, and the actual engine accepting a late summary under the old implementation. This is meaningful regression evidence, including an actual old behavior failure rather than only missing-new-helper errors.

The original full suite's **647 passed in 476.45s** applies to old bytes only. Final `fullsuite-v2.txt` remained partial at inspection; its eventual final result must be independently checked before launch. The diff of `src/`, `pyproject.toml`, `uv.lock`, `runs/`, the offset configuration and the reviewed spec/plan is empty. PDF user changes are preserved. The production root does not exist.

## Required findings resolved

| Finding | Verified implementation and regression evidence | Assessment |
|---|---|---|
| R1 — independently recompute raw gates | `report_offset_profile.py:62`, `:110-112` computes strict raw E/F fractions from fresh saved-array metrics and the original unprojected zero baseline for every score row, requiring equality with stored gates. `:239` reports the independently recomputed endpoint gates. The gate mutation refreshes the superficial history/commit receipts and is rejected under new bytes. | Resolved; no recalibration of the gate baseline. |
| R2 — new launch-code attribution | `train_offset_profile.py:54-59` requires local code/provenance equality, verifies source/runtime participants on the cleared production gate, and binds fingerprint, revision and the complete gate code record when present. `report_offset_profile.py:71` applies this to saved checkpoints and the child audit. Mutations and separate fingerprint/revision regressions exercise the intended checks. The private fixture's code boundary remains confined to its supplied private gate; public `verify_gate` still requires a cleared frozen gate. | Resolved. |
| R3 — final durable writes and cap | `train_offset_profile.py:66-79` charges journal and summary writes, checks after the elapsed-record durable write, and irreversibly records an invalid wall-budget outcome if late. `run_owned` uses this finalizer. Tests inject late work/summary writes, including an actual real-model controller completion that becomes late on summary persistence. Saved and returned validity/stop reason both fail closed. | Resolved for strict-cap validity. |
| R4 — original supervisor invocation | `train_offset_profile.py:61-64` binds ordinal, full gate object, canonical configuration and root to the supervisor record; `:409` checks this before numerical entry. The next launch-code validation binds gate code fingerprint/revision. `run_offset_profile.py` persists the canonical config identity in its exclusive invocation. A replaced still-cleared gate is rejected by the real-file regression. | Resolved; an engineering identity clarification, with no scientific protocol change. |

For R3, persisted elapsed time is a measured durable-write boundary. An invalid corrective summary write can make the in-memory returned elapsed value slightly later than the persisted measurement; both remain invalid with the same wall-budget stop reason. This is a conservative failed outcome, not an avenue for successful execution outside the cap. File-descriptor cleanup follows the finalizer; it is not numerical work or another durable scientific artifact write.

## Oracle and child followup

`offset_profile_core.py:154` now emits a dedicated independent-reference projection vector and complete correction receipt: count/atoms, raw signed mean/MSE/variance, before/after bias, correction and algebraically predicted post-statistics, with raw/projected reference all-field hashes. `train_offset_profile.py:359` retains an immutable NPZ and receipt binding phase, ordinal, incoming/outgoing updates, ordered fitting IDs/input hash, config/gate, actual incoming/candidate and vector identity. Reference work remains explicitly discarded; the actual production candidate is still accepted only once.

`report_offset_profile.py:162-192` checks the oracle update pair and preceding accepted incoming identity, exact oracle artifact membership/hashes, raw and projected reference all-field identities reconstructed from serialized tensors, the two reference bias values, NPZ residuals against retained full-batch reference energies and pinned targets, and all correction statistics. Existing full-gradient, independent Adam/moment, accepted-state and E/F comparison checks remain. This resolves the previous reference receipt schema and incoming-state-binding gaps. The existing tiny saved audit exercises the fuller artifacts successfully without invoking the model.

The fresh-process private fixture now enters `probe_owned` only after a real token-bound claimed entry and reproduces **14 role entries** from saved state at absolute error at most `1e-10`, with zero training updates and parameter initializations. It uses the same numerical child boundary as production, while the public probe still requires canonical configuration, production root, reviewed endpoint and source/gate checks. This test exposed the existing-staging-directory mkdir failure; `Store.child` now reopens an existing child through the same directory lease, retaining parent/inode/symlink ownership checks. Existing child nonzero/missing-completion and substitution tests remain. The private fixture proves the successful mechanism, not production-width performance or the 144-entry endpoint result.

## Ancillary utilities and remaining launch/closeout work

Inspected `make-launch-gate.py` and `close-saved.py` read-only. The gate utility requires passing final log lines, freezes code/config/source count and declared participant/review/log hashes, and writes exclusively. Its `implementation_review: PASS_resolved` label is supplied by the primary's invocation; the utility does not independently infer a verdict from review text or establish that a log was produced against those bytes. The primary must use this actual PASS followup and the completed matching final suite, as already required by the reviewed workflow. This is a process requirement, not a remaining implementation defect.

The saved-closeout utility regenerates JSON/Markdown twice and checks byte equality, captures raw inventory, verifies exact unique ZIP membership/CRC and each member hash, checks unchanged raw inventory and frozen source/gate, and separately times its work. Plotting uses the 62 matched postzero grid with original raw baseline fractions and a caption retaining fitted-support, sparse-observation, construction-effect and historical-reference limits. It leaves independent saved-result review pending. These are source inspection findings: no closeout artifact has yet been generated or visually verified. Archive, report regeneration and plot correctness must be confirmed after actual closure, including a failed/incomplete attempt where applicable.

## Declined-to-judge items

I do not infer learning success, improved forces/centered fitting, excursion mechanism, convergence, generalization, or causal superiority from these engineering checks. I do not certify full-width full-support oracle compilation, fsync/serialization cost, the 6001-receipt child audit, or successful completion within 3600 seconds; the production attempt remains once-only and any failure must be preserved without retry. I do not certify the still-running final suite, final frozen launch gate, actual 6000-update state/exposure/child receipts, post-run source closure, report regeneration, archive/CRC/byte checks, figure visual QA, independent saved-result review or final Obsidian readbacks before those artifacts exist. The full pilot remains pending.

## Exact participating identities

| File | SHA256 |
|---|---|
| scripts/offset_profile_core.py | 2a416258f87a01632166a20b16d472300914fddfc5297b8f7319d76907bebb45 |
| scripts/train_offset_profile.py | 416e9084c2c13ecb6bfeb824c6b4c233c5376348d027627a9df7dee6281cfe27 |
| scripts/run_offset_profile.py | 3063a3ede84eda49eb5b5ef5f4b94a94072daeed9c4626512d81cceea05b6c30 |
| scripts/report_offset_profile.py | 25dc9af0b4a67aac177952b4b542e33cbc65921bb0953705b445f3c8c2cc050a |
| tests/test_offset_profile_core.py | 3b0883babff0a315805af7215fc1cc394e40cda6b14b7af14c171aa5889cd277 |
| tests/test_offset_profile_runs.py | 6c8e15efdfd90d4455ef42daf285ed2564bd6cd76ca51e45174b5d72ba27384d |
| tests/test_offset_profile_report.py | e3190613c02091eb0cf8ed3ab6b7b5a4ab6b9489fbe97eae60f0f5bfbc4f2de0 |
| configs/aspirin-offset-profile.json | 03048bf92954cfe7d62d509af792595690d98b53f8a10f05ea9f55ffd2b3ec4a |
| docs/superpowers/specs/2026-10-02-offset-profile-design.md | b7c2d5ca7fb0fa86a5fac99e2fd5db5b37301c35b290ac4e5d7a0d55369e8e63 |
| docs/superpowers/plans/2026-10-02-offset-profile.md | bfad4a08f3f056313750a1286b9df4136cf08f9b88f0cf489b05b1303c518baf |
| reports/execution/offset-profile-2026-10-02/implementation-review-resolution.json | a3e605b673f856d4ad6ed7e294ba111ce034044d8dbc738edd053219859361c3 |
| reports/execution/offset-profile-2026-10-02/implementation-focused-v3.txt | f06d189f272ddf5cacdc369fef5ebfe7f7138d1d502b0a2917912fe1d1fb4755 |
| reports/execution/offset-profile-2026-10-02/review-repairs-final-red-witness.txt | 9ac4512164120ccb1dd4e3ea498b73823c6205c3b89727e7179e6925fb5d235a |
| reports/execution/offset-profile-2026-10-02/make-launch-gate.py | 258c69cd6b0103e03eb2621caf554398f5ef8e5f0dc8e78181d571743be29f08 |
| reports/execution/offset-profile-2026-10-02/close-saved.py | 1b0f98bcb00af2f1fe4123f6dc9a31f64e8180f51a587268b695ac646720ab9a |
