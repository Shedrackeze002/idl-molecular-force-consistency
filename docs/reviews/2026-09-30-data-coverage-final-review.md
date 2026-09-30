# Coverage milestone final review

30 September 2026. Fresh independent gpt-6-astra reviewer, read-only; initial range c256dba..52f2893, followed by review of the single retention fix pass. Earlier independent scientific/engineering plan reviews remain in `2026-09-30-data-coverage-plan-review.md`.

The reviewer independently regenerated the frozen preflight/IDs/reference hashes; checked every arm artifact hash, complete saved histories and tracked evidence; recomputed final fitting/common-anchor MSE from saved predictions and labels within1e-14; confirmed identical fresh initialization hashes, B overlap2/not-fitted14 and C fitted16. Five targeted source/controller tests passed, and the energy figure was visually checked. Parent visually checked all3figures and separately verified A replay/all checkpoint counts/errors. Single-seed training-only limitations and exposure differences are explicit.

## Important issue and resolution

Initial controller could replace a runner exception after partial artifacts with FileExistsError: `_blocked` attempted to recreate the current directory, and the outer catch repeated the same mistake. This could lose root/downstream summaries and mislabel an attempted arm as unexecuted. It did not occur in the completed matrix; numerical outcomes remain trustworthy.

Three new regressions first reproduced the bug: partial history/checkpoint, already completed arm summary, and outer bookkeeping failure after a completed arm. The fix preserves existing summaries/history/checkpoints, writes a separate controller-failure record, keeps original errors, distinguishes execution from preflight failures and blocks only unexecuted arms. All3regressions passed in0.50s; the reviewer independently reran them in0.50s and confirmed no remaining Critical/Important issue. Full post-review suite:82passed in98.94s. No research training was repeated.

## Parent assessment and rulings

The Important finding is accepted and resolved. Critical/minor findings: none. Deferred minor items: none. Reviewer set aside full-pilot/architecture selection, generalization/causal effects, OS/process-kill crash recovery and user-owned proposal changes. Parent rulings: future protocol/model choices require separate two-review plans; no generalization/causal or crash-safe transactionality claims are made; proposal.pdf deletion and Group9.pdf remain user-owned and unstaged. These scope rulings do not weaken handled-failure logging or measured controls.

Final verdict: this diagnostic milestone is verified and complete. Execution succeeded; all three learning gates failed. Full pilot remains pending. Keep the local branch and all review records; no push/merge/deletion. The later controller-only fix is separate from the preserved research execution revision356a361; its source fingerprint stays recorded in the run/report.
