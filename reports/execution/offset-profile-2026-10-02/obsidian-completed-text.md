
## 2026-10-03 — Offset profiling trial completed and saved-only audited

Current state supersedes the earlier running/halfway notes. The once-only width 32, seed 0, CPU float64, 128-fitted-example trial completed 6,000 accepted updates and 6,001 accepted bias calibrations in 1,899.5848710830032 numerical seconds. The fresh-process endpoint audit exactly matched 144 role entries; both NumPy Adam/full centered-objective oracles passed at 1/3001. No retry or parameter reinitialization. Unchanged reviewed implementation: 656 tests passed before launch. Full training pilot remains pending.

The independent saved-only result review PASS is retained in the repository (including its discovered/corrected draft wording error). All 18,324 raw files match the unique exact ZIP archive member set and every member hash/CRC; 601 source pins, original 343 capacity and 247 spike files, code/runtime and protected earlier evidence are unchanged. No new predictions or training during closeout.

Compared with historical width 32 update 6000, fitting energy MSE is 9.205634486459346% lower and force MSE 3.156351982739647% lower. Both role/overall strict 10%-of-original-baseline learning gates still FAIL (fitting E 0.2900057685250047,F 0.45100597487761285; anchor E 0.11418055530775573,F 0.42369062245390743). The anchor raw energy improved 1.628393577300702%, while centered energy variance worsened 24.41971159078873%. All 16 anchors are fitted group 01 examples, not held out.

Balanced group endpoint trail: raw energy 4 lower/4 higher, centered variance 3 lower/5 higher, forces 7 lower/1 higher; group 44 raw energy+119.28131254959813%, group 78_2 F+8.146900877762131%. Initial fittingE reduction 11.54503000616968% is by construction, not learning. Sampled raw energy maxima decrease does not establish continuous stability or mechanism. One seed/rate/historical reference/changed optimization/all-fitted support limits claims.

Correction OFFSET-PAPER-GRID-CORRECTION-001: update 3001 is shared in both saved histories and INCLUDED in 62 matched postzero scores; update 0 is excluded. The canonical report/figure were already correct; earlier erroneous paper draft is preserved.

Repository authority: reports/offset-profile-paper-findings.md; reports/offset-profile-diagnostics.json/.md; reports/offset-profile-component-comparison.json; docs/reviews/2026-10-03-offset-profile-saved-results.md; reports/execution/offset-profile-2026-10-02/final-closeout.json; reports/PAPER_EVIDENCE_LOG.md IDs OFFSET-TRIAL-COMPLETION-001, OFFSET-SAVED-CLOSEOUT-001, OFFSET-CLOSEOUT-PRESERVATION-001, OFFSET-PAPER-GRID-CORRECTION-001, OFFSET-SAVED-RESULT-REVIEW-001, OFFSET-TRIAL-FINAL-001. Paper findings SHA25669077e1632d87613f1af42faf150ab87d5f7c2b91cf25ed3a702da936c4dc172. Numerical archive SHA25691375bfe0e628d6cb374e186cd64e01b23a8f1ade837df057e5e1234ab42c7fa.

Next candidate lever: unresolved force fitting. It needs its own plan and two independent plan reviews before implementation; no next-lever experiment or full pilot has been launched.
