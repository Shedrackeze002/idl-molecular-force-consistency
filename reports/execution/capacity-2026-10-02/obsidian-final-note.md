

## Capacity comparison completed and independently verified, 2 October 2026

Both fresh width32 and width64 arms completed 6,000 updates each: 12,000 verified persisted updates and 1,536,000 core fitting exposures. Execution and whole-pair saved-result/archive review passed. Both joint learning gates failed; the full training pilot remains pending. Production source: b45ff2a66923af14a51dce7190b3808810b2a478.

| Width | Fitting energy / baseline | Fitting force / baseline | Both-role joint gate |
|---|---:|---:|---|
|32|0.3194094334870974|0.4657052724792342|Failed|
|64|0.2118608476964685|0.5639486501605391|Failed|

At the chosen rate and budget, width64 has 33.7% lower fitting energy MSE and 21.1% higher force MSE. Anchor energy alone passes for width64, but remaining criteria fail. The largest saved fitting energy observations are 3.6020684618113443 times original baseline for width32 at3,700 and130.1420012818957 times baseline for width64 at5,000, followed by sampled recovery. This does not identify a mechanism or establish continuous stability/convergence. All anchors are fitted;144 role entries represent128 distinct structures. One seed/rate and changing parameter shapes/optimization geometry limit interpretation.

The numerical/primary-report/evidence closure took5,185.547022624989seconds within10,800seconds. Later saved-only archive/figures/review/notes work is separately recorded. All343archive members matched retained raw hashes and passed CRC.585tests passed before production, implementation review cleared and whole-pair saved-result review passed. No production retries or new numerical work occurred during closeout. Failed tests, implementation repairs, unsuccessful learning gates, operational cache warnings and sampled excursions remain in the balanced evidence trail.

Authoritative repository: /Users/shedrackeze/Desktop/CMU/INTRO DEEP LRNG/Project/Codex. See reports/capacity-findings-2026-10-02.md, reports/capacity-diagnostics.json, docs/reviews/2026-10-02-capacity-saved-result-review.md and reports/PAPER_EVIDENCE_LOG.md entries CAPACITY-INDEPENDENT-RESULT-REVIEW-001 and CAPACITY-FINDINGS-VISUAL-001. Obsidian is navigation; repository evidence is numerical authority.

Possible next work is a saved-checkpoint spike/residual/gradient diagnostic before selecting another training change. This is unexecuted and requires a new detailed plan plus two independent plan reviews.
