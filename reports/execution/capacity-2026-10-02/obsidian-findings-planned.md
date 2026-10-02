# Research findings to carry into the paper

These are navigation notes, not a separate numerical evidence source. Read the exact values, source/run identities and limits in [the repository paper evidence log](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/reports/PAPER_EVIDENCE_LOG.md).

- DATA-002: chemical graph mapping and an oxygen permutation can resolve an endpoint that looks far apart under index-fixed alignment. This motivates explicit atom-permutation handling.
- DATA-005: unit provenance needs source/calculator evidence; XYZ headers alone are insufficient. Hessian conventions are a separate issue.
- DATA-006/006A: a paracetamol terminal remains outside the frozen endpoint threshold; it is reported and gated rather than dropped or accommodated by a looser threshold.
- DATA-007: two aspirin filename endpoint pairs differ from frozen geometric matching. Preserve named and resolved identities; this is a conditional archived-data observation, not a claim of corrupted data or new physical connectivity.
- DATA-008: whole-connection splitting does not imply unseen minima or independent observations. All aspirin reference minima are training-seen in the frozen first split.
- IMPLEMENTATION-001: untrained symmetry and derivative checks can establish implementation behavior before any learned-accuracy claim. A zero initialization needs deliberately nontrivial parameters for these checks.
- ENGINEERING-001/001A/002: energy loss can fall while force fitting remains weak. Compare absolute force error with zero-force predictions and preserve failed learning gates; random-initialization reductions can overstate learning. The one reviewed initialization adjustment did not resolve acceptance.

All are pre-comparison findings. None establishes that consistency training improves molecular shape recovery. New findings should be logged in the repository first, then linked here when useful.

## Diagnostic follow-up, 30 September 2026

- DIAGNOSTIC-002: exact linear readout fitting on two fixed backbones leaves substantial real-force error. This isolates a conditional feature limitation, not total network capacity.
- DIAGNOSTIC-003: an easy synthetic teacher sharing the student initial backbone is learned under the fixed pipeline; synthetic success does not prove DFT fit or generalization.
- DIAGNOSTIC-004: using all16 frames per update did not close the real-force gap. Preserve this failure alongside both earlier failed attempts; full pilot stays pending.
- IMPLEMENTATION-004/004A: the original run lacks intermediate full-subset histories; figure generation first failed on that missing file. Plot only verified endpoints, never interpolate unseen metrics or substitute minibatch loss.

[Diagnostic report and exact evidence](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/reports/force-learning-diagnostics.md) explains the controls, identities and claim limits.


## Coverage comparison, 30 September 2026

- IMPL-COVERAGE-001/002/003: label-blind frozen subset selection, full-batch-equivalent sequential accumulated gradients, fixed-cap controls and input-only exhaustive checkpoint probes were verified before production. Failed engineering/fixture checks are preserved alongside passing checks.
- COVERAGE-001/002/003 PRE/RESULT and COVERAGE-MATRIX-RESULT: original16 replay, diverse16 and nested diverse128 completed all fixed budgets but failed their joint force/energy acceptance gates because force error remained high. Energy fitting passed its individual threshold. This negative result is part of the paper evidence, not a discarded trial.
- Larger coverage did not close the force-fitting gap under this model/loss/budget. One seed, changed difficulty and 8x frame exposures per update for128 prevent attributing differences solely to sample count or making generalization/capacity claims. B fits only2original anchors; its14other anchors are still global training records.

[Coverage report, complete metrics, timing and checksums](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/reports/data-coverage-diagnostics.md). Repository evidence remains authoritative. Next is a separately saved, twice-reviewed architecture/loss-gradient plan; full pilot remains pending.


- IMPL-COVERAGE-004: independent review verified the measured artifacts and identified a handled-exception retention bug. Three failed regressions were repaired and independently cleared; all82tests pass. Numerical trials were preserved without rerunning training. Review records remain retained.


## 30 September 2026 — Frozen representation and loss gradients

FINDING-GRADIENT-001: symmetry-related necessary force floors are far below the failed gates; cutoff graphs are connected with complete two-hop reach, which is descriptive rather than capacity proof.
FINDING-GRADIENT-002: initial scaled energy-gradient dominance and strong final fitting gradient opposition; local evidence only, not Adam trajectory or causal attribution. The final C anchor has positive alignment, preserving mixed evidence.
FINDING-GRADIENT-003: every case/objective passes the predeclared directional criterion, with one failed larger step explicitly retained beside its passing smaller step. Parameter/optimizer/source identities unchanged.
GRADIENT-REPORT-001: [Authoritative report and exact numbers](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/reports/representation-gradient-diagnostics.md); [append-only evidence log](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/reports/PAPER_EVIDENCE_LOG.md).114engineering tests passed; final independent implementation/evidence review pending. No new training/weights/model/labels or full pilot. Next proposed investigation: reviewed loss/optimization control.



## Frozen diagnostic completion — final review and balanced repairs

IMPL-GRADIENT-004/005: independent review completed after a usage-limit interruption. It supported all saved research observations and found two Important failure-handling/reporting bugs. Both were reproduced with failed engineering fixtures and repaired; all 117 tests pass. The reviewer did not re-review repairs; author regressions/full suite verify them. No remaining Critical/Important findings or minors. Raw diagnostic execution/figures remain unchanged, with no research rerun or learning acceptance. Refreshed report hashes and every failed check are in the append-only repository log. Next: a separately planned/twice-reviewed loss/optimization control.


- FINDING-BALANCE-001/002: Fixed balancing improves force error while sacrificing energy acceptance; force-only improves further but still fails the fixed force gate. Removing the energy term alone did not resolve fitting at this budget. One seed, same model/data/update budget, fully fitted anchor; no capacity or optimizer-causal claim. Force-only absolute energy gauge remains unconstrained, with raw energy retained and no fitted offset. See [fixed loss-balance report](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/reports/loss-balance-diagnostics.md) and repository BALANCE evidence records.
- VERIFY-BALANCE-001: Exact22point replay, fresh2,000-update states/checkpoints, allsource/calibration hashes reconcile; auxiliary verifier failure and report-layout regression retained and corrected, without research rerun. Final independent review complete; its one replay-tolerance guard gap is repaired and author-verified. All152full-suite tests pass; no production rerun.


Final verified implementation/evidence record: [loss-balance execution record](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/reports/loss-balance-execution-record.md). Review records and failed engineering checks retained in repository/workspace; Obsidian remains navigation.


## Frozen optimizer observations — 30 September2026

FINDING-OPTSTEP-001/002: one exact3case saved-checkpoint diagnosis, local savedAdam versus two fixed equal-norm reference directions; allfinitepoints and bothFDsteps retained, including fourlarger-step failures. Tiny actualAdam fittingforce reductions and local-slope/finite-change reversals are evidence for finite-step caution, not a causal optimizer or capacity conclusion. No persistent update or learninggatechange; fullyfittedanchor. Numerical authority and identities: [optimizer-step report](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/reports/optimizer-step-diagnostics.md) and repository evidence log OPTSTEP entries.185fullverificationtests passed preproduction; independenttechnicalreview reconciled numericalevidence and found one reportingcompleteness issue. Author repair passed193fulltests; saved research unchanged/unrepeated. See repository finalreviewrecord for exactscope and response-delivery limitation.


- LIT-FINDINGS-001 / LIT-DISAGREE-002: twelve selective primary-paper cards, explicit versions/access/limitations and opposing evidence. Supplied manuscript uses direct plus conservative-pathway fine-tuning and warns about consensus failures/uncalibrated disagreement. PIWSL uses Taylor/spatial weak labels. No invention-of-consistency/disagreement or verified novelty claim.
- CONV-PLAN-001 / CONV-ENG-001/002/003 / CONV-REVIEW-001: twice-reviewed fixed-budget continuation, failed engineering attempts retained; two reporting gaps found independently and repaired with focused regressions. Final full suite/numerical review pending.
- FINDING-CONV-REPLAY-001: replay completed its fixed extension; original force gates remain failed. Balanced/force-only comparison pending. Exact counters/metrics/hashes are in the repository log.


- FINDING-CONV-001/002 / VERIFY-CONV-001: completed three fixed saved-Adam continuations to 6000 updates; all execution checks pass but every original learning gate fails. Extra budget improves fitted force error, with terminal force changes still 2–3%; no convergence/capacity/optimizer-cause/generalization or reference-recovery conclusion. Fully fitted anchors remain distinct from held-out evidence.
- VERIFY-CONV-001: 246 final checks pass; one fresh independent whole-branch reviewer reconciled all saved numerical records and both figures, with no new Critical/Important or actionable Minor issue. Earlier two Important report guards repaired by author regressions; numerical production not repeated. Exact failures, identities and limits remain in the repository.
- LIT-FINDINGS-001 / LIT-DISAGREE-002: twelve selective primary-paper cards and four remaining PDF access links saved. Close disagreement/consistency prior work acknowledged; consensus failures prevent agreement-as-correctness claims. Proposed whole-connection recovery distinction is untested, and publication status is not inferred.


- LIT-ACCESS-003: user-supplied Landscape17, Zhao transition-state benchmark, GDML and Forces Are Not Enough PDFs verified and copied byte-for-byte. All 12 registered papers now have local PDFs; previous failed attempts and exact version history preserved. Landscape17 bibliography corrected to its 2026 publisher edition. Article availability does not imply all separate supplements or exhaustive reproduction.
- LIT-LOCAL-003: local method reading reinforces conservation-versus-reference correctness, task/correction-dependent direct-force performance, and separate failure/stability reporting alongside conditional quality metrics. These are literature observations and future evaluation considerations, not new project training results or adopted thresholds.

- WIDTHRATE-PLAN-001 / WIDTHRATE-PLAN-CHECK-001 / WIDTHRATE-PLAN-AMEND-001 / WIDTHRATE-PLAN-REVIEW-001: proposed fresh width/rate sensitivity matrix, not measured success. Two independent reviews/material re-reviews cleared plan controls; transaction persistence, runtime-only deadlines and hidden parent work explicitly specified. Failed historical-schedule assumption plus mistaken diagnosis/correction retained. No model/probe/training/test execution for this plan; same original gates, fitted anchors, one-seed and timing/decay limits. Human artifact review pending before native implementation. [Plan and review record](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/docs/reviews/2026-09-30-width-rate-plan-review.md); repository numerical evidence remains authoritative.

- WIDTHRATE-IMPL-START-001 / WIDTHRATE-ENG-001/002 / VERIFY-WIDTHRATE-PREPROD-001: approved native implementation, all failures retained;320full checks passed before one fixed production matrix at18d250f. Numerical matrix currently running, final outcomes pending. WIDTHRATE-REVIEW-RUNNING-001: independent static review caught global elapsed consistency, interrupted-preflight work accounting and non-replay initial-baseline report gaps. These are report safeguards; numerical controls/run remain unchanged, no adaptive retry. All precise results/identities belong in repository evidence.

- WIDTHRATE-FAIL-001: once-only first cell completed 6000 persisted updates/768000 exposures and exactly replayed 63 historical common measurements, then failed fresh checkpoint metadata verification. Three cells blocked; raw failed checkpoint/history unchanged, no restore or numerical retry. The integer-to-string sorted-key mechanism explains rejection without establishing successful restoration.
- WIDTHRATE-ENG-003 / VERIFY-WIDTHRATE-FINAL-001: all inadequate/failed engineering trials retained. Author metadata and failure-accounting repairs, plus three report provenance guards, passed 327 final tests in 212.53s; this does not retrospectively validate production.
- WIDTHRATE-REPORT-001: independent saved-only reconciliation of 145 artifacts and figure. Fitting energy/force original fractions 0.3220330430785254 / 0.45318885284127763; fully fitted anchors 0.10905757620503544 / 0.42746961258301264. Both strict joint gates fail. No factorial contrast, capacity/optimizer-cause finding, generalization result, terminal-window summary or pilot clearance. Parent reference144 known; failed probe work/duration unknown. Repository is numerical authority; see final execution/review records for all limits and deferred Minor wording notes.

## Current status — checkpoint-copy verification and CLI access correction

Updated 2026-10-01T15:23:15.554988+00:00. This current-state section supersedes older running/planned/pending-navigation summaries.

The separately reviewed checkpoint-copy verification passed once: the saved6000-update parameter/Adam state reproduced144prediction frames exactly, maximum difference0, with zero training updates/optimizer proposals/warmup. Final388tests passed253.19s; independent saved-result review found no material inconsistency. All145original raw files remain unchanged. Original width/rate matrix remains failed, both joint learning gates failed, three cells blocked, contrasts unavailable and full pilot pending. A successful copy does not retroactively validate the original trial or establish DFT accuracy/generalization/physical shape recovery.

Repository authority: reports/PAPER_EVIDENCE_LOG.md (WIDTHRATE-RECOVERY-FINDING-001, WIDTHRATE-RECOVERY-REVIEW-002, WIDTHRATE-RECOVERY-ARCHIVE-003), reports/width-rate-recovery-verification.json/md, reports/width-rate-recovery-execution-record.md, docs/reviews/2026-10-01-width-rate-recovery-final-review.md; verified preservation archive reports/execution/width-rate-recovery-2026-10-01. All failed/incomplete engineering checks and original/review folders retained. Next numerical milestone requires a separately prospective width/rate matrix plan and two independent reviews before execution.

CLI access correction: the executable and app were available. The restricted Codex sandbox could not detect Obsidian and returned a misleading application-not-found message; permitted host execution successfully read these notes. Earlier CLI-unavailable/pending-navigation statements are superseded by this verified synchronization. No installation, app settings or CLI registration changed. Use the authorized host-execution path for future Obsidian CLI operations; repository remains numerical authority.

## Current status — fresh replacement width/rate diagnostic

Updated 2026-10-01T19:27:54.303983+00:00. This section supersedes earlier planned/running/pending-navigation summaries.

Replacement matrix status: completed; saved execution/interpretation valid: True. Full training pilot remains pending.

Verified committed updates: 24000; frame exposures: 3072000; counts complete: True. Four-cell contrasts available: True.

width16-lr1e-3: completed; persisted update 6000; fitting joint gate False; fully fitted-anchor joint gate False; failure/stop fixed_budget.

width16-lr3e-4: completed; persisted update 6000; fitting joint gate False; fully fitted-anchor joint gate False; failure/stop fixed_budget.

width32-lr1e-3: completed; persisted update 6000; fitting joint gate False; fully fitted-anchor joint gate False; failure/stop fixed_budget.

width32-lr3e-4: completed; persisted update 6000; fitting joint gate False; fully fitted-anchor joint gate False; failure/stop fixed_budget.

Original failed matrix and zero-update checkpoint-copy recovery remain separate immutable ancestry. Original unreceipted child effort remains unknown. One seed, fully fitted anchors, fixed-rate endpoints and sequential timing do not establish generalization, capacity causality, late-decay performance or efficiency rankings.

Final author repair suite passed476tests335.52s. Independent two-review plan gate cleared; five final source-review Important findings repaired and tested. Failed/blocked/abandoned engineering records and all model outcomes are preserved.

Repository authority: reports/PAPER_EVIDENCE_LOG.md (WIDTHRATE-REPLACEMENT-OUTCOME-INDEX-001 and each structured OUTCOME cell), reports/execution/width-rate-replacement-2026-10-01/corrected-report/width-rate-replacement-diagnostics.json/md (current; initial relative-path report preserved; REPORT-CORRECTION-001 links the identical raw/arm records), reports/width-rate-replacement-execution-record.md, docs/reviews/2026-10-01-width-rate-replacement-final-review.md, and reports/execution/width-rate-replacement-2026-10-01. The final artifact checker records scoped completeness after navigation and commit closure; this note is navigation, not numerical authority.


## Current status — late learning-rate lever, preproduction

The next controlled experiment is authorized. Two independent amended plan reviews passed. Fresh implementation review found four Important reporting/accounting/lineage gaps, preserved and repaired;45focused checks passed. Final full regression/follow-up gates remain pending at this update. No new production training yet.

Frozen experiment: two4000-update continuations from identical verified width32/.001 update6000 parameters and Adam moments, constant.001 versus late.0003 through8000 then.0001. Same128 fitted/16 fully fitted anchors, seed0CPUfloat64, balanced loss and original joint acceptance criteria. No test evaluation, GPU trial or full pilot.

Saved-only residual analysis found endpoint energy MSE dominated by centered residual variance; common squared mean error is tiny. This does not explain the historical spike or amend the schedule. Repository authority: PAPER_EVIDENCE_LOG.md LATE-RATE-PLAN/REVIEW/SAVED-OBSERVATIONS/ENGINEERING entries, docs/superpowers/plans/2026-10-01-late-rate.md, reports/execution/late-rate-2026-10-01. Obsidian is navigation only.



## Current status — late learning-rate diagnostic outcome

This section supersedes the preproduction note. Actual once-only experiment finished; saved execution validity: True. Full pilot remains pending.

constant: fitting energy/force fractions 0.17711402941595777 / 0.3246987513617705; original fitting-and-fitted-anchor joint gate False.

late_decay: fitting energy/force fractions 0.19119941859209563 / 0.3445582755614849; original fitting-and-fitted-anchor joint gate False.

New computed updates 8000; fitting exposures 1024000. Both schedules inherit the same update-6,000 state and Adam moments; fixed 4,000 extra updates each, 128 fitted frames and 16 fully fitted anchors, seed 0, CPU float64. No test evaluation, GPU run or main consistency/shape-recovery result. One bundled late-decay schedule and single chosen starting trajectory do not prove optimality, convergence or generalization.

Repository authority: reports/late-rate-diagnostics.json/md, PAPER_EVIDENCE_LOG.md LATE-RATE-* entries and reports/execution/late-rate-2026-10-01. All negative/engineering/review outcomes retained; Obsidian is navigation only.



### Paper observation — late-rate trajectory and endpoint tradeoff

Evidence IDs: LATE-RATE-OUTCOME-VERIFICATION-001 and LATE-RATE-TRANSIENT-001. Constant-rate fitting energy error spiked to 39.7647 times the original baseline at saved update 8,001 (raw 1.4918430382 eV²) and recovered by 8,100. Force error also rose. All eight groups show high energy error at that saved point; the constant schedule did not change at the boundary. The verified independent Adam/full-gradient boundary check passed, supporting correct execution without identifying the mechanism.

Late decay showed no comparable spike across 43 saved observations (energy fractions 0.1911994–0.2037762), yet its final fitting energy and force MSE were 7.95% and 6.12% higher than the constant arm. Both joint gates failed. Sparse observations cannot rule out behavior between scores; one checkpoint, one seed and fitted supports do not establish convergence or generalization.

Saved endpoint residual decomposition: a global energy offset accounts for only about 0.042% of constant fitting energy MSE and 0.000029% for late decay. Group-specific offsets and spike mechanisms remain unresolved; gates were never recentered. Negative learning results, failed engineering checks and accepted review repairs all remain in repository evidence.

Authoritative saved outcomes: reports/late-rate-diagnostics.json; descriptive notes: reports/execution/late-rate-2026-10-01/saved-outcome-notes.md/json and saved-spike-observations.json; figures: reports/figures/late-rate-curves.png and late-rate-curves-log.png. Exact report rebuild and numerical archive byte verification passed; original folders remain. Full pilot is pending. Next proposed lever is a separately reviewed controlled representation-capacity check, with GPU parity before a broader pilot. This proposal has not been executed.



## Current scope — capacity plan, 2 October 2026

The late learning-rate diagnostic is complete and independently reconciled; both joint fitting criteria failed. A new controlled capacity plan is proposed, with two independent scientific/engineering reviews pending. No new numerical experiment has started.

Prospective fresh 32-versus-64-feature models, seed 0, CPU float64, fixed rate 0.0003, same 128 fitted frames and 16 fully fitted anchors, same loss and zero-head initialization rule, 6,000 updates each. Width changes parameter shape and optimization geometry; this does not isolate pure expressivity. Exact10% joint raw energy-and-force gates remain. There is no inherited checkpoint state, adaptive schedule, retry or held-out result. Full pilot remains pending.

Repository authority: CAPACITY-PLAN-001 in reports/PAPER_EVIDENCE_LOG.md; configs/aspirin-capacity.json; docs/superpowers/specs/2026-10-02-capacity-design.md and docs/superpowers/plans/2026-10-02-capacity.md; reports/execution/capacity-2026-10-02. Configuration and numerical implementation cannot proceed until both reviews and material amendments clear.
