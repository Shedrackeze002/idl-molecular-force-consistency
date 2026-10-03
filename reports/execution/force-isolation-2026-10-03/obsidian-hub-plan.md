# IDL MLIP project: current state

Updated 30 September 2026. Proposal submitted. Active study: Landscape17 energy-force consistency and reference-shape recovery, aspirin primary with separately trained paracetamol replication. Earlier rMD17/NVE planning is historical.

## Current gate

Aspirin source/unit/geometry preparation, frozen connection splits and untrained physical checks passed. The energy-derived baseline and trainer are implemented; the final full suite passed 246 tests in 242.34 seconds after report-review repairs. The preproduction suite passed 238 tests; all failed engineering trials remain preserved. Both original learning attempts and the real-label full-batch control remain failed. The easy shared-backbone synthetic teacher passed; that does not establish real-label fitting.

The twice-reviewed coverage plan is now executed: original16 replay, diverse16 and nested diverse128 each completed 2,000 updates with identical fresh initial parameter/optimizer hashes and exhaustive fresh-process checkpoint verification. All three joint fitting and original-anchor gates failed because force fitting stayed above threshold. More coverage under these fixed settings did not resolve acceptance. Balanced outcomes, all settings, failed checks, source/run identities, metrics and limitations are preserved in the repository. Full pilot and scientific comparisons remain pending; no test evaluation or GPU trial occurred. Paracetamol remains independently blocked by one unmatched branch.

The twice-reviewed representation/loss-gradient diagnostic is also executed once: all eight frozen cases verified, with no parameter/optimizer update. Necessary symmetry floors are far below the unchanged force gates; local energy/force gradients strongly oppose each other at the three final fitting snapshots. This supports investigating loss balance but does not establish Adam dynamics or the cause of failed fitting. One larger finite-difference step failed; its smaller step passed under the preregistered criterion. Both results, every engineering failure, raw identities and limitations remain preserved. See FINDING-GRADIENT-001/002/003 and GRADIENT-REPORT-001. Independent implementation/evidence review is complete. It found two failure-routing/reporting issues, both repaired and verified by regressions and the full suite; raw research observations were unchanged and not repeated.

The twice-reviewed fixed loss-balance matrix is now executed once: replay, balanced and force-only each completed2,000updates/256,000training exposures with identical fresh states and exhaustive checkpoint verification. Replay matched all22fitting/anchor energy/force measurements exactly. Balancing reduced fitting force error35.9%; force-only43.0%; neither met the unchanged90%reduction criterion. Balanced energy also lost acceptance; force-only absolute energy gauge is unconstrained and its raw errors are preserved without recentering. Full pilot remains pending. All31coverage/38raw-calibration/1compact identities reconcile. See FINDING-BALANCE-001/002 and VERIFY-BALANCE-001. Independent final review complete: one replay-tolerance guard gap repaired and author-verified with152full-suite tests; no remaining Critical/Important issues or minors. Saved research was not repeated. No further numerical trial authorized by this result.

The twice-reviewed frozen optimizer diagnostic is now executed once: all3saved2000-update states verified,36FDcells/72steprecords/18prescribedtemporaryprobes, no persistent training update. FourlargerFDsteps failed andcorrespondingsmallersteps passed under the preregisteredcriterion; all retained. SavedAdam steps slightlylowered fittingforceerror; balanced/force-only reference directions had negative infinitesimal forceslopes but increasederror atbothprescribedfinite distances. This is local nonlinear evidence, not convergence/capacity/optimizercausality or generalization. All100raw+2compact source identities and source parameter/optimizer states unchanged; fullpilotpending. See FINDING-OPTSTEP-001/002 and VERIFY-OPTSTEP-001. Independent technical review reconciled the saved numerical evidence and delivered one Important report-completeness issue. Author reproduced eight omission/corruption failures, fixed the schema guard and passed193full-suite tests; saved research was not repeated. Review scope and response-delivery limitation are recorded in the repository final review record.

## Authoritative repository records

- [Frozen optimizer diagnostic report](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/reports/optimizer-step-diagnostics.md)
- [Fixed loss-balance report](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/reports/loss-balance-diagnostics.md)
- [Frozen representation and gradient report](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/reports/representation-gradient-diagnostics.md)
- [Coverage diagnostics and exact evidence](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/reports/data-coverage-diagnostics.md)
- [Paper evidence log](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/reports/PAPER_EVIDENCE_LOG.md)
- [Baseline status and both failed checks](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/reports/baseline-pilot.md)
- [Data preparation status](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/manifests/landscape17-preparation-status.json)
- [Physics verification](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/reports/baseline-verification.md)
- [Force-learning diagnosis](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/reports/force-learning-diagnostics.md)
- [Run instructions](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/docs/run-baseline.md)
- [Reviewed implementation plan](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/docs/superpowers/plans/2026-09-29-data-baseline-implementation.md)
- [Reviewed learning-gate amendment](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/docs/superpowers/plans/2026-09-29-learning-gate-amendment.md)
- [Project instructions](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/AGENTS.md)
- [Submitted proposal](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/Group9.pdf)

## Standing workflow

Preserve review folders, packages, test logs and progress ledgers after completion. Shedrack asked to leave them in place for review tracking; remove them only on explicit request.

Two independent subagents review every implementation plan and material amendment before execution. Record every experiment/trial as we go in the append-only repository evidence log, whether it succeeds, fails, is blocked or abandoned. Before execution, save rationale, fixed settings, controls and acceptance rule; afterward append actual outcome, stop reason, artifact identities and limitations. Distinguish proposals from executed trials and preserve failures for a balanced paper trail. Append corrections rather than overwrite earlier observations. Obsidian keeps current state/navigation and links to evidence, while the repository holds authoritative numerical results.

[[Research Findings]] records the paper-facing finding categories.

## Compute and next step

CMU cluster, Colab and Kaggle remain available; GPU execution remains unverified. Local branch codex/landscape17-baseline and all review folders remain preserved.

The twice-reviewed fixed continuation completed exactly once at code a24c653, extending replay, balanced and force-only from their actual saved 2000-update Adam states to 6000 with unchanged fitting support/model/LR/weights. All three execution, source, oracle and checkpoint checks passed. Every original joint learning gate and the separate force-only force gate failed. Fitting force error remaining is approximately 87.7%, 45.3% and 42.4% of original error, against a below-10% criterion. Final-window force reductions of 2–3% do not establish convergence. Full pilot and scientific comparison remain pending; no held-out or physical-recovery claim follows from fully fitted anchors.

The same independent reviewer completed the whole-branch code and saved-result/figure reconciliation. Two Important reporting defects were repaired and author-verified; no further Critical/Important or actionable Minor finding. Full suite: 246 passed. No research rerun. Current publication statuses and a novelty claim remain unverified.

A selective 12-primary-paper reading foundation includes the supplied anonymous When Forces Disagree manuscript. Its direct/conservative fine-tuning and consensus/calibration limits are close prior work. Whole-connection reference-minimum recovery at matched accuracy/cost remains an untested possible distinction. All four previously missing article PDFs were supplied and verified; all 12 registered papers now have local checksum-verified PDFs. Publisher/preprint versions, targeted read scope and earlier failed-download attempts remain documented. See LIT-ACCESS-003 / LIT-LOCAL-003.

- [Fixed continuation outcomes](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/reports/convergence-budget-diagnostics.md)
- [Execution and review record](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/reports/convergence-budget-execution-record.md)
- [Primary-paper reading map and resolved access record](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/reports/literature/2026-09-30/READING_MAP.md)
- [Paper evidence log](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/reports/PAPER_EVIDENCE_LOG.md): FINDING-CONV-001/002, VERIFY-CONV-001 and LIT entries.
- [Preserved execution archive](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/reports/execution/convergence-budget-2026-09-30)

Next: a separately reviewed recovery/replacement amendment for the stopped width/rate diagnostic; the full pilot remains pending. No numerical retry, altered schedule or new intervention is automatic. Repository evidence holds exact numerical authority; this note provides current-state navigation. Archive identities and local completion are recorded there.

## Reviewed next diagnostic — proposed, unexecuted

Two independent scientific/engineering plan reviews and material re-reviews cleared the crossed fresh16/32features × fixedAdam.001/.0003 diagnostic,6000updates per cell. Same128fit/16fully-fittedanchor and immutable original joint gates; one seed, no held-out/generalization/capacity or pilot-clearance claim. Fresh16/.001 must replay all63historical observations. Global boundary cap11400s; resource/numerical feasibility remains unverified. No implementation/tests/probes/training executed for this plan. Native execution established; human concrete artifact review pending. Engineering review tightened matched versioned state/history commit receipts and honest parent/global deadline accounting; failed planning assumptions and original drafts preserved.

- [Design](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/docs/superpowers/specs/2026-09-30-width-rate-design.md)
- [Implementation plan](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/docs/superpowers/plans/2026-09-30-width-rate.md)
- [Both independent reviews and rulings](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/docs/reviews/2026-09-30-width-rate-plan-review.md)
- Repository evidence: WIDTHRATE-PLAN-001 / WIDTHRATE-PLAN-CHECK-001 / WIDTHRATE-PLAN-AMEND-001 / WIDTHRATE-PLAN-REVIEW-001.

## Active approved implementation — 1 October2026 Kigali

Human explicitly approved the concrete plan. Native implementation and320preproduction checks passed; production launched exactly once at18d250f on CPU/float64, frozen16/32width × .001/.0003rate and6000updates per cell. First full128-frame independent Adam oracle passed; matrix currently running. No final learning acceptance or full-pilot clearance. Fresh final reviewer found three saved-report verification/accounting gaps; author repair/regressions will follow without numerical rerun. Source records, failed implementation checks and all prescribed intermediate observations retained in repository/workspace. Earlier human-review-pending statements describe the planning stage and are superseded by this explicit approval/execution update.

## Current status — 1 October 2026, final width/rate record

The approved once-only matrix stopped after the first cell reached 6000 persisted updates (768000 exposures). All 63 historical common measurements matched exactly, but fresh checkpoint verification failed because JSON converts integer historical keys to strings and changes their sorted order. The other three cells are blocked. Both original fitting and fully fitted anchor joint learning gates fail; no width/rate comparison or full-pilot clearance.

Author repairs passed 327 tests in 212.53s, including a real tiny fresh-process metadata regression; failed production was never restarted or re-probed. Independent reviewer reconciled all 145 raw report artifact hashes and the four-panel figure. Known parent reference work is 144 frames; failed child work and duration remain unknown. Two Minor report wording limitations are preserved in the final review; terminal-window summaries are unavailable for the invalid trial.

Repository authority: `reports/PAPER_EVIDENCE_LOG.md` (WIDTHRATE-FAIL-001, WIDTHRATE-ENG-003, VERIFY-WIDTHRATE-FINAL-001, WIDTHRATE-REPORT-001), `reports/width-rate-diagnostics.json/md`, `reports/width-rate-execution-record.md`, `docs/reviews/2026-09-30-width-rate-final-review.md`. Original runs/review folders retained; durable archive `reports/execution/width-rate-2026-09-30`. Next step: separately reviewed recovery/replacement amendment, not an automatic retry. Earlier running/pending-review notes describe earlier stages and are superseded by this current status.

## Current status — checkpoint-copy verification and CLI access correction

Updated 2026-10-01T15:23:15.536036+00:00. This current-state section supersedes older running/planned/pending-navigation summaries.

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


### Capacity lever — implementation and review gates, 2 October 2026

The two independent plan reviews and amended followups passed. A separate implementation review found three material guard/accounting/closure gaps; all were reproduced, repaired and cleared in a saved followup. The focused checks passed64 tests. See `CAPACITY-ENGINEERING-002/003` and `CAPACITY-IMPLEMENTATION-REVIEW-GATE-001` in the repository evidence log for every failed engineering trial and repair. Full-suite details and launch identity are in `reports/execution/capacity-2026-10-02/preproduction-gate.json`.

Next is one fresh32-versus64 comparison: seed0, CPUfloat64, fixed learning rate .0003, unchanged loss weights/normalization/128 fitting structures and16 fitted anchors,6000 committed updates per width. Oracles check actual updates1/3001; independent child initializes its declared fresh template and restores the terminal checkpoint. Exclusive root and child attempt guards prevent retries. The width32/width64/global caps are3000/7200/10800seconds.12000 updates and1536000 fitting exposures are prospective complete-pair counts, not measured results at this note's publication.

Capacity production is queued after the verified local source checkpoint. No capacity training result is claimed in this note. All anchors are fitted;144 role predictions cover128 distinct structures. Width changes initialization shape and optimization geometry; one seed/rate/fitted-support diagnostic cannot establish pure expressivity, generalization, convergence or the main force-consistency hypothesis. Full pilot remains pending. Authoritative current numbers/status are the repository receipts and `reports/capacity-diagnostics.json/md` once produced; later notes will supersede this queued status.


### Capacity comparison started once — 2 October 2026

All585 regression tests passed in468.33seconds; exact reviewed code/runtime/source identities and launcher are bound in the preproduction gate. Local numerical source checkpoint: `b45ff2a66923af14a51dce7190b3808810b2a478`. The once-only launcher is running; width32 fresh baseline, update1 independent oracle and early saved observations exist. Width64 is scheduled only if width32 completes validly. Endpoints and learning gates are pending; no capacity conclusion is available yet.

Live authority: `runs/aspirin-capacity-2026-10-02/` immutable histories/checkpoints/commit/phase receipts and `reports/execution/capacity-2026-10-02/production-stdout.txt`; evidence IDs `CAPACITY-FULL-SUITE-GATE-001`, `CAPACITY-PRE-001`, `CAPACITY-LAUNCH-001`. Every trial/review failure remains preserved. This note supersedes the prior queued status; final saved-result review/archive/readback will follow actual completion or failure. Full pilot remains pending.


### First capacity arm independently verified — 2 October 2026

Fresh width32 completed6000 updates/768000 fitting exposures within its cap. Independent child restored the final state and reproduced all144 role predictions with maximum absolute error0. A saved-only reviewer independently reconciled all63 observations/state snapshots, raw endpoint/group metrics, gates, actual oracles1/3001 and ancillary work;146 case files remained unchanged. Review: `docs/reviews/2026-10-02-capacity-width32-saved-review.md`; paper evidence `CAPACITY-WIDTH32-OUTCOME-001` and `CAPACITY-WIDTH32-SAVED-REVIEW-001`.

Both learning gates failed. Original-baseline fractions: fitting energy0.3194094334870974/force0.4657052724792342; fitted anchors energy0.11607064219031449/force0.4471150476714813. Saved energy maximum atupdate3700 was3.6020684618113443×fitting baseline and2.1998201468692145×anchor baseline; later saved recovery and another4900 spike are retained. Sparse observations establish neither continuous stability nor mechanism.

Width64 is now running fresh under the identical rate/support/update protocol. The matched width comparison, global cap/report closure and final archive/review remain pending; do not treat this first-arm result as a complete capacity conclusion. Full pilot remains pending. Repository receipts/logs remain numerical authority.



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



## Saved capacity spike diagnostic: reviewed plan, execution pending (2 October 2026)

User authorized diagnosing saved spike checkpoints before another training change. Both independent amended plan reviews PASS. Scope: width32 saved states3600/3700/3800/6000, width64 4900/5000/5100/6000. Retrospective maxima and immediate saved neighbors, all fitted structures; descriptive residual offset/centered-variance and scaled loss-gradient observations, no causal/continuous stability/generalization or actual Adam-direction claim.

No training, random parameter initialization or optimizer proposal is planned. Existing source343rawfiles and original capacity reports remain immutable. Restore parameter arrays, verify scores against history/reference, measure independent unweighted/weighted joint derivatives and atom-count-aware head-bias derivative, retain both fixed FDsteps. Instrument known returned work separately from finite-validated and unknown in-flight work. Total1800second cap includes60second reserve and guards/imports/compilation.

Focused83engineering checks passed25.97seconds; full regression and independent implementation review pending. No research diagnostic yet. Plan/design: /Users/shedrackeze/Desktop/CMU/INTRO DEEP LRNG/Project/Codex/docs/superpowers/plans/2026-10-02-capacity-spike.md and corresponding spec. Evidence entries SPIKE-PLAN-001/SPIKE-PLAN-REVIEW-AMEND-001/SPIKE-PLAN-GATE-001/SPIKE-ENGINEERING-001/002/SPIKE-FOCUSED-GATE-001 preserve repairs, unsuccessful trials and controls. Full training pilot remains pending.


## Saved capacity spike diagnostic completed (2 October 2026)

The previously pending diagnostic is now executed once. Both independent amended plan reviews and the final independent implementation review PASS; all615 regression tests passed467.29seconds (focused87passed34.18seconds). Eight restored states completed numerical verification in36.18193837499712seconds, inside the1800second total cap. No training updates, random parameter initializations or optimizer proposals occurred. Full training pilot remains pending.

Peak fitting energy squared-mean shares are86.16454542671367% (width32,u3700) and99.74418949031484% (width64,u5000). The squared-mean changes explain99.38447026696703% and99.93977749791236% of the respective pre-to-peak rawMSE increases. The wider peak mean error is+2.206811459762732eV; width32 is−0.34123452819711597eV. These are offset-dominated sampled excursions, not proof of their optimizer cause. Endpoint offsets are already tiny, so constant-offset correction alone does not resolve the failed endpoint joint gates.

Backbone energy/force gradient cosines at the peaks are+0.6557963389656083 and+0.7010270670585239, with weighted energy-to-force whole-norm ratios3.330270691702372 and4.4741114391072765. The peaks therefore do not show instantaneous opposing gradients, although surrounding states do. Scaled Euclidean slopes are not historical Adam directions. Atomic head-bias force derivative is0 in all8states; large energy gradients also occur in the backbone, so prediction offsets cannot be attributed to bias-only updates.

Thirty of32 individual FDsteps pass. Retained failures: width64u5100 energy h1e-4, width64u6000 force h1e-4; both smaller1e-5steps pass. All16case/objective checks satisfy the predeclared at-least-one-step rule, with no adapted tolerances or numerical retry. All128unique structures were fitted;16anchors are a subset. Selection was posthoc maxima, saved neighbors and endpoints, not a generalization or continuous-stability test.

Original343capacity rawfiles/347sourcepins and capacity evidence prefix remain intact. New diagnostic247rawfiles are archived with CRC/memberhashes verified. Known work:72predictioncalls/1152roleentries,256gradientcalls/4096structureexposures,512vectorFDcalls/8192exposures; unknown in-flight work0 at completion. Packaging failed once on an unresolved temporary symlink and was repaired with saved-only work; font-cache warnings and both failedlogs are retained. All unsuccessful engineering and review attempts remain documented.

Numerical authority: /Users/shedrackeze/Desktop/CMU/INTRO DEEP LRNG/Project/Codex/reports/capacity-spike-diagnostics.json. Paper synthesis: /Users/shedrackeze/Desktop/CMU/INTRO DEEP LRNG/Project/Codex/reports/capacity-spike-paper-findings.md. Evidence: /Users/shedrackeze/Desktop/CMU/INTRO DEEP LRNG/Project/Codex/reports/PAPER_EVIDENCE_LOG.md (SPIKE entries); exact source/launch/review/archive identities under reports/execution/spike-diagnostic-2026-10-02. Obsidian is navigation; repository evidence is numerical authority.

Next candidate remains unexecuted: design a controlled offset-handling comparison against the unchanged width32 reference, retain raw energy/force gates and residual mean/variance through training, and obtain two independent plan reviews. It may reduce excursions; it is not established as a solution to endpoint force fitting. No GPU or full-pilot job launched.

Independent saved-result reviewPASS: /Users/shedrackeze/Desktop/CMU/INTRO DEEP LRNG/Project/Codex/docs/reviews/2026-10-02-capacity-spike-saved-results.md. Residual/group/gradient arithmetic, source/state/history/reference/work/cap records and both archives reconciled independently; reports regenerate byte-for-byte. No model work repeated. FD acceptance arithmetic was independently audited, but central plus/minus objective vectors were not independently reevaluated. Diagnostic source checkpoint:b013acdae3f4566a8b41d7b417bf43ebe7583c5f. CanonicalJSON SHA-256:0e3a6f797effe68b63f9c3321141a9374062f69fc9f767b6853b0e70b2bdfc7c. Raw archive SHA-256:d814f9d4db37f17c9af0f125b15a4ac986cb6aa9aa633c5191cbda10fc11a47f.


## Offset profiling — implementation under review (3 October 2026)

Both final plan reviews passed before coding. The controlled6000-update width32 bias-profile intervention is still unexecuted.32focused engineering checks pass, including a private3update real-model fixture and saved-only state/projection audits; independent implementation review/fullsuite are next. Historical reference, original rawzero baselines and strict10% jointgate remain fixed. Mean-zero fitting bias is enforced, not evidence of learned calibration or force improvement. Research authority: reports/PAPER_EVIDENCE_LOG.md OFFSET-PLAN-GATE-001 and OFFSET-ENGINEERING-TASK2-TASK3-001; plan docs/superpowers/plans/2026-10-02-offset-profile.md. All failed checks and review folders retained. Full pilot pending.


## Offset profiling — once-only trial launched (3 October 2026)

Both final planreviews and independent implementation followup passed;41focused tests and656fullsuite tests passed before launch. The one reviewed6000-update width32 analytical fitting-bias profiling trial is now running with original rawzero baselines/rate/weights/supports. Numerical/learning outcomes remain pending. Source revision ca31c8b4bc67dc7ec067dd514dca3c80c713c75e; frozen gate reports/execution/offset-profile-2026-10-02/prelaunch-gate.json, raw root runs/aspirin-offset-profile-2026-10-02. Evidence OFFSET-TRIAL-PROSPECTIVE-001 and OFFSET-TRIAL-LAUNCH-001. No retry or tuning; preserve all results/failures. Full pilot pending.


## Offset profiling — numerical midpoint witnesses and fluctuation retained

Both independent production oracles at outgoing1/3001 passed; update3001 used actual nonzero incoming bodymoments. The combined1e-10 absolute/1e-8 relative rule was used, not absolute-only acceptance. The trial continues toward fixed6000updates and once-only fresh-child closure, so final execution/learning remain pending. Initial projectedzero fittingenergy MSE fraction0.8845496999383032 versus original unprojected baseline is an imposed11.545% reduction; forces unchanged. Fresh fittingenergy MSE rose at3000 then fell at3001 despite nearzero pooledmean: centered-error fluctuation remains, and calibration does not establish monotonic accuracy or continuous stability. Raw/evidence authority OFFSET-RUNNING-INITIAL-001 and OFFSET-RUNNING-HALFWAY-001 in reports/PAPER_EVIDENCE_LOG.md; running-halfway-observations.json preserves exact values. Paper methods draft reports/offset-profile-methods-note.md records derivation/controls/limits. No retry/tuning/fullpilot/held-out claim.


## 2026-10-03 — Offset profiling trial completed and saved-only audited

Current state supersedes the earlier running/halfway notes. The once-only width 32, seed 0, CPU float64, 128-fitted-example trial completed 6,000 accepted updates and 6,001 accepted bias calibrations in 1,899.5848710830032 numerical seconds. The fresh-process endpoint audit exactly matched 144 role entries; both NumPy Adam/full centered-objective oracles passed at 1/3001. No retry or parameter reinitialization. Unchanged reviewed implementation: 656 tests passed before launch. Full training pilot remains pending.

The independent saved-only result review PASS is retained in the repository (including its discovered/corrected draft wording error). All 18,324 raw files match the unique exact ZIP archive member set and every member hash/CRC; 601 source pins, original 343 capacity and 247 spike files, code/runtime and protected earlier evidence are unchanged. No new predictions or training during closeout.

Compared with historical width 32 update 6000, fitting energy MSE is 9.205634486459346% lower and force MSE 3.156351982739647% lower. Both role/overall strict 10%-of-original-baseline learning gates still FAIL (fitting E 0.2900057685250047,F 0.45100597487761285; anchor E 0.11418055530775573,F 0.42369062245390743). The anchor raw energy improved 1.628393577300702%, while centered energy variance worsened 24.41971159078873%. All 16 anchors are fitted group 01 examples, not held out.

Balanced group endpoint trail: raw energy 4 lower/4 higher, centered variance 3 lower/5 higher, forces 7 lower/1 higher; group 44 raw energy+119.28131254959813%, group 78_2 F+8.146900877762131%. Initial fittingE reduction 11.54503000616968% is by construction, not learning. Sampled raw energy maxima decrease does not establish continuous stability or mechanism. One seed/rate/historical reference/changed optimization/all-fitted support limits claims.

Correction OFFSET-PAPER-GRID-CORRECTION-001: update 3001 is shared in both saved histories and INCLUDED in 62 matched postzero scores; update 0 is excluded. The canonical report/figure were already correct; earlier erroneous paper draft is preserved.

Repository authority: reports/offset-profile-paper-findings.md; reports/offset-profile-diagnostics.json/.md; reports/offset-profile-component-comparison.json; docs/reviews/2026-10-03-offset-profile-saved-results.md; reports/execution/offset-profile-2026-10-02/final-closeout.json; reports/PAPER_EVIDENCE_LOG.md IDs OFFSET-TRIAL-COMPLETION-001, OFFSET-SAVED-CLOSEOUT-001, OFFSET-CLOSEOUT-PRESERVATION-001, OFFSET-PAPER-GRID-CORRECTION-001, OFFSET-SAVED-RESULT-REVIEW-001, OFFSET-TRIAL-FINAL-001. Paper findings SHA25669077e1632d87613f1af42faf150ab87d5f7c2b91cf25ed3a702da936c4dc172. Numerical archive SHA25691375bfe0e628d6cb374e186cd64e01b23a8f1ade837df057e5e1234ab42c7fa.

Next candidate lever: unresolved force fitting. It needs its own plan and two independent plan reviews before implementation; no next-lever experiment or full pilot has been launched.


### Offset trial final preservation clarification (2026-10-03)

After the independent numerical audit, one macOS .DS_Store folder-metadata file appeared. All 18,324 numerical files remain byte-for-byte unchanged; the current folder total is 18,325 including that incidental file. The original 18,324-member numerical ZIP/inventory stay fixed; metadata is preserved in place, as a separate copy and in incidental-metadata.zip. Independent narrow preservation followup PASS: docs/reviews/2026-10-03-offset-profile-metadata-followup.md (SHA2567045345ce336ac30e357236a51e2cfc8dc7d44700e25123329aa803245bf09cc). Repository evidence OFFSET-CLOSEOUT-METADATA-001 and final-closeout.json distinguish numerical data from this extra file. Results and failed learning gates unchanged.

Completion CLI sync first failed with Vault not found; the diagnostic was preserved and not mistaken for an actual note backup. Vault-first CLI recovery and real backups/exact readbacks succeeded for both notes. A computer-use inspection stalled on pending permissions and provided no app state; no permission was granted or security setting changed. That operational delay is separate from numerical elapsed time.

### Force-fitting next-lever plan ready (3 October 2026)

Current milestone: the force-only gradient ablation with fitting-energy gauge calibration has a saved implementation plan and two independent final amended-plan PASS reviews. This is PROPOSED and UNEXECUTED. Energy-gradient coefficient will change from 0.014584365670390719 to 0; force coefficient stays 1.9854156343296092. The width32 model, original source-zero state, 128 fitting structures/16 fitted group01 anchors, seed0, rate0.0003, microbatch16, 6000-update budget, scoring grid including3001 and fitting-energy bias calibration stay fixed. Energy labels still set the gauge and support scoring; this is not energy-free supervision.

Primary force gate requires both raw force-MSE fractions strictly below0.1 of original unprojected baselines. The unchanged four-metric joint gate is reported separately. Energy shape, centered variance and unfavorable group outcomes remain visible. One historical control/seed/molecule, fitted anchors and sparse scores limit interpretation; no held-out, causal mechanism or full-pilot claim.

Both reviewers checked the initial plan, then the amended final bytes. Amendment A requires nonzero readout/body force gradients/incoming body Adam moments and retained per-block fixture evidence, avoiding vacuous zero-head tests. Exact 18,963 source pins, old evidence prefix and runtime passed preservation checks. Prior control results remain unchanged. Reviewer failed lookup/exact-key-union attempts and LIT007 section3.3 locator correction are retained append-only. Computer-use app/screenshot access was successfully verified after the user granted it; this is separate from the earlier permission-wait failure and numerical time.

Next milestone is primary implementation of the new guarded contract/worker/child/supervisor/report, meaningful regression fixtures, a fresh full suite and independent implementation review, then a frozen launch gate before the once-only trial. Experiment configuration and training implementation remain pending. No new initialization, prediction, training, test-suite run or public run root exists at this planning milestone. Full training pilot remains pending.

Repository authority: docs/superpowers/plans/2026-10-03-force-isolation.md; docs/superpowers/specs/2026-10-03-force-isolation-design.md; docs/reviews/2026-10-03-force-isolation-scientific-plan-followup.md; docs/reviews/2026-10-03-force-isolation-engineering-plan-followup.md; reports/execution/force-isolation-2026-10-03/plan-gate.json; reports/PAPER_EVIDENCE_LOG.md IDs FORCE-ISOLATION-PLANNING-001, FORCE-ISOLATION-DESIGN-001, FORCE-ISOLATION-PLAN-REVIEW-001 and FORCE-ISOLATION-PLAN-FINAL-001. Plan gate SHA256 27a1f144fd82c18a35e2e2e71a605ec29c1e5a1f61460ed8c281eb199eadde9f.
