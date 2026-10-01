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
