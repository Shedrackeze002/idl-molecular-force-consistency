# IDL MLIP project: current state

Updated 30 September 2026. Proposal submitted. Active study: Landscape17 energy-force consistency and reference-shape recovery, aspirin primary with separately trained paracetamol replication. Earlier rMD17/NVE planning is historical.

## Current gate

Aspirin source/unit/geometry preparation, frozen connection splits and untrained physical checks passed. The energy-derived baseline and trainer are implemented; the preproduction full suite passed238tests before the continuation implementation was executed. Report-review repairs have28focused checks passing; the final full suite remains pending while training runs. Both original learning attempts and the real-label full-batch control remain failed. The easy shared-backbone synthetic teacher passed; that does not establish real-label fitting.

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

CMU cluster, Colab and Kaggle remain available; GPU execution remains unverified. Local branch codex/landscape17-baseline. The fixed continuation plan and its fresh-process/persistence amendments were independently reviewed scientifically and technically before implementation. Exactly one run is now active at code a24c653, continuing replay, balanced and force-only from their actual saved2000-update Adam states to6000, with unchanged128fitting/16fully fitted anchor, weights/model/LR. No validation/test selection or new data.

Replay completed6000updates with oracle, checkpoint and source checks passed; original joint force gates still fail. Balanced is running; force-only follows only if execution remains valid. Whole-matrix interpretation is pending. Every partial/failure/blocked outcome is retained. No numerical retry or adaptive extension. Source and checkpoint identities remain in repository evidence, not this navigation note.

A selective12primary-paper reading foundation is saved, including the supplied anonymous When Forces Disagree ICLR2026 manuscript, published PIWSL, Landscape17, Bigi2025/2026 and task-evaluation context. Prior disagreement/consistency methods and consensus/calibration limits are acknowledged; matched-accuracy/cost whole-connection reference-minimum recovery remains an untested possible distinction. Four local PDF downloads are still unavailable via transport/automation; primary web full text is readable, and access links are in the reading map.

- [Fixed continuation plan](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/docs/superpowers/plans/2026-09-30-convergence-budget.md)
- [Primary-paper reading map](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/reports/literature/2026-09-30/READING_MAP.md)
- [Paper evidence log](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/reports/PAPER_EVIDENCE_LOG.md): CONV/LIT entries.

An independent whole-branch review found two reporting trust/completion gaps; author regressions are repaired with focused checks. Final numerical review, full suite, archive and report remain pending. Research execution code/settings are unchanged during the active run. Preserve every review workspace.
