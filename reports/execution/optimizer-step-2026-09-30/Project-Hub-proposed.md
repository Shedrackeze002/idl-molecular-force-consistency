# IDL MLIP project: current state

Updated 30 September 2026. Proposal submitted. Active study: Landscape17 energy-force consistency and reference-shape recovery, aspirin primary with separately trained paracetamol replication. Earlier rMD17/NVE planning is historical.

## Current gate

Aspirin source/unit/geometry preparation, frozen connection splits and untrained physical checks passed. The energy-derived baseline and trainer are implemented; all185verification tests pass after the optimizer diagnostic implementation. Both original learning attempts and the real-label full-batch control remain failed. The easy shared-backbone synthetic teacher passed; that does not establish real-label fitting.

The twice-reviewed coverage plan is now executed: original16 replay, diverse16 and nested diverse128 each completed 2,000 updates with identical fresh initial parameter/optimizer hashes and exhaustive fresh-process checkpoint verification. All three joint fitting and original-anchor gates failed because force fitting stayed above threshold. More coverage under these fixed settings did not resolve acceptance. Balanced outcomes, all settings, failed checks, source/run identities, metrics and limitations are preserved in the repository. Full pilot and scientific comparisons remain pending; no test evaluation or GPU trial occurred. Paracetamol remains independently blocked by one unmatched branch.

The twice-reviewed representation/loss-gradient diagnostic is also executed once: all eight frozen cases verified, with no parameter/optimizer update. Necessary symmetry floors are far below the unchanged force gates; local energy/force gradients strongly oppose each other at the three final fitting snapshots. This supports investigating loss balance but does not establish Adam dynamics or the cause of failed fitting. One larger finite-difference step failed; its smaller step passed under the preregistered criterion. Both results, every engineering failure, raw identities and limitations remain preserved. See FINDING-GRADIENT-001/002/003 and GRADIENT-REPORT-001. Independent implementation/evidence review is complete. It found two failure-routing/reporting issues, both repaired and verified by regressions and the full suite; raw research observations were unchanged and not repeated.

The twice-reviewed fixed loss-balance matrix is now executed once: replay, balanced and force-only each completed2,000updates/256,000training exposures with identical fresh states and exhaustive checkpoint verification. Replay matched all22fitting/anchor energy/force measurements exactly. Balancing reduced fitting force error35.9%; force-only43.0%; neither met the unchanged90%reduction criterion. Balanced energy also lost acceptance; force-only absolute energy gauge is unconstrained and its raw errors are preserved without recentering. Full pilot remains pending. All31coverage/38raw-calibration/1compact identities reconcile. See FINDING-BALANCE-001/002 and VERIFY-BALANCE-001. Independent final review complete: one replay-tolerance guard gap repaired and author-verified with152full-suite tests; no remaining Critical/Important issues or minors. Saved research was not repeated. No further numerical trial authorized by this result.

The twice-reviewed frozen optimizer diagnostic is now executed once: all3saved2000-update states verified,36FDcells/72steprecords/18prescribedtemporaryprobes, no persistent training update. FourlargerFDsteps failed andcorrespondingsmallersteps passed under the preregisteredcriterion; all retained. SavedAdam steps slightlylowered fittingforceerror; balanced/force-only reference directions had negative infinitesimal forceslopes but increasederror atbothprescribedfinite distances. This is local nonlinear evidence, not convergence/capacity/optimizercausality or generalization. All100raw+2compact source identities and source parameter/optimizer states unchanged; fullpilotpending. See FINDING-OPTSTEP-001/002 and VERIFY-OPTSTEP-001. Independent final implementation/evidence review is pending.

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

CMU cluster, Colab and Kaggle remain available per Shedrack; GPU execution remains unverified. Branch codex/landscape17-baseline. Frozen optimizer probes completed without new training; original learning gates failed. Current step: independent finalreview, then separately save/twice-review a followup before any convergence/optimizer/capacity intervention. No automatic sweep, newseed, longerbudget, validation/test/fullpilot. Preserve all trials, including failedFDsteps. Repository evidence remains numerical authority.


Final verified implementation/evidence record: [loss-balance execution record](file:///Users/shedrackeze/Desktop/CMU/INTRO%20DEEP%20LRNG/Project/Codex/reports/loss-balance-execution-record.md). Review records and failed engineering checks retained in repository/workspace; Obsidian remains navigation.
