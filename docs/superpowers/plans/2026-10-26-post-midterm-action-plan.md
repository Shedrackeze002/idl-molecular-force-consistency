# Post-midterm action plan and progress checklist

Created: 26 October 2026. Status: active working tracker. Next task: **P07** (Idris's loader workstream). **P27** remains partially checked and pending live course requirements.

**Goal:** Complete an evidence-backed comparison of force consistency, molecular shape recovery and computational cost, then deliver the final course artifacts.

**Approach:** Verify the existing MACE model, evaluate unchanged recovery controls, and then investigate movement caps. Use the existing checkpoints before deciding whether further training or replication is justified.

**Basis:** [Reviewed next-stage brief](../../research/POST_MIDTERM_NEXT_STAGE_2026-10-26.md), [MACE evaluator contract](../../mace-evaluator-contract.md), and [team workflow](../../../TEAM.md).

This is the master action list. Each numerical work package gets a separate technical specification, implementation checklist, two independent plan reviews and a bounded launch record. This tracker does not replace those documents.

## How we will use this list

- Work through the dependency order below. The physical/cost workstream can proceed in parallel once its own plan is reviewed.
- Confirm course requirements in P27 alongside P04; do not wait for the research experiments to finish.
- Check an item only when its stated completion criterion is met. Add its completion date and a link to the evidence beside it.
- Keep unsuccessful results visible. A completed experiment can have a negative scientific outcome; executing it does not mean its hypothesis passed.
- Leave blocked work unchecked and record the blocker and next action in the activity log. A documented scope decision can defer conditional work.
- At each accepted result, append the evidence log, update this checklist, and publish the bounded code/result/status update after checking the candidate files.
- Preserve original checkpoints, failed trials and user changes. Numerical execution uses a fixed source revision and fresh output directories.
- Plan around local CPU and confirmed Colab capacity. Unconfirmed ORCHARD or cluster access must not become a dependency. Record compute estimates before any new run.

Owner names below retain the recorded workstreams; they do not claim newly confirmed availability, agreement or completed contributions. External messages require separate instructions from the coordinator.

## 0. Starting position

- [x] **P01 — Reconcile the current research state.** Completed 26 October. The midterm, accepted recovery reports and representation outcomes were checked. Evidence: [next-stage brief](../../research/POST_MIDTERM_NEXT_STAGE_2026-10-26.md).
- [x] **P02 — Review the next-stage priorities.** Completed 26 October. Independent scientific and engineering reviews support MACE first, then a separately specified movement-cap study. Evidence: the brief's review section; detailed dispositions retained locally.
- [x] **P03 — Verify the retained MACE packet files.** Completed 26 October. All 11 inventoried files pass the metadata-only verifier. This does not establish loading, runtime compatibility or prediction agreement. Evidence: [verification receipt](../../../reports/evidence/2026-10-26/mace-packet-metadata-verification.json). The authenticated packet manifest and payload are retained separately from the public repository.

## 1. Prepare the execution source and handover

Lead: Shedrack. Coordinate the loader/reproduction work with Idris and evaluator integration with Olufunke.

- [x] **P04 — Establish the clean execution checkout.** Completed 26 October. Isolated branch at research snapshot `3fa5969`; 13 source files, six historical scorer bindings and 11 packet members authenticated. Environment metadata/source checks pass; synthetic baseline: 140 passed, one archive-dependent skip and 19 subtests passed. Original worktrees preserved. Evidence: [source readiness](../../../reports/evidence/2026-10-26/post-midterm-source-readiness.json).
- [x] **P05 — Confirm handover access and current ownership.** Completed 26 October under the coordinator's instruction that GitHub presence confirms receipt. Remote handover and contract match retained bytes. Olufunke's integration role is explicitly confirmed. No published loader was visible; private implementation and private model/data access remain unverified. Evidence: [GitHub handover check](../../../reports/evidence/2026-10-26/github-handover-verification.json). Preserve Idris's loader ownership and avoid duplicate implementation.

## 2. Admit MACE and verify its predictions

Lead: Idris. Shedrack supports integration and review. P06 requires P04; P07 requires P05–P06. P08 requires the implementation checks in P07.

- [x] **P06 — Freeze and review the loader/parity plan.** Completed 26 October. The [specification](../../research/MACE_LOADER_PARITY_SPEC_2026-10-26.md) and [implementation plan](2026-10-26-mace-loader-parity.md) fix identities, all 974 validation records, mapping, units, absolute tolerances, failure handling and resource ceilings. Independent scientific and engineering reviews approve; production prevalidation order is clarified. Evidence: [review acceptance](../../../reports/evidence/2026-10-26/mace-loader-parity-plan-acceptance.json). No implementation or inference has run.
- [ ] **P07 — Implement and verify the loader.** Done when the checked model bytes are bound to the constructed calculator, an admission receipt is saved, and meaningful refusal/bridge tests pass. Wrong identities, malformed outputs and operational failures must remain visible. No hidden retry or checkpoint substitution.
- [ ] **P08 — Run the bounded prediction check.** Done when the admitted calculator is compared with saved validation predictions using the reviewed record set and exact ID joins. Require all 974 records for parity acceptance; retain a budget-limited attempt as incomplete. Save errors, runtime details and complete or incomplete status.
- [ ] **P09 — Independently audit and publish the parity outcome.** Done when prediction reductions and identities are independently checked and the result is published with its limits. Record Idris's independent reproduction evidence separately from the coordinator's earlier rescore. Failed parity blocks real MACE recovery until resolved through a recorded plan.

## 3. Evaluate MACE with the unchanged recovery procedure

Leads: Olufunke for runner/report integration; Idris for admitted MACE inference. P10 follows accepted parity in P09; P11–P13 proceed in order.

- [ ] **P10 — Freeze and review the unchanged recovery protocol.** Done when the ten original branch starts, six target-minimum starts, classifier, no-relaxation control and artifact identities are fixed. Retain the 1,500-force-evaluation branch budget, 0.01 eV/Å maximum-component threshold and existing geometry rule. Both plan reviews must accept the exact protocol and budget.
- [ ] **P11 — Add and verify the MACE recovery driver.** Done when the driver uses the existing evaluator without E3x-specific admission assumptions, persists verified MACE identity, and passes synthetic, failure-handling and relevant regression checks. The shared calculator bridge alone does not complete this task.
- [ ] **P12 — Run the unchanged MACE recovery controls.** Done when all planned outcomes, or an honest incomplete/failure record, are saved in a fresh output directory. Retain raw classifications, force calls and timing; do not select another model or procedure from the results.
- [ ] **P13 — Audit the outcomes and record the next decision.** Done when saved results are independently reviewed and published, with a MACE/E3x comparison and explicit limits. Record whether to proceed to movement caps or resolve an operational issue. MACE success or failure alone does not identify the cause of E3x failure.

If MACE readiness blocks progress, record a reviewed cap-first scope decision before starting Stage 4. Do not mark P09–P13 complete merely to bypass them.

## 4. Test whether movement caps improve recovery

Lead: Olufunke for evaluator changes, with Shedrack for experimental controls and analysis. Requires P13 or the recorded cap-first fallback. P14–P17 proceed in order.

- [ ] **P14 — Freeze and review the movement-cap experiment.** Done when a small cap set, exact per-atom vector operation, query accounting, resource budget, outcomes and interpretation rule are fixed before new results. Include all five original E3x checkpoints and their original ten starts. Add MACE only after accepted P09 prediction parity and an accepted P13 unchanged-recovery control record, including on the cap-first fallback. Specify reference-start controls and their budget explicitly. Retain uncapped history, thresholds and classification rules.
- [ ] **P15 — Implement and verify the cap intervention.** Done when tests show that accepted atom movements obey the declared bound and the uncapped path retains its established behavior. Verify rejected-query accounting, numerical failures and uniform application across models. Complete relevant regression checks and implementation review before launch.
- [ ] **P16 — Execute the fixed cap comparison.** Done when every planned model/start/cap outcome, or honest failed/incomplete status, is saved. No outcome-driven extra cap, checkpoint, threshold or retry may enter the comparison.
- [ ] **P17 — Audit, publish and decide whether recovery is informative.** Done when paired connection-level outcomes, wrong stops, nonconvergence, connectivity, movements, force calls and timing are reviewed and published. The energy baseline must recover at least one branch and exceed no-relaxation recovery to pass adequacy. Record failure honestly; adequacy does not prove a consistency benefit. If needed, propose one separately reviewed standard-optimizer control rather than an open-ended search.

## 5. Complete physical-consistency and compute analysis

Lead: Aaron, with Idris for MACE hardware/reproduction details. This workstream can start after P04 under its own reviewed plan; it need not wait for P17.

- [ ] **P18 — Freeze and review the physical/cost analysis plan.** Done when checkpoint membership, force–energy metrics, loop construction, quadrature controls, timing inputs, hardware settings, warmup, synchronization and resource limits are fixed. Both reviews must accept the scope before new numerical work.
- [ ] **P19 — Measure physical consistency.** Done when force–energy disagreement and closed-loop work are calculated with numerical-integration controls, independently checked and saved. Distinguish quadrature error from nonconservative work and retain model-specific failures.
- [ ] **P20 — Measure comparable prediction timing.** Done when the admitted models are timed on the same hardware and fixed workloads, with compilation, warmup and steady prediction separated. Record unavailable comparisons explicitly; do not infer speed from training exposure or incompatible hardware.
- [ ] **P21 — Audit and publish the physical/cost results.** Done when the analysis, source identities, timing scope and uncertainty are reviewed and published. Keep existing training costs separate. Any matched-wall-time training comparison needs a new prospective plan and budget.

## 6. Fix the final scope and evaluate the held-out test

Lead: Shedrack, with all workstream owners. P22 follows P17 and P21. P23–P26 proceed in order.

- [ ] **P22 — Record the final scope decision.** Done when the team records which claim the evidence can support, whether further seeds or paracetamol are justified, and which extensions are deferred. Paracetamol also requires resolution of its preparation block. Add reviewed tasks for each selected extension and complete their execution and analysis before P23; otherwise record the extension as deferred. Additional training needs separate reviews and compute estimates; it is not automatic.
- [ ] **P23 — Freeze the final evaluation protocol.** Done when checkpoints, model selection, recovery procedure, starts, metrics, pooling, seed treatment, connection-level uncertainty and failure rules are written and reviewed. Preserve the original negative results and label reused-validation analyses post-hoc.
- [ ] **P24 — Record separate authorization for held-out test evaluation.** Done when authorization covers the exact frozen protocol, inputs and budget. Until then, the test remains closed. This checklist is not that authorization.
- [ ] **P25 — Execute the frozen test evaluation.** Done when the planned results or complete failure/incomplete record are saved without selecting new settings from test outcomes. Record protocol deviations; any required correction needs its own documented decision.
- [ ] **P26 — Audit final results and claim limits.** Done when independent checks reconcile predictions, recovery categories, connection-level summaries, uncertainty, costs and all reported claims. A negative or limited result remains a valid final outcome when supported by evidence.

## 7. Complete the course and reproducibility deliverables

Leads: all members for their report sections and contribution evidence; Shedrack coordinates integration. Writing and packaging can progress alongside earlier stages, using only completed evidence.

- [ ] **P27 — Confirm current course requirements and make the delivery schedule.** In progress, 26 October. Current public project deliverables/weights are checked; live final deadlines and rubric remain unconfirmed. The saved rubric link is inaccessible and browser access to Canvas is policy-restricted. Evidence: [course check status](../../team/COURSE_DELIVERY_STATUS_2026-10-26.md). Do not treat the saved proposal-era dates as current confirmation.
- [ ] **P28 — Finish the report and figures.** Done when methods, results, unsuccessful trials, limitations and contributions match accepted evidence; sources remain editable; the final PDF compiles and passes visual checks. Confirmed course requirements determine format and length.
- [ ] **P29 — Verify the reproducibility release and contributions.** Done when source/configuration/artifact identities, environment instructions and a clean-checkout reproduction check are recorded. Publish bounded results with verified access, preserve each author's commits, and reconcile the contribution ledger without inventing completion credit.
- [ ] **P30 — Finish the presentation and submission checks.** Done when the presentation follows confirmed instructions, report/video artifacts receive team review, and submission receipts are retained. Record peer-review questions and required responses when that stage opens. Artifact preparation and confirmed submission remain separate facts.

## Activity log

Append a row after each completed item, material blocker or scope decision. Keep evidence paths with the task as well, so a checked box remains understandable on its own.

| Date | Task | Status or decision | Evidence or next action |
|---|---|---|---|
| 26 October 2026 | P01–P03 | Starting-state review and packet metadata check complete | Reviewed next-stage brief; local source/review record. No new inference or recovery. |
| 26 October 2026 | Tracker | Draft created and independently reviewed | Next execution task: P04; confirm course requirements through P27 in parallel. Detailed work-package reviews remain separate. |
| 26 October 2026 | P04 | Complete | Isolated source and retained environment/scorer/packet identities verified; 140 synthetic/regression passes, one expected skip, 19 subtests. No model load. |
| 26 October 2026 | P05 | Complete under coordinator's receipt convention | GitHub handover confirmed; Olufunke integration confirmed. Idris private work and private packet access remain unverified. |
| 26 October 2026 | P06 | Complete | Detailed specification and implementation plan accepted by both technical reviews. Next P07 retains Idris's implementation ownership. |
| 26 October 2026 | P27 | Partial; live course checks pending | Public deliverables checked; rubric inaccessible and Canvas host policy prevents browser read. No deadline confirmation. |
