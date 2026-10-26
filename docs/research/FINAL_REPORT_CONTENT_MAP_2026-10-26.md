# Final-report content and evidence map

26 October 2026. P28 is in progress. This map prepares the editable content draft from accepted evidence; it is not a completed final report or confirmed submission format.

## Baseline and integration status

PR #13 merged into `post-midterm-mace-readiness` at commit `47b727b4efa4be91ada585be343e1ddb509b7116`. Its complete Git tree matches reviewed closeout commit `8842091`. PRs #12, #10, #9 and #8 remain open in the existing dependency chain; this merge has not yet brought the complete stack into `main`.

Use the [canonical midterm source](../../reports/midterm/group9_Midterm_IDL.tex) for the established research question, methods, terminology and references. Extend its representation discussion using the [accepted closeout](../../reports/REPRESENTATION_STUDY_CLOSEOUT_2026-10-26.md). Preserve the published midterm as a separate artifact.

## Draft sections

| Section | Content available now | Evidence | Remaining input |
|---|---|---|---|
| Abstract | Existing research question and measured prediction/recovery outcomes | Canonical midterm; accepted result reports below | Revise after the final study scope and remaining results are fixed |
| Introduction and prior work | Force–energy consistency hypothesis, graph-model context and checked reference list | [Midterm](../../reports/midterm/group9_Midterm_IDL.tex) | Targeted literature refresh if new mechanisms or methods are introduced |
| Data and preprocessing | Whole-connection splits, training-only normalization, 21-atom aspirin, seen target minima and historical model-free fixture disclosure | [Split manifest](../../manifests/landscape17-aspirin-split.json); [midterm claim bindings](../../reports/midterm/additional-checks.json) | Any approved replication scope; retain the existing paracetamol preparation block |
| Model and training methods | Energy-derived, direct and consistency formulations; shared E3x recipe, selection and matching rules | [Controlled study update](../../reports/E3X_PROJECT_UPDATE_2026-10-18.md); [completion audit](../../reports/evidence/2026-10-18/controlled/completion-audit.json) | Describe later methods only when implemented and verified |
| Predictive results | Completed controlled E3x comparison and the single-seed MACE reference | Controlled study update; [MACE Run A](../../reports/mace-runA-fit.md) | Independently admitted MACE adapter/parity; distinguish original rescore from the new adapter work |
| Original recovery and diagnosis | Zero original-start recovery, reference-start controls, disconnected endpoints and observed movement jumps | [Recovery](../../reports/E3X_VALIDATION_RECOVERY_2026-10-17.md); [diagnosis](../../reports/E3X_RECOVERY_DIAGNOSIS_2026-10-17.md); [paired traces](../../reports/E3X_PAIRED_TRACE_2026-10-18.md) | MACE unchanged controls and any separately reviewed movement-cap intervention |
| Representation comparison | BEST, LAST and late20 results, paired seeds, parameter controls, trajectories and scoped training timings | [Closeout report](../../reports/REPRESENTATION_STUDY_CLOSEOUT_2026-10-26.md); [acceptance](../../reports/representation-closeout-2026-10-26/acceptance.json); [tables and figures](../../reports/representation-closeout-2026-10-26/README.md) | Ready for draft integration; preserve fixed-recipe and validation-selection limits |
| Physical consistency and cost | Existing finite-record agreement results and recorded fitting/scoring times | Controlled study update; [timing definitions](../../reports/midterm/additional-checks.json) | Aaron's closed-loop controls and comparable prediction-time measurements |
| Final evaluation | Protocol-freeze requirements and the reserved test partition | [Action tracker, P22–P26](../superpowers/plans/2026-10-26-post-midterm-action-plan.md) | Reviewed final protocol, separate test authorization, execution and accepted results |
| Discussion and limitations | Recovery hypothesis remains unresolved; E64 representation ordering is descriptive; selection and compute differences are disclosed | Accepted reports above | Update claims after remaining results; keep negative and incomplete outcomes visible |
| Reproducibility and contributions | Existing source/configuration/checkpoint identities and verified authored work | [Contribution ledger](../team/CONTRIBUTIONS.md); closeout input/artifact inventories | P29 clean-checkout reproduction and each member's verified contribution evidence |

## Claim boundaries for the draft

- Label MACE prediction accuracy separately from its still-unmeasured recovery. Its architecture, exposure and training conditions differ from E3x.
- Keep the original 0/10 E3x recovery outcome. A revised procedure must have its own result and post-hoc validation label.
- Describe E64's lower observed representation-study errors under the tested E64-selected recipe. Both scalar and scalar/vector models retain equivariant gradient forces.
- Preserve both seeds and BEST/LAST/late20. Repeated checks and shared branches do not create additional independent samples.
- Distinguish recorded training times from synchronized inference timing, controlled efficiency, and total study cost.
- Keep pending analyses visibly pending. Assigned ownership is not completed contribution credit.

## Immediate drafting sequence

- [ ] Create a separate editable final-report project using the established course style and a visible draft notice.
- [ ] Draft the Introduction, Data and completed Methods from the accepted sources, preserving definitions and selection limits.
- [ ] Populate completed prediction, original recovery/diagnosis and representation sections with source-bound tables and figures.
- [ ] Add explicit pending sections for MACE controls, movement-cap results, physical/timing analyses and authorized final test evaluation.
- [ ] Start the P29 artifact index alongside the manuscript, linking each reported number to its configuration, source and accepted evidence.
- [ ] Obtain two independent content reviews and compile/render the draft. Resolve P27's live rubric before claiming final-format or page-limit compliance.

Shedrack coordinates the draft and source map. Idris retains MACE loading/reproduction, Olufunke retains recovery integration, and Aaron retains physical/cost analysis. Members retain authorship of their substantive implementation, analysis and report sections.

P28 remains unchecked until the final report and figures satisfy their full completion criteria. Current tracker progress remains seven of 31 tasks complete.
