# Research experiment index

Current through 3 October 2026. This is the shared scientific index. Complete original journals, raw runs and launch/audit packages are retained separately. Reports preserve failed, invalid and blocked outcomes. The package records existing observations and adds no new model result.

| Investigation | Rationale and fixed scope | Outcome and interpretation | Evidence |
|---|---|---|---|
| Source and units | Audit Landscape17 branches, endpoint geometry, duplicate classes and physical units before training. | Source-grounded unit conversion and frozen preparation implemented. Aspirin passes; paracetamol `01_2/plus` remains unmatched and blocked. | [Source audit](landscape17-source-audit.md), [preparation status](../manifests/landscape17-preparation-status.json) |
| Original learning check | Seed 0, 16 frames from connection 01, batch 4, Adam 0.001, 2,000 updates; strict 10% own-initial error gate. | Random readout force fraction 0.22444164; failed. No full pilot. | [Baseline report](baseline-pilot.md), [learning records](baseline-learning-checks.json) |
| Zero-readout amendment | Change readout initialization only, preserving subset, model, budget and gate. | Force fraction 0.74723604; failed. A smaller absolute error did not meet the stricter own-initial gate. | [Baseline report](baseline-pilot.md), [physics checks](baseline-physics-zero-readout.json) |
| Synthetic teacher and real-label full batch | Check an exactly representable teacher and sampling effects under frozen model controls. | Teacher passed at 300 updates; real-label full batch failed at 2,000. Teacher success is an engineering control. Fixed-body readout solutions are conditional diagnostics, not network capacity bounds. | [Force-learning report](force-learning-diagnostics.md) |
| Coverage | Original 16 replay, diverse 16 and nested diverse 128; unchanged loss/model/rate and 2,000-update budget. | Execution/reload verified; all applicable joint learning gates failed. More examples under these controls did not resolve force fitting. | [Coverage report](data-coverage-diagnostics.md) |
| Representation and gradients | Eight fixed diagnostic cases; feature geometry, floors, loss-gradient identities and finite differences. | Completed verified diagnosis. Larger finite-difference step failed; smaller step passed. Local gradient opposition does not establish optimizer dynamics or a causal bottleneck. | [Gradient report](representation-gradient-diagnostics.md) |
| Loss balance | Replay, fixed balanced weights and force-only control, each 2,000 updates. | All real-force acceptance gates failed. Error reductions and force-only behavior do not establish joint success. | [Loss-balance report](loss-balance-diagnostics.md) |
| Optimizer steps | Bounded local gradient/moment probes and finite-difference witnesses, without new training. | Diagnostic checks and failed numerical witnesses retained. Neither convergence nor capacity cause established. | [Optimizer report](optimizer-step-diagnostics.md) |
| Continuation budget | Three prespecified arms extended to 6,000 updates without resets. | All unchanged joint learning gates failed; additional budget did not establish convergence. | [Continuation report](convergence-budget-diagnostics.md) |
| First width/rate matrix | Prespecified width/rate comparison with immutable source and fresh-process verification. | First cell reached 6,000 updates, but checkpoint metadata verification failed; execution interpretation invalid. Remaining three cells blocked. | [Original width/rate report](width-rate-diagnostics.md) |
| Verification-only recovery | Protected copy; no training; exact prediction verification for the failed cell. | 144 role predictions reproduced with maximum discrepancy 0.0. This supports the bounded recovered record; learning gates still fail and the original failed attempt remains preserved. | [Recovery report](width-rate-recovery-verification.md) |
| Width/rate replacement | Four new bounded cells with fixed controls and complete adverse outcomes. | Earlier replacement report was invalid due to receipt/preflight mismatch. Corrected report is authoritative; no accepted joint gate. | [Corrected replacement report](execution/width-rate-replacement-2026-10-01/corrected-report/width-rate-replacement-diagnostics.md), [historical invalid report](width-rate-replacement-diagnostics.md) |
| Late learning rate | Two continuations from the same 6,000-update width 32 state. | Both arms failed the strict 10% joint gate. Saved-only verification establishes artifact agreement, not spike mechanism. | [Late-rate report](late-rate-diagnostics.md) |
| Capacity | Controlled wider models and long bounded fitting, with fixed physical conventions. | Learning acceptance remained unresolved. Numerical floors are conditional; these results are not held-out model comparisons. Failed and partial cases remain in the report. | [Capacity report](capacity-diagnostics.md), [findings](capacity-findings-2026-10-02.md) |
| Saved spike states | Analyze previously saved checkpoints and raw energy residual components without new training. | Sampled offset/variance and gradients are evidence at those states. Continuous stability and causal spike explanation remain unproved; failed checks are retained. | [Spike report](capacity-spike-diagnostics.md), [paper findings](capacity-spike-paper-findings.md) |
| Offset profiling | Width 32, seed 0, rate 0.0003, 128 fitting frames and 16 fitted anchors; analytical global fitting-energy bias; 6,000 updates. | Valid completion. Fitting force MSE 0.00635194479621658; force fraction 0.451005974878. All four strict fractions exceed 0.1, so joint gate failed. Anchor centered variance worsened 24.41971159078873%, despite pooled raw gains. | [Offset findings](offset-profile-paper-findings.md), [component comparison](offset-profile-component-comparison.json) |
| Force-gradient isolation | Proposed follow-up preserves energy gauge and fixed force weight 1.9854156343296092 while setting energy-gradient weight to zero. | Plan only. Implementation and execution pending; no measured result. | [Plan](../docs/research/plans/2026-10-03-force-isolation.md), [design](../docs/research/specs/2026-10-03-force-isolation-design.md) |
| Literature intake | Targeted primary reading to inform bounded diagnostics and the later literature review. | Sources, versions, read sections and claim boundaries retained. Failed downloads and access limitations do not count as read evidence. This is not a completed systematic review. | [Reading map](literature/2026-09-30/READING_MAP.md), [registry](literature/2026-09-30/registry.json) |

Unsuccessful checks, partial executions, blocked arms and superseded invalid reports remain evidence rather than deleted attempts. Raw directories and archive inventories are required for exact historical reproduction; obtain the separately retained packages and verify their identities. Historical source hashes in reports identify original artifacts. [The export manifest](research-export-manifest.json) records derived path normalization; [the reproduction note](../docs/REPRODUCTION.md) states package limits.

The main shape-recovery hypothesis, full pilot, direct/consistency model comparison, GPU execution and independently trained paracetamol replication remain unexecuted. Fitted anchors are not held-out evidence. No causal mechanism or generalization claim follows from the current fitting diagnostics.

## Update through 15 October 2026

The 3 October snapshot above is historical. Subsequent work includes completed, unsuccessful force-gradient isolation; an incomplete/invalid degree-two diagnostic; the completed MACE Run B diagnostic; the full-data MACE Run A reference; and the first shared-trainer energy/direct engineering fits. See [dated evidence](evidence/2026-10-15/README.md), [attempt index](evidence/2026-10-15/attempt-index.md), [MACE result](mace-runA-fit.md) and [negative E3x finding](e3x-first-engineering-fits.md).

MACE Run A final validation force MAE is 0.00525983286381519 eV/Å pooled and 0.005348956868537277 eV/Å equal-connection; RMSE is 0.02641038976153195 eV/Å. Both strict CPU/log checks pass. Two independent saved-array and integrity reviews accept the single-seed working-reference interpretation. Selected epoch 1987 remains secondary. No causal, test or recovery claim follows.

Both first E3x fits completed 1,000 updates with finite LAST/BEST reloads. Both BEST checkpoints remain at update 0, with zero-force MAE 0.06250438966560673 eV/Å. Final energy-derived/direct MAEs are 0.07046767329260104/0.06622239554560543 eV/Å. Energy MAE improves while force validation worsens; no causal mechanism, recipe selection, adequacy or matching is established. Independent E3x arithmetic checks saved group reductions and checkpoint controls, not a full-array rescore. Successful engineering execution and unsuccessful learning outcomes are both retained.

The failed tiny-subset gates remain diagnostic outcomes under adopted protocol v3. Scientific adequacy, all-arm costs/common budget, matched comparisons across seeds, relaxation and the main test remain pending. Historical source guards are unchanged; publication provenance differs from original training provenance. Complete journals and original packages remain retained separately. The compact export manifest records original/published SHA values and location-only transformations. No new training was performed for publication.


### RESEARCH-RESULTS-2026-10-18 — accepted comparison and negative recovery

Published the accepted full-data energy baseline, corrected BEST scoring, all nine controlled fits and predictions, eight consistency diagnostics, fifty branch recovery evaluations and 155 fixed force probes. C passes A1 at 34.587408 meV/Å. Only λ=0.1 matches direct under the original rule: mean force MAEs 38.024188 versus 38.186315 meV/Å, with lower finite-record disagreement, higher auxiliary energy error and about 1.985 times fit-plus-score time. Every model recovered 0/10, against a no-relaxation control of 0/10; A2 failed. No recovery benefit, main-hypothesis or test-performance claim follows.

All fifty endpoint stopping flags reproduce. Every model misses the force threshold at all eleven exact reference minima. All 23 converged dual-head endpoints are disconnected; seven are fully isolated with zero force. Both converged energy endpoints remain connected, while a nonconverged energy endpoint also separated. The actual solver caps scale rather than displacement and checks trial convergence first. A failed reference-start control cannot prove no nearby stationary point exists.

Earlier tiny-subset, learning, environment/import and BEST serialization-order failures remain retained; scoring repair changed no learned weights. The fixed nine-fit schedule itself completed without retry. Full original histories/checkpoints remain retained, while public compact exports preserve original/exported identities and numerical parity. See reports/E3X_PROJECT_UPDATE_2026-10-18.md and reports/evidence/2026-10-18/export-manifest.json.

### PAIRED-TRACE-2026-10-18 — accepted diagnostic, original A2 unchanged

One fixed package completed four historical replays and thirty reference-start controls, using 38,853 force attempts in 74.898 seconds locally. All four endpoints reproduce exactly with unchanged counts, stop flags and categories. The selected direct trace accepts a largest atom displacement of 436.721 Å; the separated energy trace accepts 268.047 Å. These post-hoc selected paths do not establish prevalence, physical dissociation, sole solver causality or consistency benefit.

Operational reference-start own matches are energy 4/6, direct seed 0 1/6, and 0/6 for each other model. Success does not establish an exact stationary point or stable minimum; failure does not establish nonexistence. All raw position/force arrays, query journals, returned solver records and compact results are exported; large derived per-query graph summaries remain retained and are regenerable. Both independent audits accept; all earlier evidence and original A2 failure remain unchanged. No additional run or intervention follows automatically.


### VERIFICATION-SCOPE-2026-10-18 — structural archive checks

The publication default suite passed 502 tests with 3 skips. Nine existing model-free fixtures parsed all aspirin archive records, including held-out test paths, for structural/start-set assertions. No trained checkpoint was scored on test data or selected from these checks. The scientific diagnostic remained restricted to its 21 bound validation/reference members. This distinguishes archive handling from trained-model test evaluation; future routine verification will explicitly restrict the archive fixtures. See reports/evidence/2026-10-18/verification-scope.json.


### MACE-EVALUATOR-PREPARATION-2026-10-18 — synthetic bridge accepted

Prepared shared ASE calculator bridge outside the frozen evaluator/source inventories. Two independent implementation reviews accepted the final source. Both existing environments pass the same 11 synthetic unittest cases; focused 113 tests and 19 subtests, plus 132 additional tests and one expected skip, pass with broad archive fixtures disabled. All 54 inventoried frozen source/scorer files unchanged. Retained initial missing-module RED and a reproduced/fixed dummy element X constructor failure; a reviewer command needed corrected PYTHONPATH. Existing regression code emits 44 NumPy deprecation warnings. No dependencies, solver or checkpoint bytes changed.

Metadata-only verification checked primary final EMA Run A file/hash and accepted score report. Zero checkpoint deserializations, trained-model inference calls or archive payload reads. Provenance tags do not verify weights. Idris retains checkpoint-loader/admission/reproduction work, Olufunke runner/reporter integration. Actual parity and unchanged procedure on 10 validation starts and 6 reference controls follow the midterm draft under their execution record. No recovery/portability/throughput/causal or test-generalization claim. See reports/MACE_EVALUATOR_PREPARATION_2026-10-18.md and reports/evidence/2026-10-18/mace-bridge-preparation.json.


### MIDSEMESTER-2026-10-18: handover and substantive working draft

Prepared an actionable MACE handover and separately retained 15.5 MB validation-only packet. The archive contains 11 regular relative members, with ten inventoried file hashes independently verified. It includes primary final EMA model bytes, validation inputs/memberships and saved predictions, environment lock, accepted source identities, energy gauge and a checksum verifier. No model was deserialized or queried. The public handover records the final archive digest; weights and data are excluded from this publication. Idris retains the loader/admission/reproduction contribution.

Created a five-page report in the unmodified required course style, with two original figures, force/energy and recovery results, explicit changes from the proposal, limitations, truthful roles and three next actions. All 1,203 curve points were independently reduced from five group sums/counts; the largest history/rescore BEST difference is 6.94e-18 eV/Å, within the stated 1e-12 extraction tolerance. A first exact-equality check and premature figure invocation failed and were corrected without scientific data changes.

Independent reviews requested a 32-source hash/pointer ledger, corrected A1 threshold wording and fixed clipped labels/initial plot range. All early high-error points are now shown. First layout versions exceeded five pages or split paragraphs around floats; concise prose and explicit placement resolved them without dropping scientific caveats. Earlier versions remain retained. Both final reviews approve the five-page working draft; all pages were rendered and visually checked, without overfull boxes or unresolved citations.

No new scientific experiment, solver change, test scoring or remote computation was performed. Team section review, contribution confirmation and live Canvas instructions remain pending. Writing began on 18 October at the coordinator's request, superseding the prior 20 October start. See reports/midsemester-2026-10-18/EVIDENCE_MAP.md and claim-sources.json.


### MIDSEMESTER-REVIEW-REVISION-2026-10-18-001 — report and handover corrections

Additional review required wording and handover changes. Two independent plan reviews approved the qualified revisions. The report now states that every model recovers 0/10 original branches, all six target classes are seen in training, and only two of four selected traces separate. All 23 converged dual-head endpoints are disconnected. The cutoff removes interfragment coupling, while internal fragment forces may remain; automatic exact-zero force applies to the seven fully isolated cases. Added the energy-only recipe selection, explicit supervised force heads, disagreement ranges, per-model stopping counts, bounded reference-control interpretation, redistributed responsibilities and specific future diagnostics. Paracetamol remains deferred pending aspirin feasibility. Scientific outcomes, the fixed matching rule and failed A2 are unchanged.

Saved-value arithmetic gives baseline trajectory population SD 2.235/2.981/3.097 meV/Å over 101 checks at updates 150,000–200,000. This is correlated training variation, not measurement uncertainty. Across nine later fits, final-minus-BEST gaps span 0.0814–7.3997 meV/Å; the matched mean gap is 0.16213. Definitions and input hashes are retained in selection-sensitivity.json. All 33 claim-source bindings verify, including a refreshed previously stale contribution-document binding.

Handover version 2 adds the validated macOS arm64/Python 3.12.13 scoring host, explicit fresh-environment commands, cross-OS failure handling, eight raw source-file mappings and a metadata-only verifier. The original lock-resolution host remains unknown. The accepted environment passes; a deliberately wrong expected source pin fails. The initial missing-helper check is retained. No fresh installation or model loading occurred. Version 1 remains preserved. Version 2 has 12 regular archive members, 11 inventoried files and unchanged model/data/prediction/lock bytes. Archive SHA-256: ff92c74b83d3103a2542f1564d93be3b153f5342faef2675f9845d9d99958119. Idris retains loader, admission and parity work. Two independent reviews precede coordinator approval; private delivery and integration acceptance are not claimed complete.

Initial revised PDF builds overflowed to six pages. Preserved those versions, shortened repeated prose and adjusted figure layout/markers to restore five pages without changing course style, fonts or margins. A verification command initially used a wrong snapshot subpath; correcting it restored the check. Historical lookups in the isolated checkout were absent and repeated at the research root. All 54 frozen scientific/evaluator/scorer files remain unchanged. Final artifact reviews and remote publication verification are recorded separately.

The working-draft label remains. The PRIVATE repository and open PR #10 → #9 → #8 require grader access and a specific reviewed revision before submission, alongside actual contribution confirmation and live Canvas checks. No training, model inference, scientific archive parsing, solver change, test scoring, remote allocation, Drive change, outbound message or merge.


### MIDSEMESTER-REVIEW-REVISION-2026-10-18-002 — revised draft accepted for team circulation

Both final independent reviews approve the corrected report and handover. Final five-page PDF SHA-256: 8e965fc0ae7b05fe60250132d80c2e8b529e20cc2d5ecef31057a9b1cbc44521; TeX SHA-256: 2e9a37def7cfe773d35586de6093014d0e0bf3840e5f8301898e55b9af2122b7. All five rendered pages were inspected. The original course style and plot arrays are unchanged; both terminal and editor compilers succeed without overfull boxes or unresolved references. All 33 claim-source hashes, 54 frozen sources, 12 packet archive members and 11 member hashes verify. Metadata-only environment PASS and wrong-pin rejection are retained. Editable version 2 report ZIP contains 14 verified source/figure/PDF members; version 1 is preserved.

The report is ready for team circulation, not submission. Staff access, specific final reviewed revision, member confirmations, live Canvas checks and final draft-label removal remain open. Updating existing PR #10 with these corrections; no merge or new scientific execution. Private review material and handover weights/data stay outside GitHub.


### MIDSEMESTER-RUBRIC-AUDIT-2026-10-18-001 — substantive requirements met; release checks open

Audited the unchanged five-page report against the saved course grading PDF's actual Midterm column, visually checking source columns and contribution notes. All substantive document criteria are covered: required style/length, accessible problem and objective, linked literature/background, inputs/outputs, data preparation, metrics/loss, implemented baseline, multiple initial iterations, own baseline plots, challenges/risks, three next actions, bibliography, basic roles and clickable GitHub link. Formal audit records each report location and distinguishes encouraged, mandatory and final-only criteria. No positive recovery or additional training is required merely to satisfy the saved midterm document rubric. Course style is byte-identical, five pages include references, and the PDF is unchanged.

Connected course announcement posted October 17 confirms October 23 as the midterm date. Exact Canvas instructions/timezone/time, actual contribution confirmation, final draft approval/label removal and staff access to the private reviewed repository revision remain open. Live sheet/template retrieval attempts failed; public landing page yielded no assignment details. Topic-specific guided-project instructions were not applied to Path 3. No grade guarantee, current-Canvas confirmation, model run, submission or permission/visibility change. See reports/MIDSEMESTER_RUBRIC_AUDIT_2026-10-18.md.


### FACULTY-FEEDBACK-2026-10-20-002 — reviewed assessment and literature/resource findings

Both independent scientific and operational reviews approve the saved feedback assessment. Existing E3x models already use equivariant graph message passing; current shared-backbone force-formulation results are not a symmetry/graph ablation. Proposed scalar versus tensor-feature energy-model comparison remains an option, with invariant-energy gradient forces correctly described as equivariant. A direct max-degree-zero toggle would violate the present vector-head contract. No new architecture or experiment is adopted.

Retain completed midterm work; consider one bounded architecture study afterward and a separate long-range feasibility direction. Missing interactions beyond7Å are not established as the cause of failed relaxation, and increasing a cutoff alone does not isolate explicit long-range physics. LES/eSEN abstracts and current model-domain documentation are literature leads, not reproduced findings. Matbench requested pages were unavailable/403; official repository used, no exact current ranking claimed. Molecular pretrained options require domain, label/gauge, charge/spin, overlap/license and resource checks; foundation-model comparisons do not isolate symmetry. Source commits/files are pinned in reports/evidence/2026-10-20/faculty-feedback-sources.json.

Official CMU ORCHARD pages confirm PI group application, conditional General one-GPU/12-hour queue, Bootstrap one-GPU/30-minute preemptible jobs and25GBhome quota. ACCESS-ID onboarding is not access approval; storage/API charges and pilot terms need current confirmation. Recommended next access step is PI sponsorship, professor fallback if needed, without making midterm contingent on provisioning. No application, message or job occurred. Reviewed memo: reports/FACULTY_FEEDBACK_ASSESSMENT_2026-10-20.md. The first raw-byte source comparison refused the already published configuration because its JSON keys were sorted. Parsed values match exactly; both original and published file hashes are now distinguished in the source receipt. No configuration or scientific artifact changed to resolve this check. Current report, all scientific source/models/results, teammate ownership and main-test boundary remain unchanged.


### COMPUTE-PLANNING-2026-10-20-001 — Colab-only remote GPU assumption confirmed

Coordinator explicitly instructs planning as though ORCHARD will not be available; Colab is the sole assumed remote GPU resource until ORCHARD access is granted. ORCHARD sponsorship discussions are separate and not on the delivery critical path. Do not rely on pending PSC/other cluster access or guaranteed GPU assignment. The coordinator immediately clarified that current compute-unit balance is not a study limit: up to $100 can buy 1,000 additional CU at his quoted rate, so useful experiments should not be deferred merely for a balance top-up. Track estimated/actual usage and plan replenishment; preserve reviewed numerical budgets and checkpoint recovery. Colab hardware availability, memory and session behavior remain relevant. The first uncommitted wording treated balance too restrictively and is preserved privately with this correction; no experiment was curtailed or run. This is a resource constraint, not a new architecture/experiment budget or launch approval. Local CPU work is not excluded. No code/model/report change, allocation, contact, account/Drive mutation or numerical run.


### REPRESENTATION-PLAN-2026-10-20-002 — amended specification, implementation tasks and scenario budget accepted

Both independent scientific and engineering re-reviews approve the amended plan. Proposed comparison fixes energy-derived forces and training recipe: E64 scalar/vector features, mechanically parameter-matched scalar SP, same-width scalar S64, two paired seeds and at most six fresh200000-update fits. Both model families retain equivariant force outputs. The scalar recipe inherits energy-reference tuning; capacity, seed and validation-reuse limitations are explicit. Historical CPU models are context, not controlled GPU-run substitutes. Primary equal-connection MAE, BEST/LAST and pooled squared-error RMSE definitions, signed paired deltas and descriptive claims are fixed. No new parameter initialization or fit ran.

Review corrections preserve original validation_minima_diagnostics (even if empty) with train/validation and manifest, while test numerical payload stays excluded; define copied-state prediction/optimizer/RNG probes and GPU tolerances without advancing live state; exclude future BEST from restored history; authenticate original bytes before named-tree checks; define p95seconds/update as100-updateblocktime/100; account for BEST,periodic,elapsed-time and final Drive writes plus repeated-session overhead and unknown-rate scenarios. Standalone inference benchmarking remains with Aaron. No six-hour notebook pause, permanent stale-owner receipt or exact-GPU-name restriction is planned. Original draft and both review rounds remain retained separately.

Five cost scenarios independently verify: six fits with2h allowance and25percentreserve at illustrative10CU/h cost175/325/625/925CU for2/4/8/12h-per-fit assumptions; at4h and5.3CU/h,172.25CU. Scenarios exclude unknown extra reconnect cost and are not measured GPU forecasts or live prices. The filled launch budget requires actual JAX/E3xGPU readiness, synchronized end-to-end timing, full validation/checkpoint costs and recorded rate; MACE GPU success is insufficient. Current balance is replenishable within the coordinator's stated1000CU purchasing envelope; ORCHARD remains optional.

Accepted documents: docs/research/REPRESENTATION_COMPARISON_SPEC_2026-10-20.md and REPRESENTATION_COMPARISON_PLAN_2026-10-20.md. Scenario/source receipts in reports/evidence/2026-10-20. All54 frozen source files and current report remain unchanged. Plan ready for staged implementation; implementation review, bounded remote timing and filled launch-record review precede scientific dispatch. No training, inference, main-test query, package install, Drive mutation, contact or GPU allocation occurred. Midterm release and teammate ownership remain separate.


### REPRESENTATION-CPU-REVISION-2026-10-20-001 — execution plan simplified after independent review

Execution review identifies that local CPU fits and full-state trainer resume already exist. Coordinator confirms overnight CPU availability. Revised specification/plan now reuse accepted E64 Cseed0/energyseed1 conditionally on1000-update reproduction and fit only scalar SP/S64seed0/1, at mostfourfits or twoifdeduplicated. Original settings/data/sample streams remain fixed, and the mean of the last20scheduledvalidationchecks190500–200000 is co-reported with BEST/LAST. CPU trainer timings4874.94s/4104.61s are historical;4108.55s was externalstageaccounting, and5–7hqueue is provisional, not measured scalar timing.

Drop active GPU readiness, per-checkpoint step probes, device-migration policy, checkpoint-count/reconnect formulas and CU scenario table; earlier drafts/scenarios preserved in history. Two amended scientific/engineering plan reviews APPROVE. They found an off-schedule wall-stop can contaminate BEST: retain one authenticated scheduled-checkpoint fallback, require state/BEST/control history on the500updategrid, reject off-grid wall-stop state and never repair it by merely filtering JSON. If no eligible state remains, report incomplete rather than silently startover. No new storage format/service or frozen-source edits.

Midterm caption collision corrected by placing both table captions above their tables as expected by the unchanged course style. A factual message-passing GNN description added; no numerical claim, plotarray, objective, role or result changed. Five-page PDF recompiled/rendered; the prior report remains retained. Professor clarification drafted for the coordinator, not sent. Private grader access, open PRstack integration, actualcontributions, liveCanvas and teamapproval/draftnotice remain pending. No optional prospective study sentence added without coordinator choice. Solver is a critical path for a separately reviewed procedure control, not an established sole cause.

No parameter initialization/count, model/array/checkpoint load, reproduction, scalar timing, new scientific fit, GPU allocation, Drive mutation, package install or outgoing contact occurred while amending the plan/report. All54frozen scientific/evaluator/scorer files unchanged. Next: implement the thinwrapper/config/physics controls, verify counts/reproduction/resume, then reviewed serial CPU launch.


### REPRESENTATION-CPU-IMPLEMENTATION-2026-10-20-001 — synthetic verification and reviewed wrapper

Implemented the reviewed CPU protocol and shape count, small segment/resume wrapper, independent endpoint analysis and four-cell local notebook outside frozen source. Two final independent implementation reviews approve verification. E64has43521parameters; mechanically selected scalarSPwidth88has43473 (0.1103percentgap), samewidthS64has25473. This is capacity evidence, not prediction quality.48synthetic/regressiontests passed before final metadata guard;15finalfocusedtests pass, with mainarchive unavailable. All54frozen scientific/evaluator/scorer files unchanged.

Retained regression failures and fixes: empty-run directory orchestration metadata; corrupt completion/checkpoint fallback; partial scoring in fresh attempts; wrong E64 reference seed/recipe mapping; decoded embeddedBEST after raw hash authentication; corrupt/nonfinite old timing recorded unknown. Cached metric/prediction checksums and reused scorer helper identities retained. No owner protocol, GPU list, six-hour pause, source rewrite or model retry.

Real1000update E64 reproduction and2000update hard-interruption/resume verification remain pending; historical reference reuse is conditional. No full scalar scientificfit, test/recovery evaluation, package install or paid allocation occurred. Next commit the execution identity, perform bounded CPU verification, review the filled launch receipts, then serial SP0,S640,S641,SP1. Midterm release/admin checks and teammate work remain separate.


### REPRESENTATION-CPU-PREFLIGHT-2026-10-20-001 — reproduction and deliberate interruption checks PASS

Fresh E64seed0 through1000updates matches the complete originalC0 validation events at0/500/1000 to1e-10absolute, including perconnection support. Original config/provenance/history/BEST/LAST hashes captured before and unchanged after verification. Static exact recipe/source/environment/seed and current prepareddata identities are pinned for both historical E64 references. This licenses conditional reuse, not a fresh200000update reproduction or a new predictive result.

Fresh SP88seed0 uninterrupted2000updates compared with a separate worker group sentSIGKILL after saved1000/before1500 and resumed to2000. Parameters, optimizer, RNG, authenticated decodedBEST/control and full scheduled history agree to1e-10, with exact nonfloating identities. A completed re-invocation skips without checkpoint change. Signal delivery and waited outerprocess are recorded; descendant exit was not independently observed. Verification states remain separate and will not seed full fits.

Counts, synthetic physics/configuration tests and actual resume/reproduction receipts support the filled CPU launch record. Four full scalar fits remain pending final launch review. No main-test/recovery evaluation, longrange/foundationmodel run, GPU allocation, CU cost or source rewrite. All failures/corrections remain retained; no numerical retuning or hidden trial replacement.


### REPRESENTATION-CPU-LAUNCH-2026-10-20-001 — serial science queue running

Actual SPseed0 progress observed at update8500of200000 on localCPU at2026-10-20T23:27:31.353783+02:00; completion is not claimed. Four declared fresh fits run serially SP0,S640,S641,SP1. Both reviewed preflight and launch conditions pass, existing E64seed0/1 reused after required reproduction and source/recipe/data/seed binding. Reference BEST/LAST rescored independently; late20trajectory metrics are available, not new training. Verification states are separate.

Execution revision79dba2745350e16aa11e6f4e39b5011229ec5fd0 stays pinned, while results publish through the separate clean checkout. Idle-sleep prevention runs with the job. No ColabGPU or CU, test scoring or recovery. Training progress is not selected-checkpoint quality, study completion or representation superiority. Each completed/failed/incomplete outcome will be published with its evidence and interpretation-review status.


### REPRESENTATION-CPU-OUTCOME-SP-0-397019eacc41 — CPU fit outcome, 2026-10-20T23:58:55.845286+02:00

SP-0: status completed. BEST/LAST/late20 equal-connection force MAE (eV/angstrom): 0.04104627288284203/0.05252223343762975/0.04627642265021941. Raw output/source/checkpoint identities retained; independent scientific result review pending. Fixed-recipe two-seed validation evidence only, with no test, recovery, causal symmetry or population-level superiority claim. Failures/incomplete work remain visible and are not replaced with hidden tuning/retries. See reports/evidence/2026-10-20/representation/SP-0/outcome.json.


### REPRESENTATION-CPU-PUBLICATION-RECOVERY-2026-10-21-001 — completed result reconciled and serial CPU queue resumed

SP-0 completed all 200,000 updates in 1,971.80 seconds (about 32.9 minutes). Independently scored equal-connection validation force MAE: BEST 41.0463, LAST 52.5222 and late20 mean 46.2764 meV/angstrom. E64 seed0 reference BEST 34.5874 and late20 mean 39.0084 meV/angstrom. This is a single-seed fixed-recipe observation; paired comparison and independent interpretation remain pending.

The successful result push was followed by an immediate PR-head assertion failure, which stopped the queue before the next fit. All six original published files were subsequently verified at immutable commit 10f5c4e and the missing local receipt reconstructed without another training run. The exact failed API response was not recorded, so delayed metadata refresh is a hypothesis. Seven verification tests and two independent technical reviews approve the operational repair. Model code, data, settings, execution revision and update budgets remain unchanged.

The unchanged runner verified and skipped SP-0, reused its verified scores/publication, and started S64-0; actual progress observed at update 7000. S64-1 and SP-1 remain pending. Local CPU only, no CU cost. No main-test/recovery evaluation or new tuning. Failure evidence and original artifacts remain retained. See reports/evidence/2026-10-21/representation-publication-recovery.json. Earlier running-status entries are historical.


### REPRESENTATION-CPU-OUTCOME-S64-0-87cd6aab8cb5 — CPU fit outcome, 2026-10-21T08:31:38.018560+02:00

S64-0: status completed. BEST/LAST/late20 equal-connection force MAE (eV/angstrom): 0.04026445483872265/0.049631881936485205/0.04406187550769058. Raw output/source/checkpoint identities retained; independent scientific result review pending. Fixed-recipe two-seed validation evidence only, with no test, recovery, causal symmetry or population-level superiority claim. Failures/incomplete work remain visible and are not replaced with hidden tuning/retries. See reports/evidence/2026-10-20/representation/S64-0/outcome.json.


### REPRESENTATION-CPU-OUTCOME-S64-1-ac5e3075213e — CPU fit outcome, 2026-10-21T09:02:18.527932+02:00

S64-1: status completed. BEST/LAST/late20 equal-connection force MAE (eV/angstrom): 0.03881998344246991/0.04219450804343614/0.04247125108972674. Raw output/source/checkpoint identities retained; independent scientific result review pending. Fixed-recipe two-seed validation evidence only, with no test, recovery, causal symmetry or population-level superiority claim. Failures/incomplete work remain visible and are not replaced with hidden tuning/retries. See reports/evidence/2026-10-20/representation/S64-1/outcome.json.


### REPRESENTATION-CPU-OUTCOME-SP-1-ee65c6980ee7 — CPU fit outcome, 2026-10-21T09:34:58.681023+02:00

SP-1: status completed. BEST/LAST/late20 equal-connection force MAE (eV/angstrom): 0.04166924613571395/0.043646744874561286/0.04574342503314263. Raw output/source/checkpoint identities retained; independent scientific result review pending. Fixed-recipe two-seed validation evidence only, with no test, recovery, causal symmetry or population-level superiority claim. Failures/incomplete work remain visible and are not replaced with hidden tuning/retries. See reports/evidence/2026-10-20/representation/SP-1/outcome.json.


### TEAM-MEETING-EXPLAINER-2026-10-21-001 — evidence-backed team and professor meeting guide

Prepared a 25-page A4 guide covering the original question, data and branch references, model/loss/metric definitions, successful and failed diagnostics, MACE B/A, the controlled E3x comparison, recovery failure, post-hoc controls, professor feedback, representation results, compute/reproducibility lessons, current responsibilities, next actions and 14 questions with answers. It distinguishes a historical E3x fitting plateau, the measured zero-recovery floor and advice about research positioning. No dataset-wide saturation, sole solver cause or general symmetry superiority is claimed.

The current saved queue contains all four completed scalar fits and two reused E64 fits. Independent saved-result arithmetic and 24 remote per-fit file identities passed checks. The two-seed SP-minus-E64 differences are +6.464886 BEST/+6.991188 Late20 meV/angstrom; S64 gives +4.649346/+4.247827. These are fixed-recipe descriptive observations, not best-in-family, causal, statistical or recovery superiority. The broader result interpretation remains pending independent scientific review. The guide performs no new fitting or model inference.

Two independent content reviews and a bounded technical-language review passed after corrections. The guide uses ASD-STE100 Issue 9 writing rules and glossary terms; it does not claim language-conformance certification. All 25 final pages were rendered and inspected. Body sentence/paragraph checks and character bounds passed; 66 source file identities and primary links accompany the editable source. Official standard/dictionary, private review records, checkpoints and test arrays are excluded.

PDF SHA-256: 4c83235ae0a0cb9152f95d1f152df3532d36e0ceb5c1a2bcbc253a8267777484. See reports/team-explainer-2026-10-21/README.md and verification.json. This is a meeting guide, not the five-page course submission. Private grader access, live course instructions, member contribution approval and final release remain separate tasks.


### TEAM-MEETING-NOVICE-REWRITE-2026-10-21-001 — full revision for readers without prior background

The coordinator found that the earlier guide still assumed knowledge of forces, energy, molecular shape and experiment groups. A glossary alone did not meet the intended audience. Preserving that edition, rewrote the entire meeting guide to explain technical concepts where they occur. The revised 41-page guide includes a self-contained spoken opening, an explicitly illustrative energy landscape, worked error calculations and explanations beside every main results table. Definitions distinguish predicted force magnitude from prediction error, energy slopes from parameter gradients, seed spans from intervals, and scalar internal features from rotation symmetry of force outputs.

Two independent plan reviews completed before authoring. Whole-draft review findings were applied; final written review turns were interrupted and are not recorded as final approvals. Final editorial, numerical and artifact checks passed. All 41 final pages were rendered and visually inspected. Sentence and paragraph checks, expected page titles and footers, text bounds, six unchanged central numerical tables and six completed representation-fit rows passed. The guide retains the same 66 source identities. It uses ASD-STE100 Issue 9 writing conventions without claiming formal language certification.

No new experiment, trained test scoring, solver result or scientific acceptance is introduced. Earlier success, failure and incomplete outcomes remain visible. The representation interpretation still has fixed-recipe, two-seed and validation-only limits. This meeting guide remains separate from the five-page course submission and its access, contribution, live-instruction and approval checks.

PDF SHA-256: 3826de308116e625f2587f409cb2815bd4c74eb027ffa7bfe9338929197ab7b7. See reports/team-explainer-2026-10-21-novice/README.md and verification.json. The earlier technical edition remains preserved.


### MIDTERM-STE-ALTERNATE-2026-10-21-001 — separate explanatory five-page report

Prepared group9_Midterm_IDL.pdf as a separate midterm version using the approved novice-explanation approach and ASD-STE100 Issue 9 descriptive-writing conventions. The report explains concepts at first use and retains the force-formulation question, prior failed checks, MACE B diagnostic, controlled predictive results, all-model E3x recovery failure, post-hoc controls and claim limits. A short preliminary representation paragraph distinguishes internal-feature comparison from removal of rotation symmetry. The earlier report and long meeting guide remain preserved.

Two independent plan and draft reviews completed; required corrections to force/energy sign, experiment chronology and first-use meanings were applied. Final checks pass: five pages including references; unchanged course style and typography; all five original predictive rows, five recovery rows and baseline-figure bytes verified; original report PDF unchanged. All five final pages were rendered and visually inspected. Body/abstract checks cover 173 descriptive sentences, maximum 21 whitespace words, with no paragraph above six sentences. Captions, tables and references received separate inspection. This is not formal dictionary-level STE certification.

Early layout attempts exceeded five pages. Redundant prose and manual page breaks were removed; flexible floats then interrupted paragraphs. Anchored placement corrected the flow while retaining five pages. No font/margin reduction, experiment, trained test scoring or scientific acceptance occurred. Saved-rubric coverage passes; live instructions, contribution confirmation, private-repository grader access and final team approval remain open.

PDF SHA-256: ff4e92693738980279fcf47e8e6f91e8b2cb86a01e283de45051cba7e68438e0. Source, rubric map, numerical bindings and artifact checks are in reports/midterm-ste-2026-10-21/. Publication to this review branch is not submission or PR integration.

The staged formatting check initially stopped on trailing spaces in TeX. Trimming those spaces and recompiling corrected it. All five rendered pages are pixel-identical to the visually inspected edition. Final PDF SHA-256: 49920b1cf1d6c080da1fd451b9739f95766a5118c777f5301109fd1b58b3570b. No scientific content, typography or layout changed.


### MIDTERM-STE35-2026-10-21-001 — explanatory report revision

Revised 36 passages in a separate five-page group9_Midterm_IDL edition. Two independent plan/draft reviews completed; clarified the loss definition and preserved the preliminary representation limit. Final body/abstract checks find 125 sentences, maximum 29 words, within the new ceiling. All five pages were rendered and visually inspected. Numerical-token inventory including multiplicity, equations, tables, bibliography, course style and source figure are unchanged. The prior strict explanatory edition, original course report and long novice guide remain preserved. No experiment, trained test scoring or scientific acceptance changed.

See reports/midterm-ste35-2026-10-21/README.md, verification.json and source-records.json. Final PDF SHA-256: 1da46801ee70f1df40285e14d50cb3fb7bcd851b7c51094ff070f6ee6f3b04b4. Working-draft, contribution, private-access, live-instruction and approval checks remain open; this is not submission or PR integration.


### MIDTERM-CORRECTED-B-2026-10-23-001 — selected explanatory report corrected

Applied B1-B10 to selected version B: measured rather than matched cost; faculty long-range advice distinguished from the current cutoff; correct closed-loop work and quadrature limits; defined representation controls/tuning/symmetry and records; restored test-selection qualifier and nonconverged energy trace; linked eleven minima to six targets; precise record/atom/component force MAE; exact late window; seed 0 figure caption. The stopping rule now explicitly says maximum absolute force component. Consistent terminology and novice explanations are retained.

Two independent plan/source reviews passed. Six saved exported representation scores were checked without loading models; the limited both-seed ordering matches the report, while broader interpretation remains preliminary. Final artifact checks pass: five pages including references, unchanged course style/font/margins, all pages rendered and visually inspected,126 body/abstract sentences, maximum 28 words and at most six sentences per paragraph. Table and loss blocks and original figure bytes are unchanged. Bibliography titles now carry the same five primary destinations. Required disclosures remain; no new gaps or causal result is added.

Earlier layouts exceeded five pages; bounded prose concision and removal of redundant visible URLs restored length without shrinking typography. Prior report versions remain preserved. Optional figure-marker polish is deferred. No experiment, model rescore, trained test evaluation or scientific acceptance changed.

The report uses immutable research-code/evidence snapshot 66d082158100a0c81b9a7815f7ac2a65f155b7ee, verified to exist; it predates this corrected PDF. The repository was verified private on 23 October; grader access is unconfirmed. Working-draft wording is removed, but final team approval, four contribution confirmations and live Canvas instructions and access remain open. No visibility change, invitation, merge or submission occurred.

PDF SHA-256: 0381515aeab84b87a7b308006cdebbc8000b8354b3d88351527d61ba516e720f. See reports/midterm-final-review-2026-10-23/REVIEW_CHANGES.md, RELEASE_CHECKLIST.md, source-records.json and verification.json. Content finalization is not submission readiness.


### MIDTERM-R3-2026-10-23-001 — four manuscript corrections

Applied R1-R4: visible GitHub address and fixed research snapshot; removed internal release instructions from PDF; restored direct-force motivation with conditional cost wording; corrected the shared direct-force-output statement. The coordinator confirmed assigned roles for Aaron and Idris with no newly confirmed completed deliverables. Four contribution confirmations remain open.

B1-B10 remain intact. Both tables, loss equation, original figure and course style are unchanged. Optional clarifications identify the visible C BEST triangle at update 184500, message-passing rounds, CPU timing, the 128-record diagnostic subset and the first DOI. Two independent plan/source checks passed. All five pages were rendered and inspected; maximum abstract/body sentence length is 28 words, with no paragraph above six sentences. Initial URL justification produced an underfull warning and stretched spacing; local ragged-right alignment fixed it. An initial formatting script used the wrong working-directory-relative path and made no edit; the corrected invocation completed. No experimental result changed.

Independent content clearance is pending. Grader access to the private repository, all four contribution confirmations, live Canvas instructions and exact due time, and team approval remain unconfirmed. No visibility change, invitation, merge or submission occurred.

PDF SHA-256: a09cbdfd39e352846dd00b60346147a7d0e400ffc4cdd0d228e0e633e6ff2888. See reports/midterm-round3-2026-10-23/REVIEW_CHANGES.md, verification.json and RELEASE_CHECKLIST.md. Earlier editions remain preserved.


### MIDTERM-CONTENT-CLEAR-2026-10-23-001 — manuscript review closed

Independent full manuscript review returned CONTENT CLEAR with no required corrections. B1-B10 and R1-R4 pass, along with the saved substantive rubric, physics/statistics disclosures, links and language profile. The exact source/PDF pins match the reviewed publication 24e3af63c69f3e6d9c34cae847d22eb0213e2f1e. The PDF remains five pages, SHA-256 a09cbdfd39e352846dd00b60346147a7d0e400ffc4cdd0d228e0e633e6ff2888. Local, delivery and remote bytes were rechecked. No manuscript edit or rebuild followed this clearance.

Optional marker visibility and force-direction wording polish are deferred. The final review did not re-fetch publisher metadata, reopen raw representation scores or rebuild the PDF. Existing evidence boundaries remain; content clearance is not a new scientific experiment or acceptance of a broader representation interpretation.

Submission remains ADMINISTRATIVELY BLOCKED: staff access to the private fixed snapshot is unverified, all four contribution confirmations are outstanding, live Canvas instructions/exact due time/timezone are unchecked, and exact-PDF team approval is pending. Assigned roles remain as confirmed by the coordinator. Repository privacy alone does not prove staff lack access. No permission change, invitation, merge or submission occurred. See reports/midterm-round3-2026-10-23/CONTENT_CLEARANCE.json and RELEASE_CHECKLIST.md.

Publication preflight initially stopped because whitespace stripping misread the first porcelain-status path. No commit or push occurred on that attempt. Preserving the status prefix corrected the file-list check; manuscript bytes were unaffected.


### MIDTERM-FACULTY-CONTEXT-2026-10-23-001 — requested attribution clarification

Applied the coordinator's exact 28-word replacement to the first sentence of Section 6, naming feedback from a professor of material science. The field is user-supplied context. This is the only TeX change from the preserved content-cleared edition. No science, table, loss, figure or course-style change occurred. Five-page build/language checks pass; pages 1-4 render identically and page 5 was visually inspected. New PDF SHA-256: 8475210956eef8fce338ea4c708cfa39ff8f368f261960459c894b0cf6b843b0. The prior clearance binds the previous PDF; this single-sentence derivative is locally verified and has not received a separate full review. The four administrative release checks remain open, including approval of this exact updated PDF. No experiment, permission change, merge or submission occurred. See reports/midterm-faculty-context-2026-10-23/verification.json.


### MIDTERM-SUBMISSION-TEXT-2026-10-23-001 — remove internal navigation from report

Removed the printed representation-evidence path from Section 6 at the coordinator's request. It remains in supporting documentation and the evidence records; the report retains its scientific limitation and visible GitHub link. This is the only source change from publication e7bc691ed86248d80a67b6eb7b3975abf0507219. Faculty attribution is preserved. Five-page build and language checks pass; pages 1-4 render identically and page 5 was inspected. PDF SHA-256: c8e958669bd808f0e6bbcfdc3b0f07cdf4694052c4285e490888c2a0fecbd3ab. Prior PDFs remain in Git history and local archived records. No science, table, equation or figure changed; no experiment, merge, permission change or submission occurred. Administrative checks remain open.


### MIDTERM-MAIN-TEXT-LIMIT-2026-10-23-001 — references excluded from page count

The coordinator explicitly clarified that the five-page limit excludes references. This supersedes earlier total-file page-count assumptions; it is user-confirmed guidance, not live Canvas verification. The report now has five main-text pages plus a reference-only sixth page. Restored explanations previously shortened for total length: conjunctive recovery objective, E3x/MACE roles, explicit split counts, graph components/inputs/outputs/channels and Adam/update/seed descriptions. No template/font/margin changes were needed.

Two independent plan/final checks passed. All six pages were rendered and inspected, with no clipping or caption collision. Language checks cover 131 abstract/body sentences, maximum 28 words, at most six sentences per paragraph. Both tables, loss equation, original figure, bibliography, scientific numbers and prior B1-B10/R1-R4 corrections remain unchanged. Latest professor attribution and omission of the printed internal evidence path are retained. No new experiment or interpretation acceptance occurred.

The earlier full CONTENT CLEAR verdict remains bound to its preserved exact PDF; this edition has bounded technical review and local verification. Staff access, four contribution confirmations, live Canvas upload/deadline/timezone and exact-PDF team approval remain open. No merge, permission change or submission occurred. PDF SHA-256: fed98f1b53707dba48a413081293139be8981680710ea470987a2c305438876b. Current sources, map and checks: reports/midterm-maintext-2026-10-23/.


### MIDTERM-REPORT-STYLE-2026-10-23-001 — formal abstract, voice and structure

Applied coordinator feedback: one six-sentence abstract paragraph with introduction/problem/method/result/conclusion; no first-person pronouns throughout manuscript prose; standard academic headings unless course requirements specify otherwise. The saved rubric report criteria were read and no mandatory exact printed heading/order instruction was identified. All content categories remain covered; this does not establish live Canvas compliance.

Relocated the unchanged relaxation-procedure description into Methodology and the existing conclusion into Conclusion and Future Work. Fixed the training-section cross-reference after renumbering. Two bounded independent plan/final reviews passed. All six pages were rendered and inspected: five main-text pages plus reference-only page6. Abstract/body checks cover 130 sentences, maximum 28 words and at most six sentences per paragraph. Both tables, loss, figure, values, settings and bibliography/link destinations are unchanged. One underfull vertical-box warning on the baseline figure page has no observed overlap/clipping. Prior B1-B10/R1-R4 corrections, faculty attribution and internal path removal remain; 'our' becomes 'the' in the faculty sentence under the newer voice instruction.

The initial link-comparison check included the new internal section link and stopped before publication. External destinations were then compared separately; the generated Section 4.2 reference was checked in the PDF. No manuscript edit was needed for that check repair. No experiment, new interpretation acceptance, permission change, merge or submission occurred. Prior full content clearance remains bound to its previous exact PDF; this edition has bounded technical reviews and local checks. Staff access, contribution confirmations, live course upload/deadline and exact-PDF team approval remain open. PDF SHA-256: 528d8da053452e3410e9462b92f576eb844ff2284b02035d698688234ce929ac. Sources and verification: reports/midterm-report-style-2026-10-23/.


### MIDTERM-F1-F2-2026-10-23-001 — two requested wording fixes

The final manuscript review required F1/F2 and otherwise passed. Applied only the two requested replacements: corrected faculty-attribution grammar and limited the 64-channel statement to the three compared model types. No optional edits or additional independent reviews, as the coordinator instructed. Exact source diff checked; five main pages plus reference-only page6, no first-person prose and the 35-word profile pass. Pages2/5 visually inspected; pages1/3/4/6 render identically. Scientific numbers, tables, loss, figure and references unchanged. PDF SHA-256: 6b50703b81dc5f1e9061565ffa75f9ebe7ce63c4042acee4cc1c8947a07f6c16. Earlier exact source/PDF archived locally and preserved in Git history. Staff access, contribution confirmations, live Canvas instructions/deadline and team approval of this exact new PDF remain open. No experiment, permission change, merge or submission occurred.


### MIDTERM-REFERENCE-2026-10-23-001 — fixed research snapshot

The midterm now links research code and evidence to completed representation snapshot 3fa5969e5a0c2fdeaebc76a05bace20799715203. Source, evaluation, configurations, manifests, scripts and evidence match the report branch at verification. Only the manuscript URL and displayed commit identifier changed. Tables, equations, numerical results, figure and bibliography remain unchanged. The rebuilt PDF has five main pages and a reference-only sixth page; numerical, link and page-layout checks pass. PDF SHA-256: 885e22ef7b3424ee9dc5a4344af1d87bb0846b3569ba510ecf1c50fa73e43fba. No experiment or submission occurred.


### MIDTERM-CANONICAL-2026-10-23-001 — report organization

Designated reports/midterm as the canonical report directory. Moved seven earlier editions into reports/archive/midterm with each file preserved byte-for-byte; relocations.json maps original paths and hashes to archived locations. The canonical TeX, PDF, style and figure bytes are unchanged. Canonical PDF SHA-256: 885e22ef7b3424ee9dc5a4344af1d87bb0846b3569ba510ecf1c50fa73e43fba. Current navigation and team review assignments point to the canonical source. No experiment, scientific result or submission changed.
