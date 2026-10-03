> Research protocol copy. Dates, settings and checkboxes describe the saved protocol, not current experiment completion or a new launch approval. See [the experiment index](../../../reports/PAPER_EVIDENCE_LOG.md) for outcomes. Historical launch records referenced below are retained separately.

# Frozen Optimizer-Step Diagnostic Implementation Plan


**Goal:** Observe weighted-loss derivatives, exact saved-state Adam proposal and fixed equal-norm references at the three completed checkpoints without continuing training.
**Spec:** docs/research/specs/2026-09-30-optimizer-step-diagnostic-design.md.
**Architecture:** isolated numeric helpers -> immutable source/snapshot controller -> saved-JSON report. Reuse current gradient/weightedloss/checkpoint infrastructure, no baseline/trainer edits or newdependencies.

## Global constraints

- Frozen configs/aspirin-optimizer-step-diagnostic.json: previous69sources+calibrationcompact and31loss-source+losscompact verified BEFORE arrays;100raw+2compact recheck aftercases/end. Exactsourceconfig/input/order/count/weights/state/reference andbothrawmetric identities. Finaloptimizer/update2000, no sourcewrites.
- Same128fit/16fully-fittedanchor/model/LR/scales; fit-derived statefulAdam/resetAdammatchednorm/steepestmatchednorm, scales0.1/1, FDh1e-4/1e-5 with1e-8/1e-4 at-least-one criterion perapplicableobjective/direction/role. Frozenidentity1e-10/1e-8. Strictprescribedsettings validated beforeloader, notarbitraryconfigtolerances.
- Every temporarytrial retained; no winner/committedupdate/resume/sweep/seed/architecture/budget change/pilot/GPU/test/validation. Statehashesunchanged;0trainingexposures, diagnostics separatelycounted. Zero/unavailabledirections explicit, notfakeverification/convergence.
-3orderedcases,600s totalbetweenoperationcap; preservedpartials/blockedstatuses/operationalrepairroute; interpretation gated onvalidity. Preregister andlogoutcomes immediately. Oneexactconfig invocation, no retry.

## Review Focus

1. Stateful proposal uses weighted fitting gradient and actual savedmoments/count+1; manualmoment and builder identities, epsilon/dtype correct.
2. Equal-norm reset/steepest controls are artificial local directions; no training reset, causal/capacity/convergence/generalization claim or bestdirectionselection.
3. Complete100raw+2compact gates beforearrays, allmetadata/state/input/protocol/FD/scales/tolerances strict; verify originals unchanged before/after allcases.
4. Directional FD verifies BOTH scaledobjectives onfitting/anchor along fitting-derived unitdirections; bothhresults retained, at-leastone gate; zero/unavailable directions explicit.
5. Partial/budget/invalidcase/preflight failure preservesexactattemptcounts/artifacts andblocksnextcases; reportcannotpromote counterfactualgate tolearning/pilotacceptance.

### Task 1: Numeric directions and frozen snapshot analysis

**Create:** src/force_consistency/optimizer_diagnostics.py; tests/test_optimizer_diagnostics.py.
**Interfaces:** adam_directions(params,optstate,gradient,learning_rate)->(directions,verification); direction_geometry(ge,gf,direction,weights)->dict; analyze_optimizer_snapshot(state,fitting,anchor,model,norm,weights,config)->dict.

- [x] Testfirst RED: manualnonzero mu/nu/count Adam leaves/counter2001 inclbiascorrection/eps; weightedquadraticdirection/slopes/samenorm/negativegradient; zero/missingdirections; finiteFDwrongsignreject; fixedstate/counter hashes; tiny nonzero-head model weightedgradient/fittingandanchordirections/allprobes saved withoriginalbaselines.
- [x] Implement minimalnumerichelpers, reuse splitgradients/weightedaccumulator/referencebuilder and coverage_metrics; allgradients/chunks atfixedparams, exactthree directions andtwoscales, unitFDbothsteps andobjectives, callbacks preservepartialcells/counts. Snapshotstateparameters/optimizer untouched; prospectiveintermediateoptimizer discarded.
- [x] Run .venv/bin/pytest -q tests/test_optimizer_diagnostics.py; expectedGREEN. LogRED/GREEN evidence andscopedcommit.

### Task 2: Source-gated case controller and CLI

**Create:** src/force_consistency/optimizer_runs.py; tests/test_optimizer_runs.py. **Modify:** cli.py only.
**Interfaces:** verify_optimizer_sources(config); optimizer_preflight(config,check_budget)->(fitting,anchor,model,norm,states,source_record); execute_optimizer_cases(...)->dict; diagnose_optimizer_steps(config)->dict.

- [x] RED tests omitted/staleallowlists/compact/controlsettings beforeloader; real preflight sourceconfig/records/hashes and rehashedincorrectweights/count/arrays rejected; no checkpoint sourcewrite. Protectedroots, uncaughtexception/partial FD failure/wallblocked/unavailable directions andrepairroute; exactnewCLI validity andpendingpilot.
- [x] Implement strictfrozennumericconfig and31newsourcegates then existing balance_preflight69gates/data. Reconcile historicalsourceconfig/compact/references and2000counters/baselines; loadstates againsthistoricalprov/template only. Savedpredictionallframeverification. Current/historicalcode separate.
- [x] Preregister controller/cases withsourcehashes/inputweights/directions/settings/gates/limits beforeexecution; incrementalpartial/status/summary andactualcounts, caseoutcomes/hasheseachtime,100raw+2compact endchecks. Budgetbefore/aftermajorpreflight andchunks; no sourcefilewrites/research retry.
- [x] Run .venv/bin/pytest -q tests/test_optimizer_runs.py tests/test_optimizer_diagnostics.py; expectedGREEN. Full152prior+new GREEN beforeoneproduction, commitcode/evidence.

### Task 3: Frozen once-only diagnostic and balanced report

**Create:** scripts/report_optimizer_steps.py; tests/test_optimizer_report.py; reports/optimizer-step-diagnostics.md/json/figures/executionrecord/finalreview.

- [x] Report RED tests preflightzeroattempts/partialblocked, failedFD/probe gating, zero/unavailable counts, falselearningclaims. JSON-only rawloss/fractions/slopes/displacements/FDallcells/compute/source/attemptcounts, no bestdirection orfictitiouszero count updates.
- [x] Fullsuite GREEN thenexactreviewedconfig ONCE. Reconcileallsource/state/hash/cellcounts/objectives/manualAdam/builderequality/FD/probes; retainall failures, no retries/retraining. Generateplots, visuallyinspect, appendfindings/limitations/hashes.
- [x]   andauthorverified Important/Criticalrepairs withRED->GREEN/fullsuite, no unreviewednumericalrerun. Markplan/ledgercomplete, durablearchive/localcommit, retainworkspaces/userchanges/branch.

## Next decision

Onlyvalidlocaloutcomes can motivate a separatelyreviewed followup. Hypotheticalfeasibleimprovement is notgateclearance; noautomaticlongertraining/optimizer/capacitychange. Fullpilot andscientificcomparisonpending.

Execution status: completed30September2026. Three once-only frozen cases verified; full193tests passed125.07s after one author-verified report-completeness repair. Independent technicalreview and response-delivery limitation recorded in docs/reviews/2026-09-30-optimizer-step-final-review.md. No new training or automatic followup authorized.
