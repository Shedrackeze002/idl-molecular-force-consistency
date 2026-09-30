# Fixed convergence-budget implementation plan

> REQUIRED: superpowers:executing-plans, native execution after two independent scientific/engineering reviews. Preserve all workspaces, user changes and local feature branch. No implementation delegation.

**Goal:** Test sensitivity to a fixed4000additional updates with unchanged saved Adam states/data/model/weights; retain balanced outcomes and literature basis.
**Architecture:** immutable source preflight → isolated continuation runner/controller → savedJSON-only report. Shared trainer/loss/optimizer infrastructure unchanged.
**Tech Stack:** locked JAX/Flax/Optax/e3x float64CPU; standard JSON/checkpoints and matplotlib; no new dependency.
**Spec:** docs/superpowers/specs/2026-09-30-convergence-budget-design.md.

## Global constraints

- Frozen configs/aspirin-convergence-budget.json,118raw+3compact pins/controlmetadata before arrays and afterarms/end; source states/update/count2000 immutable. Whole source ancestry/inputs/normalization/raworiginal baselines verified, no arbitrary tolerance relaxation.
- Three arms replay→balanced→force-only, end6000/additional4000/globalcountcontinuity;512000new/cumulative768000committed exposures percompletearm,128discardedwarmup separately.
- Same128/16fully-fittedanchor/seed0/model/LR.001/weights/microbatch16; no LR/weight/reset/schedule/architecture/GPU/validation/test/selection/energyrecenter.
- First2001 equals prioractualAdam scale1 rawprobe within1e-10/1e-8; source2000+first2001+100interval2100…6000 exactly42observations. Raw original strict .1jointgates retained, forceonlyforce separate; fullpilotpending regardless.
- Terminal window5500→6000 signed relativechange, abs<.01 descriptive only; missing/zero denominator unavailable, never theorem or acceptance.
- Arm1200s/controller3900s boundarycaps includeverification/output; partial counts/sources/states/failure retained, laterarmsblocked. No overwrite/researchretry/budgetadaptation. Current/historical code distinct.

## Review focus

1. Resume allowed only in isolated extension with actualsavedmoment/counter and originalbaseline; no generictrainer resume or reset/rebase.
2. Allsettings/newoptimizer-prerequisiteJSON gates beforepreparedloader, including completeprobe matrix, declaredsourceweights and hashes.
3. Interrupted update attempted/unknown work versus synchronizedcommittedupdate; wall/freshprobe/source failure blocksnextarm and reportinterpretation.
4. Actualglobal/localexposures/history42/scheduledcounter/oracle/lastcheckpointnewprocess matched, no blocked6000claims.
5. Terminal-window labels limited, no convergence/capacity/physicaltaskcausality/generalization; literature protocolmatches/differences documented, no unsupportednoveltyclaim.

### Task 1: Immutable preflight and isolated continuation

**Create:** src/force_consistency/convergence_runs.py,tests/test_convergence.py. **Modify:** src/force_consistency/cli.py.
**Interfaces:** convergence_preflight(config)->(fitting,anchor,model,norm,states,sources,optimizer_records); run_convergence_arm(config,state,fitting,anchor,model,norm,original_baselines,oracle,root,provenance)->dict; execute_convergence_matrix(root,runner)->dict; diagnose_convergence(config)->dict.

- [x] RED missingmodule test first: realpreflight immutablepin/counters; omitted/stale/rehashedbadcontrols beforeparent arrays. Tiny mixed-derivative model nonzeroAdamstates continued3updates match uninterruptedweightedreference, no reset/baseline rebase/source mutation, exactexposures/scheduled2001oracle/warmup. Partialwall/exception/nonfinite/newprocess/badcount/oracle blocks, protectedroot/CLI executionvalid distinction. Explicit KeyboardInterrupt during an unsynchronized update, atomic history-write failure and checkpoint-save failure tests; preserve previously persisted verified checkpoints/history and distinguish pending unknown work.
- [x] Implement strict frozenextension controls andoptimizer raw/compact completeness gates BEFOREoptimizer_preflight; isolatedsource-state-copy continuation usingexisting weightedstep, reserved budget boundaries/synchronizedcompletion and incremental history/checkpoints/provenance. Atomic JSON writes and versioned atomic checkpoint files preserve prior verified snapshots on persistence failure; explicit interrupt handling records attempted versus synchronized finite committed work, marks execution invalid and blocks later arms even if final persistence fails (controller records independent failure evidence). Existingcoverage_probe reused inside a new isolated fresh-process wrapper that also verifies checkpoint update/Adam count6000, exact normalization, final state hashes and exact checkpoint SHA256 in its receipt; sourcebytes neverwrite. Corruption tests must preserve predictions while changing count or normalization and reject both.
- [x] Controller logs prospective/immediateoutcomes, allpartial/blocked statuses and repairroutes, actualcounts; pin/state checksafterarms/end. Strictjointoriginalgates andseparateforceonlyforce, fullpilotpending.
- [x] Run .venv/bin/pytest -q tests/test_convergence.py; expectedGREEN. Recordeveryfailedengineeringtrial/RED/GREEN andscopedcommit. Fullsuite beforeproduction inTask2.

### Task 2: JSON-only reporting and one fixed research execution

**Create:** scripts/report_convergence.py,tests/test_convergence_report.py,reports/convergence-budget-diagnostics.md/json/figures/executionrecord/finalreview.
**Interfaces:** terminal_window(history,window)->dict; verified_arm(arm,config)->bool; generate(root,output)->dict.

- [x] RED report tests actual0/partial/blocked/localglobalexposures; missing/duplicate42schedule/nonfinite/counters/checkpoint/oracle/source/rawbaseline cannotpass; zero/missingwindow andsignedrelative arithmetic; forceonlynotjoint/fullpilot; no fictitiousconvergence/capacityclaim.
- [x] MinimalJSON-onlyreport exactallhistories/raw/pergroup/equalgroup/originalfractions/weightedloss/exposures/timings/gates/terminalwindow; actualcounts checked, all3source+trial hashes. Plotallprescribedmeasuredpoints andoriginalgates againstglobalupdates, noselectedcheckpoints. Missingorpartial data retained, interpretationgated.
- [x] Full193prior+newchecks GREEN,codecommit thenexactreviewedconfigurationONCE. Verifyfirststeporacle42rows/globalcount6000, originalgates/exposures/failedoutcomes andsourceimmutability. Append researchfindingsimmediately, savedJSONreport/visualinspection; no retry.
- [x] Finish/selectiveprimaryliterature registry withversions/access/checksums/readsections/relatedness/methoddifferences/researchinferences; askCMUaccesslinks onlyforstillblockedfulltext. Citeopposing evidence, separatefromourresults/noveltytheorem.
- [x] Fresh independentfinalreview, author-verifiedImportant/Criticalrepair(onepass) withRED/GREEN/fullsuite, no numericalrerun. ObsidianCLIbackup/update/scopedreadback; durablearchive/localcommit; markledgercomplete/preserveallrecords.

Completion: all tasks executed; final246-test full suite and independent saved-result review passed. All scientific learning gates fail. Archive/local preservation identities are in the execution record and append-only evidence log.
