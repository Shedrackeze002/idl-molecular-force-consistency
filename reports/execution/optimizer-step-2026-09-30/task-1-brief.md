### Task 1: Numeric directions and frozen snapshot analysis

**Create:** src/force_consistency/optimizer_diagnostics.py; tests/test_optimizer_diagnostics.py.
**Interfaces:** adam_directions(params,optstate,gradient,learning_rate)->(directions,verification); direction_geometry(ge,gf,direction,weights)->dict; analyze_optimizer_snapshot(state,fitting,anchor,model,norm,weights,config)->dict.

- [ ] Testfirst RED: manualnonzero mu/nu/count Adam leaves/counter2001 inclbiascorrection/eps; weightedquadraticdirection/slopes/samenorm/negativegradient; zero/missingdirections; finiteFDwrongsignreject; fixedstate/counter hashes; tiny nonzero-head model weightedgradient/fittingandanchordirections/allprobes saved withoriginalbaselines.
- [ ] Implement minimalnumerichelpers, reuse splitgradients/weightedaccumulator/referencebuilder and coverage_metrics; allgradients/chunks atfixedparams, exactthree directions andtwoscales, unitFDbothsteps andobjectives, callbacks preservepartialcells/counts. Snapshotstateparameters/optimizer untouched; prospectiveintermediateoptimizer discarded.
- [ ] Run .venv/bin/pytest -q tests/test_optimizer_diagnostics.py; expectedGREEN. LogRED/GREEN evidence andscopedcommit.

