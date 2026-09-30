### Task 2: Source-gated case controller and CLI

**Create:** src/force_consistency/optimizer_runs.py; tests/test_optimizer_runs.py. **Modify:** cli.py only.
**Interfaces:** verify_optimizer_sources(config); optimizer_preflight(config,check_budget)->(fitting,anchor,model,norm,states,source_record); execute_optimizer_cases(...)->dict; diagnose_optimizer_steps(config)->dict.

- [ ] RED tests omitted/staleallowlists/compact/controlsettings beforeloader; real preflight sourceconfig/records/hashes and rehashedincorrectweights/count/arrays rejected; no checkpoint sourcewrite. Protectedroots, uncaughtexception/partial FD failure/wallblocked/unavailable directions andrepairroute; exactnewCLI validity andpendingpilot.
- [ ] Implement strictfrozennumericconfig and31newsourcegates then existing balance_preflight69gates/data. Reconcile historicalsourceconfig/compact/references and2000counters/baselines; loadstates againsthistoricalprov/template only. Savedpredictionallframeverification. Current/historicalcode separate.
- [ ] Preregister controller/cases withsourcehashes/inputweights/directions/settings/gates/limits beforeexecution; incrementalpartial/status/summary andactualcounts, caseoutcomes/hasheseachtime,100raw+2compact endchecks. Budgetbefore/aftermajorpreflight andchunks; no sourcefilewrites/research retry.
- [ ] Run .venv/bin/pytest -q tests/test_optimizer_runs.py tests/test_optimizer_diagnostics.py; expectedGREEN. Full152prior+new GREEN beforeoneproduction, commitcode/evidence.

