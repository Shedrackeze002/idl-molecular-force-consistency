### Task 2: Frozen calibration, replay, matrix controller and CLI

**Files:** create src/force_consistency/loss_balance_runs.py, tests/test_loss_balance_runs.py; modify cli.py.
**Interfaces:** balance_preflight(config)->(train,manifest,norm,selected,historical,calibration); verify_balance_replay(current,source,tolerance)->dict; execute_balance_matrix(root,runner,replay)->dict; diagnose_loss_balance(config)->dict. CLI diagnose-loss-balance --config.

- [ ] Tests stale/incomplete31pins or calibration hash precede loader, bad ratio/count/update/IDs/state/weights rejects, historical final not passed as initial; engineering tiny-run fresh hashes/checkpoint probe. Replay both partition/errors/allsource22updates, missing/duplicate/mismatch blocks balanced/control. Protected dirs, partial exceptions retained without collisions, wall/nonfinite/downstream blocked and correct repair nextstep. RED then implementation.
- [ ] Reuse load_gradient_sources for existing full source checks; verify calibration JSON hash plus complete38raw artifact index under calibration_run before loader and exact formula/source metadata against verified fresh state/subsets. Save current versus historical provenance/config/subsets/source calibration. No calibration rerun or adaptation. Each new arm invokes run_coverage_arm with isolated weights factory, immutable fixed-baseline gates and fresh expected hashes.
- [ ] Log per-arm rationale/config/controls/gates before running, outcomes/stops/hashes immediately. Incremental root summaries, failed attempted versus blocked distinct. Controller validity all arms finite/fullcap/freshprocess/replay, never defined by learning success. Freshprocess predictions via existing coverage_probe.
- [ ] Run .venv/bin/pytest -q tests/test_loss_balance_runs.py tests/test_loss_balance.py tests/test_coverage_matrix.py; expect GREEN. Full suite117prior+new before production; commit code/prospective evidence.

