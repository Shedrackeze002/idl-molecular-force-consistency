### Task 1: Isolated weighted losses/updates and trainer injection

**Files:** create src/force_consistency/loss_balance.py, tests/test_loss_balance.py; modify src/force_consistency/coverage_runs.py run_coverage_arm signature/step selection only.
**Interfaces:** validate_weights(weights:dict)->tuple[float,float]; weighted_loss(params,batch,model,normalization,weights)->(scalar,raw_metrics); accumulated_weighted_loss_and_grad(params,batches,model,normalization,weights)->((scalar,raw_metrics),grad_tree); build_weighted_step(model,normalization,learning_rate,weights)->step callable with prepare hook; run_coverage_arm(...,*,step_factory=None).

- [ ] Write failing independent weighted objective/gradient and3/3/1chunk tests, witness mixed force-backbone gradients, negative/nonfinite/allzero rejection, positive coefficient sum2, single Adam count/update equality and force-only bias stability. (1,1) exact delegated baseline; incoming trees unchanged.
- [ ] Watch RED missing helper; implement isolated functions/cache JIT weighted loss, sequential chunk weighting then one Adam apply;1/1 delegate existing builder exactly, no baseline loss changes. Test actual factory injection with tiny2-update/fresh-process run and default monkeypatch regression.
- [ ] Run .venv/bin/pytest -q tests/test_loss_balance.py tests/test_coverage.py; expect GREEN. Log meaningful failed checks/full suite appropriately; commit scoped.

