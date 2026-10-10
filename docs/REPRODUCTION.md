# Reproduction scope

The subsequent branch-reference interface repair adds `endpoint_references.py`, its generator, classified companion and tests; it documents the legacy filename field without changing frozen split/classification evidence or existing test bodies. Default discovery was 20 modules when that repair was written (the export's 19 original default modules plus `tests/test_endpoint_references.py`); after PR4's review it includes 25, because the five shape-recovery study modules also entered `pyproject.toml`'s default `testpaths` (see Test boundary). The repair's targeted data checks and collection results are recorded in [the interface report](../reports/endpoint-reference-interface.md); the export-era results below remain historical, and no broader numerical suite was rerun for this repair.

This package supports source inspection, small synthetic/physical fixtures and the documented baseline data workflow. Complete historical reproduction also requires the separately retained original source/run/launch packages. Frozen configurations and source guards retain their original identities and do not approve a launch from this export. Obtain the matching package from the project lead, verify the recorded identities, and use its original environment and source revision before attempting an historical reproduction.

## Scientific exports

Core `src/`, every test body, frozen `configs/`, manifests and the environment lock are unchanged. One diagnostic script has protocol-directory metadata references updated to `docs/research/`; no numerical calculation is changed. Package test-discovery metadata explicitly lists the supported default files. Scientific protocol copies omit personal dispatch/navigation boilerplate and identify their dated prospective status.

Report JSON normalizes machine-local checkout prefixes and protocol-directory paths. Numeric leaves, arrays, settings, outcomes, validity and acceptance gates are unchanged under the controlled mapping. Original and exported report SHA256 identities are in [the export manifest](../reports/research-export-manifest.json). Hashes embedded in old reports refer to original canonical artifacts. A source guard that checks an original byte hash must reject a normalized export; do not rehash or relax it to make a launch pass.

The earlier invalid width/rate replacement report is retained as a failed historical record. Its [corrected counterpart](../reports/execution/width-rate-replacement-2026-10-01/corrected-report/width-rate-replacement-diagnostics.md) is authoritative for the completed replacement matrix. Both large JSON exports use Git LFS. Full raw numerical archives and original audit/launch material are separately retained, so this is not a complete raw-run reproduction bundle. Archived raw metadata and historical timing labels do not constitute new instructions or new reviewer attribution.

The historical source-bound `scripts/report_width_rate_replacement.py` and personal navigation script are separately retained. Do not use an omitted historical reporter as a current command. Other source-bound launch/report scripts may also require their original pinned packages and fail closed without them. The baseline data commands and default tests do not supply new launch approval.

## Test boundary

All 36 original test modules remain byte-identical at their original paths. Some historical tests also provide synthetic helpers imported by current tests. `pyproject.toml` lists 19 original default modules explicitly, so helpers retain their original paths without bringing source-bound historical tests into default collection. This reduces regression coverage; the omitted modules include synthetic/controller tests as well as archival checks. No equivalent replacement coverage is claimed.

The attempted complete original set on the export produced **168 failed, 482 passed, 3 skipped and 3 errors** in 337.75 seconds. The saved failures require original runs, processed arrays, audit/source-pin files, protected canonical-byte identities or the separately retained audit reporter. The initial attempt also lacked a checkout-relative runtime and did not explicitly pin inherited child imports; the default verification pins parent and child imports to exported source. The complete failed attempt remains retained. It is not reported as a passing full suite.

Default verification (10 October 2026 evaluator-validation follow-up): the existing locked Python 3.12 environment, explicit checkout-source imports for parent and child processes, no historical data overlay, and `LANDSCAPE17_ARCHIVE` pointing to the separately retained frozen archive. On macOS, `pytest --collect-only -q` collects **470 cases across 25 `testpaths` modules**; `pytest -q` reports **467 passed, 3 skipped in 197.06s (0:03:17)**. These results cover the correction to merged PR #4 (`ff6117450387c65358e96aa3b47ee3e533c38d81`) and are bound to the source hashes and commands in [the follow-up verification record](../reports/execution/evaluator-validation-followup-2026-10-10/verification.json). Collection alone is not an assertion-level result.

Historical verification, superseded for the current tree: the export-era result was **240 passed, 3 skipped in 183.60 seconds**, with 243 collected cases. The historical 656-test prelaunch result belongs to the original implementation. Earlier PR revisions documented 379 and 394 collected cases; accepted head `d4f6374` collected 406 before this follow-up added 64 cases. An earlier Windows rerun recorded source-bound historical-module failures (capacity, width/rate and offset-profile report/run modules and `os.O_DIRECTORY` incompatibility), and an earlier concurrent edit briefly caused a runner import failure. Those historical failures remain evidence for their dated attempts; they are not the outcome of the clean follow-up verification above.

Default modules of the shape-recovery study (added to the default list after the export froze it, during PR4's review): `tests/test_shape_recovery_outcomes.py`, `tests/test_shape_recovery_relax.py`, `tests/test_shape_recovery_report.py`, `tests/test_shape_recovery_runner.py` and `tests/test_shape_recovery_starts.py` are also in `pyproject.toml`'s default `testpaths`, so ordinary collection includes them.

Archive prerequisite: the real-data tests in `tests/test_shape_recovery_starts.py` read the frozen Landscape17 archive (SHA-256 `3870ecfd6a5404c6654efba22f003f645e2f4dffffad1b5a22770d3437b3027b`). They resolve it from `LANDSCAPE17_ARCHIVE` when that environment variable is set, and otherwise fall back to `data/Landscape17.zip` in the checkout. The archive is a separately retained artifact, not a checkout requirement, so the real-data tests SKIP with that reason when neither location has the zip and the rest of the default suite still passes green. A run with the archive set, as used for the verification below, runs them in full.

Run the default supported set from this checkout:

```sh
uv run --frozen pytest -q
```

Default modules:

- `tests/test_capacity_report.py`
- `tests/test_capacity_runs.py`
- `tests/test_capacity_spikes.py`
- `tests/test_connections_units.py`
- `tests/test_endpoint_references.py`
- `tests/test_diagnostics.py`
- `tests/test_geometry_matching.py`
- `tests/test_gradient_diagnostics.py`
- `tests/test_loss_balance.py`
- `tests/test_loss_balance_report.py`
- `tests/test_model_physics.py`
- `tests/test_offset_profile_core.py`
- `tests/test_offset_profile_report.py`
- `tests/test_optimizer_diagnostics.py`
- `tests/test_representation.py`
- `tests/test_shape_recovery_outcomes.py`
- `tests/test_shape_recovery_relax.py`
- `tests/test_shape_recovery_report.py`
- `tests/test_shape_recovery_runner.py`
- `tests/test_shape_recovery_starts.py`
- `tests/test_source_xyz.py`
- `tests/test_splits_prepare.py`
- `tests/test_training_contract.py`
- `tests/test_width_rate_recovery_report.py`
- `tests/test_width_rate_runs.py`

Historical modules outside default collection, with unchanged originals retained here for inspection and helper imports:

- `tests/test_capacity_contract.py`: Historical capacity source-pin file and saved width/rate source states are separately retained.
- `tests/test_convergence.py`: Historical coverage/optimizer runs and processed manifest are separately retained; the first attempt also lacked its checkout-relative runtime.
- `tests/test_convergence_report.py`: Historical diagnostic source runs are separately retained.
- `tests/test_coverage.py`: Historical real-label checkpoint verification and processed arrays are separately retained. Synthetic helpers remain available to current tests.
- `tests/test_coverage_matrix.py`: Historical coverage run and its frozen source configuration are separately retained.
- `tests/test_gradient_report.py`: Historical representation-gradient run and case observations are separately retained.
- `tests/test_gradient_runs.py`: Historical coverage, representation and loss-balance source states are separately retained.
- `tests/test_late_rate.py`: Original late-rate source pins, run state and protected canonical-byte report identities require the original package.
- `tests/test_loss_balance_runs.py`: Historical coverage, gradient and loss-balance raw states are separately retained.
- `tests/test_offset_profile_runs.py`: Original offset-profile source pins and audit gates require the original launch package. Synthetic controller helpers remain available to current report tests.
- `tests/test_optimizer_report.py`: Historical gradient/optimizer observations and source states are separately retained.
- `tests/test_optimizer_runs.py`: Historical gradient/coverage source states and processed arrays are separately retained.
- `tests/test_width_rate_preflight.py`: Original width/rate source states and immutable canonical report pins require the original package; path-normalized exports deliberately have different identities.
- `tests/test_width_rate_recovery.py`: Original width/rate review gate and protected source package are separately retained.
- `tests/test_width_rate_replacement.py`: Original width/rate/recovery audit gates and protected source package are separately retained.
- `tests/test_width_rate_replacement_report.py`: Historical source-bound report audit script is separately retained, along with the original pinned packages.
- `tests/test_width_rate_report.py`: Original width/rate source/audit package and immutable canonical report pins require the original package.

Invoke these files explicitly only with the matching historical packages available. Their original guards remain in force. A new numerical/controller failure is not excused by this boundary; investigate it before publication or launch.

The main shape-recovery study, full training pilot and GPU portability remain pending. Existing diagnostic results do not establish held-out accuracy, continuous stability or a causal spike mechanism.
