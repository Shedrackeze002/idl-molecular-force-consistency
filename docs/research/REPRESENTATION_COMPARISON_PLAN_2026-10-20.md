# Representation comparison implementation plan

Implementers: read the linked study specification first. Complete each testable task in order and obtain the two independent technical reviews required by the team workflow. This plan describes future implementation; its checkboxes are not completion claims.

**Goal:** Compare scalar-only and scalar/vector graph features at fixed training exposure, with a parameter control and a recoverable Colab workflow.
**Architecture:** Reuse frozen E3x energy-model, loss, sampler and validation functions. Add a small study package outside historical source inventories for config/counting, run/resume, Drive storage and analysis. Preserve historical CPU code and artifacts.
**Tech stack:** Existing Python3.12/JAX/E3x/Flax/Optax lock, NumPy, standard-library JSON/hash/ZIP utilities, Colab notebook and Drive mount.
**Spec:** [Study specification](REPRESENTATION_COMPARISON_SPEC_2026-10-20.md).

## Global constraints

- E64degree1/width64; SPdegree0/nearest parameter count within1%; S64degree0/width64; seeds0/1; at mostsix fresh fits.
- Energy-derived forces for every configuration; batch4,Adam0.003,float64,200000updates,validationevery500, training-only normalization, existing26/5whole-connection train/validation split.
- Frozen scientific/evaluator/scorer files and historical results unchanged. New package path: `experiments/representation_study/`; new configs in `configs/representation-study/`; new tests explicitly invoked, because the default pytest list is curated.
- Remote GPU assumption Colab; unit replenishment available; no new runtime operation authorized by plan acceptance. No test payload, tuning, source upgrade or solver change.

## Review focus

1. A scalar energy's force equivariance must not be confused with internal equivariant channels; test nonzero outputs, not only zero initialization.
2. Configuration identity and dtype must survive CPU/GPU metadata differences and dict serialization ordering without accepting a different objective or dataset.
3. A disconnected Colab server must resume full optimizer/RNG/BEST state from the latest complete checkpoint; interrupted/corrupt writes must not erase the prior generation.
4. Timing must include sampling/transfers/bookkeeping and full validation/save overhead; CPU/MACE or FLOP estimates cannot masquerade as E3x GPU cost.
5. Selection must use all five validation connections with the same401checks, preserve BEST0/incomplete outcomes, and never open test numerical files.

### Task1 — Configuration, parameter matching and physics controls

Files: create `experiments/representation_study/__init__.py`, `protocol.py`, `configs/representation-study/proposal.json`, `tests/test_representation_protocol.py`.
Interfaces: `make_configs(scalar_width: int) -> dict[str, dict]`; `match_scalar_width(counts: dict[int,int], target: int) -> dict`; `count_model_parameters(model_config: dict) -> int` using synthetic shape tracing. IDs, seeds and tie rules as spec. Source hashes/count receipt are immutable inputs to the filled launch record.

- [ ] Write failing tests for degree/head exclusions, exactsharedsettings, nearestcount/tielower,1%bound, duplicate SP/S64 elimination and deterministic batch-index pairing.
- [ ] Add nonzero synthetic four-atom checks for E64 and scalar features: shapes distinguish degree0/1, dtypefloat64, energy invariance, force rotation/reflection/translation/permutation covariance, netforce/torque and force-loss gradients finite/nonzero. Reuse the current deterministic geometry/rotation fixture and tolerances (`atol1e-8,rtol1e-7`, finite differences≤1e-5), without editing its historical test.
- [ ] Observe the failing new tests before minimal implementation. Count parameters with synthetic shapes only; no scientific data read or fit.
- [ ] Run `PYTHONPATH="$PWD:$PWD/src" python -m pytest tests/test_representation_protocol.py -q` with the unavailable archive sentinel used by current routine tests; verify explicit cases and unchanged source inventory. Commit only this task's files.

### Task2 — Full-state checkpoint storage and interruption recovery

Files: create `experiments/representation_study/storage.py`, `tests/test_representation_resume.py`.
Interfaces: `commit_checkpoint(root: Path, state_bytes: bytes, control: dict, identity: dict, probes: dict) -> dict`; `latest_complete(root: Path, identity: dict) -> dict | None`. Return generation, update, statebytes, control and verified probes; retain every rejected candidate's reason. Signature fields and serialization schema freeze before runner implementation.

- [ ] Write failing synthetic tests for interrupted copy before completion receipt, corrupt newest payload falling back with explicit rejection, saved optimizer/sampler/modelRNG/BEST/history, dict-order-neutral named-tree identity, incompatible objective/dataset refusal, compatible-device metadata acceptance with the spec's exact probe inputs/tolerances and unchanged live state, rejecting future BEST contamination, repeated resume and local-lock release on process death.
- [ ] Pin that only small receipts are inventoried and only selected/fallback payloads are read. A no-op resume cannot create a fake trained update. Concurrent obvious generation/update collision must fail rather than overwrite evidence.
- [ ] Implement local staging/uniqueDrivegeneration/receiptlast/checksum semantics. No persistent stale owner, no distributed service, no deletion/automatic model retry.
- [ ] Verify with fake temporary storage and synthetic checkpoints, including split/uninterrupted updates matching optimizer/RNG/selection and predictions to1e-10 CPU. Run `PYTHONPATH="$PWD:$PWD/src" python -m pytest tests/test_representation_resume.py -q`; commit.

### Task3 — Resumable runner, honest timing and independent scoring

Files: create `experiments/representation_study/runner.py`, `analysis.py`, `tests/test_representation_runner.py`, `tests/test_representation_analysis.py`.
Interfaces: `run_fit(config: dict, bundle, storage_root: Path, identity: dict) -> dict`; `measure_cost(config: dict, bundle, storage_root: Path) -> dict`; `score_checkpoint(config: dict, generation: dict, bundle) -> dict`; `summarize_pairs(rows: list[dict]) -> dict`. Import current initialization/build_step/sampler/validate primitives; never patch their source or identity guards.

- [ ] Write failing synthetic tests for identical per-seed sample streams across architectures,401validationchecks at fullbudget,firststrictBEST, BEST0 disclosure, finite-output failure, fullstate recovery at a scheduled-validation boundary and between boundaries, no duplicate future validation prefix, and validation per-group weighting/partial batches.
- [ ] Refuse main-test file opens in loader tests; consume the original train/validation/validation_minima_diagnostics interface (including an empty diagnostic file if originally empty), with original manifest identities. Primary selection remains974validationrecords; test numerical payload stays excluded. No copying a whole preparation directory that contains test arrays into the notebook.
- [ ] Test delayed mock sampler/transfer/checkpoint work enters elapsed forecast, host/device timing synchronization occurs, score/compile costs are separated, incomplete timing stays incomplete, predictedCU uses an identified rate or remains an explicitly unknown-rate scenario, seconds/update divides100-updateblocks by100, all BEST/periodic/time/final writes enter the bound, reconnect/setup costs are counted once, and current balance does not cap updates.
- [ ] Implement the fixed protocol, save interval/progress, selected-generation scoring and paired descriptive summary, with scalar-minus-E64 MAE deltas and pooled Cartesian-component RMSE computed from squared errors. Leave standalone prediction benchmarking with Aaron. Authenticate original payload/control bytes and embeddedBEST before named-PyTree shape/dtype checks; do not reencode restored dictionaries for byte equality. Report parametercounts/device segments/BEST/LAST/seed and connection differences without unsupported statistical claims.
- [ ] Run the four new explicit test modules plus existing physics/arm-training/checkpoint regressions with broadarchivefixturesdisabled. Verify all frozen file hashes. Commit and obtain implementation review; no real-label scientific fit yet.

### Task4 — Short Colab notebook and reviewed timing/launch record

Files: create `notebooks/representation_comparison.ipynb`, `experiments/representation_study/colab.py`, `tests/test_representation_notebook.py`, and `docs/run-representation-study.md`.
Interface: `run_notebook(settings: dict) -> dict` dispatches readiness/timing or a filled reviewed fit record; a separate mode prevents an unreviewed timing probe from silently becoming full training. In either mode Run All is an end-to-end action; no mid-notebook review confirmation cell or None-filled blockers. The launch record is prepared before the production notebook is handed over.

- [ ] Write failing offline tests for compactcellstructure, progress at setup/compile/checkpoint/train, explicitsource/datahashes, no testpackage, noalarm/READY/manualstaleowner/runtimeGPUnameban, no six-hour segment cap, and restart choosing latestverifiedcheckpoint without repeating completedfits.
- [ ] Implement compatible pinned environment setup, Drive mount, explicitdataexport identities, measured readiness/timing and run/resume. Store science locally during updates and durably in Drive at scheduled boundaries. Keep a visible completed/failed summary and data-download instructions.
- [ ] Run `PYTHONPATH="$PWD:$PWD/src" python -m pytest tests/test_representation_notebook.py -q`; inspect notebook JSON/sourcehashes without allocatingGPU. Record existingGPUcompatibilitytests as unexecuted until the separate remote stage.
- [ ] After implementation review and an authorized Colab session, fill exactcounts/runtime/lock/sourcehashes, readiness/resume receipts, measuredforecast and actualrate into `reports/execution/representation-study-2026-10-20/launch.json`. Keep any failure as evidence; no hiddenretry/tuning. Two reviews of the filled launch record precede its scientificdispatch.
- [ ] Publish accepted deliverables/results through the existing clean workflow. Preserve private review material separately, and do not put tool identities or the raw model/data packet in shared history.

## Acceptance and handoff

The plan is ready when specification, future interfaces, explicitverification paths and budget scenarios have two independent reviews with material findings resolved. This is different from implementationacceptance/GPUreadiness/scientificacceptance. The immediate next implementation deliverable is Tasks1–2, followed by the runner/notebook and bounded timing verification. Report release and teammate work continue in parallel.
