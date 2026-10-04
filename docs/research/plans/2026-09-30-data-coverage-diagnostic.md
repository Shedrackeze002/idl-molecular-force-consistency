> Research protocol copy. Dates, settings and checkboxes describe the saved protocol, not current experiment completion or a new launch approval. See [the experiment index](../../../reports/PAPER_EVIDENCE_LOG.md) for outcomes. Historical launch records referenced below are retained separately.

# Data Coverage Diagnostic Implementation Plan


**Goal:** Test narrow versus diverse16-frame fitting and128-frame fitting on matched training connections without changing model size or optimizer/loss settings.

**Architecture:** Freeze label-blind, nested subsets from prepared training metadata; use the existing energy model and train-only loader. Extend diagnostics with a full-subset gradient-accumulation runner and fitting/anchor evaluation. Save all three prospective controls and failures, then report outcome-dependent next steps without launching further experiments.

**Tech Stack:** Existing locked Python3.12/JAX0.9.2/Flax0.12.6/E3x1.0.2/Optax/NumPy; float64 CPU. No new dependency.

**Spec:** `docs/research/specs/2026-09-30-data-coverage-diagnostic-design.md`.

## Global Constraints

- This user turn is planning only; no training or implementation starts in it. Native execution is the existing method.
- Exact planned subsets: `manifests/aspirin-data-coverage-subsets-planned.json`, identity8139e442fca011efcbeb5f2d28d701c5f36030d61f045e5e652b98eb18b1be47. Exact settings/reference hashes: `configs/aspirin-data-coverage-diagnostic.json`.
- Support order `01,23,77,58_2,44,22,78_2,01_2`; PCG64 seed20260930; half-up rational rank transfer; no label-dependent selection or replacement group.
- Frozen aspirin prepared/split IDs, source subset hash338e75a921b30820b70cfba47d1f06e7356a6d2232905d88975d088844403a3c and train-only scales unchanged. Open train.npz ONLY.
- Seed0, zero head,16features/degree1/2iterations/8basis/5A; Adam1e-3, float64; deterministic full-subset updates; microbatch16;2000updates/1800s per arm; no learning early stop, no resume.
- C's eight microbatches produce ONE weighted gradient and ONE Adam update. All chunks run sequentially at identical incoming parameters. Initial parameter/optimizer hashes must match across A/B/C. Same original joint objective; no additional loss weights.
- Scores at0/every100/250/final; fitting and original16 gates separate;10% immutable initial thresholds unchanged. A replay atol1e-10/rtol1e-8 on all stored100-update metrics before B/C.
- Protected fresh directories, finite-state gates, exhaustive input-only new-process checks<=1e-10, balanced prospective/actual evidence. Preserve user proposal changes and review folders.
- All previous failed runs stay immutable. No full pilot, GPU trial, scientific arm, seed sweep, capacity/loss/lr change, budget extension or test access.

## Review Focus

1. Unequal trajectory lengths, string group IDs, numeric frame ordering and rank collisions: reject ambiguity; exactly reproduce fixed IDs, no silent replacement.
2. Accumulation and final partial chunks: match normalized full-subset loss/gradients/Adam; no optimizer step or parameter change between chunks.
3. Anchor overlap: B has only2 fitted anchor frames, C has16; save fitted/not-fitted flags and never call this scientific held-out evaluation.
4. Replay/identity/wall/nonfinite failure: retain current failure and downstream blocked statuses; finite learning failure must not be conflated with execution failure.
5. Compute and acceptance comparisons:128 uses8times exposures per update; independent own/anchor gates and measured-cost reporting prevent attributing extra computation or label difficulty solely to sample count.

### Task 1: Frozen subset and identity contract

**Files:** Create `src/force_consistency/coverage.py`, `tests/test_coverage.py`; use existing planned manifest/config.
**Interfaces:** Consume `load_training_partition(directory, expected_split_hash) -> (PreparedPartition,dict,Normalization)` and source IDs. Produce `select_coverage_subsets(partition: PreparedPartition, manifest: dict, anchor_ids: list[str], seed: int=20260930) -> dict` containing ordered support, rank schedules and each arm's IDs/indices/hash/anchor overlap; `verify_coverage_subsets(selected: dict, planned: dict) -> None`; `take_subset(partition, record_ids) -> PreparedPartition`, readonly arrays.

- [ ] Write tests `test_exact_nested_selection_and_rank_distribution` asserting A16/B16/C128 distinct IDs, B⊂C, A⊂C, equal group/branch counts, planned group/order/hashes, B-anchor overlap2. Assert selection unchanged by arbitrary label perturbations.
- [ ] Write `test_train_only_source_and_rank_failures` with held-out NPZs deleted, train-only load guard, numeric frames2/10 order, stale hash, outside-training role/group, duplicate IDs and insufficient/colliding ranks rejected without replacement.
- [ ] Run `.venv/bin/pytest -q tests/test_coverage.py`: expected red for missing helpers.
- [ ] Implement selection using exact spec's rational half-up mapping and existing array/manifest checks. Runtime must regenerate planned payload deterministically; no source-label reads for selection.
- [ ] Rerun coverage/prepare tests: expected green; review exact ID manifest equality; commit scoped helpers/tests.

### Task 2: Equivalent full-subset optimizer and exhaustive diagnostics

**Files:** Modify `src/force_consistency/train.py` only to expose/reuse normalized loss; create `src/force_consistency/coverage_runs.py`; extend `tests/test_coverage.py` and training tests if needed.
**Interfaces:** Produce `normalized_loss(params, batch, model, normalization) -> (scalar,metrics)` preserving current build_step mathematics. `accumulated_loss_and_grad(params, batches, model, normalization) -> ((scalar,metrics),gradient_tree)`; `build_accumulated_step(model, normalization, learning_rate) -> callable(params,optstate,batches)->(params,optstate,metrics)` weights each chunk by frame_count/full_count and applies Adam once. `coverage_metrics(params, fitting, anchor, model, microbatch_size=16) -> dict` includes exhaustive counts/per-group/raw MSE/baselines and anchor fitting overlap. `run_coverage_arm(config, fitting, anchor, root, provenance) -> dict`; `coverage_probe(root) -> None`.

- [ ] Write `test_accumulated_step_matches_full_batch` on seven varied geometries split3/3/1: compare scalar loss, BOTH raw MSEs, all gradient leaves and all first-update parameter/optimizer leaves to production full batch with atol/rtol1e-10. Assert only one optimizer update and parameters fixed between chunks. Repeat continuity for two updates.
- [ ] Write `test_existing_step_objective_unchanged` comparing current production step to shared-loss implementation, including nonzero mixed force gradients; watch red before exposing helper.
- [ ] Write exhaustive fitting/anchor tests asserting no missing/duplicate frames, correct component counts, B overlap2 and separate14 not-fitted metrics, empty-partition null ratios, fixed own/anchor thresholds and per-group normalization.
- [ ] Implement shared loss and accumulated gradient/Adam semantics; reuse existing finite/provenance/checkpoint helpers. Run full-gradient equivalence tests: expected green before production use.
- [ ] Write runner tests for2000 fixed-cap semantics using tiny caps, no gate early stop, wall/incomplete/failure distinctions, immutable baselines, matching fresh initial state hashes, first observed passing scheduled update, preserved initial/final histories, protected directory and no resume. New-process probe test deletes held-out files and checks every saved fitting/anchor frame at1e-10.
- [ ] Implement exhaustive microbatched evaluation, elapsed/exposure history and checkpoints with current/historical identities separated. No general bundle loader. Run `.venv/bin/pytest -q tests/test_coverage.py tests/test_training_contract.py tests/test_diagnostics.py`: expected green. Commit scoped implementation/tests.

### Task 3: Reviewed matrix controller, execution and balanced report

**Files:** Modify `src/force_consistency/cli.py`; add CLI `diagnose-coverage --config`; create `reports/data-coverage-diagnostics.md`/`.json` after execution and figure generator under `scripts/`; append existing evidence log; update run guide/README only with measured status.
**Interfaces:** `diagnose_coverage(config: dict) -> dict` executes verified A, then B, then C. A replay gate compares all saved reference100-update metrics, required2000updates and fresh-process checkpoint verification; failed/short A blocks B/C. B numerical/identity/short-budget failure blocks C; a finite B learning failure does NOT block C. Root status separates execution validity from individual fit and anchor acceptance.

- [ ] Write controller tests for replay mismatch/stale reference/corrupt array blocking without model fitting, learning-failed B permitting C, nonfinite/short-budget B blocking C, accurate planned/blocked summaries and separate execution/learning exit status.
- [ ] Implement controller/CLI and prospective evidence append with unique record IDs, current config/subset/source hashes and per-arm outcomes. Integration test tiny caps with training-only fixture and fresh-process reload; expected red then green.
- [ ] Run full `.venv/bin/pytest -q`: expected all existing67 plus new tests green. Save failures and verification in the retained execution ledger; commit source before experiment.
- [ ] Recheck exact settings and candidate IDs against reviewed spec; append prospective matrix entry. Run `.venv/bin/force-consistency diagnose-coverage --config configs/aspirin-data-coverage-diagnostic.json` once. Expected research outcome UNKNOWN; no extra attempt. Record A replay as an executed trial, not assumed existing evidence.
- [ ] Inspect A/B/C outcomes, checkpoints, timing/exposures, raw/common-anchor metrics, fit/anchor gates and hashes. Plot update and elapsed-time views with complete source identities; distinguish fitted/not-fitted anchors. If any cap truncates, mark comparison incomplete rather than substituting another budget.
- [ ] Save report/compact hashes, append every success/failure/block,  Fresh independent final code/evidence review; important fixes require failed regression then green full suite. Commit measured artifacts; no automatic full pilot.

## Decision after this plan

If C passes BOTH fit and original16 gates with verified complete budget, save a separate two-review baseline-protocol amendment before full training. If C fails either, return to the planned architecture/loss-gradient investigation with this new evidence. Partial improvements motivate a hypothesis, not a selected successful run or relaxed threshold. No result is expected or claimed in this plan.
