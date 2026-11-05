# Saved-output representation-study closeout

26 October 2026. Proposed closeout of the completed [representation study](REPRESENTATION_COMPARISON_SPEC_2026-10-20.md). The coordinator approved proceeding with saved-output analysis while the MACE workstream continues. This adds no fitting, model inference, checkpoint selection or recovery experiment.

## Question and claim boundary

Compare scalar-only internal graph features with scalar/vector features under the already executed E64-selected training recipe. All models predict forces by differentiating an invariant energy, so all force outputs remain equivariant. This analysis cannot isolate symmetry causality, establish best-in-family performance or demonstrate recovery/test generalization.

Primary contrast: SP minus E64, paired by seeds 0 and 1. Secondary contrast: S64 minus E64. E64 has 43,521 parameters, SP width 88 has 43,473, and S64 width 64 has 25,473. Retain the actual 0.1103% SP parameter gap. Do not tune scalar settings, replace seeds or choose another comparison after viewing the closeout.

## Fixed inputs

Use the six saved score exports at `reports/evidence/2026-10-20/representation/{E64,S64,SP}-{0,1}/score.json`, parameter counts, implementation/preflight/reproduction/resume/launch receipts and the original study specification. Preserve each export's byte identity.

Retained numerical inputs live under the coordinator's existing research-data root. Record an explicit path map; historical paths name an earlier checkout location. Resolve only these known paths against the retained root, preserve original records, and keep private absolute paths out of public exports:

- New scalar fits: `runs/aspirin-representation-cpu-2026-10-20/{S64,SP}-{0,1}/segments/000001`.
- Reused E64 seed 0: `runs/aspirin-e3x-v3-scientific-recovery-2026-10-15/C`.
- Reused E64 seed 1: `runs/aspirin-e3x-controlled-2026-10-16/energy-1`.
- Twelve cached BEST/LAST prediction files: the two E64 reference score directories and the four scalar `analysis/000001` directories under the representation run root.
- Labels: only `data/processed/aspirin/validation.npz` and its manifest's validation metadata. Never open `train.npz`, `test.npz` or the main source archive.

Freeze a before-analysis file inventory with SHA-256 and byte lengths. Authenticate cached scores with their existing `score.sha256`, predictions with the cached score hashes, checkpoints with exported score hashes, and original E64 files with the historical launch's reference pins. Verify the prepared manifest identity, split hash and validation-array hashes. Validate architecture, seed, objective, complete-budget summary, data identity and numerical-source fingerprint across the six runs. Published scores may differ from raw scores only by the documented normalized score-directory field.

## Checkpoint and history verification without model restoration

Read checkpoint envelopes and their control metadata with the plain msgpack decoder. Authenticate the envelope, opaque model payload hash, provenance and control hash. Do not deserialize the model payload or initialize a model. The embedded BEST model state remains opaque bytes; compare it with the BEST checkpoint payload.

Use the authenticated LAST checkpoint control's validation history, and require the standalone `validation.json` to match. Require exactly 401 events at updates `0,500,...,200000`, with no duplicates, missing events or future events. BEST is the first strict minimum, LAST is update 200000, and late20 is the arithmetic mean at `190500,191000,...,200000`. Check those reductions and BEST update against the saved score and selection metadata. BEST control history is its legitimate selected-update prefix, not a second 401-event history. Inventory both historical fresh-process reload proofs and verify their role, update, status and checkpoint bindings; opaque payload inspection does not freshly reconstruct parameter-state updates.

## Independent endpoint reduction

Read cached NPZ predictions with `allow_pickle=False`. Validate exact ID support, unique IDs, shapes, dtypes, finiteness and groups; join with the manifest by exact record ID and preserve atom order. Expect 974 records, 21 atoms and five distinct validation groups: `13`, `18`, `18_2`, `66`, `78`.

For BEST and LAST independently recompute per-connection and equal-connection force MAE, pooled force MAE, residual molecular energy MAE, pooled Cartesian force RMSE and the zero-force reference. RMSE is the square root of the mean squared component error. Use absolute tolerance `1e-14` plus relative tolerance `1e-12` for differences caused by reduction order; retain actual maxima. Do not adjust predictions or recenter energies.

The public claim remains validation-only. A fresh reduction of cached predictions is not a fresh model prediction, an independent training run or a hardware-portability result.

## Summaries, timing and figures

For every family and seed, retain BEST, LAST and late20 errors, selected update and parameter count. Report family means, both seed values and scalar-minus-E64 paired differences for all three checkpoint/trajectory summaries: BEST, LAST and late20. Positive deltas favor E64. Call signs mixed if seeds disagree or BEST and late20 differ in sign; keep LAST visible. Report paired connection-level endpoint differences descriptively, without treating branches, frames or repeated checkpoints as independent samples. Do not add hypothesis tests or confidence intervals with two seeds/five reused connections.

Training timing comes from all recorded completed segments. Distinguish whole-fit elapsed, compilation, warmup, validation and steady compiled-update time. Missing or invalid timing remains unknown. Report historical E64 reuse and different execution dates; these timings do not establish a controlled efficiency or inference-speed comparison. Preserve the recorded publication interruption separately from numerical completion.

Create reproducible CSV/JSON tables and publication-quality PNG/PDF figures: BEST/LAST/late20 by family with both seeds visible; paired scalar-minus-E64 differences; and unsmoothed validation histories with a late-window view. Use consistent meV/Å units, visible seed labels and descriptive captions. Figures must not imply confidence intervals or clipping of selected excursions.

## Implementation and acceptance

Create a small saved-data analysis script and a separate plotting script. They may import NumPy, msgpack and Matplotlib plus the standard library. Do not import JAX, Torch, Flax, the trainer or the existing scoring module, whose import path reaches model code. Keep all historical sources and inputs unchanged.

Use fresh output directories and refuse overwrite. Preserve any failed analysis attempt with its failure reason. Unit tests must cover corrupted bindings, schedule/tie/window boundaries, unequal-group pooling, exact ID joins, near-zero reductions, paired sign changes, invalid timing and existing-directory refusal. Rehash all inventoried inputs after analysis and again at final verification.

Closeout passes only when all six candidates, twelve endpoint files, fixed histories and independent metric reductions verify; generated tables/figures reconcile; inputs remain unchanged; and both independent saved-result reviews accept the bounded interpretation. Retain failures rather than rerun training or inference to repair reporting.

Deliver a readable Markdown research report, editable analysis/plotting source, CSV/JSON reductions, figures, verification inventory and reviewed acceptance record. Record this independent workstream as a new tracker item without renumbering P01–P30. Keep MACE and teammates' workstreams unchanged. Publication is prepared as a separate bounded, reviewable change.
