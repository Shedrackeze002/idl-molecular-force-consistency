# MACE loader and validation prediction check — specification

26 October 2026. Status: accepted by independent scientific and engineering plan reviews. This is P06 of the [action tracker](../superpowers/plans/2026-10-26-post-midterm-action-plan.md). Idris retains implementation, admission and independent reproduction; Shedrack prepares the coordination plan and reviews. Olufunke retains recovery integration. No new implementation or inference is reported here; implementation and launch checks remain separate.

## Purpose and scope

Bind the verified Run A final EMA artifact to a real calculator, then check that the shared evaluator bridge reproduces its saved validation predictions. This establishes adapter fidelity on the fixed validation inputs; it establishes neither recovery success nor predictive generalization.

Use the existing shared bridge. Add a small loader, a parity worker and a bounded local supervisor outside frozen source. Do not change the bridge, evaluator, training code, dependencies, checkpoints or original predictions. A MACE recovery driver is a later work package.

## Inputs and trusted identities

The v2 private handover is the only numerical input package. It contains 974 validation records, their memberships and saved final predictions. Do not open training or test partitions, other models, selected-model exports or remote storage.

| Artifact | Required SHA-256 |
|---|---|
| Original v2 archive | `ff92c74b83d3103a2542f1564d93be3b153f5342faef2675f9845d9d99958119` |
| Extracted manifest | `3a872e1886c37b0e069e4ecaaa426bbab6a4a771db7c805f21741ee67e22717c` |
| Primary final EMA model, epoch index 2047 | `fbd128215c0d74f5d9e5d3ccde778763221f987535dcc8ce4f4f8471a5b35407` |
| Validation extxyz | `b1d5ef64f434ee9dffb523874ccada47023be4acec57cbbd2f544e201e5ec148` |
| Validation memberships | `2536021ee9966bf9fd930c83bfd92ed63f91bacca3f51e251368caf5db7466e8` |
| Saved final validation predictions | `22bf7d88815ae3a84ecb8717c497f11311f058c1a862f8efd0e08a5efda0d3cd` |

The manifest inventories 11 regular files and is self-excluded. Authenticate its pinned bytes and verify every inventoried member's length and hash. Refuse absolute/traversal member paths, symlinks and missing members before model construction. Use the extracted packet directly; no new extraction framework is required. The archive digest is the retained distribution identity, not a substitute for member verification.

Use source snapshot `3fa5969e5a0c2fdeaebc76a05bace20799715203` as the clean baseline. Bind new code and unchanged bridge/evaluator hashes at implementation completion. Retained scorer code is separately authenticated by the historical rescore records; its old absolute paths are not execution instructions.

## Environment and calculator

Require Python 3.12.13 and the seven package versions and eight raw source-file hashes in the authenticated `ACCEPTED_IDENTITY.json`. Run the packet's metadata-only environment verifier first. Require the accepted macOS arm64 scoring platform for this bounded attempt. Any platform or source difference requires a separate reviewed amendment rather than silent substitution.

Construct one primary-model calculator on CPU with one Torch thread, float64, `enable_cueq=False`, `enable_oeq=False`, and `compile_mode=None`. Use the accepted constructor recipe from the retained scorer. No dependency installation, GPU allocation, training, extra seed or alternate checkpoint is part of this work.

Copy the authenticated 23,397,415 model bytes into the fresh private attempt directory. Hash the copy before construction and after prediction. Pass that exact snapshot path to the pinned constructor and save its identity in the admission receipt. Do not load an unverified path or use a caller provenance dictionary as proof. The original packet remains read-only.

The generic bridge continues to mark caller tags `checkpoint_identity_verified=False`. Save a separate admission receipt linking input hashes, actual snapshot path, constructor settings, runtime/source checks and execution code. Do not relabel the generic bridge's metadata to claim stronger verification.

## Fixed input mapping and outputs

Validate the full input set before constructing the model. Load saved NPZ data with `allow_pickle=False`. Require exact fields `record_ids`, `energy`, `forces`; 974 unique nonempty IDs; float64 finite energy `(974,)`; and float64 finite forces `(974,21,3)`.

Require exactly 974 extxyz records with unique IDs, 21 atoms each, finite coordinates, nonperiodic structures and a consistent atom-symbol order. Require equal ID sets across extxyz, memberships and saved predictions. Each membership has one validation connection; the authenticated sidecar fixes the set as `13`, `18`, `18_2`, `66`, `78`. Preserve `18_2` as distinct from `18`; do not infer replacement membership from filenames or integer conversion.

Process records in the saved prediction ID order, using exact ID joins. Preserve the atomic row order of each authenticated extxyz frame. No rotation, translation, permutation, projection, energy re-centering or fitted correction is allowed.

For each frame, call the shared arm's force accessor once and energy accessor once. Preserve Å coordinates, eV/Å forces and the residual molecular energy in eV. `REF_energy` is already residual. Do not apply the absolute-energy offset `-17647.258992926312 eV` during parity.

The fixed call budget is 974 force requests plus 974 energy requests. These are accessor requests, not guaranteed model-forward counts. Both accessors may trigger a calculator evaluation, and constructor work has separate scope. No extra learned probes, relaxation steps or replacement records are allowed.

Save the joined record IDs, full returned predictions, per-record maximum absolute differences, global maxima, input/source hashes, counts and elapsed time. Preserve completed rows after failure without presenting partial agreement as full parity.

## Acceptance and limits

Parity passes only when all 974 records and all 61,362 force components are present, finite and admitted, and these absolute bounds hold:

- Maximum energy difference from saved residual predictions: at most `1e-10 eV`.
- Maximum force-component difference from saved predictions: at most `1e-10 eV/Å`.
- Relative tolerance: zero. Do not rely on a library's default relative tolerance.
- Exactly one model construction, 974 force requests and 974 energy requests; no model retry or fallback.

These tolerances are prospective engineering criteria for reproducing the accepted local scoring recipe, not newly measured agreement. A later platform comparison requires its own criteria. Prediction agreement is not a new independent training result.

## Resource and failure policy

Use one fresh attempt directory, refused if it already exists. All model snapshots and numerical outputs remain private. Use one process group and a shared 1,800-second wall cap covering setup, admission, model construction, prediction and persistence. Permit at most 10 additional seconds for termination and failure recording; a run crossing 1,800 seconds cannot pass.

Limit the attempt directory to 256 MiB, including the copied model, predictions and logs. Bound each stdout/stderr log at 16 MiB; overrun is a failure. Check time and output size during execution and again after final persistence. Do not claim a memory benchmark or portable performance result from these ceilings.

On timeout, terminate the owned process group, wait five seconds, then kill remaining owned children and wait. Preserve status and partial files; do not restart. A numerical discrepancy is `PARITY_FAIL`; identity, malformed-input, construction and numerical errors are distinct operational failures. A timeout or interrupted/partial run is incomplete and never passes.

Synthetic unit tests may use small analytic fixtures and fake constructors. Production cannot accept injected factories or fake identities. Record implementation checks, source commit, plan/review hashes, exact command and resource settings in a launch receipt before loading the model. Two implementation reviews and a complete launch check precede P08. This specification and plan review alone do not declare runtime or parity success.

## Outcomes and next step

Retain positive, negative and incomplete outcomes. An independent saved-output audit recomputes parity with an exact ID join, validates counts and inputs, and checks raw hashes without loading the model. Publish code, summary and supporting small evidence after review; exclude model/data snapshots and personal operating records.

Accepted parity permits preparation of the separately reviewed unchanged ten-start/six-reference recovery control. Failed parity blocks that control. Idris authors his own implementation and reproduction evidence; no missing contribution is filled with coordinator-authored code under his name.
