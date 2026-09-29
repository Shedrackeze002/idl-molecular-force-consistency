# Landscape17 data pipeline and baseline pilot

Date: 29 September 2026

Status: user-approved design, revised after two independent subagent reviews on 29 September 2026. Both reviewers found it ready for implementation planning. Implementation and training have not started.

## Purpose and first milestone

We will implement the submitted study, beginning with reproducible Landscape17 preparation and a verified energy-derived E3x baseline for aspirin. The first milestone is a usable dataset, an audited connection split, and a small training run with saved validation metrics and measured cost. This will establish whether the data and baseline are sound before we compare direct and consistency-trained forces.

The wider study retains the submitted design: a shared E3x backbone, energy-derived forces, ordinary direct forces, a consistency-trained direct arm, and a same-split MACE reference. Aspirin is primary; paracetamol uses separately trained models. Static force error, reference-shape recovery, physical consistency, and compute cost remain separate outcomes. This first milestone does not include the full experiment grid or a publication claim.

## Current evidence

The active repository is `Codex/`, with origin `https://github.com/Shedrackeze002/idl-molecular-force-consistency.git`. The working tree already contains deletion of `proposal.pdf` and an untracked `Group9.pdf`; these user changes must be preserved. The submitted research scope is in `proposal.tex` and `CLAIM_AUDIT.md`.

On 29 September, the public archive was downloaded into memory for inspection. Its size was 53,019,632 bytes and its SHA-256 was `3870ecfd6a5404c6654efba22f003f645e2f4dffffad1b5a22770d3437b3027b`, matching the saved sample provenance. The complete selected data inventory is:

| Molecule | Atoms per sampled frame | Connection directories | Descent frames | Reference minima | Reference transition states |
| --- | ---: | ---: | ---: | ---: | ---: |
| Aspirin | 21 | 37 | 6,944 | 11 | 37 |
| Paracetamol | 20 | 9 | 1,869 | 4 | 9 |

All 405 XYZ files listed in the local sample checksum manifest passed verification. Both sampled branches for each molecule decrease in stored energy as their numeric frame index increases. The maximum absolute force components in the final sampled frames are:

| Molecule, connection 01 | Plus branch | Minus branch |
| --- | ---: | ---: |
| Aspirin | 0.00625404 | 0.00270845 |
| Paracetamol | 0.00153362 | 0.00508297 |

These are raw stored force values. The archive's XYZ headers do not declare units. The source paper uses eV and angstroms in its force criteria, but preparation must trace the archive's serialization and conversion conventions before labeling converted arrays. A low endpoint force alone is insufficient to establish branch identity or structural equivalence.

This is an initial inspection, not the completed dataset audit. No complete duplicate search, geometry matching, unit verification, or split assignment has been performed. The downloaded archive was not retained by this exploratory inspection.

## Compute and implementation choice

Shedrack has access to the CMU cluster, Google Colab, and Kaggle. We will prepare and check data locally, use a single CMU GPU for the main pilot when its scheduler and device are identified, and keep Colab/Kaggle as portable alternatives. Actual allocation, runtime, and GPU memory will determine the full experiment budget. Platform names alone do not establish a GPU model or available hours.

The local machine is an Apple M5 Max MacBook Pro with 48 GB memory. The current Python runtime has NumPy, SciPy, and PyTorch; JAX, Flax, E3x, ASE, and pytest were not found. Development dependencies will live in an isolated environment. CPU checks and a deliberately small pilot should remain runnable without a GPU; scientific timing comparisons will use one recorded GPU environment.

We recommend retaining E3x/JAX as proposed, rather than changing the backbone to an unrelated PyTorch architecture. The official E3x force-field example provides a useful implementation reference, but its MD17 random splits, units, and hyperparameters are not our experimental protocol. A notebook-only implementation would be convenient for exploration but would make repeatable cluster runs and tests harder. The chosen structure is a small Python package with command-line entry points and thin notebook or batch-job launchers added only when the core path works.

## Data preparation contract

1. Download the versioned archive into ignored local storage, verify its size and SHA-256, and extract only safe relative paths. Preserve the original archive and XYZ contents. A checksum mismatch stops preparation.
2. Parse extended XYZ records using the declared `Properties` schema. Validate atom counts, expected elements, nonperiodicity, finite coordinates/energies/forces, and stable atom ordering within each molecule. Preserve molecule, exact source path, connection ID, branch, and numeric frame index for every record.
3. Preserve connection identifiers as strings. Names such as `010`, `310`, and `01_2` cannot be treated as simple integers. Resolve each path directory to exactly one transition-state record using the reference filenames and verified geometry. Both branches and that transition state remain one indivisible connection group.
4. Establish units from author-provided serialization code or dataset metadata. Record source units, target units, numerical conversion constants, and supporting source locations. If this evidence is ambiguous, audit output remains usable but model-ready export fails with an explicit unit-resolution error. Do not silently borrow rMD17 units.
5. Produce a per-connection inventory, branch-length distribution, frame checksums, exact duplicate report, and an invariant-geometry screen for near duplicates. Exact duplicates must compare species and positions, not merely entire file bytes. Geometry-equivalent records with different atom ordering need a species-aware matching check. Keep raw identities even when records are redundant.
6. Match each branch's final geometry against all reference minima with translation, rotation, inversion, and chemically valid atom-permutation handling. A distance-matrix screen may shortlist candidates; it cannot certify a match on its own. Require a unique accepted reference equivalence class, not a unique atom permutation: multiple mappings to one equivalent reference shape are valid. Permutations must preserve species and coordinate-inferred molecular connectivity between the two structures; save a deterministic representative mapping and scores for all candidate classes. Ambiguous or unmatched branches remain flagged and cannot enter the shape-recovery evaluation. Numeric branch order and endpoint energy alone do not establish this map. The preparation matching policy must be frozen before assigning splits, separately from the later model-endpoint success policy.
7. Save arrays separately for each molecule, plus a JSON provenance manifest and a human-readable audit. Use angstrom, eV, and eV/angstrom only after verification. Keep energies and coordinates in float64 during preparation. Training energy offsets are computed from training labels only, saved with checkpoints, and restored for reporting; force labels remain consistent with the energy units.

## Split and leakage policy

The proposed primary group counts are aspirin 26 training / 5 validation / 6 test, and paracetamol 5 / 2 / 2. These counts are design choices, not frozen splits. A reproducible seeded assignment over sorted string identifiers will be created only after the connection audit passes. The split seed, algorithm, archive checksum, exact group IDs, and manifest hash must be recorded. No model results influence the assignment.

Hold out both descent branches and the associated transition state together. Training minima may be included only when connected to a training group. A reference minimum shared by training and held-out connections is explicitly labeled seen; endpoint evaluation does not claim unseen minima. Evaluation-only reference geometries and labels are accessible to the audit and evaluator, not to fitting or normalization.

Do not merge all connections merely because they share a minimum, since this can collapse the network into one group. Instead, preserve connection grouping and label endpoint overlap. Exact or geometry-equivalent duplicate frames that cross partitions must be listed and excluded from the held-out static-error sample under a predetermined rule, with removal counts reported. Freeze duplicate tolerances, transformations, exclusion precedence, and minimum eligible coverage before assigning splits; save raw and eligible record memberships. Connection-level relaxation starts must also be screened for overlap with fitted structures. If this leaves too few eligible starts or connection groups under that declared coverage rule, stop and revise the protocol before training; do not silently choose favorable test frames.

Test data are inaccessible to training and checkpoint selection. The pilot records validation error only. Later study configuration will freeze exact nonstationary branch starts, structural matching tolerances, relaxation budgets, and statistical analysis before any test evaluation.

## Baseline model and training contract

Use one configurable equivariant message-passing backbone with an invariant scalar energy output. Sum atomic energy contributions and differentiate the molecular energy to obtain forces, `F = -grad(E)`. The backbone must return its untruncated equivariant features from the first pilot. The energy head consumes even scalar features; the later direct head will consume odd-parity degree-one features representing polar vectors, with the correct Cartesian convention. Scalar truncation belongs inside the energy head, not the shared backbone. Keep backbone settings identical across study arms and record head-only parameter differences.

Start with a deliberately small configuration, separate from the eventual research configuration: 16 channels, maximum degree 1, two message-passing iterations, eight radial basis functions, and a smooth 5 angstrom cutoff. These are proposed engineering settings to measure feasibility, not tuned or verified accuracy settings. For molecules this small, a fixed all-pairs edge list with smooth cutoff weights avoids moving-neighbor discontinuities during force differentiation.

Train on energies and forces with explicit documented loss weights. The force term is a mean over atom components; the energy term is a mean over structures. The energy head predicts a residual in physical eV: `E_phys = E_train_offset + E_residual`, and `F = -grad(E_residual)` with coordinates in angstroms. Compute the offset and center labels in NumPy float64 before transfer to JAX. Apply energy and force loss scales to physical residuals afterward; do not independently standardize targets and then differentiate without the chain-rule conversion. All scale estimates use training labels only. Explicitly enable and assert float64 for the first pilot and derivative checks, including parameter and coordinate dtypes. A later precision change must be recorded and reverified. Draw training connections uniformly and then draw their frames, so that a long path does not automatically dominate. Never sample from validation or test to fill a batch.

Every run writes its complete configuration, random seed, split hash, code revision or dirty-tree fingerprint, dependency versions, device, numeric precision, parameter count, checkpoint, and validation history. Record compilation time separately from steady-state update time. GPU operations must be synchronized for timing; peak memory reporting must identify the method and its limitations. No measured quantity is called GPU-hours for a CPU run.

First verify learning on a tiny fixed training subset; this is a debugging check and is not a generalization result. Then run a capped pilot on the audited training partition and record held-out validation MAE in verified physical units. Checkpoint selection uses validation only. A pilot's purpose is to show learning and measure cost, not to support the scientific hypothesis.

## Verification and failure handling

Preparation tests must reject corrupted checksums, malformed XYZ schemas, missing branches, nonfinite labels, ambiguous transition-state identities, unresolved units, and cross-partition contamination. A deterministic re-run must produce identical manifests and array contents. Use synthetic malformed records and small checked-in raw examples; do not require downloading the full archive to run basic tests.

For the model, test energy invariance and force equivariance under rotation, inversion, translation, and atom permutation; finite-difference agreement of energy-derived forces; net-force and net-torque residuals; stable differentiation through the cutoff; and batch-versus-single predictions. Use multiple finite-difference step sizes and report tolerances appropriate to precision. Check a nontrivial parameter state so zero-initialized outputs cannot make the tests pass vacuously. Validate parameter gradients of the force loss, since force supervision differentiates through an energy derivative.

Stop a training run on nonfinite losses, persist failure context, and do not report its partial history as a successful pilot. Checkpoint round trips must reproduce predictions. Evaluate every eligible validation record exactly once, including the final partial batch. Accumulate absolute-error sums and component counts rather than averaging unequal batch means. Primary pooled force-component MAE and equal-connection mean force-component MAE use branch frames and associated transition states; reference minima are unique-record diagnostics outside checkpoint selection. Select checkpoints by the equal-connection mean, with the earlier checkpoint winning ties, and preserve this convention across the later study arms. Connection-group count remains visible beside frame count. Shared endpoints and alternate paths can create dependence; frames and seed-by-connection repetitions are not additional independent data. Save endpoint-pair and minimum-equivalence-class identities for the later statistical protocol.

## Proposed repository layout

| Location | Responsibility |
| --- | --- |
| `src/force_consistency/data/` | Download verification, XYZ parsing, inventories, matching, preparation, splits |
| `src/force_consistency/models/` | Shared E3x backbone and energy-derived baseline |
| `src/force_consistency/train.py` | Seeded training, validation, timing, checkpointing |
| `configs/` | Explicit pilot configurations; full study settings added later |
| `tests/` | Data failures, leakage checks, model physics, checkpoint checks |
| `data/processed/` | Ignored derived arrays with provenance manifests |
| `manifests/` | Reviewable frozen splits and source/protocol identities |
| `reports/` | Dataset audit and clearly labeled pilot findings |
| `runs/` | Ignored checkpoints and run histories |

The existing proposal, guide, raw sample, and their source attribution remain intact. The first implementation will add install instructions and a portable single-device launch path; scheduler templates need actual cluster details rather than invented account or partition names.

## Completion criteria and next stages

This milestone is complete when the archive is reproducibly prepared, units and identities are evidenced, overlap is reported, a leakage-controlled split is frozen, physical model checks pass, a tiny-subset learning check succeeds, and one capped baseline pilot saves its validation history and compute record. Model-ready export requires a unique reference-class assignment for every required branch; unresolved branch matching or unit evidence permits a raw audit deliverable but blocks training and milestone completion.

The next milestone adds the ordinary direct head to the same backbone and verifies its transformation behavior. Only after both pilots work will we predeclare consistency strengths, validation MAE matching rules, paired relaxation settings, and the full run budget. MACE, repeated seeds, paracetamol, and statistical analysis follow in separate work units. The October 23 midterm remains the target for audited data and working conservative/direct pilots.

## Sources and review checklist

- Submitted design: `proposal.tex`, `CLAIM_AUDIT.md`, and `Group9.pdf` in this repository.
- Dataset record: https://doi.org/10.6084/m9.figshare.29949230.v1
- Inspected archive: https://ndownloader.figshare.com/files/57341180
- Landscape17 paper and author-code links: https://arxiv.org/html/2508.16425
- Official E3x force-field example: https://e3x.readthedocs.io/stable/examples/md17_ethanol.html

Self-review and two independent subagent reviews checked this design against the submitted question, the actual archive names and counts, the sample endpoints, and the available local environment. Proposed split counts and model settings are clearly distinguished from measured findings. Unit evidence, branch matching, leakage, and physical checks are explicit prerequisites. Compute access is user-confirmed; exact devices and scheduler settings are not assumed. The full evaluation protocol is deliberately outside this first implementation milestone and must be fixed before test outcomes are examined. Review findings and decisions are saved in `docs/reviews/2026-09-29-design-review.md`.
