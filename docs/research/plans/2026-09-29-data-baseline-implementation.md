> Research protocol copy. Dates, settings and checkboxes describe the saved protocol, not current experiment completion or a new launch approval. See [the experiment index](../../../reports/PAPER_EVIDENCE_LOG.md) for outcomes. Historical launch records referenced below are retained separately.

# Landscape17 Preparation and Baseline Pilot Implementation Plan


**Goal:** Produce an audited Landscape17 dataset, reproducible connection splits, and a verified, capped aspirin energy-derived E3x pilot with validation metrics and measured cost.

**Architecture:** A small Python package separates source verification, parsed records, geometric identity, split eligibility, and model training. The data commands create canonical manifests; training loads only authorized training/validation arrays. One equivariant backbone retains scalar and vector features, with a head-local energy readout.

**Tech Stack:** Python 3.12, NumPy, SciPy, ASE, NetworkX, JAX, Flax, E3x, Optax, and pytest. Resolve and lock compatible package versions in the isolated environment, recording the lock digest; CPU and NVIDIA installs use the same core versions with backend-specific JAX extras. No macOS GPU dependency.

**Spec:** `docs/research/specs/2026-09-29-data-baseline-design.md`, including the reconciled findings in `docs/reviews/2026-09-29-design-review.md`.

**Status:** prepared for two independent reviews and user execution handoff. No implementation action is authorized by this document alone.

## Global Constraints

- Preserve the existing deletion of `proposal.pdf` and untracked `Group9.pdf`; do not stage unrelated files.
- Use only Landscape17 labels for these experiments; never mix rMD17 labels into the core training data.
- Source archive: file ID `57341180`, 53,019,632 bytes, SHA-256 `3870ecfd6a5404c6654efba22f003f645e2f4dffffad1b5a22770d3437b3027b`.
- Preserve both branches and their transition state as one string-ID connection group.
- Split counts: aspirin 26/5/6, paracetamol 5/2/2; freeze assignment only after the required audit passes.
- Require evidenced angstrom/eV/eV-per-angstrom units before model-ready export; never infer unit conventions from rMD17 or numerical magnitude.
- Compute training offsets/scales only from training labels, and center in NumPy float64 before JAX transfer.
- The first pilot and derivative checks explicitly enable and assert float64, including parameters and coordinates.
- Pilot backbone: 16 channels, degree 1, two iterations, eight radial basis functions, smooth cutoff 5 angstrom, fixed all-pairs edges without self-edges.
- Evaluate each eligible validation frame once; primary MAE uses branch/TS records and unique reference-minimum errors are separate diagnostics. Select checkpoints by equal-connection mean force-component MAE, earlier update wins exact ties. Record pooled MAE too.
- Test labels/records are inaccessible to fitting, model selection, and pilot metrics. Audit access to reference/test data is separate.
- Preparation gates apply per selected molecule. Aspirin is the first pilot; a paracetamol gate failure blocks its own export/training and must remain reported, without silently removing the failed connection or changing thresholds.
- Counts describe connection groups, not guaranteed independent samples. Do not count frames or repeated seeds as independent observations.
- Record compilation separately from steady-state timing and verify any requested GPU backend; do not silently fall back to CPU.

## Review Focus

1. Nonstandard string IDs and repeated endpoint-pair connections must resolve without integer coercion or guessed branch semantics (Task 2).
2. A chemically equivalent endpoint may require atom reordering; index-fixed alignment or unrestricted species swapping must not certify identity (Task 3).
3. Near-duplicate screens must not silently miss records at bucket boundaries, or hide inconsistent force labels (Tasks 3–4).
4. Large energy offsets and JAX's default precision must not break the physical force derivative or relative-energy loss (Tasks 4–5).
5. Partial validation batches, resumed random sampling, and unavailable GPU requests must not distort metrics or claim reproducibility (Task 6).

## Shared types and policy values

`FrameRecord` is an immutable record with `record_id: str`, `molecule: str`, `kind: Literal['path','minimum','transition_state']`, `source_path: str`, `connection_id: str | None`, `branch: Literal['plus','minus'] | None`, `frame_index: int | None`, `species: tuple[str,...]`, `positions: ndarray[N,3]`, `energy: float`, `forces: ndarray[N,3]`, and `source_sha256: str`. Parsed floating arrays use float64. Record ID is the archive-relative source path; duplicate geometries retain different raw IDs.

`PreparedPartition` contains float64 `positions`, centered `energy_residual`, `forces`, atomic numbers, ordered record IDs, and group/role memberships. `PreparedBundle` contains only `train`, primary branch/TS `validation`, unique-record `validation_minima_diagnostics`, their manifests, and `Normalization`; no test arrays. Validation diagnostics contain only eligible minima connected to validation groups and not already seen in training; a minimum connected to several validation groups appears once, and a minimum shared with test follows validation-over-test precedence. It has no primary connection membership and cannot select checkpoints. Test-only reference minima are inaccessible to the loader. `Normalization` records `energy_offset_eV`, `energy_loss_scale_eV`, and `force_loss_scale_eV_per_A`.

Freeze preparation policy in `configs/preparation.json` before generating assignments: full-atom orthogonal RMSD after chemical graph mapping; duplicate tolerance `1e-5` angstrom; endpoint-identification tolerance `0.05` angstrom; reference equivalence tolerance `1e-5` angstrom. These are proposed preparation defaults, not future model-success criteria. Molecular graphs use species-labelled nodes and bonds at distance greater than 0.4 angstrom and at most `1.2 * (ASE covalent_radius_i + covalent_radius_j)`; preserve the graph policy and ASE version in the manifest. A graph mismatch or exhaustive-search failure yields an unresolved identity, not a negative match.

Separate exact stored-coordinate duplicates from roundoff-equivalent and near-duplicate geometries. Hard label rejection applies only when coordinates are exactly equal after the accepted atom permutation (without approximate alignment), with energy tolerance `1e-7` eV and force-component tolerance `1e-6` eV/angstrom. For rigidly transformed/roundoff-equivalent or near geometries, save aligned energy/force differences for diagnosis and retain leakage exclusions; do not reject merely because the differences exceed serialization-rounding tolerance. A stronger transformed-label rejection policy requires independent source-specific numerical evidence before adoption. Freeze this policy before checking discrepancies.

Use `numpy.random.Generator(PCG64(20260929))` to permute sorted connection IDs separately for each molecule; persist the actual lists, not only the seed. Duplicate exclusion precedence is training > validation > test. Higher-priority partitions retain records, lower-priority partitions exclude matching records; preserve provenance and counts. Each held-out connection must retain at least one eligible nonstationary frame per branch (`max(abs(force)) > 1e-3` eV/angstrom) and at least one eligible static-evaluation record. Every expected group must remain represented. Failed coverage blocks model-ready export and requests protocol revision before fitting.

Canonical identity is SHA-256 of UTF-8 JSON with sorted keys, fixed separators, no NaN, and no timestamps or absolute paths. Binary arrays are hashed in canonical little-endian float64/int64 form with shape/dtype recorded. ZIP container timestamps are not scientific identity; compare decompressed arrays and manifests on rerun.

### Task 1: Verified source and schema-aware parser

**Files:** Create `pyproject.toml`, `src/force_consistency/__init__.py`, `src/force_consistency/data/{types,source,xyz}.py`, `tests/test_source_xyz.py`; modify `.gitignore`. Lock dependencies when this task executes; ignore environments, full raw archives, processed arrays, and runs.

**Interfaces:** `fetch_archive(destination: Path) -> SourceManifest`; `parse_xyz(text: str, *, source_path: str, molecule: str) -> FrameRecord`; `read_records(archive: Path, molecules: tuple[str,...]) -> list[FrameRecord]`. `SourceManifest` supplies source URL/version, size, SHA-256 and member inventory. Export exception types `SourceError` and `SchemaError`.

- [ ] Write failing tests that reject checksum mismatch, absolute/traversal archive members and links, bad atom counts, periodic frames, duplicate/missing schema fields, malformed force shapes, nonfinite numbers, and truncated rows. Assert a permuted Properties-field layout yields the same arrays as a normal layout.
- [ ] Run `python -m pytest tests/test_source_xyz.py -q`; confirm failures represent missing behavior before implementation.
- [ ] Implement streaming download to a temporary file, verify before atomic promotion, and safe extraction or direct-member reads. Keep original bytes. Parse declared schema/types and metadata using quoted-header handling; require one XYZ frame per member and expected H/C/N/O composition for the selected molecule. Parse path IDs and numeric indices without guessing TS destinations.
- [ ] Run the parser tests and read all checked-in sample XYZ files without networking. Assert 405 parsed records, 21 aspirin atoms and 20 paracetamol atoms, unchanged source digests and finite labels. This confirms parsing, not units.
- [ ] Commit only Task 1 files and the dependency lock.

### Task 2: Connection inventory and evidenced unit conversion

**Files:** Create `src/force_consistency/data/{audit,units}.py`, `configs/preparation.json`, `tests/test_connections_units.py`, `manifests/landscape17-unit-evidence.json`, `reports/landscape17-source-audit.md`.

**Interfaces:** `inventory_connections(records: Sequence[FrameRecord]) -> ConnectionInventory`; `verify_units(evidence: Mapping[str,object]) -> UnitContract`; `convert_record(record: FrameRecord, units: UnitContract) -> FrameRecord`. Inventory stores string connection ID, both ordered branch record lists, TS ID, endpoint-pair metadata, raw counts and any unresolved status. Unit evidence includes versioned source URL/commit, serialization/conversion location, source units and constants.

- [ ] Write failing tests preserving `010`, `310`, `01_2`, and parallel/self connections; reject missing branches, duplicate frame indices, gaps in numeric sequence, inconsistent species order within a branch, and ambiguous TS lookup. Verify absent/unsupported unit evidence blocks model-ready conversion while a raw inventory can still be reported.
- [ ] Run `python -m pytest tests/test_connections_units.py -q` and confirm relevant failures.
- [ ] Build the TS lookup from parsed reference filenames `(minimum_i, minimum_j, optional_suffix)` and their exact contracted directory names; require exactly one lookup hit and validate composition and reference presence. The first branch frame is displaced from a TS and must not be required to equal its coordinates. Save geometry-distance diagnostics without treating a nearest-TS guess as evidence.
- [ ] Trace units through author-provided DFT serialization/calculator code or metadata, freezing the source evidence. Implement explicit unit conversions and record constants. If this trace cannot establish units, stop conversion and deliver the raw audit; do not fill the evidence file with an assumption.
- [ ] Run Task 1–2 tests. Inspect the full verified archive and report 37/9 groups, 6,944/1,869 path frames, 11/4 minima and 37/9 TS records for aspirin/paracetamol. Enumerate any missing/extra or malformed records instead of accepting aggregate counts alone.
- [ ] Commit source, tests, frozen policy/evidence and audit, excluding the archive and derived arrays.

### Task 3: Chemical identity, duplicate audit and endpoint mapping

**Files:** Create `src/force_consistency/data/{geometry,matching}.py`, `tests/test_geometry_matching.py`; extend the audit report and preparation manifest.

**Interfaces:** `chemical_graph(record: FrameRecord, policy: PreparationPolicy) -> nx.Graph`; `align_records(a: FrameRecord, b: FrameRecord, policy: PreparationPolicy) -> AlignmentResult`; `audit_duplicates(records: Sequence[FrameRecord], policy: PreparationPolicy) -> DuplicateAudit`; `map_endpoints(inventory: ConnectionInventory, records: Sequence[FrameRecord], policy: PreparationPolicy) -> EndpointMap`. Results include species-preserving permutation, orthogonal matrix, translation, RMSD, search-complete flag and unresolved reason.

- [ ] Write failing tests for rotated/translated/inverted and reordered synthetic geometries, graph-isomorphic atom reordering, illegal chemical mappings, several mappings to one class, two competing reference classes, incomplete search, and an accepted-RMSD boundary pair. Test correctly transformed equivalent force labels, reject contradictory labels only on exact stored-coordinate matches, and keep two near geometries with legitimately different forces from a synthetic stiff conservative potential valid while recording their leakage relationship.
- [ ] Run `python -m pytest tests/test_geometry_matching.py -q` and establish failures before implementation.
- [ ] Use NetworkX labelled-graph isomorphisms between the two coordinate-derived graphs and Kabsch alignment over both determinant signs. Exhaust permitted mappings to minimize full-atom RMSD; choose lexicographically first mapping for exact ties. Bound searches at 100,000 candidate permutations per pair and mark a reached bound unresolved. Do not report search-exhausted pairs as nonduplicates. Save mapping/transform diagnostics.
- [ ] Construct the invariant descriptor by concatenating sorted interatomic distances within each unordered species-pair block. For N-atom structures and accepted full-atom RMSD epsilon, query candidates with SciPy `cKDTree.query_pairs(r=N*epsilon+1e-10, p=2, eps=0)` in matching molecule/composition blocks. The `1e-10` angstrom margin accounts for numerical screening error. This is conservative because the sum of squared pair-distance differences after accepted alignment is at most `N * sum_i ||delta_i||^2 = N^2 * epsilon^2`, and sorting species-pair blocks cannot increase the optimal squared difference. No approximate query pruning. Confirm each candidate with the complete chemical aligner; failed or incomplete searches remain unresolved and block leakage certification. Equivalent reference classes use the duplicate identity rule; ambiguous transitive clusters that fail all-pairs equivalence are unresolved.
- [ ] Match every branch terminal record to all reference classes using the frozen `0.05` angstrom identification threshold. Save all scores, chosen class, permutation, and status. Classify exact stored-coordinate equality separately from aligned roundoff/near equivalence; save serialization conventions and transformed-label differences under the declared policy without calling ordinary near-geometry force differences corruption.
- [ ] Run Task 1–3 tests. Reproduce the aspirin minus/minimum-1 close match while demonstrating index-fixed RMSD approximately 0.68845 angstrom; require a valid graph mapping and final RMSD below 0.01 angstrom rather than prescribing a swap without checking chemistry. Report unresolved branch identities without silently dropping them.
- [ ] Commit Task 3 changes and scientific audit outputs.

### Task 4: Frozen splits, eligible memberships and training-only normalization

**Files:** Create `src/force_consistency/data/{splits,prepare}.py`, `tests/test_splits_prepare.py`, `manifests/landscape17-{aspirin,paracetamol}-split.json`; extend `pyproject.toml` with `force-consistency` CLI and `src/force_consistency/cli.py`.

**Interfaces:** `assign_groups(inventory: ConnectionInventory, policy: PreparationPolicy) -> SplitManifest`; `apply_eligibility(records: Sequence[FrameRecord], splits: SplitManifest, duplicates: DuplicateAudit, endpoints: EndpointMap) -> EligibilityManifest`; `fit_normalization(train: Sequence[FrameRecord]) -> Normalization`; `export_prepared(...) -> PreparedManifest`; `load_training_bundle(directory: Path, *, expected_split_hash: str) -> PreparedBundle`.

- [ ] Write failing tests for exact group counts/disjoint membership, deterministic PCG64 assignment, kept string IDs, branch/TS co-membership, duplicate precedence and removal counts, shared-minimum seen flags, failed branch coverage, unmatched endpoints and stale policy/archive hashes. Assert changing validation/test labels cannot change training normalization; training loader never opens test arrays. A minimum shared among validation groups or validation/test appears once in eligible validation diagnostics, never in primary group MAE; training-seen and test-only minima are absent from that diagnostic partition.
- [ ] Run `python -m pytest tests/test_splits_prepare.py -q` and verify failures.
- [ ] Implement raw/eligible manifests and the declared coverage gate. Include each unique training-connected minimum only once; held-out-only minima remain evaluation-only. Preserve per-record roles so shared minima are not arbitrarily assigned to one connection for path sampling.
- [ ] Compute offset as the mean of unique eligible training energies in float64. Use `max(std(E_train), 1e-3 eV)` for the energy residual loss scale and `max(sqrt(mean(F_train**2)), 1e-3 eV/angstrom)` for force scale. These scale only residual terms. Center energy arrays in float64; derivative semantics remain physical eV/angstrom.
- [ ] Export atomic-number arrays, physical coordinates/forces, centered energies, record IDs, group/role mappings and canonical digests per partition. Commands: `force-consistency audit --archive data/Landscape17.zip` and `force-consistency prepare --molecule aspirin --policy configs/preparation.json`. Audit covers both selected molecules; model-ready preparation takes an explicit molecule and must fail on its unresolved units, connection mapping, required branch/reference-class mapping, or cross-partition duplicate certification. Check every branch of that molecule before assigning splits. A failing paracetamol preparation cannot silently omit connection 01_2, raise its matching tolerance, or make an already-certified aspirin pilot access paracetamol labels. Unique validation minima occupy a separate diagnostic partition; primary validation includes branch/TS records only.
- [ ] Run Task 1–4 tests and prepare the verified archive twice into separate ignored directories. Compare canonical manifests and decompressed array hashes, not compressed container bytes. If any held-out connection fails coverage, stop before fitting and record the required protocol revision.
- [ ] Test that an unmatched paracetamol branch fails paracetamol preparation while independently certified aspirin preparation remains possible. Record both statuses; do not describe the full two-molecule dataset as model-ready until both pass.
- [ ] Commit manifests, policy and report with code/tests; keep derived arrays ignored.

### Task 5: Shared equivariant backbone and physical energy-derived forces

**Files:** Create `src/force_consistency/models/{backbone,energy}.py`, `tests/test_model_physics.py`, `configs/aspirin-baseline-pilot.json`.

**Interfaces:** `Backbone(config: ModelConfig).__call__(atomic_numbers, positions, dst_idx, src_idx) -> equivariant_features`; `EnergyModel(config: ModelConfig).__call__(atomic_numbers, positions, dst_idx, src_idx, batch_segments, batch_size) -> residual_energy_eV`; `energy_and_forces(params, batch) -> (residual_energy_eV, forces_eV_per_A)`. Define `ModelConfig` in `backbone.py`; input position/parameter dtypes are asserted float64.

- [ ] Write failing physics tests using nontrivial seeded parameters: invariant energy under translations, permutations and O(3) transforms; transformed forces; batch/single agreement; finite differences at steps `1e-3`, `1e-4`, `1e-5` angstrom; net force/torque residuals; finite force-loss parameter derivatives; continuous forces on both sides of the 5 angstrom cutoff; nonzero scalar and vector feature outputs.
- [ ] Run `python -m pytest tests/test_model_physics.py -q`, confirming the unimplemented interface fails.
- [ ] Adapt the official E3x architecture with scalar and polar-vector features retained after the last message pass. The energy head reads only even scalars and sums atomic residual contributions. Use explicit float64 for inputs, Flax parameters and computation; assert `jax_enable_x64`. Derive `F=-grad(E_residual)`; physical offset is added only for reporting. No direct force head is implemented in this milestone.
- [ ] Physics acceptance uses absolute/relative tolerances `1e-8/1e-7` for symmetry and batch checks. Require central differences to approach analytic forces over the stated step sweep with minimum max-component error at most `1e-5` eV/angstrom. Test cutoff energies/forces tend to matching limits with decreasing displacement; do not assert two distinct nearby geometries have identical predictions. Record tolerances with results; do not loosen them merely to make a failing implementation pass.
- [ ] Run Task 1–5 tests and record parameter counts and verified irrep conventions in `reports/baseline-verification.md`.
- [ ] Commit model, configuration, tests and verification report.

### Task 6: Seeded training, exhaustive validation, checkpointing and cost record

**Files:** Create `src/force_consistency/{train,metrics,runs}.py`, `tests/test_training_contract.py`; extend CLI and pilot configuration.

**Interfaces:** `sample_training_batch(bundle: PreparedBundle, rng_state, batch_size: int) -> (Batch, rng_state)`; `validate(params, bundle: PreparedBundle, model, batch_size: int) -> ValidationMetrics`; `train(config: RunConfig) -> RunSummary`; `save_checkpoint(path, state, provenance) -> None`; `load_checkpoint(path, expected_provenance) -> TrainState`. State includes parameters, optimizer state, update, sampler RNG, model/JAX RNG, normalization and provenance. Validation metrics include evaluated IDs/counts, pooled and equal-connection force MAE, energy MAE and per-group sums.

- [ ] Write failing tests evaluating 11 eligible validation records with batch size 4 exactly once and checking component-count-weighted sums. Assert equal-connection selection differs correctly from pooled selection on unequal paths. Check rejection of wrong split/backend, nonfinite updates, altered checkpoint manifests, parameter dtype drift, and resume sampling continuity.
- [ ] Run `python -m pytest tests/test_training_contract.py -q` and verify failures.
- [ ] Use Optax Adam, learning rate `1e-3`, batch size 4, random seed 0, validation every 100 updates and at update 0/final. Loss is mean squared energy residual divided by training energy scale squared plus mean squared physical force residual divided by training force scale squared. Sampling selects a training connection uniformly, then selects uniformly from its eligible path/TS/associated-minimum record references; shared minima may be referenced by several groups, but retain one unique stored record. Record these effective sampling memberships.
- [ ] Implement full validation traversal with correct partial-batch masks or a separate final shape, no dropped frames. Average primary branch/TS per-group component MAEs equally for checkpoint selection; persist both aggregations and full evaluated-ID coverage. Evaluate each eligible reference-minimum diagnostic once, with its IDs/errors in a separate nonselection result. No test evaluation command runs during the pilot.
- [ ] Verify requested CPU/GPU using JAX device enumeration. Save complete lock/config/split/policy hashes, code revision plus tracked/untracked code fingerprint, platform/device, precision, parameter count, and random state. Time compilation/warm-up separately; synchronize every timed region. Peak GPU memory can be unavailable, but must be explicitly marked unavailable with collection method recorded. GPU-hours apply only to actual GPU elapsed time.
- [ ] Checkpoint atomically via a temporary path, then reload in a new process. Resume must match uninterrupted sampling/state on the same pinned CPU environment within `1e-10` tolerances; GPU nondeterminism limitations must be recorded rather than promising bitwise identity across platforms.
- [ ] Run Task 1–6 tests and commit the trainer and reporting contract.

### Task 7: Learning check, capped pilot and reproducible handoff

**Files:** Create `configs/aspirin-baseline-learning-check.json`, `reports/baseline-pilot.md`, `docs/run-baseline.md`; update `README.md`, `reports/PAPER_EVIDENCE_LOG.md`, and CLI help. No new cluster credentials/account settings are invented.

**Interfaces:** CLI commands `force-consistency train --config <path>` and `force-consistency summarize-run --run <directory>`. Summarize reads saved run/checkpoint metadata and metrics; it does not recompute test results. Training defaults to `--backend cpu` unless the GPU is explicitly requested.

- [ ] Write a failing end-to-end test running a tiny synthetic training/validation bundle without downloading data. Assert valid physical units, manifest identity, saved checkpoint, exhaustive validation, and a failure summary when a required preflight is missing. Keep the integration fixture separate from research data.
- [ ] Run `python -m pytest tests/test_training_contract.py -q` and implement the minimal CLI/report path.
- [ ] For the real learning check, take 16 evenly indexed eligible path frames from the lexicographically first training group and save exact IDs; exclude validation/test. Use fixed-subset batches, Adam `1e-3`, maximum 2,000 updates and 30 minutes, whichever first. Require both energy and force MSE below 10% of their initial values. Save initial/final metrics and call this a tiny-subset learning acceptance check, not negligible-error overfitting. If it fails, investigate before a full pilot.
- [ ] Run the audited aspirin pilot with batch size 4, maximum 1,000 updates and 60 minutes, whichever first. The wall budget includes compilation, validation and checkpointing; record actual completed updates and reason for stopping. Pilot success requires all preparation gates, including unique branch/reference mapping, plus at least 100 updates, finite losses, full validation and a reloadable checkpoint. A numerical run without the preparation gates cannot complete this milestone. Report validation change without claiming a scientific effect or minimum generalization accuracy.
- [ ] Run `python -m pytest -q` once after final code changes; record actual pass/fail and numerical checks. Save the dataset audit, physics verification, learning-check metrics, pilot validation history and measured runtime/device details. Append important observations to the paper evidence log with exact source/run paths, hashes, metric definitions, status, limits and verification evidence. A CPU pilot is an engineering result; compare scientific speed only on recorded identical GPU settings later.
- [ ] Document a fresh-environment CPU install and NVIDIA GPU install from the same lock, data preparation commands, unit/provenance gates, and one portable single-device launch. Verify an available GPU platform if access details permit; otherwise identify GPU portability as unverified. Add CMU scheduler settings only after the actual cluster account/partition/device is known.
- [ ] Commit final documentation and scientific summaries, excluding large checkpoints, raw archive and generated arrays. Deliver what passed, unresolved prerequisites, measured cost and the remaining direct-model milestone.

## Self-review and execution handoff

This plan covers the design's first milestone and both design reviews. Matching and unit evidence have explicit failure states; scientific identity is separate from incidental metadata; precision and metric-selection conventions are fixed. Shared minima are stored once but retain memberships and seen flags. The later direct-force model, consistency sweep, MACE, test relaxations and inference-speed claims are intentionally outside this milestone.

Before execution, two independent reviewers must check this exact saved plan and any material revision. The user then reviews the reconciled plan and chooses native or delegated implementation as required by the writing-plans workflow. Native execution is recommended for this first milestone because the parser, geometry identities and split manifests have tightly connected interfaces; independent review remains mandatory. No coding, environment installation or training begins during plan review.
