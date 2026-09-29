# Paper evidence log

This is an append-only record of observations, design decisions, and hypotheses that may inform the eventual paper. An entry's status matters: a data audit can establish a preprocessing issue, but it does not establish the effect of consistency training. No model has been trained as of 29 September 2026.

Each new entry should include a stable ID, date, evidence paths/checksums, method and numeric convention, verification status, relevance to the paper, limits, and what must be checked next. Preserve an earlier entry when a later check changes its interpretation; add a correction linked to its ID. Hypotheses, model diagnostics, and measured experiment results must be labeled separately.

## 29 September 2026: initial archive and sample observations

### DATA-001: frame count and group count describe different scales

**Status:** directly verified archive inventory, before preprocessing or training.

The inspected Landscape17 v1 archive contains 6,944 aspirin pathway frames in 37 connection directories and 1,869 paracetamol frames in nine directories. There are 11/4 reference minima and 37/9 reference transition states, respectively. These two selected molecules therefore provide 8,813 pathway frames and 46 connection groups. These are raw counts; no final split or eligible evaluation count exists yet.

**Evidence:** archive file ID 57341180, size 53,019,632 bytes, SHA-256 `3870ecfd6a5404c6654efba22f003f645e2f4dffffad1b5a22770d3437b3027b`; [versioned dataset](https://doi.org/10.6084/m9.figshare.29949230.v1). Counts were obtained by reading ZIP member names and grouping XYZ members under each molecule's `SteepestDescent/` and `Conformations/` directories. The archive was inspected in memory and not retained during this check.

**Paper relevance:** the methods and limitations should distinguish many correlated frames from few connection groups. Connections may share minima or endpoint pairs; 46 groups is not proof of 46 statistically independent observations. Repeated model seeds do not create new molecular connections.

**Next check:** save the complete per-record inventory and manifest, verify connection-to-TS identity, and quantify eligible coverage after overlap screening.

### DATA-002: atom-order alignment can misclassify a branch destination

**Status:** numerical sample observation, independently reproduced by the parent and a scientific-review subagent. Full matching implementation remains unverified.

For aspirin connection `01`, rigid alignment of `coordsminus_122.xyz` with fixed atom indices against `coordsmin_1.xyz` gives all-atom RMSD **0.688446809856698** in raw coordinate units. Reordering the reference so zero-based indices **7 and 9** exchange places, with every other index unchanged, and then aligning gives **0.0010909448463713434**. These are oxygen atoms, corresponding to one-based atom rows 8 and 10. The fitted orthogonal transformation is recorded by the reproduction method; the full matcher permits both determinant signs.

For context, the same fixed-index alignment of the aspirin plus endpoint `coordsplus_110.xyz` to `coordsmin_0.xyz` gives RMSD **0.003096956893101937**. The much larger fixed-index minus score can therefore arise from atom mapping, rather than a different reference conformation.

**Method:** read XYZ positions, subtract each structure's centroid, compute the singular value decomposition of `A.T @ B`, and minimize the all-atom squared-distance sum over the rigid transformation. RMSD is `sqrt(mean(sum((A @ R - B)**2, axis=1)))`. The close match was reproduced with NumPy 2.0.2. Coordinate units are provisionally expected to be angstroms from the source conventions; formal archive unit tracing is still a model-ready export gate.

**Chemical check:** under the proposed distance-derived graph rule, edges have distance greater than 0.4 and at most 1.2 times the summed covalent radii, using H 0.31, C 0.76, N 0.71, and O 0.66 in the same provisional coordinate units. Each structure has 21 edges, and its adjacency matrix agrees exactly after the stated mapping. This supports the mapping under that rule; it is not independent bond-order validation or proof of every archive match.

**Evidence files and SHA-256:**

| Repository-relative file | SHA-256 |
| --- | --- |
| `data/landscape17_sample/aspirin/SteepestDescent/01/coordsminus_122.xyz` | `0236d1fda0447bf56a77bdc5ef59d0b967842d980bacf73752cf499c09af1063` |
| `data/landscape17_sample/aspirin/Conformations/coordsmin_1.xyz` | `3d6c27301862b836e7ace6259d42eef0c6a18f5f99a8537f5709d7778c5d75f1` |
| `data/landscape17_sample/aspirin/SteepestDescent/01/coordsplus_110.xyz` | `1c798d7c46c52281db8824c6c5d1bb2406342b5ab86b23e183e4e7374b80ca9a` |
| `data/landscape17_sample/aspirin/Conformations/coordsmin_0.xyz` | `8a2c7abacc04958b3b674b8412e7ab8be8a6ec5e57c0abf789c05f2b50060e7a` |

**Paper relevance:** motivate species- and connectivity-aware permutation/rotation/inversion matching in the evaluation methods. An index-only success metric could falsely record a wrong endpoint. This example is a preprocessing observation, not a result about a trained potential and not a new claim that symmetry-aware matching was invented here.

**Next check:** establish units, validate the full deterministic graph/isomorphism matcher, quantify how often this issue occurs, and save all competing reference-class distances. Reference identity is an equivalence class, so several valid atom mappings to one class do not make an endpoint ambiguous.

### DATA-003: a saved branch endpoint need not have negligible force

**Status:** directly computed from raw sample forces, before training.

The maximum absolute stored force components at connection `01` terminal frames are:

| Molecule / branch | Final frame | Maximum absolute force component, raw stored units |
| --- | --- | ---: |
| Aspirin plus | `coordsplus_110.xyz` | 0.00625404 |
| Aspirin minus | `coordsminus_122.xyz` | 0.00270845 |
| Paracetamol plus | `coordsplus_071.xyz` | 0.00153362 |
| Paracetamol minus | `coordsminus_092.xyz` | 0.00508297 |

The sampled aspirin branches contain 111/123 frames; paracetamol branches contain 72/93. Each sampled branch has no upward step in its stored energies under a `1e-10` raw-unit comparison tolerance. This concerns four sampled branches, not every archive path.

**Evidence:** the listed XYZ files and their `forces:R:3` header fields, with the unchanged sample checksums. Method: maximum over the absolute values of all atom force components; numerical frame indices establish traversal order.

**Paper relevance:** identify destinations geometrically rather than assuming the final saved frame is an exactly converged stationary point. The [source paper](https://arxiv.org/html/2508.16425) describes approximate descent paths; our later claim concerns recovery of the archived reference destination under a specified optimizer, not an optimizer-independent basin assignment.

**Limit:** force units have not yet been formally traced from archive serialization. Do not present these four numbers as eV/angstrom results or infer that reference minima violate their convergence criteria. These files are descent frames, distinct from the reference minimum records.

**Next check:** audit all branches, verify units and destination mappings, and record endpoint force distributions separately from stationary-point force distributions.

### DATA-004: source integrity is verified; scientific suitability is still being audited

**Status:** direct checksum verification, independently repeated.

All **405** XYZ files listed by `data/landscape17_sample/SHA256SUMS` match their recorded SHA-256 values. The manifest itself hashes to `0bf5d89d62b11d9ff64dd9071bef5f92cec1f38776506727fc3c41fc385fe07c`. The sample contains 399 descent frames and six reference structures, not a training split.

**Paper relevance:** include versioned source and integrity checks in the reproducibility material. Checksums establish that files match the frozen sample; they do not establish unit correctness, absence of leakage, or accurate chemistry.

**Next check:** carry the source digest into prepared arrays, split manifests and every run record.

## 29 September 2026: review decisions to preserve in the methods

### METHOD-001: validation weighting must be fixed before comparing force models

**Status:** design decision, not a measured finding.

Path lengths vary, so pooled component MAE and the equally weighted mean of connection MAEs answer different questions. Primary validation includes eligible branch frames and associated transition states. We will report both aggregates, select pilot checkpoints by equal-connection mean force-component MAE, and retain that convention across the later force-model arms. Validation traverses every eligible frame exactly once, including partial batches. Reference-minimum errors are unique-record diagnostics and do not select checkpoints.

**Relevance:** record this convention before any validation-guided checkpoint selection or matched-MAE comparison. Shared endpoints still create dependence; equal weighting does not by itself solve the statistical problem.

### METHOD-002: energy scaling must preserve the force derivative

**Status:** mathematical implementation requirement, not an experimental finding.

The model predicts an energy residual in physical eV, with the training-only energy offset restored for reporting. Forces come from the negative coordinate derivative of this physical residual. Loss scales apply afterward. Independent energy and force standardization without the chain-rule conversion would change the relationship being tested. Centering the large raw energies before precision reduction avoids unnecessary loss of relative-energy information.

**Relevance:** preserve the exact offset, units, loss scales and dtypes in run manifests; reverify any later precision change. The first pilot uses explicitly asserted float64.

### HYPOTHESIS-001: consistency training may improve destination recovery

**Status:** untested submitted hypothesis.

At comparable validation force error and measured cost, increasing energy-force consistency may improve recovery of reference destinations from held-out connections. No current entry supports a positive or negative answer. That conclusion requires the later controlled runs, paired starts, uncertainty analysis and prior-art comparison.

### METHOD-003: near-duplicate exclusion does not imply identical labels

**Status:** mathematical distinction identified during both implementation-plan reviews; no contradictory raw label has been demonstrated.

Two structures can be close enough to exclude one from a held-out score, while still having legitimately different forces. For example, a hypothetical conservative potential `E(x)=k*x^2/2` gives `F(x)=-k*x`, so a displacement `delta` changes the force by `-k*delta`. A small displacement can produce a force change above serialization precision if the local curvature is large.

We will distinguish exact stored-coordinate label consistency from near-duplicate leakage screening. Near pairs keep their measured energy/force differences for diagnosis, without being declared corrupted solely from that difference. This policy must be frozen before inspecting discrepancies.

**Relevance:** the methods should explain both the geometry threshold and the scope of label checks. A stricter leakage exclusion rule is not evidence that two reference calculations should return exactly equal forces.

### DATA-005: retained archive inventory and source-grounded units

**Status:** full selected archive parsed and source conventions traced during implementation; no independent DFT recalculation.

The retained, checksum-verified archive parses into **8,874 records**: 8,813 path frames, 15 minima, and 46 transition states. Every selected connection has both contiguous numeric branch sequences and one uniquely resolved reference TS filename. This verifies source structure, not endpoint geometry or leak-free splits.

`manifests/landscape17-unit-evidence.json` records the angstrom/eV/eV-per-angstrom assignment, identity conversion factors, versioned author code and frozen source hashes. TopSearch release 0.0.3 (commit `01e29d35c025bc3494db913f4dee526e82da4f9e`) returns ASE potential energy and forces. The ASE Psi4 calculator converts Hartree energies and Hartree/Bohr gradients to ASE units, while the published Figshare methods specify eV/angstrom force criteria. Snapshot files are under `reports/source_evidence/`.

**Correction to the earlier provisional-unit status in DATA-002/DATA-003:** the implementation now uses this source-grounded unit assignment, so the listed RMSDs are interpreted in angstroms and listed raw forces in eV/angstrom. The archived XYZ headers still lack explicit unit fields; the private export script was not independently inspected, and no DFT label was recomputed. Those limits remain part of the provenance.

**Additional source caveat:** the Figshare v1 metadata explicitly states that Hessian eigenvalues require multiplication by **9442.713128494901** to obtain eV/angstrom squared. Hessians are outside this baseline pipeline; future Hessian analysis must retain that conversion rather than silently reusing the stored values.

**Next check:** finish full-archive symmetry matching, duplicate screening and eligible connection coverage before model-ready export.

### DATA-006: one paracetamol branch fails the frozen destination criterion

**Status:** full raw-data geometry audit, before any fitting. The declared 0.05 angstrom threshold was not adjusted after observing this case.

Of 92 branch endpoints, 91 have a unique accepted reference class: all 74 aspirin branches and 17 of 18 paracetamol branches. The remaining source is `Landscape17/paracetamol/SteepestDescent/01_2/coordsplus_138.xyz`. Its closest chemically aligned reference is minimum 1, at RMSD **0.06431211546693098 angstrom**, above the frozen preparation threshold. Its maximum force component is **0.01034903 eV/angstrom**, with energy **-14023.35601746059 eV** under the source-grounded unit assignment.

Against minima 0/1/2/3, complete chemically constrained orthogonal-alignment RMSDs are **0.34597055528879134 / 0.06431211546693098 / 1.4095231220018807 / 1.3724159316498081 angstrom**. This is an unmatched archive endpoint under our rule, not evidence of a learned-model failure, a new minimum, or corrupted labels.

The complete duplicate screen found **zero** accepted pairs within **1e-5 angstrom**, with no incomplete candidates or exact-label conflicts. This narrow threshold does not establish absence of broader geometric similarity or correlated frames.

**Evidence:** checksum-verified `data/Landscape17.zip`; `reports/landscape17-endpoints.json` and `reports/landscape17-duplicates.json`; `configs/preparation.json`; the unit manifest. Matching uses full graph-isomorphism enumeration and orthogonal alignment; the candidate screen is mathematically conservative. Both prepared arrays and matching code still require independent implementation review.

**Consequence:** paracetamol model-ready export remains blocked; do not delete this connection or relax the threshold to obtain a passing audit. Aspirin can be prepared independently if all its own gates pass, since the first baseline pilot is aspirin-only. A separate reviewed decision is needed to resolve the paracetamol destination contract before its replication experiments.

### DATA-006A: complete-score provenance supplement

The complete (unshortlisted) paracetamol 01_2 branch/reference comparisons quoted in DATA-006 are now retained in `reports/landscape17-paracetamol-01_2-complete-endpoint-check.json`, including source hashes, full alignment status, and all reference scores. The general endpoint report records null for references rejected by the conservative descriptor screen; it should not be cited as the source of an uncomputed full alignment score.

### DATA-007: geometric destinations can differ from filename endpoint pairs

**Status:** full-archive data observation under the frozen chemically constrained matching policy; before fitting.

For aspirin, connection directory `44` uniquely associates with TS filename `coordsts_4_4.xyz`, but its archived branch terminals match references 4 and 10. Connection `66` associates with `coordsts_6_6.xyz`, while the branch terminals match references 6 and 9. All four destinations satisfy the frozen 0.05 angstrom identification criterion. Every other fully matched connection's geometric pair agrees with its named endpoint pair in this check.

**Evidence:** `reports/landscape17-aspirin-44-complete-endpoint-check.json`, `reports/landscape17-aspirin-66-complete-endpoint-check.json`, and the full endpoint report. Complete comparisons include source and reference hashes, distances, energies and forces. The grouping identity is unchanged: each directory still contributes both branches and its uniquely identified TS.

**Consequence:** keep the named pair as source metadata, but derive destination classes and minimum training/seen memberships from the verified branch geometry. Filenames alone are insufficient to establish branch destinations. This does not establish corrupted data, a new physical connection, or incorrect source-paper conclusions; the result is conditional on our matching and reference-class conventions.

## DATA-008 — Frozen aspirin preparation and reference coverage (29 September 2026)

Status: verified preparation, before fitting. Two independent invocations produced identical canonical manifest identity `4b9f220201d6483e2b2a8da74c10330c3762942ff4c6fabe6389ee27249c43aa` and decompressed-array identities; split hash `73d5475bfc1f1158da62b939f5431b1781db886443cece2cdb297c8dd645133a`. Paths: `data/processed/aspirin/manifest.json`, `data/processed/aspirin-repeat/manifest.json`, `manifests/landscape17-aspirin-split.json`. Assignment is PCG64(20260929), sorted string IDs, 26/5/6 groups. Eligible unique record counts: train 4891, primary validation 974, test 1127, validation-minimum diagnostics 0. All 11 reference minima belong to at least one training connection under the verified geometric classes. Their shared training memberships are retained; none may contribute to primary checkpoint selection. This split therefore measures held-out connection paths/TS, not unseen reference minima or independent molecular samples. No cross-partition duplicate exclusion was needed at the frozen 1e-5 A tolerance.

Training-only normalization: offset -17647.258992926312 eV, energy scale 0.21583988927697104 eV, force scale 0.1268863230978139 eV/A. Held-out-label perturbation tests leave normalization unchanged. Fitting loader opens train/validation/diagnostic arrays only. Paracetamol command failed before output creation on 01_2/plus; no paracetamol split or success manifest is emitted. See DATA-006/006A. These are preparation gates, not model results. Next check: float64 model physics and actual learning.

## IMPLEMENTATION-001 — Physical derivative checks before training (29 September 2026)

Verified, untrained E3x energy model. See `reports/baseline-verification.md`, `reports/baseline-physics.json`, `tests/test_model_physics.py`, and `configs/aspirin-baseline-pilot.json`. 4737 float64 parameters. Seed 19; central-difference errors [1.5030025307671835e-07, 9.454223792726424e-09, 8.784594329735995e-09] eV/A at h=[1e-3,1e-4,1e-5] A. O(3), translation/permutation, batch/single, net-force/torque, smooth-cutoff limits and force-loss parameter gradients passed fixed checks. Degree-one polar-vector features remain available in the shared backbone; only the energy head takes even scalars. This is an implementation check; it does not show force accuracy, generalization or comparative physical-consistency benefit. Next check: real fixed-training-subset learning and capped validation pilot.

## ENGINEERING-001 — First tiny-subset learning gate failed (29 September 2026)

Status: incomplete learning acceptance; full pilot not started. Saved run: `runs/aspirin-learning-check-seed0`, exact subset IDs in `learning-subset.json`, configuration in `config.json`, final checkpoint `last.msgpack`, full history in `validation.json`/`losses.json`. Summary SHA-256 `c435ffd94b77a243876dae289ecd5201407c1d17237f76c041315d5abf62770d`. Seed0, Adam1e-3, batch4,16numericallyeven frames from first lexicographic training connection01; all frozen source/preparation/physics gates passed. 2000updates in 26.59973204205744 CPU seconds, stopped at the predeclared update cap. Initial/final energy MSE 32.23256077457698/0.007751758500612726 eV^2. Initial/final force-component MSE 0.0435718463614834/0.009779336686872274 eV^2/A^2. Force fraction remaining 0.2244416407269122, above the unchanged0.10 acceptance criterion. This failure is retained and cannot be replaced with a selected successful seed or relaxed metric. A new process reproduced the final checkpoint.

Focused diagnostics are saved in `reports/learning-check-diagnostics.json`: finite nonzero supervised gradients, target net force/torque and adjacent-source energy/work checks. They diagnose code/data scaling; they are not a model-comparison result or proof of source correctness. Budget/capacity/optimizer explanations remain hypotheses until tested. Two independent reviewers are checking the failure before any material plan amendment.

## ENGINEERING-001A — Baselines and reviewed prospective initialization test

Correction/supplement to ENGINEERING-001: the first final force MSE is only3.4956% better than the zero-force baseline0.010133561081129106 eV^2/A^2. Reduction from a random initialization is not evidence of strong force fitting. Both independent reviewers reproduced this and checked raw/prepared label correspondence. Small nonzero reference net-force/torque components have projected MSE floor about7.75e-9, far below the failed gate. The adjacent source work correlations>.995 support sign/scale but do not independently validate DFT.

Both reviewers approved the exact prospective `docs/superpowers/plans/2026-09-29-learning-gate-amendment.md`. Exactlyone zero-readout attempt preserves seed, fixed16IDs, backbone, loss/optimizer, caps and10% criterion. It tests initialization conditioning; causality is not assumed. Production zero initialization and deliberately nontrivial physics state are separately verified; no numerical tolerances are loosened. Full-subset history and simple baselines will be recorded. Learning resume is disabled because resetting initial acceptance metrics would redefine the gate. GPU dependencies are now in the lock, but GPU execution is unverified. Next check: the one prospective run, or report the gate still failed without further tuning.

## ENGINEERING-002 — Prospective zero-readout attempt also failed (29 September 2026)

Linked follow-up to ENGINEERING-001/001A; executed only after both independent reviewers cleared `docs/superpowers/plans/2026-09-29-learning-gate-amendment.md`. Run `runs/aspirin-learning-check-zero-readout-seed0`; summary SHA-256 `22333ac423ae0e031d09cd5837fcbd4294226aeb0e83d5372b3378d80e9adc3c`. Same seed0, same16IDs/order and subset hash338e75a921b30820b70cfba47d1f06e7356a6d2232905d88975d088844403a3c, same backbone/loss/Adam/batch/caps. Only the energy readout initializer changed. Initial/final energy MSE0.061298358531283896/0.0004897094577467109 eV²; force MSE0.010133561081129104/0.0075721620581139844 eV²/A². Force fraction remaining0.7472360404690309 exceeds unchanged0.10 criterion. Own initial force error equals the zero-force baseline. Improvement over that baseline25.276396% remains inadequate for the reviewed gate.

2000updates in26.417110207956284 CPU seconds; max-update stop; new-process checkpoint predictions reproduced. Full-subset MSE recorded every100updates and full primary validation retained. See `reports/baseline-learning-checks.json` for precise artifact hashes/config/provenance/metric histories and `reports/baseline-pilot.md` for scope. This one attempt supports that zero readout did not resolve the gate; it does not prove a capacity/budget/conditioning cause or a scientific benefit. The full pilot remains pending under the reviewed stop condition. Next step requires a new reviewed diagnostic plan; no further tuning or test evaluation was performed.
