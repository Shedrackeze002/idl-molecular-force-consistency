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
