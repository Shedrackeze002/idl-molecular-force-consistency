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

## IMPLEMENTATION-002 — Final review regression trials (29 September 2026)

Scope: code correctness, not additional molecular learning. The fresh whole-branch reviewer independently ran53tests and identified two important gaps: resumed fitting forgot validation-selection history/best state, and a methane symmetry alignment could conceal conflicting labels on exactly identical stored coordinates. Both parent regressions reproduced failures before fixes: `test_resumed_selection_keeps_earlier_best_and_exact_tie` selected update3 instead of earlier update1; `test_symmetric_exact_positions_do_not_hide_conflicting_labels` classified identical coordinates as merely aligned. Two failures in14.47s. Exact fixtures are in the committed tests; this is an implementation diagnostic, not a Landscape17 corruption finding.

Fix trial: independently identify exact species/coordinate/graph-preserving permutation for hard label comparisons; preserve threshold behavior for near geometries. Persist integrity-checked validation history, historical best score/update and corresponding best state in the checkpoint envelope; restore it and earlier-tie selection on resume. No research run, architecture setting or learning threshold changed. Regression verification and full-suite result will be appended below. Both recorded real learning checks were fresh runs, so the resume bug did not cause their failures. Landscape17 duplicate audit had zero accepted pairs, so the exact-symmetry bug does not change those observed counts. Next check: passing regressions/full suite and complete duplicate re-audit.

## IMPLEMENTATION-002A — Regression fixes and source rechecks passed

Verification supplement to IMPLEMENTATION-002: both failures became passing regressions (2 passed in 15.96 s), followed by the full suite (55 passed in 38.59 s). The full duplicate audit on all 8,874 selected records remains zero accepted pairs, zero unresolved comparisons and zero exact-label conflicts. A fresh post-review aspirin preparation reproduces canonical identity `4b9f220201d6483e2b2a8da74c10330c3762942ff4c6fabe6389ee27249c43aa`; frozen split/physical arrays/normalization remain unchanged. Source checksum and thresholds are unchanged. Full fixes/review and execution trials are retained in `docs/reviews/2026-09-29-final-implementation-review.md` and `reports/implementation-execution-record.md`.

No additional research learning trial was run. Both failed fixed-subset attempts remain tied to their original code/lock/config identities; current corrected code has not been benchmarked by a new molecular run. Existing checkpoints retain inference compatibility; new fitting resume requires saved selection controls rather than resetting historical best selection. Full pilot remains pending, paracetamol remains blocked, and GPU execution remains unverified.

## METHOD-004 — Balanced experimental trail (user reaffirmation, 29 September 2026)

Standing rule: document every experiment/trial whether it succeeds, fails, is blocked or abandoned. Before execution, save rationale, fixed settings, controls, acceptance rule and reviewed plan/configuration; append actual outcomes, stop reasons, hashes, limitations and next checks afterward. Preserve all attempts and distinguish proposals from executed trials. Earlier entries plus the saved pre-execution plans/configurations trace the two learning checks and their original failures; this rule makes the future before/after log explicit. Implementation/regression trials are recorded as code diagnostics, never converted into scientific model-comparison evidence. Obsidian is current-state/navigation memory; this repository log is authoritative for paper claims.

## DIAGNOSTIC-001 — Prospective force-learning diagnosis (30 September 2026)

Status: planned, not executed; two independent reviews pending. User authorized proceeding after discussion of both failed gates. Exact prospective plan: `docs/superpowers/plans/2026-09-30-force-learning-diagnostics.md`; fixed settings and historical artifact SHA256s: `configs/aspirin-force-diagnostics.json`. Existing source data, split, 16 training IDs, normalization, model size, seed0, Adam1e-3, 2000-update/1800-second cap and 10% own-initial-MSE rule preserved. No held-out arrays will be opened.

A: conditional linear readout fits on fresh and historical final backbones, force-only and original normalized joint objective, fixed rcond1e-10; numerical reconstruction required. B: one exactly representable teacher control, initial student backbone shared with teacher, seed23 kernel Normal(0,.1), centered energy, original loss scales, all16 examples per update. C: one original-label full-batch control only if B passes. Each has unknown outcome and an explicit failure/block trail. These tests distinguish hypotheses; they do not establish comparative model accuracy, DFT correctness, generalization or satisfaction of the original batch4 learning gate. Full pilot remains pending. Append review decisions before implementation and outcomes as each check finishes.

## DIAGNOSTIC-001A — Reviews cleared corrected prospective protocol

Both independent reviews are saved in `docs/reviews/2026-09-30-diagnostic-plan-review.md`. Scientific follow-up explicitly cleared the corrected plan. Per-atom head bias uses mean molecular residual divided by atom count; rank-truncated readout solves are not whole-network capacity bounds. Engineering safeguards add nonzero-head reconstruction on both bodies, a holdout-free new-process probe, and explicit batch16. No research trial has run yet. Implementation tests below are engineering trials with red/green outcomes recorded separately.

## IMPLEMENTATION-003 — Diagnostic implementation trials

Before molecular execution: new loader/readout tests first failed4/4 because interfaces were absent. After implementation,9/10 passed; remaining failure was an in-memory tuple-vs-JSON-list fixture comparison, corrected without changing loader behavior. Task1 verification10/10 passed. New controller tests first failed5/9 because functions were absent; after implementation12/12 passed in25.31s, including weighted objective, force signs/axes, atom bias, teacher degeneracy/centering, immutable arrays, cap/history, nonfinite failure summary, blocked real run, historical hash/model/normalization gates and full CLI/new-process probe with all held-out files deleted. A synthetic test checkpoint uses a declared update-counter fixture, not a molecular training result. Full suite and exact production config preflight follow before experiments. Failed test traces retained in this plan's execution workspace and summarized here.

## IMPLEMENTATION-003A — Diagnostic regression suite passed

Full suite67/67 passed in57.25s before any molecular diagnostic execution. Existing55 tests remain passing;12 new tests cover the reviewed diagnostic contracts. No existing model/loss settings or preparation identities changed. The next invocation is the exact prospective CPU config; outcome remains unknown.

## DIAGNOSTIC-001A — Frozen readout trial preregistered (30 September 2026)

Training-only engineering diagnosis; full pilot remains pending.

```json
{
  "config_hash": "b6c55a5a5aa40d7e30d0423f953793c635c7b2d433e6ebac59ad3cada906d556",
  "controls": "fresh and historical backbones, joint and force-only",
  "rcond": 1e-10,
  "run_directory": "runs/aspirin-force-diagnostics-2026-09-30"
}
```

## DIAGNOSTIC-001A — Frozen readout trial completed (30 September 2026)

Training-only engineering diagnosis; full pilot remains pending.

```json
{
  "artifact": "runs/aspirin-force-diagnostics-2026-09-30/readout.json",
  "sha256": "6f760b0f374d33e090fb3f54c40948a27434df7e990c3c1487d1caebda3d4237",
  "solutions": {
    "fresh": [
      {
        "coefficient_l2": 3.3285634544307707,
        "energy_mse_eV2": 0.009370411512161263,
        "force_mse_eV2_per_A2": 0.008237971281002272,
        "mode": "force_only",
        "rank": 16
      },
      {
        "coefficient_l2": 5.043436835152683,
        "energy_mse_eV2": 0.0007420096271186995,
        "force_mse_eV2_per_A2": 0.00879886195790133,
        "mode": "joint",
        "rank": 17
      }
    ],
    "historical_final": [
      {
        "coefficient_l2": 0.9630752766324349,
        "energy_mse_eV2": 0.0024365124422744406,
        "force_mse_eV2_per_A2": 0.007373451310725372,
        "mode": "force_only",
        "rank": 16
      },
      {
        "coefficient_l2": 0.5784706646425077,
        "energy_mse_eV2": 0.00033510516512580505,
        "force_mse_eV2_per_A2": 0.007501567117503857,
        "mode": "joint",
        "rank": 17
      }
    ]
  }
}
```

## DIAGNOSTIC-001B — Representable teacher trial preregistered (30 September 2026)

Training-only engineering diagnosis; full pilot remains pending.

```json
{
  "batch_size": 16,
  "config_hash": "b6c55a5a5aa40d7e30d0423f953793c635c7b2d433e6ebac59ad3cada906d556",
  "gate_fraction": 0.1,
  "kernel_std": 0.1,
  "max_updates": 2000,
  "seed": 23
}
```

## DIAGNOSTIC-001B — Representable teacher trial outcome (30 September 2026)

Training-only engineering diagnosis; full pilot remains pending.

```json
{
  "accepted": true,
  "failure_reason": null,
  "final_mse": {
    "energy_mse_eV2": 7.629829221195148e-06,
    "force_mse_eV2_per_A2": 2.0318703322910553e-05
  },
  "fraction_remaining": {
    "energy_mse_eV2": 0.08482084275536658,
    "force_mse_eV2_per_A2": 0.009713584025373742
  },
  "initial_mse": {
    "energy_mse_eV2": 8.995229207048184e-05,
    "force_mse_eV2_per_A2": 0.002091782319464598
  },
  "status": "passed",
  "stop_reason": "learning_acceptance",
  "updates": 300
}
```

## DIAGNOSTIC-001C — Real full-batch trial preregistered (30 September 2026)

Training-only engineering diagnosis; full pilot remains pending.

```json
{
  "batch_size": 16,
  "config_hash": "b6c55a5a5aa40d7e30d0423f953793c635c7b2d433e6ebac59ad3cada906d556",
  "gate_fraction": 0.1,
  "max_updates": 2000,
  "subset_hash": "338e75a921b30820b70cfba47d1f06e7356a6d2232905d88975d088844403a3c"
}
```

## DIAGNOSTIC-001C — Real full-batch trial outcome (30 September 2026)

Training-only engineering diagnosis; full pilot remains pending.

```json
{
  "accepted": false,
  "executed": null,
  "failure_reason": null,
  "final_mse": {
    "energy_mse_eV2": 0.00021554077745390633,
    "force_mse_eV2_per_A2": 0.007240959200598679
  },
  "fraction_remaining": {
    "energy_mse_eV2": 0.003516256921364446,
    "force_mse_eV2_per_A2": 0.7145522825221747
  },
  "initial_mse": {
    "energy_mse_eV2": 0.061298358531283896,
    "force_mse_eV2_per_A2": 0.010133561081129106
  },
  "status": "incomplete",
  "stop_reason": "max_updates",
  "updates": 2000
}
```

## IMPLEMENTATION-004 — Figure-generation failure preserved

First postprocessing attempt wrote the compact diagnostic JSON but could not plot because the original random-head run predates saved full-subset `learning-history.json`. Failure: FileNotFoundError for that original run path. No molecular trial was rerun, altered or extended. Correction: show that run's verified initial/final full-subset points only, using its saved summary, while plotting the two available histories normally. Never substitute minibatch losses or invent intermediate full-subset metrics. Figure source hashes are saved in the compact JSON; regenerated report/figure verification follows. This is an artifact-generation trial, not a model-learning result.

## DIAGNOSTIC-002 — Fixed-feature force fits remain insufficient

Executed as preregistered on the original16 training frames; fixed rcond1e-10. Fresh/historical force-only MSEs0.008237971281002272/0.007373451310725372 eV²/A², ranks16/16. Joint MSEs0.00879886195790133/0.007501567117503857, ranks17/17. Both actual and nonzero-head reconstructions passed, solved-head maximum error<5.50e-15. These conditional readout solutions exceed the zero-head gate0.0010133561081129106; they do not bound differently trained backbones. See tracked `reports/force-learning-diagnostics.json` for singular values/coefficients/full identities. Earlier automatically generated DIAGNOSTIC-001A trial headings refer to the frozen-readout preregistration/outcome; the separately titled DIAGNOSTIC-001A review entry is the review supplement. Use this unique DIAGNOSTIC-002 record for paper citations to the readout result.

## DIAGNOSTIC-003 — Exactly representable teacher passed

Known in-class synthetic teacher shares initial student backbone, seed23 readout Normal(0,.1), centered molecular energy; original scales/seed0/Adam/fullbatch16. Passed at300updates, not a budget extension. Energy MSE8.995229207048184e-05 ->7.629829221195148e-06; force MSE0.002091782319464598 ->2.0318703322910553e-05. Fractions0.08482084275536658/0.009713584025373742 both<.10. CPU10.746022709063254s, fresh-process max prediction difference2.1510571102112408e-16. Pipeline can learn this deliberately easy target; this is not DFT correctness, generalization or a scientific comparison. Teacher pass enabled exactly one original-label full-batch control.

## DIAGNOSTIC-004 — Real full-batch force criterion still failed

Same original16IDs/seed0/zero head/backbone/Adam1e-3/loss scales, only sampling changed from batch4 to all16. Stopped at2000-update cap; no retry. Energy MSE0.061298358531283896 ->0.00021554077745390633; force MSE0.010133561081129106 ->0.007240959200598679. Force fraction0.7145522825221747>.10, improvement28.544771747782526% over zero forces. CPU34.36673712497577s; new-process max difference8.881784197001252e-16. This one sampling change did not resolve acceptance and does not retroactively clear batch4 learning. Representation/nonlinear optimization remain hypotheses; full pilot stays pending. No held-out array, extra seed, larger architecture or later scientific arm was evaluated.

## IMPLEMENTATION-004A — Diagnostic artifacts verified

Figure-generation correction succeeded: original random-head initial/final points only, two available molecular histories and separate teacher panel. PNG visually inspected; PDF also saved. All compact-record run and figure-input SHA256s checked against retained artifacts, and both checkpoint outcome/counter checks passed. Compact JSON SHA256 `2165ffb58ed9768b20b00b1c05d18dd73db3ddf9ff813c2dc0b31cab4aa73e54`. Report `reports/force-learning-diagnostics.md`, figure `reports/figures/force-learning-diagnostics.png`. Outcomes are validly executed diagnostics with real acceptance failed, not a full-pilot success.

## IMPLEMENTATION-005 — Fresh final diagnostic review cleared implementation

Independent gpt-6-astra review identified no critical, important or minor defect. It independently passed12 diagnostic tests in24.56s and matched22run artifact hashes,3figure-input hashes and generator identity. Parent's earlier full suite67/67 remains the full-suite verification; reviewer did not independently rerun all67. Both controls' update counters and outcomes independently matched saved evidence. Full review, explicit exclusions and parent rulings are in `docs/reviews/2026-09-30-diagnostic-final-review.md`. Parent visually inspected the corrected figure; GPU/DFT generalization/capacity/cause remain unverified or unresolved. Budget is checked between operations, not guaranteed by preemptive interruption. No new molecular trial or full pilot followed review.

## METHOD-005 — Retain review folders and logs

User instruction, 30 September 2026: leave review folders in place; they help preserve review history. The diagnostic scratch workspace remains retained, alongside its durable execution archive. `AGENTS.md` now instructs future work to preserve review packages, logs and ledgers after completion and avoid automatic cleanup. This is a workflow preference, not a new experiment or change to reported outcomes.

## HYPOTHESIS-001 — Training-set quantity versus diversity (30 September 2026)

User question: would increasing the number of examples help? Current source subset was rechecked: all16 path frames come from aspirin connection01, eight from each descent branch; subset identity `338e75a921b30820b70cfba47d1f06e7356a6d2232905d88975d088844403a3c`. This narrow geometry coverage is a limitation of the engineering diagnostic, not a newly measured explanation for its failure. More varied training connections might improve nonlinear fitting, but no experiment has established that here. The observed failure is on training examples, so extra examples alone are not a guaranteed remedy.

The general distinction between poor training fit and data-limited generalization is supported by Goodfellow, Bengio and Courville, Deep Learning, chapter11 sections11.3/11.5: https://www.deeplearningbook.org/contents/guidelines.html . That guidance is not a molecular-force benchmark or proof about this model.

Proposed, unexecuted diagnostic: compare narrow versus diverse training-only subsets before attributing the failure to model capacity; separately examine a larger diverse subset (illustratively128 examples) with matched model/loss settings, explicit exposure/compute accounting and evaluation on the unchanged original16 alongside the expanded subset. Exact sampling, subset identities, budgets and acceptance rules need a saved plan and two independent reviews before execution. Keep held-out arrays unopened and preserve the original failures. No larger-data run or new scientific result occurred in this discussion.

## PLAN-COVERAGE-001 — Prospective data-diversity and quantity controls

Status: saved plan, unexecuted; two independent reviews pending. User requested a next-step plan following HYPOTHESIS-001. Spec `docs/superpowers/specs/2026-09-30-data-coverage-diagnostic-design.md`; implementation plan `docs/superpowers/plans/2026-09-30-data-coverage-diagnostic.md`; fixed candidate settings `configs/aspirin-data-coverage-diagnostic.json`. Planned subset manifest identity `8139e442fca011efcbeb5f2d28d701c5f36030d61f045e5e652b98eb18b1be47` includes exact IDs/indices/hashes for original16, diverse16 and nested diverse128.

Label-blind planning preflight read prepared metadata and original IDs only, with PCG64(20260930), fixed support01/23/77/58_2/44/22/78_2/01_2, rational half-up transfer of original within-branch ordinal ranks. All selected branches have sufficient distinct ranks; subset sizes16/16/128 verified, both16sets nested in128. No labels were fitted, no model initialized, no product code changed and no training run occurred. This preflight verifies proposed selection feasibility, not learned performance.

Plan holds model/loss/scales/seed/Adam fixed; all arms use deterministic full-subset updates, with128 accumulated over8microbatches then one Adam update. Fresh original16 replay guards source/optimizer consistency before other arms. Fitting and unchanged original16 gates are separate; diverse16 fits only2anchor frames and128 fits all16. Computational cost and record difficulty differ, so outcomes cannot establish pure sample-count causality or scientific generalization. Per-arm2000updates/1800s, no learning early stop or tuning. Full pilot remains pending regardless of this planning preflight. Append reviewer decisions before implementation and each prospective/actual trial if later executed.

## PLAN-COVERAGE-002 — Both independent plan reviews cleared

Scientific and engineering reviewers independently verified the proposed subset selection, canonical identities, frozen replay artifacts and comparison controls. Both found no material blocker. Decisions/limitations are saved in `docs/reviews/2026-09-30-data-coverage-plan-review.md`. Engineering clarifications incorporated: matching initial parameter/optimizer hashes across arms, sequential microbatch accumulation, and first observed passing scheduled update rather than an exact crossing time. IDs, budgets, gates and settings are unchanged. No implementation or training started; all prospective outcomes remain unknown. This review completion is not learning success or full-pilot authorization.

### IMPL-COVERAGE-001 — Frozen subset implementation verification (2026-09-30)
- Type: engineering check, not a trained-model finding. User authorized unchanged reviewed plan execution.
- Tests first failed because the new coverage module was absent; implementation then passed all 8 coverage/preparation tests in 0.57s. Retained logs: `.superpowers/sdd/2026-09-30-data-coverage-diagnostic/task1-{red,green}.log`.
- Runtime selection exactly regenerates frozen manifest `8139e442fca011efcbeb5f2d28d701c5f36030d61f045e5e652b98eb18b1be47`; guards cover duplicate/outside/mixed-role paths, numeric ordering and rank collisions, with no replacement groups. Arbitrary label perturbation leaves selection unchanged.
- No training run executed yet; optimizer equivalence remains the next required check.

## IMPL-COVERAGE-002 — Gradient and run-control engineering checks

The implementation now shares the original normalized loss without changing its mathematics. Unequal3/3/1 microbatches match full-batch loss, both MSEs, every gradient leaf and two successive Adam parameter/optimizer states within1e-10 absolute/relative tolerance. Nonzero force gradients reach the backbone after a nonzero head update. Sequential accumulation holds incoming parameters fixed and applies Adam once.

All28 coverage/training-contract/previous-diagnostic tests passed in74.20s. New fixtures verify fixed caps despite earlier scheduled gate passage, immutable initial values, fitted/not-fitted anchor scoring, empty null gates, wall/nonfinite failure retention, identical initial parameter/optimizer hashes across subset shapes and all-frame saved-input fresh-process probes. Logs remain under `.superpowers/sdd/2026-09-30-data-coverage-diagnostic/`.

Failed engineering checks preserved: missing helpers in test-first runs; overly exact float ratio assertion (0.049999999999999996 vs0.05), corrected to a tolerance; controller integration fixture used a wrong dataclass field name, corrected after source inspection. These are implementation/fixture trials, not research failures. No new aspirin coverage research arm has run yet.

## IMPL-COVERAGE-003 — Matrix controller and full regression verification

All79 tests passed in101.94s before research execution (67previous tests plus12new coverage tests). Retained log: `.superpowers/sdd/2026-09-30-data-coverage-diagnostic/full-preproduction.log`. The synthetic2atom end-to-end fixture executed a one-update reference and all three one-update controls with real train-only preparation and all-frame fresh-process probes, after removing held-out arrays. Deliberately corrupted prepared forces then blocked all fitting. This is engineering verification, not an aspirin learning result. The stale-reference, missing/mismatched replay and incomplete/nonfinite blocking branches were tested separately. A finite failed learning gate permits C.

## COVERAGE-MATRIX-PRE — Approved fixed matrix preregistration

User authorized execution of the saved plan after both independent reviews. Exact configuration `configs/aspirin-data-coverage-diagnostic.json`; canonical frozen subset identity8139e442fca011efcbeb5f2d28d701c5f36030d61f045e5e652b98eb18b1be47. Source/hash/selection preflight regenerated16/16/128IDs and matched all frozen identities. Prepared/split/normalization, seed0, zero-head model, Adam1e-3, original normalized joint loss and float64 CPU stay fixed.

A is a new2000-update replay of the original16full-batch control, checked against every historical100-update metric. B uses16frames across8connections; C uses128frames across the same8connections, nested with A/B, one accumulated Adam update per full subset. Each arm has1800s including initialization/compile/warmup/scoring/reload, no learning early stop or resume. Fit and common-anchor gates each require both MSEs below0.10immutable initial values. A replay or any operational/short-budget failure blocks downstream; finite learning failure in B permits C. Actual outcomes unknown at preregistration. No extra seed, GPU, scientific arm, full pilot, longer cap, loss/capacity/LR change or test access is authorized by this matrix.

Limitations: one seed per condition; B changes fitted difficulty and overlaps only2anchors; C fits all16anchors and uses8times exposures per update. Save raw units, per-group/overlap/count baselines, scheduled ratios, synchronized timing, frame exposures, checkpoint/source hashes, all failures and blocks. Conclusions are training-only diagnostics, not causal/statistical/generalization evidence. Decision after C requires its BOTH gates; protocol amendment (if passed) or architecture/gradient investigation (if failed) must be separately planned and reviewed.

## COVERAGE-001-PRE — narrow16 trial preregistered (2026-09-30)

Training-only diagnostic; no scientific generalization claim; full pilot remains pending.

```json
{
  "anchor_gate": "Both final anchor MSEs < 0.1 fixed original initial",
  "config": {
    "backend": "cpu",
    "comparison_updates": [
      0,
      100,
      250,
      500,
      1000,
      1500,
      2000
    ],
    "data_directory": "data/processed/aspirin",
    "early_stopping": false,
    "evidence_log": "reports/PAPER_EVIDENCE_LOG.md",
    "expected_prepared_identity": "4b9f220201d6483e2b2a8da74c10330c3762942ff4c6fabe6389ee27249c43aa",
    "expected_split_hash": "73d5475bfc1f1158da62b939f5431b1781db886443cece2cdb297c8dd645133a",
    "expected_subsets_identity": "8139e442fca011efcbeb5f2d28d701c5f36030d61f045e5e652b98eb18b1be47",
    "gate_fraction": 0.1,
    "learning_rate": 0.001,
    "max_updates": 2000,
    "microbatch_size": 16,
    "model": {
      "cutoff_A": 5.0,
      "energy_readout_init": "zero",
      "features": 16,
      "iterations": 2,
      "max_degree": 1,
      "radial_basis": 8
    },
    "reference_run": "runs/aspirin-force-diagnostics-2026-09-30/real",
    "reference_sha256": {
      "checkpoint-verification.json": "24e231afcdd58e5f34a759bf974a87dbe9e01f6d45adffd9324c99b4aa240e7b",
      "config.json": "d8bfbe676f0e03f0de23bdc2b4b44610f3f5c8f21c67034cb31091b8765ba53f",
      "history.json": "13488ec9153c6a0a85b7256cc408e14986a78b9e5f66070f60d66234810961a9",
      "provenance.json": "4c5c05915e11250f3bddabae841469039232a6db03ac316074a8467ccb0913d9",
      "summary.json": "3bab8e9f66cc747dacb6744cca0b93f316fd8568962fe736c7480b7382f7a650"
    },
    "replay_tolerance": {
      "atol": 1e-10,
      "rtol": 1e-08
    },
    "run_directory": "runs/aspirin-data-coverage-2026-09-30",
    "seed": 0,
    "status": "planned_unexecuted",
    "subsets_manifest": "manifests/aspirin-data-coverage-subsets-planned.json",
    "validation_interval": 100,
    "wall_seconds": 1800
  },
  "fit_gate": "Both final MSEs < 0.1 own initial",
  "limitations": "n=1; B changes record difficulty; C uses 8x exposures per update; training-only anchor",
  "rationale": "Replay",
  "run_directory": "runs/aspirin-data-coverage-2026-09-30/narrow16",
  "subset": {
    "anchor_fit_overlap_ids": [
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_000.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_015.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_031.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_046.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_062.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_077.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_093.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_108.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_001.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_016.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_032.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_047.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_063.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_078.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_094.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_110.xyz"
    ],
    "identity": "338e75a921b30820b70cfba47d1f06e7356a6d2232905d88975d088844403a3c",
    "indices": [
      37,
      52,
      68,
      83,
      99,
      114,
      130,
      145,
      161,
      176,
      192,
      207,
      223,
      238,
      254,
      270
    ],
    "record_ids": [
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_000.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_015.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_031.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_046.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_062.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_077.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_093.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_108.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_001.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_016.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_032.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_047.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_063.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_078.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_094.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_110.xyz"
    ]
  }
}
```

## COVERAGE-001-RESULT — narrow16 trial outcome (2026-09-30)

Training-only diagnostic; no scientific generalization claim; full pilot remains pending.

```json
{
  "anchor_gate": {
    "accepted": false,
    "fraction_remaining": {
      "energy_mse_eV2": 0.003516256921364377,
      "force_mse_eV2_per_A2": 0.7145522825221748
    },
    "gate_fraction": 0.1,
    "threshold_mse": {
      "energy_mse_eV2": 0.00612983585312839,
      "force_mse_eV2_per_A2": 0.0010133561081129107
    }
  },
  "artifact_sha256": {
    "checkpoint-verification.json": "f39a1a3d6cacfc6eb6418e7a695bb550ac9350bc9e08cd2aa20d80ad256a6295",
    "history.json": "9970a673c501b566e8f8de0b83581ce81f40cba9a71a2547111f9fe46b0af812",
    "last.msgpack": "b5c792314ea6b34d46bc5b2b104abbc7b1e647fd6c81cea55ec49e94b18cb6b7",
    "summary.json": "c3b0aa387b7cbd2cc57723e6232677eea19abb3f03dc1834cd8cdf1b32128452"
  },
  "elapsed_seconds": 37.41735974990297,
  "execution_valid": true,
  "failure_reason": null,
  "fit_gate": {
    "accepted": false,
    "fraction_remaining": {
      "energy_mse_eV2": 0.003516256921364377,
      "force_mse_eV2_per_A2": 0.7145522825221748
    },
    "gate_fraction": 0.1,
    "threshold_mse": {
      "energy_mse_eV2": 0.00612983585312839,
      "force_mse_eV2_per_A2": 0.0010133561081129107
    }
  },
  "frame_exposures": 32000,
  "initial_state_hashes": {
    "optimizer": "8fc30b9a817f70448819b69c458e2e1cacc034cde02880066eab48edebd4f01d",
    "params": "b6351d17922c7646322cb85f142c9566dc5380b6ca5676898c157d92dd93becd"
  },
  "status": "completed",
  "stop_reason": "max_updates",
  "updates": 2000
}
```

## COVERAGE-002-PRE — diverse16 trial preregistered (2026-09-30)

Training-only diagnostic; no scientific generalization claim; full pilot remains pending.

```json
{
  "anchor_gate": "Both final anchor MSEs < 0.1 fixed original initial",
  "config": {
    "backend": "cpu",
    "comparison_updates": [
      0,
      100,
      250,
      500,
      1000,
      1500,
      2000
    ],
    "data_directory": "data/processed/aspirin",
    "early_stopping": false,
    "evidence_log": "reports/PAPER_EVIDENCE_LOG.md",
    "expected_prepared_identity": "4b9f220201d6483e2b2a8da74c10330c3762942ff4c6fabe6389ee27249c43aa",
    "expected_split_hash": "73d5475bfc1f1158da62b939f5431b1781db886443cece2cdb297c8dd645133a",
    "expected_subsets_identity": "8139e442fca011efcbeb5f2d28d701c5f36030d61f045e5e652b98eb18b1be47",
    "gate_fraction": 0.1,
    "learning_rate": 0.001,
    "max_updates": 2000,
    "microbatch_size": 16,
    "model": {
      "cutoff_A": 5.0,
      "energy_readout_init": "zero",
      "features": 16,
      "iterations": 2,
      "max_degree": 1,
      "radial_basis": 8
    },
    "reference_run": "runs/aspirin-force-diagnostics-2026-09-30/real",
    "reference_sha256": {
      "checkpoint-verification.json": "24e231afcdd58e5f34a759bf974a87dbe9e01f6d45adffd9324c99b4aa240e7b",
      "config.json": "d8bfbe676f0e03f0de23bdc2b4b44610f3f5c8f21c67034cb31091b8765ba53f",
      "history.json": "13488ec9153c6a0a85b7256cc408e14986a78b9e5f66070f60d66234810961a9",
      "provenance.json": "4c5c05915e11250f3bddabae841469039232a6db03ac316074a8467ccb0913d9",
      "summary.json": "3bab8e9f66cc747dacb6744cca0b93f316fd8568962fe736c7480b7382f7a650"
    },
    "replay_tolerance": {
      "atol": 1e-10,
      "rtol": 1e-08
    },
    "run_directory": "runs/aspirin-data-coverage-2026-09-30",
    "seed": 0,
    "status": "planned_unexecuted",
    "subsets_manifest": "manifests/aspirin-data-coverage-subsets-planned.json",
    "validation_interval": 100,
    "wall_seconds": 1800
  },
  "fit_gate": "Both final MSEs < 0.1 own initial",
  "limitations": "n=1; B changes record difficulty; C uses 8x exposures per update; training-only anchor",
  "rationale": "Broader connection support",
  "run_directory": "runs/aspirin-data-coverage-2026-09-30/diverse16",
  "subset": {
    "anchor_fit_overlap_ids": [
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_000.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_001.xyz"
    ],
    "identity": "3c77271af9509a980fc7f462710475535ba047a3467bd71106706867315060a4",
    "indices": [
      37,
      161,
      1616,
      1699,
      4162,
      4268,
      3651,
      3751,
      2507,
      2627,
      1484,
      1579,
      4414,
      4504,
      519,
      628
    ],
    "record_ids": [
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_000.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_001.xyz",
      "Landscape17/aspirin/SteepestDescent/23/coordsminus_010.xyz",
      "Landscape17/aspirin/SteepestDescent/23/coordsplus_010.xyz",
      "Landscape17/aspirin/SteepestDescent/77/coordsminus_025.xyz",
      "Landscape17/aspirin/SteepestDescent/77/coordsplus_030.xyz",
      "Landscape17/aspirin/SteepestDescent/58_2/coordsminus_040.xyz",
      "Landscape17/aspirin/SteepestDescent/58_2/coordsplus_033.xyz",
      "Landscape17/aspirin/SteepestDescent/44/coordsminus_067.xyz",
      "Landscape17/aspirin/SteepestDescent/44/coordsplus_054.xyz",
      "Landscape17/aspirin/SteepestDescent/22/coordsminus_054.xyz",
      "Landscape17/aspirin/SteepestDescent/22/coordsplus_062.xyz",
      "Landscape17/aspirin/SteepestDescent/78_2/coordsminus_072.xyz",
      "Landscape17/aspirin/SteepestDescent/78_2/coordsplus_067.xyz",
      "Landscape17/aspirin/SteepestDescent/01_2/coordsminus_109.xyz",
      "Landscape17/aspirin/SteepestDescent/01_2/coordsplus_094.xyz"
    ]
  }
}
```

## COVERAGE-002-RESULT — diverse16 trial outcome (2026-09-30)

Training-only diagnostic; no scientific generalization claim; full pilot remains pending.

```json
{
  "anchor_gate": {
    "accepted": false,
    "fraction_remaining": {
      "energy_mse_eV2": 0.01838750683983036,
      "force_mse_eV2_per_A2": 0.9387592519896504
    },
    "gate_fraction": 0.1,
    "threshold_mse": {
      "energy_mse_eV2": 0.00612983585312839,
      "force_mse_eV2_per_A2": 0.0010133561081129107
    }
  },
  "artifact_sha256": {
    "checkpoint-verification.json": "f39a1a3d6cacfc6eb6418e7a695bb550ac9350bc9e08cd2aa20d80ad256a6295",
    "history.json": "f9b66c8e496ba8ca82d626df2a38980f41c7dc1edf28bb97f21028a8aa8fc367",
    "last.msgpack": "7156487d43587f9cdfe563f14db79948d465b0bf43995ff67e07e3c6ac180d8f",
    "summary.json": "af772ad611b5d153b85265861375419df80f2b93c20407c0304f6ca9876fb5e6"
  },
  "elapsed_seconds": 33.923090540920384,
  "execution_valid": true,
  "failure_reason": null,
  "fit_gate": {
    "accepted": false,
    "fraction_remaining": {
      "energy_mse_eV2": 0.02669173143721766,
      "force_mse_eV2_per_A2": 0.8967607042970483
    },
    "gate_fraction": 0.1,
    "threshold_mse": {
      "energy_mse_eV2": 0.003622249218106596,
      "force_mse_eV2_per_A2": 0.0017939664221939815
    }
  },
  "frame_exposures": 32000,
  "initial_state_hashes": {
    "optimizer": "8fc30b9a817f70448819b69c458e2e1cacc034cde02880066eab48edebd4f01d",
    "params": "b6351d17922c7646322cb85f142c9566dc5380b6ca5676898c157d92dd93becd"
  },
  "status": "completed",
  "stop_reason": "max_updates",
  "updates": 2000
}
```

## COVERAGE-003-PRE — diverse128 trial preregistered (2026-09-30)

Training-only diagnostic; no scientific generalization claim; full pilot remains pending.

```json
{
  "anchor_gate": "Both final anchor MSEs < 0.1 fixed original initial",
  "config": {
    "backend": "cpu",
    "comparison_updates": [
      0,
      100,
      250,
      500,
      1000,
      1500,
      2000
    ],
    "data_directory": "data/processed/aspirin",
    "early_stopping": false,
    "evidence_log": "reports/PAPER_EVIDENCE_LOG.md",
    "expected_prepared_identity": "4b9f220201d6483e2b2a8da74c10330c3762942ff4c6fabe6389ee27249c43aa",
    "expected_split_hash": "73d5475bfc1f1158da62b939f5431b1781db886443cece2cdb297c8dd645133a",
    "expected_subsets_identity": "8139e442fca011efcbeb5f2d28d701c5f36030d61f045e5e652b98eb18b1be47",
    "gate_fraction": 0.1,
    "learning_rate": 0.001,
    "max_updates": 2000,
    "microbatch_size": 16,
    "model": {
      "cutoff_A": 5.0,
      "energy_readout_init": "zero",
      "features": 16,
      "iterations": 2,
      "max_degree": 1,
      "radial_basis": 8
    },
    "reference_run": "runs/aspirin-force-diagnostics-2026-09-30/real",
    "reference_sha256": {
      "checkpoint-verification.json": "24e231afcdd58e5f34a759bf974a87dbe9e01f6d45adffd9324c99b4aa240e7b",
      "config.json": "d8bfbe676f0e03f0de23bdc2b4b44610f3f5c8f21c67034cb31091b8765ba53f",
      "history.json": "13488ec9153c6a0a85b7256cc408e14986a78b9e5f66070f60d66234810961a9",
      "provenance.json": "4c5c05915e11250f3bddabae841469039232a6db03ac316074a8467ccb0913d9",
      "summary.json": "3bab8e9f66cc747dacb6744cca0b93f316fd8568962fe736c7480b7382f7a650"
    },
    "replay_tolerance": {
      "atol": 1e-10,
      "rtol": 1e-08
    },
    "run_directory": "runs/aspirin-data-coverage-2026-09-30",
    "seed": 0,
    "status": "planned_unexecuted",
    "subsets_manifest": "manifests/aspirin-data-coverage-subsets-planned.json",
    "validation_interval": 100,
    "wall_seconds": 1800
  },
  "fit_gate": "Both final MSEs < 0.1 own initial",
  "limitations": "n=1; B changes record difficulty; C uses 8x exposures per update; training-only anchor",
  "rationale": "Extra within-connection coverage at matched support",
  "run_directory": "runs/aspirin-data-coverage-2026-09-30/diverse128",
  "subset": {
    "anchor_fit_overlap_ids": [
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_000.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_015.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_031.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_046.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_062.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_077.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_093.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_108.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_001.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_016.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_032.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_047.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_063.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_078.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_094.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_110.xyz"
    ],
    "identity": "b667e61ebf7c44e88695466f2a2ba44451a87f9adb76728658e4ae3926715da9",
    "indices": [
      37,
      52,
      68,
      83,
      99,
      114,
      130,
      145,
      161,
      176,
      192,
      207,
      223,
      238,
      254,
      270,
      1606,
      1616,
      1627,
      1637,
      1648,
      1658,
      1669,
      1679,
      1690,
      1699,
      1710,
      1719,
      1730,
      1739,
      1750,
      1760,
      4137,
      4149,
      4162,
      4175,
      4188,
      4200,
      4213,
      4226,
      4239,
      4253,
      4268,
      4282,
      4297,
      4311,
      4326,
      4341,
      3611,
      3624,
      3638,
      3651,
      3665,
      3678,
      3692,
      3705,
      3719,
      3729,
      3741,
      3751,
      3763,
      3773,
      3785,
      3796,
      2440,
      2456,
      2474,
      2490,
      2507,
      2523,
      2541,
      2557,
      2574,
      2587,
      2600,
      2613,
      2627,
      2640,
      2653,
      2667,
      1430,
      1441,
      1452,
      1462,
      1474,
      1484,
      1496,
      1506,
      1518,
      1530,
      1543,
      1555,
      1567,
      1579,
      1592,
      1605,
      4342,
      4354,
      4366,
      4377,
      4390,
      4401,
      4414,
      4425,
      4438,
      4448,
      4460,
      4470,
      4482,
      4492,
      4504,
      4515,
      410,
      425,
      441,
      456,
      473,
      488,
      504,
      519,
      535,
      548,
      561,
      574,
      588,
      601,
      614,
      628
    ],
    "record_ids": [
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_000.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_015.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_031.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_046.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_062.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_077.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_093.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsminus_108.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_001.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_016.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_032.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_047.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_063.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_078.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_094.xyz",
      "Landscape17/aspirin/SteepestDescent/01/coordsplus_110.xyz",
      "Landscape17/aspirin/SteepestDescent/23/coordsminus_000.xyz",
      "Landscape17/aspirin/SteepestDescent/23/coordsminus_010.xyz",
      "Landscape17/aspirin/SteepestDescent/23/coordsminus_021.xyz",
      "Landscape17/aspirin/SteepestDescent/23/coordsminus_031.xyz",
      "Landscape17/aspirin/SteepestDescent/23/coordsminus_042.xyz",
      "Landscape17/aspirin/SteepestDescent/23/coordsminus_052.xyz",
      "Landscape17/aspirin/SteepestDescent/23/coordsminus_063.xyz",
      "Landscape17/aspirin/SteepestDescent/23/coordsminus_073.xyz",
      "Landscape17/aspirin/SteepestDescent/23/coordsplus_001.xyz",
      "Landscape17/aspirin/SteepestDescent/23/coordsplus_010.xyz",
      "Landscape17/aspirin/SteepestDescent/23/coordsplus_021.xyz",
      "Landscape17/aspirin/SteepestDescent/23/coordsplus_030.xyz",
      "Landscape17/aspirin/SteepestDescent/23/coordsplus_041.xyz",
      "Landscape17/aspirin/SteepestDescent/23/coordsplus_050.xyz",
      "Landscape17/aspirin/SteepestDescent/23/coordsplus_061.xyz",
      "Landscape17/aspirin/SteepestDescent/23/coordsplus_071.xyz",
      "Landscape17/aspirin/SteepestDescent/77/coordsminus_000.xyz",
      "Landscape17/aspirin/SteepestDescent/77/coordsminus_012.xyz",
      "Landscape17/aspirin/SteepestDescent/77/coordsminus_025.xyz",
      "Landscape17/aspirin/SteepestDescent/77/coordsminus_038.xyz",
      "Landscape17/aspirin/SteepestDescent/77/coordsminus_051.xyz",
      "Landscape17/aspirin/SteepestDescent/77/coordsminus_063.xyz",
      "Landscape17/aspirin/SteepestDescent/77/coordsminus_076.xyz",
      "Landscape17/aspirin/SteepestDescent/77/coordsminus_089.xyz",
      "Landscape17/aspirin/SteepestDescent/77/coordsplus_001.xyz",
      "Landscape17/aspirin/SteepestDescent/77/coordsplus_015.xyz",
      "Landscape17/aspirin/SteepestDescent/77/coordsplus_030.xyz",
      "Landscape17/aspirin/SteepestDescent/77/coordsplus_044.xyz",
      "Landscape17/aspirin/SteepestDescent/77/coordsplus_059.xyz",
      "Landscape17/aspirin/SteepestDescent/77/coordsplus_073.xyz",
      "Landscape17/aspirin/SteepestDescent/77/coordsplus_088.xyz",
      "Landscape17/aspirin/SteepestDescent/77/coordsplus_103.xyz",
      "Landscape17/aspirin/SteepestDescent/58_2/coordsminus_000.xyz",
      "Landscape17/aspirin/SteepestDescent/58_2/coordsminus_013.xyz",
      "Landscape17/aspirin/SteepestDescent/58_2/coordsminus_027.xyz",
      "Landscape17/aspirin/SteepestDescent/58_2/coordsminus_040.xyz",
      "Landscape17/aspirin/SteepestDescent/58_2/coordsminus_054.xyz",
      "Landscape17/aspirin/SteepestDescent/58_2/coordsminus_067.xyz",
      "Landscape17/aspirin/SteepestDescent/58_2/coordsminus_081.xyz",
      "Landscape17/aspirin/SteepestDescent/58_2/coordsminus_094.xyz",
      "Landscape17/aspirin/SteepestDescent/58_2/coordsplus_001.xyz",
      "Landscape17/aspirin/SteepestDescent/58_2/coordsplus_011.xyz",
      "Landscape17/aspirin/SteepestDescent/58_2/coordsplus_023.xyz",
      "Landscape17/aspirin/SteepestDescent/58_2/coordsplus_033.xyz",
      "Landscape17/aspirin/SteepestDescent/58_2/coordsplus_045.xyz",
      "Landscape17/aspirin/SteepestDescent/58_2/coordsplus_055.xyz",
      "Landscape17/aspirin/SteepestDescent/58_2/coordsplus_067.xyz",
      "Landscape17/aspirin/SteepestDescent/58_2/coordsplus_078.xyz",
      "Landscape17/aspirin/SteepestDescent/44/coordsminus_000.xyz",
      "Landscape17/aspirin/SteepestDescent/44/coordsminus_016.xyz",
      "Landscape17/aspirin/SteepestDescent/44/coordsminus_034.xyz",
      "Landscape17/aspirin/SteepestDescent/44/coordsminus_050.xyz",
      "Landscape17/aspirin/SteepestDescent/44/coordsminus_067.xyz",
      "Landscape17/aspirin/SteepestDescent/44/coordsminus_083.xyz",
      "Landscape17/aspirin/SteepestDescent/44/coordsminus_101.xyz",
      "Landscape17/aspirin/SteepestDescent/44/coordsminus_117.xyz",
      "Landscape17/aspirin/SteepestDescent/44/coordsplus_001.xyz",
      "Landscape17/aspirin/SteepestDescent/44/coordsplus_014.xyz",
      "Landscape17/aspirin/SteepestDescent/44/coordsplus_027.xyz",
      "Landscape17/aspirin/SteepestDescent/44/coordsplus_040.xyz",
      "Landscape17/aspirin/SteepestDescent/44/coordsplus_054.xyz",
      "Landscape17/aspirin/SteepestDescent/44/coordsplus_067.xyz",
      "Landscape17/aspirin/SteepestDescent/44/coordsplus_080.xyz",
      "Landscape17/aspirin/SteepestDescent/44/coordsplus_094.xyz",
      "Landscape17/aspirin/SteepestDescent/22/coordsminus_000.xyz",
      "Landscape17/aspirin/SteepestDescent/22/coordsminus_011.xyz",
      "Landscape17/aspirin/SteepestDescent/22/coordsminus_022.xyz",
      "Landscape17/aspirin/SteepestDescent/22/coordsminus_032.xyz",
      "Landscape17/aspirin/SteepestDescent/22/coordsminus_044.xyz",
      "Landscape17/aspirin/SteepestDescent/22/coordsminus_054.xyz",
      "Landscape17/aspirin/SteepestDescent/22/coordsminus_066.xyz",
      "Landscape17/aspirin/SteepestDescent/22/coordsminus_076.xyz",
      "Landscape17/aspirin/SteepestDescent/22/coordsplus_001.xyz",
      "Landscape17/aspirin/SteepestDescent/22/coordsplus_013.xyz",
      "Landscape17/aspirin/SteepestDescent/22/coordsplus_026.xyz",
      "Landscape17/aspirin/SteepestDescent/22/coordsplus_038.xyz",
      "Landscape17/aspirin/SteepestDescent/22/coordsplus_050.xyz",
      "Landscape17/aspirin/SteepestDescent/22/coordsplus_062.xyz",
      "Landscape17/aspirin/SteepestDescent/22/coordsplus_075.xyz",
      "Landscape17/aspirin/SteepestDescent/22/coordsplus_088.xyz",
      "Landscape17/aspirin/SteepestDescent/78_2/coordsminus_000.xyz",
      "Landscape17/aspirin/SteepestDescent/78_2/coordsminus_012.xyz",
      "Landscape17/aspirin/SteepestDescent/78_2/coordsminus_024.xyz",
      "Landscape17/aspirin/SteepestDescent/78_2/coordsminus_035.xyz",
      "Landscape17/aspirin/SteepestDescent/78_2/coordsminus_048.xyz",
      "Landscape17/aspirin/SteepestDescent/78_2/coordsminus_059.xyz",
      "Landscape17/aspirin/SteepestDescent/78_2/coordsminus_072.xyz",
      "Landscape17/aspirin/SteepestDescent/78_2/coordsminus_083.xyz",
      "Landscape17/aspirin/SteepestDescent/78_2/coordsplus_001.xyz",
      "Landscape17/aspirin/SteepestDescent/78_2/coordsplus_011.xyz",
      "Landscape17/aspirin/SteepestDescent/78_2/coordsplus_023.xyz",
      "Landscape17/aspirin/SteepestDescent/78_2/coordsplus_033.xyz",
      "Landscape17/aspirin/SteepestDescent/78_2/coordsplus_045.xyz",
      "Landscape17/aspirin/SteepestDescent/78_2/coordsplus_055.xyz",
      "Landscape17/aspirin/SteepestDescent/78_2/coordsplus_067.xyz",
      "Landscape17/aspirin/SteepestDescent/78_2/coordsplus_078.xyz",
      "Landscape17/aspirin/SteepestDescent/01_2/coordsminus_000.xyz",
      "Landscape17/aspirin/SteepestDescent/01_2/coordsminus_015.xyz",
      "Landscape17/aspirin/SteepestDescent/01_2/coordsminus_031.xyz",
      "Landscape17/aspirin/SteepestDescent/01_2/coordsminus_046.xyz",
      "Landscape17/aspirin/SteepestDescent/01_2/coordsminus_063.xyz",
      "Landscape17/aspirin/SteepestDescent/01_2/coordsminus_078.xyz",
      "Landscape17/aspirin/SteepestDescent/01_2/coordsminus_094.xyz",
      "Landscape17/aspirin/SteepestDescent/01_2/coordsminus_109.xyz",
      "Landscape17/aspirin/SteepestDescent/01_2/coordsplus_001.xyz",
      "Landscape17/aspirin/SteepestDescent/01_2/coordsplus_014.xyz",
      "Landscape17/aspirin/SteepestDescent/01_2/coordsplus_027.xyz",
      "Landscape17/aspirin/SteepestDescent/01_2/coordsplus_040.xyz",
      "Landscape17/aspirin/SteepestDescent/01_2/coordsplus_054.xyz",
      "Landscape17/aspirin/SteepestDescent/01_2/coordsplus_067.xyz",
      "Landscape17/aspirin/SteepestDescent/01_2/coordsplus_080.xyz",
      "Landscape17/aspirin/SteepestDescent/01_2/coordsplus_094.xyz"
    ]
  }
}
```

## COVERAGE-003-RESULT — diverse128 trial outcome (2026-09-30)

Training-only diagnostic; no scientific generalization claim; full pilot remains pending.

```json
{
  "anchor_gate": {
    "accepted": false,
    "fraction_remaining": {
      "energy_mse_eV2": 0.024702018838049,
      "force_mse_eV2_per_A2": 0.8769996293603259
    },
    "gate_fraction": 0.1,
    "threshold_mse": {
      "energy_mse_eV2": 0.00612983585312839,
      "force_mse_eV2_per_A2": 0.0010133561081129107
    }
  },
  "artifact_sha256": {
    "checkpoint-verification.json": "cf07aa2554610daac41ef3e83a1335092e751bf8db7ecc30ccf2cf5128f1a319",
    "history.json": "2e037af82e7fae42f7811fb661b39202b331bb42e8499ec45c0c3ba73d18629f",
    "last.msgpack": "7305113575499ebe51ca3e30b87d5bb644888b913b845e3fee5b95e222f4f6c3",
    "summary.json": "0e374b8ed313f8004aec6219af3b2e45f0a868082f00ced907a6f4c9d8b88a8b"
  },
  "elapsed_seconds": 256.7245559170842,
  "execution_valid": true,
  "failure_reason": null,
  "fit_gate": {
    "accepted": false,
    "fraction_remaining": {
      "energy_mse_eV2": 0.08738058573272221,
      "force_mse_eV2_per_A2": 0.9784403043672875
    },
    "gate_fraction": 0.1,
    "threshold_mse": {
      "energy_mse_eV2": 0.003751676427546992,
      "force_mse_eV2_per_A2": 0.001408394821807022
    }
  },
  "frame_exposures": 256000,
  "initial_state_hashes": {
    "optimizer": "8fc30b9a817f70448819b69c458e2e1cacc034cde02880066eab48edebd4f01d",
    "params": "b6351d17922c7646322cb85f142c9566dc5380b6ca5676898c157d92dd93becd"
  },
  "status": "completed",
  "stop_reason": "max_updates",
  "updates": 2000
}
```

## COVERAGE-MATRIX-RESULT — Completed comparison and negative learning result

All3arms completed2,000updates within1800s caps, same initial parameter/optimizer hashes and fresh-process maximum prediction differences0 on every fitting/anchor frame. A replay maximum MSE difference6.5052130349130266e-18 across21historical checkpoints. Valid execution does not mean learning acceptance. All energy individual thresholds passed; all force individual thresholds failed, so every joint fitting and anchor gate failed.

Final force fractions remaining: A fitting/anchor0.7145522825221748; B fitting0.8967607042970483/anchor0.9387592519896504; C fitting0.9784403043672875/anchor0.8769996293603259. C thus reduced fitting force MSE2.156% and original-anchor force MSE12.300%, below the predeclared90% requirement. No gate ever passed at an observed scheduled update.

C cost256,000training-frame exposures and256.7245559170842s versus A32,000/37.41735974990297s and B32,000/33.923090540920384s. Warmup exposures are separately retained. Recorded evaluation counts refer to metric scoring passes; saved-reference and fresh-process probe predictions are additional verification work. B fits2anchor frames (final force MSE0.020278909839792178) and does not fit14(final0.00797498341775791); differing initial difficulty prevents comparing these values as a transfer benefit. All anchors are globally training records.

Interpretation: broader/larger coverage alone under these fixed settings did not clear the force gate. This does not identify capacity, gradient balance or optimization as the cause, establish a sample-count effect, or imply poor scientific generalization. Next is a separately planned/twice-reviewed architecture/loss-gradient investigation. No extra attempt/seed/tuning/GPU/full-pilot/test/scientific comparison followed.

Tracked report/compact evidence and3measured-point figures were generated from preserved raw artifacts; generator checksum and Matplotlib version are in compact JSON. Plot creation succeeded after benign unwritable-cache warnings; source settings/research outputs were unchanged. Both force and overlap figures visually checked; energy figure check pending.

```json
{
  "root_summary_sha256": "54e7f6d7e9995dbca90bdca3417457b075e99037dc9ecd6d6a1c0647ac102a9f",
  "compact_evidence_sha256": "6c5b3233360a6baf9b155162fbae1fdf43b0d1f9b209f79997b468976d40dbb4",
  "execution_revision": "356a3615e4693010ea9fc38d5874337991eb06dd",
  "code_fingerprint": "77dd790e53d96fa924de73feb03b812ce611772c5de50dcd63a7c2cd8475852f",
  "frozen_subset_identity": "8139e442fca011efcbeb5f2d28d701c5f36030d61f045e5e652b98eb18b1be47"
}
```

## IMPL-COVERAGE-004 — Independent review and handled-failure retention repair

Independent final reviewer regenerated preflight, checked all saved hashes/history/init/overlap counts and recomputed final MSE within1e-14; measured matrix remains trustworthy. One Important bug: a runner exception after creating artifacts could be masked by directory collision, losing root/downstream failure records. Three regressions first failed, then passed in0.50s after a single controller-only fix; independent follow-up also passed3tests. Existing summaries/history/checkpoints stay preserved; attempted failure and blocked arms are distinct. Raw production research runs were not changed or repeated.

Full final regression suite82passed in98.94s; fresh review cleared all Critical/Important findings, with no minors. All3figures now visually checked (completes COVERAGE-MATRIX-RESULT pending energy check). Review/rulings and test/log identities: `docs/reviews/2026-09-30-data-coverage-final-review.md`, `reports/data-coverage-execution-record.md`. Both failed initial assertions/fixtures and the final retention regressions remain preserved. Obsidian hub/findings refreshed through CLI with original notes backed up; numerical authority remains repository evidence. All workspaces/packages/logs retained. Local branch only; full pilot/new architecture trials remain pending separate reviewed plans.

## PLAN-GRADIENT-001 — Prospective invariant-representation and loss-gradient investigation

User authorized next investigation after failed coverage controls. Saved spec/plan `docs/superpowers/{specs,plans}/2026-09-30-representation-gradient-diagnostic*`; config `configs/aspirin-representation-gradient-diagnostic.json` pins31coverage source artifacts. Status planned/unexecuted; two independent reviews pending. No labels numerically analyzed, parameters fitted/updated or product code modified in this planning stage.

Chosen bounded probes: necessary rigid-motion force floor and unchanged cutoff graph on16/16/128fixed training sets, then separate normalized energy/force gradients at eight fixed fresh/final fitting/common-anchor cases. SVD1e-12, microbatch16, joint-gradient identity1e-10/1e-8, FD1e-4/1e-5 with1e-8/1e-4 tolerance,600s cap. No training/optimizer update/architecture/loss/label/gate changes, test access or full pilot. Floor obstruction is model-target compatibility, not corruption or sufficiency; local gradient conflict is not Adam dynamics or causal failure. Exact outcomes unknown. Every success/failure/block will be retained.

## PLAN-GRADIENT-002 — Both independent reviews cleared

Scientific and engineering reviewers independently verified all31source hashes and frozen state/settings; both cleared the exact plan for implementation and one bounded execution, with no material blocker. Accepted clarifications are in spec/review record: complete source allowlist before arrays, component-space projected-prediction leakage, numeric-path branch metadata, explicit force/energy norm ratio and raw gradient-descent directional derivatives, budget before preflight, parameterization/loss-scale dependence and directional rather than full-component FD verification. Sources/cases/tolerances/caps remain unchanged. No new numerical research result yet; full pilot pending.

## IMPL-GRADIENT-001 — Rigid-motion/graph engineering checks

Prospective rationale: establish the reviewed projection/graph algebra before reading research labels; analytic fixtures, independent least-squares oracle, rigid/scale covariance, collinear/single-atom rank, invalid-input rejection, pooled/equal-group metadata and strict cutoff/two-hop boundary cases. Initial expected RED: missing representation module (retained task1-red.log). Implemented pure observational projection; no labels changed. All13new tests passed in0.16s; full suite95passed (task1-suite.log). This is engineering validation, not a new research outcome; production600s analysis remains unexecuted. Logs retained in `.superpowers/sdd/2026-09-30-representation-gradient-diagnostic/`.

## IMPL-GRADIENT-002 — Split-gradient engineering validation

Prospective rationale/settings: unchanged scaled energy and force objectives,7-frame witness with3/3/1chunks, algebraically known positive/negative/orthogonal/zero directions, mixed force-backbone derivatives, both finite-difference steps, intentionally wrong gradient sign, deterministic fallback, immutable state and retained pre-FD observations on budget stop. Missing-helper RED preserved in task2-red.log. First7tests plus coverage tests15/15passed30.00s; added budget retention test first failed (DID NOT RAISE), then all16focused tests passed29.81s after callbacks/checks were implemented. Independent joint objective matches the split sum. These are artificial engineering fixtures, not research comparisons. Full suite114passed115.04s before production analysis; logs retained.

## IMPL-GRADIENT-003 — Source/controller engineering validation and execution preflight

Missing-controller RED preserved. All11controller tests passed8.46s: complete31hash allowlist/settings precede train loader; real source arrays/checkpoints/predictions/fresh hashes verify; rehashed corrupt checkpoint/config/arrays reject; case order/counts, immutable state, partial exceptions, pre-attempt budget blocks, protected output and CLI execution-status semantics pass. Full suite114passed115.04s. Production controller checks600s from before preflight, preserves partial gradient and both-step observations, and blocks pending cases without training. The exact frozen config will execute once after committing this tested code. Outcome remains unknown; no extra attempt/tuning/pilot authorized. Report generation is observational JSON-only presentation.
