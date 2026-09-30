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

## GRADIENT-EXECUTION-PRE — Frozen analysis preregistered

Frozen training-only diagnostics; no optimizer update or learning acceptance.

```json
{
  "config": {
    "backend": "cpu",
    "data_directory": "data/processed/aspirin",
    "decomposition_atol": 1e-10,
    "decomposition_rtol": 1e-08,
    "direction_seed": 20260930,
    "direction_zero_tolerance": 1e-12,
    "evidence_log": "reports/PAPER_EVIDENCE_LOG.md",
    "expected_prepared_identity": "4b9f220201d6483e2b2a8da74c10330c3762942ff4c6fabe6389ee27249c43aa",
    "expected_split_hash": "73d5475bfc1f1158da62b939f5431b1781db886443cece2cdb297c8dd645133a",
    "expected_subsets_identity": "8139e442fca011efcbeb5f2d28d701c5f36030d61f045e5e652b98eb18b1be47",
    "fd_atol": 1e-08,
    "fd_rtol": 0.0001,
    "fd_steps": [
      0.0001,
      1e-05
    ],
    "gradient_identity_atol": 1e-10,
    "gradient_identity_rtol": 1e-08,
    "learning_rate": 0.001,
    "microbatch_size": 16,
    "model": {
      "cutoff_A": 5.0,
      "energy_readout_init": "zero",
      "features": 16,
      "iterations": 2,
      "max_degree": 1,
      "radial_basis": 8
    },
    "output_reserve_seconds": 10,
    "prediction_constraint_atol": 1e-08,
    "run_directory": "runs/aspirin-representation-gradient-2026-09-30",
    "seed": 0,
    "source_run": "runs/aspirin-data-coverage-2026-09-30",
    "source_sha256": {
      "config.json": "3575dad49acf97b8f780f953a397c776cb5f3dd96729f1e6236837d8e99a5f64",
      "diverse128/anchor-inputs.npz": "5c638b71fe14615e53ae0b27a0c69c1fff1fd572f9f7191130cb2116e86264fc",
      "diverse128/checkpoint-verification.json": "cf07aa2554610daac41ef3e83a1335092e751bf8db7ecc30ccf2cf5128f1a319",
      "diverse128/config.json": "dfbdb9129e00a2984341a1efd80e901fda20565d3bcc7cc53ae04d2ed5ac6b51",
      "diverse128/fitting-inputs.npz": "9c56de4bae8058f4c18aab338aaee4e45c42f527905e48656f49900239fa701f",
      "diverse128/history.json": "2e037af82e7fae42f7811fb661b39202b331bb42e8499ec45c0c3ba73d18629f",
      "diverse128/last.msgpack": "7305113575499ebe51ca3e30b87d5bb644888b913b845e3fee5b95e222f4f6c3",
      "diverse128/provenance.json": "1502d258d63cf8a69e9ee67fcd9852855b9dbc3deb2010c33010d8b56db81063",
      "diverse128/reference.npz": "5ebb9dfb89f2c5df095a8d6f1dfe5fa2e1901b6dcfcf3159b417ccad5b724afd",
      "diverse128/summary.json": "0e374b8ed313f8004aec6219af3b2e45f0a868082f00ced907a6f4c9d8b88a8b",
      "diverse16/anchor-inputs.npz": "5c638b71fe14615e53ae0b27a0c69c1fff1fd572f9f7191130cb2116e86264fc",
      "diverse16/checkpoint-verification.json": "f39a1a3d6cacfc6eb6418e7a695bb550ac9350bc9e08cd2aa20d80ad256a6295",
      "diverse16/config.json": "dfbdb9129e00a2984341a1efd80e901fda20565d3bcc7cc53ae04d2ed5ac6b51",
      "diverse16/fitting-inputs.npz": "ae6550bbda77af6eb1617a859cc8ce118ab90c6bd109c3a25b0d3827faf9a65c",
      "diverse16/history.json": "f9b66c8e496ba8ca82d626df2a38980f41c7dc1edf28bb97f21028a8aa8fc367",
      "diverse16/last.msgpack": "7156487d43587f9cdfe563f14db79948d465b0bf43995ff67e07e3c6ac180d8f",
      "diverse16/provenance.json": "80bed026608d4455f4ab81a1fbc17520c511a1772c39fc448eada96f6dd34791",
      "diverse16/reference.npz": "f48c42e9697e2c93a6a8393c1378d3faecf1ad08a3c598518cf39fafae2c1c85",
      "diverse16/summary.json": "af772ad611b5d153b85265861375419df80f2b93c20407c0304f6ca9876fb5e6",
      "narrow16/anchor-inputs.npz": "5c638b71fe14615e53ae0b27a0c69c1fff1fd572f9f7191130cb2116e86264fc",
      "narrow16/checkpoint-verification.json": "f39a1a3d6cacfc6eb6418e7a695bb550ac9350bc9e08cd2aa20d80ad256a6295",
      "narrow16/config.json": "dfbdb9129e00a2984341a1efd80e901fda20565d3bcc7cc53ae04d2ed5ac6b51",
      "narrow16/fitting-inputs.npz": "5c638b71fe14615e53ae0b27a0c69c1fff1fd572f9f7191130cb2116e86264fc",
      "narrow16/history.json": "9970a673c501b566e8f8de0b83581ce81f40cba9a71a2547111f9fe46b0af812",
      "narrow16/last.msgpack": "b5c792314ea6b34d46bc5b2b104abbc7b1e647fd6c81cea55ec49e94b18cb6b7",
      "narrow16/provenance.json": "5de39bfac0af7376beee7e9790b6028443cff7d8d3983006b4a1ba0b5617b6ad",
      "narrow16/reference.npz": "95f1258fba9882b7c7ae600f704f0465e7c6011ef149d2127ccf5a414ee9ecaa",
      "narrow16/summary.json": "c3b0aa387b7cbd2cc57723e6232677eea19abb3f03dc1834cd8cdf1b32128452",
      "provenance.json": "b60d13789966f2ce2fbb11a3c4de73f0886f2e952ba9e0254892c35c554f7dcc",
      "subsets.json": "d3c3db7a92f3ffb0748039f63d7a0734bf38214032037ab67780d6182ad764b6",
      "summary.json": "54e7f6d7e9995dbca90bdca3417457b075e99037dc9ecd6d6a1c0647ac102a9f"
    },
    "status": "planned_unexecuted",
    "subsets_manifest": "manifests/aspirin-data-coverage-subsets-planned.json",
    "svd_rcond": 1e-12,
    "wall_seconds": 600
  },
  "limits": "One bounded execution; no retries or training",
  "source_config_hash": "302b32fdb0e22c79e47d399902e8765f4dd6f7566ffb0ec59622af1d1d2f63c3"
}
```

## GRADIENT-PHASE-A-PRE — Necessary constraints preregistered

Frozen training-only diagnostics; no optimizer update or learning acceptance.

```json
{
  "acceptance": "Projection/reconstruction checks, saved prediction leakage<=1e-8 and raw-MSE decomposition1e-10/1e-8; original gates unchanged",
  "config_hash": "302b32fdb0e22c79e47d399902e8765f4dd6f7566ffb0ec59622af1d1d2f63c3",
  "subsets": {
    "anchor_identity": "338e75a921b30820b70cfba47d1f06e7356a6d2232905d88975d088844403a3c",
    "ordered_groups": [
      "01",
      "23",
      "77",
      "58_2",
      "44",
      "22",
      "78_2",
      "01_2"
    ],
    "prepared_identity": "4b9f220201d6483e2b2a8da74c10330c3762942ff4c6fabe6389ee27249c43aa",
    "rank_denominators": {
      "minus": 122,
      "plus": 110
    },
    "rank_numerators": {
      "minus": [
        0,
        15,
        31,
        46,
        62,
        77,
        93,
        108
      ],
      "plus": [
        1,
        16,
        32,
        47,
        63,
        78,
        94,
        110
      ]
    },
    "selection": {
      "01": {
        "minus": {
          "pool_count": 123,
          "ranks": [
            0,
            15,
            31,
            46,
            62,
            77,
            93,
            108
          ],
          "record_ids": [
            "Landscape17/aspirin/SteepestDescent/01/coordsminus_000.xyz",
            "Landscape17/aspirin/SteepestDescent/01/coordsminus_015.xyz",
            "Landscape17/aspirin/SteepestDescent/01/coordsminus_031.xyz",
            "Landscape17/aspirin/SteepestDescent/01/coordsminus_046.xyz",
            "Landscape17/aspirin/SteepestDescent/01/coordsminus_062.xyz",
            "Landscape17/aspirin/SteepestDescent/01/coordsminus_077.xyz",
            "Landscape17/aspirin/SteepestDescent/01/coordsminus_093.xyz",
            "Landscape17/aspirin/SteepestDescent/01/coordsminus_108.xyz"
          ],
          "small_set_record_id": "Landscape17/aspirin/SteepestDescent/01/coordsminus_000.xyz"
        },
        "plus": {
          "pool_count": 111,
          "ranks": [
            1,
            16,
            32,
            47,
            63,
            78,
            94,
            110
          ],
          "record_ids": [
            "Landscape17/aspirin/SteepestDescent/01/coordsplus_001.xyz",
            "Landscape17/aspirin/SteepestDescent/01/coordsplus_016.xyz",
            "Landscape17/aspirin/SteepestDescent/01/coordsplus_032.xyz",
            "Landscape17/aspirin/SteepestDescent/01/coordsplus_047.xyz",
            "Landscape17/aspirin/SteepestDescent/01/coordsplus_063.xyz",
            "Landscape17/aspirin/SteepestDescent/01/coordsplus_078.xyz",
            "Landscape17/aspirin/SteepestDescent/01/coordsplus_094.xyz",
            "Landscape17/aspirin/SteepestDescent/01/coordsplus_110.xyz"
          ],
          "small_set_record_id": "Landscape17/aspirin/SteepestDescent/01/coordsplus_001.xyz"
        }
      },
      "01_2": {
        "minus": {
          "pool_count": 124,
          "ranks": [
            0,
            15,
            31,
            46,
            63,
            78,
            94,
            109
          ],
          "record_ids": [
            "Landscape17/aspirin/SteepestDescent/01_2/coordsminus_000.xyz",
            "Landscape17/aspirin/SteepestDescent/01_2/coordsminus_015.xyz",
            "Landscape17/aspirin/SteepestDescent/01_2/coordsminus_031.xyz",
            "Landscape17/aspirin/SteepestDescent/01_2/coordsminus_046.xyz",
            "Landscape17/aspirin/SteepestDescent/01_2/coordsminus_063.xyz",
            "Landscape17/aspirin/SteepestDescent/01_2/coordsminus_078.xyz",
            "Landscape17/aspirin/SteepestDescent/01_2/coordsminus_094.xyz",
            "Landscape17/aspirin/SteepestDescent/01_2/coordsminus_109.xyz"
          ],
          "small_set_record_id": "Landscape17/aspirin/SteepestDescent/01_2/coordsminus_109.xyz"
        },
        "plus": {
          "pool_count": 95,
          "ranks": [
            1,
            14,
            27,
            40,
            54,
            67,
            80,
            94
          ],
          "record_ids": [
            "Landscape17/aspirin/SteepestDescent/01_2/coordsplus_001.xyz",
            "Landscape17/aspirin/SteepestDescent/01_2/coordsplus_014.xyz",
            "Landscape17/aspirin/SteepestDescent/01_2/coordsplus_027.xyz",
            "Landscape17/aspirin/SteepestDescent/01_2/coordsplus_040.xyz",
            "Landscape17/aspirin/SteepestDescent/01_2/coordsplus_054.xyz",
            "Landscape17/aspirin/SteepestDescent/01_2/coordsplus_067.xyz",
            "Landscape17/aspirin/SteepestDescent/01_2/coordsplus_080.xyz",
            "Landscape17/aspirin/SteepestDescent/01_2/coordsplus_094.xyz"
          ],
          "small_set_record_id": "Landscape17/aspirin/SteepestDescent/01_2/coordsplus_094.xyz"
        }
      },
      "22": {
        "minus": {
          "pool_count": 87,
          "ranks": [
            0,
            11,
            22,
            32,
            44,
            54,
            66,
            76
          ],
          "record_ids": [
            "Landscape17/aspirin/SteepestDescent/22/coordsminus_000.xyz",
            "Landscape17/aspirin/SteepestDescent/22/coordsminus_011.xyz",
            "Landscape17/aspirin/SteepestDescent/22/coordsminus_022.xyz",
            "Landscape17/aspirin/SteepestDescent/22/coordsminus_032.xyz",
            "Landscape17/aspirin/SteepestDescent/22/coordsminus_044.xyz",
            "Landscape17/aspirin/SteepestDescent/22/coordsminus_054.xyz",
            "Landscape17/aspirin/SteepestDescent/22/coordsminus_066.xyz",
            "Landscape17/aspirin/SteepestDescent/22/coordsminus_076.xyz"
          ],
          "small_set_record_id": "Landscape17/aspirin/SteepestDescent/22/coordsminus_054.xyz"
        },
        "plus": {
          "pool_count": 89,
          "ranks": [
            1,
            13,
            26,
            38,
            50,
            62,
            75,
            88
          ],
          "record_ids": [
            "Landscape17/aspirin/SteepestDescent/22/coordsplus_001.xyz",
            "Landscape17/aspirin/SteepestDescent/22/coordsplus_013.xyz",
            "Landscape17/aspirin/SteepestDescent/22/coordsplus_026.xyz",
            "Landscape17/aspirin/SteepestDescent/22/coordsplus_038.xyz",
            "Landscape17/aspirin/SteepestDescent/22/coordsplus_050.xyz",
            "Landscape17/aspirin/SteepestDescent/22/coordsplus_062.xyz",
            "Landscape17/aspirin/SteepestDescent/22/coordsplus_075.xyz",
            "Landscape17/aspirin/SteepestDescent/22/coordsplus_088.xyz"
          ],
          "small_set_record_id": "Landscape17/aspirin/SteepestDescent/22/coordsplus_062.xyz"
        }
      },
      "23": {
        "minus": {
          "pool_count": 83,
          "ranks": [
            0,
            10,
            21,
            31,
            42,
            52,
            63,
            73
          ],
          "record_ids": [
            "Landscape17/aspirin/SteepestDescent/23/coordsminus_000.xyz",
            "Landscape17/aspirin/SteepestDescent/23/coordsminus_010.xyz",
            "Landscape17/aspirin/SteepestDescent/23/coordsminus_021.xyz",
            "Landscape17/aspirin/SteepestDescent/23/coordsminus_031.xyz",
            "Landscape17/aspirin/SteepestDescent/23/coordsminus_042.xyz",
            "Landscape17/aspirin/SteepestDescent/23/coordsminus_052.xyz",
            "Landscape17/aspirin/SteepestDescent/23/coordsminus_063.xyz",
            "Landscape17/aspirin/SteepestDescent/23/coordsminus_073.xyz"
          ],
          "small_set_record_id": "Landscape17/aspirin/SteepestDescent/23/coordsminus_010.xyz"
        },
        "plus": {
          "pool_count": 72,
          "ranks": [
            1,
            10,
            21,
            30,
            41,
            50,
            61,
            71
          ],
          "record_ids": [
            "Landscape17/aspirin/SteepestDescent/23/coordsplus_001.xyz",
            "Landscape17/aspirin/SteepestDescent/23/coordsplus_010.xyz",
            "Landscape17/aspirin/SteepestDescent/23/coordsplus_021.xyz",
            "Landscape17/aspirin/SteepestDescent/23/coordsplus_030.xyz",
            "Landscape17/aspirin/SteepestDescent/23/coordsplus_041.xyz",
            "Landscape17/aspirin/SteepestDescent/23/coordsplus_050.xyz",
            "Landscape17/aspirin/SteepestDescent/23/coordsplus_061.xyz",
            "Landscape17/aspirin/SteepestDescent/23/coordsplus_071.xyz"
          ],
          "small_set_record_id": "Landscape17/aspirin/SteepestDescent/23/coordsplus_010.xyz"
        }
      },
      "44": {
        "minus": {
          "pool_count": 133,
          "ranks": [
            0,
            16,
            34,
            50,
            67,
            83,
            101,
            117
          ],
          "record_ids": [
            "Landscape17/aspirin/SteepestDescent/44/coordsminus_000.xyz",
            "Landscape17/aspirin/SteepestDescent/44/coordsminus_016.xyz",
            "Landscape17/aspirin/SteepestDescent/44/coordsminus_034.xyz",
            "Landscape17/aspirin/SteepestDescent/44/coordsminus_050.xyz",
            "Landscape17/aspirin/SteepestDescent/44/coordsminus_067.xyz",
            "Landscape17/aspirin/SteepestDescent/44/coordsminus_083.xyz",
            "Landscape17/aspirin/SteepestDescent/44/coordsminus_101.xyz",
            "Landscape17/aspirin/SteepestDescent/44/coordsminus_117.xyz"
          ],
          "small_set_record_id": "Landscape17/aspirin/SteepestDescent/44/coordsminus_067.xyz"
        },
        "plus": {
          "pool_count": 95,
          "ranks": [
            1,
            14,
            27,
            40,
            54,
            67,
            80,
            94
          ],
          "record_ids": [
            "Landscape17/aspirin/SteepestDescent/44/coordsplus_001.xyz",
            "Landscape17/aspirin/SteepestDescent/44/coordsplus_014.xyz",
            "Landscape17/aspirin/SteepestDescent/44/coordsplus_027.xyz",
            "Landscape17/aspirin/SteepestDescent/44/coordsplus_040.xyz",
            "Landscape17/aspirin/SteepestDescent/44/coordsplus_054.xyz",
            "Landscape17/aspirin/SteepestDescent/44/coordsplus_067.xyz",
            "Landscape17/aspirin/SteepestDescent/44/coordsplus_080.xyz",
            "Landscape17/aspirin/SteepestDescent/44/coordsplus_094.xyz"
          ],
          "small_set_record_id": "Landscape17/aspirin/SteepestDescent/44/coordsplus_054.xyz"
        }
      },
      "58_2": {
        "minus": {
          "pool_count": 107,
          "ranks": [
            0,
            13,
            27,
            40,
            54,
            67,
            81,
            94
          ],
          "record_ids": [
            "Landscape17/aspirin/SteepestDescent/58_2/coordsminus_000.xyz",
            "Landscape17/aspirin/SteepestDescent/58_2/coordsminus_013.xyz",
            "Landscape17/aspirin/SteepestDescent/58_2/coordsminus_027.xyz",
            "Landscape17/aspirin/SteepestDescent/58_2/coordsminus_040.xyz",
            "Landscape17/aspirin/SteepestDescent/58_2/coordsminus_054.xyz",
            "Landscape17/aspirin/SteepestDescent/58_2/coordsminus_067.xyz",
            "Landscape17/aspirin/SteepestDescent/58_2/coordsminus_081.xyz",
            "Landscape17/aspirin/SteepestDescent/58_2/coordsminus_094.xyz"
          ],
          "small_set_record_id": "Landscape17/aspirin/SteepestDescent/58_2/coordsminus_040.xyz"
        },
        "plus": {
          "pool_count": 79,
          "ranks": [
            1,
            11,
            23,
            33,
            45,
            55,
            67,
            78
          ],
          "record_ids": [
            "Landscape17/aspirin/SteepestDescent/58_2/coordsplus_001.xyz",
            "Landscape17/aspirin/SteepestDescent/58_2/coordsplus_011.xyz",
            "Landscape17/aspirin/SteepestDescent/58_2/coordsplus_023.xyz",
            "Landscape17/aspirin/SteepestDescent/58_2/coordsplus_033.xyz",
            "Landscape17/aspirin/SteepestDescent/58_2/coordsplus_045.xyz",
            "Landscape17/aspirin/SteepestDescent/58_2/coordsplus_055.xyz",
            "Landscape17/aspirin/SteepestDescent/58_2/coordsplus_067.xyz",
            "Landscape17/aspirin/SteepestDescent/58_2/coordsplus_078.xyz"
          ],
          "small_set_record_id": "Landscape17/aspirin/SteepestDescent/58_2/coordsplus_033.xyz"
        }
      },
      "77": {
        "minus": {
          "pool_count": 101,
          "ranks": [
            0,
            12,
            25,
            38,
            51,
            63,
            76,
            89
          ],
          "record_ids": [
            "Landscape17/aspirin/SteepestDescent/77/coordsminus_000.xyz",
            "Landscape17/aspirin/SteepestDescent/77/coordsminus_012.xyz",
            "Landscape17/aspirin/SteepestDescent/77/coordsminus_025.xyz",
            "Landscape17/aspirin/SteepestDescent/77/coordsminus_038.xyz",
            "Landscape17/aspirin/SteepestDescent/77/coordsminus_051.xyz",
            "Landscape17/aspirin/SteepestDescent/77/coordsminus_063.xyz",
            "Landscape17/aspirin/SteepestDescent/77/coordsminus_076.xyz",
            "Landscape17/aspirin/SteepestDescent/77/coordsminus_089.xyz"
          ],
          "small_set_record_id": "Landscape17/aspirin/SteepestDescent/77/coordsminus_025.xyz"
        },
        "plus": {
          "pool_count": 104,
          "ranks": [
            1,
            15,
            30,
            44,
            59,
            73,
            88,
            103
          ],
          "record_ids": [
            "Landscape17/aspirin/SteepestDescent/77/coordsplus_001.xyz",
            "Landscape17/aspirin/SteepestDescent/77/coordsplus_015.xyz",
            "Landscape17/aspirin/SteepestDescent/77/coordsplus_030.xyz",
            "Landscape17/aspirin/SteepestDescent/77/coordsplus_044.xyz",
            "Landscape17/aspirin/SteepestDescent/77/coordsplus_059.xyz",
            "Landscape17/aspirin/SteepestDescent/77/coordsplus_073.xyz",
            "Landscape17/aspirin/SteepestDescent/77/coordsplus_088.xyz",
            "Landscape17/aspirin/SteepestDescent/77/coordsplus_103.xyz"
          ],
          "small_set_record_id": "Landscape17/aspirin/SteepestDescent/77/coordsplus_030.xyz"
        }
      },
      "78_2": {
        "minus": {
          "pool_count": 95,
          "ranks": [
            0,
            12,
            24,
            35,
            48,
            59,
            72,
            83
          ],
          "record_ids": [
            "Landscape17/aspirin/SteepestDescent/78_2/coordsminus_000.xyz",
            "Landscape17/aspirin/SteepestDescent/78_2/coordsminus_012.xyz",
            "Landscape17/aspirin/SteepestDescent/78_2/coordsminus_024.xyz",
            "Landscape17/aspirin/SteepestDescent/78_2/coordsminus_035.xyz",
            "Landscape17/aspirin/SteepestDescent/78_2/coordsminus_048.xyz",
            "Landscape17/aspirin/SteepestDescent/78_2/coordsminus_059.xyz",
            "Landscape17/aspirin/SteepestDescent/78_2/coordsminus_072.xyz",
            "Landscape17/aspirin/SteepestDescent/78_2/coordsminus_083.xyz"
          ],
          "small_set_record_id": "Landscape17/aspirin/SteepestDescent/78_2/coordsminus_072.xyz"
        },
        "plus": {
          "pool_count": 79,
          "ranks": [
            1,
            11,
            23,
            33,
            45,
            55,
            67,
            78
          ],
          "record_ids": [
            "Landscape17/aspirin/SteepestDescent/78_2/coordsplus_001.xyz",
            "Landscape17/aspirin/SteepestDescent/78_2/coordsplus_011.xyz",
            "Landscape17/aspirin/SteepestDescent/78_2/coordsplus_023.xyz",
            "Landscape17/aspirin/SteepestDescent/78_2/coordsplus_033.xyz",
            "Landscape17/aspirin/SteepestDescent/78_2/coordsplus_045.xyz",
            "Landscape17/aspirin/SteepestDescent/78_2/coordsplus_055.xyz",
            "Landscape17/aspirin/SteepestDescent/78_2/coordsplus_067.xyz",
            "Landscape17/aspirin/SteepestDescent/78_2/coordsplus_078.xyz"
          ],
          "small_set_record_id": "Landscape17/aspirin/SteepestDescent/78_2/coordsplus_067.xyz"
        }
      }
    },
    "selection_seed": 20260930,
    "split_hash": "73d5475bfc1f1158da62b939f5431b1781db886443cece2cdb297c8dd645133a",
    "subsets": {
      "diverse128": {
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
      },
      "diverse16": {
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
      },
      "narrow16": {
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
  }
}
```

## GRADIENT-PHASE-A-RESULT — Necessary constraint outcome

Frozen training-only diagnostics; no optimizer update or learning acceptance.

```json
{
  "artifact_sha256": "ed8ff3e418c136c7b8e24777658a62ff5aba0ab5fbe5891d35cbcf862d16134b",
  "pooled": {
    "diverse128": {
      "compatible_mse": 0.014083939185454,
      "floor_fraction_of_zero_force": 6.413411977582667e-07,
      "floor_mse": 9.03261621934256e-09,
      "floor_obstructs_gate": false,
      "gate": 0.001408394821807022,
      "raw_mse": 0.014083948218070219,
      "translation_mse": 6.1526045585488705e-09
    },
    "diverse16": {
      "compatible_mse": 0.01793965469434784,
      "floor_fraction_of_zero_force": 5.310908753795357e-07,
      "floor_mse": 9.527591975644952e-09,
      "floor_obstructs_gate": false,
      "gate": 0.0017939664221939815,
      "raw_mse": 0.017939664221939815,
      "translation_mse": 5.8688520855158395e-09
    },
    "narrow16": {
      "compatible_mse": 0.010133553332115677,
      "floor_fraction_of_zero_force": 7.646880861894121e-07,
      "floor_mse": 7.749013429412125e-09,
      "floor_obstructs_gate": false,
      "gate": 0.0010133561081129107,
      "raw_mse": 0.010133561081129106,
      "translation_mse": 4.644641210085968e-09
    }
  },
  "status": "completed"
}
```

## GRADIENT-CASE-01-PRE — Case preregistered

Frozen training-only diagnostics; no optimizer update or learning acceptance.

```json
{
  "acceptance": "Joint gradient identity, at least one of both FD steps per objective, immutable parameter/optimizer hashes; no learning gate change",
  "case": "fresh-A-fitting",
  "config_hash": "302b32fdb0e22c79e47d399902e8765f4dd6f7566ffb0ec59622af1d1d2f63c3",
  "rationale": "Local split loss gradients at frozen incoming parameters",
  "record_count": 16,
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
  ],
  "source_hashes": {
    "optimizer": "8fc30b9a817f70448819b69c458e2e1cacc034cde02880066eab48edebd4f01d",
    "params": "b6351d17922c7646322cb85f142c9566dc5380b6ca5676898c157d92dd93becd"
  },
  "update": 0
}
```

## GRADIENT-CASE-01-RESULT — Case outcome

Frozen training-only diagnostics; no optimizer update or learning acceptance.

```json
{
  "elapsed_seconds": 2.6062803330132738,
  "executed": true,
  "failure_reason": null,
  "retained_artifacts": {
    "observations.json": "871edf2e85cd8fdb5fdb9fc7d6119729217bacc08d4753e1a21f817e31f67cbc",
    "partial.json": "dee75a5af21c7806b348c1929cab1543b294d3b11c863385464d0d093a11acbf",
    "prospective.json": "49e76563d5c0e66561756d4aed61e344ee176a3de6e8863234598ad9d92040d3",
    "summary.json": "e368f7bb5e0f60865e932f56f7a90c169a80d17857b816782ffa4117de6bcf99"
  },
  "state_hashes_after": {
    "optimizer": "8fc30b9a817f70448819b69c458e2e1cacc034cde02880066eab48edebd4f01d",
    "params": "b6351d17922c7646322cb85f142c9566dc5380b6ca5676898c157d92dd93becd"
  },
  "state_hashes_before": {
    "optimizer": "8fc30b9a817f70448819b69c458e2e1cacc034cde02880066eab48edebd4f01d",
    "params": "b6351d17922c7646322cb85f142c9566dc5380b6ca5676898c157d92dd93becd"
  },
  "status": "completed",
  "stop_reason": null,
  "summary_sha256": "e368f7bb5e0f60865e932f56f7a90c169a80d17857b816782ffa4117de6bcf99",
  "verification_valid": true
}
```

## GRADIENT-CASE-02-PRE — Case preregistered

Frozen training-only diagnostics; no optimizer update or learning acceptance.

```json
{
  "acceptance": "Joint gradient identity, at least one of both FD steps per objective, immutable parameter/optimizer hashes; no learning gate change",
  "case": "fresh-B-fitting",
  "config_hash": "302b32fdb0e22c79e47d399902e8765f4dd6f7566ffb0ec59622af1d1d2f63c3",
  "rationale": "Local split loss gradients at frozen incoming parameters",
  "record_count": 16,
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
  ],
  "source_hashes": {
    "optimizer": "8fc30b9a817f70448819b69c458e2e1cacc034cde02880066eab48edebd4f01d",
    "params": "b6351d17922c7646322cb85f142c9566dc5380b6ca5676898c157d92dd93becd"
  },
  "update": 0
}
```

## GRADIENT-CASE-02-RESULT — Case outcome

Frozen training-only diagnostics; no optimizer update or learning acceptance.

```json
{
  "elapsed_seconds": 0.08751545893028378,
  "executed": true,
  "failure_reason": null,
  "retained_artifacts": {
    "observations.json": "f2a23f14002b4da1bb3d2653e47b7fadf8f63c4c1f051f61194bc30066520cba",
    "partial.json": "abdcaa23b7054550fb280cbf1cae22be9376852c46f1d54d93962a526b339a69",
    "prospective.json": "8e53f321235b0a68e03011e182557e7bafde4c965a1adb51ca68db199d807c79",
    "summary.json": "932362c1d7e180a4454dad3e95d628c55892df82bc1ceb1eb3f000c5be214d0e"
  },
  "state_hashes_after": {
    "optimizer": "8fc30b9a817f70448819b69c458e2e1cacc034cde02880066eab48edebd4f01d",
    "params": "b6351d17922c7646322cb85f142c9566dc5380b6ca5676898c157d92dd93becd"
  },
  "state_hashes_before": {
    "optimizer": "8fc30b9a817f70448819b69c458e2e1cacc034cde02880066eab48edebd4f01d",
    "params": "b6351d17922c7646322cb85f142c9566dc5380b6ca5676898c157d92dd93becd"
  },
  "status": "completed",
  "stop_reason": null,
  "summary_sha256": "932362c1d7e180a4454dad3e95d628c55892df82bc1ceb1eb3f000c5be214d0e",
  "verification_valid": true
}
```

## GRADIENT-CASE-03-PRE — Case preregistered

Frozen training-only diagnostics; no optimizer update or learning acceptance.

```json
{
  "acceptance": "Joint gradient identity, at least one of both FD steps per objective, immutable parameter/optimizer hashes; no learning gate change",
  "case": "fresh-C-fitting",
  "config_hash": "302b32fdb0e22c79e47d399902e8765f4dd6f7566ffb0ec59622af1d1d2f63c3",
  "rationale": "Local split loss gradients at frozen incoming parameters",
  "record_count": 128,
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
  ],
  "source_hashes": {
    "optimizer": "8fc30b9a817f70448819b69c458e2e1cacc034cde02880066eab48edebd4f01d",
    "params": "b6351d17922c7646322cb85f142c9566dc5380b6ca5676898c157d92dd93becd"
  },
  "update": 0
}
```

## GRADIENT-CASE-03-RESULT — Case outcome

Frozen training-only diagnostics; no optimizer update or learning acceptance.

```json
{
  "elapsed_seconds": 0.6467581250471994,
  "executed": true,
  "failure_reason": null,
  "retained_artifacts": {
    "observations.json": "86e590a45e3bc5b8a896ab184ccaa4622b5200c62d909fef256955d779640fa1",
    "partial.json": "adfa1b2ecab15ace385fdb5ae6ee21a4f61f183cb3732855da02c6c7d96b3d6c",
    "prospective.json": "903e3e3622af7ee9fe1016e9a762f12b6eba0577d6e8cdb0acfbcbba09d96641",
    "summary.json": "8c52ef2f528d9af3a7c448057e8e5d489bdaa55a9327980f342fbe751862384c"
  },
  "state_hashes_after": {
    "optimizer": "8fc30b9a817f70448819b69c458e2e1cacc034cde02880066eab48edebd4f01d",
    "params": "b6351d17922c7646322cb85f142c9566dc5380b6ca5676898c157d92dd93becd"
  },
  "state_hashes_before": {
    "optimizer": "8fc30b9a817f70448819b69c458e2e1cacc034cde02880066eab48edebd4f01d",
    "params": "b6351d17922c7646322cb85f142c9566dc5380b6ca5676898c157d92dd93becd"
  },
  "status": "completed",
  "stop_reason": null,
  "summary_sha256": "8c52ef2f528d9af3a7c448057e8e5d489bdaa55a9327980f342fbe751862384c",
  "verification_valid": true
}
```

## GRADIENT-CASE-04-PRE — Case preregistered

Frozen training-only diagnostics; no optimizer update or learning acceptance.

```json
{
  "acceptance": "Joint gradient identity, at least one of both FD steps per objective, immutable parameter/optimizer hashes; no learning gate change",
  "case": "final-A-fitting",
  "config_hash": "302b32fdb0e22c79e47d399902e8765f4dd6f7566ffb0ec59622af1d1d2f63c3",
  "rationale": "Local split loss gradients at frozen incoming parameters",
  "record_count": 16,
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
  ],
  "source_hashes": {
    "optimizer": "517e4854dfbaeafca9d37c054436aaad12b031e2cef430389e0fe86b1bcdff99",
    "params": "66d7a8fa4dc158428dcbed6c78c8bd57213337160939b50bd2c127e8ff885643"
  },
  "update": 2000
}
```

## GRADIENT-CASE-04-RESULT — Case outcome

Frozen training-only diagnostics; no optimizer update or learning acceptance.

```json
{
  "elapsed_seconds": 0.08842833305243403,
  "executed": true,
  "failure_reason": null,
  "retained_artifacts": {
    "observations.json": "76baa62b97b3d99e15f3ad9ef36dd11e01a1b91c55325664104cf6aa8a72d315",
    "partial.json": "0235d07bac59cd238774d4903a56cebd36e4fe12d9efd1454ade4f43559e1dd6",
    "prospective.json": "e137e849baf7ad2e212e41a0759ff9a198fd018d65fb100541539a376ffdb3b9",
    "summary.json": "0f0a6c02febf58e86f87aa18a9506c993ba2f1e6aa0bb999243cbf1a5c531e02"
  },
  "state_hashes_after": {
    "optimizer": "517e4854dfbaeafca9d37c054436aaad12b031e2cef430389e0fe86b1bcdff99",
    "params": "66d7a8fa4dc158428dcbed6c78c8bd57213337160939b50bd2c127e8ff885643"
  },
  "state_hashes_before": {
    "optimizer": "517e4854dfbaeafca9d37c054436aaad12b031e2cef430389e0fe86b1bcdff99",
    "params": "66d7a8fa4dc158428dcbed6c78c8bd57213337160939b50bd2c127e8ff885643"
  },
  "status": "completed",
  "stop_reason": null,
  "summary_sha256": "0f0a6c02febf58e86f87aa18a9506c993ba2f1e6aa0bb999243cbf1a5c531e02",
  "verification_valid": true
}
```

## GRADIENT-CASE-05-PRE — Case preregistered

Frozen training-only diagnostics; no optimizer update or learning acceptance.

```json
{
  "acceptance": "Joint gradient identity, at least one of both FD steps per objective, immutable parameter/optimizer hashes; no learning gate change",
  "case": "final-B-fitting",
  "config_hash": "302b32fdb0e22c79e47d399902e8765f4dd6f7566ffb0ec59622af1d1d2f63c3",
  "rationale": "Local split loss gradients at frozen incoming parameters",
  "record_count": 16,
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
  ],
  "source_hashes": {
    "optimizer": "b78d1d84eb8d100874d6dfb719db819789b5721e5ec9bf22a08d7a2dd4b85719",
    "params": "fd5ed13e2a7a020d61c7059330f0bf029cfec84d71558cd76b67c1bb303232b9"
  },
  "update": 2000
}
```

## GRADIENT-CASE-05-RESULT — Case outcome

Frozen training-only diagnostics; no optimizer update or learning acceptance.

```json
{
  "elapsed_seconds": 0.0853824169607833,
  "executed": true,
  "failure_reason": null,
  "retained_artifacts": {
    "observations.json": "a4475e427b8a6671390a01ce1820ba09703a39924e43b5b38dc44011996d0b38",
    "partial.json": "9765b1d85caecd49f6f19e720b8176d4c1687727aa21700743f90857822462f9",
    "prospective.json": "cea7b82f42e72392574721204c996745dbb2e4f815da3ec44522e4621cebc18b",
    "summary.json": "4a3659c9755b83419599cc89b1a199cdd02fbe524b3610c90fa3272146457f32"
  },
  "state_hashes_after": {
    "optimizer": "b78d1d84eb8d100874d6dfb719db819789b5721e5ec9bf22a08d7a2dd4b85719",
    "params": "fd5ed13e2a7a020d61c7059330f0bf029cfec84d71558cd76b67c1bb303232b9"
  },
  "state_hashes_before": {
    "optimizer": "b78d1d84eb8d100874d6dfb719db819789b5721e5ec9bf22a08d7a2dd4b85719",
    "params": "fd5ed13e2a7a020d61c7059330f0bf029cfec84d71558cd76b67c1bb303232b9"
  },
  "status": "completed",
  "stop_reason": null,
  "summary_sha256": "4a3659c9755b83419599cc89b1a199cdd02fbe524b3610c90fa3272146457f32",
  "verification_valid": true
}
```

## GRADIENT-CASE-06-PRE — Case preregistered

Frozen training-only diagnostics; no optimizer update or learning acceptance.

```json
{
  "acceptance": "Joint gradient identity, at least one of both FD steps per objective, immutable parameter/optimizer hashes; no learning gate change",
  "case": "final-C-fitting",
  "config_hash": "302b32fdb0e22c79e47d399902e8765f4dd6f7566ffb0ec59622af1d1d2f63c3",
  "rationale": "Local split loss gradients at frozen incoming parameters",
  "record_count": 128,
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
  ],
  "source_hashes": {
    "optimizer": "311d3e2925e1d6b5183ae9346a279b9047ec4fae7a9c162fb83454d758e0e87c",
    "params": "a05e254c1de7b419f2f659af0e902a137a8380f7e6a6c3bbc423947c9332e236"
  },
  "update": 2000
}
```

## GRADIENT-CASE-06-RESULT — Case outcome

Frozen training-only diagnostics; no optimizer update or learning acceptance.

```json
{
  "elapsed_seconds": 0.6350165409967303,
  "executed": true,
  "failure_reason": null,
  "retained_artifacts": {
    "observations.json": "0e57685416e8bbae6a626b4ffa275d4de772a68047c6cc97eaad0ca251520e2e",
    "partial.json": "4228f1c4ad536f2e77d477a39e43d5456869bd83b33f83da661f0d941a45fb5f",
    "prospective.json": "915c2f830a403c87045d5e36e6820f80580066837853f05a544361c86131844c",
    "summary.json": "979fbba5e20e2e5b52d5840841713b5004882c2e166f391676c4d5c7b77e6000"
  },
  "state_hashes_after": {
    "optimizer": "311d3e2925e1d6b5183ae9346a279b9047ec4fae7a9c162fb83454d758e0e87c",
    "params": "a05e254c1de7b419f2f659af0e902a137a8380f7e6a6c3bbc423947c9332e236"
  },
  "state_hashes_before": {
    "optimizer": "311d3e2925e1d6b5183ae9346a279b9047ec4fae7a9c162fb83454d758e0e87c",
    "params": "a05e254c1de7b419f2f659af0e902a137a8380f7e6a6c3bbc423947c9332e236"
  },
  "status": "completed",
  "stop_reason": null,
  "summary_sha256": "979fbba5e20e2e5b52d5840841713b5004882c2e166f391676c4d5c7b77e6000",
  "verification_valid": true
}
```

## GRADIENT-CASE-07-PRE — Case preregistered

Frozen training-only diagnostics; no optimizer update or learning acceptance.

```json
{
  "acceptance": "Joint gradient identity, at least one of both FD steps per objective, immutable parameter/optimizer hashes; no learning gate change",
  "case": "final-B-anchor",
  "config_hash": "302b32fdb0e22c79e47d399902e8765f4dd6f7566ffb0ec59622af1d1d2f63c3",
  "rationale": "Local split loss gradients at frozen incoming parameters",
  "record_count": 16,
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
  ],
  "source_hashes": {
    "optimizer": "b78d1d84eb8d100874d6dfb719db819789b5721e5ec9bf22a08d7a2dd4b85719",
    "params": "fd5ed13e2a7a020d61c7059330f0bf029cfec84d71558cd76b67c1bb303232b9"
  },
  "update": 2000
}
```

## GRADIENT-CASE-07-RESULT — Case outcome

Frozen training-only diagnostics; no optimizer update or learning acceptance.

```json
{
  "elapsed_seconds": 0.09102804190479219,
  "executed": true,
  "failure_reason": null,
  "retained_artifacts": {
    "observations.json": "18fea2408fc7ce2b320639042c6f23e515a63f722744f5874b2104a8614284d4",
    "partial.json": "2a1c278c160badfe320dae1ad1877dc8ee14082a8a86d91c069c6c20b3c3ae64",
    "prospective.json": "9bb2a18c0b89f02fff7730ba3b55f032239f473f464ab771a0bb72b449924891",
    "summary.json": "82efdcc9d90a887556fcb7189246bac5be111a060caed1e3a39af39f75528663"
  },
  "state_hashes_after": {
    "optimizer": "b78d1d84eb8d100874d6dfb719db819789b5721e5ec9bf22a08d7a2dd4b85719",
    "params": "fd5ed13e2a7a020d61c7059330f0bf029cfec84d71558cd76b67c1bb303232b9"
  },
  "state_hashes_before": {
    "optimizer": "b78d1d84eb8d100874d6dfb719db819789b5721e5ec9bf22a08d7a2dd4b85719",
    "params": "fd5ed13e2a7a020d61c7059330f0bf029cfec84d71558cd76b67c1bb303232b9"
  },
  "status": "completed",
  "stop_reason": null,
  "summary_sha256": "82efdcc9d90a887556fcb7189246bac5be111a060caed1e3a39af39f75528663",
  "verification_valid": true
}
```

## GRADIENT-CASE-08-PRE — Case preregistered

Frozen training-only diagnostics; no optimizer update or learning acceptance.

```json
{
  "acceptance": "Joint gradient identity, at least one of both FD steps per objective, immutable parameter/optimizer hashes; no learning gate change",
  "case": "final-C-anchor",
  "config_hash": "302b32fdb0e22c79e47d399902e8765f4dd6f7566ffb0ec59622af1d1d2f63c3",
  "rationale": "Local split loss gradients at frozen incoming parameters",
  "record_count": 16,
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
  ],
  "source_hashes": {
    "optimizer": "311d3e2925e1d6b5183ae9346a279b9047ec4fae7a9c162fb83454d758e0e87c",
    "params": "a05e254c1de7b419f2f659af0e902a137a8380f7e6a6c3bbc423947c9332e236"
  },
  "update": 2000
}
```

## GRADIENT-CASE-08-RESULT — Case outcome

Frozen training-only diagnostics; no optimizer update or learning acceptance.

```json
{
  "elapsed_seconds": 0.08587008400354534,
  "executed": true,
  "failure_reason": null,
  "retained_artifacts": {
    "observations.json": "34774a60fcb8fd8d06139eaa2228fe5a337d32cd3389a391f22f3841f76fecc5",
    "partial.json": "d3d6c277e20a041bb5077e2ed7d32b62c4b12a88dfd366a77ec10b6dc5edfbc4",
    "prospective.json": "70f5ca3fdbba3942b0a45e12731a920bc30717bc62e380d3c4217030f1c678ae",
    "summary.json": "71ba6b8f6f70b55d679f73e8274774714c00c69e16ac1bca280ed537ab64a27d"
  },
  "state_hashes_after": {
    "optimizer": "311d3e2925e1d6b5183ae9346a279b9047ec4fae7a9c162fb83454d758e0e87c",
    "params": "a05e254c1de7b419f2f659af0e902a137a8380f7e6a6c3bbc423947c9332e236"
  },
  "state_hashes_before": {
    "optimizer": "311d3e2925e1d6b5183ae9346a279b9047ec4fae7a9c162fb83454d758e0e87c",
    "params": "a05e254c1de7b419f2f659af0e902a137a8380f7e6a6c3bbc423947c9332e236"
  },
  "status": "completed",
  "stop_reason": null,
  "summary_sha256": "71ba6b8f6f70b55d679f73e8274774714c00c69e16ac1bca280ed537ab64a27d",
  "verification_valid": true
}
```

## GRADIENT-EXECUTION-RESULT — Frozen analysis outcome

Frozen training-only diagnostics; no optimizer update or learning acceptance.

```json
{
  "case_statuses": {
    "final-A-fitting": "completed",
    "final-B-anchor": "completed",
    "final-B-fitting": "completed",
    "final-C-anchor": "completed",
    "final-C-fitting": "completed",
    "fresh-A-fitting": "completed",
    "fresh-B-fitting": "completed",
    "fresh-C-fitting": "completed"
  },
  "elapsed_seconds": 8.960575666977093,
  "execution_valid": true,
  "failure_reason": null,
  "next_step": "separately_review_intervention_from_local_evidence",
  "phase_A_status": "completed",
  "status": "completed",
  "summary_sha256": "6d0027b14169bfaa72a6117d786412b8d0909a916ae13f4e0aa562d384907e16"
}
```

## FINDING-GRADIENT-001 — Necessary symmetry floor does not obstruct the force gates

One frozen analysis completed8.960575666977093s, execution revision b68b017. For A/B/C, pooled rigid-floor MSE7.749013429412125e-09/9.527591975644952e-09/9.03261621934256e-09 eV²/Å² versus unchanged gates0.0010133561081129107/0.0017939664221939815/0.001408394821807022. Every floor is far below its gate; necessary translation/rotation constraints alone do not explain the failed fitting. All sampled rigid ranks6; every strict5Å graph connected/no isolates/two-hop reach1; no pairs within1e-8Å of cutoff. Connectivity is not capacity/integrability sufficiency. Raw labels/gates unchanged; no calculator error or generalization claim.

## FINDING-GRADIENT-002 — Scaled energy dominance at initialization and local opposition after fitting

At fresh A/B/C, energy parameter-gradient norms247.20812241730232/47.75515555607004/72.4114827715311; force norms0.5877787788899789/0.9816442915714093/0.5319166048734169. Initial backbone energy/force gradients structurally zero because the head is zero; force bias derivative structurally zero throughout.

At final fitting A/B/C, whole-tree energy-force cosine-0.9645968197169623/-0.9924599548492967/-0.7056331347992232 (full precision in companion/raw JSON is authoritative). Raw ordinary minus-joint-gradient directional slopes are positive for A energy, B force and C force: local objective opposition. C's original-anchor cosine is positive0.3297979, so this pattern is not uniform across all observations. Parameterization and original scales matter; no actual Adam update direction or training-trajectory causality was measured. This motivates a separately reviewed loss/optimization control before assuming more data/capacity is the solution; it does not prove the failure's cause or clear any learning gate.

## FINDING-GRADIENT-003 — Directional verification with preserved unsuccessful step

All8case gradient-sum checks passed (largest component difference5.717648576819556e-15); all16objective directional checks met the predeclared at-least-one-step criterion. Of32individual steps,31passed. Final-C-fitting energy step1e-4 failed: error0.0002249861811485232 versus autodiff1.3307488075266263; step1e-5 passed with error2.2498705722906465e-06. Both are retained without retry/tuning. FD verifies selected directions, not every component. All incoming parameter/optimizer hashes unchanged; all31historical source hashes unchanged and historical predictions reproduced exactly. Full pilot remains pending; no training, optimizer application or test access occurred.

## GRADIENT-REPORT-001 — Balanced report and measured figures

Saved `reports/representation-gradient-diagnostics.md/json` and2figures, visually inspected after changing norm plots to separate points (fixed cases are not a trajectory). Generator reads preserved JSON only. Exact source and figure/generator identities are in companion JSON; retrospective interpretation supplements the per-phase/case prospective/outcome entries above. Independent final implementation/evidence review pending.

```json
{
  "compact_evidence_sha256": "27cb8d51072a8bb74fc2962c8b47355af76207bee6b3075040c723792d57152e",
  "figures": {
    "representation-force-floors.png": "db621e3145c5ee89492e871d379bde090cbad74979f049ad6b02b37dac799455",
    "representation-gradient-geometry.png": "38121a40550c1bce278b6d85b014d037e53681fdf12442bd579169310fbabaa7"
  },
  "generator_sha256": "3eb27f40704fe53292f7446a71a9c30fb40a5bc5ac69d04daf8515caeeb7613f",
  "report_sha256": "bf067a2246a170ba1a7fc98fabd3a679513d2dce9339b4db6e05d7e514dd788c",
  "summary_sha256": "6d0027b14169bfaa72a6117d786412b8d0909a916ae13f4e0aa562d384907e16"
}
```

## MEMORY-GRADIENT-001 — Project navigation refresh and verification correction

Obsidian hub/findings updated through CLI after repository outcomes, originals backed up in retained workspace. First readback assertion failed because the hub uses grouped FINDING-GRADIENT-001/002/003 rather than a standalone FINDING-GRADIENT-003 string. The write succeeded. Read-only verification of the correct grouped ID and report link passed for both notes; no duplicate append/rewrite occurred. This failed engineering assertion is preserved, not a research failure. Final independent review remains pending.

## IMPL-GRADIENT-004 — Independent review, interrupted review and balanced reporting repairs

Fresh independent final reviewer reconciled31sources/38raw artifacts/all8cases and supported retained successful observations. First reviewer turn stopped at usage limit; after user explicitly said proceed, the reviewer resumed and completed. No production diagnostic was repeated. Reviewer's exact pooled-sum assertion initially failed by ±1.6543612251060553e-24; a1e-15relative check passed, ordinary summation rounding rather than numerical research discrepancy. Focused review32original tests passed; concurrently authored routing regression failed as expected (1failed32passed10.75s).

Two Important defects: invalid snapshot execution still recommended intervention; failure/preflight reports could claim eight attempts and universal FD success. Author routing regression first failed5.45s; two report regressions first failed0.55s. After one fix pass, all14focused controller/report tests passed11.01s. Invalid execution now routes to repair; report counts/gated interpretation derive from actual outcomes. No Critical/Minor findings, no declined behaviors. Full suite pending at this entry; final completion follows verified suite. Successful raw research observations, settings, figures and previous failed steps are unchanged. Review details in `docs/reviews/2026-09-30-representation-gradient-final-review.md`.

## IMPL-GRADIENT-005 — Final verified implementation and preserved research evidence

Full final suite117passed126.17s after both Important fixes. No remaining Critical/Important review findings; no minors or declined behaviors. Independent reviewer assessed the pre-fix research evidence; author verified the two repairs through RED-to-GREEN and full suite, without a second review. The one research execution, all31historical source hashes, all38raw artifact hashes, all8case/constraint numbers and both visually inspected figures are unchanged. Previous report/generator identities in GRADIENT-REPORT-001 are historical; these refreshed identities supersede presentation hashes only. No learning gate cleared, no training/model/loss/label change, no extra attempt/pilot. Next: separately review a controlled loss/optimization investigation; local opposition is evidence, not causal attribution. Keep branch/records/user proposal changes.

```json
{
  "compact_evidence_sha256": "0f25af23455154c730e5380d9bb7e95470cc283140d68b96f0becb20ddb11477",
  "final_suite": "117passed126.17s",
  "generator_sha256": "f94d924cfa67a0aa608795ac233d23c6d1a74a0b1c23f3b11de5d104a56e875b",
  "report_sha256": "d426910c2151d1e736f3b93922dd1946a27d8e031a3a677a91d3d326032bd45a",
  "research_repeated": false,
  "review_sha256": "b8ed7e236598805506632c156af28e74815c5ce0be9582ea032de18ef7466245",
  "verification_logs": {
    "final-fixes-green.log": "2e0e7c3fb944fbdacbdcb9d41654d539096261068d58c06261811bfd899e9cda",
    "final-fixes-suite.log": "07d9b4c519b63c705f3b9c246aa73d374aee09271ec145b32b51ae1af12bf4e8",
    "final-report-red.log": "7630442897605c8c8715748a26496a62753d58037de9254bb3fa08583b923518",
    "final-routing-green.log": "e5128813d83681255ee0ac7f661e2cdc5bc4d2593317b45fe16f439f94b960fa",
    "final-routing-red.log": "7b22689ade69ce1c9f90a13d303574d484c8a6c21239412dd6877b677691ecc7"
  }
}
```

## MEMORY-GRADIENT-002 — Final navigation and artifact reconciliation

Obsidian hub/findings finalized through CLI and read back;117tests/review repairs/next reviewed control agree with repository state. Original/intermediate/final notes retained. Final31source/38raw/report/generator/2figure hashes and8immutable state cases reconcile; no numerical trial repeated.

```json
{
  "completion-ledger.md": "91c8f5e5fa340058d787e665aec715cf7392c5a10c29a21ac77fc3edee1e402c",
  "obsidian-findings-final.md": "6bd66aed1d08ea6901d9e043c8c5387a21e1e968f9f62473a24a50501b4ef6ce",
  "obsidian-hub-final.md": "631e58cc613aceb62ce554cd0a890280665fd3d1a535b695adb72159fc7cfdc5"
}
```

## PLAN-BALANCE-001 — Prospective controlled fixed loss-balance diagnostic

User Proceed authorizes the next controlled loss-balancing investigation. Saved spec/plan `docs/superpowers/{specs,plans}/2026-09-30-loss-balance-diagnostic*` and frozen config `configs/aspirin-loss-balance-diagnostic.json`; status proposed/unexecuted, two reviews pending. No product code or new numerical training trial yet. Scope:128fitting/16fully-fitted anchor, exact replay -> one fixed train-only initial-gradient balance -> force-only ablation; eachfreshseed0/2000updates/.001Adam/1800s cap, samearchitecture/settings. Calibration uses preserved fresh-C whole-gradient norm ratio136.13314964807924, weights0.014584365670390719/1.9854156343296092 (sum2), no adaptive/sweep/recalibration. All31source hashes and compact diagnostic hash pinned. Original raw energy/force gates remain; force-only force success cannot clear joint learning acceptance. Rationale: test objective tradeoff after low symmetry floors/local opposition; n=1training diagnostic, not Adam-causal/capacity/generalization claim. Failed/blocked outcomes retained prospectively and immediately. Prior missing config filename during exploration was corrected to existing aspirin-data-coverage-diagnostic.json; no state modified.

## PLAN-BALANCE-002 — Both independent plan reviews cleared

Scientific/engineering reviews cleared exact3arm protocol, no material blockers. Independently verified31coverage/38calibrationraw/compacthash/counts/update0/IDs/state/norms/weights and22historicalobservations. Accepted reporting/implementation clarifications in saved spec/review: whole-tree bias inclusion and initial-only calibration, force-only unconstrained absolute energy gauge (no posthocshift), explicit arm weights/factory agreement, dynamic default replay, both-partition replay and initial baseline reconciliation, preserved replay-comparison failure, nonzero-head derivative witness. No settings/arms/caps changed; no new training result yet.

## IMPL-BALANCE-001 — Weighted training controls, unsuccessful checks retained

Task1 missing-helper RED and Task2 missing-controller RED failed at collection as expected; both logs retained in `.superpowers/sdd/2026-09-30-loss-balance-diagnostic/`. Task1 independent weighted full-batch/uneven3/3/1 gradients, force mixed derivatives, Adam update/count, force-only bias, exact1/1default and tiny2update checkpoint/fresh-process tests passed16checks35.95s. Added caller-mutation regression failed1check6.08s: builder captured mutable dictionary; fixed by snapshotting validated coefficients. Final17focused checks passed38.75s. Task2's28focused checks passed45.34s, including pre-loader identity/formula gates, real frozen calibration, both-partition22point replay, preservation of completed replay on comparison failure, partial checkpoint retention, protected directories, valid learning failures continuing and operational CLI success. No production research training yet; tests are engineering evidence only.

## IMPL-BALANCE-002 — Report and immutable calibration regression trials

Four RED checks failed0.73s for missing report generator and missing raw-calibration recheck helper, retaining full output in task3-red.log. Recheck now verifies all38raw calibration files before/after each arm and at completion, in addition to31coverage sources/compact hash. Optional fitting-baseline guard implements accepted reviewer clarification alongside existing anchor guard; no default objective/budget/data change. Report tests cover zero-attempt preflight failure, partial17update failure, and force-only force passage without joint success. Focused GREEN/full-suite outcomes follow; all new training outcomes remain unknown.

## IMPL-BALANCE-003 — Pre-production verification

Report/controller16focused checks passed20.58s. Four metadata characterization checks passed0.55s: even consistently rehashed calibration records with wrong count/update/ordered IDs/state are rejected before the data loader. Full pre-production suite142passed128.27s; the four newly added metadata checks were separately verified afterward (no product code changed). Report consumes saved JSON only, preserves actual attempted/completed/blocked counts and gates interpretation on operational validity. All117prior tests remain green. No production numerical outcome exists at this point. Starting exact reviewed configuration once after local implementation commit; every arm is freshly initialized, controls/gates/source pins remain fixed, and operational/replay failure blocks later arms.

## BALANCE-EXECUTION-PRE — Frozen fixed loss controls preregistered (2026-09-30)

Training-only diagnostic; no scientific generalization claim; full pilot remains pending.

```json
{
  "config": {
    "arms": {
      "balanced": {
        "energy": 0.014584365670390719,
        "force": 1.9854156343296092
      },
      "force-only": {
        "energy": 0.0,
        "force": 2.0
      },
      "replay": {
        "energy": 1.0,
        "force": 1.0
      }
    },
    "backend": "cpu",
    "calibration_case": "fresh-C-fitting",
    "calibration_energy_norm": 72.41148277153111,
    "calibration_force_norm": 0.531916604873417,
    "calibration_path": "reports/representation-gradient-diagnostics.json",
    "calibration_ratio": 136.13314964807924,
    "calibration_run": "runs/aspirin-representation-gradient-2026-09-30",
    "calibration_sha256": "0f25af23455154c730e5380d9bb7e95470cc283140d68b96f0becb20ddb11477",
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
    "replay_tolerance": {
      "atol": 1e-10,
      "rtol": 1e-08
    },
    "run_directory": "runs/aspirin-loss-balance-2026-09-30",
    "seed": 0,
    "source_run": "runs/aspirin-data-coverage-2026-09-30",
    "source_sha256": {
      "config.json": "3575dad49acf97b8f780f953a397c776cb5f3dd96729f1e6236837d8e99a5f64",
      "diverse128/anchor-inputs.npz": "5c638b71fe14615e53ae0b27a0c69c1fff1fd572f9f7191130cb2116e86264fc",
      "diverse128/checkpoint-verification.json": "cf07aa2554610daac41ef3e83a1335092e751bf8db7ecc30ccf2cf5128f1a319",
      "diverse128/config.json": "dfbdb9129e00a2984341a1efd80e901fda20565d3bcc7cc53ae04d2ed5ac6b51",
      "diverse128/fitting-inputs.npz": "9c56de4bae8058f4c18aab338aaee4e45c42f527905e48656f49900239fa701f",
      "diverse128/history.json": "2e037af82e7fae42f7811fb661b39202b331bb42e8499ec45c0c3ba73d18629f",
      "diverse128/last.msgpack": "7305113575499ebe51ca3e30b87d5bb644888b913b845e3fee5b95e222f4f6c3",
      "diverse128/provenance.json": "1502d258d63cf8a69e9ee67fcd9852855b9dbc3deb2010c33010d8b56db81063",
      "diverse128/reference.npz": "5ebb9dfb89f2c5df095a8d6f1dfe5fa2e1901b6dcfcf3159b417ccad5b724afd",
      "diverse128/summary.json": "0e374b8ed313f8004aec6219af3b2e45f0a868082f00ced907a6f4c9d8b88a8b",
      "diverse16/anchor-inputs.npz": "5c638b71fe14615e53ae0b27a0c69c1fff1fd572f9f7191130cb2116e86264fc",
      "diverse16/checkpoint-verification.json": "f39a1a3d6cacfc6eb6418e7a695bb550ac9350bc9e08cd2aa20d80ad256a6295",
      "diverse16/config.json": "dfbdb9129e00a2984341a1efd80e901fda20565d3bcc7cc53ae04d2ed5ac6b51",
      "diverse16/fitting-inputs.npz": "ae6550bbda77af6eb1617a859cc8ce118ab90c6bd109c3a25b0d3827faf9a65c",
      "diverse16/history.json": "f9b66c8e496ba8ca82d626df2a38980f41c7dc1edf28bb97f21028a8aa8fc367",
      "diverse16/last.msgpack": "7156487d43587f9cdfe563f14db79948d465b0bf43995ff67e07e3c6ac180d8f",
      "diverse16/provenance.json": "80bed026608d4455f4ab81a1fbc17520c511a1772c39fc448eada96f6dd34791",
      "diverse16/reference.npz": "f48c42e9697e2c93a6a8393c1378d3faecf1ad08a3c598518cf39fafae2c1c85",
      "diverse16/summary.json": "af772ad611b5d153b85265861375419df80f2b93c20407c0304f6ca9876fb5e6",
      "narrow16/anchor-inputs.npz": "5c638b71fe14615e53ae0b27a0c69c1fff1fd572f9f7191130cb2116e86264fc",
      "narrow16/checkpoint-verification.json": "f39a1a3d6cacfc6eb6418e7a695bb550ac9350bc9e08cd2aa20d80ad256a6295",
      "narrow16/config.json": "dfbdb9129e00a2984341a1efd80e901fda20565d3bcc7cc53ae04d2ed5ac6b51",
      "narrow16/fitting-inputs.npz": "5c638b71fe14615e53ae0b27a0c69c1fff1fd572f9f7191130cb2116e86264fc",
      "narrow16/history.json": "9970a673c501b566e8f8de0b83581ce81f40cba9a71a2547111f9fe46b0af812",
      "narrow16/last.msgpack": "b5c792314ea6b34d46bc5b2b104abbc7b1e647fd6c81cea55ec49e94b18cb6b7",
      "narrow16/provenance.json": "5de39bfac0af7376beee7e9790b6028443cff7d8d3983006b4a1ba0b5617b6ad",
      "narrow16/reference.npz": "95f1258fba9882b7c7ae600f704f0465e7c6011ef149d2127ccf5a414ee9ecaa",
      "narrow16/summary.json": "c3b0aa387b7cbd2cc57723e6232677eea19abb3f03dc1834cd8cdf1b32128452",
      "provenance.json": "b60d13789966f2ce2fbb11a3c4de73f0886f2e952ba9e0254892c35c554f7dcc",
      "subsets.json": "d3c3db7a92f3ffb0748039f63d7a0734bf38214032037ab67780d6182ad764b6",
      "summary.json": "54e7f6d7e9995dbca90bdca3417457b075e99037dc9ecd6d6a1c0647ac102a9f"
    },
    "status": "planned_unexecuted",
    "subsets_manifest": "manifests/aspirin-data-coverage-subsets-planned.json",
    "validation_interval": 100,
    "wall_seconds": 1800
  },
  "config_hash": "50f4b550d178fcd0fb219d97d6658f94774700d1c43182ea765915529b1205f7",
  "limits": "One exact3armmatrix; no sweep/retry/adaptation; training-only fitted anchor"
}
```

## BALANCE-REPLAY-PRE — Arm preregistered (2026-09-30)

Training-only diagnostic; no scientific generalization claim; full pilot remains pending.

```json
{
  "acceptance": "Original raw energy AND force gates on fitting/anchor; force-only force success reported separately",
  "anchor_ids": [
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
  "arm": "replay",
  "config_hash": "410c77ee6e7a5a5c0b74577455971ecde7a096784bd28c328f9ce77ba515097d",
  "expected_initial_hashes": {
    "optimizer": "8fc30b9a817f70448819b69c458e2e1cacc034cde02880066eab48edebd4f01d",
    "params": "b6351d17922c7646322cb85f142c9566dc5380b6ca5676898c157d92dd93becd"
  },
  "limits": "n=1; anchor fully fitted; fixed whole-tree initialization calibration; force-only energy gauge unconstrained",
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
  ],
  "updates": 2000,
  "wall_seconds": 1800,
  "weights": {
    "energy": 1.0,
    "force": 1.0
  }
}
```

## BALANCE-REPLAY-RESULT — Arm outcome (2026-09-30)

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
    "anchor-inputs.npz": "5c638b71fe14615e53ae0b27a0c69c1fff1fd572f9f7191130cb2116e86264fc",
    "checkpoint-verification.json": "cf07aa2554610daac41ef3e83a1335092e751bf8db7ecc30ccf2cf5128f1a319",
    "config.json": "ce9bed374dd8e43ee0d3f742e11ded968aa6ecb998b2a3a7023739c8fdcde105",
    "fitting-inputs.npz": "9c56de4bae8058f4c18aab338aaee4e45c42f527905e48656f49900239fa701f",
    "history.json": "a0dec9c24f1a4d0b38fcff61e66bf28bc979c99e7dd5c04c01bbbc354f0b73a5",
    "last.msgpack": "1966c79103e542e9bcf2b7deceebf0102e182a7b1532b3398c46281bdc53a831",
    "provenance.json": "ccae192660c6b4adb657cd30d58fdb2c2482fbbddea011e0b042464e7eddc88c",
    "reference.npz": "5ebb9dfb89f2c5df095a8d6f1dfe5fa2e1901b6dcfcf3159b417ccad5b724afd",
    "summary.json": "125f303c78710b7697d6e829da82e2903c8b4f2760c165825fb3fb4b99b5237b"
  },
  "elapsed_seconds": 243.63797516690101,
  "executed": true,
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
  "updates": 2000,
  "warmup_frame_exposures": 128
}
```

## BALANCE-BALANCED-PRE — Arm preregistered (2026-09-30)

Training-only diagnostic; no scientific generalization claim; full pilot remains pending.

```json
{
  "acceptance": "Original raw energy AND force gates on fitting/anchor; force-only force success reported separately",
  "anchor_ids": [
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
  "arm": "balanced",
  "config_hash": "80b5c491bbd77527730318721e456699b553d9097d6f93e2b10e4d9cb2ec8997",
  "expected_initial_hashes": {
    "optimizer": "8fc30b9a817f70448819b69c458e2e1cacc034cde02880066eab48edebd4f01d",
    "params": "b6351d17922c7646322cb85f142c9566dc5380b6ca5676898c157d92dd93becd"
  },
  "limits": "n=1; anchor fully fitted; fixed whole-tree initialization calibration; force-only energy gauge unconstrained",
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
  ],
  "updates": 2000,
  "wall_seconds": 1800,
  "weights": {
    "energy": 0.014584365670390719,
    "force": 1.9854156343296092
  }
}
```

## BALANCE-BALANCED-RESULT — Arm outcome (2026-09-30)

Training-only diagnostic; no scientific generalization claim; full pilot remains pending.

```json
{
  "anchor_gate": {
    "accepted": false,
    "fraction_remaining": {
      "energy_mse_eV2": 0.14981029569238652,
      "force_mse_eV2_per_A2": 0.5562927944341675
    },
    "gate_fraction": 0.1,
    "threshold_mse": {
      "energy_mse_eV2": 0.00612983585312839,
      "force_mse_eV2_per_A2": 0.0010133561081129107
    }
  },
  "artifact_sha256": {
    "anchor-inputs.npz": "5c638b71fe14615e53ae0b27a0c69c1fff1fd572f9f7191130cb2116e86264fc",
    "checkpoint-verification.json": "cf07aa2554610daac41ef3e83a1335092e751bf8db7ecc30ccf2cf5128f1a319",
    "config.json": "446d95ef952249d64ca5598d2c9ad713af96c7758b87190b5beab2a1ec4da589",
    "fitting-inputs.npz": "9c56de4bae8058f4c18aab338aaee4e45c42f527905e48656f49900239fa701f",
    "history.json": "3d5f8541f36996b67cc061a753dbeafdf110d76a14306964ced5388b03973384",
    "last.msgpack": "bdc60383661ec8000d1d5e76918560390432d194e0b169eb2758729dce97e099",
    "provenance.json": "46bc63737b77c3aec012432ee0a41f1324319e1b4185fe0556d5dc83d42d13d7",
    "reference.npz": "a87b9890d8ea7df874a182a70573b146afc795a36a2566c4966a3b2d2cd3466e",
    "summary.json": "abde4733a1119ade837775fedb5ed93ef5fcebbd2543cce1b51bde1305005aa9"
  },
  "elapsed_seconds": 242.5499108750373,
  "executed": true,
  "execution_valid": true,
  "failure_reason": null,
  "fit_gate": {
    "accepted": false,
    "fraction_remaining": {
      "energy_mse_eV2": 0.4220107073579012,
      "force_mse_eV2_per_A2": 0.6411855899465196
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
  "updates": 2000,
  "warmup_frame_exposures": 128
}
```

## BALANCE-FORCE-ONLY-PRE — Arm preregistered (2026-09-30)

Training-only diagnostic; no scientific generalization claim; full pilot remains pending.

```json
{
  "acceptance": "Original raw energy AND force gates on fitting/anchor; force-only force success reported separately",
  "anchor_ids": [
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
  "arm": "force-only",
  "config_hash": "bcadd5ed7b1d0f56935c401d1abe8ebcbb2c6e7903b4379a6a7b840ba8593406",
  "expected_initial_hashes": {
    "optimizer": "8fc30b9a817f70448819b69c458e2e1cacc034cde02880066eab48edebd4f01d",
    "params": "b6351d17922c7646322cb85f142c9566dc5380b6ca5676898c157d92dd93becd"
  },
  "limits": "n=1; anchor fully fitted; fixed whole-tree initialization calibration; force-only energy gauge unconstrained",
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
  ],
  "updates": 2000,
  "wall_seconds": 1800,
  "weights": {
    "energy": 0.0,
    "force": 2.0
  }
}
```

## FINDING-BALANCE-001 — Fixed balance improves force error but loses energy acceptance

Replay completed2000updates and exhaustive fresh-process verification; all22both-partition E/F observations matched historical C exactly (maximum MSE difference0). Balanced completed2000updates/256000exposures, same fresh parameter/optimizer hashes, fresh-process verified,242.5499108750373s. Its fitting force fraction0.6411855899465196 (35.88144100534804% reduction from initial) improves over replay0.9784403043672875, while fitting energy fraction0.4220107073579012 fails the unchanged<0.1gate (replay0.08738058573272221 passed energy alone). Anchor E/F fractions0.14981029569238652/0.5562927944341675 also fail joint acceptance. This is a measured fixed-loss tradeoff, n=1fully-fitted train anchor; no causal attribution, tuning selection, capacity verdict or held-out claim. Force-only arm running; no full matrix interpretation yet.

```json
{
  "balanced": {
    "anchor_gate": {
      "accepted": false,
      "fraction_remaining": {
        "energy_mse_eV2": 0.14981029569238652,
        "force_mse_eV2_per_A2": 0.5562927944341675
      },
      "gate_fraction": 0.1,
      "threshold_mse": {
        "energy_mse_eV2": 0.00612983585312839,
        "force_mse_eV2_per_A2": 0.0010133561081129107
      }
    },
    "fit_gate": {
      "accepted": false,
      "fraction_remaining": {
        "energy_mse_eV2": 0.4220107073579012,
        "force_mse_eV2_per_A2": 0.6411855899465196
      },
      "gate_fraction": 0.1,
      "threshold_mse": {
        "energy_mse_eV2": 0.003751676427546992,
        "force_mse_eV2_per_A2": 0.001408394821807022
      }
    }
  },
  "balanced_summary_sha256": "abde4733a1119ade837775fedb5ed93ef5fcebbd2543cce1b51bde1305005aa9",
  "replay": {
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
    }
  },
  "replay_summary_sha256": "125f303c78710b7697d6e829da82e2903c8b4f2760c165825fb3fb4b99b5237b"
}
```

## BALANCE-FORCE-ONLY-RESULT — Arm outcome (2026-09-30)

Training-only diagnostic; no scientific generalization claim; full pilot remains pending.

```json
{
  "anchor_gate": {
    "accepted": false,
    "fraction_remaining": {
      "energy_mse_eV2": 55075.171126502115,
      "force_mse_eV2_per_A2": 0.5059369229844992
    },
    "gate_fraction": 0.1,
    "threshold_mse": {
      "energy_mse_eV2": 0.00612983585312839,
      "force_mse_eV2_per_A2": 0.0010133561081129107
    }
  },
  "artifact_sha256": {
    "anchor-inputs.npz": "5c638b71fe14615e53ae0b27a0c69c1fff1fd572f9f7191130cb2116e86264fc",
    "checkpoint-verification.json": "cf07aa2554610daac41ef3e83a1335092e751bf8db7ecc30ccf2cf5128f1a319",
    "config.json": "52bcb13d741a24cdde22644245f0bc9a8a20704aa9bf62640887e61ef8b99d18",
    "fitting-inputs.npz": "9c56de4bae8058f4c18aab338aaee4e45c42f527905e48656f49900239fa701f",
    "history.json": "3eedbee2a881391caf0c01dc811f586ce77f1785f56e8e71e35b8b8a0a12b269",
    "last.msgpack": "3b90337de97438f9ac5e83143265403a9fce59a175cac5653cec9e5077e00224",
    "provenance.json": "ee3745b7beec73306168cbdc94588ac940a29be770fdda780d0fe9dacfe9bfe3",
    "reference.npz": "07f5aacc1cbe91b57f21de888f14416ace7a10b3f45496184eec5adfaf3ad600",
    "summary.json": "42e9158b66498ee9365911abeb697257ccdae6448a58743bebb850ce801cef7e"
  },
  "elapsed_seconds": 232.94204191700555,
  "executed": true,
  "execution_valid": true,
  "failure_reason": null,
  "fit_gate": {
    "accepted": false,
    "fraction_remaining": {
      "energy_mse_eV2": 90243.08877300369,
      "force_mse_eV2_per_A2": 0.5699438664822202
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
  "updates": 2000,
  "warmup_frame_exposures": 128
}
```

## BALANCE-EXECUTION-RESULT — Fixed loss matrix outcome (2026-09-30)

Training-only diagnostic; no scientific generalization claim; full pilot remains pending.

```json
{
  "arm_statuses": {
    "balanced": "completed",
    "force-only": "completed",
    "replay": "completed"
  },
  "balanced_joint_accepted": false,
  "execution_valid": true,
  "failure_reason": null,
  "force_only_force_accepted": false,
  "next_step": "review_fixed_loss_outcomes_before_new_intervention",
  "source_hashes_unchanged": true,
  "status": "completed",
  "summary_sha256": "7d74cb95ae16f7503da5098fcfb0a6b7bbf9f74a6edc975238dac47a154bdd49"
}
```

## FINDING-BALANCE-002 — Force-only improves fitting but fails the unchanged force gate

Exact3arm matrix completed validly; no retries, adaptation, extra updates or test/validation evaluation. Force-only2000updates/256000training exposures,128discarded warmup exposures,232.94204191700555s, same fresh state and exhaustive fresh-process checkpoint verification. Fitting force fraction0.5699438664822202 (43.00561335177798% reduction from initial), anchor force fraction0.5059369229844992 (49.40630770155008% reduction), both remain above0.1. Balanced and force-only acceptance are false; full pilot remains pending. Force-only raw energy fractions90243.08877300369/55075.171126502115 are retained with unchanged offset: absolute energy gauge is unconstrained, and these failures alone do not prove destructive coupling. Removing the energy objective did not resolve force fitting at this fixed budget. No capacity theorem, Adam-causal finding, verdict on other weights/optimizers/budgets or held-out result. Next: review fixed outcomes before another separately twice-reviewed intervention.

```json
{
  "all_sources_unchanged": true,
  "execution_revision": "176af0427b053a5094f8c1e221193defd9a3a507",
  "force_only": {
    "anchor_gate": {
      "accepted": false,
      "fraction_remaining": {
        "energy_mse_eV2": 55075.171126502115,
        "force_mse_eV2_per_A2": 0.5059369229844992
      },
      "gate_fraction": 0.1,
      "threshold_mse": {
        "energy_mse_eV2": 0.00612983585312839,
        "force_mse_eV2_per_A2": 0.0010133561081129107
      }
    },
    "elapsed_seconds": 232.94204191700555,
    "fit_gate": {
      "accepted": false,
      "fraction_remaining": {
        "energy_mse_eV2": 90243.08877300369,
        "force_mse_eV2_per_A2": 0.5699438664822202
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
    "warmup_frame_exposures": 128
  },
  "summary_sha256": "7d74cb95ae16f7503da5098fcfb0a6b7bbf9f74a6edc975238dac47a154bdd49"
}
```

## VERIFY-BALANCE-001 — Saved-artifact reconciliation and retained verifier/report failures

Independent JSON/checkpoint reconciliation verified all31coverage+38raw-calibration+1compact hashes, exact22observations/arm, both fixed initial baselines, ordered input IDs,128fit/16fully-fitted anchor,8064/1008force components, arm coefficient config/provenance agreement, fresh state identities,2000serialized Adam counts/update counters,256000exposures, checkpoint hashes and fresh-process probes, all final raw gates and replay0difference. The first auxiliary reconciliation script failed with KeyError anchor_frame_count: that summary key does not exist; corrected verifier uses saved checkpoint-verification anchor_record_count16. Original failure log retained, corrected reconciliation passed; no research rerun. First report build succeeded with nonblocking font-cache permission warnings, retained. Visual inspection/readback found duplicated historical provenance and broken table rows; regression failed1check9.21s, repair preserves all raw records/hashes and renders equal-group metrics as rows, omits nonmetric flags and redundant provenance copies. Presentation-only rebuild/full final verification follows. Initial replay preflight/initialization overlapped focused verification; observed elapsed times are not clean throughput comparisons.

## REPORT-BALANCE-001 — Verified measured report and navigation

Final147full-suite checks passed117.17s after report regression repair; alloriginal117checks remain covered. JSON-only report rebuilt from saved3arm outcomes, tables/force and energy figures visually inspected; all22scheduled observations retained, no bestcheckpoint/tuning. Compact presentation omits duplicated historical provenance while raw records and complete checksums remain preserved. Both loss-control acceptance flags false; full pilot pending. Source/hash/checkpoint independent reconciliation passed. Obsidian hub/findings refreshed viaCLI with backups/readback; final independent review pending, no production rerun.

```json
{
  "compact_sha256": "c7c35276a38e4f17b785510e745eb75e8c76b2fd3f591b787a8bc43b7f034e4f",
  "figures": {
    "loss-balance-energy.png": "720895c780df417c17063514328be70e41cbaca43d191171a3c5101b30e15838",
    "loss-balance-force.png": "6499f0b7135971a09c06efa26064120191a9f186e12dbd81d2c863fea9ac2250"
  },
  "final_suite": "147passed117.17s",
  "generator_sha256": "405dc96c6f1f77d0112c543df8ecdc8a4255eb033e4c51798343d0c84d5ef5aa",
  "reconciliation_sha256": "c340a83cf727c94e8c38d0e097399d541735567de4b238069452d244d3fb994a",
  "report_sha256": "e83aa5dddd6115c8329195b577109803973707cba7ff6f8cc118d83f4375a37a",
  "summary_sha256": "7d74cb95ae16f7503da5098fcfb0a6b7bbf9f74a6edc975238dac47a154bdd49"
}
```

## IMPL-BALANCE-004 — Independent final review and replay-tolerance guard repair

Fresh independent gpt-6-astra reviewer confirmed all31coverage+38raw-calibration+compact/report/figure identities,88exactreplay values, input/config/weights, endpoint gates and2000serializedAdamcounts.19focused regressions passed10.65s, two numerical/source-loading fixtures intentionally deselected; fullsuite inspected as author evidence. One Important defect: preflight did not enforce prescribed replay tolerance, permitting altered config to loosen identity acceptance. Current measured configuration used exact1e-10/1e-8 and matched all88values exactly, so actual results remain valid. No Critical/minors. Review in docs/reviews/2026-09-30-loss-balance-final-review.md, witness/reconciliation/logs retained. Declined broaderweights/capacity/optimizer/budgets/heldout/cause/pilot/newproduction judgments remain outside scope, as prescribed.

Author verified root cause: configured tolerance forwarded unchecked. Four preloader regressions failed0.84s; complete five missing/loose/nonfinite/malformed/extra-key regressions failed0.78s. Narrow exacttolerance guard now rejects before source/data access, tests also require allthree arms blocked on altered config. Focused/full outcomes follow. This enforces existing reviewed settings, no material protocol revision or research rerun. Reviewer evaluated pre-fix code and supported measured artifacts; author verification, not a second review, will establish the repair.

## IMPL-BALANCE-004A — Nonfinite fixture contract correction

First repair-focused run retained1failed25passed10.98s. The preloader rejection passed, but the NaN downstream fixture expected saved blocked summaries for a configuration that cannot be serialized by existing strictJSON writer (allow_nan=False). Stack trace identified rejection before config persistence/data loading, not weakened tolerance acceptance. Corrected the engineering assertion: missing/loose/malformed/extra-key JSON configurations save blocked arms; nonfinite input raises strictJSON ValueError before any arm exists. No trainer/source/research change; exactpreflight tolerance enforcement remains. Originalfailed log and corrected outcome retained. This supplements the earlier generic blocked-arm claim for unpersistable nonfinite input.

## IMPL-BALANCE-005 — Final verified review repair and unchanged research evidence

Exacttolerance guard and corrected strictJSON engineering fixtures passed26focused checks11.55s; full152checks passed110.24s. No remaining Critical/Important findings or minors; reviewer independently assessed pre-fix evidence, author verified repair without secondreview. Measuredmatrix executed once, all3arms valid, balancedjoint/forceonlyforce gatesfalse, fullpilotpending; original/recent failedlearning trials retained. All31coverage+38rawcalibration+compact/run/report/generator/figures unchanged after guard repair. No additionaltraining, newweights/adaptation/seed/architecture/budget/GPU/test/validation trial. Next: review fixedlossoutcomes beforeanother separatelytwice-reviewed intervention. Localbranch/reviewfolders preserved; proposal.pdf deletion and Group9.pdf untouched.

```json
{
  "compact_sha256": "c7c35276a38e4f17b785510e745eb75e8c76b2fd3f591b787a8bc43b7f034e4f",
  "final_suite": "152passed110.24s",
  "focused": "26passed11.55s",
  "generator_sha256": "405dc96c6f1f77d0112c543df8ecdc8a4255eb033e4c51798343d0c84d5ef5aa",
  "logs": {
    "final-fix-suite.log": "7899a8084a94975ee5a582d823a5784e427e85bbacd190741e559fecb3e191f5",
    "final-reconciliation.log": "c340a83cf727c94e8c38d0e097399d541735567de4b238069452d244d3fb994a",
    "final-tolerance-complete-red.log": "4031cf6c5c51e7f4463e1da913c4515ce756f0cc4fae318ea43faf5113e93a51",
    "final-tolerance-corrected-green.log": "fa2b08ca2668c03cf3aa5fffaf0370c8383576c405fe3c16e78ced35b4263b6d",
    "final-tolerance-green.log": "dbc261032a720bf6aec18c92f33283748d9b8107301750adb0a77805d9c04c64",
    "final-tolerance-red.log": "ba2188d431753956b603ec19c7eb2c6b19a28bde437dec0b0fb9347aebc0ea2c"
  },
  "report_sha256": "e83aa5dddd6115c8329195b577109803973707cba7ff6f8cc118d83f4375a37a",
  "research_repeated": false,
  "review_sha256": "e450ef4e87a20b58d429a8abf16346396161b84f3f291f0f9807a852423bc60e",
  "summary_sha256": "7d74cb95ae16f7503da5098fcfb0a6b7bbf9f74a6edc975238dac47a154bdd49"
}
```

## MEMORY-BALANCE-001 — Final navigation verification and retained assertion failure

Obsidian hub/findings finalized throughCLI, backups/readbacks retained. Hub verified immediately. First findings assertion failed because it searched the entire append-only findings history for any reviewpending text, including an older gradient-stage note followed by its completion. Current loss-control section already correctly stated reviewcomplete/guardrepair/152tests. Scoped currentsection read-only verification passed; no duplicate note append/write occurred. Repository retains authoritative numerical results; navigation agrees on failedlearninggates, pendingpilot and separatelyreviewed nextintervention.

## ARCHIVE-BALANCE-001 — Durable completion records, preserved local workspace

Complete verified execution/review/test/notes/failedcheck records copied to reports/execution/loss-balance-2026-09-30; 47artifact byte/hash identities verified, reviewpackage losslessly compressed and verified. Original workspace/allreviewfolders and allrawruns remain in place. ManifestSHA256 ebab5da3dc451e31c66cbaedd72ffa797143c9e76a5425841f0852e68cb8d8f3. Localbranch retained; no merge/push/cleanup. No new product change after152test green; no numerical research rerun.

## PLAN-OPTSTEP-001 — Prospective frozen optimizer-step diagnosis

Latest user Proceed authorizes the next twice-reviewed force-fitting investigation. Saved design/plan/config for immutable optimizer-step observations: three fixed loss-control2000update checkpoints,128fit/16fully-fittedanchor; no trainingcontinuation or sourcewrites. Compare exact saved-state Adam proposal with two fixed equal-norm references, temporary0.1/1displacements, bothobjectives/roles, twoFDsteps and rawfixedbaseline errors. Prespecified100raw+2compact sourcegates beforearrays,600s totalcap, allstates unchanged/0trainingexposures. Status proposed/unexecuted; two independent reviews pending. Rationale: recordedforceonly fittingMSE declined0.008694317757203524 at1500to0.008027059902742316 at2000, but this is a fixed-window sourcecurve observation, not convergence/capacity proof. Balanced and replay also declined across samewindow; allendpoints remainfailedgates. No slope/probe computed yet, no optimizer/capacity/budget/LR/weights/labels change, no sweep/selectedresult/GPU/pilot/test/validation. Every planned/failed/blocked/actualresult retained; preliminarychoices grounded in existingresults, no newnumericaltrial before reviews.

## PLAN-OPTSTEP-002 — Scientific clearance and operation-accounting clarification

Scientific reviewer cleared protocol (implementation/numericalresults still unexecuted). Independently verified100distinctraw+2compact hashes,31losscompactindex,3statecount2000/finitefloat64mu/nu/nonnegativevariance, forceonlybias/momentszero, identicalarrays/scales/initidentities/weights/22histories andfixed-window observations. Reviewer first auxiliary decoder raised KeyError update by treating checkpointenvelope asstate; decoding nestedpayload then passed, no writes/numericalprobe. Engineering metadata verification confirms sameidentities; finalverdictpending. Clarified synchronizedcompleted-frame accounting withstarted/incomplete operations for interruptedcalls, isolatedcachedgradient chunkwrappers forfinebudgetcounts, no inventedwork or numericchange. FDpassnotrobusttiny-sign certificate andnormmatchnotperparameter/computematch. No newresearchtrial yet.

## PLAN-OPTSTEP-003 — Both reviews cleared before native implementation

Both independent reviews cleared exact3case protocol/oneexecution with no materialblocks. Metadata102pins/sourcecounters/moments/weights/inputs/gates verified; no numericalprobes. Followup correctedengineeringoverstrongoldloaderchunkinstrumentationwording: declaredopaqueparentisallowed with600s budgetbeforeafter,208completedpredictionframes onlyafterreturn, interruptedcallstarted/unconfirmed, newlossourcesstrictJSONgatebeforearrays andchunkpredictions. Fullavailablematrix36FDcells/72steps/18probes; deriveactualcounts, zerog/nonzeromomentumoracle, nofakezero-work orverification. Acceptedcautions/countclarifications savedspec/review/ledger; no settings/controlchange. Productimplementation beginsnative/testfirst, everyfailure/outcome retained.

## OPTSTEP-ENGINEERING-001 — Numeric diagnostic RED/GREEN

Task1 test-first collection failed as intended for missing optimizer_diagnostics (task1-red.log); no numerical research ran. Implemented saved Adam moments/count+1/epsilon independent formula, equal-norm hypothetical controls, both unit-direction FD steps/objectives/roles, fixed probes and synchronized started/completed per-chunk work records. Focused engineering checks: 5 passed in7.66s (task1-green.log). Scalar, quadratic and seven-frame model witnesses verify zero-gradient/nonzero-momentum behavior, weighted/builder identity, immutable states,12FDcells/24steps/6probes and168FD/84probe completed frame passes. These are engineering checks, not Landscape17 research outcomes. Sources unchanged; production remains unexecuted. Next: strict source controller.

## OPTSTEP-ENGINEERING-002 — Controller source schema correction

Task2 RED: missing optimizer_runs,1collectionerror0.05s. First focused run:22passed/1failed8.71s. The real preflight rejected compact provenance because the report intentionally omits historical_reference and per-arm provenance; independent raw/compact comparisons now use those exact documented projections, while checking full historical raw provenance separately. Auxiliary read-only schema inspection also queried a nonexistent compact execution key (KeyError), corrected by reading report generator. No production/model probes ran; all source bytes preserved. Rehashed invalid metadata remains an engineering rejection trial. Focused retry pending.

## OPTSTEP-ENGINEERING-003 — Selected subset identity schema

Second focused run22passed/1failed11.15s (task2-green-corrected.log): selected subset payload carries no identity key; planned manifest identity is separately verified by parent preflight. Corrected the extra source-provenance comparison to the frozen expected_subsets_identity already audited against planned manifest, retaining exact selected-payload comparison. This failed at a source gate before new checkpoint probes; no scientific result. Retry pending.

## OPTSTEP-ENGINEERING-004 — Source/controller verification GREEN

Task2 final focused23passed10.67s; expanded27passed11.41s (task2-green-final.log/task2-expanded.log). Real read-only preflight checks100raw+2compact identities,128fitting/16anchor memberships/input arrays, three final Adam/update2000 states, original initial/final raw metrics, saved predictions and immutable source files. Completed prediction accounting208parent+432loss-source frames. Engineering corruptions of settings/pins/rehashedweights/rehashedsavedpositions/loadedAdamcount fail, and interruption retains started-unconfirmed work; invalid/exception/budget case blocks later cases while retaining partials. Full matrix unexecuted; no research conclusions from these tests. Next: JSON-only report and whole-suite verification before exact once-only research invocation.

## OPTSTEP-ENGINEERING-005 — Report and whole-suite gates before production

JSON-only report RED6failed0.04s (missing generator); GREEN6passed8.75s. Report derives actual FD/probe counts from retained cells/steps/probes, preserves unavailable/started-unconfirmed work, rejects forged verification success, and never promotes probe outcomes to learning/full-pilot acceptance. Full regression suite185passed124.95s (full-before-production.log), including152prior checks; diff whitespace check passed. Exact research configuration has not run yet. Code will be committed before its single invocation. Obsidian sandbox CLI initially could not connect; authorized scoped CLI read succeeded with host access, both notes backed up in retained workspace. No vault content changed yet.

## OPTSTEP-EXECUTION-PRE — Frozen optimizer diagnostics preregistered

Frozen training-only diagnostics; no optimizer update or learning acceptance.

```json
{
  "config": {
    "adam": {
      "b1": 0.9,
      "b2": 0.999,
      "eps": 1e-08,
      "eps_root": 0.0,
      "nesterov": false
    },
    "arms": {
      "balanced": {
        "energy": 0.014584365670390719,
        "force": 1.9854156343296092
      },
      "force-only": {
        "energy": 0.0,
        "force": 2.0
      },
      "replay": {
        "energy": 1.0,
        "force": 1.0
      }
    },
    "backend": "cpu",
    "calibration_case": "fresh-C-fitting",
    "calibration_energy_norm": 72.41148277153111,
    "calibration_force_norm": 0.531916604873417,
    "calibration_path": "reports/representation-gradient-diagnostics.json",
    "calibration_ratio": 136.13314964807924,
    "calibration_run": "runs/aspirin-representation-gradient-2026-09-30",
    "calibration_sha256": "0f25af23455154c730e5380d9bb7e95470cc283140d68b96f0becb20ddb11477",
    "cases": [
      "replay",
      "balanced",
      "force-only"
    ],
    "data_directory": "data/processed/aspirin",
    "diagnostic_wall_seconds": 600,
    "direction_zero_tolerance": 0.0,
    "directions": [
      "stateful_adam",
      "reset_adam_matched_norm",
      "steepest_matched_norm"
    ],
    "early_stopping": false,
    "evidence_log": "reports/PAPER_EVIDENCE_LOG.md",
    "expected_prepared_identity": "4b9f220201d6483e2b2a8da74c10330c3762942ff4c6fabe6389ee27249c43aa",
    "expected_split_hash": "73d5475bfc1f1158da62b939f5431b1781db886443cece2cdb297c8dd645133a",
    "expected_subsets_identity": "8139e442fca011efcbeb5f2d28d701c5f36030d61f045e5e652b98eb18b1be47",
    "fd_atol": 1e-08,
    "fd_rtol": 0.0001,
    "fd_steps": [
      0.0001,
      1e-05
    ],
    "gate_fraction": 0.1,
    "gradient_atol": 1e-10,
    "gradient_rtol": 1e-08,
    "learning_rate": 0.001,
    "loss_balance_compact_path": "reports/loss-balance-diagnostics.json",
    "loss_balance_compact_sha256": "c7c35276a38e4f17b785510e745eb75e8c76b2fd3f591b787a8bc43b7f034e4f",
    "loss_balance_run": "runs/aspirin-loss-balance-2026-09-30",
    "loss_balance_sha256": {
      "balanced/anchor-inputs.npz": "5c638b71fe14615e53ae0b27a0c69c1fff1fd572f9f7191130cb2116e86264fc",
      "balanced/checkpoint-verification.json": "cf07aa2554610daac41ef3e83a1335092e751bf8db7ecc30ccf2cf5128f1a319",
      "balanced/config.json": "446d95ef952249d64ca5598d2c9ad713af96c7758b87190b5beab2a1ec4da589",
      "balanced/fitting-inputs.npz": "9c56de4bae8058f4c18aab338aaee4e45c42f527905e48656f49900239fa701f",
      "balanced/history.json": "3d5f8541f36996b67cc061a753dbeafdf110d76a14306964ced5388b03973384",
      "balanced/last.msgpack": "bdc60383661ec8000d1d5e76918560390432d194e0b169eb2758729dce97e099",
      "balanced/provenance.json": "46bc63737b77c3aec012432ee0a41f1324319e1b4185fe0556d5dc83d42d13d7",
      "balanced/reference.npz": "a87b9890d8ea7df874a182a70573b146afc795a36a2566c4966a3b2d2cd3466e",
      "balanced/summary.json": "abde4733a1119ade837775fedb5ed93ef5fcebbd2543cce1b51bde1305005aa9",
      "config.json": "56952e95c9b605fa5fbfd3d84a30fd8679ccc928690f529b738ae778bca2422b",
      "force-only/anchor-inputs.npz": "5c638b71fe14615e53ae0b27a0c69c1fff1fd572f9f7191130cb2116e86264fc",
      "force-only/checkpoint-verification.json": "cf07aa2554610daac41ef3e83a1335092e751bf8db7ecc30ccf2cf5128f1a319",
      "force-only/config.json": "52bcb13d741a24cdde22644245f0bc9a8a20704aa9bf62640887e61ef8b99d18",
      "force-only/fitting-inputs.npz": "9c56de4bae8058f4c18aab338aaee4e45c42f527905e48656f49900239fa701f",
      "force-only/history.json": "3eedbee2a881391caf0c01dc811f586ce77f1785f56e8e71e35b8b8a0a12b269",
      "force-only/last.msgpack": "3b90337de97438f9ac5e83143265403a9fce59a175cac5653cec9e5077e00224",
      "force-only/provenance.json": "ee3745b7beec73306168cbdc94588ac940a29be770fdda780d0fe9dacfe9bfe3",
      "force-only/reference.npz": "07f5aacc1cbe91b57f21de888f14416ace7a10b3f45496184eec5adfaf3ad600",
      "force-only/summary.json": "42e9158b66498ee9365911abeb697257ccdae6448a58743bebb850ce801cef7e",
      "provenance.json": "8e45255e0daa5b7e0f52cb12d05f612470724edb451a310bdf699d621e839ae2",
      "replay/anchor-inputs.npz": "5c638b71fe14615e53ae0b27a0c69c1fff1fd572f9f7191130cb2116e86264fc",
      "replay/checkpoint-verification.json": "cf07aa2554610daac41ef3e83a1335092e751bf8db7ecc30ccf2cf5128f1a319",
      "replay/config.json": "ce9bed374dd8e43ee0d3f742e11ded968aa6ecb998b2a3a7023739c8fdcde105",
      "replay/fitting-inputs.npz": "9c56de4bae8058f4c18aab338aaee4e45c42f527905e48656f49900239fa701f",
      "replay/history.json": "a0dec9c24f1a4d0b38fcff61e66bf28bc979c99e7dd5c04c01bbbc354f0b73a5",
      "replay/last.msgpack": "1966c79103e542e9bcf2b7deceebf0102e182a7b1532b3398c46281bdc53a831",
      "replay/provenance.json": "ccae192660c6b4adb657cd30d58fdb2c2482fbbddea011e0b042464e7eddc88c",
      "replay/reference.npz": "5ebb9dfb89f2c5df095a8d6f1dfe5fa2e1901b6dcfcf3159b417ccad5b724afd",
      "replay/summary.json": "125f303c78710b7697d6e829da82e2903c8b4f2760c165825fb3fb4b99b5237b",
      "subsets.json": "d3c3db7a92f3ffb0748039f63d7a0734bf38214032037ab67780d6182ad764b6",
      "summary.json": "7d74cb95ae16f7503da5098fcfb0a6b7bbf9f74a6edc975238dac47a154bdd49"
    },
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
    "probe_scales": [
      0.1,
      1.0
    ],
    "replay_tolerance": {
      "atol": 1e-10,
      "rtol": 1e-08
    },
    "run_directory": "runs/aspirin-optimizer-step-2026-09-30",
    "seed": 0,
    "source_run": "runs/aspirin-data-coverage-2026-09-30",
    "source_sha256": {
      "config.json": "3575dad49acf97b8f780f953a397c776cb5f3dd96729f1e6236837d8e99a5f64",
      "diverse128/anchor-inputs.npz": "5c638b71fe14615e53ae0b27a0c69c1fff1fd572f9f7191130cb2116e86264fc",
      "diverse128/checkpoint-verification.json": "cf07aa2554610daac41ef3e83a1335092e751bf8db7ecc30ccf2cf5128f1a319",
      "diverse128/config.json": "dfbdb9129e00a2984341a1efd80e901fda20565d3bcc7cc53ae04d2ed5ac6b51",
      "diverse128/fitting-inputs.npz": "9c56de4bae8058f4c18aab338aaee4e45c42f527905e48656f49900239fa701f",
      "diverse128/history.json": "2e037af82e7fae42f7811fb661b39202b331bb42e8499ec45c0c3ba73d18629f",
      "diverse128/last.msgpack": "7305113575499ebe51ca3e30b87d5bb644888b913b845e3fee5b95e222f4f6c3",
      "diverse128/provenance.json": "1502d258d63cf8a69e9ee67fcd9852855b9dbc3deb2010c33010d8b56db81063",
      "diverse128/reference.npz": "5ebb9dfb89f2c5df095a8d6f1dfe5fa2e1901b6dcfcf3159b417ccad5b724afd",
      "diverse128/summary.json": "0e374b8ed313f8004aec6219af3b2e45f0a868082f00ced907a6f4c9d8b88a8b",
      "diverse16/anchor-inputs.npz": "5c638b71fe14615e53ae0b27a0c69c1fff1fd572f9f7191130cb2116e86264fc",
      "diverse16/checkpoint-verification.json": "f39a1a3d6cacfc6eb6418e7a695bb550ac9350bc9e08cd2aa20d80ad256a6295",
      "diverse16/config.json": "dfbdb9129e00a2984341a1efd80e901fda20565d3bcc7cc53ae04d2ed5ac6b51",
      "diverse16/fitting-inputs.npz": "ae6550bbda77af6eb1617a859cc8ce118ab90c6bd109c3a25b0d3827faf9a65c",
      "diverse16/history.json": "f9b66c8e496ba8ca82d626df2a38980f41c7dc1edf28bb97f21028a8aa8fc367",
      "diverse16/last.msgpack": "7156487d43587f9cdfe563f14db79948d465b0bf43995ff67e07e3c6ac180d8f",
      "diverse16/provenance.json": "80bed026608d4455f4ab81a1fbc17520c511a1772c39fc448eada96f6dd34791",
      "diverse16/reference.npz": "f48c42e9697e2c93a6a8393c1378d3faecf1ad08a3c598518cf39fafae2c1c85",
      "diverse16/summary.json": "af772ad611b5d153b85265861375419df80f2b93c20407c0304f6ca9876fb5e6",
      "narrow16/anchor-inputs.npz": "5c638b71fe14615e53ae0b27a0c69c1fff1fd572f9f7191130cb2116e86264fc",
      "narrow16/checkpoint-verification.json": "f39a1a3d6cacfc6eb6418e7a695bb550ac9350bc9e08cd2aa20d80ad256a6295",
      "narrow16/config.json": "dfbdb9129e00a2984341a1efd80e901fda20565d3bcc7cc53ae04d2ed5ac6b51",
      "narrow16/fitting-inputs.npz": "5c638b71fe14615e53ae0b27a0c69c1fff1fd572f9f7191130cb2116e86264fc",
      "narrow16/history.json": "9970a673c501b566e8f8de0b83581ce81f40cba9a71a2547111f9fe46b0af812",
      "narrow16/last.msgpack": "b5c792314ea6b34d46bc5b2b104abbc7b1e647fd6c81cea55ec49e94b18cb6b7",
      "narrow16/provenance.json": "5de39bfac0af7376beee7e9790b6028443cff7d8d3983006b4a1ba0b5617b6ad",
      "narrow16/reference.npz": "95f1258fba9882b7c7ae600f704f0465e7c6011ef149d2127ccf5a414ee9ecaa",
      "narrow16/summary.json": "c3b0aa387b7cbd2cc57723e6232677eea19abb3f03dc1834cd8cdf1b32128452",
      "provenance.json": "b60d13789966f2ce2fbb11a3c4de73f0886f2e952ba9e0254892c35c554f7dcc",
      "subsets.json": "d3c3db7a92f3ffb0748039f63d7a0734bf38214032037ab67780d6182ad764b6",
      "summary.json": "54e7f6d7e9995dbca90bdca3417457b075e99037dc9ecd6d6a1c0647ac102a9f"
    },
    "status": "planned_unexecuted",
    "subsets_manifest": "manifests/aspirin-data-coverage-subsets-planned.json",
    "validation_interval": 100,
    "wall_seconds": 1800
  },
  "config_hash": "5e10be5e29633022a450076553ff17f59f7e34fcc3cf54a564647043a036ccc2",
  "limits": "Exactly3savedstates; no research retry or training;600s boundary cap"
}
```

## OPTSTEP-CASE-01-PRE — Case preregistered

Frozen training-only diagnostics; no optimizer update or learning acceptance.

```json
{
  "anchor_ids": [
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
  "case": "replay",
  "config_hash": "5e10be5e29633022a450076553ff17f59f7e34fcc3cf54a564647043a036ccc2",
  "directions": [
    "stateful_adam",
    "reset_adam_matched_norm",
    "steepest_matched_norm"
  ],
  "fd_steps": [
    0.0001,
    1e-05
  ],
  "fitting_ids": [
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
  ],
  "limits": "Temporary local displacements only; no committed updates, winner selection or learning/pilot clearance",
  "scales": [
    0.1,
    1.0
  ],
  "source_update": 2000,
  "state_hashes": {
    "optimizer": "311d3e2925e1d6b5183ae9346a279b9047ec4fae7a9c162fb83454d758e0e87c",
    "params": "a05e254c1de7b419f2f659af0e902a137a8380f7e6a6c3bbc423947c9332e236"
  },
  "weights": {
    "energy": 1.0,
    "force": 1.0
  }
}
```

## OPTSTEP-CASE-01-RESULT — Case outcome

Frozen training-only diagnostics; no optimizer update or learning acceptance.

```json
{
  "artifact_sha256": {
    "observations.json": "afe41c645bd6241be729fd41a1d7df8a256d86222c4bb3052d8b8c58a1cb3278",
    "partial.json": "afe41c645bd6241be729fd41a1d7df8a256d86222c4bb3052d8b8c58a1cb3278",
    "prospective.json": "59341366cb041de254656535bfad5d3d18ca288ce2f31495c4c07f8d2a5790c9",
    "summary.json": "a92a86fd2a7245822383e544e67ebc9592dee43782144270fd3e5d38f218be2d"
  },
  "compute": {
    "completed_frame_evaluations": {
      "fd": 1728,
      "gradient_energy": 144,
      "gradient_force": 144,
      "probe": 864,
      "reference_builder": 128,
      "source_predictions": 144,
      "weighted_gradient": 128
    },
    "operations": [
      {
        "category": "gradient_energy",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_energy",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_energy",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_energy",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_energy",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_energy",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_energy",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_energy",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_force",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_force",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_force",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_force",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_force",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_force",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_force",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_force",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_energy",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "gradient_force",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "weighted_gradient",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "weighted_gradient",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "weighted_gradient",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "weighted_gradient",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "weighted_gradient",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "weighted_gradient",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "weighted_gradient",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "weighted_gradient",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "reference_builder",
        "completed_frames": 128,
        "planned_frames": 128,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "source_predictions",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "source_predictions",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "source_predictions",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "source_predictions",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "source_predictions",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "source_predictions",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "source_predictions",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "source_predictions",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "source_predictions",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      }
    ]
  },
  "counts": {
    "fd_cells": 12,
    "fd_steps": 24,
    "probes": 6
  },
  "executed": true,
  "failure_reason": null,
  "state_hashes_after": {
    "optimizer": "311d3e2925e1d6b5183ae9346a279b9047ec4fae7a9c162fb83454d758e0e87c",
    "params": "a05e254c1de7b419f2f659af0e902a137a8380f7e6a6c3bbc423947c9332e236"
  },
  "state_hashes_before": {
    "optimizer": "311d3e2925e1d6b5183ae9346a279b9047ec4fae7a9c162fb83454d758e0e87c",
    "params": "a05e254c1de7b419f2f659af0e902a137a8380f7e6a6c3bbc423947c9332e236"
  },
  "status": "completed",
  "stop_reason": null,
  "verification_valid": true
}
```

## OPTSTEP-CASE-02-PRE — Case preregistered

Frozen training-only diagnostics; no optimizer update or learning acceptance.

```json
{
  "anchor_ids": [
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
  "case": "balanced",
  "config_hash": "5e10be5e29633022a450076553ff17f59f7e34fcc3cf54a564647043a036ccc2",
  "directions": [
    "stateful_adam",
    "reset_adam_matched_norm",
    "steepest_matched_norm"
  ],
  "fd_steps": [
    0.0001,
    1e-05
  ],
  "fitting_ids": [
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
  ],
  "limits": "Temporary local displacements only; no committed updates, winner selection or learning/pilot clearance",
  "scales": [
    0.1,
    1.0
  ],
  "source_update": 2000,
  "state_hashes": {
    "optimizer": "dea2d079b2a0bdd295152c55e134de429d6d75ddb9610ea93ec737cda1d45405",
    "params": "25483221ba39fa383cd989491f04e43089bd553928354a3056ba9b4e0b3330de"
  },
  "weights": {
    "energy": 0.014584365670390719,
    "force": 1.9854156343296092
  }
}
```

## OPTSTEP-CASE-02-RESULT — Case outcome

Frozen training-only diagnostics; no optimizer update or learning acceptance.

```json
{
  "artifact_sha256": {
    "observations.json": "12be229893b45f77d232b4c7ab875ea7f805d638ed6706fdd96323df14df371e",
    "partial.json": "12be229893b45f77d232b4c7ab875ea7f805d638ed6706fdd96323df14df371e",
    "prospective.json": "11f1fd9cdd4848d875ac9955818c243648f33c0ef83be71dfdc22b6bff9479f3",
    "summary.json": "eb552e1804d90190b5d44a0caaa00f106df32ea8410e4e83ac2174dc52bd97e5"
  },
  "compute": {
    "completed_frame_evaluations": {
      "fd": 1728,
      "gradient_energy": 144,
      "gradient_force": 144,
      "probe": 864,
      "reference_builder": 128,
      "source_predictions": 144,
      "weighted_gradient": 128
    },
    "operations": [
      {
        "category": "gradient_energy",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_energy",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_energy",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_energy",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_energy",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_energy",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_energy",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_energy",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_force",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_force",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_force",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_force",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_force",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_force",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_force",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_force",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_energy",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "gradient_force",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "weighted_gradient",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "weighted_gradient",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "weighted_gradient",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "weighted_gradient",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "weighted_gradient",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "weighted_gradient",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "weighted_gradient",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "weighted_gradient",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "reference_builder",
        "completed_frames": 128,
        "planned_frames": 128,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "source_predictions",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "source_predictions",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "source_predictions",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "source_predictions",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "source_predictions",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "source_predictions",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "source_predictions",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "source_predictions",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "source_predictions",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      }
    ]
  },
  "counts": {
    "fd_cells": 12,
    "fd_steps": 24,
    "probes": 6
  },
  "executed": true,
  "failure_reason": null,
  "state_hashes_after": {
    "optimizer": "dea2d079b2a0bdd295152c55e134de429d6d75ddb9610ea93ec737cda1d45405",
    "params": "25483221ba39fa383cd989491f04e43089bd553928354a3056ba9b4e0b3330de"
  },
  "state_hashes_before": {
    "optimizer": "dea2d079b2a0bdd295152c55e134de429d6d75ddb9610ea93ec737cda1d45405",
    "params": "25483221ba39fa383cd989491f04e43089bd553928354a3056ba9b4e0b3330de"
  },
  "status": "completed",
  "stop_reason": null,
  "verification_valid": true
}
```

## OPTSTEP-CASE-03-PRE — Case preregistered

Frozen training-only diagnostics; no optimizer update or learning acceptance.

```json
{
  "anchor_ids": [
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
  "case": "force-only",
  "config_hash": "5e10be5e29633022a450076553ff17f59f7e34fcc3cf54a564647043a036ccc2",
  "directions": [
    "stateful_adam",
    "reset_adam_matched_norm",
    "steepest_matched_norm"
  ],
  "fd_steps": [
    0.0001,
    1e-05
  ],
  "fitting_ids": [
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
  ],
  "limits": "Temporary local displacements only; no committed updates, winner selection or learning/pilot clearance",
  "scales": [
    0.1,
    1.0
  ],
  "source_update": 2000,
  "state_hashes": {
    "optimizer": "5a1a5b60f53b32db88eebe22475db2b8ae7799145658ce842618bd1d6d3203e9",
    "params": "0e4dd54f6de15a4440dcafcfb23a1df7afae375953d51f1f92d00f7b2984ef33"
  },
  "weights": {
    "energy": 0.0,
    "force": 2.0
  }
}
```

## OPTSTEP-CASE-03-RESULT — Case outcome

Frozen training-only diagnostics; no optimizer update or learning acceptance.

```json
{
  "artifact_sha256": {
    "observations.json": "a0ebb2ecbd1e57396b46d65569ae1850ef6cfa917a7559cf4c5c292779fab730",
    "partial.json": "a0ebb2ecbd1e57396b46d65569ae1850ef6cfa917a7559cf4c5c292779fab730",
    "prospective.json": "8f1724df10f3b1c0d228b17218223a17f90a6c9a67189fbf6a6b2cd6369a6405",
    "summary.json": "e5864c1ce659662f8c62b67b0d3c68fa321ba4c1a72b9735663bc43127d2b9f4"
  },
  "compute": {
    "completed_frame_evaluations": {
      "fd": 1728,
      "gradient_energy": 144,
      "gradient_force": 144,
      "probe": 864,
      "reference_builder": 128,
      "source_predictions": 144,
      "weighted_gradient": 128
    },
    "operations": [
      {
        "category": "gradient_energy",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_energy",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_energy",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_energy",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_energy",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_energy",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_energy",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_energy",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_force",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_force",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_force",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_force",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_force",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_force",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_force",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_force",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "gradient_energy",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "gradient_force",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "weighted_gradient",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "weighted_gradient",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "weighted_gradient",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "weighted_gradient",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "weighted_gradient",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "weighted_gradient",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "weighted_gradient",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "weighted_gradient",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "reference_builder",
        "completed_frames": 128,
        "planned_frames": 128,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "fd",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "source_predictions",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "source_predictions",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "source_predictions",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "source_predictions",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "source_predictions",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "source_predictions",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "source_predictions",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "source_predictions",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "source_predictions",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "fitting",
        "status": "completed"
      },
      {
        "category": "probe",
        "completed_frames": 16,
        "planned_frames": 16,
        "role": "anchor",
        "status": "completed"
      }
    ]
  },
  "counts": {
    "fd_cells": 12,
    "fd_steps": 24,
    "probes": 6
  },
  "executed": true,
  "failure_reason": null,
  "state_hashes_after": {
    "optimizer": "5a1a5b60f53b32db88eebe22475db2b8ae7799145658ce842618bd1d6d3203e9",
    "params": "0e4dd54f6de15a4440dcafcfb23a1df7afae375953d51f1f92d00f7b2984ef33"
  },
  "state_hashes_before": {
    "optimizer": "5a1a5b60f53b32db88eebe22475db2b8ae7799145658ce842618bd1d6d3203e9",
    "params": "0e4dd54f6de15a4440dcafcfb23a1df7afae375953d51f1f92d00f7b2984ef33"
  },
  "status": "completed",
  "stop_reason": null,
  "verification_valid": true
}
```

## OPTSTEP-EXECUTION-RESULT — Frozen optimizer diagnostic outcome

Frozen training-only diagnostics; no optimizer update or learning acceptance.

```json
{
  "case_statuses": {
    "balanced": "completed",
    "force-only": "completed",
    "replay": "completed"
  },
  "elapsed_seconds": 15.104326874949038,
  "execution_valid": true,
  "failure_reason": null,
  "next_step": "separately_review_followup_from_local_evidence",
  "source_hashes_unchanged": true,
  "status": "completed",
  "summary_sha256": "3255465553f165b8979475a23681c9a490bef098f05bdb1387e316561cbd4b9b"
}
```

## FINDING-OPTSTEP-001 — Once-only frozen optimizer observations

Measured research: one invocation at codefef6197, exact reviewed configs/aspirin-optimizer-step-diagnostic.json. All3cases completed in15.104326874949038s within600s;100raw+2compact source hashes unchanged;36FDcells/72steps/18probes,0committed training exposures. Parameter/optimizer hashes unchanged ineachcase; sourceupdate/Adamcount2000, proposed2001discarded. Manual moments/delta and referencebuilder passed. Raw summarySHA256 `3255465553f165b8979475a23681c9a490bef098f05bdb1387e316561cbd4b9b`; compactSHA256 `333000d73c50b454c13960745fb89e06298052e17058294537f71ff38a87f92e`. See every metric/block/FDstep in raw case observations and reports/optimizer-step-diagnostics.json.

At scale1, saved-state Adam fitting force MSE relative changes: {"balanced": -0.00011973294957468017, "force-only": -0.00014439968447732543, "replay": -7.1773320751145064e-06}. These are tiny local reductions, not gate clearance/convergence/causality. Replay anchor force increases slightly; balanced and force-only anchors decrease. Both reference directions at the balanced and force-only fitting endpoints have negative force slopes, yet every prescribed finite fitting force probe (scales0.1/1) increases force MSE. UnitFDs support the local derivatives under locked tolerance, exposing limits of inferring finite-step benefit from local slopes. Equal whole-tree norms do not equalize parameter displacement/compute. No winner chosen or persistent update. Force-only headbias weightedgradient and direction remainzero; absoluteenergy errors are preserved, no recentering. Fully fitted anchor,one seed,no generalization/capacity theorem. Original learning failures/fullpilotpending remain.

## FINDING-OPTSTEP-002 — Preserve all failed FD steps

All36cells passed the preregistered at-least-one-of-both criterion. Four largerh1e-4 steps failed: replay/steepest/fitting/energy; balanced/steepest/fitting/energy andforce; force-only/steepest/fitting/force. Correspondingh1e-5 steps passed. All72steprecords are retained without selecting a replacement run or hiding failed steps. Max errors are listed in saved records; FD tolerance passage does not certify near-zero derivative signs. All18temporarypoints retained; no numerical retry. Negative slopes plus finite increases at prescribed distances are local nonlinear observations, not evidence that another optimizer would fail in training.

## VERIFY-OPTSTEP-001 — Post-run reconciliation and plot inspection

Controller valid,100raw+2compact hashes rechecked separately after production. Sourcepredictions208parent+432loss; percase completedframes: energygradient144,forcegradient144,weightedgradient128,referencebuilder128,sourcepredictions144,FD1728,probe864. Across3cases totaldiagnosticpasses10480 under recorded category convention; no incomplete operations. Figures optimizer-step-force.png/energy.png visually inspected: allprescribedpoints, bothroles, readablelabels, differences fromsource inpercent, axesindependent perpanel. Report generation succeeded with temporary matplotlibcache after sandbox cache warnings, not a numerical failure. No hidden heldout evaluation. Independent final review pending; originalstates2000unchanged.

## OPTSTEP-FINAL-REVIEW-001 — Independent review and verified reporting repair

Fresh gpt-6-astra/high review at0a02b43..e1679d1 delivered33focusedtests passed20.93s and independent100raw+2compact/36FD/72steps/18probes/norm/blocksum/manual/builder/hash reconciliation; no numerical research defect identified. One Important report-completeness defect: empty/missing direction/role/objective/step/probe coverage with recomputedcounts could falsely pass report interpretation guard. Author eightvariant RED8failed9.32s; one strictschema repair pass GREEN14passed9.89s. Explicit zero/NA/unavailable states retained; complete probe matrix required, no fictitious success from omissions. Fullsuite and JSON-only report rebuild pending; numerical execution not repeated. Reviewer final-response failed dueaccountusage after delivering technicalfindings/reconciliation; record scopes what was received and does not claim formal independentclearance. See docs/reviews/2026-09-30-optimizer-step-final-review.md. CLI backup/updates of bothObsidiannotes verified; repository numericalauthority unchanged.

## OPTSTEP-REPORT-002 — Final report-only repair verified

Full193tests passed125.07s (full-after-review.log); no failures, prior152checks retained. SavedJSON-only rebuild remains interpretation-valid with36FDcells/72steps/18probes;100raw+2compact andall18rawoutputfiles independently hash-reconciled;0unknownoperations/0committedtraining. CompactcurrentSHA256 `f841592a80e09b205b5422d196fa22088d704d63a03693e1d143ae806be53fc5`; generatorSHA256 `7bb82a4ccb58251d9ea76eec81af260e9f736fc0ed6024b8c90720cb0baac68a`. This newreport identity supersedes initial compactidentity in FINDING-OPTSTEP-001 solely because completenessguard/generator changed; rawsummary/sources remainidentical. Figures unchanged byteforbyte after rebuild, already visually inspected. No independentformalclearance asserted beyond deliveredtechnicalreview andauthorverifiedrepair. Pilotpending; nextinterventionrequiresseparatereview.

## OPTSTEP-ARCHIVE-001 — Final retained evidence archive

Task3check14passed9.76s; full193passed125.07s. BothObsidian currentsections verified throughCLI with193tests and exactreview scope; initial/intermediate/finalnotes retained. Durablearchive reports/execution/optimizer-step-2026-09-30 holds38workspacefiles plus verified18file rawoutputZIP. ManifestSHA256 `e0f7eea272c283e940830067cba436bca34af6ad97aa0fbb70a8fab26617873f`; ZIPCRC andallarchived/sourcebytes verified. Existingworkspaces/reviews/runs and userproposal changes preserved. Localbranch only, no merge/push/delete. Milestonecomplete; nextconvergence/optimizer/capacity work requiresnewtwicereviewedplan. Originallearninggates failed/fullpilotpending.
