# Force-gradient isolation with matched energy-gauge calibration

## Intent and boundary

The user authorized proceeding to the next task after the completed offset-profile trial. This milestone prepares a concrete next-lever plan with two independent reviews. Proposed experiment and implementation are UNEXECUTED until their gates are recorded. Primary implementation is the established execution method; subagents review only. This is a controlled diagnostic with a new guarded runner, so it gets a written design and implementation plan.

Question: under the latest width32/6000-update setting, is removing the remaining energy-loss gradient sufficient to improve force fitting or reach the original force threshold? Do not treat the question as an established explanation of the failed fit.

Three approaches were considered. Recommended: a matched force-only gradient ablation, changing the energy coefficient from0.014584365670390719 to0.0 while keeping all other scientific settings. A larger architecture/representation would change several factors before the objective tradeoff is isolated. A new rate/schedule sweep would create search/selection and would not isolate the energy term. Neither alternative is included.

## Existing evidence and limits

`reports/offset-profile-paper-findings.md` and its independent saved-result review establish a valid completed joint control. Fitting force MSE fraction0.45100597487761285 and fitted-anchor fraction0.42369062245390743 remain above0.1. Centered anchor energy variance worsened24.41971159078873%; group errors are heterogeneous. All16 anchors are fitted group01 structures, not held-out evaluation.

Earlier `reports/loss-balance-diagnostics.md` force-only work also FAILED the force gate: it used width16,2000updates, rate0.001, force coefficient2 and no energy-gauge calibration. It is context only, not the numerical control or proof this new ablation will succeed.

LIT007, Christensen/von Lilienfeld, local `reports/literature/2026-09-30/gradients2020.txt`, section on composition learning (printed pp13–14), describes the energy integration-constant ambiguity under force information. Its kernel learning curves do not prescribe neural Adam budgets or predict this experiment. The present conservative model determines forces as negative coordinate derivatives; a global additive energy shift changes neither forces nor their parameter gradients mathematically. Both statements need implementation checks; neither establishes accurate reference-shape recovery.

## Exact scientific controls

Restore `runs/aspirin-capacity-2026-10-02/width32/snapshots/0.msgpack` directly, with source provenance. No random parameter initializer or endpoint continuation. Preserve source params/Adam count and moments/RNG/normalization. Model:32 features,13,569 parameters,degree1,two message-passing iterations,eight radial features,5 Å cutoff,21 atoms. CPU,float64,seed0,128 fitting structures (eight groups ×16),16 fitted anchors from group01, original ordered IDs and exact input bytes, microbatch16, Adam rate0.0003.

The single scientific change relative to completed `runs/aspirin-offset-profile-2026-10-02` is energy coefficient0.0 instead of0.014584365670390719. Force coefficient MUST remain1.9854156343296092; do not rescale it to2 or re-estimate weights. Loss remains force coefficient × raw force-component MSE / original squared force loss scale. The energy term contributes zero gradient. Compute/retain raw E/F scores; all finite-value rules remain active. Shared body and readout kernel parameters remain trainable; only scalar atomic energy-head bias is excluded from Adam and has zero moments.

Retain the same full-fitting global bias projection before training and after every accepted step: delta_bias=−mean(fitting energy errors)/21. Anchor/group energies are never recalibrated separately. Energy labels are still used for this gauge and scoring. Name the method **force-only gradient ablation with fitting-energy gauge calibration**; do not call it energy-free or force-only supervision. The calibration is not a learned energy-shape objective.

The newly projected update-zero all-field hash MUST equal the completed control projected-zero hash `b22c5108b3e08c088f20f2bc862e01ddb6df9fd3961ce949f951e41f1512e4bc` before warmup/accepted updates. Shared original-zero full hash is `6ebbe0d9ff683e10c09b1ab373faf5d62fcc199b8109caf5ed9e4f0723ca028e`. Retain original zero-baseline replay and source/projected identities separately. If exact equality fails, invalidate execution; do not recalibrate/tune to rescue it.

## Budget and observations

Exactly one public invocation, one fresh-process endpoint child,6000 accepted updates,6001 accepted initial/training calibrations. No early stop,search,adaptive weights,rate changes,retry,resume,label replacement,new examples,extra seed or data access. Public root `runs/aspirin-force-isolation-2026-10-03` must be absent and exclusively reserved.

Charged numerical wall3600seconds including source/runtime checks, imports, compilation,warmup,scoring,oracles,child and final durable summary writes;180seconds finalization reserve. Supervisor3660seconds, TERM then10second grace/KILL, no retry. A timeout/incomplete source/state/child/audit prevents numerical endorsement. Timings are operational observations, not a clean throughput comparison.

63 immutable scored states:0,1,every100 through6000,and3001.62 shared postzero comparisons INCLUDE3001 and exclude projected0; initial projected0 is a separate equality witness. No best-checkpoint selection. Two full128 independent pure-force-gradient/NumPy Adam oracles at outgoing1/3001, actual incoming moments, combined per-leaf atol1e-10+rtol1e-8; do not substitute an absolute-only maximum rule. Once-only child predicts128+16=144 role entries, max absolute error≤1e-10, no training/initialization.

Accepted training gradient frames768000; accepted initial/training calibration frames768128; warmup gradient128+calibration128 discarded. Full reference oracle work, all score/replay/child exposures and unsuccessful attempted calls are counted separately. Immutable vectors/receipts/state/history/predictions and raw returned latest partial work use the same durable-write and ownership protections as the control.

## Acceptance and interpretation

Execution validity is independent of learning. Primary force-only gate: BOTH fitting and fitted-anchor raw force MSE fractions strictly below0.1 of immutable ORIGINAL UNPROJECTED source-zero force MSE. Joint gate remains BOTH raw energy and force fractions below0.1 on BOTH roles, reported separately and never implied by passing the force-only gate. Original fitting E/F baselines0.03751676427546992 eV² /0.014083948218070219 eV²/Å²; anchor0.061298358531283896 /0.010133561081129106. No projected-zero baseline reset,RMSE substitution or gauge-based acceptance.

Compare raw endpoints/fractions/signed relative changes, force-only/joint outcomes and raw sampled maxima against the verified completed offset-profiled joint control, with identical ordered support/grid and exact initial projected state. Retain pooled/equal-group/per-group E/F, means,mean squares and population centered variance. Record improvement,unchanged error and deterioration without adding a post-hoc effect threshold or statistical significance claim. Group44/78_2 adverse control outcomes remain visible. Energy shapes and cross-group relative energies can deteriorate even when mean error is analytically zero; they remain empirical outcomes.

If force-only passes but joint fails, this diagnostic supports force fitting under these settings without clearing the project gate. If force-only fails, removing this term is insufficient in this specific setting, not proof of inadequate architecture, convergence,optimizer failure or label incompatibility. A descriptive better endpoint cannot establish a general causal conflict mechanism. One seed/rate/molecule, historical control, changed objective, fitted single-group anchors,sparse scores,construction-based energy calibration and finite-difference/gradient checks limit claims. Main held-out reference-shape study and full training pilot remain pending.

## Artifact and code architecture

Freeze all old source/config/scripts/tests/docs/raw bytes bound by planning `source-pins.json`:18,963 exact paths, SHA256 `eda3ae20c34d16a5fea4260d6a15a5ac31356e2a384df00e4dd08e29a0e30664`. Exact membership is the union of the old601-source map, completed control18324 numerical manifest, old gate artifact keys, and the15 explicit context paths in source-context-verification generation. Reconstruct and check this union; reject missing/extra numeric membership and symlinks. Only already declared `.DS_Store` is excluded as incidental control metadata, retained separately. Context source report/gate/archive hashes are part of this union. Protect the entire previous45,863,751-byte evidence-log prefix SHA256 `3c45058c51e90a4c17390b9184218eeb7030eede0a1f85bc57b6b1e0df34f246`; future evidence is append-only.

Create four new scripts only: `scripts/force_isolation_contract.py` pure config/source/control/review/launch guards; `scripts/train_force_isolation.py` own guarded worker/child; `scripts/run_force_isolation.py` once-only supervisor; `scripts/report_force_isolation.py` saved-only audited control comparison. Three new test files and `configs/aspirin-force-isolation.json` accompany them. Production configuration is generated from exact proposed-settings values plus hash-bound plan/source paths; canonical ID is frozen in the pure contract after both final plan PASS reviews.

Reuse unchanged offset numerical core, Journal/Store/state/prediction/ownership helpers and saved-state tensor audits by import. Do NOT use old `run_owned` as a production shortcut: it unconditionally calls old canonical `verify_inputs` at6000, and its child launcher points to the old worker. Own the small orchestration engine/child/launch finalization in the new worker, preserving order and boundaries; retain explicit code ancestry and test equivalence on a private tiny joint-coefficient fixture. No monkeypatch/rebinding of old guards, bypass flag or modification of old modules. The new reporter passes explicitly verified source/gate to the old saved-array audit, and implements its own comparison root/force gate/labels. Old reports remain regenerable under old frozen guards.

Before launch: two final plan PASS reviews, resolved material amendments, focused meaningful regressions, full suite at final participant bytes, independent implementation followup, immutable prelaunch gate including new and imported participants/config/runtime/revision, prospective budget/work record, exclusive invocation ordinal. Source/state/review failures resolve operationally without a research retry.

## Closure and preservation

Saved-only new report/paper findings, exact62-grid figure with caption/visual QA, numerical-only inventory/ZIP exact membership/all hashes/CRC, byte-for-byte report regeneration, independent saved-result review, append-only balanced evidence and CLI-only Obsidian backups/readbacks. If incidental metadata appears, retain and inventory separately rather than silently altering numerical membership. No additional production/model calls to finish reporting. Preserve all failed,blocked,abandoned,interrupted reviewer/engineering/operational trials and user PDF changes; no review folder deletion. Reuse suitable dedicated checkout, no new environment/GPU/install. End with a concrete recorded next decision; no unplanned next lever.
