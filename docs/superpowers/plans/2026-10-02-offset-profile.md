# Controlled Energy-Offset Profiling Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline, task by task. Primary implements; subagents review only. Preserve every review workspace.

**Goal:** Execute one source-bound analytical energy-bias profiling trial against historical width32 capacity reference, preserving original raw gates.
**Architecture:** Standalone numerical core, pure guarded training/child controller, saved-only reporter. Reuse frozen helpers without src or historical artifact changes.
**Tech Stack:** Existing CPUfloat64 JAX/Flax/E3x/Optax/NumPy/msgpack; Python JSON/files/ZIP and scientific plotting.
**Spec:** docs/superpowers/specs/2026-10-02-offset-profile-design.md

## Global Constraints

- Dedicated checkout/runtime; preserve PDFs/reviews. No install/src edits/historical rerun/GPU/fullpilot.
- One intervention; source update0 parameter/optimizer hashes.6000accepted updates,6001 initial+outgoing calibrations.128fit/16 fitted anchors,21atoms,microbatch16,source seed0,rate0.0003,weights E0.014584365670390719/F1.9854156343296092,width32/degree1/iterations2/radial8/cutoff5Å.
- Original normalization and unprojected raw zero-baseline strict<0.1 gates for both E/F/bothroles.63score updates0,1,every100 through6000,and3001.
- Exclusive runs/aspirin-offset-profile-2026-10-02 ordinal1; no retry/adaptation. Total3600s includes180s reserve; supervisor3660s+10sTERM grace; child actual remaining time.
- Exact sources/launch maps before numerics and at closure/reporting; bind original gate. Distinguish predicted projection statistics/fresh scores and accepted/returned/discarded/reference/unknown work.
- Two independent plan reviews/followups before implementation; implementation review and final focused/fullsuite gates before launch; independent saved-result review before completion.

## Review Focus

1. Different microbatch means/unequal chunks: global profiled derivative, never per-chunk centering.
2. Bias-only shift/nonzero incoming moments: force invariance, independent Adam oracle, zero biasmu/nu, immutable input.
3. Failed calls/late caps/missing summary: retain known/unknown work and returned outputs, block validity/retry.
4. Source/root/child/gate substitution: reject before entry and invalidate scientific endorsement.
5. Initial projection/fitted-anchor dependence: original baseline fixed, single fitting-derived model bias for raw roles.

### Task1: analytical projection and verified update

**Files:** scripts/offset_profile_core.py; tests/test_offset_profile_core.py.
**Interfaces:** project_bias(params,errors,atom_counts)->(newparams,receipt); profile_step(state,batches,model,norm,weights,rate,call,retain)->(candidate,actual_gradient,receipt); profile_oracle(incoming,candidate,actual_gradient,fullbatch,model,norm,weights,rate,call,retain)->receipt; restore_state(path,provenance)->state. call(kind,count,fn,*args,on_returned=...) handles ready/finite accounting; retain persists partial/returned artifacts.

- [ ] RED tests: errors[-3,-1,1,3] with21atoms leavebias unchanged andvariance5; constant+4 givesdelta−4/21,variance5. Onlybias changes; sourceparams/moments unchanged. Empty/nonfinite/varying counts reject; hand-derived2chunk fixture detects local versus global objective.
- [ ] Run .venv/bin/pytest -q tests/test_offset_profile_core.py and retain RED.
- [ ] Implement projection validation/immutable tree and known Optax tuple decoding without initializer. Mask only bias after rawaggregate tolerance<=1e-10; reject nonzero biasmoments. Original weighted gradient at full-support calibrated state, OptaxAdam, then full-support energy-only outgoing projection. Predicted poststatistics labeled; compilation does not execute updates.
- [ ] Add RED real small-model/nonzero-readout/unequal-chunk tests against full globally centered objective and NumPyAdam, incomingcount0 and3/nonzero bodymoments. Force invariance1e-12; gradient/state1e-10/1e-8, immutable incoming state. Expected missing profile_step/oracle failure.
- [ ] Implement core/oracle, run GREEN and scoped checkpoint commit.

### Task2: guarded once-only training and child

**Files:** configs/aspirin-offset-profile.json; scripts/train_offset_profile.py; scripts/run_offset_profile.py; tests/test_offset_profile_runs.py; execution sourcepins/progress.
**Consumes:** Task1 functions/callbacks and frozen cc/lr/cov source/model/checkpoint helpers.
**Produces:** verify_inputs(config)->sourcefacts; verify_gate(config,expected=None)->gate; execute(config)->summary; probe(root,token)->receipt. Raw config/code/provenance/sharedzero/replay/initialprojection,63score snapshots/history/predictions,6001 projection vectors/receipts,work/oracles/child/summary.

- [ ] RED real-file tests for knownreturned/nonfinite/unknown interruption, latecall, exclusive duplicate root/probe, substitutedgate/sourcebytes, initialprojection versus originalbaseline, wrongatoms/biasmoments, provenance mismatch and last-published immutable commit. Inject small operations only below real file/control boundaries; private tiny real-model3update integration has no production CLI bypass.
- [ ] Implement canonical-config/exact source-map guards, source blueprint/hash/baseline binding. Record historicalreference, randominitializations0, sourcezero beforeprojection. Reserve root/start totalclock before imports; save frozen source/gate. No modelinitializer.
- [ ] Implement journal ready/finite/attempted counts and returnedoutput retention, accepted/computed/persisted distinctions. Failure preserves partialgradient/candidate/projectionvector/state/unknown flags and halts. Separate6001 calibrations from6000updates and warmup/reference work.
- [ ] Score both roles on63grid, raw predictions/residual/group stats; zero-source replay separate, baseline neverreset. Publish immutable state/history/predictions then receipt last. Save full oracle gradients/reference/actual outputs at1/3001, accepted actualcandidate once.
- [ ] Implement exclusive/token-bound child with config/provenance/source/state/history/input/gate/directory binding, no randominit, source blueprint/zero-bias-moment checks and endpoint144role predictions absolute1e-10. Retain childwork/stdout/stderr/entry/completion even on timeout, actual remainingglobalbudget, no retry.
- [ ] Implement frozen supervisor/ordinal1 outputfiles and processgroup cap. Core calls honor global reserve; all guards/compile/scoring/oracle/child/primaryclosure charged; latecalls invalidate.
- [ ] Run core/controller/relevant state/source regressions GREEN; freeze implementation for independent review, resolve material issues RED→GREEN. Final focused/fullsuite logs/scoped sourcecheckpoint before launch.

### Task3: saved report, controlled execution and closeout

**Files:** scripts/report_offset_profile.py; tests/test_offset_profile_report.py; reports/offset-profile-diagnostics.json/.md; reports/offset-profile-paper-findings.md; execution archive/figure/review/Obsidianbackups; append PAPER_EVIDENCE_LOG.md.
**Consumes:** Task2 rawwork/status/state/score/projection, original raw343 reference and source/launch maps.
**Produces:** build_report(root)->dict; render(root,outdir)->dict; independently verified saved closeout.

- [ ] RED fixtures for incomplete/missing-summary/invalidchild/gate/source drift suppressing endorsement; originalbaselines despite projection; population/group variance distinction; matching62postzero score maxima; densecorrections versus sparsereference separatelylabeled; exact archive/reference member sets.
- [ ] Implement saved-only state/history/prediction/projection/receipt audits, fresh source/gate checks, raw gates and matchinggrid outcomes/correction summaries. Decode arrays only; no model evaluation/retry. Rebuild JSON/Markdown byte-for-byte in resolved temporary directory; inventory/CRC/allmemberbytes checked.
- [ ] Run final report/focused/fullsuite verification before launch; record reviewPASS/config/artifact hashes/sourcecommit in frozen gate. Prospective rationale/fixedcounts in evidence, Obsidian plannedgates CLIbackup/readback. Launch supervisor ONCE.
- [ ] Monitor phases/progress; preserve excursions/failures without tuning/restart. After closure: saved-only reports/audits, matchinggrid rawenergy/force figure/caption/visualQA, archive and independent saved-result review. Preserve failed/interrupted reviews and repairs.
- [ ] Balanced paper findings retain construction effects, empirical centeredenergy/force results, gates/negative checks/historical/fitted/sparse limits; record abandoned paths. CLIupdate both Obsidian notes with backups/readbacks; final scopedresult commit, no productionrerun, fullpilotpending.

## Self-review and authorization ruling

All spec requirements and five reviewfocus risks map to tasks above. Task1call/retain contract feedsTask2; Task2 rawartifact contracts feedTask3. Primary inline as existing session; user authorized proceeding with this comparison, no redundant human permission gate for reversible implementation. Material scientific revisions require both independent followups. No production artifacts exist at plan stage.

## Engineering clarification A binding task details

Task1 returns actual masked derivative explicitly; script-local per-microbatch callbacks wrap raw kernels, not aggregate frozen helpers. Task2 source guards601exactpaths plus separately pinned spikeinventory; provenance distinguishes all-field sourcezero from all-field projectedzero and binds unchanged model/sampler/normalization at every state. Projection/accepted-step receipts bind ordinal/phase/update/orderedinputs/config/gate/state hashes/NPZ hash, score commits bind accepted receipt. NPZ/msgpack ready outputs precede finite checking/strictJSON. Task2/3 tests inject NaNoutput, ID/receipt swap, orphancommit, root identity/symlink, childnonzero/missingcompletion and substitutedgate; failed outputs/knownunknownwork remain retained and unendorsed. These concrete requirements resolve review acceptance checks before implementation.
