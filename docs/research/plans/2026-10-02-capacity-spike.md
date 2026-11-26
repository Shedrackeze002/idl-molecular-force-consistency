> Research protocol copy. Dates, settings and checkboxes describe the saved protocol, not current experiment completion or a new launch approval. See [the experiment index](../../../reports/PAPER_EVIDENCE_LOG.md) for outcomes. Historical launch records referenced below are retained separately.

# Saved capacity spike diagnostic implementation plan


**Goal:** Describe energy-residual and loss-gradient changes at fixed saved capacity pre/peak/recovery/endpoint states without training.
**Architecture:** Standalone analysis script reuses frozen source helpers, pure source guards, msgpack payload restore and existing gradient utilities; separate saved-report function only reads artifacts. Exclusive bounded numerical execution saves partial work and failures.
**Tech stack:** Existing Python/JAX/Flax/NumPy/msgpack runtime; no dependencies installed.
**Spec:** docs/research/specs/2026-10-02-capacity-spike-design.md

## Global constraints

- Preserve all343 source rawfiles, participating src/lock files, original capacity reports, original source files and review/test directories.
- Fixed8cases/order and1800s cap/60s reserve; CPUfloat64; no retry/training/optimizer proposals/random initialization.
- Fitting128/anchors16 (all fitted), microbatch16; production weights/config unchanged.
- Every meaningful trial recorded, including failed engineering/derivative checks and blocked numerical cases.
- Two independent scientific/engineering plan reviews and material followups before implementation. Implementation review before numerical launch.

## Review focus

1. Tampered/omitted source or state identity must fail before numerical work.
2. Offset/gradient ratios with zero denominator must remain undefined, not invented zeros.
3. Weighted gradients must use the historical weights and independent weighted-objective identity.
4. Timeout/verification failure must preserve partial observations/work and stop subsequent cases.
5. Invalid observations must never become endorsed findings by merely existing on disk.

### Task1: Pure controls, residual reduction and source restoration

Files: create scripts/diagnose_capacity_spikes.py; tests/test_capacity_spikes.py; configs/aspirin-capacity-spike.json; reports/execution/spike-diagnostic-2026-10-02/source-pins.json.
Interfaces: verify_inputs(config)->capacity_report; residual_summary(energy_error,force_error,groups)->dict; load_snapshot(case_root,update,provenance)->decoded_state after existing pure serialized audit.

- [ ] Write meaningful tests for residual identity on constant errors[2,2], centered[-1,1], mixed[1,3], zero[0,0], nonfinite/shape rejection and unequal-group pooled vs equal-group aggregation.
- [ ] Run RED and preserve output.
- [ ] Implement pure reductions, frozen case/config identity, complete source map hashes, original report canonical check; reject changes to orderedIDs/memberships, norm/count/schema/state hashes using source audits.
- [ ] Test complete pin-map allowlist/tampered files before numerical import, wrong history state, named zero-MSE share and unchanged synthetic decoded state.
- [ ] Run GREEN and preserve all failures including fixtures/inspection trials.

### Task2: Immutable numerical analysis and failure retention

Same script/test files.
Interfaces: analyze_case(state,parts,model,norm,weights,history_row,check,on_partial)->dict; execute(config)->summary; gradient_metrics(ge,gf,gj,weighted_joint,weights)->dict; render(run_root,output)->report reading savedJSON/NPZ only.

- [ ] Add tests using existing small model witness for weighted identity/nonunit weights, head-bias analytic E derivative and structurally zero F derivative; preserve source before/after and verify predictions/history tolerances.
- [ ] Add injected analyzer/budget tests ensuring exclusive root, oneattempt, retained phase/partial/completed observations after late failure, downstream blocked cases and actual known work. Report tests ensure invalid observation files cannot appear as verified results, missingcases blocked and every selected case retained.
- [ ] Run RED; implement minimal standalone orchestration. Use existing gradient_geometry with explicitly relabeled slope scales; script-local instrumented loops call the frozen _objectives, _gradient, _weighted_gradient and _predict kernels. Do not use broad split/accumulation/FD helpers lacking per-microbatch completion hooks. Implement the existing FD unit-gradient/zero fallback rule with fixed steps and per-plus/minus microbatch completion receipts. Never call initialize_state/buildstep/optimizer update. Restore actual payload using _read_checkpoint plus msgpack_restore following pure audit, pass onlyparams to model.apply. Save rawprediction/error arrays percase and gradients serialized with clear scales.
- [ ] Reconcile pooled/pergroup/equalgroup scores with historical rows; exact endpoint references. Validate weighted/unweighted loss/gradient identities, energy bias analytic derivative, force bias zero, FDcriterion, source hash immutable. Preserve phase counters aftereach completed microbatch and unknown in-flight work on abruptprocess failure rather than fabricated totals.
- [ ] Render saved-only descriptive residual changes/block norms/ratios/cosines and local negative-joint-gradient slopes explicitly distinct from actual Adam directions. Gate interpretation on all case checks and runstatus, preserve failures.
- [ ] Run focused tests, relevant existing gradient/state/report tests and full suite once ifsource concerns/regressions justify. Independent implementation review and materially repaired followup before launch. Record full participating source/runtime identities and scripts/config/reviews hashes.

### Task3: One bounded execution and evidence closure

- [ ] Append prospective rationale, fixedcases, posthocselection limitations, controls, checks and plannedwork separately in evidence; checkpoint analysis source/reviews scoped with original source files excluded.
- [ ] Freeze prelaunch gate and launch once withstdout/stderr/logs, operational killtimeout1860s, no retry. Record loadedstates/definitioninstances separately from initializations and nooptimizerwork.
- [ ] Monitor phases and append meaningful completed observations/failures; no changes to numerical code afterlaunch.
- [ ] Saved-only canonical report, rawhash inventory/archive CRC/members and independent result review; retain invalid/blocked outcomes and no scientific interpretation until verified.
- [ ] Paper-facing findings and all trial evidence; ; scoped local resultcheckpoint. Any next intervention remains proposed, requiringnewplan+tworeviews.

## Self-review

Design controls map to task1 source/residual checks, task2 weighted/FD/failure/status checks and task3 verified evidence closure. Selected cases follow actual history rows. Utility unweighted geometry must not be mislabeled production-weighted geometry. No hidden training branch or template initializer. User's proceed authorizes this bounded saved-checkpoint diagnostic; avoid duplicate authorization requests. Dedicated existing checkout retained for path-bound ignored source artifacts.

## Amendment A task coverage

- Task1 freezes signed residual convention prediction-minus-target and distinguishes pooled/global from group offsets.
- Task2 tests n_i=21 atomic-bias derivative, weighted/unweighted slope formulas with a!=b and known opposing vectors, named zero derivative directions; no per-molecule bias assumption.
- Task2 tests interrupted microbatch in-flight journal remains unknown, completion recorded only after block_until_ready/finite-check, FD first failed/second passed criterion with both retained, post-compute cap overrun keeps observations but invalidates case/run; no later numerical work.
- Task2 instruments all kernels in script-local loops with pre-call budget/source directory checks, completion receipt and partial save. Independent weighted/unweighted joint kernels remain separate from the split-gradient sum. Frozen src stays unchanged.
- Task3 freezes total1800s inclusive60s reserve, operational supervisor1860s. Any overrun/missingcase invalidates complete-run interpretation; valid earlier case observations are separately labeled. Expected work72/256/512 microbatch calls and1152/4096/8192 structure exposures for prediction/gradient/FD respectively. All source-state hashes and data/code identities checked before and after.

Ruling: user proceed authorizes bounded diagnostic, and established inline method is preserved. The two requested plan reviews replace redundant approval handoffs; they do not authorize a training intervention. Material Amendment A goes through both reviewers before implementation.

Final engineering clarification: budget begins immediately after exclusive root/ordinal1reservation, before guards/imports/compilation. Task2 tests known ready/returned work retained on finite-check failure versus unknown in-flight work, full decoded-state hash audit on failure paths, and global source-integrity failure invalidating all endorsements. FD zero threshold1e-12, seed20261002. These clarify Amendment A without changing selected states or scientific thresholds.
