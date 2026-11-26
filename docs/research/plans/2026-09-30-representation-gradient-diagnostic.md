> Research protocol copy. Dates, settings and checkboxes describe the saved protocol, not current experiment completion or a new launch approval. See [the experiment index](../../../reports/PAPER_EVIDENCE_LOG.md) for outcomes. Historical launch records referenced below are retained separately.

# Representation and Gradient Diagnostic Implementation Plan


**Goal:** Measure necessary force-representation constraints and separate energy/force parameter gradients at frozen coverage states, without training or changing settings.
**Architecture:** Pure NumPy rigid-motion projection/graph probes; JAX microbatched split-gradient/finite-difference probes; train-only, hash-gated checkpoint controller and report. Reuse existing loss, checkpoint and coverage selection contracts.
**Tech Stack:** Locked Python3.12/JAX0.9.2/E3x1.0.2/NumPy/Flax; float64 CPU, no new dependencies.
**Spec:** `docs/research/specs/2026-09-30-representation-gradient-diagnostic-design.md`.

## Global Constraints

- Exact frozen sources/config in `configs/aspirin-representation-gradient-diagnostic.json`; same prepared/split/subset identities and16/16/128IDs. Hash-check all source files before train-only array access.
- Original seed0 zero-head16feature/degree1/2iteration/8basis/5Å model, train scales and saved final2000-update states. Historical/current provenance separate. No optimizer update or new training run.
- Eight fixed cases in spec order, microbatch16, sequential frame-weighted gradients, no parameter movement between chunks. Hash before/after including ephemeral finite-difference probes.
- Centered rigid-motion columns, column normalization and SVD rcond1e-12; net force/torque and pooled/component-weighted floors. No projected label substitution or altered gate.
- Gradient identity atol1e-10/rtol1e-8. Unit-gradient direction (fallback seeded Rademacher for norm<=1e-12); FD1e-4/1e-5, at least one within atol1e-8/rtol1e-4; save both. Prediction-Q tolerance1e-8; scalar decomposition atol1e-10/rtol1e-8.
- Whole-analysis600s with10s reserve, operational blocks, fresh protected outputs, no resume/extra attempt. Log prospectively/after each stage, failures and abandoned cases.
- Preserve old gates, all prior artifacts and user proposal changes. No full pilot/GPU/test/scientific arm/seed/architecture/loss/LR/capacity/budget change.

## Review Focus

1. Degenerate/collinear geometry and unit-dependent SVD rank: valid lower bound with documented numerical rank, no singular inverse.
2. Misinterpreting label rigid components as corruption, a capacity bound, projected-label acceptance or causal mechanism: preserve raw labels and claims.
3. Uneven chunks/empty or zero gradient blocks: original loss derivative identity, correct counts, null undefined cosine/ratios.
4. A source state/array mismatch or partial exception must preserve attempted observations and block only pending cases; no collision overwrites.
5. FD probes and graph descriptions cannot be mistaken for optimizer trials/Adam trajectory, expressive sufficiency or a changed learning protocol.

### Task 1: Necessary constraints and graph diagnostics

**Files:** create `src/force_consistency/representation.py`, `tests/test_representation.py`.
**Interfaces:** `rigid_projection(positions: np.ndarray,forces: np.ndarray,rcond: float=1e-12)->dict` returns parallel/compatible vectors, rank, singular values, physical net force/torque, raw/translation/floor/compatible MSE and checks. `partition_constraints(partition: PreparedPartition,predicted_forces: np.ndarray|None=None)->dict` returns per-record and pooled/equal-group values, counts and unchanged IDs. `cutoff_graph(positions: np.ndarray,cutoff_A: float=5.,hops: int=2)->dict` returns exhaustive geometry graph statistics.

- [x] Write analytic translation/rotation/harmonic/superposition and least-squares-oracle tests; degenerate collinear/single-atom, rigid transformations and coordinate scaling must preserve floor. Invalid shape/nonfinite/empty input rejects. Watch missing-module RED.
- [x] Implement orthogonal SVD projection and invariant prediction/decomposition checks, pooled force-component averaging and metadata. Graph strict cutoff/no self edges, exact connectivity/two-hop counts; verify disconnected and cutoff-boundary fixtures.
- [x] Run `.venv/bin/pytest -q tests/test_representation.py`; expect all tests GREEN; log any failure and commit scoped helpers/tests.

### Task 2: Frozen split gradients and numerical verification

**Files:** create `src/force_consistency/gradient_diagnostics.py`, `tests/test_gradient_diagnostics.py`.
**Interfaces:** `split_gradients(params,partition,model,normalization,microbatch_size=16)->(losses,energy_grad,force_grad,joint_grad)`; `gradient_geometry(energy_grad,force_grad)->dict`; `verify_directional_derivatives(params,partition,model,normalization,energy_grad,force_grad,steps=(1e-4,1e-5),seed=20260930)->dict`; `analyze_snapshot(state,partition,model,normalization,config)->dict` returns finite values, all/per-block/per-leaf geometry, FD results, immutable state identities/counts.

- [x] Write7frame3/3/1 chunk split-vs-full joint-gradient tests; exact sum identity, fixed params, nonzero mixed force-backbone flow after nonzero witness head. Gradient algebra fixtures pin positive/negative/orthogonal/zero cosines/dot/norms. Watch RED missing helpers.
- [x] Implement original scaled losses only. Stable whole-tree dot/norm summaries, backbone/head-kernel/head-bias/per-leaf entries; null zero norms. Verify existing objective derivative independently through accumulated helper.
- [x] Write FD tests for energy/force gradients, wrong sign causing failure, both-step results retained, fixed fallback direction and unchanged params/optimizer. At zero head backbone force gradient/bias zero is recorded, not diagnosed as broken. Implement and run `.venv/bin/pytest -q tests/test_gradient_diagnostics.py tests/test_coverage.py`; expect GREEN; commit.

### Task 3: Hash-gated controller, execution and balanced report

**Files:** create `src/force_consistency/gradient_runs.py`, `tests/test_gradient_runs.py`, `scripts/report_representation_gradients.py`; modify CLI; create reports after execution.
**Interfaces:** `load_gradient_sources(config)->(train,manifest,normalization,subsets,fresh_state,final_states,historical_provenance)` verifies all frozen hashes/settings/predictions and training arrays before compute. `diagnose_gradients(config:dict)->dict` saves PhaseA and eight cases; operational failure blocks pending cases while retaining completed ones. CLI `diagnose-gradients --config` exits on execution validity, never declares learning acceptance.

- [x] Write stale-source-before-loader, corrupt state/settings, fresh state identity and saved prediction input checks. Controller fixtures cover full case order/count/IDs, protected outputs, no optimizer updates, injected partial failure after artifacts and wall cap with accurate attempted/blocked statuses. RED then implement; use real small-model snapshot components when integration cost permits.
- [x] Save config/current and historical provenance, exact per-phase/case prospectives/results with hashes and limits. Training-only loader; source checkpoints/inputs verified via existing code. Reload finite model state at frozen2000 counter, match fresh hashes, no old writes. Verify root/case incremental summaries and immutable sources.
- [x] Run full `.venv/bin/pytest -q` before research analysis; expect prior82 plus new tests GREEN. Commit implementation and prospective evidence.
- [x] Execute exact fixed config once, wall600s. Log every actual phase/case, partial failure or block immediately. Research outcome UNKNOWN; no further intervention if unresolved.
- [x] Reconcile floors/raw MSE/gates/graph support and gradient identities/FD/nulls/source hashes. Save report/JSON and measured scientific plots, explicitly separate geometry necessity/local gradients/Adam dynamics. Visually inspect figures, record generator/checksums.
- [x] Fresh independent final review; resolve material findings with RED-to-GREEN regressions and full passing suite. Commit results and local fixes, no merge/push/cleanup or full pilot.

## Decision after this milestone

Implementation/source verification failure requires repair before mechanistic interpretation. Necessary-floor obstruction motivates reviewed source/model-target contract audit, not label alteration. Valid low floors plus gradient evidence motivate a separately reviewed intervention, never automatically identify capacity or authorize changed weights/training. Save unresolved conclusions too.

Execution completion: all three tasks, one exact production execution, independent final review and two RED-to-GREEN failure-handling repairs complete. Full final suite117passed126.17s. See review/evidence/execution record; no production rerun, optimizer update or learning gate change.
