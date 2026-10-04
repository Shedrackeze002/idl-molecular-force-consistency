> Research protocol copy. Dates, settings and checkboxes describe the saved protocol, not current experiment completion or a new launch approval. See [the experiment index](../../../reports/PAPER_EVIDENCE_LOG.md) for outcomes. Historical launch records referenced below are retained separately.

# Paired Relaxation and Reference-Shape Recovery Implementation Plan

**Goal:** Build and verify the reference-shape recovery evaluator that all model arms of the main study will be scored with, starting from a fixture-tested slice and ending in one frozen paired-relaxation procedure.
**Architecture:** New pure evaluation package plus a saved-only reporter. Reuse the unchanged preparation, geometry and matching helpers. No change to `src/force_consistency/data/`, frozen configs, manifests or historical runs.
**Tech Stack:** Existing Python 3.12 environment, NumPy/SciPy/ASE/networkx. Relaxation is deterministic force-driven minimisation with no learned component.
**Spec:** `docs/research/specs/2026-10-03-shape-recovery-design.md`.

## Global Constraints

- Scope is the exact spec and `configs/aspirin-shape-recovery.json`. Aspirin only. Paracetamol remains blocked at `01_2/plus` and is out of scope for this milestone.
- Reuse `align_records`, `classify_endpoint`, `chemical_graph`, `descriptor` and `evaluate_partition` unchanged. Do not reimplement alignment, matching or pooling.
- Consume the frozen split `73d5475bfc1f1158da62b939f5431b1781db886443cece2cdb297c8dd645133a` read-only. Test groups (25, 00, 07_2, 56, 910, 67) stay closed until every threshold, start set and scoring rule is frozen and committed.
- Every arm is relaxed with the same procedure, the same start structure, the same step budget and the same convergence rule. No arm-specific settings, no early stopping for a favoured arm, no retry.
- Report matched-reference, unmatched-converged, and nonconverged outcomes with explicit denominators. A converged direct-force endpoint is a force-field fixed point, not automatically an energy minimum.
- No energy, stability, mechanism or generalisation claim is made by the evaluator. It measures endpoints only.
- Two independent plan reviews before implementation. Material revisions go back to both reviewers. Execution of a test-group evaluation is a separate later authorisation.

## Review Focus

1. Start-selection rule. Does the frozen rule give a genuinely nonstationary, informative start (see Measured inputs), and is it fixed from validation data only?
2. Convergence rule. Is the force threshold defensible against the reference minima force level, and is the criterion stated per atom or as a maximum component?
3. Outcome taxonomy. Are matched, unmatched, ambiguous, unresolved and nonconverged kept distinct, and is the seen-reference-minimum flag preserved?
4. Determinism and symmetry. Does the relaxation reproduce bitwise from a saved start, and does final matching use the same symmetry-aware comparison as preparation?
5. Leakage. Does anything read test frames, test endpoints or test outcomes before freeze?

## Measured inputs for this design

Produced by this reviewer from the public archive (SHA-256 `3870ecfd6a5404c6654efba22f003f645e2f4dffffad1b5a22770d3437b3027b`) using the repository's own data package. These are inputs to the decision, not results of the study. Candidate rules were compared on selection data only; the held-out test figures in this section are an audit of what was withheld, not a basis for any choice.

- Aspirin archive: 6,992 records, 11 reference minima, 37 transition states, 6,944 path frames.
- Reproduced exactly: endpoint classification for all 74 aspirin branches, the split identity, the 4,891 / 974 / 1,127 partition counts, zero duplicate pairs, zero unresolved duplicates, zero exact-label conflicts.
- Held-out evaluation set: 22 branches across 5 validation and 6 test connections.
- All 11 reference minima are `seen_in_training`. The test connections' target minima are used by training connections as well, so this is a held-out-path and withheld-transition-state split, not a held-out-shape split.
- Reference minima residual force: maximum component between `5.347e-04` and `2.745e-03` eV/angstrom.
- Start-rule comparison on selection data (62 branches) and on validation alone (10 branches), RMSD to that branch's matched reference minimum, maximum force component at the start frame:

| Rule | Selection, 62 branches: left / force | Validation, 10 branches: left / force |
|---|---|---|
| First stored frame | 0.5712 / 8.44e-01 | 0.7608 / 9.81e-01 |
| Last nonstationary frame | 0.0000 / 3.41e-03 | 0.0000 / 3.48e-03 |
| 10% of TS-to-minimum | 0.0544 / 8.75e-02 | 0.0631 / 1.37e-01 |
| 25% of TS-to-minimum | 0.1450 / 1.96e-01 | 0.1860 / 2.84e-01 |

The same ordering holds on training alone, validation alone, and the two pooled, so the decision is stable across selection subsets.

The last-nonstationary-frame rule starts essentially at the target: the remaining distance to the matched reference minimum is below `5e-4` angstrom on every branch, and the force signal the model is asked to follow is about `3.4e-03` eV/angstrom, close to the reference minima's own residual. A relaxation from that start cannot distinguish models. The rule should not be adopted.

The first-frame rule gives the largest remaining descent and the strongest force signal, and is defined on all 74 branches. The 10%-of-TS rule gives a middle option and never starts later than the last nonstationary frame. The design freezes the first-frame rule as primary and keeps the 10%-of-TS rule as a predeclared sensitivity analysis. Both are fixed before any test outcome is seen.

Frame order was also checked directly rather than assumed. Frame 0 is within 0.10 angstrom of the branch's transition state on 66 of 74 branches and within 0.25 angstrom on all 74, and a branch travels a median of 0.6479 angstrom from its first to its last stored frame. Frame 0 is therefore a genuine near-transition-state start, though it is not always the single closest frame: the closest frame sits at index 0 to 3 on 64 branches and as late as index 40 on one branch. The rule is stated as the first stored frame, not as the closest frame.

---

### Task 1: Frozen start set and outcome taxonomy

**Files:** Create `configs/aspirin-shape-recovery.json`; `evaluation/start_set.py`; `evaluation/outcomes.py`; `tests/test_shape_recovery_starts.py`; `tests/test_shape_recovery_outcomes.py`. Planning records under `reports/execution/shape-recovery-2026-10-03/`. Do not modify preparation source, frozen configs or existing tests.

**Interfaces:** `build_start_set(split_manifest, records, policy, rule:dict) -> dict` returns per-connection, per-branch the start record id and its stored frame index, together with the start structure read from the archive and the rule identity. It does not read endpoint status. `classify_outcome(final: FrameRecord, reference: FrameRecord, policy, *, budget_exhausted: bool, converged: bool) -> dict` returns one of `matched_reference`, `unmatched_converged`, `ambiguous`, `unresolved`, `nonconverged`, with the measured RMSD, the permutation and rotation, and the reference class.

- [ ] Write failing tests for the start rule: every one of the 74 branches yields exactly one start; the start is stored frame 0 of that branch; the returned structure is byte-identical to the archive frame it names; the rule is reproduced from the frozen split manifest alone and never reads test endpoint status.
- [ ] Write failing tests for the outcome taxonomy: a structure placed at each reference minimum classifies as `matched_reference` to that class only; a structure midway between two minima classifies as `ambiguous`; a run that hits the step cap without meeting the force rule classifies as `nonconverged` and is never counted as a failure of shape recovery; a run with no finite force classifies as `unresolved` and is retained.
- [ ] Run the focused tests and preserve the actual RED output.
- [ ] Implement `build_start_set` and `classify_outcome`. Record the chosen start record ids and a canonical hash of the start set.
- [ ] Commit config, both modules, both test files and the recorded start-set hash.

### Task 2: Frozen relaxation procedure

**Files:** Create `evaluation/relax.py`; `tests/test_shape_recovery_relax.py`.

**Interfaces:** `relax(start, force_fn, procedure:dict) -> dict` returns the final positions, the step count, the force history summary, the convergence status, the number of force evaluations, and a deterministic failure status. `force_fn` is any callable mapping positions and atomic numbers to a `(N, 3)` force array in eV/angstrom.

- [ ] Write failing tests using analytic potentials with known minima: a harmonic well must converge to its analytic minimum within a stated tolerance; a double well started on either side must converge to the correct basin; a constant nonzero force field must be reported as `nonconverged` at the step cap with the endpoint retained and not scored as shape recovery.
- [ ] Write a failing test for determinism: the same start, procedure and force function must reproduce the final structure bitwise across two fresh processes.
- [ ] Write a failing test for the force-call budget: an arm that exceeds its budget is marked incomplete and is not silently truncated into a success.
- [ ] Implement `relax` with the procedure fixed in the spec. Record the force-evaluation count exactly.
- [ ] Commit the module, tests and the recorded analytic fixtures.

### Task 3: Adapter, reporter and the synthetic end-to-end slice

**Files:** Create `evaluation/adapter.py`; `scripts/report_shape_recovery.py`; `tests/test_shape_recovery_report.py`; `reports/shape-recovery-fixture-diagnostics.json` and `.md`.

**Interfaces:** `arm_from_checkpoint(path, contract) -> dict` returns a callable conforming to the shared model interface agreed with Shedrack: record identity, atomic-number species encoding, preserved atom order, positions in angstrom, force output shape `(N, 3)` in eV/angstrom, optional scalar energy in eV, and an explicit status when a quantity is unavailable. `evaluate_starts(start_set, arm, procedure) -> dict` returns per-branch and per-connection outcomes. `build_report(...) -> dict` and `render(...) -> dict` write the diagnostics pair.

- [ ] Write failing tests for the adapter contract: missing or nonfinite forces are reported unavailable and never replaced with zeros; a direct-force arm with no meaningful energy is accepted for shape recovery and is refused for any energy-based metric; atom order and species encoding survive a round trip.
- [ ] Write failing tests for the reporter: forged or missing outcomes are rejected; per-connection denominators sum to the evaluated branch count; the seen-reference-minimum flag is preserved per branch; a zero-denominator connection is reported as unavailable rather than as zero percent.
- [ ] Implement the adapter, the runner and the reporter.
- [ ] Run the slice end to end on a synthetic fixture arm that has a known analytic target, across a small subset of training connections only. Verify that the slice reports the known outcome and that it refuses the test partition.
- [ ] Produce the fixture diagnostics pair and a one-page note stating what the fixture proves and what it does not.
- [ ] Commit the slice, tests, fixture outputs and note.

## Self-review and handoff

Every spec section maps to Task 1 (start set and taxonomy), Task 2 (relaxation), or Task 3 (adapter, reporter and the fixture slice). The interfaces reuse the existing geometry and matching helpers and add no replacement scorer. The scientific claim, the frozen evaluation policy and the test partition stay distinct. No test outcome is read before the start set, the procedure and the thresholds are committed. This plan does not report an unexecuted evaluation as a result, and this milestone does not evaluate any real model arm.

## Open questions for the team

1. **Shared adapter agreement.** Task 3 depends on the model interface Shedrack owns. The concrete items needed are listed in `PROJECT_HANDOFF.md` under "Common model interface". This must be agreed before Task 3 can start, though Tasks 1 and 2 do not depend on it.
2. **Where the evaluation package lives.** `docs/team/PROJECT_HANDOFF.md` suggests an evaluation package for Olufunke and Emmanuel. Proposed path is top-level `evaluation/` with owner-specific subpackages, so the two workstreams do not collide. Needs to be an agreed interface before either side implements.
3. **Reviewers.** Two independent technical reviewers for this plan, as `TEAM.md` requires.
4. **The last-nonstationary-frame rule is not usable.** The measurement above should be recorded in the thread so the rule is not reintroduced later.
5. **Test groups stay closed.** The first test evaluation needs its own authorisation after freeze, separate from this implementation approval.
