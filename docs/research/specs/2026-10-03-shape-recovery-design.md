> Research protocol copy. Dates, settings and checkboxes describe the saved protocol, not current experiment completion or a new launch approval. See [the experiment index](../../../reports/PAPER_EVIDENCE_LOG.md) for outcomes. Historical launch records referenced below are retained separately.

# Reference-shape recovery and paired relaxation design

3 October 2026. Status: proposed, unexecuted. This milestone writes the evaluator protocol, freezes the paired start set and the outcome taxonomy, and verifies a synthetic fixture slice. It does not evaluate any trained model and does not open the test partition.

## 1. What this evaluates

The main study compares an energy-derived E3x model, an ordinary direct-force model, a consistency-trained direct-force model and a same-split MACE reference. Static force error is a proxy. The claim in the proposal title is about the shape a molecule reaches, and this evaluator is what measures that.

Each arm begins from the same frozen start structure and is moved by its own predicted forces under one fixed procedure. A run succeeds only if it converges and its final structure matches that branch's reference DFT minimum under a symmetry-aware structural comparison.

This evaluator measures endpoints. It does not establish that a converging model is physically stable, that a converged direct-force endpoint is a minimum of any scalar potential, or that any result generalises beyond the frozen split.

## 2. Data and frozen inputs

- Archive: public Landscape17 v1, SHA-256 `3870ecfd6a5404c6654efba22f003f645e2f4dffffad1b5a22770d3437b3027b`, 53,019,632 bytes, CC BY 4.0.
- Molecule: aspirin only. Paracetamol is blocked by the unmatched endpoint `paracetamol/01_2/plus` and is out of scope.
- Split manifest: `manifests/landscape17-aspirin-split.json`, identity `73d5475bfc1f1158da62b939f5431b1781db886443cece2cdb297c8dd645133a`, read-only.
- Partitions as frozen: train 26 connections / 4,891 records, validation 5 / 974, test 6 / 1,127. Test connections are `25`, `00`, `07_2`, `56`, `910`, `67`.
- Evaluation set: 22 held-out branches, being both branches of the 5 validation and 6 test connections.
- Reference minima: 11, all `seen_in_training`. The test connections' target minima are also used by training connections.
- Policy: `configs/preparation.json` unchanged, including `nonstationary_force = 0.001` eV/angstrom.

The seen-minimum structure is recorded because it changes what the study can claim. Paths and transition states are held out; shapes are not. Every reported result carries this limit.

## 3. Frozen start-selection rule

Primary rule: for each of the 22 held-out branches, the start structure is the first stored frame of that branch.

Stored frame order was verified rather than assumed. Across all 74 aspirin branches, frame 0 lies within 0.10 angstrom of that branch's transition state on 66 branches and within 0.25 angstrom on all 74, while the branch travels a median of 0.6479 angstrom (range 0.1989 to 1.2943) from its first to its last stored frame. The stored order therefore runs from the transition-state end outward, and frame 0 is a genuine near-transition-state start. It is not always the single closest frame, which is at index 0 to 3 on 64 branches and as late as index 40 on one branch, so the rule is stated as the first stored frame and not as the closest frame.

Sensitivity rule, predeclared and reported alongside: the first frame whose RMSD to that branch's matched reference minimum is at or below 10 percent of the transition-state-to-minimum RMSD for the same branch.

Both rules are computed from the frozen split manifest and the archive alone. Neither is adjusted after any outcome is seen.

The choice between candidate rules was made on **selection data only**, being the 26 training and 5 validation connections. The 6 test connections were not used to choose the rule, in the same way they are not used to choose model settings. Test-side figures appear in this document only as an audit of what was withheld.

Candidate rules on selection data (62 branches). Every "RMSD left to target" cell is the atomic-coordinate RMSD, in angstrom, of the rule's chosen stored frame against that branch's matched reference minimum (the branch's `reference_class` in `reports/landscape17-endpoints.json`) under the repository's symmetry-aware structural comparison — `align_records` in `src/force_consistency/data/geometry.py`: mean-centred positions, graph-isomorphism atom correspondence (lexicographically smallest permutation at equal RMSD), best rigid rotation including reflection — and the "last nonstationary" frame is the branch's last stored frame whose maximum per-atom force component exceeds the preparation policy's `nonstationary_force` of `1e-3` eV/angstrom:

| Rule | Defined | RMSD left to target, median | min-max | Force at start, median | Nonstationary |
|---|---|---|---|---|---|
| First stored frame | 62/62 | 0.5749 | 0.1730-1.2553 | 8.44e-01 | 100% |
| 10 percent of TS | 62/62 | 0.0562 | 0.0183-0.1202 | 8.75e-02 | 100% |
| Last nonstationary frame | 62/62 | 0.0025 | 0.0003-0.0277 | 3.41e-03 | 100% |

On validation alone (10 branches) the same ordering and the same separation appear, so the decision does not depend on the training connections either:

| Rule | Defined | RMSD left to target, median | Force at start, median |
|---|---|---|---|
| First stored frame | 10/10 | 0.7629 | 9.81e-01 |
| 10 percent of TS | 10/10 | 0.0740 | 1.37e-01 |
| Last nonstationary frame | 10/10 | 0.0022 | 3.48e-03 |

The last-nonstationary-frame rule is rejected on selection data. Its median remaining distance is `0.0025` angstrom (range `0.0003` to `0.0277`), well above the `5e-4`-angstrom scale a near-endpoint start would need, and its median start force is `3.413e-03` eV/angstrom, which is within 1.24 times the reference minima's own upper residual bound of `2.745e-03` eV/angstrom. A relaxation from that start cannot separate models on shape recovery.

The first-frame rule is primary because it gives both the largest remaining descent and the strongest force signal. The 10-percent rule is the sensitivity check because it asks whether any result depends on starting close to the transition state.

## 4. Relaxation procedure

One procedure, identical for every arm:

- Integrator: steepest descent on the predicted force field, with a backtracking step-size rule. No learned step size, no adaptive schedule beyond the fixed rule.
- Convergence: maximum per-atom force component at or below `1e-2` eV/angstrom. This is set above the reference minima's own residual range so that a model is not penalised for matching reference-level noise, and it is stated as a maximum component rather than a norm.
- Budget: a fixed maximum number of force evaluations per branch, recorded in the config. The budget is the same for every arm and is not extended for a failing arm.
- Termination: on the convergence rule, on budget exhaustion, or on a nonfinite force. No early stopping is offered to any arm.
- Determinism: the same start, procedure and force function must reproduce the final structure bitwise across a fresh process.

Force-call counts, wall-clock per branch and the number of evaluations to convergence are recorded per branch so that cost can be compared later. Cost comparison is not part of this milestone.

## 5. Outcome taxonomy

Every branch ends in exactly one category, and the categories never collapse into each other:

| Category | Meaning |
|---|---|
| `matched_reference` | Converged and the final structure matches the branch's own reference minimum under the symmetry-aware comparison. A success. |
| `unmatched_converged` | Converged to a force-field fixed point that is not the branch's reference minimum. Not a success, and not a failure to converge. |
| `ambiguous` | The final structure is within tolerance of more than one reference equivalence class. |
| `unresolved` | The comparison could not be completed under the policy search limit, or a nonfinite value occurred. |
| `nonconverged` | The budget was exhausted, or termination was triggered by a nonfinite force, before the convergence rule was met. |

Reported denominators are per connection and per partition, and matched-reference, unmatched-converged and nonconverged counts are reported separately. A converged fixed point is never described as a minimum, and a nonconverged run is never counted as a wrong shape.

Each branch keeps a `seen_minimum` flag taken from the frozen manifest, so a held-out connection whose target shape was seen through a training connection is visible in every table.

## 6. Interface with the model arms

The evaluator consumes one callable per arm, conforming to the shared adapter agreement that Shedrack owns: record identity, atomic-number species encoding, preserved atom order, positions in angstrom, force output shape `(N, 3)` in eV/angstrom, optional scalar molecular energy in eV, and an explicit status when a quantity is unavailable.

An unavailable quantity stays unavailable and is never replaced with zeros. A direct-force arm that has no meaningful energy is fully valid for shape recovery and is refused for any energy-based metric, with the refusal recorded rather than silently skipped.

## 7. Selection, freeze and test discipline

- Model settings and any comparison checkpoints are selected on training and validation data only.
- The start set, the relaxation procedure, the convergence rule, the outcome taxonomy and the report schema are committed before any test outcome is produced.
- The test partition is evaluated once, after freeze, under a separate authorisation from the one that approves this implementation.
- If validation force-error ranges across models do not overlap, the analysis reports the accuracy, cost and recovery tradeoff and does not claim a matched-error superiority.
- Uncertainty is grouped by whole connection. Frames and seeds are not treated as additional independent connections.

## 8. What this design does not do

- It does not evaluate a trained model in this milestone. The fixture slice uses an analytic arm with a known target on training connections only.
- It does not claim that a consistent model is physically conservative at finite penalty weight.
- It does not test generalisation to unseen shapes, because every aspirin reference minimum is training-seen.
- It does not run or depend on GPU hardware, and it makes no compute claim.

## 9. Verification of this design's inputs

Produced by the reviewer from the public archive using the repository's own data package, before this document was written:

- `read_records` and `inventory_connections` parse 6,992 aspirin records, 11 minima, 37 transition states and 6,944 path frames.
- `map_endpoints` reproduces all 74 committed aspirin branch classifications with zero field-level mismatches.
- `assign_groups` reproduces the frozen split identity exactly, with matching train, validation and test membership.
- `audit_duplicates` reports zero duplicate pairs, zero unresolved duplicates and zero exact-label conflicts.
- `apply_eligibility` reproduces 4,891 train, 974 validation and 1,127 test records.
