# Paired relaxation traces and reference-start controls, 18 October 2026

Status: completed once and accepted by independent engineering and scientific saved-result reviews. The original validation recovery result remains 0/10 for every evaluated model, with A2 failed. This diagnostic does not replace that outcome.

## What ran

Four fixed post-hoc replays used the original BEST checkpoints, original validation starts and unchanged relaxation procedure: direct seed 0, consistency λ=0.1 seed 0 and energy C seed 0 on aspirin/18/plus, plus energy C seed 0 on aspirin/13/plus. Each reproduced its saved endpoint exactly, with maximum coordinate difference 0.0 Å and identical force/step counts, stopping flags, failure reasons and official category. Historical paths were not recorded, so this establishes reproduction of the saved outcomes, not bit-identical historical trajectories.

Thirty additional controls started each of the same five models at each of the six distinct classified own references, minima 1, 3, 6, 7, 8 and 9. These were fixed before launch. The unchanged force threshold is maximum absolute Cartesian component at most 0.01 eV/Å; own-reference geometry requires the existing symmetry-aware 0.05 Å tolerance. No model, strength, checkpoint, threshold, solver or output was selected from the diagnostic outcomes. The main test remained closed.

The package used 38,853 force attempts across 34 relaxations, below the 51,000 ceiling. Numerical execution took 74.898 seconds and the parent supervisor took 75.758 seconds, without timeout, retry or extra energy/auxiliary-gradient query. The 172 saved files total 424,269,373 bytes, below 1 GiB. These are local diagnostic timings, not GPU billing or deployment costs.

## Observed coordinate jumps

| Selected replay | Largest accepted per-atom vector displacement (Å) | First accepted disconnected state, zero-based query index | Outcome reproduced |
|---|---:|---:|---|
| Direct seed 0, 18/plus | 436.720784 | 986 | Force-converged, own geometry unmatched |
| Consistency λ=0.1 seed 0, 18/plus | 0.086492 | None | Nonconverged, connected endpoint |
| Energy C seed 0, 18/plus | 0.006083 | None | Nonconverged, connected endpoint |
| Energy C seed 0, 13/plus | 268.047200 | 2 | Nonconverged, separated endpoint |

The direct replay's first accepted disconnection occurs at query 986 with a 104.249071 Å maximum atom displacement. Its largest 436.720784 Å jump occurs later, at query 1004. Terminal query 1016 moves an atom by up to 32.281 Å and lands at an all-isolated, exactly zero-force geometry. This is distinct from claiming the largest jump was the first separation. The energy 13/plus replay first disconnects at query 2, which is also its largest accepted jump. The preceding energy state has minimum interatomic separation 0.116071 Å and maximum force component 576.446471 eV/Å; scale 0.4 Å²/eV then yields the observed large displacement. These measurements describe an observed replay path, not physical molecular dissociation.

The traces establish that the unchanged procedure accepts extreme coordinate jumps with both energy-derived and direct forces. Its cap bounds scale rather than displacement, and its agreement test does not require energy decrease. This is stronger evidence than the previous endpoint-only inference. It does not establish the procedure as the sole cause, exclude model error or component drift elsewhere, or estimate how often such jumps occur in unselected branches. The paired branch was chosen after inspecting earlier outcomes; its differences do not establish a consistency recovery advantage.

## Reference-start behavior

| Frozen model | Force-converged own matches from six reference starts |
|---|---:|
| Energy C, seed 0 | 4/6 |
| Direct, seed 0 | 1/6 |
| Consistency λ=0.1, seed 0 | 0/6 |
| Direct, seed 1 | 0/6 |
| Consistency λ=0.1, seed 1 | 0/6 |

The four energy successes are minima 1, 3, 6 and 7, with symmetry-aware endpoint RMSDs 0.029406, 0.010850, 0.011421 and 0.007178 Å. Direct seed 0 succeeds at minimum 1 with RMSD 0.027784 Å. Energy minimum 8 converges but misses geometry (RMSD 0.080398 Å); minimum 9 exhausts the force budget (RMSD 0.119685 Å). Every model/reference outcome, stop reason and incomplete or rejected geometry fact is retained in the saved records.

These successes show that exact-reference force mismatch does not rule out a nearby threshold-converged own match. They do not establish an exact stationary point, a stable minimum or its energy curvature. A failed reference-start control means this frozen procedure did not establish the joint force/geometry result within its budget. It cannot prove no nearby stationary point exists or that all recovery failure lies solely in the model. The six references are shared, seen minima rather than independent unseen-minimum or population samples. Keep seed-specific counts separate; this is not a new matched recovery estimate.

## Verification and next decision

The coordinator decoded all 38,853 saved position/force pairs and recomputed the full observer summaries. Independent engineering and scientific audits separately reconstructed decisions, graph connectivity, displacements, component motion, raw net force and torque, final classifications and every control result. The scientific audit reports 947,793 raw-array checks. All 172 file hashes reproduce. All 468 earlier protected output files and frozen scientific sources remained unchanged. No learned-force query was issued during result review.

The diagnostic supports an explicit methodological limitation of the frozen procedure while retaining model and reference-control limitations. It does not justify silently adding a displacement cap, changing the training recipe or rerunning validation with a selected solver. MACE under the same procedure remains a separate reviewed reference check after the midterm draft. Any later procedure version needs fixed standard settings, comparable treatment of every model and a clear post-hoc label for validation results. Midterm writing remains October 20–23.

Accepted records: reports/execution/e3x-paired-trace-2026-10-18/. Raw traces: runs/aspirin-e3x-paired-trace-2026-10-18/. Full checkpoints and earlier unsuccessful attempts remain preserved. GitHub publication of the accepted result and supporting evidence is the next delivery step.
