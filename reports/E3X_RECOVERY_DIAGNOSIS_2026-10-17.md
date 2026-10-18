# E3x recovery diagnosis — 17 October 2026

Status: completed and accepted by both independent engineering and scientific saved-result reviews. The original failed-A2 recovery result remains unchanged.

The diagnosis finds two concrete limitations: the learned forces do not reproduce stationarity at the exact reference minima, and several force-converged direct/consistency endpoints have no remaining interactions inside the model cutoff. The saved stopping flags themselves reproduce. These observations narrow the investigation but do not establish the historical cause of all failed recoveries.

## Fixed diagnostic and provenance

The same five accepted BEST checkpoints each evaluated ten original starts, ten saved endpoints and eleven reference minima: **155 molecular force evaluations in 40 batched supplier calls**. No training, new relaxation, alternative checkpoint/strength or test-data access occurred. All ten start predictions per model reproduce the accepted validation scores within the fixed 1e-10 tolerances. The fifty recomputed endpoint force flags agree with their saved convergence facts. Original A2 remains failed; all original categories and recovery counts remain unchanged.

The source/locks, five helpers, evaluator and all 317 historical result files remain byte-identical. The six supervised diagnostic stages used 41.758033 seconds locally; coordinator audit used 1.064377 seconds. These exclude unmeasured administrative/reviewer overhead and are not deployment or GPU-cost measurements.

## Exact-reference stationarity mismatch

All eleven archived reference minima have maximum absolute force components between **0.535 and 2.745 meV/Å**, below the operational **10 meV/Å** stopping threshold. Every model predicts a larger-than-threshold maximum component at every reference, including all six distinct own references used by the ten validation branches. The reference labels were inspected rather than assumed to be exactly zero.

| Frozen model | Predicted maximum component across eleven reference minima, meV/Å | References above 10 meV/Å |
|---|---:|---:|
| Energy C, seed 0 |43.28–66.73|11/11|
| Direct, seed 0 |41.43–70.90|11/11|
| Consistency λ=0.1, seed 0 |56.60–141.03|11/11|
| Direct, seed 1 |30.19–59.26|11/11|
| Consistency λ=0.1, seed 1 |41.35–80.15|11/11|

A passing average force-error gate does not guarantee correct local stationary points. Conversely, these measurements at the exact reference coordinates do **not** exclude a stationary point within the existing 0.05 Å geometric tolerance. The consistency model is not uniformly better on this diagnostic: seed 0 has a larger reference-force range than its ordinary-direct partner. Two seeds and eleven shared structures support descriptive observations, not significance or a universal ranking.

## Force-converged endpoints outside the interaction graph

All fifty saved endpoints remain own-reference geometry nonmatches when inspected diagnostically, including the twenty-five originally nonconverged endpoints. There are no hidden own matches rescued merely by ignoring the force gate. Diagnostic geometry does not amend their official categories.

After exact-zero outputs appeared in the saved force records, an **exploratory post-hoc reduction** inspected all fifty endpoints at each model's already frozen 5 Å interaction cutoff. This radius is an architectural parameter, distinct from the covalent-radius chemical-graph criterion. It is not a new tuned matching tolerance or a preregistered comparative metric.

| Model | Force-converged endpoints | Disconnected at 5 Å among those | Exactly zero force with no interaction pairs |
|---|---:|---:|---:|
| Energy C, seed 0 |2|0|0|
| Direct, seed 0 |4|4|2|
| Consistency λ=0.1, seed 0 |6|6|4|
| Direct, seed 1 |4|4|0|
| Consistency λ=0.1, seed 1 |9|9|1|

All 23 force-converged dual-head endpoints split into 8–21 interaction components. Seven have no atom pair within 5 Å: all 21 atoms are isolated, minimum pair distance is 5.013–8.935 Å, and every predicted force component is exactly zero. The frozen smooth-cutoff message basis and bias-free polar force head are consistent with zero vector features when every interaction vanishes. These are geometrically wrong flat-force endpoints, which the existing geometry gate correctly rejects as recovered references. This observation does not demonstrate physical molecular dissociation or reveal the sequence of events that produced the endpoints. A separate static implication is that the auxiliary scalar energy head can become position-independent in this strict all-isolated regime, so its gradient and the direct vector head can both vanish and satisfy the consistency residual trivially. This phase did not measure auxiliary gradients or residuals at those endpoints; this is an architectural implication, not an additional measured result or evidence that the penalty caused separation. Finite-cutoff energy models can also have flat separated-atom regimes; the two observed energy-converged endpoints stayed connected in this run.

## Physical-force and stopping checks

Raw net force and centroid torque are retained without projection. The energy-derived arm has net force around 1e-15 eV/Å and torque around 1e-14 eV over the sampled structures, consistent with its invariant-energy gradient construction. Direct/consistency arms have nonzero values on the common starts and references. Endpoint extrema are separately conditioned on model and official stopping category: their geometries differ, and very large nonconverged outliers must not be presented as a controlled arm comparison. Net force/torque observations do not by themselves prove conservativity or its absence.

The maximum-component stopping rule is Cartesian-frame dependent; RMS force component and vector norms have different meanings. The same stored frame and unchanged threshold are used throughout. No discrepancy was found in the fifty recorded stopping flags. This checks endpoint accounting, not whether the relaxation procedure is optimal or physically reliable along an unobserved path.

The relaxer computed steps, force histories and final maximum force, but the runner did not retain those fields. Historical trajectories cannot be reconstructed from endpoints/counts. Also, the current growth cap bounds the scale in displacement=scale×force; it is not a force-independent 1 Å displacement limit. Both limitations remain documented without changing the source or original results.

## Next decision

A separately reviewed traced replay of a fixed failing branch can locate when its interaction graph separates and relate that event to force magnitudes, accepted step scales and displacement. The replay would be diagnostic, with original checkpoint/start/settings retained, and would not replace the accepted negative result. No replay or corrective solver/training intervention is launched here.

Evidence packet: reports/execution/e3x-recovery-diagnosis-2026-10-17/. Raw records: runs/aspirin-e3x-recovery-diagnosis-2026-10-17/. Main test remains closed; midterm writing remains scheduled for October 20–23.

Accepted evidence: acceptance (retained audit record), engineering review (retained audit record), scientific review (retained audit record).
