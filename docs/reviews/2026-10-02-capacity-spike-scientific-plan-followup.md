# Capacity spike scientific plan followup, 2 October 2026

Status: **PASS** for Amendment A. Both material scientific specification findings from the initial review are resolved. This authorizes implementation under the reviewed bounded diagnostic; it is not numerical execution verification or a scientific result.

Exact reviewed design SHA-256: `b674a96d27f3867c4f40b5ae85747e292c89e01f2ff9f46263463ddac5afc835`.
Exact reviewed plan SHA-256: `7cac3ef0cbfa0536bdb1067dd5e8ab5f20518404521714ee4cbbf0be5ebe492a`.
The initial NEEDS_REPAIR review remains preserved in `2026-10-02-capacity-spike-scientific-plan.md`.

## Repairs verified

1. The design now fixes prediction-minus-target residuals and the molecular effect of a shared atomic bias. The analytic unweighted scaled-energy derivative is `2 mean(n_i e_i)/s_E²`, with 21 atoms for aspirin; production weighting multiplies by the energy coefficient. The force derivative is structurally zero. Tolerances are fixed, and the amendment's test coverage includes this atomic count. This resolves the possible factor-of-21 error. The analytic check remains local algebra; global offset must not be attributed to the bias parameter alone.
2. The design defines the production-weighted joint gradient and the two unweighted scaled-objective slopes along its negative, then defines weighted-component slopes by multiplication with the corresponding positive coefficients. It explicitly relabels helper outputs at whole/block/leaf levels and excludes raw-MSE and Adam-direction interpretations. A nonunit, unequal-weight test with opposing gradients is required. Parameterization dependence and invariant cosine under positive weighting are also stated. These definitions resolve the initial scale/direction ambiguity.

The instrumented-kernel amendment preserves the scientific objectives while making attempted/completed work explicit. Independent unweighted and weighted joint kernels are separate from the split sum, and all microbatches see the same saved parameters. The FD rule retains both fixed steps and the existing zero-gradient fallback without optimizing parameters. Planned work reconciles as 8 states × 9 prediction batches = 72 calls/1152 role entries; 8 × 4 gradient objectives × 8 batches = 256 calls/4096 exposures; and 8 × 2 FD directions × 2 steps × 2 sides × 8 batches = 512 calls/8192 exposures. FD vector calls evaluate both loss components without doubling structure exposures. These are invocation records, not training or primitive-forward counts.

The total cap is now unambiguously 1800 seconds including a 60-second reserve. Completed valid earlier cases can be described individually after a later failure, but complete-run contrasts require all eight cases and final source/cap checks to pass. This does not relax the no-retry rule or scientific status gate.

## Limitations retained for implementation/result review

- Finite differences establish agreement along the selected direction only, not exhaustive correctness of all derivative components. Preserve direction kind, both steps and every failed step.
- A positive raw-MSE increase may combine a negative component change with a larger positive one. Keep increment shares signed and unclipped; no share exists for nonpositive total increase. Pooled variance includes between-group means and is not generally the mean of group variances.
- Selected maxima and neighbors are post hoc and sampled; no continuous trajectory, excursion duration, causal optimizer mechanism, convergence, generalization or isolated expressivity claim follows.
- Predictions are all fitted, gradients use the 128 fitting structures only, and the 16 anchors duplicate fitted structures. Original raw learning gates remain unchanged.

This followup read only the amended documents and verified their file hashes. No model/gradient/prediction work, raw-file changes or implementation occurred. No additional inspection failures occurred.

## Final engineering clarification hash check

Status: **PASS**, unchanged scientific approval. Final design SHA-256 is `d9c5b564ba0771c17f7c94cbf41a7d553e8838ec39b55c657cb932a607feb96b`; final plan SHA-256 is `eb4bf915c5ca7a1efc1a7b90246a40ef8b59bd98afbf4a695cbc073ea7377e1e`. Both were read and hashes verified after the clarification was appended.

The clarification starts the numerical cap before guards/imports/compilation, distinguishes ready returned work from finite-validated work and unknown in-flight work, audits all decoded state fields on failure paths, and invalidates every endorsement if global source integrity fails. These strengthen the existing controls without changing selected states, derivative/metric criteria or inferential scope. The FD fallback seed is now fixed at `20261002` and zero threshold at `1e-12`; this removes an unspecified setting without adapting it to any result. No numerical model work or raw edits occurred.
