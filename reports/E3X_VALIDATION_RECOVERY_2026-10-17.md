# E3x validation recovery, 17 October 2026

Status: accepted negative validation result after independent engineering and scientific saved-result reviews. The recovery gate failed; no comparative recovery benefit is established.

The frozen energy-baseline recovery gate failed: zero of ten branches recovered their classified own reference, compared with zero own-reference matches in the measured no-relaxation control. Both ordinary-direct seeds and both matched consistency λ=0.1 seeds also recovered zero of ten branches each. The comparison therefore remains descriptive; it does not establish a recovery advantage or refute the general research hypothesis.

| Accepted checkpoint | Correct-reference recovery | Force-converged, geometry unmatched | Nonconverged | Force attempts |
|---|---:|---:|---:|---:|
| Energy C, seed 0 | 0/10 | 2 | 8 | 13,883 |
| Direct, seed 0 | 0/10 | 4 | 6 | 12,226 |
| Consistency λ=0.1, seed 0 | 0/10 | 6 | 4 | 11,375 |
| Direct, seed 1 | 0/10 | 4 | 6 | 13,321 |
| Consistency λ=0.1, seed 1 | 0/10 | 9 | 1 | 8,197 |

Every pass uses the same ten first-frame starts from five validation connections. The branch force-evaluation cap is 1,500, force convergence is maximum absolute component ≤0.01 eV/Å, and own-reference geometry requires the frozen symmetry-aware 0.05 Å rule. There were 59,002 force attempts across 50 model–start evaluations, below the fixed 75,000 maximum. Twenty-four nonconverged branches exhausted their budget; one consistency seed-1 branch stopped for no progress. No operational failures, retries, alternative checkpoints, changed tolerances or extra starts occurred.

The measured no-relaxation control made zero model calls and zero steps: own-geometry count0/10; nearest-own-reference class context 5/10. Nearest assignment is context, not recovered geometry or an additional gate. All five models replayed the saved 974 validation predictions in batches and five representative single-molecule calls within 1e-10 tolerances, with maximum absolute differences 4.26e-14–5.69e-14 eV/Å. Adapter fidelity does not imply successful relaxation.

The 25 force-converged endpoints are all geometry-unmatched. An independent audit using only ten selected validation starts and eleven reference minima exactly reproduces the control, all 110 nearest scores and all 50 endpoint categories. The frozen matcher excludes 24 converged own-reference candidates through its descriptor bound and one through a complete alignment (RMSD 0.109590853 Å > 0.05 Å). Optional explicit own alignment finds graph nonisomorphism for 23 descriptor-excluded dual-head endpoints. These are graph-mismatch facts, not complete RMSDs or proof of chemical bond breaking. The reviewer retained two failed auxiliary audit assumptions and their corrections; no stored category or result changed. More declared force convergence under the consistency model is not evidence of correct shape recovery or an energy minimum. All ten own-reference minima occur in training metadata; this study holds out transition connections toward seen minima. There are five independent validation connections, two seeds per matched arm, and one energy-baseline seed. The five paired connection differences and both mean arm recovery rates are all zero. Selection used validation force error, so the unchanged matching and small validation set limit generalization claims.

Numerical training source, locks, accepted checkpoints/scores and 181 controlled output files are unchanged. Main-test data stayed closed. Summed supervised stage wall time is 129.625468 s on the local CPU, including control, replays and passes; this is neither complete study cost nor a deployment/GPU benchmark. Initial implementation failures, the review-discovered refusal defect, regression evidence and all raw branches remain preserved.

Next: diagnose saved endpoint geometry and stopping behavior before proposing another experiment. Keep this negative result fixed; any new settings or data intervention needs a separate reviewed plan.

Evidence: `reports/execution/e3x-validation-recovery-2026-10-17/` contains the adopted plan/reviews, implementation checks, frozen launch records, completion audit and output inventory; `runs/aspirin-e3x-validation-recovery-2026-10-17/` contains raw rows and per-model reports.

Accepted evidence: result acceptance (retained audit record), engineering review (retained audit record), scientific review (retained audit record).
