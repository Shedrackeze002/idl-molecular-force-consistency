# Offset profiling: paper findings

Saved-data findings from the once-only 6,000-update trial. Primary closeout verified; independent saved-result review pending. These findings are research observations, with the limitations below retained for the paper.

The intervention reduced pooled endpoint error relative to the historical width 32 run, but it did not pass the predeclared joint learning gate. The result supports continuing the force-fitting investigation; it does not establish convergence or held-out performance.

## Trial and comparison

`runs/aspirin-offset-profile-2026-10-02` restored the historical width 32 update-zero state without random parameter initialization. CPU float64; width 32 (13,569 parameters); degree 1; two message-passing iterations; eight radial features; 5 Å cutoff; learning rate 0.0003; energy weight 0.014584365670390719; force weight 1.9854156343296092; microbatch 16; seed 0. The 128 fitting structures span eight groups, 16 per group. All 16 anchors are fitted structures from group 01. They are not a held-out set.

One global additive energy bias was calibrated from fitting energies before training and after each accepted update. Its Adam moments were zeroed, and only its gradient was masked after a numerical zero check. Anchor scores use that same fitting-derived bias; there is no anchor or group-specific recalibration. This changes the optimization procedure relative to the historical reference.

Execution completed 6,000 accepted updates and 6,001 accepted calibrations. Numerical elapsed time was 1,899.5848710830032 seconds, including charged finalization. No retry, extra parameter initialization, unknown work or in-flight numerical work remained. There are 62 shared postzero score observations; intervention-only update 3001 and projected update 0 are excluded from the matched historical grid.

## Raw endpoint results

MSE units: energy eV²; forces eV²/Å². Relative change is (profiled−historical)/historical. Fractions use the original unprojected zero-state MSE, so the baseline is shared and fixed.

| Role | Metric | Historical update 6000 | Profiled update 6000 | Change (%) | Historical baseline fraction | Profiled baseline fraction |
|---|---|---:|---:|---:|---:|---:|
| fitting | Energy | 0.0119832084234968 | 0.0108800780562791 | -9.2056344865 | 0.319409433487 | 0.290005768525 |
| fitting | Forces | 0.00655896894247982 | 0.00635194479621658 | -3.1563519827 | 0.465705272479 | 0.451005974878 |
| anchor | Energy | 0.00711493983993826 | 0.0069990806165559 | -1.6283935773 | 0.11607064219 | 0.114180555308 |
| anchor | Forces | 0.00453086764587091 | 0.00429349480213828 | -5.2390151796 | 0.447115047671 | 0.423690622454 |

The strict gate requires both energy and force endpoint fractions below 0.1 for each role. All four profiled fractions remain above 0.1: both role gates and the overall joint gate fail. Execution validity and engineering verification do not turn this into a successful learning-gate result.

## What offset removal explains, and what it leaves unresolved

At the initial state, fitting raw energy MSE fell 11.54503000616968% solely from the analytical projection; force MSE was unchanged. That immediate reduction is by construction and must not be presented as learning.

On the 62 shared postzero observations, the fitting raw energy maximum fell from 0.13513795338588072 to 0.033201892516970574 eV² (75.43111192296662% lower), and the anchor maximum fell from 0.1348453640671307 to 0.036222284060598034 eV² (73.13790925540063% lower). These are sampled maxima. Removing a mean offset directly reduces fitting raw MSE, and dense intervention corrections have no matched dense historical counterpart. The observations do not establish continuous stability or a spike mechanism.

At update 6000, the historical fitting mean-error square was already only 4.4911314561212563e-08 eV². Fitting centered variance fell from 0.011983163512182258 to 0.010880078056279094 eV² (9.205294201166092% lower). The new fitting mean-error square was 9.464039006018321e-29 eV². The endpoint gain is therefore mostly a difference in learned centered errors, rather than removing a large remaining historical endpoint offset. This is a descriptive comparison, not a causal decomposition of the training dynamics.

The anchor result is mixed. Raw energy MSE fell 1.628393577300702%, while centered energy variance rose from 0.004883396577606338 to 0.006075907937692252 eV² (24.41971159078873% higher). Its mean-error square fell from 0.002231543262331928 to 0.0009231726788636489 eV². Keeping only the raw score would hide this deterioration in centered errors.

Centered fitting fluctuations remain: fitting energy fractions are 0.420133179867439 at update 3000 and 0.30901383668434784 at update 3001, despite fitting mean errors about 3.3029e-15 and 6.9666e-15 eV. Force fractions are 0.5733154659398794 and 0.5581048241028662. Mean-zero enforcement does not imply a smooth error trajectory. Four previously saved spike states (3600,3700,3800,6000) have explicit component comparisons in `offset-profile-component-comparison.json`; they are a separate saved-state comparison, not additional matching-grid scores.

## Group differences retained

At the endpoint, fitting raw energy MSE improved in four groups and worsened in four; centered within-group variance improved in three and worsened in five; force MSE improved in seven and worsened in one. Relative changes below use each group’s historical endpoint.

| Group (16 structures each) | Raw energy change (%) | Centered variance change (%) | Force change (%) |
|---|---:|---:|---:|
| 01 | -1.62839358 | +24.41971159 | -5.23901518 |
| 01_2 | +27.08897142 | +50.87257417 | -0.30081059 |
| 22 | -67.15026733 | +13.11726788 | -9.87484450 |
| 23 | -65.45416861 | -9.13896340 | -4.03762508 |
| 44 | +119.28131255 | +14.30232714 | -9.80357995 |
| 58_2 | +4.62883701 | -25.28687050 | -2.47852666 |
| 77 | -8.30056736 | -22.80795140 | -8.10461767 |
| 78_2 | +87.76375438 | +59.03828977 | +8.14690088 |

In particular, group 44 raw energy MSE rose 119.28131254959813%, and group 78_2 force MSE rose 8.146900877762131%. There is no universal per-group gain. Pooled centered variance is the sum of the weighted within-group variance and variance of group means. The historical values are 0.008643143536930417+0.0033400199752518416 eV²; the profiled values are 0.007917220641529289+0.0029628574147498063 eV². Both pooled terms decreased, despite five groups having worse centered variance.

## Verification and limits

Both full centered-objective/independent NumPy Adam oracles passed at updates 1 and 3001. The latter maximum state discrepancy is 1.3664092080034607e-10, evaluated with combined per-leaf absolute tolerance 1e-10 plus relative tolerance 1e-8. It exceeds the absolute term alone; do not describe every state difference as below 1e-10. The once-only fresh-process prediction audit had maximum absolute discrepancy 0.0 over 144 role entries. The unchanged reviewed implementation had 656 tests passing before launch; this is engineering evidence, not additional model training.

Saved-only closeout regenerated the canonical JSON and Markdown byte-for-byte, checked all 18,324 raw files against 18,324 unique exact archive members, checked each member hash and archive CRC, and verified raw preservation. Endpoint arithmetic was independently recomputed from saved prediction arrays and unchanged targets, including group membership and variance identities. No model calls, new predictions or training were performed for closeout. The figure was visually inspected. Matplotlib emitted host font-cache warnings; the figure was created successfully, and that warning remains in the closeout output.

One seed, one rate, one molecule, a historical reference, changed optimizer dynamics, fitted anchors from a single group, and sparse matched observations limit inference. There is no held-out/generalization result, causal spike-mechanism proof, convergence proof or full training pilot. The next candidate lever concerns unresolved force fitting; it requires its own saved plan and two independent plan reviews before implementation.

## Evidence identities

Paper evidence entries: `OFFSET-TRIAL-COMPLETION-001`, `OFFSET-SAVED-CLOSEOUT-001` (plus subsequent review/final entries). Repository data are the numerical authority.

- `reports/offset-profile-diagnostics.json` — SHA256 `33f5b57fc74b363de07026b6401bf79f029b24fbb46bcc356bac4256bf875f26`.
- `reports/offset-profile-component-comparison.json` — SHA256 `5baa1ab4f75d126789ccd05d95101876ab5b787a1c8ee7d3a03cb6b3924b3276`.
- `reports/execution/offset-profile-2026-10-02/independent-arithmetic.json` — SHA256 `42785d90a90f78c595559c6827cd8387f600db4cbd4c13721625462fc986f467`.
- `reports/execution/offset-profile-2026-10-02/saved-closeout-verification.json` — SHA256 `3695e32696c150c2adffd90a8c72c1d5cbf16939a5e0ccffd0a1b8c1ea2c8336`.
- `reports/execution/offset-profile-2026-10-02/numerical-run.zip` — SHA256 `91375bfe0e628d6cb374e186cd64e01b23a8f1ade837df057e5e1234ab42c7fa`.
- `reports/execution/offset-profile-2026-10-02/matched-raw-mse.png` — SHA256 `424c9df45d105f40a323c9ff8c7b2d8997e3b0e5d6c8c7866bb31f3ba95f2bb0`.
