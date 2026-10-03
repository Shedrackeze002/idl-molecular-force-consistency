# Offset-profile independent saved-result review — 3 October 2026

Verdict: **PASS for corrected paper draft SHA256 `0ede62384b0ac7851989c98aabb4240b9926d9def7632c0e5f7dfd2d731ff8c7` and the numerical artifacts identified below.** One factual matched-grid claim required correction and is resolved. No remaining material numerical, provenance, or paper-claim issue was identified within this saved-only scope. This is a verified bounded diagnostic with failed learning gates; it is not convergence, held-out generalization, causal mechanism evidence, or a full pilot.

## Scope and independent evidence

Reviewed the current AGENTS.md, append-only paper evidence log, fixed configuration, scientific/engineering final plan reviews, implementation resolution/followup, frozen prelaunch gate, launch/invocation/process records, completed once-only raw run, canonical reports, four historical spike-state comparisons, saved closeout record and independent primary arithmetic. HEAD is `ca31c8b4bc67dc7ec067dd514dca3c80c713c75e` on the existing checkout. The frozen gate binds the two final plan PASS reviews, implementation repair PASS and completed final suite. Its participant bytes remain unchanged; retained `fullsuite-v2.txt` ends in **656 passed in 488.59s (0:08:08)**. This review reused that evidence and did not rerun tests.

Reviewer evidence is preserved at `reports/reviews/offset-profile-saved-results-2026-10-03/`: `audit.py`, `audit-results.json`, `audit-output.txt`, reconstructed canonical JSON/Markdown under `regenerated/`, `supplemental-results.json`, `grid-correction-followup.json`, exact `reviewed-paper-findings.md`, and `progress.md`. The independent NumPy arithmetic path directly reads prediction/target arrays; the separate report reconstruction applies the saved-state/receipt audit. It makes no training, model, initializer, fresh-child or new prediction calls. Only reviewer records were written. Frozen source/configuration/tests/gate and all production/historical raw bytes were preserved; user changes to `proposal.pdf` and `Group9.pdf` were untouched.

## Required correction resolved

The first spacing-cleaned draft, SHA256 `9de7272b493048adb50ab8e86e7fb28097afd0da29e8fbafba5c48277e7e1d66`, incorrectly called update 3001 intervention-only and excluded from matching. Both raw histories contain the same 63 scored updates: 0, 1, every hundred through 6000, and 3001. Their postzero intersection therefore has 62 observations **including 3001**. Canonical `matching_rows` and the plotted grid were already correct. The corrected draft says this explicitly and excludes projected update 0 from the postzero comparison. Independent text comparison confirms this is the only change from the prior draft. The primary retained the prior draft and correction record; my initial failed grid assertion is also retained in `audit-first-grid-failure.txt`.

## Independently reproduced findings

| Role | Metric | Historical endpoint | Profiled endpoint | Relative change | Profiled original-zero fraction |
|---|---|---:|---:|---:|---:|
| fitting | Energy MSE, eV² | 0.01198320842349682 | 0.010880078056279096 | -9.205634486459346% | 0.2900057685250047 |
| fitting | Force MSE, eV²/Å² | 0.0065589689424798165 | 0.006351944796216577 | -3.156351982739647% | 0.45100597487761285 |
| fitted anchor | Energy MSE, eV² | 0.007114939839938265 | 0.006999080616555901 | -1.628393577300702% | 0.11418055530775573 |
| fitted anchor | Force MSE, eV²/Å² | 0.004530867645870908 | 0.004293494802138282 | -5.239015179552853% | 0.42369062245390743 |

Every fraction exceeds the strict 0.1 threshold. Both role gates and the joint gate **FAIL**. Fractions use original unprojected source-zero MSE, not projected-zero MSE or RMSE. Ordered role IDs, identical input bytes, unique support and one group membership per structure were independently checked. All 16 anchors belong to fitting group 01; they provide no independent held-out evaluation.

Raw energy equals population centered variance plus squared mean for every checked role/group/state. Fitting centered variance decreases 9.205294201166092% at the endpoint; the old pooled squared mean is already only `4.4911314561212563e-08` eV². Thus the descriptive endpoint difference mostly concerns centered errors, without attributing a causal training mechanism. Anchor centered variance instead increases 24.41971159078873% (`0.004883396577606338` to `0.006075907937692252` eV²), despite its small raw MSE improvement. The draft retains this negative result.

The eight fitting groups each contain 16 structures. Raw energy improves in four and worsens in four; centered variance improves in three and worsens in five; force improves in seven and worsens in one. The unfavorable group44 raw-energy increase of 119.28131254959813% and group78_2 force increase of 8.146900877762131% are correct. Weighted within/between group variance sums reproduce pooled variance for both arms. The pooled terms decline even while five individual groups have worse centered variance; the draft makes no universal per-group claim.

Initial fitting raw MSE falls 11.545030006169677% under projection, with fitting force predictions bitwise identical and centered variance unchanged. This is an algebraic construction effect, not learning. The saved update3000/3001 means, variances and E/F fractions independently reproduce the reported centered-error fluctuation. The 62-point raw maxima and reductions agree with the matched histories. Four saved historical spike-state prediction arrays at 3600, 3700, 3800 and 6000 independently reproduce every pooled and group component in `offset-profile-component-comparison.json`, with ordered support checked. These four-state components are correctly distinguished from the full 62-point raw comparison and dense intervention-only correction receipts.

## Execution, closure and preservation

Saved audits verify all **6000 accepted update links**, **6001 initial/training projection vectors and receipts**, all **63 immutable score/state/history commitments**, fitting-derived bias and zero bias moments, full-field source/projected/accepted identities, unchanged RNG/normalization fields, and original raw gates. The logged attempted/returned/valid call counts agree: 48000 training gradient microbatch calls and 48000 calibration microbatch calls each expose 768000 frames, with 6000 accepted Adam proposals. Warmup/reference work is retained separately and does not increment accepted updates. Initial calibration exposes 128 frames. No unknown, unaccepted or late work remains.

Both production oracles pass direct saved-tensor per-leaf checks using `atol=1e-10 + rtol=1e-8`. At update1, gradient/state/prediction maximum errors are `1.0547118733938987e-15`, `1.1449174941446927e-16`, and `4.163336342344337e-17`. At update3001, they are `2.5135449277513544e-13`, `1.3664092080034607e-10`, and `4.551914400963142e-14`. The latter state error exceeds the absolute term alone and passes the combined rule, exactly as the draft says. Its actual incoming update3000 body moments are nonzero and incoming/candidate/reference projection identities are bound to accepted states.

The once-only child attempt/entry/completion/process and endpoint binding pass saved verification. Child maximum error is 0.0 over 144 role entries, with Adam count6000 and no training or random initialization. I additionally compared preserved returned final fitting/anchor chunks to endpoint arrays bitwise. I did not launch another child or infer independent behavioral proof beyond its saved receipts and artifacts.

Numerical elapsed time is **1899.5848710830032 seconds**, below3600 and including the recorded final durable-write boundary. The production supervisor returns exit0 with hashes matching its stdout/stderr; numerical and supervisor elapsed scopes are distinguished. Launch revision, runtime/dependency/code fingerprint, configuration, original invocation and gate agree. The completed suite applies to unchanged reviewed source bytes; its engineering success does not change the failed learning result.

Independent exact inventories check **343 historical capacity files**, **247 historical spike files**, all **601 pinned source paths**, all frozen gate participants/live code, and the protected **45677796-byte evidence-log prefix**. The new raw inventory matches **18324 files** before and after audit. ZIP membership is exact and unique at **18324**, every member SHA256 agrees with raw bytes, and every CRC passes. Canonical JSON and Markdown reconstruction is byte-for-byte identical. The retained figure was visually inspected: four readable log-MSE-fraction panels, roles/metrics/axes/legend and threshold are clear; its caption states fitted supports and sparse-line limitations. The primary font-cache warnings are retained and are not a figure failure.

## Claim limits and preserved attempts

The corrected draft retains initial projection by construction, failed gates, anchor-centered deterioration, group heterogeneity, one seed/rate/molecule, historical reference with changed optimizer dynamics, fitted group01 anchors, sparse observations and lack of matched dense historical corrections. It does not claim a causal spike mechanism, continuous stability, convergence, held-out generalization or a completed full training pilot. Continuing unresolved force-fitting investigation is stated as future work with its own plan/two-review requirement, not a demonstrated result.

Preserved reviewer reconnaissance failures: nonexistent `.md` implementation-resolution lookup (actual `.json`), nonexistent historical `predictions/6000.npz` lookup (actual `reference.npz`), excessive metadata output truncation, and a correction-text assertion that expected an extra article before “projected update 0.” These were read-only lookups or saved-array/text checks. The latter was resolved after exact diff inspection showed only the requested scientifically correct sentence. No failed numerical experiment, production retry, suite rerun or raw mutation occurred.

## Exact reviewed artifact identities

| Artifact | SHA256 |
|---|---|
| corrected `reports/offset-profile-paper-findings.md` | `0ede62384b0ac7851989c98aabb4240b9926d9def7632c0e5f7dfd2d731ff8c7` |
| `reports/offset-profile-diagnostics.json` | `33f5b57fc74b363de07026b6401bf79f029b24fbb46bcc356bac4256bf875f26` |
| `reports/offset-profile-diagnostics.md` | `85e6307989f02557d628e1fb440262e0d618b2eefa83c47072cee31708d7e9f1` |
| `reports/offset-profile-component-comparison.json` | `5baa1ab4f75d126789ccd05d95101876ab5b787a1c8ee7d3a03cb6b3924b3276` |
| execution `independent-arithmetic.json` | `42785d90a90f78c595559c6827cd8387f600db4cbd4c13721625462fc986f467` |
| execution `saved-closeout-verification.json` | `3695e32696c150c2adffd90a8c72c1d5cbf16939a5e0ccffd0a1b8c1ea2c8336` |
| execution `numerical-run-inventory.json` | `52156beb72da98b0c1da1801efa1b57ffeaaed21d36259f170a426cd11e97ea8` |
| execution `numerical-run.zip` | `91375bfe0e628d6cb374e186cd64e01b23a8f1ade837df057e5e1234ab42c7fa` |
| execution `prelaunch-gate.json` | `4991f948ba1e19ec26e157bab57d9920e232b43c64d3ca9ac0b23d3ea0af23e4` |
| execution `source-pins.json` | `c527a7b19c4ebe4aa681add2e2a649306a4d3f44a9e029eacb5a7b9b3d876882` |
| run `summary.json` | `18fb37a18e5f421631ce77fcec4ca8de5c381c38765f87afdad4661f9ddf9bfd` |
| run `predictions/6000.npz` | `519baa224165c0925b5bee4a4d0be7455790f8e5a90f801acce3697b4d801f66` |
| execution `matched-raw-mse.png` | `424c9df45d105f40a323c9ff8c7b2d8997e3b0e5d6c8c7866bb31f3ba95f2bb0` |

Reviewer result files contain the numerical detail and prior audited draft identity. This verdict is hash-bound; a subsequent paper status/link-only edit can be checked separately without rerunning any numerical work. Primary final evidence-log/Obsidian status synchronization remains a closeout task; this review does not certify updates that have not yet been made.
