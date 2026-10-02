# Offset-profile scientific plan review

Decision: **PASS** for the saved plan and design identified below. No blocking scientific repair is required. This is a plan review, not verification of a future implementation or endorsement of unexecuted model results. The second independent plan review, implementation review and prelaunch verification remain required by AGENTS.md.

## Reviewed identities and scope

- Current source HEAD: `5e1c81f5322ecc7e92cfcc3238d285ffc1949387`.
- Design: `docs/superpowers/specs/2026-10-02-offset-profile-design.md`; SHA-256 `ba4b422791426592be29b765fd85345781c9a01a77d08af647f0d06240f5b6cd`.
- Plan: `docs/superpowers/plans/2026-10-02-offset-profile.md`; SHA-256 `91ab94fb05d19321908b621e3d11431ae259809bc65e8ce6334d3d72c3456a0d`.
- Read AGENTS.md, current paper evidence closeout, capacity/spike paper findings and JSON reports, source width32 provenance, capacity configuration, existing loss/update/model/normalization kernels, and current Project Hub/Research Findings navigation. Repository artifacts were used as numerical authority.
- Pure standard-library hashing found zero missing/changed entries in the 347-entry spike source map (343 capacity raw files) and the separate 20-entry ancestral capacity source map. This does not replace the proposed new launch/closure guards or a complete production inventory.
- No numerical imports, model calls, training, tests, dependency work, or raw/source modifications occurred. Only this review record was created.

## Scientific assessment

### Global projection and envelope derivative

The frozen model applies a single scalar atomic head bias before summing atomic energies. For all 21-atom structures, write `E_i(theta,b)=E_body_i(theta)+21b`. With equally weighted fitting records, the proposed correction `delta_b=-mean(e)/21` is the exact minimizer of the energy squared-error component for fixed nonbias parameters. Positive fixed energy weight and fixed normalization scale do not change that optimum. The resulting objective is `w_E Var_fit(e)/s_E^2 + w_F MSE_fit(F_error)/s_F^2`.

At the global optimum, `mean_fit(e)=0`; therefore the derivative of the original raw weighted loss with respect to each nonbias parameter equals the derivative of the globally centered objective. The chain term involving `db*/dtheta` vanishes because the raw objective's derivative with respect to bias is zero. Forces are unaffected by this coordinate-independent bias. This justification requires the complete fitting support at the same incoming parameters. It would fail if each microbatch calibrated or centered independently, used unequal chunk means without record-count weights, changed atom counts, or selected different fitting labels.

The design correctly requires one initial and one outgoing full-support projection, incoming projected gradients, unmasked aggregate bias-gradient tolerance, then exact bias masking and zero bias moments. It preserves original Adam on all other leaves. An independent full-support centered-objective derivative and unequal-chunk/different-mean engineering fixtures explicitly cover the key global-versus-local error. No differentiation through calibration is needed for the proposed envelope update.

### Controls, gate and initial-state difference

The historical reference is clearly labeled and immutable. Source width32 has 13569 parameters and the same model configuration, rate, weights and ordered fitted supports. Its initial parameter/optimizer hashes are respectively `c63c26e38b4c7c65ae0f4658ce3cb31052cede8f940ec394cb40bbf7cf4ed4f4` and `f0a71cbf26a1dec088cf8c5a5338eb72413ab1868544a4c6a0ff5cbdaf210184`. Restoring this exact state without random initialization is a suitable paired-start control.

The intervention update0 is projected and thus need not be the historical update0. The design explicitly retains the shared unprojected state/replay and separate projection receipt, and keeps raw gates anchored to the original unprojected baseline. Verified baseline MSE values are:

| Role | Energy eV² | Force eV²/Å² |
|---|---:|---:|
| Fitting | 0.03751676427546992 | 0.014083948218070219 |
| Fitted anchors | 0.061298358531283896 | 0.010133561081129106 |

The corresponding strict 10% thresholds are 0.003751676427546992/0.001408394821807022 for fitting and 0.00612983585312839/0.0010133561081129107 for anchors. The reference endpoint fitting fractions match the plan: 0.3194094334870974/0.4657052724792342. Bias fitting uses all 128 fitting structures once, including the 16-anchor subset; separately calibrating anchors would invalidate this comparison.

The initial projection and ongoing bias constraint are components of the intervention, not learned performance. Fitting mean suppression and its raw-MSE contribution are enforced by construction. Centered energy variance and force-error changes are empirical outcomes. The prior endpoint offsets are tiny; this trial cannot claim that a constant correction alone repairs the historical endpoint failure. The design preserves that distinction.

### Observation grid, verification and evidence

The source report contains exactly the proposed 63 score updates: 0, 1, each hundred through 6000, and 3001. The 62 matching postzero observations give a fair sampled-grid trajectory/maxima comparison despite different update0 construction. Dense projection receipts have no dense historical counterpart; their maxima must remain separately labeled. One seed, historical reference, changed update rule and fitted-only supports limit generalization, causal attribution and continuous-stability claims, as stated in the design.

The production oracles at outgoing1/3001 are well chosen: the second uses actual incoming update3000 moments, rather than a zero-moment surrogate. Full-support centered differentiation and separate NumPy bias-corrected Adam avoid sharing the production accumulation/Optax update path. Independent reference calibration must remain separate from the production projection helper. Retaining actual/reference gradients, predictions, optimizer comparisons and receipts gives a meaningful saved-only audit; a scalar maximum-difference receipt alone would not satisfy this plan.

The once-only token-bound child checks endpoint restoration and all 144 role prediction entries without training or random initialization. The anchors overlap fitting, so these are 128 unique structures, not 144 independent samples. Predicted postprojection statistics are separately labeled from fresh measured scores, appropriately limiting what algebraic receipts establish between scored states.

The 3600-second total cap, 180-second reserve, charged imports/oracles/scoring/child/closure, external timeout and no-retry rule are scientifically acceptable fixed operational constraints. Whether the implementation can finish within them is unmeasured. Incomplete, nonfinite, late, child-failed and identity-drift attempts cannot count as valid execution or learning success; the plan preserves known-returned and unknown-in-flight work and prohibits numerical retries or favorable checkpoint selection. Archive/review/report regeneration is saved-only and separately timed.

## Actionable implementation acceptance checks

These implement existing reviewed requirements; they do not amend the scientific design:

1. Weight each microbatch contribution by its record count divided by 128, and evaluate all gradients at one incoming projected state. Fail the unequal-chunk/global-centering regression if either rule is broken.
2. Compare both the unmasked aggregate bias derivative and complete nonbias gradient against the independent profiled objective. Retain both vectors/trees, incoming identity, reference Adam state, independent calibration output, and actual outgoing predictions at each oracle.
3. Publish shared-zero identity/replay separately from projected update0, and derive every raw gate from the four original baseline values above. Include anchor original-baseline tests.
4. Keep projection-imposed mean reduction distinct from centered-energy/force fitting changes; use the 62 matching postzero grid for reference maxima and preserve dense correction/preprojection summaries separately.
5. Treat missing oracle/child/completion evidence, unknown in-flight work, cap overrun or source/gate drift as invalid execution while retaining partial observations and stop reasons. Verify the production receipts implement this before launch.

## Preserved inspection errors and limits

- An exploratory shell glob `scripts/*late*` had no matches, so that batched search did not execute; subsequent explicit source-file reads replaced it.
- Guessed paths `scripts/train_capacity_control.py`, `src/force_consistency/model.py`, and `reports/execution/spike-diagnostic-2026-10-02/obsidian-sync` were absent. File inventory located the actual capacity controller and `models/energy.py`; no Obsidian-sync directory was needed for this plan review.
- `runs/aspirin-capacity-2026-10-02/width32/commits/0.json` was an absent guessed receipt path; this is not evidence that source update0 is missing. The raw source-map check found no absent members, and the reviewed design requires restoration from the actual saved snapshot identity.
- Broad JSON/history and Project Hub reads exceeded output limits. Later selective standard-library JSON extraction independently recovered current source grid, initial hashes, baselines, endpoint fractions and statuses; targeted navigation lines recovered the completed diagnostic and unexecuted offset candidate. No missing output was treated as reviewed evidence.
- This review did not inspect future code, measure runtime, decode optimizer arrays, or repeat diagnostic derivatives. Those remain implementation/prelaunch/result verification obligations.

PASS applies only to the two document hashes above. Material revisions require follow-up review before implementation.
