# Capacity spike scientific plan review, 2 October 2026

Status: **NEEDS_REPAIR**. Two specification gaps should be resolved and independently re-reviewed before implementation. The saved-state diagnostic itself is scientifically appropriate within its stated descriptive scope.

This was a read-only review of the design, plan, capacity findings/report, immutable scored histories, model readout, split-gradient/finite-difference utilities, weighted objective and pure checkpoint audits. No model was initialized or evaluated, no gradients or predictions were computed, and no raw files were edited. Python was used only to read saved JSON and select scalar fields.

Reviewed plan SHA-256: `82c89bfe93c6707e673d7f5716eb79ef5f9135f4b94a15c90630b56f06b5947f`.
Reviewed design SHA-256: `7594891d54e6c684352544d537fceb918a1023caecf9008b8ef2f259951b1cfd`.
Capacity canonical report SHA-256: `31fc7af10a6b87dcb584fb73593d279d2cc57b61756d8a388e1927b8296769ec`.

## Required revisions

**P2: Operationally define the head-bias analytic check.** Design line 25 and plan lines 42/45 name a check without its equation, sign convention or atom-count factor. The energy head produces an atomic scalar and sums it over atoms (`models/energy.py:21-22`). A bias increment therefore changes molecular energy by `n_i * delta_b`, not simply `delta_b`. Define signed residual as `e_i = prediction_i - target_i`, use the stored energy loss scale `s_E`, and compare autodiff against `dL_E/db = 2 mean(n_i * e_i) / s_E²`; this reduces to `42 mean(e_i)/s_E²` for the 21-atom aspirin structures. The production-weighted derivative is `w_E * dL_E/db`; force-objective bias derivative is structurally zero because this additive atomic term has no position derivative. Set the check tolerance and record both expected and observed values. The test witness must have more than one atom and nonzero mean residual so omission of the atom factor cannot pass. This check establishes local algebra, not that a saved global offset originated from the bias alone.

**P2: Specify slope objectives and direction separately for weighted geometry.** Design line 25 and plan line 46 ask for weighted/unweighted geometry and negative-joint-gradient slopes, but the existing helper names both outputs `*_raw_joint_descent_slope` and calculates `-(||g_E||² + dot)` and `-(||g_F||² + dot)`. Passing `w_E g_E, w_F g_F` into it yields derivatives of the weighted component objectives along `-(w_E g_E + w_F g_F)`. They are not derivatives of the unweighted scaled objectives along that direction. Freeze the labels/equations: for unweighted scaled objectives along the production-weighted negative joint gradient, report `s_E = -w_E ||g_E||² - w_F <g_E,g_F>` and `s_F = -w_F ||g_F||² - w_E <g_E,g_F>`. If weighted-component slopes are also retained, identify them as `w_E*s_E` and `w_F*s_F`. The plain unweighted-joint direction remains `-(g_E+g_F)`. For each block, describe its value as that block's contribution or a direction confined to the block. Add a nonunit-weight toy check distinguishing these conventions. These slopes describe infinitesimal Euclidean directions; they neither reproduce Adam's moment/preconditioned direction nor predict a finite training update.

## Scientific controls accepted

- The maximum saved fitting energy MSE is at update 3700 for width32 and 5000 for width64. Immediate scored neighbors 3600/3800 and 4900/5100 exist and their next saved energy MSEs are lower. At width32 the pre/peak/next fitting raw energy MSEs are `0.017976363969020193`, `0.13513795338588072`, `0.017303336020938115` eV². At width64 they are `0.009555389496062851`, `4.882506784430785`, `0.00932241419185751` eV². These establish the selection rule and sampled recovery, not excursion duration or a causal event.
- The retrospective maxima, neighboring observations and endpoints are fixed before this diagnostic, and all eight cases remain visible. This is post hoc descriptive sampling, with no unbiased prevalence estimate or aligned cross-width comparison.
- Raw MSE decomposition as mean residual squared plus population centered variance is valid. A global mean residual describes the best common additive correction on the inspected support, but does not identify the parameter source of that error or prove an optimizer mechanism. Endpoints cannot substitute for peak residuals.
- Original raw energy/force learning gates are preserved. Scaled losses and gradient geometry are separate observations, and recentering must never alter acceptance.
- The 16 anchors duplicate fitted structures within the 128 fitting structures; 144 role entries are not held-out evidence. No generalization, convergence, causal Adam instability, isolated capacity mechanism or main force-consistency result follows.
- Independent weighted-objective differentiation is required rather than defining the check by the split sum alone. State/schema/history and endpoint-reference checks make restoration reproducible without a template initializer.
- Both predeclared finite-difference steps are retained, at least one must pass for each objective, and no post hoc tolerance/step adaptation is authorized. Failure stops later cases and preserves observations without endorsement.

## Reporting cautions for implementation review

1. Directional finite differences validate only the inspected direction for each objective. A unit-gradient direction is not an independent random direction and does not prove every parameter/block derivative correct; the zero-norm seeded direction and direction kind must remain explicit. Joint derivative identities and the bias analytic check complement it. Do not describe the combination as an exhaustive gradient oracle.
2. An increase contribution can be negative or exceed the total increase when another component decreases. Report signed squared-mean and variance differences and any shares without clipping to `[0,1]`; no increase share exists for a nonpositive denominator. Per-group decompositions do not average into pooled variance without the between-group mean term. Keep pooled/group/equal-group conventions explicit.
3. Euclidean norms and slopes depend on the parameterization. Use within-width state contrasts descriptively; do not infer a representation bottleneck or rank expressivity from cross-width gradient magnitudes.
4. History/reference agreement and finite-difference checks are verification conditions, not a new learning-success criterion. Preserved incomplete cases must remain status-gated in summaries and figures.

## Inspection trail and limitations

Two read-only path guesses found absent `src/force_consistency/model.py`, `training.py` and later `weighted_runs.py`; the correct files were located by repository search (`models/energy.py`, `train.py`, `loss_balance.py`). One broad saved-history JSON print was truncated because it included memberships/record IDs; a follow-up restricted output to selected scalar MSE fields and confirmed all stated values and maxima. These were inspection-output failures only, with no model work or source mutation. No execution feasibility or derivative accuracy claim was made from this review.

Relevant source hashes: gradient diagnostics `e3324b57a98899eb8038a64cd0ad6874932df31874d1d2de7217960b7c55ccd4`; atomic energy model `e137f319deae85bb6374c0ef9844a785ea1cb60cd0eafd45fafbed7498f5e0ec`; loss balance `0ff64dcbad7a44f93b362d43d82336d500780cb41f893df27e7539b2f616dfbd`.
