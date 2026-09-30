# Independent final implementation and evidence review

Reviewed c1d8d6d..7aa532e on 30 September 2026, against the saved design, plan, review focus and AGENTS.md. No production training, recalibration or scientific computation was repeated. Only reviewer-owned verification artifacts and this review were written.

## Strengths

- Weighted objectives preserve nested force derivatives, hold parameters fixed during frame-weighted accumulation, and apply Adam once. The independent objective tests include uneven chunks and a nonzero-head mixed-derivative witness. Replay dynamically selects the original default builder; normalized_loss is unchanged.
- Complete source/calibration allowlists precede loading; fresh-state and both initial metric guards prevent historical final-state initialization. Completed replay mismatch retains evidence and blocks later arms. Learning failure does not abort the prescribed matrix.
- Saved evidence independently reconciled: all 31 source files, 38 raw calibration files, compact calibration, report artifact and figure hashes; identical input identities and initial states; each arm 2000 updates, 256000 training exposures, 128 discarded warmup exposures and 22 observations. Serialized checkpoint update and Adam counters both equal 2000 for every arm. All 88 replay metric comparisons match exactly.
- Endpoint fractions and acceptance flags independently recomputed from immutable source initial metrics. Balanced joint and force-only force acceptance are both false. The report distinguishes fitted anchors, force-only energy gauge freedom, whole-tree bias contribution and the absence of Adam/capacity/generalization conclusions.

## Issues

### Critical

None found.

### Important

1. **Freeze replay tolerance at preflight.** `src/force_consistency/loss_balance_runs.py:56-60` checks frozen budget/gates but omits `replay_tolerance`; line 150 passes arbitrary configured tolerance directly to verification. Changing it to `{atol: 1e9, rtol: 1e9}` reaches the data loader, and a saved-history fixture with a deliberate +1 force-MSE discrepancy passes replay with that configuration while the prescribed tolerance rejects it. This defeats the mandatory replay identity gate before candidate training for an altered input configuration. Reject missing, malformed or changed tolerances before data loading (exact prescribed `atol=1e-10`, `rtol=1e-8`), and add a regression proving downstream execution remains blocked. Witness retained in `.superpowers/sdd/2026-09-30-loss-balance-diagnostic/independent-tolerance-witness.log`. The measured run used the correct tolerance and matched exactly, so its results are unaffected.

### Minor

None raised.

## Verification

Reviewer-authored JSON/hash reconciliation script and passing result: `.superpowers/sdd/2026-09-30-loss-balance-diagnostic/independent-final-reconcile.py` and `independent-final-reconcile.json`. Saved histories, config/provenance and checkpoint counters inspected independently of the author's reconciliation conclusion. Focused failure/controller/report regressions: **19 passed, 2 deselected in 10.65s**, retained in `independent-final-tests.log`; deselected tests would load/model-check scientific source data or run a tiny training fixture. Previously recorded full-suite results were inspected as author evidence, not claimed as independently rerun.

## Declined to judge

- New coefficient pairs, architectures, optimizers, larger budgets or held-out performance: outside this frozen diagnostic and unauthorized.
- Causal attribution of optimization failure or capacity limits: the design explicitly does not support these conclusions.
- Repeating production training or recomputing gradient calibration: prohibited for this review; saved artifacts and implementation were inspected instead.
- Full-pilot readiness: requires a separately reviewed protocol amendment; current joint gates failed.

## Assessment

**Ready to merge: With fixes.** The measured evidence and scientific interpretation reconcile, but the controller should enforce the reviewed replay tolerance before future invocation. Fixing this guard requires engineering checks only, with no repeat of production training.

## Author repair verification (after independent review)

The Important tolerance finding was independently reproduced by the author. Four initial regressions failed0.84s, five complete variants failed0.78s. Exact tolerance is now enforced before source/data access. First focused run1failed25passed10.98s exposed a fixture expecting strictJSON NaN input to produce saved blocked records; writer already rejects it before loading. Corrected assertion verifies early rejection/no arms for NaN, and saved blocked arms for allpersistablealtered configs.26focused checks passed11.55s; full152tests passed110.24s. No production computation repeated; all numerical artifacts/reports/figures/source hashes unchanged. This is author verification of the fix, not a second independent review. No remaining Critical/Important findings or deferred minors. Wider questions remain outside scope as listed above.
