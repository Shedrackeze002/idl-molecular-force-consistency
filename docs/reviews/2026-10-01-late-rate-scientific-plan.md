# Scientific review: late learning-rate continuation plan

Date: 2026-10-01. Scope: scientific validity and evaluation controls only; no training or implementation performed.

Reviewed `AGENTS.md`, `docs/superpowers/specs/2026-10-01-late-rate-design.md`, `docs/superpowers/plans/2026-10-01-late-rate.md`, `configs/aspirin-late-rate.json`, the corrected width/rate report and saved observation notes, and selected parent contracts/history. Independently verified all 16 entries in `reports/execution/late-rate-2026-10-01/source-pins.json` match the current bytes and its configured SHA256 `55fb10bef1abc25783d8b6376dbfcb4896da71db1f6216210a01249224b42e30`. The pinned parent update6000 history contains all 64 original observations, including the update2000 spike; a separate unpinned historical file is unnecessary for this analysis.

## Gate recommendation

**PASS for implementation of the specified bounded diagnostic.** Findings: 0 Critical, 0 Important, 3 advisory clarifications below. This is a plan assessment, not production clearance or assurance that late decay will improve learning. Final implementation review, independent numerical verification, full regression verification, and pre-registration remain required before production.

## Scientific controls that are adequate

- The question follows directly from continued late loss movement and failed joint gates. A smaller fixed rate from initialization does not answer late continuation from a common trained state.
- Both arms inherit the identical verified width32/.001 update6000 parameters, nonzero Adam moments/count, examples, normalization, loss weights, precision, and update0 acceptance baselines. Preserving Adam state prevents an optimizer reset from becoming a second intervention.
- Constant continuation provides the missing equal-update control. Each arm receives exactly4000 additional full128-example updates; inherited work and discarded oracle/probe work are separated from new training exposures.
- Fixed arm order, endpoint, score grid, time caps, no early stopping/retry/winner selection, and blocking later work after failure prevent success-dependent extension. Partial results cannot produce a complete endpoint contrast.
- Raw energy and force acceptance criteria remain unchanged. Energy residual mean/variance is descriptive and never removes energy bias from the acceptance metric. The historical constant-mean energy field is correctly distinguished from prediction-error drift.
- Anchors remain explicitly fitted, with no held-out/generalization claim. The parent was chosen on training diagnostics and the report preserves that selection condition. Single-seed and finite-budget limitations are stated.
- The parent report remains a four-case diagnostic with failed learning gates; beginning this continuation does not reinterpret it as a successful baseline or run the full pilot.

## Advisory clarifications

1. **Treat the two-stage schedule as one intervention.** The constant-versus-decay endpoint comparison cannot isolate the contribution of the .0003 phase from the .0001 phase, or establish that either is optimal. Exact fix: include that limitation in the final report and evidence entry; retain the frozen schedule rather than infer phase-specific causality from sampled curves.

2. **Make contrast orientation and denominator handling explicit.** “Signed differences and ratios” is adequate as a plan, but interpretation is clearer with `late_decay minus constant` and `late_decay divided by constant`, separately for both metrics and roles; a ratio should be absent with a recorded reason if its denominator is zero. Exact fix: define these conventions in the report schema/tests and report text. Differences carry units; ratios are dimensionless. Neither is statistical significance.

3. **Keep saved descriptive analysis from becoming undeclared schedule selection.** It is reasonable to extract residuals and the previously observed spike before execution, but the reviewed rates/budgets are already fixed. Exact fix: record that this analysis does not alter the frozen protocol. Any material alteration prompted by it needs a linked prospective amendment and follow-up reviews before training. Report spikes only at observed score locations, with no claim of the true continuous-time maximum or an established mechanism.

## Residual limits

This experiment measures conditional fitting behavior on one inherited trajectory and two predefined schedules. It cannot establish held-out force accuracy, molecular relaxation stability, capacity sufficiency, the main consistency-training hypothesis, a generally best learning-rate schedule, convergence, or hardware efficiency. The two supports overlap completely for the16 anchors, so they are correlated diagnostics rather than independent replications. Equal-group metrics on the balanced128-example support are descriptive and may coincide with record-weighted metrics; they do not create additional independent evidence.

No change to the experimental arms or acceptance gate is requested. The advisory report clarifications can be resolved during implementation without changing the numerical protocol.
