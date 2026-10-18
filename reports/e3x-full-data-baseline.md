# Full-data E3x baseline candidates and validation adequacy

16 October 2026. Status: accepted after independent engineering and scientific saved-result reviews. These are three completed seed-0 energy-derived fits; direct and consistency comparisons have not run at this budget.

## Result and scope

All three predeclared configurations completed 200,000 updates, and fresh CPU inference from their saved BEST checkpoints reproduces the logged validation errors. The corrected A1 assessment selects configuration C, whose equal-connection force MAE is 0.03458740829384409 eV/Å against the fixed 0.0500 eV/Å floor. This is an informative force-learning result at the achieved accuracy level. It does not establish convergence, a joint energy/force criterion, shape recovery or the project's main comparative hypothesis.

| Configuration | Force-selected BEST update | Equal-connection force MAE (meV/Å) | Pooled force MAE (meV/Å) | Energy residual MAE (meV/molecule) |
|---|---:|---:|---:|---:|
| C | 184,500 | 34.587408 | 34.591506 | 126.990138 |
| B | 176,500 | 40.369870 | 40.350670 | 28.177781 |
| A | 198,500 | 42.108456 | 42.196107 | 28.094417 |

We evaluated every one of the 974 validation records across five held-out connections. The zero-force reference is 0.06250438966560673 eV/Å, independently recomputed from labels and matched against update zero. C reduces that error by 44.6640%. Its individual connection force MAEs range from 30.389863 to 38.663359 meV/Å. Every candidate has 401 scheduled validation entries, a nonzero first-minimum BEST, completed full-U LAST and successful fresh-process BEST/LAST prediction checks. Full predictions, reference arrays, IDs and per-connection reductions are retained.

## Frozen configurations and exposure budget

A uses width 32 and learning rate 0.001. B uses width 64 and learning rate 0.003. Both give the normalized energy and force terms unit weights. C retains B's width and rate, with energy weight 0.014584365670390719 and force weight 1.9854156343296092. All use the shared degree 1 backbone, two interactions, eight radial functions, 5 Å cutoff, zero-initialized readout, plain Adam, batch 4, float64, seed 0 and validation every 500 updates. There was no adaptive tuning, fourth configuration or early stopping. The fixed 200,000 updates give 800,000 sampled exposures, approximately 163.57 effective passes with replacement over 4,891 training records.

C→B→A was the predeclared serial order. Selection uses the lowest equal-connection validation force MAE at the force-selected BEST checkpoint. A1 is one assessment of the same three configurations, not three independent hypothesis tests. The threshold was fixed with the zero-force validation value already known. Selection and the floor use the same validation connections, so the result is optimistic and has no independent test interpretation. Later seeds will be reported without re-deciding A1.

## What the comparison suggests, and its limits

The weighting change is associated with 14.3237% lower BEST force MAE for C than B in this single-seed comparison. There is a cost: C's energy residual MAE at its force-selected checkpoint is 4.507 times B's. We should carry this force-energy trade-off into the paper and assess whether the achieved force field supports relaxation. The values come from separately force-selected checkpoints; they do not isolate fixed-update training effects or establish repeatability across seeds. Comparing A and B also changes width and learning rate together, so we cannot attribute their difference to width alone.

The independently scored MACE Run A reference remains stronger on force MAE: 5.348957 meV/Å equal-connection versus C's 34.587408. Architecture and training exposure budgets differ, so this is context rather than a controlled efficiency or architecture comparison. Earlier short E3x fits retained BEST at update 0, and the previous tiny-subset learning checks left force-MSE fractions around 0.32–0.98. Those unsuccessful observations remain in the evidence trail; the cause of that earlier plateau is not resolved by this full-data result.

Only train and validation data are used here. The previous label-consistency report covers all 26 training and five validation connections, with both branches, but its exact historical generator bytes remain unauthenticated. No validation relaxation result selected a model or checkpoint. A2, the direct/consistency two-seed comparison, matched-error decision and six-connection test evaluation remain pending.

## Scorer failure and transparent correction

The original independent scorer rejected all three checkpoints with `BEST embedded state mismatch`. Its comparison reserialized a restored state, while Flax restores dictionaries in template order. For each real model, all 68 state leaves had exactly equal values, shapes and dtypes; six dictionary order differences changed the serialization. Each authenticated original checkpoint payload was already byte-identical to the embedded BEST state. The fresh-process prediction checks had passed at maximum differences below 3.4e-14, with tolerance 1e-10.

The repair compares the authenticated original payload bytes directly with the embedded state and reports state/control digests. Corrupted payload/control, wrong provenance and even a nextafter-sized altered leaf are rejected. All other data, source, full-budget, history, selection, zero-force and full-prediction checks remain. Regression evidence includes six initial failing cases, one additional failing control-digest assertion, 29 focused passes and 493 default passes with 12 expected skips. The initial review findings and revisions are retained.

Training revision 8ead1bd03142a4c95addbf1bf06d8e7a515cf4a7 and numerical fingerprint a046011a0efdfab2835b00e42f446e691b453b1f30854f8accce26c6292f4da5 are unchanged. The scoring implementation is local revision 26756042c1c84d6c712a21ef629f7a0ee8e8466e. Separate score-only records change exactly the scorer helper hash, 377f97eb… to 581f727a…, with explicit old/new bindings. All 48 original run files, including failed outcomes and raw A1=false, remain byte-identical. The corrected assessment supersedes that operational false result on the same three consumed scientific slots. No weights were modified, no training was repeated, and no extra configuration was introduced.

## Measured operational cost

Summed training-process elapsed time was 3.350262 CPU-hours, including each trainer's setup, compilation, validation, writes and checkpoint probes. This excludes separate failed scorers, earlier invalid import dispatch, implementation/testing, repair investigation and other coordination overhead; it is not total project effort or CPU-core-hours. Filesystem timestamps place the final A summary near 01:29 Kigali on 16 October, not an exact process-finish clock. These calendar observations are separate from summed process durations.

The repair's three fresh scorer processes took 7.120796 s (C), 6.485928 s (B) and 6.062905 s (A), with 19.895287 s for the repair driver overall. Each was below both its original approximately 26 s allowance and the separate reviewed 180 s repair cap. That does not retroactively turn the failed original scoring calls into successes. The earlier 4.602196-hour CBA figure was a forecast including reserve, not measured cost. Local CPU work used no Colab compute units. Some failed-dispatch/repair-investigation overhead was not independently timed, so a complete operational total is unavailable.

## Evidence and next use

Validation trajectories (retained audit record) show the force improvement and energy trade-off; markers indicate the force-selected BEST in both panels. PDF figure (retained audit record) and SVG are retained for export.

The source of record is the correction manifest (retained audit record), independent root reductions (retained audit record), full corrected scores (retained audit record), corrected A1 (retained audit record), and the [balanced evidence log](PAPER_EVIDENCE_LOG.md). Both saved-result reviews are accepted; the acceptance receipt (retained audit record) freezes C and its checkpoint identity for subsequent runs. Next use is the selected fixed recipe for the remaining seed and direct/consistency fits, together with A2's validation relaxation and measured no-relaxation control under their reviewed protocols. Those jobs have not launched.
