# Fresh width/rate: independent final review

Reviewer `/root/width_rate_final_review`, gpt-6-astra/high, read-only. One fresh whole-branch review from the twice-reviewed plan base `439adba`; static review during the once-only run, then saved-results/figure reconciliation after it stopped. No implementation delegation, numerical rerun, checkpoint restoration or duplicate suite.

## Important findings and author verification

1. Global elapsed 0 could coexist with four 1000-second sequential cells. Author reproduced it and enforces finite global time at least the actual parent, once-only shared initialization and cell durations, within 1µs clock tolerance and the strict global cap.
2. Interrupted preflight omitted completed initialization and unknown pending work. Author reproduced it and derives operation counts from the latest started/completed/unknown event, without double counting event pairs.
3. A coherently rehashed non-replay update 0 could use update 100 metrics. Author reproduced it and binds all four raw update-0 baselines and shared initialization metrics to immutable historical originals.
4. Failed probes recorded zero time and omitted known reference frames. Actual failed artifacts expose this; author reproduced it, records parent reference frames before launching the child, marks attempted child work unknown on failure, and records elapsed failure time for future runs. Derived reporting preserves old raw receipts, uses hash-bound reference work, and labels old failed probe duration unavailable.

Separately, metadata-only autopsy diagnosed the actual checkpoint rejection: historical integer keys sorted numerically before saving, then lexicographically as strings after JSON reading. Author reproduced the exact 200/1000 ordering condition and normalizes provenance once before both strict serializations. The strict shared loader is unchanged. An initial inadequate fixture passed because its key order did not differ; the corrected fixture failed before repair. No production checkpoint was modified or re-probed.

Four initial report regressions failed; 37 report tests then passed in 17.90s. Subsequent provenance/accounting regressions failed before repair; 66 focused runner/report checks passed in 86.18s. Full final suite evidence is recorded in the execution record. Repairs are author-verified; the reviewer did not re-review repaired code. No frozen configuration or historical source was amended.

## Saved-result and figure reconciliation

The reviewer independently verified 326 source pins, saved original code identities before post-run repair, all 64 checkpoint/history envelope payload/control bindings and history prefixes, and 145 final report raw-artifact hashes. All 63 common historical energy/force measurements match exactly, maximum difference 0. The final authoritative receipt records 6000 persisted updates, 768000 fitting exposures and 128 discarded warmup exposures. First-step independent Adam oracle passed. Final subprocess checkpoint verification failed; three later cells were blocked.

Final fractions of original fitting energy/force MSE: 0.3220330430785254 / 0.45318885284127763. Fully fitted anchor fractions: 0.10905757620503544 / 0.42746961258301264. Both strict joint gates fail. The failed matrix permits no factorial contrast, winner, capacity/optimizer-causality claim, generalization result or full-pilot clearance.

The final four-panel figure was independently viewed. All 64 prescribed observations remain; energy spikes at 3900 and original thresholds are visible; trajectory is labeled partial/invalid, no blocked-cell curves invented. Derived accounting correctly retains 144 completed parent reference frames, two shared initializations, unknown failed probe work/duration and `work_counts_complete=false`. Raw failed fields remain unchanged.

**Final saved-output verdict:** no additional Critical or Important inconsistency. Output SHA256: JSON `548158a2e36515a847af63c326e237ac456a65fa7ec1bf5a9064b7a0397efd1a`; figure `cb73691d372d29b4e290f5a1d7268cfa9d3c8eeb531c48224654eef2966e7560`; Markdown `1d9813c12c19ec5de3d253ceb2e98122a076b27fae8188a9ccead8170dfff482`.

## Deferred Minor findings

- Figure axis says “committed update”; accurate for this run (all 64 observations persisted), but a general failed-write trajectory could include a synchronized, uncommitted scored row.
- Markdown boilerplate says descriptive 5500→6000 changes are in JSON, but `terminal_window` is empty for this invalid execution. **Correction for readers: terminal summaries are unavailable in this report.** No endpoint or convergence inference should be drawn from this boilerplate.

## Declined scope and author rulings

- Initially deferred saved-results/figures: now completed with the identities above; cost was a pending verdict until artifacts existed.
- Author repair correctness and test results: native author regression/full-suite evidence, not independent numerical verification; cost is lack of a second repair-code audit, per the one author fix-pass workflow.
- Physical checkpoint restoration/predictions: explicitly not retried after the failed once-only trial; stays unverified, cost is no retrospective execution clearance.
- Blocked-cell scientific outcomes: unavailable, not inferred; cost is no width/rate comparison.
- Coherent deliberate fabrication of every new artifact: outside software internal-provenance checks; no external attestation or stronger trust certificate claimed. Cost is trust in the execution environment.

Repository evidence log and execution archive preserve all trials and review records.
