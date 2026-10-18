# Research progress, 18 October 2026

The controlled fitting, scoring, validation recovery and two diagnostic phases are complete and independently reviewed. The main test remains closed. Midterm writing is scheduled for October 20–23.

## Predictive comparison

The selected energy baseline C passes the frozen A1 force-error floor. All nine subsequent controlled fits reached 200,000 updates. Two-seed mean equal-connection validation force MAEs are:

| Formulation | Force MAE (meV/Å) | Matched to direct under the original rule |
|---|---:|---|
| Energy-derived | 34.89 | Reference arm |
| Direct | 38.19 | Comparator |
| Consistency λ=0.1 | 38.02 | Yes |
| Consistency λ=1 | 34.68 | No |
| Consistency λ=10 | 39.73 | No |

Only λ=0.1 meets the unchanged matching rule. It improves finite-record force versus auxiliary-energy-gradient agreement, with higher auxiliary energy error and about 1.985 times the observed fit-plus-score time. These findings do not establish global conservativity or recovery benefit. MACE remains an independently scored external predictive reference at 5.349 meV/Å equal-connection validation force MAE; its relaxation recovery is unmeasured. The dated baseline report is a 16 October snapshot, so its then-pending comparison status is superseded here.

## Original negative recovery result

Every evaluated model recovered 0/10 classified own references, against a no-relaxation control of 0/10. A2 failed. All 25 force-converged endpoints were geometrically wrong. Whether consistency improves recovery remains unanswered.

A fixed diagnosis reproduced all 50 stopping flags. All five models exceeded the 10 meV/Å force threshold at all 11 exact reference minima, whose label forces stayed below 2.745 meV/Å. This does not exclude nearby threshold-converged points within 0.05 Å. All 23 converged dual-head endpoints were disconnected at the frozen 5 Å cutoff; seven had every atom isolated and exactly zero force. The two converged energy endpoints remained connected. A separate nonconverged energy endpoint also separated, so the observation is not exclusive to dual-head models.

The actual procedure caps scale, not coordinate displacement, and tests trial convergence before force agreement. The strict all-isolated limit makes the consistency residual uninformative for the frozen E3x architectures. Auxiliary gradients at those seven endpoints were not measured. See [the addendum](E3X_RECOVERY_DIAGNOSIS_REVIEW_ADDENDUM_2026-10-18.md) and [known issues](../docs/EVALUATOR_KNOWN_ISSUES.md).

## Accepted trace and reference-start diagnostic

All four selected historical outcomes reproduced exactly, and all thirty fixed reference-start controls completed. The package used 38,853 force attempts and about 75 seconds locally, without retry. The selected direct failure accepted a largest atom displacement of 436.721 Å; the separated energy case accepted 268.047 Å. The largest jump is not necessarily the first separation, and the selected cases do not estimate population rates or prove sole causality.

Operational force-converged own matches from six shared reference starts were energy C0 4/6, direct seed 0 1/6, and 0/6 for each of the other three models. Success does not establish an exact stationary point or stable minimum. Failure does not prove nearby stationary points are absent. The original A2 result remains unchanged. Read [the complete trace report](E3X_PAIRED_TRACE_2026-10-18.md).

Evidence: [export manifest](evidence/2026-10-18/export-manifest.json), [controlled acceptance](evidence/2026-10-18/controlled/acceptance.json), [recovery acceptance](evidence/2026-10-18/recovery/acceptance.json), [diagnosis acceptance](evidence/2026-10-18/diagnosis/acceptance.json), [trace acceptance](evidence/2026-10-18/paired-trace/result_acceptance.json). Original failures, checkpoint identities and export limitations remain documented. No new solver or training intervention is implied.


## Verification scope

The publication default suite passed 502 tests with 3 skips. Nine existing model-free archive fixtures loaded all aspirin records, including held-out test-path coordinates and labels, for structural/start-set checks. No trained checkpoint was scored on test data and no model, solver or parameter was selected from those checks. The scientific diagnostic itself read only its 21 bound validation-start/reference members. Routine future default checks should set LANDSCAPE17_ARCHIVE to an unavailable sentinel and run the guarded validation-only loader separately. See [the verification receipt](evidence/2026-10-18/verification-scope.json).
