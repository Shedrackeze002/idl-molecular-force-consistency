

### Paper observation — late-rate trajectory and endpoint tradeoff

Evidence IDs: LATE-RATE-OUTCOME-VERIFICATION-001 and LATE-RATE-TRANSIENT-001. Constant-rate fitting energy error spiked to 39.7647 times the original baseline at saved update 8,001 (raw 1.4918430382 eV²) and recovered by 8,100. Force error also rose. All eight groups show high energy error at that saved point; the constant schedule did not change at the boundary. The verified independent Adam/full-gradient boundary check passed, supporting correct execution without identifying the mechanism.

Late decay showed no comparable spike across 43 saved observations (energy fractions 0.1911994–0.2037762), yet its final fitting energy and force MSE were 7.95% and 6.12% higher than the constant arm. Both joint gates failed. Sparse observations cannot rule out behavior between scores; one checkpoint, one seed and fitted supports do not establish convergence or generalization.

Saved endpoint residual decomposition: a global energy offset accounts for only about 0.042% of constant fitting energy MSE and 0.000029% for late decay. Group-specific offsets and spike mechanisms remain unresolved; gates were never recentered. Negative learning results, failed engineering checks and accepted review repairs all remain in repository evidence.

Authoritative saved outcomes: reports/late-rate-diagnostics.json; descriptive notes: reports/execution/late-rate-2026-10-01/saved-outcome-notes.md/json and saved-spike-observations.json; figures: reports/figures/late-rate-curves.png and late-rate-curves-log.png. Exact report rebuild and numerical archive byte verification passed; original folders remain. Full pilot is pending. Next proposed lever is a separately reviewed controlled representation-capacity check, with GPU parity before a broader pilot. This proposal has not been executed.
