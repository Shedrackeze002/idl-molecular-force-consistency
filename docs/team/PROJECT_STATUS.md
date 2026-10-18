# MACE preparation update — 18 October 2026

The shared ASE calculator bridge is synthetic-tested in both existing environments and independently reviewed. [Preparation result](../../reports/MACE_EVALUATOR_PREPARATION_2026-10-18.md) and [contract](../mace-evaluator-contract.md) distinguish this engineering result from pending checkpoint loading/parity/recovery. Idris retains loader/reproduction and Olufunke runner/reporter integration. Real MACE inference follows the midterm draft; the solver is unchanged. Earlier research results below remain authoritative for their recorded scope.

# Project status and progress

See [the18October research update](../../reports/E3X_PROJECT_UPDATE_2026-10-18.md) for current accepted evidence. Earlier status below is historical.

# Project status — 15 October 2026

The data and evaluator infrastructure are established, MACE Run A is a verified reference, and all three E3x training paths are implemented. The controlled energy/direct/consistency comparison and shape-recovery hypothesis remain unanswered.

| Area | Verified status | Next |
|---|---|---|
| Data | 4,891 training / 974 validation / 1,127 test structures across 26/5/6 aspirin connections; classified endpoint references | Preserve split, units and atom order; keep test closed |
| MACE | Run B diagnostic and 2,048-epoch Run A independently scored | Idris reproduction and shared inference adapter |
| E3x | Three training paths reviewed; first energy/direct fits completed 1,000 updates | Wider-model and consistency timing, common budget and 12-run resource ledger, then three scientific candidates |
| Learning | Both first E3x BEST checkpoints remain at update 0; force validation worsens despite improved energy error | Preserve the negative result; no adequacy or recipe-selection claim |
| Recovery | Olufunke's evaluator PR #4 and follow-up fixes in PR #6 are merged | Trained-model integration and measured validation no-relaxation control |
| Physics/cost | Emmanuel owns physical consistency, cost and connection analysis; completion is not yet verified | Analytic fixtures, synchronized timing and report section |
| Replication | Paracetamol endpoint 01_2/plus remains unmatched | Separate reviewed resolution after the aspirin pipeline works |

MACE final validation force MAE is 5.259833 meV/Å pooled and 5.348957 meV/Å equal-connection; RMSE is 26.410390 meV/Å. One seed and validation-influenced selection limit claims. No test, relaxation or causal result is established.

First E3x initial and selected force MAE is 62.504390 meV/Å. Final energy-derived/direct MAEs are 70.467673/66.222396 meV/Å. These short engineering fits consume no scientific candidate slots.

Declared candidates: A uses width 32, learning rate 0.001 and equal weights; B uses width 64, learning rate 0.003 and equal weights; C repeats B with the preserved balanced coefficients. The proposed 50,000 updates remains provisional. Measure all arm types, then fix resources, overhead, reserve and the deadline budget for all 12 runs before any candidate starts. Each new plan receives two independent reviews. Nothing is running at this publication snapshot.

Midterm draft target: 20 October. Saved course deadline: 23 October, to verify against live announcements. Writing ownership follows the workstreams and is not completed contribution credit. Draft methods, MACE results and balanced preliminary E3x findings now.

Reports: [MACE Run A](../../reports/mace-runA-fit.md), [first E3x fits](../../reports/e3x-first-engineering-fits.md), [evidence and attempts](../../reports/evidence/2026-10-15/README.md), [reproduction/access](../reproduce-oct15-results.md). The original proposal and 3 October handoff are historical context.
