# Late learning-rate continuation diagnostic

Execution valid: True; full pilot: pending.

| Arm | Original-baseline fitting E fraction | Original-baseline fitting F fraction | Joint learning gate |
|---|---:|---:|---|
| constant | 0.17711403 | 0.32469875 | False |
| late_decay | 0.19119942 | 0.34455828 | False |

Known computed additional updates: 8000; fitting exposures: 1024000.

Contrast orientation: late_decay minus constant; ratios late_decay divided by constant. See JSON for raw units, groups, anchors, histories, residuals, all work records and verification errors.

- One selected checkpoint, one seed and fitted supports; no held-out or main-hypothesis result.
- The two-stage reduction is one bundled intervention; individual phase effects and optimality are untested.
- Fixed finite budget and sparse scores do not prove convergence or identify every spike/mechanism.
- Raw energy/force gates use immutable update0 baselines; residual decomposition never recenters them.
- Sequential wall times are engineering records, not hardware efficiency rankings.
