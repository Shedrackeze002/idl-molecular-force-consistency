# Copied-checkpoint recovery verification

Recovery status: **passed**; copied verification valid: **True**.

New training updates: **0**. Numerical verifier attempts: **1**. Completed/unknown new work: `{"optimizer_proposals": 0, "prediction_frames": 144, "template_initializations": 1, "template_initializer_frames": 16, "training_updates": 0, "warmup_frames": 0}`.

Controller elapsed: 5.668380625000282 seconds; measured verifier elapsed: 4.785355457999685 seconds. Later report/archive work is outside the controller cap.

The original matrix remains **failed**, both original learning gates **failed**, three cells **blocked**, contrasts **unavailable**, full pilot **pending**. This separate receipt is never grafted onto the old trial.

Original accounting is inherited completely in the JSON, including per-arm oracle/evaluation/reference work and parent/shared initialization receipts. The old failed verifier duration and unreceipted work remain unavailable; combined accounting is incomplete.

- A passing copy receipt reproduces saved endpoint predictions under the pinned implementation; it is not DFT accuracy or physical recovery.
- Original matrix remains failed, original learning gates failed and three cells blocked; no hybrid matrix, contrasts, winner or pilot clearance.
- Original child work/duration remains unknown; original and recovery effort are separate, never complete combined totals or efficiency evidence.
- No new training, optimizer proposal, warmup, curve remeasurement or held-out evaluation. Future matrix work needs a separate reviewed plan.
