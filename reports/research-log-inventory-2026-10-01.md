# Saved research report inventory

Read-only coverage of all17existing top-level JSON reports. No experiment rerun or independent metric audit. Every report below is now directly indexed with its checksum in repository evidence.

| Saved report | Outcome fields | Previously directly indexed |
|---|---|---|
| reports/baseline-learning-checks.json | {"status": "learning gate failed; full pilot pending"} | True |
| reports/baseline-physics-zero-readout.json | {} | False |
| reports/baseline-physics.json | {} | True |
| reports/convergence-budget-diagnostics.json | {"execution_valid": true, "full_pilot": "pending", "interpretation_valid": true, "status": "completed"} | True |
| reports/data-coverage-diagnostics.json | {"execution_valid": true, "full_pilot": "pending", "status": "completed"} | True |
| reports/force-learning-diagnostics.json | {"execution_valid": true} | True |
| reports/landscape17-aspirin-44-complete-endpoint-check.json | {} | True |
| reports/landscape17-aspirin-66-complete-endpoint-check.json | {} | True |
| reports/landscape17-duplicates.json | {} | True |
| reports/landscape17-endpoints.json | {} | True |
| reports/landscape17-paracetamol-01_2-complete-endpoint-check.json | {} | True |
| reports/learning-check-diagnostics.json | {} | True |
| reports/loss-balance-diagnostics.json | {"execution_valid": true, "full_pilot": "pending", "status": "completed"} | True |
| reports/optimizer-step-diagnostics.json | {"execution_valid": true, "full_pilot": "pending", "interpretation_valid": true, "status": "completed"} | True |
| reports/representation-gradient-diagnostics.json | {} | True |
| reports/width-rate-diagnostics.json | {"execution_valid": false, "full_pilot": "pending", "interpretation_valid": false, "status": "failed"} | True |
| reports/width-rate-recovery-verification.json | {"full_pilot": "pending", "status": "passed", "verification_valid": true} | True |

Previously unindexed artifact: reports/baseline-physics-zero-readout.json. It remains intact and is added to the evidence index.

The zero-readout-named physics file deliberately perturbs its energy head; its filename must not be interpreted as a trivial zero-force test. It records untrained analytical checks, not DFT accuracy or learned physical recovery.

- Filename/hash search tests direct indexing only, not semantic completeness of every historical log paragraph.
- This snapshot does not certify every raw run, figure, source PDF, trial rationale or analytical derivation across the entire project.
- New comparison has its own reviewed completeness reconciler; missing or unreceipted work remains unknown.
