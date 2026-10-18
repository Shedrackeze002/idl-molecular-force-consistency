# Reproducing and auditing the October 18 exports

Use the locked Python 3.12 environment and repository source. The export contains accepted A/B/C baseline scores, all nine controlled BEST validation prediction arrays, consistency diagnostics, all fixed validation-recovery and endpoint-diagnostic JSON outputs, and the accepted paired-trace/control result. Numeric, boolean and null JSON values are retained under location-only normalization; NPZ, raw trace arrays and query-journal bytes are unchanged. The manifest lists separate original and exported hashes.

Original checkpoints, large per-update histories, private operating records and exact historical launch packages remain retained by the coordinator. This publication supports saved-prediction and raw-trace auditing, but is not a self-contained model-replay archive. Request verified checkpoint/data access before inference. Do not pass normalized historical exports directly into strict original launch verifiers. A fresh run needs reviewed records and its own source identity. One loader test now uses repository-local data instead of a workstation path.

Run helper and observer checks explicitly:

```sh
LANDSCAPE17_ARCHIVE=/nonexistent/landscape17-routine-verification-disabled.zip PYTHONPATH=.:src:scripts python -m pytest tests/test_e3x_phase_tools.py tests/test_e3x_comparison.py tests/test_e3x_comparison_score.py tests/test_validation_controls.py tests/test_validation_recovery.py tests/test_e3x_recovery_adapter.py tests/test_recovery_diagnosis.py tests/test_relax_trace.py tests/test_recovery_trace_driver.py
LANDSCAPE17_ARCHIVE=/nonexistent/landscape17-routine-verification-disabled.zip PYTHONPATH=.:src:scripts python -m pytest
```

Dataset-dependent controls require the authenticated archive at data/Landscape17.zip, which is ignored and not copied into this publication. The guarded loader reads validation starts and reference minima only. The main test remains closed.

The paired raw traces contain arrays.bin, an append-only sequence of numeric NumPy arrays, and queries.jsonl with corresponding position/force offsets. Read arrays with numpy.load(..., allow_pickle=False), reconstruct query records, and apply recovery_trace.reconstruct_trace under the frozen configuration and recovery_trace.summarize_trace at 5 Å. Full derived graph summaries are omitted to avoid duplication; they can be regenerated from these byte-identical raw inputs. Per-call relaxation.json retains the full returned solver state. Geometry auditing also requires the authenticated reference structures.

The observer wraps the unchanged relaxer and adds no model query. The driver refuses repeat output/dispatch markers, binds two record reviews, and enforces 600 numerical seconds plus 60 seconds for existing-artifact finalization. Traces and reference controls are diagnostic evidence, separate from original A2. No displacement cap, solver sweep, force projection or training intervention was added.


## Verification scope

The publication default suite passed 502 tests with 3 skips. Nine existing model-free archive fixtures loaded all aspirin records, including held-out test-path coordinates and labels, for structural/start-set checks. No trained checkpoint was scored on test data and no model, solver or parameter was selected from those checks. The scientific diagnostic itself read only its 21 bound validation-start/reference members. Routine future default checks should set LANDSCAPE17_ARCHIVE to an unavailable sentinel and run the guarded validation-only loader separately. See [the verification receipt](../reports/evidence/2026-10-18/verification-scope.json).
