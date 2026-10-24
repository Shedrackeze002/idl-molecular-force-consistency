# Run the CPU representation comparison

This study compares scalar and scalar/vector graph features. Both derive forces from energy. The fixed settings and interpretation are in the20October representation specification. Accepted E64 fits are reused only after a reviewed1000-update reproduction; at mostfour SP/S64fits are new. No test scoring or recovery result is produced here.

Use the **pinned execution checkout** and the existing Python3.12.13 CPU environment. Preserve its commit and numerical source through any resumable fit. The separate publication checkout can publish results without changing execution HEAD. Do not run two queue processes against the same output directory. This is ordinary single-process operation, without an owner-file recovery procedure.

A reviewed local launch record is written to `configs/representation-study/local-launch.json` before the production run. It contains actual prepared-data/output locations, parameter counts, execution/code identities and the verification receipt hashes. That machine-specific file is ignored by Git. `configs/representation-study/cpu.json` describes the portable study contract; it is not a substitute for measured count/reproduction/resume verification. Contributors need the separately supplied prepared train/validation/diagnostic partitions and original manifest; never copy test numerical payload into the run package.

From the execution checkout, with its Python environment selected:

```bash
PYTHONPATH="$PWD:$PWD/src" python -m experiments.representation_study.runner --queue configs/representation-study/local-launch.json
```

Alternatively open `notebooks/representation_comparison.ipynb` from this checkout and Run All with that CPU kernel. Four readable cells load settings, verify environment/source, run/resume the serial queue and show saved results. There is no manual confirmation cell or timed pause.

The runner uses the existing trainer and its500-update full-state checkpoints. Each invocation writes a fresh numbered segment; orchestration logs/configuration live in a separate attempts directory so the trainer receives an empty fit directory. One authenticated scheduled checkpoint copy is retained per segment. On restart the newest valid saved update wins, not file modification time. Corrupt or off-schedule saves are logged and bypassed for an older valid state; source/config identity mismatch stops explicitly. A graceful wall stop may evaluate off-grid and alter BEST, so its state is ineligible, rather than repaired by deleting JSON rows. Do not change source or suppress checksum/provenance errors to resume.

Complete fits are skipped only after summary/config/checkpoints and new-process reload proofs verify. Failed or incomplete operations remain evidence; they are not scientific success and are not silently retuned/retried. A kernel/terminal interruption can lose work since the last valid scheduled save, which is replayed on resume. Keep the Mac awake and powered for the authorized overnight run; closing the notebook is not a checkpoint operation.

BEST/LAST are independently scored on the same974validation records. Report the mean equal-connection force MAE over the20scheduled checks190500–200000 as a separate trajectory metric. Opposing BEST/late-window delta signs are mixed evidence. Source-bound raw summaries/checkpoints/predictions remain local; publish readable code, settings, scores and limits after each result. Parameter count, timing or correctness on scalar configurations is not assumed from the old E64 timing.
