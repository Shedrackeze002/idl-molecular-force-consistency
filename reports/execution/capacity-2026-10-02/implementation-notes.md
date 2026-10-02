# Capacity implementation notes

Current status: Tasks1–3 implementation and64focused checks passed; both plan gates and repaired implementation followup cleared. Full suite running; full-size production unexecuted. Historical design notes follow.

Existing reusable helpers: late_rate._parts (hash-bound ordered arrays), _boundary (full gradient vs accumulated/NumPy Adam; fixed reviewed weights), _commit (immutable snapshots/history + published receipt), _prediction_error (all-frame finite absolute reproduction), verify_code (participating files/lock/runtime), residual_summary; width_rate_runs.validate_state/canonical_state_hashes; width_rate_replacement.serialized_state_metadata (opaque serialized shape/count/hash audit), no_symlinks/claim_root.

Use fresh case model/template; source parent parameters are never loaded. Source provenance/config/contract/commit/history0 and6000/child/probe-completion reconcile pinned data/norm/baselines only. Initialize inside each arm cap; save exact0 state shape/hash/Adam-zero checks and compare raw0 metrics to source baselines. Existing model family/readout/numerical code stays unchanged.

Expected complete case work:6000committed updates/768000fitting exposures;63scores*144=9072scoring predictions (0 included);16frame model initialization;128discarded warmup exposures; oracle1/3001 each128full-gradient+128accumulated exposures and1NumPy+1accumulated discarded proposal;144endpoint-reference predictions; child1template init on16 and144endpoint predictions. Incoming3000 moments nonzero; no discarded work enters update count. Partial categories must retain started/unknown flags.

Ruling: retain existing dedicated codex/landscape17-baseline checkout instead of creating a new worktree. Inputs/ancestry and old numerical closures are path-bound, ignored runs and existing dependency environment live here; separate new files and exclusive new run root provide scope separation. Cost if wrong: accidental edits to old sources would break prior code hashes, so old files remain unchanged and fresh full suite/source checks gate production. User PDFs are excluded from staging.

Pre-flight: Task1 source/state contract -> Task2 load/run/probe; Task1 audit plus Task2 persisted receipts -> Task3 saved reporting. Config and signatures match. Plan documentation and hypothesis are proposed work, not results. All original and new workspaces remain under the user's preservation rule.
