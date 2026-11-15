# Run the admitted MACE validation parity check

This workflow implements the fixed validation-only task. It does not train,
relax molecules, access training/test partitions, or authorize model loading.

## Before numerical execution

1. Verify the released v2 packet and retain its original bytes.
2. Complete the WSL runtime and separate test lock.
3. Obtain an approved WSL admission receipt enumerating every allowed platform,
   package and source-file difference.
4. Obtain two resolved reviews of the exact implementation commit.
5. Obtain Shedrack's explicit authorization for one launch receipt. The receipt
   must pin the approved admission receipt hash, the 21-symbol species order,
   all fields below, and exactly the implementation commit reviewed twice.

The approved launch JSON is fail-closed. It records absolute `python_executable`,
`packet_root` and fresh `attempt_root` paths; `command` is exactly the interpreter,
`scripts/check_mace_parity.py`, `--packet-root`, and `--attempt-root` argv used by
the supervisor. It also records `source_baseline` as
`3fa5969e5a0c2fdeaebc76a05bace20799715203`, two or more resolved review objects
with distinct review IDs, reviewer identities and URLs bound to the same full
implementation commit, and these exact limits:

```json
{
  "resource_limits": {
    "seconds": 1800,
    "attempt_bytes": 268435456,
    "per_log_bytes": 16777216,
    "cleanup_seconds": 10
  }
}
```

The WSL approval separately binds complete, equal package declaration/version
key sets, all eight canonical installed-source identities, and raw byte receipts
for the unchanged shared bridge and evaluator. Those frozen receipts also name
their accepted baseline commit and Git blob identities. Before construction,
clean-filtered live worktree blobs are compared with immutable objects derived
from the pinned implementation and baseline commit trees; refreshing a receipt's
self-reported hashes cannot admit edited source. Missing, substituted or
contradictory launch/admission fields are refused before model construction.

Synthetic tests do not require or load the checkpoint:

```bash
PYTHONPATH=.:src:scripts python -m pytest -q \
  tests/test_mace_loader.py tests/test_mace_parity.py \
  tests/test_mace_parity_supervisor.py tests/test_mace_parity_audit.py \
  tests/test_mace_reference_adapter.py -p no:cacheprovider
```

Only after the gates above, use a fresh absolute output path:

```bash
PYTHONPATH=.:src:scripts "$python_bin" scripts/supervise_mace_parity.py \
  --packet-root "$packet_root" --output "$attempt_root" \
  --launch-receipt "$launch_receipt" \
  --wsl-admission-receipt "$wsl_admission_receipt"
PYTHONPATH=.:src:scripts "$python_bin" scripts/audit_mace_parity.py \
  --packet-root "$packet_root" --attempt-root "$attempt_root"
```

The worker authenticates 11 packet members, validates all 974 inputs before
construction, copies and hashes the primary final EMA model, constructs one
CPU/float64 calculator, then makes exactly 974 force and 974 energy accessor
requests. Acceptance is absolute-only (`rtol=0`, `atol=1e-10`) for energy and
force components.

The supervisor uses one fresh directory, a shared 1,800-second cap, 256 MiB
directory bound and 16 MiB per-log bound. Timeout, overflow, partial output,
identity refusal and numerical disagreement are retained and never retried or
relabeled as success. The independent audit reads saved arrays only; it does
not import or load MACE/Torch.
