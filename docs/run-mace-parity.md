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
   must pin the approved admission receipt hash and the 21-symbol species order.

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


## Frozen bridge identity

Before model construction, the loader authenticates `evaluation/__init__.py`,
`evaluation/adapter.py` and `scripts/mace_reference_adapter.py` against source
snapshot `3fa5969e5a0c2fdeaebc76a05bace20799715203`. Their bytes remain unchanged.
The worker requires a fresh interpreter with none of those modules preloaded.
It checks standard import origins, then compiles the authenticated source bytes
instead of reading bytecode caches. The verified bridge callable is retained
through construction. Failed imports remove the modules inserted by this check.

The admission receipt records the exact frozen file inventory. The saved audit
requires that inventory, including its paths, sizes and hashes; older receipts
without it do not establish this source authentication. These checks assume a
controlled Python runtime, without hostile import hooks or concurrent code
mutation. They do not establish numerical parity or complete launch approval.

## Exact launch receipt

Launch approval covers one exact invocation, interpreter, packet location,
fresh output location and resource budget. A status label alone is insufficient.
`launch_schema` must be the integer `1`. The receipt retains the existing
implementation commit, WSL receipt digest and 21-symbol species order, and adds:

| Field | Required binding |
|---|---|
| `python_executable` | Absolute interpreter path, preserving the virtual-environment path |
| `python_sha256` | SHA-256 of that executable's bytes |
| `packet_root`, `attempt_root` | Canonical absolute paths for this attempt |
| `command` | Exact worker command below, with no interpreter options |
| `resources` | The fixed resource map below |
| `plan` | Local filename and SHA-256 of the reviewed execution plan JSON |
| `reviews` | Two distinct local filename/SHA-256 references for resolved implementation reviews |

The exact six command elements are `python_executable`, the absolute path to
`scripts/check_mace_parity.py`, `--packet-root`, `packet_root`, `--attempt-root`,
and `attempt_root`. The worker checks both `sys.argv` and `sys.orig_argv`, so
unapproved interpreter flags cannot disappear from the comparison.

```json
{"timeout_seconds": 1800, "cleanup_seconds": 10,
 "output_limit_bytes": 268435456, "log_limit_bytes": 16777216}
```

The plan JSON must contain exactly `implementation_commit`, `python_executable`,
`python_sha256`, `command`, `packet_root`, `attempt_root`, `resources`,
`wsl_admission_sha256` and `species`, with values matching the launch receipt.
Each review JSON requires an approved status, a distinct nonempty `reviewer_id`,
the same `implementation_commit`, the exact `plan_sha256`, and integer
`unresolved_findings: 0`. Approval references use only distinct regular local
filenames and cannot overwrite generated status, prediction, temporary or log files.
Place plan and review files beside the external launch receipt. The supervisor
copies the referenced records into its newly created output directory.

Before spawning, the supervisor checks the actual interpreter, command and
resource settings. Before construction, the worker also requires matching
RUNNING supervisor evidence and no prior result files. Final saved byte counts must be nonnegative integers within the declared bounds.
Production records must
explicitly contain `fresh_output: true` and `synthetic_command: false`.
Command-injected process tests are marked synthetic and cannot pass the saved
production audit.

The saved audit checks retained identities without opening the historical
interpreter or replacing it with the auditing host's interpreter. Trusted local
review records establish approval consistency; hashes and reviewer labels do
not provide a cryptographic signature of human authorization. Incomplete old
receipts remain historical evidence and do not satisfy the new launch contract.
No proposed admission or launch record becomes approved through these checks.
