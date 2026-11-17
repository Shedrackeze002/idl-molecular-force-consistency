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
`-I -S scripts/mace_bootstrap.py --entry worker`, `--packet-root`, and `--attempt-root` argv used by
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
raw live worktree bytes are compared with immutable objects derived
from the pinned implementation and baseline commit trees; refreshing a receipt's
self-reported hashes cannot admit edited source. Missing, substituted or
contradictory launch/admission fields are refused before model construction.

The worker and auditor check NumPy and ASE origins before importing them.
Admission pins both MACE package initializers and checks package search paths
and the exact constructor module before construction.

Synthetic tests do not require or load the checkpoint:

```bash
PYTHONPATH=.:src:scripts python -m pytest -q \
  tests/test_mace_loader.py tests/test_mace_parity.py \
  tests/test_mace_parity_supervisor.py tests/test_mace_parity_audit.py \
  tests/test_mace_reference_adapter.py -p no:cacheprovider
```

## External trust root (required before any project startup)

The supported production entry point is a coordinator-provisioned copy of
`trust/mace_trusted_launcher.py` OUTSIDE the mutable checkout. Executing the
working-tree bootstrap directly is not an authorized production launch.

An administrator/coordinator must first review the launcher at the approved Git
commit, export its bytes with `git --no-replace-objects cat-file blob
<approved-commit>:trust/mace_trusted_launcher.py`, and compare its SHA-256 with
the independently supplied reviewed digest using a trusted system tool. Install
that exact copy, the approved policy and pinned receipts in a directory the
experiment account cannot modify. Do not copy them from unverified working-tree
files or obtain the policy digest from the same mutable checkout. Trust in this
external copy and the Python/Git installations is the explicit starting
assumption; a script cannot authenticate a malicious replacement of itself.

The coordinator-owned policy uses schema 1, an APPROVED/ACCEPTED status, and:

- `checkout_root`: absolute mutable source checkout;
- `execution_root`: a distinct fresh private directory outside that checkout;
- `implementation_commit`: the full reviewed commit, also pinned by both receipts;
- `baseline_commit`: `3fa5969e5a0c2fdeaebc76a05bace20799715203`;
- `python_executable`: the exact dedicated-venv executable (preserve its venv path);
- `trusted_launcher_sha256`: reviewed digest of the external launcher;
- `launch_receipt`, `launch_sha256`: external approved launch JSON and its digest;
- `wsl_admission_receipt`, `wsl_admission_sha256`: external approved WSL JSON and digest.

The approved launch's command must point to
`<execution_root>/scripts/mace_bootstrap.py`, not the original checkout. All
packet/attempt paths remain explicit approved absolute paths. The execution root
and attempt root must be fresh; failed staging is retained and requires a new
reviewed decision rather than a retry over the same directory.

Before importing or executing any project entry, the external launcher verifies
its pinned policy/receipts, the approved Git commit and every implementation and
frozen source file against raw Git blob bytes. Git replacement refs, user/global
Git configuration and checkout filters are excluded. It stages captured verified
bytes into the private execution directory, imports only the needed immutable
Git objects, and retains private read-only copies of the pinned receipts. It
never executes source files from the original checkout. The worker therefore
cannot change between the trust check and startup through a later source edit.
All runtime modules in the execution copy are read-only; protect this private
directory from other writers during the attempt.

The trusted setup time is included in the supervisor's existing shared deadline.
Audit uses the retained execution copy and rechecks its source bytes and receipts.
The mutable checkout may have changed after execution without changing the
recorded execution copy. No project import is needed to perform these checks.

Only after provisioning and all approval gates, invoke the EXTERNAL launcher:

```bash
"$python_bin" -I -S "$trusted_launcher" --entry supervisor \
  --policy "$approved_policy" --policy-sha256 "$coordinator_policy_sha256"
"$python_bin" -I -S "$trusted_launcher" --entry audit \
  --policy "$approved_policy" --policy-sha256 "$coordinator_policy_sha256"
```

The worker authenticates 11 packet members, validates all 974 inputs before
construction, copies and hashes the primary final EMA model, constructs one
CPU/float64 calculator, then makes exactly 974 force and 974 energy accessor
requests. Acceptance is absolute-only (`rtol=0`, `atol=1e-10`) for energy and
force components.

Production uses a dedicated virtual environment and the external launcher, then
the verified isolated bootstrap for all entries. `-I -S` ignores PYTHONPATH, user sites, sitecustomize,
usercustomize and executable .pth hooks. The bootstrap adds
a restricted mapping for the exact loader, bridge, worker and evaluation adapter
modules before the selected environment's site-packages. Project directories are
never added as general import paths, so local e3nn or other dependency files
cannot override installed packages. Dependency origin checks still apply. Synthetic pytest
commands above exercise library interfaces and do not authorize a production run.

The supervisor uses one fresh directory, a shared 1,800-second cap, 256 MiB
directory bound and 16 MiB per-log bound. Timeout, overflow, partial output,
identity refusal and numerical disagreement are retained and never retried or
relabeled as success. The independent audit reads saved arrays only; it does
not import or load MACE/Torch.
