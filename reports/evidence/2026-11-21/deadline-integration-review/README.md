Tested implementation: f550073eae1bd1b880ff04f8daee587c5412840b.

Author-run nine-file WSL suite: 297 passed, 3 expected skips, 45 warnings,
19 subtests passed, exit 0 in 263.39 seconds. Command, stdout, stderr, exit
status and tested commit are captured here with host paths redacted. This is
author-provided evidence; independent review remains pending. The private
packet/archive skips remain unchanged. No real checkpoint was deserialized,
real MACE calculator constructed or learned inference performed.

## Active shared deadline

The protected launcher's POSIX interval timer is armed before policy preparation
and remains active through source/receipt authentication, synchronous Git waits,
snapshot staging, import configuration and supervision. The existing worker
liveness checks use the same initial monotonic start and fixed 1,800-second
budget. Setup does not get a separate budget and is interrupted while blocked,
rather than classified only after it eventually returns.

A timeout emits terminal TIMEOUT JSON. Once a fresh attempt path has been
authenticated by the external policy/receipts, it also saves that receipt and
empty logs there. An earlier timeout emits evidence without writing to an
unverified path. Existing/racing directories are not overwritten. An interrupted
Git wait is killed/reaped. A deadline during worker handoff kills the owned
group even before its process handle reaches the monitor. Aborting execution
disarms the one-shot alarm before the existing bounded cleanup grace, preventing
a second interruption from cutting cleanup short.

Regressions exercise a 0.2-second budget with a 1.25-second source delay, a
pre-authentication delay, a blocking synchronous Git subprocess whose PID must
be reaped, and a delayed import-configuration phase after successful staging.
Existing timeout, termination, overflow and descendant-cleanup regressions pass.

The separate setup-deadline-comparison.json records an old/new measurement:

| Revision | Budget | Injected delay | Caller elapsed | Reported execution elapsed |
| --- | ---: | ---: | ---: | ---: |
| Previous 3def934 | 0.2 s | 1.25 s | 1.994 s | 1.841 s |
| Candidate f550073 | 0.2 s | 1.25 s | 0.275 s | 0.200 s |

Both attempts return TIMEOUT without launching a worker. The candidate
interrupts setup at the shared deadline; caller elapsed also includes process
startup and evidence persistence. The pinned reproduction harness uses only
source fixtures, no calculator or checkpoint:

```bash
"$python_bin" reports/evidence/2026-11-21/deadline-integration-review/reproduce-setup-deadline.py
```

## Actual protected-loader integration with inert data

The positive integration case invokes the external protected launcher under
-I -S, including its actual private supervision, worker fork, source/receipt
reauthentication, liveness checks, private loader and unchanged _run_parity().
The worker source is asserted byte-for-byte equal to the repository source.
No parity function, private loader, live-admission check or liveness function
is replaced in this subprocess.

The synthetic packet has 974 nonperiodic 21-atom frames, all five required
connection groups with the fixed counts, an eleven-member hashed manifest,
saved float64 reference predictions and plain non-model bytes. Extxyz storage
is reversed and prediction storage independently rotated to test ID/order joins.
The synthetic policy/admission inventories bind actual platform/package/source
identities and the synthetic source commit; immutable accepted bridge/evaluator
baseline objects are retained. Actual admission, source, packet, symbol-order,
copy/hash and receipt checks run without stubbing.

The only calculator replacement is an analytical ASE test double placed in the
synthetic Git tree's constructor body before its source commit and policy are
pinned. Fixture identity constants bind the inert manifest/model bytes. This
does not add a production injection option. The constructor asserts that neither
MACE nor Torch was imported and that its snapshot contains the inert byte marker.
It records every construction and force/energy accessor request independently
of the parity routine's counters. No learned model is loaded.

The captured inert-integration-summary.json verifies:

- exactly one observed construction and 974 force plus 974 energy requests;
- 974 records in the approved prediction order, with the event trace matching it;
- actual PARITY_PASS with zero absolute energy/force difference for this inert case;
- matching final and partial saved arrays, float64 dtype, receipt/prediction hashes,
  unchanged inert snapshot hash after construction and prediction;
- actual private supervisor WORKER_COMPLETED with exit code zero.

These are synthetic pipeline results, not reproduction of the saved learned
MACE predictions or the project's reported scientific scores.

External launcher candidate SHA-256:
74f9ff15982c790aab75e19676f34ff771513c1a950c2d18992fb5619cdac818.

The historical WSL proposal remains unchanged and PROPOSED. Protected
provisioning, implementation review acceptance, platform admission and explicit
bounded-launch authorization remain required before any real-model execution.
