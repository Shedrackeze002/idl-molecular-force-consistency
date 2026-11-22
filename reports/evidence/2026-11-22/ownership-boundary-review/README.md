# Worker and attempt ownership review fixes

Tested implementation: `4c94765610b9201965dc4b545fb8af2a2298e47b`.
Previous reviewed head: `ffa7ae3b4945cd18aa94bfe507e03b0711ad65d5`.
Review: [6012834925](https://github.com/Shedrackeze002/idl-molecular-force-consistency/pull/18#pullrequestreview-6012834925).

## P1: continuous worker ownership

The protected launcher masks SIGALRM and the four terminal signals only while
acquiring resources and recording ownership. The parent registers its worker
handle before unmasking after fork. That private registry survives readiness,
factory cleanup and return, so the supervisor can stop/reap the child even if
the caller's local assignment never happens. Readiness itself is interruptible.
Cleanup also kills an owned child interrupted before setsid establishes its
process group. The child resets the inherited alarm and terminal handlers and
unblocks those signals before worker execution.

The actual external-launcher regressions queue real signals immediately after
fork (before handle registration), and in the factory's final cleanup before
the return reaches the caller. Both boundaries run under SIGALRM, SIGTERM,
SIGINT, SIGHUP and SIGQUIT. All ten cases retain terminal status, a non-null
exit code and a reaped child. The early-fork case delays setsid deliberately.
A separate readiness case delays the ready byte for 1.25 seconds and arms an
alarm for 0.15 seconds inside the readiness phase; it times out promptly and
reaps the child. Another case checks the worker's actual unblocked signal mask.

## P2: directory acquisition and registration

The same narrow mask makes successful mkdir, the local created flag and the
private owned-attempt registration atomic with respect to signal delivery.
The five-signal regression interrupts immediately after successful mkdir and
verifies retained terminal status and empty stdout/stderr logs. SIGALRM records
TIMEOUT with worker_started false, without an evidence persistence error.

A competing-owner regression creates a directory and sentinel before the
supervisor's mkdir, queues SIGALRM, and makes that mkdir fail with EEXIST.
Timeout handling emits terminal JSON, reports the persistence refusal and
leaves the independent sentinel untouched, without adding logs or receipts.
The existing pre-existing-directory refusal tests also remain in the full suite.

## Validation

- Full WSL regression suite: **315 passed, 3 expected skips, 45 warnings,
  19 subtests passed**, exit 0 in 193.62 seconds.
- Captured focused ownership suite: **18 passed**, exit 0 in 8.98 seconds.
- ownership-regressions.json retains summaries of 16 terminal receipts: ten
  worker handoff cases, five mkdir cases and the readiness deadline case. All
  eleven worker cases additionally confirm that the child PID no longer exists.
  The independent-directory and restored-signal-mask cases are covered by the
  focused suite without claiming an owned failure receipt.
- source-scope.json compares ASTs with the previous reviewed head: private
  construct/load_arm/check_liveness, prepare, dispatch and OwnedWorker are
  unchanged. The historical WSL proposal is unchanged.

The command, exit code and stdout/stderr files accompany both captured runs.
The full suite includes the existing actual private-loader / unchanged parity
integration with 974 inert records. It passed alongside all prior deadline,
interruption, descendant cleanup, overflow, admission and audit regressions.

This is author-run WSL evidence; independent review remains pending. Tests use
synthetic Git trees, synthetic receipts and inert data only. No real checkpoint
was deserialized, real MACE calculator constructed or learned inference run.
The historical WSL proposal remains PROPOSED. These fixes do not grant platform
admission or authorize a real-model launch.

Trusted launcher SHA-256:
`dfa1cf54ffb6adc2e5e829eae531019a7762193a7dfa63513918ec323a1b2748`.
