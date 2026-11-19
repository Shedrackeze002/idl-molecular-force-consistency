Tested implementation: 73787b0cd61a92b0041a408e4dfaf12e412d4806.

Author-run nine-file WSL suite: 292 passed, 3 expected skips, 45 warnings,
19 subtests passed, exit 0 in 149.56 seconds. The command, stdout, stderr,
exit code and tested commit are retained with host paths redacted. This is
author-provided evidence, not an independent review or launch approval. The
private-packet/archive skips remain unchanged. No real checkpoint was loaded,
real MACE calculator constructed or learned prediction requested.

## Reproductions against the previous revision

Both latest findings were reproduced against raw immutable source from
f812be6a40c9008715685cee8dfdea24245a8f4f. The regression cases are extracted
from the candidate's pinned test source and run against a temporary old tree.
Both fail there as expected (pytest exit 1), independently of the passing
candidate run:

- Under -I -S, a forged __trusted_worker_spawn__ callback, NOT_APPROVED launch
  and WSL receipts, and a mismatched WSL digest let old supervisor main()
  invoke the callback in its own PID and record WORKER_COMPLETED without a child.
- With admission checks isolated using synthetic data, a direct old loader
  call invokes the supplied constructor without any supervised liveness gate.
  The constructor writes a marker and raises; only synthetic non-model bytes
  are copied, with no deserialization or learned computation.

The separate previous-head-reproductions.* files retain the expected failures
(host paths redacted and trailing whitespace trimmed). Reproduce them from a
repository clone containing the old and candidate commits with the WSL test
environment:

```bash
"$python_bin" reports/evidence/2026-11-19/closed-runtime-review/reproduce-previous-head.py
```

## Closed production execution paths

All supervision code now lives inside the protected launcher's main(). It
captures the verified worker factory in a lexical closure after policy,
receipt, commit and source authentication. The project supervisor script,
main(), supervise() and _supervise() always refuse, including under -I -S.
No project global is read as a privileged handoff and no project callback
argument can enable supervision. Captured project namespaces receive no
runtime permission object. The external supervisor/audit operator commands
and existing resource limits are unchanged.

The entire production loader is now a private child closure too. Its fixed
real constructor is not a parameter or a project global. It checks private
worker/parent/session/path/deadline liveness before packet access and again
immediately before construction. Packet/approval validation, species order,
private snapshot copy/hash, import-origin checks, constructor settings and
retained receipts are preserved. The former importable _load_admitted_arm()
injection API always refuses before packet/snapshot access or constructor
invocation, even with no-op compatibility guards or synthetic approved data.

Constructor fakes exist only in tests/mace_test_runtime.py. That test-only
harness extracts actual private function ASTs into a disposable namespace
and supplies fake dependencies there; it does not duplicate the implementation.
Production neither imports that file nor exports a runtime factory. This
keeps the existing loader, supervisor and loader-to-auditor tests on the real
runtime code without leaving a production constructor-injection seam.
The harness is included in the reviewed source inventory and pinned to LF
for raw-byte compatibility with Windows/WSL checkouts.

## Candidate coverage

The exact forged-callback/unapproved-receipt regression now refuses before
callback invocation or attempt creation. Direct supervise()/_supervise()
calls are also rejected. The direct injected-loader regression refuses before
the constructor marker or attempt/snapshot creation. Positive external launch,
real numerical-library dispatch, source/receipt tamper refusal, PID/path
liveness, all four terminal signals, spawn interruption, initial RUNNING write
interruption, error/repeated-signal descendant cleanup, partial prediction
handling and saved-output audit regressions continue to pass.

A stale library startup-isolation assertion initially expected direct
supervisor --help to succeed; it now expects the intended refusal while still
verifying that startup/dependency shadows never execute. The final full suite
was rerun against the committed correction above.

External launcher candidate SHA-256:
9492f500f2c61d26a01b0a7e5c33d00a1ddedcc2631972ea66a8a70dff401d48.

The historical WSL proposal remains unchanged and PROPOSED. Protected
provisioning, implementation review acceptance, environment admission and
explicit bounded-launch authorization remain required for real-model execution.
