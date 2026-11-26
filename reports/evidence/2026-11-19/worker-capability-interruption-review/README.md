Tested implementation: fc64c12dde98a2b4b4bf1879d08bd7ee3375c542.

Author-run nine-file WSL suite: 279 passed, 3 expected skips, 45 warnings,
19 subtests passed, exit 0 in 173.95 seconds. Command, stdout, stderr, exit code
and tested commit are retained here with public host paths redacted. This is
author-provided regression evidence, not an independent reviewer run. The
private-packet/archive skips remain unchanged. No real checkpoint was loaded,
no real MACE calculator constructed and no learned prediction requested.

The previous revision blocked the external worker option but did not protect
the checker's own isolated CLI/main() or parent interruption. This revision
adds guards to those paths and tests the actual trusted handoff, rather than
inferring protection from a non-isolated refusal test.

The trusted child installs a private closure in its verified loader and passes
it to the checker entry. The closure binds worker PID, live supervisor PID,
owned session, approved packet/attempt paths and the shared deadline. Neither
receipts nor flags/environment values mint this authority. Checker CLI startup
requires the capability before dependency imports. Checker main(), run_parity(),
the public load_admitted_arm() and the real constructor require authorization;
the parity loop rechecks it before every record. The parent loader stays
unauthorized because the closure is installed only after fork. No capability
factory or worker dispatch helper is exported from the trusted main().

Capability coverage includes ordinary and -I -S checker invocation, isolated
import-and-call main() with a synthetic parity marker, public execution API
refusals before packet/model access, a positive real checker/main() path with
only parity computation replaced by a synthetic marker, wrong-path refusal and
refusal when another fork tries to reuse the authorized worker's capability.
Existing fake-constructor library tests explicitly scope their synthetic
authority substitutions; they do not constitute production admission.

Supervisor SIGTERM, SIGINT and SIGHUP become recorded INTERRUPTED outcomes.
Cleanup terminates and reaps the owned group, including TERM-ignoring
descendants, and does not let repeated terminal signals abort cleanup. Startup
signals are deferred until the worker is assigned to cleanup state. The child
restores ordinary signal handlers and its unblocked mask. Prior supervisor
handlers are restored after supervision. Final receipt persistence is also
within the interruption/error handling scope. Cleanup that cannot remove the
group is recorded as CLEANUP_FAILED rather than successful completion.

Real-handoff regressions send all three signals to the supervisor, observe
beyond the reviewer's 1.5-second delayed marker window and assert no live worker
or completion marker remains. Other regressions interrupt the spawn ownership
window and send a second signal while a TERM-ignoring descendant is being
cleaned up. The auditor rejects INTERRUPTED/CLEANUP_FAILED receipts even with
apparently successful parity output and exit code zero.

An additional regression first failed during self-review: an unrelated monitor
error followed by the first terminal signal could abort exception cleanup.
Cleanup now enters a signal-protected phase regardless of its initiating
reason. That regression passes in this final suite and checks reaping plus
restoration of the caller's signal handler.

External launcher candidate SHA-256:
d351c704289f82c1ca0d0bc2ad214fbaf59ca631a0667b99cc167fd3c8d41f8b.

The historical WSL proposal is unchanged and PROPOSED. Protected provisioning,
implementation review acceptance, environment admission and explicit bounded
launch authorization remain separate prerequisites for real-model execution.
