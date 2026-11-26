Tested implementation: 6e81a960d57bd494a2b4d45447ac60db20dfd6fe.

Final author-run nine-file WSL suite: 288 passed, 3 expected skips, 45 warnings,
19 subtests passed, exit 0 in 172.69 seconds. The command, stdout, stderr,
exit status and tested commit are captured here with host paths redacted.
This evidence is author-provided; independent review remains pending. No real
checkpoint was deserialized, MACE calculator constructed or learned prediction
requested. The private-packet/archive skips remain unchanged.

## Previous revision reproductions

The three latest review findings were reproduced against immutable source
bytes from 83842ef97b112b5d88d5cde5ef4f033932462923 in a temporary source tree.
The old synthetic handoff tests were extended with SIGQUIT and two regression
checks. The result was the expected 3 failed, 3 passed (pytest exit 1):

- assigning a no-op _worker_capability let isolated public checker main() run
  a synthetic parity marker and return success;
- SIGQUIT killed the old supervisor while its detached synthetic worker
  survived and could write the delayed marker;
- SIGTERM immediately after the initial status write left RUNNING in the receipt.

SIGTERM, SIGINT and SIGHUP were passing controls for the old handoff.
These expected failures are recorded separately in previous-head-reproductions.*
(host paths redacted and trailing whitespace trimmed);
they are not failures of the candidate implementation. The reproduction harness
is retained as reproduce-previous-head.py. From the repository root, run it
with the dedicated WSL test environment:

```bash
"$python_bin" reports/evidence/2026-11-19/private-driver-review/reproduce-previous-head.py
```

## Candidate changes and coverage

The mutable callable authorization mechanism is removed. Checker CLI/main(),
public run_parity(), public load_admitted_arm(), the former project constructor
and the compatibility require_supervised_worker() path refuse execution.
Setting the former module/global permission fields cannot enable them.

Only the separately provisioned external launcher's private supervised child
owns the real MACE constructor and the production arm provider. After policy,
receipt and retained-source verification, it loads captured checker bytes as
a numerical library and invokes the internal routine with its own providers.
Neither constructor nor production provider is installed in a project global.
Private liveness checks bind the worker PID, live parent, owned session,
approved packet/attempt paths and the shared deadline. Origin checks and all
CPU/float64 constructor settings are retained. The numerical/adapter helpers
accept explicit fake providers in library tests; those helpers do not supply
the real constructor or grant launch authority.

The exact isolated bootstrap configure_imports() -> forged _worker_capability
-> checker main() regression now refuses before parity, loader or constructor
markers. No-op and fake-object forgeries are covered. Public execution API
refusals are also tested with the former compatibility guard replaced by a
no-op, while positive verified private-driver execution still works. Fake
provider integration tests cover the unchanged loader/parity/audit pipeline.

SIGQUIT joins SIGTERM/SIGINT/SIGHUP consistently across supervisor handlers,
deferred startup signals and child signal reset. Actual trusted-handoff tests
verify interrupted status, reaping and absence of the delayed marker for all
four signals. Existing repeated-signal and TERM-ignoring descendant cleanup
regressions continue to pass.

Fresh-directory creation, log creation and the first RUNNING write are now
inside interruption/status handling. Directory ownership is established with
signals deferred, so an error never overwrites a pre-existing/racing attempt.
Tests inject each of the four signals immediately after the initial write and
require terminal INTERRUPTED status with empty retained logs and no worker.
The production synthetic handoff tests that boundary too. A recoverable first
write error records SETUP_FAILED. Finalization and cleanup error coverage from
the previous revision is retained; the auditor rejects interrupted outcomes.

External launcher candidate SHA-256:
7131419cec1699a6f99b6650f57dc308bbbc6134366b2908104faab06c284bea.

The historical WSL proposal is unchanged and PROPOSED. Protected provisioning,
implementation review acceptance, environment admission and explicit bounded
launch authorization remain prerequisites for real-model execution.
