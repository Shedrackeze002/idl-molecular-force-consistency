Implementation tested: 0a8632794d1d39173545d29af674fd4cfd0464ca.

The expanded nine-file WSL suite returned 245 passed, 3 expected skips,
45 warnings, 19 subtests passed, exit 0 in 104.67 seconds. Raw captured stdout,
stderr, command and exit record are retained with host paths redacted.

The external launcher imports only Python standard-library modules. It refuses
to serve as a trust root from inside the mutable checkout. An independently
provisioned external policy pins the launcher, approved commit, selected Python,
receipt digests and fresh execution directory. It authenticates all project
source bytes before startup, then executes a private read-only snapshot made
from captured verified bytes. Only required immutable Git objects are staged;
no Git checkout filter, source hook, upload-pack or project module executes
during verification. Receipt bytes are captured and retained privately too.

Seventeen external-boundary tests cover positive startup through a synthetic
bootstrap/supervisor/worker chain, tampered bootstrap/worker/supervisor/loader/
frozen adapter, incorrect commit/policy/receipt/launcher digests, in-checkout
launcher refusal, existing-directory preservation, startup hooks, post-check
source changes, audit snapshot tampering and source-inventory consistency.
All tests use synthetic receipts/repositories and no real model artifacts.

Launcher candidate SHA-256:
565051c8a04f63b28d29d2d39f9488b07d02df08c41c1b40bba641702eb87cce

Deployment instructions are in docs/run-mace-parity.md. The policy template has
an intentionally unapproved status and cannot authorize execution. External
administrator/coordinator provisioning and independent digest verification are
required; this report does not claim they have occurred. The trusted Python/Git
installations and the separately protected launcher/policy are explicit trust
assumptions. Same-account or privileged writers must not be able to alter the
private execution directory during an attempt.

The historical WSL proposal remains unchanged and PROPOSED pending review of
this new boundary. No platform admission or real-model launch is claimed.
