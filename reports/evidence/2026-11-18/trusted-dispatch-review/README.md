Tested implementation: dd2f3637c440e7a38364bcd944091d6a8a9a534c.

The nine-file WSL suite returned 253 passed, 3 expected skips, 45 warnings,
19 subtests passed, exit 0 in 122.52 seconds. The command, stdout, stderr and
exit status were captured automatically; public host paths are redacted.

The bootstrap now contains import configuration only. Its CLI and importable
main() both unconditionally refuse; it contains no runpy or entry-selection
code. The old BOOTSTRAP_DISPATCH command has been removed from production.
The separately provisioned external launcher's main() first verifies the
external policy, receipts, approved Git revision and complete raw source
inventory. Only after success does it select an entry, configure imports and
execute captured verified source bytes. It does not re-read an entry or pyc
file for dispatch. Worker reauthentication through the external launcher is
retained.

Three new subprocess regressions reproduce the reviewer's importlib load and
module.main() call, with tampered worker/supervisor/audit entries. All refuse
without a marker, even with a policy-looking environment value. Existing
positive external-launch tests now use the real configuration-only bootstrap.
Library-level import-resolution probes select their synthetic entries in test
code; this is not a production bootstrap dispatch API.

External launcher candidate SHA-256:
61d3f8801605842e83cac46bdcde67206f0a17c6517ee93db46e10772f8f13f6

The historical proposal remains unchanged and PROPOSED until review acceptance.
Protected external provisioning, WSL admission and bounded-launch authorization
are still separate requirements. No real model or checkpoint was executed.
