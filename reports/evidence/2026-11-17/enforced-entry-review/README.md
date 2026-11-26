Tested implementation: f04e37c1311e123b545e710cf37e212c57945e51.

The full nine-file WSL suite returned 250 passed, 3 expected skips, 45 warnings,
19 subtests passed, exit 0. Automatically captured stdout/stderr/exit and command
are retained alongside the tested commit receipt with host roots redacted.

The working-tree bootstrap now unconditionally refuses direct script invocation
for worker, supervisor and audit. It exposes a dispatch API that the separately
provisioned external launcher calls only after verifying the snapshot's raw
source bytes. The supervisor's approved worker command also uses that external
launcher; every worker startup revalidates the retained snapshot and pinned
receipts. The reviewed launch schema binds the external launcher and policy paths.

The independently pinned policy digest is explicit on the initial invocation and
inherited by the verified supervisor's worker. Omitting it refuses before project
startup. This avoids a circular hash between a policy and its launch receipt.

Three new subprocess regressions modify worker, supervisor and auditor entries
and invoke the bootstrap directly; none writes its marker. Additional tests prove
the external worker command still works and a changed execution copy is refused
before worker startup. Existing startup-hook and import-shadow probes exercise
the dispatch API as library tests, separately from the production trust gate.

External launcher candidate SHA-256:
c9c6b6600b4a6b75ba4044774e2662d02bcd1a86eb6b0a8f70605abf60b2ed6c

The external launcher and its pinned policy still require protected coordinator
provisioning and approval. The historical WSL proposal remains PROPOSED and is
unchanged. No real checkpoint loading, model construction or inference occurred.
