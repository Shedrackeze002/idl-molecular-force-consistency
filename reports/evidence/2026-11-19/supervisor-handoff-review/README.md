Tested implementation: 142b977be96a8bc2a924a529138bb7d697a3d7af.

The nine-file WSL regression suite returned 262 passed, 3 expected skips,
45 warnings, 19 subtests passed, exit 0 in 155.75 seconds. The command, stdout,
stderr, exit status and tested commit were captured automatically. Public host
paths are redacted. The skips remain the unavailable private-packet/archive
checks; no real checkpoint was deserialized, constructed or used for prediction.

The external launcher now accepts only supervisor and audit entries. Its
prepare() API also refuses worker selection before project startup. The worker
dispatcher and single-use handoff exist only inside the verified launcher's
main(), not as importable dispatch helpers or environment/CLI permissions.

Only the verified supervisor receives the handoff. It records RUNNING status,
validates launch identity, copies retained receipts and opens stdout/stderr logs
before calling it. The child forks in a new owned process group and rechecks
the pinned policy, receipts and complete retained source tree before executing
captured worker bytes. The existing supervisor monitors that group, including
its trust checks, with the shared deadline and output/log limits. Production
requires the approved WSL/POSIX runtime; native Windows must use WSL.

The synthetic boundary fixture now uses the real bootstrap, supervisor and
launch validator with a model-free worker. It checks RUNNING status, open logs,
isolation and process-group ownership before its worker marker. Regressions
cover direct worker refusal before staging, refusal after staging with the
independent policy digest but no supervisor status, prepare() refusal, missing
supervisor handoff, and real-supervisor timeout, log and directory overflow.
Shortened bounds in overflow/timeout fixtures are synthetic committed sources,
not changes to the production contract. Production remains 1,800 seconds,
256 MiB per attempt, 16 MiB per log and 10 seconds for cleanup.

The approved launch command now identifies the supervisor entry, consistently
across the trust root, loader and auditor. All three entry refusal messages and
the bootstrap configuration refusal point to the external-launcher workflow.
The run guide explains configuration-only bootstrap use and private worker
dispatch. Refusal-message subprocess regressions check the supported commands.

External launcher candidate SHA-256:
c89d6fe6bec88f7d83933206d7211bef572bee2997274c8b3e1e39d4a873d1a6.

The historical WSL proposal remains unchanged and PROPOSED. This evidence is
not environment admission or model-launch approval. Protected provisioning,
review acceptance, environment admission and explicit bounded-launch
authorization are still required before real-model execution.
