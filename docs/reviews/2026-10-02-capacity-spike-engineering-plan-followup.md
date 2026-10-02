# Saved capacity spike diagnostic: engineering plan followup

Verdict: **PASS** for the saved Amendment A and final clarifications. The two required repairs in the initial engineering review are resolved. This clears the plan for implementation; it does not establish correctness of code that has not yet been written or validity of unexecuted numerical results.

Reviewed plan SHA256: `eb4bf915c5ca7a1efc1a7b90246a40ef8b59bd98afbf4a695cbc073ea7377e1e`.

Reviewed design SHA256: `d9c5b564ba0771c17f7c94cbf41a7d553e8838ec39b55c657cb932a607feb96b`.

## Repair verification

E1 is resolved by explicit script-local loops around existing frozen kernels rather than broad helpers without completion hooks. Before every microbatch, the loop records an attempted/in-flight call and checks budget; after readiness, it records known returned work, then validates finiteness before recording completed-valid work. Thus a returned nonfinite result remains known executed work while an interrupted call without a ready result remains unknown. The plan includes injected interruption/finite-failure tests, independent unweighted and fixed-weight joint kernels, plus/minus FD instrumentation, and retention of completed derivative steps. No participating `src/` edits are planned.

The fixed complete-run totals reconcile independently: eight states times nine prediction microbatches gives 72 calls and 1152 role entries; eight states times four gradient passes times eight fitting microbatches gives 256 calls and 4096 structure exposures; eight states times two objective directions times two step sizes times two signs times eight fitting microbatches gives 512 vector-value calls and 8192 structure exposures. Each FD vector call evaluates both objective components without doubling its exposure count. These categories are explicitly distinct from training and unique structures.

E2 is resolved by a single total 1800-second internal budget including a 60-second output reserve. The budget starts immediately after exclusive root/ordinal-1 reservation, before guards, numerical imports and compilation. No new numerical call starts after 1740 seconds. An overrun retains actual observations/work and invalidates subsequent complete-run interpretation; the separate 1860-second supervisor can terminate a hung process while retaining unknown in-flight work. Subsequent cases remain blocked and no numerical retry is allowed.

The amendments also resolve the earlier implementation-specific cautions: the energy-bias identity contains the atom-count factor; weighted component and unweighted scaled-loss slopes have explicit direction/scales; all decoded state fields are hashed on successful and failed paths; a global source-integrity failure invalidates every scientific endorsement; per-case validity is separate from complete-run and paired contrast validity. The FD zero threshold and fallback seed are fixed.

## Remaining implementation and execution gates

The implementation review must verify exact source/script/config/review manifest completeness and no numerical import before guards, historical selected receipt binding, true ready/returned versus valid/in-flight counters, failure-path state audits and cap behavior. It must check these against code and meaningful tests rather than rely on this plan verdict. Numerical launch remains gated on that review and frozen prelaunch identities. After execution, interpretation remains gated on saved-result verification and all relevant source/case/run checks.

No implementation, parameter initialization, prediction, derivative evaluation, training, optimizer work or raw-state edit occurred during this followup. The initial failed read-only path inspections remain preserved in the first engineering review. This followup used only read-only document/hash inspection and writes this review record.
