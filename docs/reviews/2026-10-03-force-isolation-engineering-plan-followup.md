# Force-isolation engineering plan amended followup

Verdict: **PASS** for the final amended specification and implementation plan below. No material engineering issue remains in this bounded amendment. The initial review and its saved-only audit evidence remain unchanged; this followup extends their plan verdict to the amended bytes. Implementation and numerical execution remain UNEXECUTED.

| Artifact | Final SHA-256 |
|---|---|
| `docs/superpowers/specs/2026-10-03-force-isolation-design.md` | `1a61f0087b0baf147d07e8a5e5ed5dca02199910195bcf21f15e58132625ba21` |
| `docs/superpowers/plans/2026-10-03-force-isolation.md` | `dbac79dd3d57f5599094cb8abe9e13661d5e596ecfcfb53af64c359b0ab5b1ce` |
| `reports/execution/force-isolation-2026-10-03/amendment-A.json` | `e60cafd5c55106d2ae265ac26b604c8c6fa44f548254728f717ccce1be2d6846` |
| Unchanged `proposed-settings.json` | `48d6bd6070904d348f250e1a9e8fd4e09cfe0fe44685af9b195e4c749fcfe2f8` |
| Unchanged `source-pins.json` | `eda3ae20c34d16a5fea4260d6a15a5ac31356e2a384df00e4dd08e29a0e30664` |
| Initial engineering review | `ee2a0c551a074e4a5e0691059e4d7685646c18ccb84c50c49ae8f89b1ac28f39` |

The preserved pre-amendment copies exactly match the initial reviewed spec SHA `ddec7093b80801e3a70f16a9db8624d4d249278f4146999fa74ed0664ea4c38c` and plan SHA `496ded0971ce8af638f3372fb48c67aedcfd03cf2381a2e47f36213fd64d86da`.

The actual diff adds one specification paragraph at line 55 and one Task 2 verification checklist item at line 51. Both explicitly require a nonzero readout kernel, a strictly nonzero shared-body force-gradient norm, actual nonzero incoming body Adam moments and retained per-block norm/assertion evidence for the energy-label independence and gauge-invariance fixtures. The plan additionally states readout-kernel trainability, scalar-bias-only projection and rejection of the initial zero-head state alone as an invariance witness. No scientific settings, restoration rule, source membership, production objective, tolerance, score grid, budget, invocation count, architecture, gate semantics or closeout scope changed.

This is a sound verification refinement. The actual model initializes the readout kernel to zero when configured that way (`src/force_consistency/models/energy.py:19-22`), so checking body-gradient equality only at that initial head can pass trivially. A nonzero readout alone does not prove body sensitivity; requiring and retaining a nonzero body force-gradient norm closes that gap. Nonzero incoming body moments ensures that the moment-preservation/Adam assertions exercise an active optimizer state. Per-block evidence makes the tested parameter support inspectable and distinguishes the shared body/readout from the intentionally masked scalar bias. The amended fixture requirements are compatible with the private test engine and the existing nonzero-kernel/nonzero-moment fixture pattern (`tests/test_offset_profile_core.py:60-76`), while leaving the once-only production restoration and zero-source identities intact.

The amendment record also preserves an append-only literature-navigation correction: composition learning is section 3.3, rather than 4.3. The actual spec/plan diff does not expand literature claims or change the experiment. That correction introduces no engineering dependency or new numerical work.

I independently checked final hashes, exact bounded diffs, unchanged settings/source-context/map identities, preserved initial documents and all artifacts covered by the original engineering evidence manifest. The original 18,963-source/control-array audit was not repeated, as explicitly scoped by the followup request; its recorded PASS and limits remain the supporting evidence. No model call, training, prediction, initialization, test suite, child launch, implementation, Obsidian action or old-source mutation occurred.

The initial engineering review's interface, pure-import, joint-row/force-gate, acyclic hash graph, participant coverage, once-only child, ownership, final durable-write, complete work-audit and saved-closeout acceptance points remain applicable. Future tests must demonstrate these nonvacuity assertions; this document does not claim that they have run. Final focused/full-suite evidence at final code bytes, an independent implementation followup, the immutable launch gate and the scientific amended-plan verdict remain separate required gates. Primary can now assess the two final plan followups before authorized implementation.
