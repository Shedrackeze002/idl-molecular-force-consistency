# Two independent reviews of the baseline implementation plan

Date: 29 September 2026

Reviewed plan: `docs/superpowers/plans/2026-09-29-data-baseline-implementation.md`.

Both the scientific and engineering subagents first reviewed the design independently. After the user-approved design was revised, both independently reviewed the implementation plan, returned actionable findings, and then re-reviewed the corrected plan. Neither wrote implementation code or changed project files.

## Initial plan findings and corrections

| Review | Finding | Verified correction |
| --- | --- | --- |
| Engineering | Duplicate candidate screening lacked an executable conservative bound. | Species-pair sorted distance descriptors use an exact Euclidean radius query of `N*epsilon + 1e-10`; both reviewers checked the bound, and the plan requires boundary tests and final chemical alignment. |
| Both | Near geometries can have legitimate label differences beyond serialization precision. | Exact stored-coordinate label checks are separate from transformed/near-pair diagnostics; near relationships still exclude leakage. A stiff conservative-potential test covers the distinction. |
| Scientific | A shared minimum's role in connection-averaged validation was ambiguous. | Primary validation uses branch/TS records only. Unique eligible reference minima occupy a separate diagnostic partition, with explicit precedence and no checkpoint-selection role. |
| Engineering | Unresolved branch destinations did not explicitly block pilot completion. | Model-ready export and milestone completion now require unique branch/reference-class mappings; raw audit delivery remains possible when mapping is unresolved. |
| Engineering | A 90% loss-reduction criterion did not establish negligible-error overfitting. | The capped check is explicitly named a tiny-subset learning acceptance check. |

## Final independent verdicts

The engineering reviewer returned: **Ready for execution under the saved handoff. No remaining actionable blocker found.** The scientific reviewer returned: **Ready for execution. No unresolved P1/P2 scientific blocker remains in the revised plan.**

These verdicts approve the implementation contract, not unbuilt code or successful training. Full unit provenance, data/geometry coverage, numerical physics tests, and actual GPU portability remain execution gates. Human review of the saved plan and selection of the execution method still apply under the writing-plans workflow.

Both reviewers also checked `reports/PAPER_EVIDENCE_LOG.md` and found its raw observations, design decisions, unit caveats, and untested hypothesis appropriately separated. The scientific reviewer independently checked the listed sample hashes and the oxygen permutation's distance-derived connectivity. The parent independently reproduced the alignment and graph-adjacency observations.

## Standing requirement

Every later implementation plan must receive two independent subagent reviews before execution, as requested by Shedrack and recorded in `AGENTS.md`. Material revisions must be checked again. Findings must be evaluated against evidence rather than accepted automatically.

## Per-molecule preparation clarification

Both independent reviewers cleared the explicit per-molecule gate amendment before Task 4: aspirin may proceed after its own gates pass; paracetamol remains blocked by its unmatched branch. Engineering requested molecule-isolation, mixed-molecule rejection and no-success-export-on-failure tests. Scientific requested unshortlisted RMSDs for the blocked branch; the saved complete endpoint check supplies them. Both returned ready for Task 4. Thresholds and source group identities remain unchanged.
