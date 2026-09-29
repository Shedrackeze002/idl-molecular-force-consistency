# Project workflow

## Implementation plan review

User instruction, 29 September 2026: "Make it a habit going forward for 2 subagents to review implementation plans before we go. Always."

Before executing any implementation plan, obtain two independent subagent reviews. Give each reviewer the current saved plan, its design/specification, and relevant project evidence. One review should focus on scientific validity and evaluation controls; the other should focus on implementation correctness and verification.

Evaluate the findings rather than accepting them automatically. Resolve material issues, record decisions and remaining limitations, and obtain follow-up review of material revisions before execution. Reviewer agreement is evidence to assess, not a guarantee of correctness. The two-review requirement applies to every implementation plan, including later milestones.

Reviews are authorized. Delegating implementation or sending messages to other user-owned chats still requires its own authorization. Preserve user changes and distinguish proposed experiments, engineering checks, and measured research results.

## Paper evidence

Keep meaningful findings in the append-only `reports/PAPER_EVIDENCE_LOG.md`. Record stable IDs, exact source or run paths and checksums, metric/matching conventions, verification status, limitations, and the next required check. Separate data observations, implementation diagnostics, method decisions and untested hypotheses from measured model results. Add corrections as linked entries rather than silently overwriting earlier observations.
