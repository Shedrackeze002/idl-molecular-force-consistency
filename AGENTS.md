# Project workflow

## Implementation plan review

User instruction, 29 September 2026: "Make it a habit going forward for 2 subagents to review implementation plans before we go. Always."

Before executing any implementation plan, obtain two independent subagent reviews. Give each reviewer the current saved plan, its design/specification, and relevant project evidence. One review should focus on scientific validity and evaluation controls; the other should focus on implementation correctness and verification.

Evaluate the findings rather than accepting them automatically. Resolve material issues, record decisions and remaining limitations, and obtain follow-up review of material revisions before execution. Reviewer agreement is evidence to assess, not a guarantee of correctness. The two-review requirement applies to every implementation plan, including later milestones.

Reviews are authorized. Delegating implementation or sending messages to other user-owned chats still requires its own authorization. Preserve user changes and distinguish proposed experiments, engineering checks, and measured research results.

## Paper evidence

Keep meaningful findings in the append-only `reports/PAPER_EVIDENCE_LOG.md`. Record stable IDs, exact source or run paths and checksums, metric/matching conventions, verification status, limitations, and the next required check. Separate data observations, implementation diagnostics, method decisions and untested hypotheses from measured model results. Add corrections as linked entries rather than silently overwriting earlier observations.

User reaffirmation, 29 September 2026: document every experiment and trial, whether it succeeds or fails, to preserve a balanced research trail for the paper. Before running an experiment, log its rationale, fixed settings, controls and acceptance rule, with the reviewed plan/configuration. Append its actual result, failure/stop reason, artifact identities and limits afterward. Preserve unsuccessful, blocked and abandoned attempts; never replace them with a selected successful run. Distinguish an unexecuted proposal from an executed trial. Summarize implementation/regression trials with their test evidence without treating them as scientific comparisons.

## Project memory

User instruction, 29 September 2026: save findings as work proceeds and use the existing Obsidian CLI when helpful. The `hermes` vault lives under `/Users/shedrackeze/Library/Mobile Documents/iCloud~md~obsidian/Documents/hermes`. Codex notes are in `Codex/IDL MLIP Project/`: `Project Hub.md` holds current state/navigation and `Research Findings.md` links paper-facing observations to evidence IDs. Use the Obsidian CLI for vault changes. Keep authoritative numbers, run identities and corrections in the repository evidence log; refresh the hub when project gates change. Do not treat an old note as current until checked against repository artifacts.

Obsidian access diagnostic, 1 October 2026: the installed CLI can report that Obsidian is missing when run inside the restricted Codex sandbox even while permitted host execution succeeds. Before declaring CLI unavailability, compare the same read-only command through the authorized host execution path. Use the existing Obsidian CLI with backups and verified readback for vault changes; preserve the access diagnosis in repository evidence.

## Preserve review records

User instruction, 30 September 2026: leave the review folders in place because they are harmless and help track reviews. Preserve review workspaces, packages, test logs and progress ledgers after completion; do not automatically delete them as cleanup. Durable execution archives supplement retained workspaces. Remove review records only when the user explicitly requests it. This project preference takes precedence over skill instructions to delete completed review workspaces.
