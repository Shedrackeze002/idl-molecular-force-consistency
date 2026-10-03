# Team workflow

This guide sets the working conventions for Group 9's Landscape17 project. Start with [the project handoff](docs/team/PROJECT_HANDOFF.md), agree ownership and interfaces, and follow [the run instructions](docs/run-baseline.md) for the relevant environment.

## Review before implementation

Every implementation or experiment plan receives two independent technical reviews before execution. One examines scientific validity, evaluation controls and claim limits; the other examines implementation correctness and verification. Give reviewers the saved plan, design and relevant project evidence.

The coordinator evaluates findings against the evidence, resolves material issues and obtains follow-up reviews of material revisions. Record remaining limitations. Plan-review acceptance does not establish that implementation, a numerical launch or a scientific result has passed. Technical reviews supplement the author's validation and a teammate's PR review.

## Balanced research evidence

Record meaningful findings in the append-only `reports/PAPER_EVIDENCE_LOG.md`, with stable IDs, exact source/run paths, checksums, metric definitions, verification status, limitations and the next required check. Use it as an index for targeted lookup.

Before an experiment, save its rationale, fixed settings, controls and acceptance rule. Afterward, record actual outcomes, failure or stop reasons and artifact identities. Keep unsuccessful, blocked and abandoned attempts alongside successes. Proposed work stays clearly distinct from executed work, and engineering checks stay distinct from measured model results. Add linked corrections rather than silently changing historical observations.

Repository evidence is authoritative. Personal notes can help navigation but do not replace saved research records.

## Contributions and protected work

Each member owns substantive implementation, meaningful verification, analysis and a report section. Use your own branch and commit identity; record completed PRs, commits, checks and report sections in [the contribution ledger](docs/team/CONTRIBUTIONS.md). Proposed assignments are not completed credit.

Protect frozen experiment sources, manifests, configurations and run artifacts. Work in separate checkouts while a source-bound numerical trial is active. Do not change its source, dependencies or Git revision during the trial. New work needs its own reviewed implementation and launch checks.

Preserve dated plans, review records, packages, test logs and progress ledgers, including interrupted or failed reviews. Keep user changes intact. Do not remove these records as routine cleanup; exact historical records may refer to documentation names or conventions that were current at their date.

## Shared repository

Commit research code, reproducible instructions, configurations, manifests, reports and evidence needed by the team. Keep personal tool configuration and private workflow instructions local. Check the staged file list before publishing. Large tracked evidence files use Git LFS as described in [the setup guide](docs/team/LFS_SETUP.md); ignored datasets and checkpoints still require separate verified access.
