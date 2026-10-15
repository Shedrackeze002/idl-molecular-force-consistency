# Team workflow

Read [the project handoff](docs/team/PROJECT_HANDOFF.md), [baseline commands](docs/run-baseline.md) and the relevant report before starting. Use the experiment index for targeted lookup.

For endpoint evaluation, read [the branch-reference contract](docs/team/ENDPOINT_REFERENCES.md). The frozen split's legacy `endpoint_pair` is filename metadata and must not select plus/minus targets.

Each member owns a substantive workstream and authors their own code, tests, evidence and report section. Use a separate branch and a bounded PR with the problem, implementation, verification and limitations. Record actual PR and commit links in the contribution ledger. Existing work is not reassigned to another member.

Before implementation, save an action plan with settings, dependencies and acceptance checks. Obtain two technical reviews and resolve material issues before proceeding; material amendments need review again. A historical launch package does not approve a new run.

Before each trial, record rationale, fixed settings, controls, data roles and acceptance criteria. Afterwards record outcome, failure or stop reason, evidence identities, limitations and next decision. Retain unsuccessful, invalid, blocked and abandoned attempts. Keep execution validity separate from learning acceptance, and synthetic fixtures separate from measured model results.

Use whole-connection data partitions and training-only preprocessing. Keep test data closed during fitting and selection. Report fitted-anchor results as fitted, and CPU timings as CPU timings. Sparse observations do not establish continuous stability or a causal mechanism.

Preserve existing source datasets, run directories, reports and historical evidence. Use a fresh protected output directory for new work, with a new source identity and verified launch gate. The coordinator adopted the revised adequacy protocol on14 October: failed tiny-subset gates remain diagnostics. New scientific launches still require reviewed source/config/budget records, and no adequacy or recovery claim follows from an engineering smoke pass.
