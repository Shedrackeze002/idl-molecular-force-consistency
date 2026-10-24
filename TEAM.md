# Team workflow

Read [the project handoff](docs/team/PROJECT_HANDOFF.md), [baseline commands](docs/run-baseline.md) and the relevant report before starting. Use the experiment index for targeted lookup.

For endpoint evaluation, read [the branch-reference contract](docs/team/ENDPOINT_REFERENCES.md). The frozen split's legacy `endpoint_pair` is filename metadata and must not select plus/minus targets.

Each member owns a substantive workstream and authors their own code, tests, evidence and report section. Use a separate branch and a bounded PR with the problem, implementation, verification and limitations. Record actual PR and commit links in the contribution ledger. Existing work is not reassigned to another member.

Before implementation, save an action plan with settings, dependencies and acceptance checks. Obtain two technical reviews and resolve material issues before proceeding; material amendments need review again. A historical launch package does not approve a new run.

Before each trial, record rationale, fixed settings, controls, data roles and acceptance criteria. Afterwards record outcome, failure or stop reason, evidence identities, limitations and next decision. Retain unsuccessful, invalid, blocked and abandoned attempts. Keep execution validity separate from learning acceptance, and synthetic fixtures separate from measured model results.

Use whole-connection data partitions and training-only preprocessing. Keep test data closed during fitting and selection. Report fitted-anchor results as fitted, and CPU timings as CPU timings. Sparse observations do not establish continuous stability or a causal mechanism.

Preserve existing source datasets, run directories, reports and historical evidence. Use a fresh protected output directory for new work, with a new source identity and verified launch gate. The coordinator adopted the revised adequacy protocol on14 October: failed tiny-subset gates remain diagnostics. New scientific launches still require reviewed source/config/budget records, and no adequacy or recovery claim follows from an engineering smoke pass.


## Publish completed results

After every completed trial or accepted research result, update GitHub with the relevant code, result, supporting evidence and current status. Include unsuccessful and interrupted outcomes with their limitations. Use a bounded change and verify the remote commit and file identities. Keep personal tool configuration, tool identities, private review instructions and operating transcripts out of shared files and commit/PR text; retain those records locally. Publication must not silently convert a pending review or incomplete run into an accepted scientific result.


## Available compute assumption — confirmed 20 October 2026

Plan remote GPU work assuming **Colab only**. ORCHARD is a possible addition after access is actually granted and usable resources are confirmed; do not depend on it, PSC provisioning or another cluster to complete the project. Shedrack is discussing ORCHARD separately. Existing local CPU work is not excluded by this GPU-planning assumption.

The coordinator clarified that the current compute-unit balance is not a study constraint: up to $100 can be used for additional Colab units, equivalent to 1,000 purchasable CU at the coordinator's quoted $10 per 100 CU. This is a planning envelope, not a claim that those units are already purchased or that the quoted rate was independently checked. Do not reduce or defer useful experiments merely because the present balance needs a top-up. Estimate and record expected/actual consumption, including setup/checkpoint overhead, so purchases can be timed appropriately. Colab GPU availability, memory, session duration and interruption recovery remain practical platform considerations; no particular accelerator is guaranteed. Reuse verified checkpoints and resumable Drive backups. Existing plan reviews and experiment budgets still apply; no run or purchase is performed by this documentation update.
