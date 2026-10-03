# Getting the project files from GitHub

Start with [PROJECT_HANDOFF.md](PROJECT_HANDOFF.md) for the project summary, remaining work and proposed ownership, then use [CONTRIBUTIONS.md](CONTRIBUTIONS.md) to record your own substantive work and commit links. The repository is private, so confirm collaborator access before treating onboarding as complete. Andrew IDs in the proposal are not verified GitHub handles.

## Clone with Git LFS

Install [Git LFS](https://git-lfs.com/) on your development machine. These commands initialize your own client and obtain the published branch, including the four large evidence files:

```sh
git lfs install
git clone --branch codex/project-handoff https://github.com/Shedrackeze002/idl-molecular-force-consistency.git
cd idl-molecular-force-consistency
git lfs pull
git lfs ls-files
```

If you already have this branch checked out, install Git LFS and run `git lfs pull` from that checkout. A file containing only `version https://git-lfs.github.com/spec/v1` is an LFS pointer; it is not the downloaded research file. Repository access and LFS availability both matter. If a quota, authentication or download error occurs, retain the error and resolve access before using the file.

Four specific files use LFS: the corrected width/rate diagnostics JSON, its implementation and raw ZIPs, and the historical root-level width/rate diagnostics JSON. Their exact paths, sizes and SHA256 hashes are in [the publication manifest](../../reports/execution/github-publication-2026-10-03/large-artifacts.json). The corrected report remains the scientific authority; the historical root report stays preserved for the evidence trail. LFS tracking does not make that historical report current.

## What the published history means

The retained research checkout remains at original source commit `17aa83c751f95f5a1aef54fd1974340bfc5a1527`. Publication converted the four oversized file blobs into LFS pointers in a separate clone. Original author and committer identities, timestamps, messages, parent relationships and every unaffected tracked file were checked, and a [complete 74-commit map](../../reports/execution/github-publication-2026-10-03/complete-commit-map.csv) records both changed IDs and unchanged identity rows. The raw LFS map contains rewritten commits only. A separate administrative publication commit adds onboarding and verification records; it is not teammate implementation credit.

Historical source pins, run identities and reviewer gates keep their original meaning. The old commit IDs may not resolve on GitHub. The map and file-byte checks explain publication equivalence but do not revalidate a source-bound numerical launch under a migrated HEAD. Any new numerical work needs its own reviewed source and launch gate. Follow the project's two-review rule for implementation plans and record successful, failed, blocked and abandoned trials in the evidence log.

The GitHub copy also does not contain every ignored raw checkpoint, downloaded dataset or run array. Obtain those separately when a reproduction requires them; do not interpret a successful clone as proof of a complete reproduction package. No new training or scientific result was produced during publication.

## Contribution workflow

Create your own working branch from the agreed team base, make your own substantive code, validation and analysis commits, and open a PR with evidence and limitations. Review the interface agreements in the handoff before starting. Do not reassign historical work to teammates or fill the contribution ledger with planned work as completed credit. The initial publication is an unmerged draft PR; commits on that branch alone do not guarantee default-branch contribution-graph visibility or course credit. Team acceptance of the proposed allocation remains pending.
