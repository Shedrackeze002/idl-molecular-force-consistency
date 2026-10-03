# Git LFS publication engineering follow-up review

Date: 3 October 2026. Bounded re-review of the amended publication plan only. The initial review and its REVISIONS REQUIRED verdict remain unchanged. No migration, push, numerical execution or source-history/configuration change was performed by this reviewer.

## Reviewed identities and verdict

- Amended plan: `docs/team/LFS_PUBLICATION_PLAN.md`, SHA256 `8df951cfdac3ce03ddbb65d9d1f61dce786e9d5c1d2e67af23091f54d2abf9f0`.
- Unchanged manifest: `reports/execution/github-publication-2026-10-03/large-artifacts.json`, SHA256 `0b5f513c86b8990b51b61d36619bb78ea0536e65eb35fd7e0ca19a4aa8890048`.
- Initial engineering review: `docs/team/reviews/2026-10-03-lfs-engineering.md`, SHA256 `389aa84080e9187338db40b66e6a79152a6bcb2b4232974fcc8d82cc06c8f878`.
- Frozen source/main identities: `17aa83c751f95f5a1aef54fd1974340bfc5a1527` / `663bc49640901904530c9a2ede3903cf43c87b9e`.

Verdict: PASS at plan level. All six material findings from the initial engineering review are resolved in the appended executable controls. This approves the bounded publication procedure, conditional on the independent preservation review and actual execution gates passing. It is not evidence that migration, upload, remote hydration, PR creation or final preservation has already succeeded.

## Resolution of initial findings

1. The plan now names retained tool/clone/receipt paths, an independent `--no-local --no-hardlinks` clone with no alternates/shared inodes, explicit source SHA branch creation, exact GitHub origin/main fetch, and explicit LFS include/exclude refs, exact four paths and raw map output. Implicit/all-ref/no-rewrite migration is prohibited.
2. The raw rewritten-commit CSV is separated from a complete 74-row map with unchanged identity rows. Coverage, unique identities, raw author/committer/message preservation, ordered parent mapping and full tree comparisons precede a distinct 75th administrative commit.
3. Historical pointers are checked from Git objects independently of filters; attributes, manifest OIDs/sizes, hydrated payload hashes and all publication-history LFS integrity are required. The ordinary-blob limit scan includes all branch ancestry. Unaffected file modes/blob identities must remain equal.
4. The remote verifier starts from GitHub with smudge disabled and its own explicitly empty LFS storage, checks no alternates/borrowed storage, pins the GitHub LFS endpoint and explicitly fetches the published branch before hydration and payload/hash checks.
5. The preservation baseline now covers source refs/config/index/status, tracked changes, existing untracked hashes, user PDF state and reviewed handoff/ledger/log identities. Only named new publication/review records and exact-prefix log append operations are permitted. No source checkout/ref/index/config changes are allowed.
6. The plan requires absent destination branch and unchanged remote main immediately before the single explicit non-force branch push, independent remote verification before draft PR creation, attachment and readback of actual PR state/base/head, and separate post-push receipts without a self-referential commit hash.

The contributor guide also records the publication mapping's limits, the need for a separate reviewed numerical launch gate under a migrated HEAD, ignored-artifact requirements and unmerged-branch contribution graph limitations. These statements appropriately prevent publication verification from being mistaken for scientific revalidation.

Implementation note: run source metadata/status preservation reads with optional Git index refresh writes disabled (for example, `GIT_OPTIONAL_LOCKS=0`) so verification itself respects the frozen source-index requirement. This is a routine way to satisfy the approved immutability control, not a remaining material plan gap.

Execution must retain actual command argv/status, validation receipts and any failed or interrupted attempts. The initial all-history source metadata scan supports exactly four payload variants; discoveries inconsistent with that scan trigger the specified halt/review gate. The official [Git LFS v3.8.0 migration manual](https://github.com/git-lfs/git-lfs/blob/v3.8.0/docs/man/git-lfs-migrate.adoc), consulted for the initial review, supports the plan's explicit-ref and rewritten-map semantics. This follow-up does not independently re-run portable-binary installation or help checks reported by the primary agent.
