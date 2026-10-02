# Capacity engineering plan follow-up: Amendment A

Review date: 2026-10-02. Reviewer: original independent engineering plan reviewer. Scope: material amendment and author rulings only, against the preserved initial review and existing helper evidence. No implementation, tests, training, model initialization or prediction was performed.

Disposition: **engineering plan cleared for implementation**. CAPACITY-ENG-P2-001 is resolved at the plan level. No new material issue found. This clears the engineering plan review only; the other independent plan review and the later implementation/preproduction gate remain required.

## Verified reviewed identities

- Amended specification SHA256: `2f156d40d80b97c714cf3e8715912389e898ce13d94e1903b60ec0c5280cabd2`.
- Amended implementation plan SHA256: `b51c8b64ddf50857c5edaaf9350cbfa3c689e09ccc38d6b2a27cb0c1d291603e`.
- Author rulings `reports/execution/capacity-2026-10-02/plan-author-rulings.json` SHA256: `7d862d0b2d0c5728d66fc7260081ba9518de7ef91a23ffec81f161caf0e6a61d`.
- Original review `docs/reviews/2026-10-02-capacity-engineering-plan.md` remains preserved at SHA256 `621f099f1f6e09a89266d0f57166e43e4e32356e9174fe3e35f6ee5a98fd15b2`.
- Frozen numerical configuration remains SHA256 `655cfd44c0eeacd2e2cb62210b2a78eb5562a417dc01e09a707dfba931f6e960`.
- Source pin map remains SHA256 `9611a194fc9c30906f6b9e6a4b66eb4590d279b0799b8299ca0133dc7b0e6fd7`; all20 named source files were independently rehashed with zero mismatches.

Specification/plan hashes match the saved author rulings. A focused filesystem check found no `capacity*.py` implementation modules at review time. This is consistent with the recorded no-implementation state; it does not substitute for the author's broader execution record.

## Material finding resolution

Amendment A now requires exclusive parent `probe-attempt.json` reservation, an unpredictable launch token, a separate exclusive public-child `probe-entry.json` claim, and a bound `probe-completion.json`. The attempt binds exact case/config/code, input/provenance/reference, saved0 and terminal checkpoint/history identities. The public `--probe ROOT --token TOKEN` guard validates these identities before importing numerical libraries or initializing a model; the pure code/runtime guard is expressly separated from numerical runner imports. Missing, forged, mismatched, repeated and concurrent attempts are rejected before numerical work. A consumed attempt remains consumed after interruption or unknown completion; no retry is authorized.

Tasks1–3 now name guard tests, private-fixture/public-route separation and saved-report reservation/entry/completion reconciliation. Completed child work remains known if a later source/cap/ownership check fails. These requirements close the original gap where a valid public child route could run more than once despite a fixed one-attempt config.

## Advisory reconciliation

All initial implementation advisories have an explicit saved disposition:

1. Numeric saved-state audit must decode and verify finite leaves, nonnegative Adam variance and exactly zero initial moments; opaque schema/count checks alone are insufficient. It remains saved-only model work.
2. Every checkpoint uses the actual width-specific saved0 shape blueprint, with an independently initialized child template checking architecture/seed. Historical dimension substitution is prohibited.
3. DirectoryLease/all-component symlink checks and fd-backed or anchored IO protect root/case ownership, including lost-directory tests. Raw orphans survive interruption; ownership loss invalidates execution and blocks later work.
4. Canonical saved JSON/Markdown must regenerate exactly, and substantive report/evidence/post-write closure checks remain within10800s. Figures, archive verification, independent saved-result review and Obsidian navigation are explicitly later closeout work, separately recorded outside the numerical cap. They cannot support a total-work or efficiency claim.
5. A valid child completion receipt retains its known template/prediction counts through later failure, with an explicit regression test. Unreturned attempt work remains unknown.

The additional scientific clarification correctly describes144 role prediction entries as128 distinct fitted structures, since the16 anchors are inside the fitting support. Width/order/rate/update/support/loss/seed/caps and the63 observation schedule remain unchanged.

## Remaining acceptance scope

These are specified controls, not verified implementation behavior. Before the exclusive production run, retain RED/GREEN guard/state/partial-accounting/real-child/closure tests, run the full suite, and obtain independent implementation review with material findings resolved. Post-write overrun must still invalidate complete contrasts, even when all numerical updates and child predictions succeeded. Preserve the initial review and this follow-up beside the implementation/results trail.
