# Width/rate plan: independent reviews and author rulings

30 September 2026. Planning only; no implementation, tests, prediction probes or training executed. User/AGENTS requires two independent reviews and re-review of material amendments. Native implementation is established; human review of this concrete artifact remains pending. Original and amended drafts are preserved in `.superpowers/sdd/2026-09-30-width-rate-plan/`.

## First scientific review

Reviewer `/root/width_rate_scientific_plan_review`: scientifically defensible; no Critical or Important findings. Read design/plan/config/manifest, existing reports and implementation, literature map and PaiNN training settings without numerical execution. This was not verification of future implementation or all301source bytes.

Two Minor findings: (1) worse fresh lower-rate endpoints at equal6000updates do not test whether lowering the rate late or using validation-triggered decay would work; (2) local sequential wall times are affected by compilation caches, order and system load and cannot establish architecture-efficiency rankings. Author accepted both and added explicit report limitations; no numerical setting changed.

The reviewer supported cloned within-width initialization, honest cross-width draw differences, fixed original zero-head baselines/weights,63-point replay, fully fitted anchors/one-seed limits, retaining learning failures, cap-limited partial contrasts and fullpilotpending after any diagnostic pass.

## First engineering review

Reviewer `/root/width_rate_engineering_plan_review`: amend before implementation; no Critical findings, two Important gaps and one Minor dependency issue. Static review verified all301source hashes, canonical config/manifest identities,22+42histories with equal duplicate2000 and63joinedpoints,64plannedpoints, historical parent-config guards and fixed-rate Adam state compatibility. No tests/models/probes/training executed; replay feasibility, runtime/memory and future numerical tolerances remain unverified.

1. **Important: persistence transaction.** Individually atomic state/history writes do not guarantee a matched recoverable pair. Existing continuation helper writes history before checkpoint and can persist a newer state in failure handling. Author verified the helper behavior and accepted: new isolated runner must write versioned checkpoint and versioned history, validate both hashes/equal endpoint, then publish an atomic commit receipt. Receipt alone defines persisted_update. Interruptions preserve the previous receipt; attempted observations/orphans/failure receipts are separate. Explicit failure injections cover each boundary; report/probe verify the receipt pair. No existing research file or helper changed.
2. **Important: runtime context and hidden preflight work.** Unchanged balance_preflight does not forward load_gradient_sources' inner callback and discards its historical fresh template. Author verified the parent call and208source-prediction frames (16+16+128fitting plus three16anchors). Accepted runtime-only ExecutionContext through controller/preflight/arms/probe, global before/after parent boundaries, started/completed/unknown operation receipts, actual elapsed and honest unknown interrupted work. Historical template work is separate from the two shared training initializations. Probe timeout uses the smaller positive remaining arm/controller allowance. Tests cover global exhaustion and once-only actual versus paired cap-charge accounting. Boundary caps cannot forcibly cancel in-flight work.
3. **Minor: dependency identities.** Author expanded immutable source pins301→326 to existing Python source modules except the planned-to-change CLI, and the dynamically imported optimizer report verifier. New code and CLI identities will be captured as execution provenance. All326existing pinned bytes verified statically.

## Material amendment and follow-up review

Unchanged scientific settings: four fresh16/32feature × .001/.0003 fixed-rate cells,6000updates each, same128fit/16fully-fittedanchor, original balanced weights/gates, no schedule, held-out use, adaptive retry or pilot clearance. Updated canonical config SHA256 `e64165f0e511ae073dae1fb6e3da49053fee110e5844b5da0471bce47ec01198`; pin manifest SHA256 `621e60dd590f18214ac6467f9b9f00eff7d1248067eb1d2c9ee19f014362e0c4`. Both original reviewers dispatched for material re-review. Findings are assessed against code/evidence; reviewer agreement is no correctness guarantee.

**Follow-up engineering verdict: cleared at plan level.** Both Important findings and dependency Minor resolved; no new Critical/Important blockers. Reviewer independently verified all326pin hashes, both amended identities and64score points. Remaining engineering caution: committed version files must remain immutable; author made that explicit in spec/plan as the intended versioned-transaction behavior, with an overwrite-regression assertion. No changed scientific settings or new protocol. Numerical equivalence, memory/runtime feasibility, transaction implementation and report strictness remain future checks.

**Follow-up scientific verdict: clearance confirmed.** No Critical/Important/remaining Minor findings. Reviewer verified the208frame parent-work derivation statically and confirmed all original settings/claims/limits remain intact; read the static-check record but did not independently rehash all326files. No edits/tests/models/probes/training by either reviewer.

**Final author ruling:** both independent plan reviews and material re-reviews are complete. This is clearance of the proposed protocol, not measured success. Human review of the concrete design/plan remains pending before native implementation. Full pilot remains pending.
