# Fixed loss-balance execution record

30 September2026. User-authorized bounded native execution on local branch `codex/landscape17-baseline`; saved plan/spec/config independently reviewed scientifically and for engineering before implementation. Plan review: docs/reviews/2026-09-30-loss-balance-plan-review.md. Final independent review pending.

Implementation revision `176af0427b053a5094f8c1e221193defd9a3a507`, source fingerprint `52140b30999e7960ced101efebbeb94fe6dadc7f816a664f6325d1964f58f997`. One production invocation, no retries/resume/sweep or scientific evaluation. Exact config `configs/aspirin-loss-balance-diagnostic.json`; run `runs/aspirin-loss-balance-2026-09-30`; raw summary SHA256 `7d74cb95ae16f7503da5098fcfb0a6b7bbf9f74a6edc975238dac47a154bdd49`.

All3arms completed2000updates/256000training exposures and128discarded warmup exposures, identical fresh parameter/Adam states,22observations and exhaustive128fit/16anchor checkpoint verification in fresh processes. Replay bothpartitions/bothmetrics at all22points matched source exactly. Serialized Adam count/update2000 independently reconciled. All31coverage+38raw-calibration+1compact hashes unchanged. Root/arm coefficients and input order reconcile. Anchor is wholly fitted and training-only.

Balanced fitting E/F fractions0.4220107073579012/0.6411855899465196; anchor0.14981029569238652/0.5562927944341675. Force-only fitting E/F fractions90243.08877300369/0.5699438664822202; anchor55075.171126502115/0.5059369229844992. Strict0.1raw threshold unchanged; balanced joint/force-only force acceptance bothfalse. No full pilot. Force-only absolute energy gauge unconstrained, no recentering. No cause/capacity/generalization claim. Full outcome/history/pergroup metrics: reports/loss-balance-diagnostics.md/json. Evidence: BALANCE/FINDING-BALANCE/VERIFY-BALANCE entries.

Verification:142fullpreproduction tests128.27s plus4metadata checks0.55s; task1final17passed36.86s, task2final33passed55.49s; initial postproduction146passed119.47s; report-layout regression1failed9.21s then4passed8.80s; final147fullsuite117.17s. Auxiliary verifier initially used nonexistent anchor_frame_count key; corrected to checkpoint-verification anchor_record_count, original failure retained. Report first build succeeded with nonblocking font-cache warnings; final JSON-only rebuild succeeded, both figures visually inspected including originalresolution forcefigure. No numerical run repeated after presentation repairs.

Observed arm times243.63797516690101/242.5499108750373/232.94204191700555s; focused tests overlapped replay preflight/initialization, so timings do not provide clean throughput comparisons. Allper-arm caps1800s retained. No new dependencies. User proposal.pdf deletion/Group9.pdf preserved. Keep local commits/workspaces; no merge/push/cleanup.

Obsidian hub/findings updated viaCLI with before/intermediate backups and successful readback; navigation points to repository numerical authority. Final review/navigation completion and durable archive follow.

Retained workspace: `.superpowers/sdd/2026-09-30-loss-balance-diagnostic/`; durable archive will supplement it. Earlier task1completion line captured precommitHEAD; ledger correction identifies actual7c8eb23..176af04range. Task3latesttest line predates reportcommit; followup ledger correction will identify currentcommit.
