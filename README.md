# Landscape17 molecular force consistency

Group 9 investigates whether consistency between energy-derived and direct forces improves stable-shape recovery on unseen transition-state connections. The planned study controls architecture, force accuracy and computational cost. The main comparison remains unexecuted.

The proposal is submitted. We have built the data pipeline, an energy-derived E3x baseline, checkpoint and physical checks, and a sequence of bounded aspirin fitting diagnostics. None of the real-label diagnostics has passed the predeclared joint learning gate. The full pilot, direct-force/consistency arms, shape-recovery evaluation and MACE reference remain pending.

Start with [the team handoff](docs/team/PROJECT_HANDOFF.md), [team workflow](TEAM.md), [baseline commands](docs/run-baseline.md) and [the experiment index](reports/PAPER_EVIDENCE_LOG.md). The [contribution ledger](docs/team/CONTRIBUTIONS.md) records each member's future authored work and verification.

Aspirin preparation is frozen at 4,891 training, 974 primary validation and 1,127 test records across 26/5/6 whole connections. All 11 reference minima are training-seen. Paracetamol preparation remains blocked by the unmatched `01_2/plus` endpoint. Test data are excluded from fitting and selection.

The latest offset-profile trial completed 6,000 updates. Fitting force MSE improved 3.1563519827% relative to the historical width 32 endpoint, but all four energy/force baseline fractions remained above the strict 0.1 gate. Anchor centered energy variance worsened 24.41971159078873%. Anchors are fitted structures, so this is diagnostic evidence, with no held-out performance claim. See [the paper findings](reports/offset-profile-paper-findings.md).

The next proposed lever is [force-gradient isolation](docs/research/plans/2026-10-03-force-isolation.md). Its implementation and execution remain pending. Read [the literature map](reports/literature/2026-09-30/READING_MAP.md), [proposal source](proposal.tex) and [claim audit](CLAIM_AUDIT.md) for scope and attribution. The proposal files describe submission-time scope; dated experiment reports give current research status.

This package contains source, tests, frozen configurations, scientific reports and selected figures. Machine-local report paths are normalized in documented research exports. Complete historical run and launch packages are retained separately, and frozen source guards are unchanged. Read [the reproduction boundary](docs/REPRODUCTION.md) and [Git LFS setup](docs/team/LFS_SETUP.md) before attempting historical reproduction. GPU execution on CMU, Colab or Kaggle remains unverified.
