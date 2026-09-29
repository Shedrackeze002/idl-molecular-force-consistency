# Two independent diagnostic plan reviews

30 September 2026. Plan: `docs/superpowers/plans/2026-09-30-force-learning-diagnostics.md`. Scientific and engineering subagents read the plan, implementation and prior failed-run evidence before execution. No research trials ran during review.

Scientific reviewer found the molecular-vs-per-atom bias ambiguity: force-only readout intercept and synthetic teacher centering require mean molecular residual divided by atom count. Corrected explicitly. Clarified rank-truncated conditional solve rather than a rigorous network capacity floor. Follow-up verdict on saved revision: **Ready for execution; no remaining material scientific blocker.** Fixed scales/subset/caps/gates intact; batch16 fitting cannot retroactively clear batch4 acceptance.

Engineering reviewer cleared corrected saved plan. Safeguards adopted: fixed nonzero head reconstruction on BOTH frozen backbones (zero-head reconstruction alone is vacuous); new-process probe reads saved control input only, tested without held-out NPZs; explicit batch_size16 in config. No remaining material implementation blocker reported. Review responses briefly hit service usage errors after delivering findings; scientific follow-up succeeded and explicitly verified corrected wording. Neither error is an experimental outcome.

Parent assessment: corrections match the E3x scalar head kernel(16,1), per-atom bias(1,), and summed molecular energies. Fixed settings remain unchanged. No additional seeds, learning-rate sweep, larger architecture or full pilot authorized by these reviews.
