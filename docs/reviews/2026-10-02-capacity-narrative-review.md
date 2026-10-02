# Capacity scientific narrative review

Date:2026-10-02. Additive wording check against the independently verified whole-pair artifacts; no repeated broad verification, model work or raw-result changes.

Initial disposition: **one factual wording correction required; other inspected claims agree with verified evidence.**

Reviewed `reports/capacity-findings-2026-10-02.md` SHA256 `b35438003efdccae3fcb1a0559d3310b68011b351dc4fa24d68482303c17783e` and `reports/execution/capacity-2026-10-02/capacity-trajectories-caption.md` SHA256 `68de68a4ea779c17c0c8a3719982facfc6966eab3ad4f61f8d538630b5340712`.

**CAPACITY-NARRATIVE-P2-001:** The findings state “full gradients accumulated in16 microbatches.” The frozen microbatch size is16 structures;128 fitting structures therefore produce **eight microbatches of16 structures each**. Correct this sentence to match the actual protocol. Parameter counts13,569/43,521 and their3.20738447932788 ratio are correct; this correction changes wording only, not numerical results or controls. Reported promptly to the author.

The endpoint energy decrease/force increase directions, percentage changes, raw eV² and eV²/Å² units, original-baseline fractions, strict joint threshold and both failed learning outcomes agree with the saved independent reductions. Oracle/child/work/cap/archive numbers agree with the completed whole-pair review. The findings distinguish update6000 residuals from the earlier width64 update5000 excursion and make no endpoint-to-spike causal inference. Finite-window changes remain separate from convergence; the proposed spike/head-body diagnostic is explicitly unexecuted and requires a reviewed plan.

The caption correctly labels logarithmic axes,63 sampled observations, connecting lines, individual versus joint thresholds and16 fitted anchors within128 distinct structures. It retains single-seed/rate/budget, held-out, expressivity and convergence limits. Figure visual-QA and cache-warning statements are the author's rendering record; this narrow wording review does not repeat visual inspection.

No additional material scientific wording issue found. Retain this initial correction beside the author's corrected document and an additive followup before declaring narrative clearance.

## Additive correction followup

The author corrected the sentence to “full gradients accumulated over eight microbatches of16 structures each.” A focused read/assertion verifies that corrected wording and the absence of “16 microbatches”; the caption bytes remain unchanged. Corrected findings SHA256 is `b068d1343e6f27cd38a7b1283f876f31749290da84e31db22328815785ee4a0d`. **Final disposition: PASS for scientific narrative/caption wording at these identities. CAPACITY-NARRATIVE-P2-001 is resolved.** The initial finding above remains preserved, and no raw scientific result was edited or rerun. The author's failed-wording/correction record is retained under CAPACITY-FINDINGS-VISUAL-001 and the result-clearance record.
