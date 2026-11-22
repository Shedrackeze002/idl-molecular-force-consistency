# Post-midterm research priorities — 26 October 2026

Status: planning brief. The coordinator requested the next project stage and reported no new mentor or midterm feedback. This brief develops the sequence already stated in the canonical midterm. It does not authorize a numerical launch or replace an implementation or experiment specification.

## Research objective

Determine whether closer force–energy agreement improves recovery of reference molecular shapes on held-out transition connections. The immediate objective is to establish an informative recovery comparison. Preserve the accepted negative results and distinguish procedure effects from model limitations.

## Verified starting position

- The controlled E3x comparison is complete for two seeds. Direct and consistency-0.1 mean validation force MAEs are 38.186 and 38.024 meV/Å. Similar observed errors do not establish equivalence.
- All five evaluated E3x checkpoints recover 0/10 original validation starts. The energy baseline therefore fails the recovery adequacy gate. The ten starts represent five connections, not ten independent connections.
- Fixed post-hoc traces show accepted per-atom movements up to 436.721 Å for direct prediction and 268.047 Å for energy-derived forces. This implicates a procedure limitation without establishing the sole cause of failure.
- The same six reference-start controls yield 4/6 recovery for energy C seed 0, 1/6 for direct seed 0, and 0/6 for the other three checkpoints. These are seen minima and post-hoc controls.
- MACE Run A completed 2,048 epochs and independently scored 5.349 meV/Å validation force MAE. MACE recovery remains unmeasured; architecture, exposure, optimization and hardware differ from E3x.
- The representation fits are complete. E64, S64 and SP two-seed mean BEST errors are 34.893, 39.542 and 41.358 meV/Å under the fixed E64-selected recipe. Scalar retuning was not performed; this does not identify feature causality.

Sources: [canonical midterm](../../reports/midterm/group9_Midterm_IDL.tex), [original recovery](../../reports/E3X_VALIDATION_RECOVERY_2026-10-17.md), [paired traces](../../reports/E3X_PAIRED_TRACE_2026-10-18.md).

## Approaches and recommendation

1. **MACE control first, then movement caps — recommended.** Follow the midterm's declared sequence. Admit the stronger predictive reference and measure its behavior under the unchanged procedure before altering that procedure. This requires the real loader and parity work.
2. **Movement-cap study first.** Address the observed extreme movements immediately on existing E3x models. This can proceed without MACE readiness, but initially lacks the stronger predictive reference. Adopt this fallback through a recorded scope decision if loader delivery blocks progress.
3. **New training or architecture expansion first.** This would test a different question and consume additional effort before recovery becomes informative. Defer this option unless a reviewed feasibility result motivates a precise training intervention.

## Stage 1: MACE admission and unchanged recovery controls

First deliverable: a source-bound loader/parity specification and readiness record, using the existing shared calculator bridge and Idris handover packet.

- Preserve ownership: Idris owns loader/admission and independent reproduction; Olufunke owns evaluator integration. Shedrack coordinates source readiness, review and shared analysis. These remain assignments rather than completed contributions.
- Use the primary final EMA artifact, epoch index 2047, SHA-256 `fbd128215c0d74f5d9e5d3ccde778763221f987535dcc8ce4f4f8471a5b35407`. Do not select another checkpoint from recovery outcomes.
- Verify package and source identities, input record/atom order, units, precision, energy convention and agreement with saved validation predictions. Fix tolerances, sample membership and resource limits before new inference.
- After admission, evaluate the unchanged ten validation branch starts and six reference starts. Keep the existing force threshold, geometry acceptance, force budget, outcome taxonomy and no-relaxation comparison fixed.
- Specify a MACE driver separately: the existing validation driver hardcodes five E3x models and E3x admission. The shared bridge alone does not make that command MACE-ready. Persist the verified MACE identity alongside the result audit.
- Report every failure and incomplete outcome. MACE is an external reference; its performance cannot isolate the effect of consistency training.

The local v2 packet passed its metadata-only verifier on 26 October: 11 inventoried files matched. This did not load the model or establish runtime compatibility, inference parity or recovery.

Detailed existing contract: [MACE evaluator contract](../mace-evaluator-contract.md); [Idris handover](../team/IDRIS_MACE_HANDOVER_2026-10-18.md).

## Stage 2: isolate the movement-cap intervention

Prepare a separate experiment specification after Stage 1, or after an explicit fallback decision. Specify a small fixed set of per-atom displacement caps before observing their results. Do not choose caps from the best recovery outcome.

Preserve the original uncapped results as the primary historical comparison. Include all five original E3x checkpoints: energy C seed 0, direct seeds 0/1 and consistency-0.1 seeds 0/1. Apply each procedure identically to their original ten starts; include MACE only after admission. Hold force and geometry thresholds, starts, budgets and classification fixed; document the cap's exact vector operation and rejected-query accounting.

Measure correct-reference recovery, wrong stops, nonconvergence, connectivity and movement diagnostics, plus force calls and elapsed time. Compare paired outcomes by connection and retain seed-specific results. Reused-validation results remain post-hoc. A capped trajectory can remain connected while failing to recover the correct shape.

Retain the baseline adequacy rule: the energy model must recover at least one original branch and exceed unchanged-start recovery before the procedure supports the comparative recovery question. Adequacy is not evidence of a consistency benefit. Prespecify the rule for interpreting cap outcomes, including complete failure, before execution. Any later standard-optimizer comparison needs a separate reviewed intervention.

## Stage 3: complete the physical and cost analysis

Aaron's stream covers force–energy disagreement, numerical closed-loop work and comparative timing. Fix loop construction and quadrature controls, and separate numerical integration error from measured nonconservative work.

Use identical hardware and synchronized prediction timing where possible. Report compilation, warmup, inference and supervised training costs separately. A matched-wall-time training comparison requires its own prospective budget and review; existing exposure or timing differences cannot substitute for it.

## Stage 4: freeze the final evaluation

Freeze checkpoint selection, procedure, controls, metrics, pooling and uncertainty analysis before requesting the separate held-out test evaluation. Do not use test outcomes to select a solver, cap, penalty or checkpoint.

Decide on additional seeds or paracetamol only after aspirin feasibility and the final claim are clear. Retain a diagnostic or negative-result paper if recovery remains uninformative; do not promise a recovery benefit.

## Immediate engineering readiness issue

The top-level checkout is an older, heavily modified research tree and lacks later implementation files cited by current documentation. A preserved later execution tree contains those files, but its Git worktree pointer still names the former project location and fails repository discovery.

Before code changes, establish a clean source checkout from the verified research snapshot and authenticate it against retained implementation/evaluator identities. Preserve the existing trees, user edits, histories and numerical outputs. Do not repair readiness by copying selected files into the old checkout or treating a broken worktree as an authenticated execution revision.

The preserved publication checkout is clean at report revision `83bd55bcb47ac0b910ba946110648d849a8936c6` and contains research snapshot `3fa5969`. Its MACE bridge, validation driver, evaluation package and bridge test match that snapshot. It can supply an isolated working branch, but does not contain the full MACE baseline environment. Bind the retained scorer and environment separately; static package identities do not establish runtime compatibility.

## Review and launch boundary

Scientific and engineering priority reviews approve the sequence under [TEAM.md](../../TEAM.md). The scientific refinements specify all five E3x checkpoints, baseline adequacy and prospective cap decision rules; the engineering refinement distinguishes the viable source checkout from separately retained environment assets. Their scope is this sequence and readiness assessment. Each implementation or numerical plan still needs its own two reviews, verification and bounded launch record.

No new model training, model loading, inference, relaxation, held-out test scoring, remote allocation or outgoing teammate message was performed for this brief. Existing midterm submission administration is separate from these research priorities.
