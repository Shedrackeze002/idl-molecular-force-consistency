# Independent data-coverage plan reviews

30 September 2026. Planning-only request. Spec `docs/superpowers/specs/2026-09-30-data-coverage-diagnostic-design.md`; plan `docs/superpowers/plans/2026-09-30-data-coverage-diagnostic.md`; candidate manifest/config remain planned and unexecuted. Native implementation is the existing method; this turn did not implement or train.

## Scientific review

Verdict: **Ready as a planning-only deliverable. No material scientific correction required.** Independently verified candidate canonical identity, support selection, rational rank mapping, training-path memberships, indices and hashes. Counts16/16/128; A⊂C and B⊂C; B fits exactly2anchor frames; C uses B's8groups with balanced branch/group counts. Five replay hashes match and reference history covers every100updates through2000.

Reviewer accepts the explicit limitations: ordinal ranks do not match physical difficulty; B changes fitted geometries; C changes coverage and computation; separate fitting/anchor gates prevent confusing easier data with recovery of the original gap. Sequential accumulation at unchanged parameters plus one Adam step preserves full-subset objective. Fresh initialization, replay, fixed caps and operational blocking are appropriate. A successful C supports a new reviewed protocol decision; a failed C leaves cause unresolved. Neither is causal/statistical proof or retroactive batch4 acceptance.

## Engineering review

Verdict: **Ready for implementation after planning approval. No material blocker found.** Independently checked canonical manifest hash, all5reference hashes, PCG64 order, distinct sizes/nesting and anchor overlap. Frame-count weighting preserves both MSE terms because all frames share21atoms; one Adam update preserves the optimizer contract. Uneven-chunk and two-update tests are meaningful. Train-only access, input-only exhaustive reload, current/historical identity separation, finite/blocked distinctions and exposure/cost reporting are explicit.

Implementation details made explicit after review: assert matching fresh initial parameter/optimizer hashes across arms; process microbatches sequentially for memory benefit; call a scheduled successful measurement the first observed passing update, not an exact threshold crossing. These clarify already specified controls without changing IDs/settings/gates/budgets. No material revision requiring a new scientific protocol was made.

## Parent assessment and state

Both reviewers cleared the saved design/protocol with the above engineering clarifications incorporated. Parent independently checked manifest/config identities and5reference hashes. No model initialization, source-code modification, dependency installation, label fitting or research training occurred. All support/ID choices are label-blind planning artifacts. Prior failures and user proposal changes remain intact. Review records remain preserved under AGENTS.md. Execution awaits the next execution instruction; no full pilot or scientific comparison is implied by plan review.
