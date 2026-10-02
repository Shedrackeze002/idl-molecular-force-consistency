# Capacity scientific plan review

Date: 2026-10-02. Reviewer: independent scientific/evaluation-controls subagent.

**Disposition: PASS for the proposed bounded diagnostic; no material scientific plan finding.** This review clears its scientific design only. The second independent plan review, resolution of any material findings, implementation verification and preproduction gate remain required before numerical execution. No capacity result has been measured by this review.

## Scope and inspected evidence

Read `AGENTS.md`, the capacity design/specification and implementation plan, `configs/aspirin-capacity.json`, its planning record and source-pin manifest, the current late-rate report, the corrected historical width/rate report and relevant append-only paper evidence entries. Inspected source provenance, support identities, original baseline values and source checkpoint/history metadata. The preserved invalid historical root report was not used as scientific authority.

Reviewed config SHA256: `655cfd44c0eeacd2e2cb62210b2a78eb5562a417dc01e09a707dfba931f6e960`.
Reviewed source-pin manifest SHA256: `9611a194fc9c30906f6b9e6a4b66eb4590d279b0799b8299ca0133dc7b0e6fd7`.
All 20 listed source files matched their pinned SHA256 during this review. This is a saved-file identity check, not new model evaluation or a replacement for production source-contract validation.

## Scientific controls assessed

| Control | Assessment |
|---|---|
| Question and intervention | Fresh width32 versus width64 under one prescribed rate is an interpretable descriptive architecture/optimization comparison. The plan explicitly rejects pure expressivity causality. |
| Data, loss and normalization | Both arms reuse exactly 128 ordered fitted records, the same 16 fitted anchors, fixed normalization and unchanged energy/force weights. Source provenance records all support memberships as fitted. |
| Freshness | No trained parameters or optimizer state are transferred. Each arm starts from the declared seed and zero readout/moments/count; template hashes and independent child initialization are planned. |
| Initialization | Same seed denotes the same initialization rule/key. Width changes shapes and resulting values; the specification states this limitation explicitly. |
| Optimization | Fixed CPU float64, model settings, full accumulated 128-frame gradients with microbatch16, Adam settings, rate0.0003 and 6,000 committed updates per arm. Proposed boundary checks at1 and3001 distinguish optimizer integrity from fitting outcomes. |
| Work matching | Each arm has 768,000 fitting exposures; a complete pair has12,000 fresh updates and1,536,000 fitting exposures. Other numerical work has separate accounting. This matches updates/exposures, not FLOPs or wall time. |
| Scoring and selection | Independently checked the configuration has exactly63 observations:0,1,100 through6000 at100-update intervals, and3001. No retries, adaptive schedule, early stopping or endpoint selection. Endpoint6000 is fixed. |
| Gate | Raw energy AND force MSE must each be strictly below10% of immutable original update0 baselines, separately on fitting and fitted anchors. Integrity can pass while all learning gates fail. Residual decomposition does not recenter a gate. |
| Contrasts | Prespecified endpoint width64 minus width32 and width64 divided by width32 for both support roles. Undefined ratios retain their reason. Secondary trajectory/group/residual observations remain descriptive. |
| Budget and failure | Per-arm caps3000/7200seconds and global10800seconds with180seconds reserved are bounded operational limits. A failed first arm blocks the second; incomplete/invalid execution cannot supply a complete pair contrast. Every negative, blocked or unknown outcome is retained. |
| Interpretation | One seed, fitted data, one selected prospective rate and finite sparse scores support neither generalization, convergence, spike mechanism, efficiency ranking nor clearance of the main study/full pilot. These limits are explicitly saved. |

The source fitting update0 MSE values are energy `0.03751676427546992 eV²` and force `0.014083948218070219 eV²/Å²`; anchor values are energy `0.061298358531283896 eV²` and force `0.010133561081129106 eV²/Å²`. They match source `histories/0.json` exactly. Although that individual update0 file is not a manifest entry, its complete observation equals the first entry in pinned `histories/6000.json`; pinned provenance also contains the baseline values. There is no unpinned baseline authority needed for the plan as written.

The corrected prior width/rate report has valid execution/interpretation and all fitting/anchor joint gates fail. Its width32 fitting endpoint fractions are0.20377621120957126/0.3898121837590496 at0.001 and0.3194094334870974/0.4657052724792342 at0.0003. These observations support a prospective controlled comparison, not proof that0.0003 is optimal or that width is the limiting mechanism. The valid late-rate report likewise retains failed gates: constant fitting fractions0.17711403/0.32469875 and late-decay0.19119942/0.34455828. Both reports keep the full pilot pending, consistent with the proposal.

## Material findings

None. No scientific amendment is required to the reviewed protocol before implementation, subject to the separate engineering review and subsequent verification gates.

## Advisory clarifications

1. **Anchor overlap and frame terminology.** Saved identities show all16 anchors belong to fitting support and to group01. The144 child prediction entries concatenate128 fitting plus16 repeated anchor-role records; there are128 distinct structures. Keep the two reporting roles and both gates, but do not describe144 as independent or unique structures, or the anchor criterion as independent generalization evidence. This refines the existing explicit fitted-anchor limitation and does not change the protocol.
2. **Trajectory language.** With63 scores, any spike/stability statement must refer to saved observations. Connected plot segments and the5500→6000 change do not resolve intervening behavior or prove stationarity. Keep full plotted peaks and the existing sparse-score limitation.
3. **Operational failure versus a scientific negative.** If width64 hits its cap or fails verification, preserve raw observations with status, without claiming it fits worse than32. Complete valid learning failures are distinct from insufficient/invalid observations. The proposed report suppression and retained failure accounting already implement this distinction.
4. **Capacity wording.** Retain “capacity diagnostic” as a project label, but state any measured result in terms of the fresh width intervention under the prescribed rate, seed and update budget. Even a substantial width64 improvement does not isolate representational capacity from changed initialization and optimization geometry.

## Review execution and limits

No source implementation, training, optimizer update, model initialization, model prediction, benchmark or test invocation occurred. Only saved-document/JSON reads and SHA256 identity checks were used. One overly broad JSON inspection was truncated; a second inspection was also truncated and used the wrong dictionary assumption for the list-valued history, causing a read-only `AttributeError`. Focused follow-up reads recovered the exact list schema, baseline equality, overlap, counts and score sequence; no conclusion depends on missing/truncated content. All original evidence and prior review folders remain intact. Only this new review artifact was written.
