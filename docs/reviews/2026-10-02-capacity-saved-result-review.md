# Independent capacity saved-result review

Date: 2026-10-02. Independent reviewer, whole-pair followup to the preserved width32 preliminary review. **Verdict: PASS for saved execution integrity, exact result reconciliation and numerical archive. No material inconsistency found. Both prespecified learning criteria failed.** This clears use of the bounded descriptive comparison; it does not clear the full training pilot or the main force-consistency hypothesis.

## Review scope and retained trials

Rechecked the frozen protocol and reviewed source identities, then inspected the completed width32 and width64 arms, controller records, primary report, finalization, production invocation/process receipts, saved closeout records and archive. `reviewer-capacity-saved-audit.py` completed successfully once in2.9797533329983708 seconds. It canonically rebuilds the report in memory from saved evidence and independently reduces saved prediction/target arrays with NumPy. A separate successful `reviewer-capacity-saved-moments.py` decodes the stored0/3000 Adam arrays without gradient or optimizer calculations. Scripts, stdout and JSON records are retained under `reports/execution/capacity-2026-10-02/`.

No training, model initialization, model prediction, child probe, production retry, suite rerun or raw artifact edit occurred. All343 numerical files were byte-identical before and after the audit. Source/lock/runtime identities match the saved585-pass/468.33-second suite evidence; that evidence was reused rather than rerun for documentation. The initial implementation findings and repaired followup, prior width32 review, unsuccessful scientific gates, all sampled excursions and original inspection failures remain preserved. This followup records a truncated combined read and a failed read of the guessed path `scripts/closeout_capacity.py`; focused full-file checks and discovery of the actual `scripts/close_capacity_saved.py` resolved those read-only issues. No numerical result was inferred from truncated output.

## Protocol, source and saved-state reconciliation

Configuration SHA256 is `655cfd44c0eeacd2e2cb62210b2a78eb5562a417dc01e09a707dfba931f6e960`, canonical configuration identity `888c9307696c1e114dddb3e24384bc30896e4c5d74b4e8279ea623243c40e26e`; the20-source map SHA256 is `9611a194fc9c30906f6b9e6a4b66eb4590d279b0799b8299ca0133dc7b0e6fd7`. Every named source pin, complete participant source/lock hash and captured dependency matches. Both arms and the once-only invocation record production revision `b45ff2a66923af14a51dce7190b3808810b2a478` and code fingerprint `2066ca4a476c5bca16a7c48312457a0efcb84da60970c3337a2b80a011113a0d`. Every source-snapshot file reproduces its launch hash. All bound preproduction artifacts, including the suite log, match their saved hashes. Suite and production revisions differ because of the scoped source checkpoint; the complete participating file set and runtime identities match exactly.

Verified fixed width32→width64 assignments, fresh seed0 CPU float64 models, constant rate.0003,6000 updates each, prescribed model/loss/budget values, exact ordered128 fitting and16 fitted-anchor IDs/memberships/array hashes, original raw baselines and unchanged normalization. Initial raw metrics are identical across widths, while parameter/optimizer identities and shapes differ. Width32 has13,569 parameters and width64 has43,521. Each starts at update/count0 with exactly zero Adam moments, rather than inheriting source training state.

Each arm has exactly63 immutable snapshots and corresponding history prefixes:0,1,every100 through6000 and3001. There are no extra saved versions. Pure serialized audits verify finite float64 leaves, nonnegative variance, width-specific actual saved0 schema, Adam counts and canonical parameter/optimizer hashes for every snapshot. Terminal commit/contract/history/summary and root-returned case record agree at6000. Each production child completed its only ordinal1 token-bound attempt, independently checked a fresh template against saved0 and reproduced all144 terminal role predictions with error0.0, count6000 and the recorded absolute1e-10 tolerance. Attempt/entry/completion hashes and exact config/case/code, directory, source/input/provenance/reference, initial and terminal bindings agree.

Both actual committed oracle boundaries bind incoming/proposed/committed state identities and prescribed discarded work. The reported absolute differences are:

| Width | Update1 | Update3001 |
|---|---:|---:|
|32|8.326672684688674e-17|1.6653345369377348e-15|
|64|2.42861286636753e-16|3.563122019656362e-15|

All are below1e-10. Separate stored-moment inspection confirms both moment trees are zero at0, and at3000 both contain7,553 nonzero elements for width32 and27,393 for width64. Thus the later boundary actually has nonzero incoming moments, rather than merely a label asserting this. The reviewer verifies saved oracle receipts/state consistency and the inspected oracle implementation, without rerunning full gradients or creating proposals.

## Exact accounting and scientific closure

Each case records6000 computed/attempted/persisted/independently receipt-verified updates and768000 fitting-frame exposures, with no unknown pending update. Pair totals are12,000 updates and1,536,000 fitting exposures. Separate pair work is two parent initializers on32 frames,256 discarded warmup exposures,18,144 scoring role entries,288 endpoint-reference role entries,512 full-gradient and512 accumulated oracle frame exposures, four independent NumPy and four discarded accumulated proposals, two child template initializers on32 frames and288 child role predictions with zero child training updates. These categories are not added into a fabricated single training-exposure count. Each arm's144 role entries cover128 distinct fitted structures; the16 anchors are a group01 subset of fitting, and repeated evaluation across widths does not make additional unique structures.

Width32 elapsed1562.8148765840015 seconds is below its3000-second cap; width64 elapsed3619.9719801249885 seconds is below7200. Controller completion is5183.623872749988 seconds and final scientific closure5185.547022624989 seconds, below10800. Ordinal1/no-retry production invocation and the matching code record agree; launcher returned exit0 and its stdout/stderr hashes reproduce exactly. No extra numerical invocation was performed by this review.

Canonical regenerated report objects and serialized JSON/Markdown bytes exactly reproduce the primary outputs. `raw-artifacts.json` exactly covers the retained numerical root excluding itself and the separate finalization marker. The finalization marker binds its inventory SHA256, both report hashes and the first45,677,796 bytes of the append-only evidence log. Independent hashing confirms that prefix remains intact despite subsequent closeout appends. Saved-only figures/archive/reviews/Obsidian are explicitly later work outside the scientific cap, not an efficiency comparison or a claim about capped total project effort.

## Independently recomputed scientific observations

Direct NumPy reductions reproduce raw endpoint aggregate MSE/RMSE, counts, target zero/mean baselines, ordered IDs, all group and equal-group metrics, residual decomposition and gate fractions. Both joint criteria require raw energy and force below10% of original0 on both roles; the width64 anchor energy alone passing does not satisfy its joint criterion.

| Width / role | Energy MSE (eV²) | Force MSE (eV²/Å²) | Original0 E fraction | Original0 F fraction | Raw joint gate |
|---|---:|---:|---:|---:|---|
|32 fitting|.01198320842349682|.0065589689424798165|.3194094334870974|.4657052724792342|Failed|
|64 fitting|.007948333482229644|.00794262358651163|.2118608476964685|.5639486501605391|Failed|
|32 fitted anchors|.007114939839938265|.004530867645870908|.11607064219031449|.4471150476714813|Failed|
|64 fitted anchors|.0021536387316987695|.0053035089487986865|.03513370966695055|.5233608310384563|Failed|

Contrast direction is independently verified as width64-minus-width32 and64/32:

| Role / metric | Raw difference | Ratio |
|---|---:|---:|
|Fitting energy|-.004034874941267176|.6632892628859276|
|Fitting force|+.0013836546440318138|1.2109561207205048|
|Anchor energy|-.004961301108239495|.3026924724801969|
|Anchor force|+.0007726413029277785|1.1705283321687636|

Width64 therefore ends with lower energy error and higher force error on both fitted roles at this one rate/seed/budget. Its largest saved energy observation is5000: fitting4.882506784430785 eV²,130.1420012818957 times its original baseline; anchor4.680865802953164 eV²,76.36200895272364 times its baseline. Next saved update5100 has fractions.24848662649600908/.03648942537263615, respectively. Other above-original-baseline fitting energy observations occur1400,1700,3200,3600 and4500; anchor observations occur1400,3200 and4500. All are retained with next saved observations in the verification JSON. Width32's largest saved energy observation remains3700 at3.6020684618113443 fitting and2.1998201468692145 anchor baseline fractions; its other preserved fitting excursions occur2300,3100 and4900. The maximum saved force MSE remains update0 for both roles and widths.

Independently recomputed5500→6000 relative changes match: width32 fitting E/F -.06663519548328774/-.028031009823560764 and anchors+.05076265818435542/-.02887724799393233; width64 fitting-.0841186602325171/-.027863712214831998 and anchors-.03028476624263746/-.023893034802308397. These finite-window changes do not establish convergence.

Endpoint energy residual mean-squared plus centered variance reproduces raw MSE without recentering a gate. Width64 fitting signed mean is-.00015582647317679416 eV, mean-squared component2.428188974271815e-08 eV² and centered variance.007948309200339902 eV². Width64 anchors have signed mean-.00015209833530760442 eV, mean-squared component2.3133903603344463e-08 eV² and centered variance.0021536155977951664 eV². Width32's corresponding decomposition remains as preserved in the preliminary review. These endpoint decompositions cannot explain the earlier sampled energy excursion; this review makes no spike-mechanism claim.

## Archive and reproducible artifact identities

`numerical-run.zip` contains exactly343 unique relative members, matching the independent retained-root inventory. ZIP CRC passes, every member's SHA256 matches its source bytes, and no member uses an absolute/traversal path. Retained numerical bytes remained unchanged. Hash checks also reproduce the saved closeout record's archive/inventory/report and PNG/SVG figure identities. The existing plot code preserves every sampled point and full peak height on logarithmic axes; author visual inspection is separate from this numerical saved-result audit. Retained font-cache warnings are closeout rendering-environment records and do not alter the numerical conclusion.

Principal hashes:

- Primary JSON: `31fc7af10a6b87dcb584fb73593d279d2cc57b61756d8a388e1927b8296769ec`.
- Primary Markdown: `c111215a040b154207c26a255aad7ac1244fe6f04c83f545c1c7f2a24b4d8ac5`.
- Bound raw inventory: `c8de92644fd7ada1af708354417051ad21628729f0bdd188bbd51c3be8628f56`.
- Evidence prefix: `c9a671d84712a9203bc25550b5bd95c685dda817d87464c22f501f56ac717584`.
- Archive: `2226d21dab8533330914aee75473a88ac4e9cf0b546141d309c0ec5fb287939a`.
- Full archive/root inventory: `bc689bd0a7deaab79b07d388492fd316e5c1e4026cc200ec22db28c38fa64ebf`.
- Independent whole-pair verification JSON: `2a1056c6bdb59b5f55bfe263999a89d320729e3673765215f2129e1cd717b3dd`.
- Independent stored-moment verification JSON: `2088b46126239a605a303a0c1d82d96abf8881c07acc5787a4087cbd4d0ae413`.

## Interpretation and remaining project scope

The saved report explicitly retains one seed, one selected rate, finite fitted-support budget, initialization-shape/optimization confounding, sparse-score limits, raw uncentered criteria, fitted-anchor status and pending full pilot. Equal updates/exposures do not imply equal compute. The available result supports a descriptive energy/force tradeoff at6000 and much larger sampled width64 energy excursions. It does not establish a pure capacity/expressivity cause, continuous stability, convergence, generalization, spike explanation, hardware efficiency, GPU parity or the main research hypothesis.

Preserve both unsuccessful criteria, all read/engineering trials and prior reviews. The author should append this final independent verdict/identities to the repository evidence log and update Obsidian navigation through its authorized backed-up/readback workflow. Those documentation appends may extend the log while retaining the verified scientific prefix. No numerical rerun is needed for this closeout.
