# Preliminary capacity width32 saved-result review

Date: 2026-10-02. Independent reviewer. **Disposition: PASS for the completed width32 arm only; no material inconsistency found.** Width64 was still running when this scope was assigned. This review does not inspect its moving files, the global report, the global wall-cap closure, a pair contrast or an archive, and does not call the experiment complete.

## Scope and preserved verification

Read AGENTS.md, the frozen configuration, amended design/implementation plan, cleared scientific/engineering plan followups, the initial implementation findings and their repaired followup, the preproduction gate and retained full-suite output. Reused the saved 585-pass/468.33-second suite evidence after independently verifying its bound log and the other gate artifacts. No suite or production rerun occurred.

Ran `reports/execution/capacity-2026-10-02/reviewer-width32-saved-audit.py` once, successfully, in 1.8485081669932697 seconds. The script restricts all case inspection to `runs/aspirin-capacity-2026-10-02/width32`; it uses authoritative source verification and the saved-only `_validated_case` audit, then separately recomputes endpoint aggregate/group/equal-group metrics, raw gates, residual decomposition and above-baseline sampled observations directly with NumPy. No training, model initialization, prediction or numerical child probe was performed. Imports alone are not counted as numerical model work.

The script, its stdout and `reviewer-width32-saved-verification.json` are preserved under `reports/execution/capacity-2026-10-02/`. The JSON retains all 146 first-arm file hashes, exact metrics and work categories, independent reductions and the two truncated read-only inspection outputs. Those inspection truncations were followed by focused reads and complete-file verification; no numerical conclusion used a truncated array or membership listing. All 146 case-file hashes remained identical before and after the audit. The repository evidence log and Obsidian remain author-owned closeout work for this review; the author should append this review's outcome and identities without replacing prior records.

## Identities and execution checks

The frozen configuration SHA256 remains `655cfd44c0eeacd2e2cb62210b2a78eb5562a417dc01e09a707dfba931f6e960`; canonical configuration identity is `888c9307696c1e114dddb3e24384bc30896e4c5d74b4e8279ea623243c40e26e`. All 20 source pins match their map SHA256 `9611a194fc9c30906f6b9e6a4b66eb4590d279b0799b8299ca0133dc7b0e6fd7`. The recorded production revision is `b45ff2a66923af14a51dce7190b3808810b2a478`; code fingerprint `2066ca4a476c5bca16a7c48312457a0efcb84da60970c3337a2b80a011113a0d`, every participant source/lock hash and all captured dependencies match the preproduction suite identities and current bytes. Later documentation commits are permitted by the recorded code contract.

Checked exact width32 case/model assignment, CPU float64, fresh seed0 state, zero initial Adam moments/count, actual width-specific saved0 blueprint, 13,569 parameters, finite state leaves/nonnegative variance, and Adam update counts for all 63 saved snapshots. The immutable history prefixes contain exactly update0, update1, every100 through6000 and update3001, with the prescribed constant .0003 rate. Every history row's canonical parameter/optimizer hashes binds its saved snapshot. Terminal `commit.json`, checkpoint contract, summary and history agree at6000; the summary SHA256 is `6da242c06ca53274d5faa349a675b317c4d6855db7405cce066c89d64b16685c`.

Exact ordered fitting/anchor IDs, memberships, array/file hashes, original update0 metrics and normalization match the authoritative source. Energy offset is -17647.258992926312 eV, energy loss scale .21583988927697104 eV and force loss scale .1268863230978139 eV/Å. Both fresh zero-head role scores reproduce the original raw baseline. This arm inherits source data/normalization only, and has no inherited trained state in its saved fresh0 checkpoint.

Both discarded oracle receipts bind incoming state, proposed state and actual committed state at1 and3001. Maximum reported absolute differences are `8.326672684688674e-17` and `1.6653345369377348e-15`, respectively, below the recorded absolute tolerance; update3001 records nonzero incoming moments. The reviewer verifies receipt/state binding and finite saved values, without recomputing model gradients. The independently executed production child has error0.0, correct6000 Adam count and saved initial template hashes. Exclusive ordinal1 attempt, token, entry and completion bind case/config/code, directory identity, fresh0 and terminal snapshots/histories, provenance, inputs and endpoint reference; child verification/completion hashes agree.

The arm reports6000 computed, attempted and receipt-persisted updates,768000 fitting-frame exposures and no unknown pending update. Separate work is one initializer on16 frames,128 discarded warmup exposures,9072 scoring role entries,144 endpoint-reference role entries, two128-frame full-gradient and two128-frame accumulated oracle evaluations, two independent NumPy and two discarded accumulated proposals, and one child template initializer on16 frames plus144 child role predictions with zero child training updates. The144 entries span128 distinct fitted structures because16 anchors are included in fitting. Case elapsed1562.8148765840015 seconds is below3000; global closure is outside this preliminary review.

## Learning result and sampled trajectory

Independent endpoint reductions reproduce every saved aggregate, per-group and equal-group metric and both raw joint-gate decisions.

| Role | Energy MSE (eV²) | Force MSE (eV²/Å²) | Energy fraction of original0 | Force fraction of original0 | Joint10% gate |
|---|---:|---:|---:|---:|---|
| Fitting | .01198320842349682 | .0065589689424798165 | .3194094334870974 | .4657052724792342 | Failed |
| Fully fitted anchors | .007114939839938265 | .004530867645870908 | .11607064219031449 | .4471150476714813 | Failed |

Execution integrity passing does not satisfy learning acceptance. The fitting energy fractions exceed the original baseline at2300 (1.4616051173839626),3100 (1.8088490321429394),3700 (3.6020684618113443) and4900 (1.3508647286352125). The largest saved energy observation is3700: fitting raw MSE .13513795338588072 eV² and anchor .1348453640671307 eV² (2.1998201468692145 times its original baseline). At the next saved update3800 these fractions are .46121610845452854 and .0766823764959413, respectively. The next saved fitting fractions after the other above-baseline observations are .48382220501503725 at2400, .5082758791312568 at3200 and .36913940909823895 at5000. Force's maximum saved MSE for both roles remains update0; these facts describe only the stored sampling schedule.

From5500 to6000, fitting energy/force relative changes are -.06663519548328774 / -.028031009823560764; anchor energy/force changes are +.05076265818435542 / -.02887724799393233. Neither establishes convergence. Fitting endpoint energy error has signed mean .00021192289768029449 eV, mean-squared component `4.4911314561212563e-08` eV² and centered variance .011983163512182258 eV². Anchor signed mean is -.047239213184937026 eV, mean-squared component .002231543262331928 eV² and centered variance .004883396577606338 eV². Each decomposition sums to its raw MSE; no gate is recentered.

## Remaining review and scientific limits

Whole-pair review must wait for completed width64, canonical JSON/Markdown reconstruction, root case-record agreement, raw inventory, append-safe evidence-prefix/report/finalization hashes, scientific global-cap closure and byte-verified archive. This preliminary PASS substitutes for none of those checks.

One seed, a selected rate, finite updates and fully fitted supports do not establish generalization, a pure expressivity/capacity cause, continuous stability, spike mechanism, convergence, hardware efficiency or the main force-consistency hypothesis. Same seed is an initialization rule across widths and does not make differently shaped parameters identical. The full training pilot remains pending. Preserve every observed unsuccessful gate and sampled spike beside subsequent results.
