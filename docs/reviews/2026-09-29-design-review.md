# Independent review of the first implementation design

Date: 29 September 2026

Target: `docs/superpowers/specs/2026-09-29-data-baseline-design.md`, original commit `d4ef0bc`.

Two distinct subagents reviewed the saved design without editing it or seeing each other's findings. The scientific reviewer checked proposal alignment, leakage, molecular matching, and statistical claims. The engineering reviewer checked data contracts, E3x/JAX feasibility, scaling, precision, and reproducibility. Both returned **ready for implementation planning**, with no demonstrated architectural blocker. Their verdicts concern the design; they do not establish passing code or successful training.

## Findings and resolution

| Reviewer | Finding | Resolution |
| --- | --- | --- |
| Engineering | Scaling must preserve the physical energy-force derivative; large absolute energies require centering before lower precision. | The design now specifies a physical eV residual head, training-only energy offset, residual-only loss scaling, and explicitly asserted float64 for this pilot. |
| Engineering | Several symmetry-equivalent mappings to one reference shape should not create ambiguity. | The accepted identity is a reference equivalence class; save a deterministic mapping and competing class distances. Permutations must preserve species and connectivity between structures. |
| Both | Preparation matching and duplicate thresholds cannot remain deferred until relaxation evaluation. | Freeze the preparation policy and coverage rules before splits; downstream model-success thresholds remain a separate later protocol. |
| Both | Validation aggregation and checkpoint selection were underspecified. | Evaluate all eligible validation records once, including partial batches; report pooled and equal-connection MAE, selecting by the latter. |
| Engineering | A scalar-only final backbone could undermine the later matched architecture. | The first backbone retains equivariant features; scalar truncation is head-local. A later force head must use polar-vector rather than pseudovector features. |
| Scientific | Whole connections are grouping units, not automatically independent observations. | Replaced the independence claim with connection-group count; record endpoint identities and prohibit counting frames or repeated seeds as independent observations. |
| Engineering | Deterministic scientific identity must be separated from incidental metadata. | The implementation plan must define canonical manifests/hashes separately from timestamps, paths, and device/run metadata. |
| Engineering | Pilot defaults and backend checks need precise contracts. | The implementation plan must pin optimizer, budget, metrics, overfit criteria, and requested-backend verification. |

## Independently checked raw example

All 405 local sample checksums passed both the initial audit and the scientific review. The parent agent independently reproduced the review's aspirin connection `01` example: fixed-index alignment of `coordsminus_122.xyz` to `coordsmin_1.xyz` gives RMSD 0.688446809856698 angstrom. Swapping reference oxygen atom indices 7 and 9 (zero-based, atom rows 8 and 10) before orthogonal alignment gives 0.0010909448463713434 angstrom. The fixed-index plus-branch match to minimum 0 is 0.003096956893101937 angstrom.

This demonstrates that atom-order-only matching would miss a close geometry match. It is not a full validation of a chemically constrained matching implementation. Such an implementation still needs an explicit connectivity policy, complete equivalence-class handling, and failure tests. The original exploratory permutation by itself establishes a geometric match, not the full chemical matching contract.

## Remaining audit gates

Archive unit provenance, full connection-to-transition-state resolution, full duplicate/removal counts, transformed-label consistency, branch/reference coverage, and sufficient eligible records are still unverified. The design correctly requires these before the relevant exports or training. The planned data audit must record failures rather than turn missing evidence into assumed correctness.

The source paper describes approximate descent paths, so later results should refer to recovery of the archived branch destination under the chosen optimizer. Archive membership does not prove an optimizer-independent basin assignment.

## Standing workflow

Shedrack explicitly requires two independent subagent reviews of every implementation plan before execution. This is recorded in `AGENTS.md` and in a user-authorized long-term preference note. Material revisions receive follow-up review; findings are verified rather than accepted automatically.
