# Representation constraints and loss-gradient diagnostic design

30 September 2026. Status: proposed, not executed. User authorized proceeding with the next investigation after the completed coverage comparison. Native bounded diagnostics extend the existing verified flow; two independent plan reviews precede implementation. No model architecture, learning protocol, labels or loss weights change in this milestone.

## Question and approach

Coverage failed all force gates despite fitting energies. Compare three approaches: immediately enlarge the network (changes capacity without identifying the mechanism); change loss weights or train longer (changes optimization before measuring it); inspect necessary representation constraints and loss gradients at preserved parameters (chosen). The investigation measures mechanisms, not a trained intervention or causal explanation.

The invariant energy model imposes zero net force and zero centered torque. For each frame, centered positions r_i and six infinitesimal rigid-motion columns Q (translations e_a and rotations e_a cross r_i) satisfy Q^T F_model=0. Orthogonal projection of the target force onto the column space of Q is an unavoidable component for this invariant model class, independent of width or optimization. Column-normalize nonzero Q columns and compute thin SVD with relative rank cutoff1e-12, recording rank/singular values. This is a numerically evaluated necessary-constraint lower bound, not a sufficiency/integrability/capacity theorem. Center positions by unweighted atom mean because force MSE weights components equally; do not mass-weight.

Projection is observational only. Never change force labels or silently use projected labels for fitting. A large rigid-motion component indicates model/target incompatibility under the enforced symmetry; it does not prove data corruption, a unit error, or an incorrect physical calculator. Numerical approximations/external constraints could violate ideal symmetries. Derivation is local algebra from this model, not imported dataset assumptions. The official E3x force-field example supports negative energy derivatives and cutoff-message semantics: https://e3x.readthedocs.io/stable/examples/md17_ethanol.html ; its ethanol accuracy/split/configuration is not evidence for this project.

## Frozen scope

Use the existing aspirin prepared identity4b9f220201d6483e2b2a8da74c10330c3762942ff4c6fabe6389ee27249c43aa, split73d5475bfc1f1158da62b939f5431b1781db886443cece2cdb297c8dd645133a, coverage subsets identity8139e442fca011efcbeb5f2d28d701c5f36030d61f045e5e652b98eb18b1be47 and all fixed16/16/128 IDs/orders. Open train.npz only plus saved training-only run artifacts. No validation/test data. Hash-pin original coverage root/config/subsets/provenance and each arm's config/provenance/summary/history/inputs/reference/checkpoint/verification before loading prepared arrays. Check source settings, normalization, memberships, array identities, checkpoint update2000 and verified predictions. Every historical artifact remains immutable.

Use seed0 zero-head fresh parameters and saved final A/B/C parameters from `runs/aspirin-data-coverage-2026-09-30`, with original model16features/degree1/2iterations/8basis/5Å, Adam1e-3 state and train-only scales. Assert recreated fresh parameter/optimizer hashes match all original arms. Capture current code/lock identity separately from historical execution provenance.

## Phase A: necessary constraints and neighborhood support

For each16/16/128 fitting subset, save per-frame and pooled translation-only and combined translation/rotation lower-bound MSE, zero-force baseline, fraction of baseline, centered net-force/torque vectors, numerical rigid rank, IDs and group/branch metadata. Save residual and projected-component checks: reconstruction, orthogonality and Pythagorean error decomposition. Combine pooled values by component count, not unequal frame/group weights. Report equal-group summaries separately.

Validate saved final fitting/anchor predictions against the same constraints; normalized-Q projection maximum component<=1e-8eV/Å. Recompute force MSE and decompose into necessary floor plus compatible residual MSE, with atol1e-10/rtol1e-8. Compare lower bound with each original immutable force gate; floor above gate+1e-10 is a numerically obstructed necessary condition. A floor below the gate only means these constraints alone do not explain failure. Do not subtract the floor from the acceptance rule or declare any old learning check passed.

For each geometry, inspect the actual strict distance<5Å cutoff graph: degree range, isolated atoms, connected components, directed pair density, fraction of ordered nonself pairs reachable within two hops, and pairs within1e-8Å of cutoff. Connectivity/two-hop coverage is descriptive, not proof of expressive sufficiency. No positions, cutoff or neighbors are changed.

## Phase B: loss-gradient decomposition at frozen states

Eight cases, fixed order: fresh-A fitting, fresh-B fitting, fresh-C fitting; final-A fitting, final-B fitting, final-C fitting; final-B original16 anchor, final-C original16 anchor. Common fresh/A anchor equals A fitting and is not duplicated. No trained state is reused as initialization for training, because no training occurs.

Use original normalized objectives L_E=energyMSE/Escale² and L_F=forceMSE/Fscale². Microbatch16, sequential at the same parameters, frame-count weighting, exactly as verified coverage accumulation. Compute both objective values and separate parameter gradients; independently obtain joint gradient using the existing normalized-loss/accumulation helper and verify g_joint=g_E+g_F (atol1e-10/rtol1e-8). Save whole-tree, backbone, head-kernel, head-bias and per-leaf L2 norms, dot products, cosines, relative norm ratio and directional derivatives of raw joint-gradient descent. Zero norms produce null cosines/ratios, never manufactured agreement/conflict. Force head-bias gradient is structurally zero; zero-head backbone force gradient is expected at initialization and is not by itself a bug.

Parameter finite-difference verification: for each E/F gradient use its unit-L2 direction if norm>1e-12, otherwise a fixed seeded Rademacher unit-tree direction. Central differences at steps1e-4 and1e-5; require at least one agreement with autodiff directional derivative (atol1e-8/rtol1e-4), record both results, accept neither by omission. For all8cases restore exactly incoming parameters after probes: shifted parameters are ephemeral function evaluations, not saved checkpoints or optimizer updates. Capture original parameter and optimizer hashes before/after; they must match. No Adam update is computed or applied. Orientation diagnoses ordinary gradient descent locally; it does not establish Adam's actual update direction, dynamics throughout training or causal loss conflict. Per-leaf norms do not measure network capacity.

## Controls, budget and operational gates

Float64 CPU, no new dependencies, one execution in a fresh protected directory, no resume/retry/tuning. Whole analysis cap600s including preflight, compilation, static scoring, finite differences and output writing; check between operations with10s output reserve, not hard interruption. Each phase/case is prospectively logged and immediately retained after outcome; failure/wall/nonfinite/identity blocks remaining cases explicitly. Save completed observations even if later stages fail. No training loop, optimizer updates, new trained teacher/capacity variant, loss/LR change, added seed or full pilot. Engineering test fixtures are separately labelled, not research trials.

Test SVD projection against analytic pure translation, pure rotation, invariant harmonic forces, superpositions, collinear/single-atom degeneracy, rotation/translation/coordinate-scale covariance and direct scalar constrained least-squares oracle. Test graph connected/disconnected/boundary/two-hop examples. Test split gradients against existing full objective with unequal3/3/1 chunks and mixed force-backbone derivatives; test finite differences and zero-norm reporting. Test stale sources before loader, modified checkpoint/settings, protected dirs, partial failures/caps, complete count/IDs and source-state preservation. Full suite passes before execution.

## Interpretation and next decision

If prediction-constraint/gradient identity/finite-difference/source verification fails: preserve result and address implementation/provenance before considering training changes. If a numerical symmetry floor obstructs a force gate: inspect source/calculator/frame/force conventions in a new reviewed plan, with no label correction now. Otherwise report gradient orientation/norm evidence and its limits. Negative cosine or dominance may motivate a reviewed loss/optimization control; absence does not establish adequate capacity. If cause remains unresolved, save that outcome and formulate a bounded capacity/intervention plan separately. None of these static diagnostics clears the learning gate, identifies a cause conclusively or authorizes full training.

## Deliverables

Frozen config, saved implementation plan and two reviews; CLI `diagnose-gradients --config`; per-phase/case JSON, preserved failures/blocks, source/config/array/state/code identities, report and scientific plots with generator identities; append-only paper evidence, retained execution/review workspace and current Obsidian navigation. Keep user proposal changes and local branch intact. A fresh final implementation/evidence reviewer checks the complete milestone; preserve every failed engineering check and substantive ruling.

## Review clarifications (both reviewers cleared; no protocol change)

The source hash map must contain the complete required31-file allowlist; reject omitted pins before any prepared-array load. Enforce prediction leakage as max(abs(P F_model)) in force-component units eV/Å; normalized-column dot products are separate descriptive values. Report force-gradient norm divided by energy-gradient norm explicitly, and raw-descent derivatives as -(||g_E||²+g_E·g_F) and -(||g_F||²+g_E·g_F). These statistics depend on parameterization and fixed loss scales; directional finite differences check selected directions, not every component. Start the budget timer before all source checks. Branch metadata comes from the validated numeric_path parser. These clarify implementation/reporting and do not change frozen sources, cases, tolerances or caps.

## Execution update

Completed once on production revision b68b017,8.9605757s, all8cases/PhaseA verified. Initial proposed status above records design history; outcomes are now in reports/representation-gradient-diagnostics.md/json and the append-only evidence log. Independent final review found two failure-handling/reporting bugs; both fixed with regressions and117passing final tests. Research artifacts unchanged; full pilot/interventions remain pending separate reviewed plans.
