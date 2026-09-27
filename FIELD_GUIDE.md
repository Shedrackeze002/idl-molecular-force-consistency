# The idea, from the beginning: can learned forces guide a molecule to the right shape?

*Research field guide, revised 26 September 2026. This explains the proposed study and its evidence. The team is still reviewing literature; no model has been trained and no result is reported here.*

## The question in one sentence

If a neural network predicts the forces that move a molecule's atoms, can making those forces agree with its predicted energy help the molecule settle into the same stable shape as a reference quantum-chemistry calculation, even when an entire route to that shape was not shown during training?

That is the question in the current [proposal](proposal.tex). It aims at a paper, but a paper-worthy answer requires an effect that survives careful controls and a final check of the literature. Neither a new algorithm nor publication acceptance is assumed.

## 1. The idea, from the beginning

A molecule is a collection of atoms at three-dimensional positions. The atoms can arrange themselves in more than one shape, also called a *conformation*. For a specified electronic-structure calculation, every arrangement has an energy. Imagine a many-dimensional landscape: valleys are stable shapes, and higher regions separate them. A force tells each atom which way it is pulled. In the simplest mathematical statement, force is the downhill slope of energy: $\mathbf F=-\nabla E$.

Calculating that reference energy and force repeatedly with quantum chemistry is expensive. A machine-learned interatomic potential (MLIP) is a faster approximation trained on examples from one chosen reference method. It is not a direct solution of quantum mechanics and does not become experimentally correct just by matching its training labels. Researchers use these models to explore structures and simulate atomic motion. Foundational neural potential work goes back at least to [Behler and Parrinello (2007)](https://link.aps.org/doi/10.1103/PhysRevLett.98.146401); [NequIP](https://www.nature.com/articles/s41467-022-29939-5) gives a modern introduction.

The usual model score is an average prediction error on held-out structures. Force mean absolute error (MAE) asks, roughly, how far the predicted force components are from the reference values at those sampled positions. A low MAE is useful, but it does not prove that following the predicted forces for many steps ends in the right valley. Small local errors can accumulate, and an unfamiliar part of the landscape may be wrong even when familiar examples are well fit.

The network must also respect geometry. Rotating a molecule should rotate its force arrows while leaving its energy unchanged. Models that build this behavior in are called *equivariant*. [PaiNN](https://arxiv.org/abs/2102.03150), [NequIP](https://www.nature.com/articles/s41467-022-29939-5), and [MACE](https://papers.neurips.cc/paper_files/paper/2022/hash/4a36c3c51af11ed9f34615b81edb5bbc-Abstract-Conference.html) are established examples. [E3x](https://arxiv.org/abs/2401.07595) provides equivariant building blocks for the proposed controlled models; using those blocks is not a claim to have invented a new architecture.

## 2. Two ways to predict forces

An **energy-based** model predicts an energy for each arrangement, then calculates forces from its slope. In exact mathematics, the resulting forces are *conservative*: going around a closed path in the space of atomic positions gives zero total work. This is a structural guarantee about the model, not a guarantee that its energy matches chemistry. A poorly trained conservative model can still guide atoms to the wrong shape.

A **direct-force** model predicts a force arrow for every atom without taking the energy derivative at inference. It may save time, but that must be measured on the actual hardware. Direct arrows can be accurate at individual sampled positions yet fail to come from any single consistent energy surface. In the landscape analogy, the model has drawn arrows without necessarily drawing a hill whose slopes explain them. The arrows may disagree with a separate energy prediction from the same network.

The third model in the proposal still predicts forces directly. During training, it receives an extra penalty when its arrows disagree with the downhill direction of its energy prediction. This is **consistency training**. A finite penalty can reduce disagreement, but it does not mathematically guarantee conservative forces. It also adds training work because an energy derivative is needed during training, even if the final direct-force inference avoids that derivative.

The study asks whether that training change improves a practical task: starting from an unfamiliar path configuration, does following the force arrows reach the reference stable shape? Better agreement is useful only if it changes that outcome at a reasonable accuracy and compute cost.

## 3. What previous research has already answered

The broad idea of learned molecular forces is established. [sGDML](https://www.nature.com/articles/s41467-018-06169-2) showed low-data conservative force learning; [PaiNN](https://arxiv.org/abs/2102.03150), [NequIP](https://www.nature.com/articles/s41467-022-29939-5), and [MACE](https://papers.neurips.cc/paper_files/paper/2022/hash/4a36c3c51af11ed9f34615b81edb5bbc-Abstract-Conference.html) are major geometry-aware model families. The proposed E3x implementation is a controlled experimental vehicle, not an architectural breakthrough.

[Bigi, Langer, and Ceriotti (2025)](https://proceedings.mlr.press/v267/bigi25a.html) examined nonconservative direct forces in molecular dynamics and geometry optimization, including physical-consistency tests. A paper whose sole conclusion is that direct forces can cause trouble would repeat known work. [eSEN](https://arxiv.org/abs/2502.12147) likewise emphasizes that static errors and nominally conservative design alone do not establish good physical behavior.

[Zhao et al. (2025)](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202506240) is especially close. It compares direct forces and energy-derived forces in a reaction transition-state-search workflow on Transition1x. Direct models could obtain low force error yet perform poorly in intended transition-state search, in part because Hessians from direct forces were inaccurate or asymmetric. The current proposal therefore cannot claim to be the first direct-versus-energy comparison on transition tasks. Zhao studies *reactive* transition-state search; the proposed experiment studies how consistency training changes **relaxation from withheld conformational descent paths**. These are related but distinct tests.

[Landscape17](https://www.nature.com/articles/s41524-025-01878-x) provides a way to test this narrower question. It identifies stable structures, transition states, and approximate downhill paths between them for several molecules, and reports that evaluated MLIPs can miss reference connections or produce extra stable structures. It also shows that adding path data can help. The present project does not claim to invent that dataset or rediscover the benefit of adding path samples.

A [public OpenReview manuscript](https://openreview.net/pdf?id=cQvBP4TZHe) already studies disagreement between direct forces and energy gradients as a diagnostic. [Takamoto, Zaverkin, and Niepert (2025)](https://arxiv.org/abs/2408.05215) already study conservation-related training losses. These works narrow the possible contribution: the proposed penalty and the disagreement score are tools for a controlled experiment, not new methods by themselves. [UMA](https://arxiv.org/html/2506.23971v2) also discusses direct and conservative training strategies at much larger scale; a two-stage strategy was part of the earlier course design but is no longer the main question.

This is a selective map of the work most relevant to the project's claim, not a survey of every MLIP paper. The exact priority boundary must be checked again before submitting a manuscript.

## 4. Why the dataset changed

The first course idea used **rMD17**, whose molecular-dynamics frames have energies and forces calculated at the PBE/def2-SVP level. It is useful for measuring prediction error near sampled trajectories. But it does not, by itself, provide the full set of reference downhill routes needed for the new question. Its nearby frames are also correlated, so a random frame split can make generalization look stronger than it is. The [rMD17 source](https://arxiv.org/abs/2007.09593) documents the dataset. The locally supplied rMD17 package was inspected during the earlier claim audit.

The current study instead uses **Landscape17** as its primary data source. Its public archive contains atomic positions, energies, and forces on paths from transition states toward connected stable shapes. The [paper](https://arxiv.org/html/2508.16425) lists 37 transition states for aspirin and nine for paracetamol, with path structures associated with them. Those counts are small enough for close scrutiny and large enough to make a whole-path holdout possible, though uncertainty will remain substantial. Aspirin is the primary molecule; paracetamol is a second check.

Landscape17's reference labels use a hybrid DFT method, $\omega$B97x/6-31G(d). The earlier rMD17 labels use PBE/def2-SVP. Those methods define different energy surfaces. Scores or energies from the two sets cannot be merged as if they came from one reference. The proposal therefore trains and compares its core models on Landscape17 alone. No new DFT calculation is required for the core design, but the archive, units, path identities, duplicate structures, and shared endpoints must be audited before training.

A *whole-path holdout* means every point on a selected reference path stays in the same training, validation, or test group. This is more demanding than scattering adjacent path frames across groups. It does not guarantee that a test path is entirely novel: two paths may share a stable endpoint already seen during training. Those endpoints will be marked as seen, and claims will concern withheld **paths**, not wholly unseen molecules or minima.

## 5. Exactly what the team proposes to compare

Three models will use the same equivariant E3x-based backbone and the same Landscape17 training split:

1. **Energy-based baseline.** Predict energy and calculate its force by differentiation.
2. **Ordinary direct-force control.** Predict forces directly, with an energy output but no consistency penalty.
3. **Consistency-trained direct force.** Predict forces directly, while penalizing disagreement with the energy-derived forces during training. Several penalty strengths will be chosen in advance and selected on validation data.

A separately trained [MACE](https://papers.neurips.cc/paper_files/paper/2022/hash/4a36c3c51af11ed9f34615b81edb5bbc-Abstract-Conference.html) model on the same split provides an established conservative reference. It helps show whether the E3x baseline is unusually weak, but one MACE comparison cannot prove that an effect generalizes across every architecture.

The intervention is the change from model 2 to model 3. The scientific question is not merely whether the penalty makes its own disagreement score smaller; that would be unsurprising. The question is whether it changes **shape recovery** after controlling, as far as the data allow, for ordinary force error. The energy-based model anchors the physically constrained end of the comparison.

The study will select model settings using training and validation paths only. It will compare checkpoints in force-MAE ranges that overlap on validation data. If the regularized and unregularized models never achieve comparable error, it cannot honestly attribute a recovery difference to consistency alone. In that case, the result will be an accuracy, cost, and recovery tradeoff rather than a matched-effect claim.

## 6. What the measurements mean

**Force error** measures the average difference between predicted and reference x, y, and z force components on held-out path structures. It reveals local fit, not where a multi-step relaxation ends. Energy error can also be reported for heads with meaningful energy outputs, but the direct head's energy is not guaranteed to govern its forces.

**Shape recovery** is the primary downstream task. Every model begins at the same preselected, slightly perturbed positions from a withheld path. The same force-driven relaxation algorithm and evaluation budget follow each model's forces. A run succeeds only if it converges and its final structure matches the connected reference DFT minimum under a symmetry-aware structural comparison. A run that stops at a different structure is counted separately from one that fails to converge. A stationary direct-force endpoint is a *force-field fixed point*, not automatically a minimum of any energy.

**Closed-loop work** is a physical-consistency check. Imagine moving the atoms around a small route and returning to the start. For forces derived exactly from one smooth energy, the net work around that closed route is zero. A persistent nonzero result for direct forces indicates a nonconservative component. Numerical integration itself creates error, so the calculation must be repeated with finer steps. Lower closed-loop work is interesting, but it does not by itself show better shape recovery.

**Cost** includes training time and GPU-hours, including the extra derivative used by consistency training. Force prediction speed is measured with the same hardware, batching, and implementation choices. Direct-force inference is not automatically faster in practice. Accuracy and cost should both be visible when comparing models.

**Uncertainty** must respect the data structure. Many frames from one path are related, so they are not independent replications. Comparisons use whole path groups and repeated training seeds, with paired starts for each model. Three seeds and a same-split MACE reference are planned. Paracetamol has only nine path groups, so its uncertainty will be wide; it should be treated as a limited replication, not proof of broad molecular transfer.

## 7. What different results would mean

- **Consistency training improves recovery at similar force error.** This would support the proposed mechanism for the tested molecules, split, backbone, and optimizer, especially if the result repeats across seeds and paracetamol. It would still need stronger replication and another prior-work search for a paper.
- **Recovery improves only because force MAE improves.** The penalty may be useful, but the evidence would not isolate a benefit of physical consistency beyond ordinary accuracy.
- **Disagreement and loop work fall, but recovery does not improve.** Better physical-consistency diagnostics do not automatically translate to the chosen task. That is a valuable negative result if uncertainty is small enough to interpret it.
- **Force-error ranges do not overlap.** A matched comparison is unavailable. Report the full tradeoff instead of forcing a causal claim.
- **Models differ across aspirin but not paracetamol, or vary strongly by seed/path.** The effect may be molecule-specific or too uncertain. More data or a redesigned test may be needed before manuscript submission.
- **The energy-based baseline remains best.** This would not be surprising given its exact mathematical constraint. The direct models could still be useful if measured inference savings outweigh the task-performance cost for a specified application.

No outcome has happened yet. A negative result can be scientifically useful, but publication depends on execution quality, statistical resolution, relation to prior work, and the venue's interest.

## 8. Feasibility, failure points, and practical sequence

The core design is feasible using the public archive, but it is ambitious for a course schedule. A single equivariant model, its direct and energy-derived heads, paired relaxations, consistency sweeps, MACE training, three seeds, and two molecules all consume time and compute. A short pilot should measure GPU memory, training time, and relaxation cost before the full grid is promised.

Several details can invalidate an apparently strong finding. Path frames or duplicates must not leak across splits. Shared minima must be identified. Energy and force units must be checked. The energy model's force should be checked against a finite-difference calculation; rotation and translation behavior should be tested. Matching criteria for final shapes, starting positions, relaxation budget, model-selection rule, and statistical unit should be fixed before test results are examined. Training cost of the regularizer must be included rather than hidden. These are design safeguards, not completed experimental results.

The planned order is straightforward. Before the October 23 midterm, parse and audit Landscape17, freeze path groups and starting positions, and verify conservative and direct-force pilots. Before the December 9 course report, run the consistency sweep, same-split MACE reference, paired relaxations, and paracetamol check. A later paper may require broader replication, deeper failure analysis, more independent paths or molecules, and a refreshed prior-art search. The course report is the first milestone, not the manuscript's evidence by itself.

## 9. Is this novel enough for a paper?

The broad direct-versus-energy comparison is **not new**. The warning that low force error may not ensure useful simulations or transition-state search is **not new**. A conservation-related penalty is **not new as a general idea**. [Bigi et al.](https://proceedings.mlr.press/v267/bigi25a.html), [Zhao et al.](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202506240), the [OpenReview manuscript](https://openreview.net/pdf?id=cQvBP4TZHe), and [Takamoto et al.](https://arxiv.org/abs/2408.05215) prevent those claims.

The **candidate contribution** is narrower: at one electronic-structure reference, with whole conformational paths withheld and the model backbone held fixed, test whether changing direct-force energy consistency alters recovery of reference stable shapes beyond any change in static force error, while accounting for training and inference cost. The intervention makes the question more informative than simply finding a correlation between inconsistency and failure. The controlled dataset and paired downstream task make a clear falsifiable experiment.

This remains a *candidate* gap. An exact priority claim would require a wider literature review, and a publishable empirical claim requires convincing measurements. If the effect is weak, unstable, or explained by MAE, the paper direction may need to change. The team should be willing to report that honestly rather than relabel an incremental result as a new method.

## 10. Where this fits in the wider field

Many other MLIP questions lie beyond this project. Researchers build models transferable across many molecules rather than one molecule at a time; [SPICE](https://www.nature.com/articles/s41597-022-01882-6) and [MACE-OFF](https://pubs.acs.org/doi/10.1021/jacs.4c07099) are relevant data and model examples. Others study long-range interactions, charge response, solvents, periodic systems, uncertainty-guided data collection, reaction discovery, and reliable thermodynamic or kinetic observables. A [uniform pretrained-model benchmark](https://pubs.acs.org/doi/10.1021/acs.jctc.6c00130) illustrates why speed, memory, stability, and the electronic-structure reference matter alongside MAE. This proposal intentionally isolates one question; it will not establish reaction rates, complete transition networks, or experimental molecular behavior.

## 11. Short glossary

- **Conformation:** one three-dimensional arrangement of the atoms in a molecule.
- **Energy landscape / potential energy surface:** the energy assigned to every possible atomic arrangement by a chosen calculation or model.
- **Force:** the direction and strength of the pull on an atom; for a conservative model it is the negative energy gradient.
- **Minimum:** a locally stable shape on an energy landscape.
- **Transition state / descent path:** a high-energy connecting region and an approximate downhill route toward a minimum.
- **DFT:** density functional theory, an approximate electronic-structure method used to provide reference labels here.
- **MLIP:** machine-learned interatomic potential, a model for molecular energies and/or forces.
- **Equivariance:** rotating the input molecule rotates predicted vector forces in the same way.
- **MAE:** mean absolute error, the average size of prediction mistakes under a specified convention.
- **Conservative force:** a force field that is exactly the negative gradient of one energy function.
- **Relaxation:** repeated movement following forces until the model reaches a stationary arrangement or the run fails/stops.
- **Closed-loop work:** accumulated force along a route returning to its start; nonzero converged work detects nonconservativity.

## 12. Suggested reading order for a beginner

1. Start with Sections 1, 2, 4, and 6 of this guide. The key distinction is *getting arrows right at sampled positions* versus *following arrows to the right destination*.
2. Read the introductions of [NequIP](https://www.nature.com/articles/s41467-022-29939-5) or [PaiNN](https://arxiv.org/abs/2102.03150) for molecular graphs, energies, forces, and symmetry.
3. Read [Bigi et al.](https://proceedings.mlr.press/v267/bigi25a.html) for direct-force physical consistency, then [Zhao et al.](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202506240) for a task where force MAE and useful search diverge.
4. Read [Landscape17](https://arxiv.org/html/2508.16425), especially its introduction, dataset description, and results about missed paths and extra minima.
5. Read [Takamoto et al.](https://arxiv.org/abs/2408.05215) and the [OpenReview manuscript](https://openreview.net/pdf?id=cQvBP4TZHe) to see why neither consistency training nor disagreement as a diagnostic can be claimed as entirely new.
6. Return to the [proposal](proposal.tex) and [claim audit](CLAIM_AUDIT.md). The proposal states what will be tested; the audit separates established facts from hypotheses and risks.

## Primary sources and further reading

- [Behler and Parrinello, neural potential energy surfaces (2007)](https://link.aps.org/doi/10.1103/PhysRevLett.98.146401).
- [sGDML, symmetry-aware conservative molecular force fields (2018)](https://www.nature.com/articles/s41467-018-06169-2).
- [Christensen and von Lilienfeld, rMD17 (2020)](https://arxiv.org/abs/2007.09593).
- [PaiNN, equivariant message passing (2021)](https://arxiv.org/abs/2102.03150).
- [NequIP, equivariant interatomic potentials (2022)](https://www.nature.com/articles/s41467-022-29939-5).
- [MACE, higher-order equivariant model (2022)](https://papers.neurips.cc/paper_files/paper/2022/hash/4a36c3c51af11ed9f34615b81edb5bbc-Abstract-Conference.html).
- [E3x, equivariant model framework (2024)](https://arxiv.org/abs/2401.07595).
- [Bigi, Langer, and Ceriotti, nonconservative force risks (2025)](https://proceedings.mlr.press/v267/bigi25a.html).
- [Zhao et al., MLIPs and transition-state search (2025)](https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202506240).
- [Takamoto, Zaverkin, and Niepert, physics-informed training losses (2025)](https://arxiv.org/abs/2408.05215).
- [Cărare et al., Landscape17 (online 2025; volume 2026)](https://www.nature.com/articles/s41524-025-01878-x).
- [Fu et al., eSEN and practical energy conservation (2025)](https://arxiv.org/abs/2502.12147).
- [When Forces Disagree, public manuscript under review (2026)](https://openreview.net/pdf?id=cQvBP4TZHe).
- [Wood et al., UMA and large-scale molecular potential training (2026 revision)](https://arxiv.org/html/2506.23971v2).
- [Eastman, Pretti, and Markland, pretrained-model benchmark (2026)](https://pubs.acs.org/doi/10.1021/acs.jctc.6c00130).
- [SPICE molecular dataset](https://www.nature.com/articles/s41597-022-01882-6) and [MACE-OFF transferable organic potentials](https://pubs.acs.org/doi/10.1021/jacs.4c07099).
