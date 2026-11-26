# From Atoms to Our Results: Landscape17 Project Explainer

A team and professor guide with no required background

Snapshot: 2026-10-21T14:40:05+02:00

Language: ASD-STE100 Issue 9 writing rules with point-of-use scientific explanations; no independent language-conformance certification

## Page 1: FROM ATOMS TO OUR RESULTS

Team and professor meeting | CMU Introduction to Deep Learning | Group 9 | 21 October 2026

Research title: Do Consistent Learned Forces Find the Right Molecular Shape?

**START HERE**

A molecule contains atoms. Our computer models predict the pushes and pulls on those atoms. We ask where those predictions move the molecule.

This guide teaches each idea before using it to discuss an experiment. You do not need prior chemistry, physics or machine-learning knowledge.

The early pages give the background to the problem. Later pages give the full research history, the results, the limits and the next tasks.

### The current answer, in ordinary language

We built models that predict atomic forces. One extra training rule made two kinds of force prediction agree much more closely.

That agreement did not make our first shape-search procedure reach the correct target. Every model in that pilot reached zero correct targets.

A separate completed comparison found lower force error when our small network kept internal directional information. The training recipe limits that conclusion.

### Reading routes

| Need | Read |
| --- | --- |
| A speech you can use immediately | Pages 2-3. Each speech section supplies its own background. |
| Understanding from zero | Pages 4-13, then the experiment history. |
| Why our direction changed | Pages 18, 28 and 31. |
| Meeting decisions | Pages 35-36. Questions and answers follow. |

The glossary is a reminder only. The main text gives the meanings beside the ideas and results that use them.

Scientific records are unchanged. The same sources support this rewritten edition [1]-[24].

## Page 2: A spoken introduction from zero

Our project starts with atoms, the tiny pieces of matter that join to form a molecule. We study an aspirin molecule with 21 atoms.

Its molecular shape means the arrangement of those atoms in three dimensions. That arrangement can change when parts of the molecule move.

Forces are pushes and pulls on the atoms. If we know the forces, a computer procedure can use them to change the shape.

A detailed physics calculation gives energy and force answers for each shape. We use these answers as our reference, but repeating the calculation is expensive.

We teach the model with examples. Each example contains an atomic arrangement, its reference energy and the reference forces.

Energy is a number associated with the arrangement. Think of a landscape where height represents energy, rather than physical height.

A stretched spring is a simple analogy: changing its shape changes stored energy and the pull it produces. The molecule is not a literal spring.

Slope means how quickly the energy height changes as atomic positions change. An energy-derived model predicts energy, then calculates forces from those slopes.

A direct-force model predicts the force arrows themselves. Our direct models also predict an energy number, but the arrows can disagree with its slope.

Consistency training adds an extra error penalty when those two predictions disagree. It encourages agreement; it does not guarantee correct answers.

**OUR ORIGINAL QUESTION**

If we encourage the direct arrows to agree with the energy slope, will the atoms reach the correct reference shape more often?

We compare models on the same starting arrangements. We compare force errors and record the computing time, so an apparent improvement has context.

The target shape comes from the reference calculations. A computer stop counts as success only when the final shape also matches that target.

Opening context and the proposed question [1]. The energy landscape is an analogy, not a measured drawing.

## Page 3: What happened and where we are

Our first small model still made large force errors on examples it had already seen. We examined the data and the training settings.

MACE, a different computer model made for atoms, later gave much closer answers on the same training examples. The data were not impossible to learn.

A network is a computer model with adjustable numbers. Layers are its stages of calculation. Training changes its numbers to reduce prediction errors.

We then trained larger versions of our small model. We built them with E3x, a software toolkit for processing atomic arrangements.

The full-data models learned useful force predictions. The agreement penalty also made direct forces much closer to their predicted energy slopes.

But every evaluated model recovered zero of ten target shapes with our first procedure. Even our energy-derived comparison model recovered zero.

This result cannot tell us which method recovers shapes better. We examined both the models and the procedure that moves the atoms.

Some models still predicted large forces at stable reference shapes. Some accepted steps also moved atoms so far that the model treated fragments as separate.

These observations do not identify one cause for every failure. We need a check using a stronger comparison model and a safer rule for moving the atoms.

Meanwhile, a professor suggested a comparison of how networks handle direction. We completed a related study of scalar and vector internal features.

Scalar features carry numbers without direction. Vector features also carry directional information. If we rotate the whole molecule, the force arrows should rotate with it. Both model types keep that rule.

The model with vector features had lower force error in both runs. We used its existing training recipe, so this is a limited comparison.

**WHERE THE MEETING SHOULD END**

We need agreed owners for the remaining model connector, shape-search controls, physical checks and timing work. The original recovery question remains open.

Completed work and limits [5]-[14]. Later pages unpack every model name, score and control in this summary.

## Page 4: Atoms, shapes and force arrows

### What is a molecular shape?

Atoms are very small units of matter, such as carbon, hydrogen and oxygen. Chemical bonds hold atoms together in a molecule.

Aspirin is the molecule we use for the main study. Our data contain one aspirin molecule at a time, with 21 atoms.

A geometry is a list of their positions. Each atom has three coordinates: positions along left-right, forward-back and up-down directions.

The coordinates locate the atoms. The pattern of distances and orientations between them gives the molecular shape, also called a conformation.

Turning the entire molecule does not change its internal shape. Bending or turning one part relative to another can change that shape.

### What does a force prediction contain?

A force is a push or pull. We show it as an arrow because it has both size and direction.

The model gives three force components for each atom. A component is the amount of force along one coordinate direction.

For 21 atoms, one force prediction contains 63 component values. These are predicted pushes and pulls, not new atomic positions.

A separate coordinate-update procedure decides how far to move each atom using the forces. A large force does not itself specify a displacement.

**WHY THE DISTINCTION MATTERS**

We can have an inaccurate force model, an unsuitable movement procedure, or both. The final shape depends on their combined behavior.

### What does correct shape mean here?

We compare the final arrangement with a stored reference arrangement. We remove whole-molecule rotation and translation before measuring their difference.

Translation moves the whole molecule without changing its internal shape. Alignment turns and shifts one geometry to overlap the reference.

Some atoms have equivalent roles in the bonding pattern. Our matching rules permit those atom exchanges, not arbitrary swaps of every atom of one element.

Data geometry, atom order and matching controls [1]-[2]. Force is an interaction, not velocity or a displacement instruction.

## Page 5: Energy and the slope of a landscape

Each atomic arrangement has an associated energy. Our reference answers come from a detailed physics calculation.

Electrons are tiny particles in atoms. Density-functional theory (DFT) models their behavior to calculate molecular energy.

DFT gives our reference energies and forces. These are approximate calculations, not laboratory measurements or exact physical truth.

One molecule can have several locally stable shapes. Each is like a valley: small changes of internal shape increase its energy.

We call such a shape a local minimum. Local means stable against nearby changes, not necessarily the lowest energy among every possible shape.

Illustration: an energy landscape with two valleys and a pass. This is not measured aspirin data.

### How does energy give forces?

Imagine changing one atomic coordinate by a very small amount. The change in energy tells us the slope along that coordinate.

The coordinate gradient collects these slopes. The negative coordinate gradient gives the force, which points downhill in this energy analogy.

A real molecular landscape has many coordinates. The drawing above only illustrates the idea; it is not an aspirin calculation.

### Why does an energy offset not change forces?

Adding the same number to every energy lifts the entire landscape without changing any slopes. The predicted forces stay the same.

We therefore use a fixed training-derived energy offset. Residual energy means the energy after we subtract that fixed number.

**KEEP THIS FOR THE DIAGNOSTICS**

A large error in the overall energy level can coexist with a smaller error in energy differences. Their effects on forces are different.

Reference method, energy convention and force derivative [1]-[3]. Illustration only; its heights and shapes are not project measurements.

## Page 6: What a learned model actually does

### Start with one example

The input tells the model the atom types and all atomic positions. For aspirin, this means 21 atoms and 63 coordinate values.

The desired output contains one reference energy and 63 reference force components. A label is one of these desired reference answers.

A neural network processes the input through numerical layers. Parameters are its adjustable numbers, like knobs that change the output.

Training means changing those parameters so the model gives answers closer to the reference labels.

Prediction means using the current parameters to produce answers. A prediction can be made on a familiar example or a new arrangement.

### What do E3x and MACE mean?

| Name | Role in our project |
| --- | --- |
| E3x | A software library of network layers that handle three-dimensional geometry. We built our small experimental networks with these layers. |
| MACE | A different atomic network architecture from prior research. We trained it on our data as an established comparison model. |

These names do not name the data. They identify different model-building tools or architectures, meaning different arrangements of computation.

### Why do we use a graph?

A graph is a set of nodes and links. We make atoms the nodes and nearby atomic neighbors the linked pairs. These neighbor links are computational links, not a claim that every pair has a chemical bond.

Message passing means that linked atoms exchange numerical information inside the network. This helps each atom account for its local surroundings.

Our E3x network uses two message-passing interactions. These are two rounds of information exchange, not two atomic-movement steps.

**ANSWER TO THE PROFESSOR**

We already use graph neural networks. A graph is how we organize atomic neighbors; it is not a new direction we still need to implement.

Model implementation and common input/output conventions [1], [7]-[8], [16]. No reference-model score is implied by its name alone.

## Page 7: Three ways to predict the forces

### 1. Energy-derived forces

The network predicts one energy for the arrangement. We calculate the coordinate gradient, the energy change along each coordinate.

We use its negative value as the force. A force field is the rule that gives forces for different atomic arrangements.

Work means energy transferred by a force as atoms move. A closed loop returns all atomic positions to their starting arrangement.

Such a force field is conservative. In the exact mathematical calculation, moving around a closed loop gives no net work.

Conservative does not mean correct. A wrong energy surface can still generate perfectly conservative but wrong forces.

### 2. Ordinary direct forces

A direct-force head predicts the force arrows itself. A head is the final output part of a network.

Our direct network also has an auxiliary energy head. Auxiliary means an additional energy output trained alongside the force output.

Ordinary direct training fits both outputs to labels. It has no extra agreement penalty comparing the direct arrows with the energy slopes.

### 3. Consistency-trained direct forces

This model keeps the same direct and energy outputs. During training, we also compare the direct arrows with the auxiliary energy slopes.

An extra error penalty increases when the two force predictions disagree. This encourages consistency, meaning agreement between these two outputs.

Lambda is the numerical strength of that penalty. Lambda zero gives ordinary direct training; positive values add the agreement rule.

**AGREEMENT IS NOT ACCURACY**

Two predictions can agree and both be wrong. We must still compare them with reference labels and measure where their forces move the atoms.

### What counts as the controlled comparison?

We keep the network body, training data and update budget similar across these E3x arms. We record parameter counts and actual computing costs.

MACE remains a separate comparison. Its architecture and training budget differ, so its result cannot isolate one E3x design choice.

Original hypothesis and actual direct/energy/consistency objectives [1], [8]. Closed-loop model measurements remain a separate unfinished task.

## Page 8: Paths, branches and connection groups

A stable shape is a local energy minimum: a valley where nearby changes of internal shape raise energy. A transition state resembles a pass between valleys.

It is unstable along a descent direction. Starting on either side can give a path downhill toward a minimum.

Landscape17 stores atomic arrangements along these descent paths. A branch is one of the two descent directions from a transition state.

A connection contains the transition state and both branches. Our connection group means all the records that belong to that connection.

Minimum A → Branch A → Transition state → Branch B → Minimum B

### Why keep a whole connection together?

Neighboring stored arrangements on a path can be very similar. Splitting neighboring records randomly could give the model an almost identical training example.

We instead put each whole connection in one partition. A partition is one of the separate data collections used for training or evaluation.

Both branches stay together. A model cannot train on one branch while we treat the other branch as a fully unseen connection.

### What does unseen mean here?

A held-out connection is a path group that the model did not use to adjust its parameters.

Different connections can still reach the same minimum. A new route can end at a familiar destination.

**THE LIMIT OF OUR SPLIT**

Our training partition contains all 11 aspirin reference minima. Held-out connections are new routes, not necessarily new destination shapes.

The same five validation connections appear across models. Running more models on those connections does not create more independent connections.

Connection-level split and shared-minimum audit [1]-[2]. The flow diagram is schematic, not a measured energy profile.

## Page 9: The data, the targets and the correction

Landscape17 is the dataset: the stored atomic arrangements, reference energies, forces and path identities that support this study. For aspirin, it contains 37 connections and 74 branches. The archive also supplies 11 reference minima, meaning stored stable target shapes.

rMD17 is another molecular dataset with different reference calculations. We did not mix its labels with these Landscape17 labels.

| Partition | Whole connections | Eligible records | Purpose |
| --- | --- | --- | --- |
| Training | 26 | 4,891 | Change model parameters using known examples |
| Validation | 5 | 974 | Select settings, select saved models and examine recovery |
| Main test | 6 | 1,127 | Final evaluation after the study choices are fixed |

A record is one eligible atomic arrangement with its labels and identity. Thousands of records do not mean thousands of independent paths.

We have not scored a trained model on the main test partition. Earlier model-free software fixtures did parse its coordinates and labels.

Those fixtures did not select models or movement settings. We keep that historical access separate from trained-model test evaluation.

### Why the target mapping needed correction

Each branch needs its own correct destination. A filename pair can name minima without giving the correct plus/minus branch order. Using the pair positions gave the wrong target to 26 of 74 branches. We changed the shared interface to use empirical endpoint classifications.

Empirical classification means matching the stored branch endpoint geometry to the reference minima. It does not mean trusting the filename.

For example, connection 66 has filename pair 6/6, but its two classified destinations are minima 6 and 9.

### Why paracetamol is not part of the current result

Å means angstrom, an atomic-distance unit equal to one ten-billionth of a meter. The matching rule uses this unit.

Paracetamol is the second molecule planned for a repeat study. One branch still misses the frozen structural-match rule.

We align the shapes and combine the distances between matching atoms. The distance score is 0.06431 Å, above 0.05 Å. We did not loosen the rule after seeing this failure.

This mismatch does not show bad labels or a new stable shape. We deferred the repeat study while resolving aspirin feasibility.

Unit audit, split and branch targets [2]; we did not independently recompute DFT.

## Page 10: Units and error scores, with an example

### The units give the scale

An angstrom, written Å, is one ten-billionth of a meter. It is useful for atomic distances; many bond lengths are of this order.

An electronvolt, written eV, is a small energy unit. A millielectronvolt, written meV, is one-thousandth of an electronvolt.

Force uses energy divided by distance: eV/Å or meV/Å. Multiply an eV/Å value by 1,000 to express it in meV/Å.

### Mean absolute error: average size of the mistakes

Mean absolute error, or MAE, ignores whether a component error is positive or negative. It averages the error sizes.

Illustration only: suppose three component errors are -2, +1 and +3 in one force unit. Their absolute sizes are 2, 1 and 3.

Their MAE is (2 + 1 + 3) / 3 = 2. This example teaches the calculation; it is not a project measurement.

### Mean squared error gives large mistakes more weight

Mean squared error, or MSE, squares each error before averaging. The illustrative squared errors are 4, 1 and 9.

Their MSE is 14 / 3 = 4.667 squared units. Root mean squared error, or RMSE, takes the square root: approximately 2.160 original units.

A force-MSE fraction of 0.1 means 10% of the starting squared error remains. It does not mean 10% of the starting RMSE remains.

### How we average the validation paths

Equal-connection force MAE first calculates each connection's average component error. It then gives each of the five validation connections equal weight.

Pooled MAE instead averages all components together. A connection with more stored records then has more influence.

**TABLE READING RULE**

Lower error is better. Compare the same metric, units and data collection. Pooled MAE, equal-connection MAE and RMSE are not interchangeable.

Metric definitions and fixed record/component counts [7]-[8]. All calculations in the worked example are illustrative.

## Page 11: Training knobs, losses and two gradients

A network parameter is an adjustable number. Training repeatedly changes these numbers to reduce an error score called the loss.

A training update is one parameter adjustment. The optimizer is the algorithm that chooses it; our final E3x recipe uses Adam.

The learning rate controls the size of an adjustment. A batch is the collection of records used to calculate one update.

Width counts internal feature channels at each atom. A channel is a parallel slot for learned information. Features are not extra reference labels.

### Why use a weighted loss?

Energy and force have different units and typical sizes. Normalization rescales their errors so we can combine them in one numerical objective.

We calculate the scales from training data only. The physical predictions still use eV for energy and eV/Å for forces.

A loss weight sets how much one error contributes to training. Changing it can improve forces while worsening energy predictions.

**OUR THREE-TERM OBJECTIVE**

The loss combines normalized energy error, normalized force error and, for consistency models, direct-force disagreement with the energy slope.

### The word gradient refers to two different calculations

A coordinate gradient asks how predicted energy changes when atomic positions change. Its negative value gives energy-derived forces.

A parameter gradient asks how the loss changes when a model parameter changes. The optimizer uses it to choose a training update.

Our early gradient-conflict diagnostics compared parameter gradients of energy loss and force loss. They did not compare two physical force vectors.

Opposing parameter gradients can pull an update toward competing objectives at a particular saved state. This does not show the entire training-failure cause.

Shared objective and normalization [3], [7]-[8]. Fixed final weights are energy 0.014584365670390719 and force 1.9854156343296092.

## Page 12: Updates, seeds and saved checkpoints

We train the final small models for 200,000 parameter updates. Each update uses a batch of four records.

This gives 800,000 sampled record exposures. Sampling uses replacement, so a record can appear again before every other record appears.

One epoch usually means a pass through the training data. The exposure count is approximately 163.57 dataset passes, not 200,000 epochs.

A seed controls random initialization and sampling. Two seeds give two training runs, but they do not create new validation connections.

### A checkpoint is a saved training state

We save the network parameters and the bookkeeping needed to continue training. This includes optimizer and random-generator state.

LAST is the final full-budget checkpoint. BEST is the first checkpoint with the smallest validation force MAE among the scheduled checks.

We calculate validation at update zero and every 500 updates. This gives 401 checks for a completed 200,000-update run.

Update zero is before learning. If BEST is zero, training did not beat that initial validation force score under the selection rule.

### Why the best score needs context

Illustration only: validation errors of 40, 35 and 39 give BEST 35 and final error 39. The selected best is not the typical result.

Looking for the lowest of many checks can make performance look optimistic. Validation also helped select our recipe.

Late20 is our average of the last 20 scheduled validation checks, from update 190,500 through 200,000.

It gives a summary of late-training behavior. It is not another selected checkpoint and does not use a new data partition.

**WHY WE NOW SHOW THREE VALUES**

BEST shows the selected checkpoint. LAST shows the endpoint. Late20 shows whether the late trajectory supports the same interpretation.

Fixed training, selection and late-window rules [7]-[8], [12]-[14]. Toy validation values are not measured project results.

## Page 13: The old gate and the revised study

A gate is a rule that decides whether the next study stage can proceed. Our first gate was a small-subset training sanity check.

It asked both energy and force MSE to become less than 10% of their own initial values on fitted examples.

A fitted example is a training record the model used to adjust its parameters. Success here would show learning, not unseen-path accuracy.

Many bounded diagnostics failed this strict joint rule. Continuing to search indefinitely would not answer the proposal's main shape-recovery question.

### We kept the failures and changed the forward decision

The amendment was a disclosed protocol change after the plateau. It did not turn any old failed run into a pass.

We used a limited full-data comparison with two new adequacy rules. Adequacy here means enough useful behavior to support the pilot comparison.

The baseline is our first energy-derived comparison model. A zero-force predictor gives no pushes or pulls; its validation MAE is 62.504 meV/Å.

| Rule | Meaning | Measured outcome |
| --- | --- | --- |
| A1: force-error rule | The selected energy baseline must have validation force MAE at most 50 meV/Å. | Selected energy model passes: 34.587 meV/Å. |
| A2: recovery rule | The baseline must recover at least 1/10 targets and more than the measured no-movement control. | Fails: baseline 0/10; control 0/10. |

The no-movement control measures whether a starting arrangement already matches its target. It uses no predicted forces.

A1 compares prediction errors; A2 compares whole search outcomes. Passing A1 does not imply passing A2.

**SELECTION LIMIT**

We already knew the zero-force validation error when setting A1. Validation also selected the model, so this is not independent test evidence.

The revised protocol let us examine the actual model comparison. The failed recovery gate still limits the conclusions.

Original proposal, preserved diagnostics and adopted forward protocol [1], [3]-[4], [7], [9].

## Page 14: First checks: can the pipeline learn?

A pipeline is the complete route from input data through training to saved predictions. We first asked whether that route worked at all.

A zero-force predictor always outputs zero pushes and pulls. It gives a useful error reference even though it learns nothing.

A randomly initialized network can start much worse than this simple predictor. Large improvement from random output can therefore mislead.

| Check | What happened | Meaning |
| --- | --- | --- |
| Random initial energy output | Final force MSE was only 3.50% below the zero-force error. | Improvement from its own poor start was not enough. |
| Zero initial energy output | Force-MSE fraction remaining: 0.74724 after 2,000 updates. | For this real subset, 74.724% of zero-force squared error remained. |
| Easy synthetic teacher | Force fraction 0.00971; energy fraction 0.08482. | Both fell below 0.1, using their own synthetic starting errors. |
| Real labels, full batch of 16 | Force fraction 0.71455 after 2,000 updates. | Using all 16 records per update still failed the force gate. |

### Why the teacher control mattered

Synthetic means we constructed the answers. The teacher used a deliberately easy target that the student network could represent.

Learning that target showed one functioning path through the training software. It did not show that the model could fit arbitrary DFT forces.

The teacher passed both thresholds after 300 updates. It was an easy constructed target, not a demonstration of reference-DFT accuracy.

The teacher fractions use its own initial errors. Their starting error is not the zero-force error from real DFT labels.

### We also held the internal network fixed

A readout is the final rule that combines internal features into outputs. We solved for readout weights while keeping the internal network unchanged.

Those solutions stayed above the force gate. This limited those fixed representations, not every representation that a trainable network could learn.

Initial learning checks, easy-teacher control and fixed-body solves [3]. These are training-set diagnostics with one seed, not held-out success.

## Page 15: Coverage, physical constraints and loss balance

### Would more varied training examples cure the fit?

Coverage means which parts of the training paths our chosen subset includes. We compared 16 nearby records, 16 varied records and 128 varied records.

| Subset, after 2,000 updates | Force-MSE fraction remaining | Plain reading |
| --- | --- | --- |
| Narrow 16 | 0.71455 | 71.455% of that subset's starting squared error remained. |
| Diverse 16 | 0.89676 | 89.676% remained. |
| Diverse 128 | 0.97844 | 97.844% remained. |

These subsets have different difficulties and starting errors. The 128-record arm also used more record exposures per update.

We did not show that more data is harmful. We showed that coverage alone did not cure fitting under the tested settings.

### Could physical symmetry make the labels impossible?

Net force is the sum of the atomic forces. Torque is their total turning effect.

For an isolated molecule without outside forcing, the totals should be zero. Whole-molecule movement and rotation should not change its energy.

An error floor is a minimum unavoidable error under specified conditions. The net-force/torque floor was far below our failed target.

So those conditions alone could not account for the large fitting gap. This did not show that every other physical constraint was satisfied.

### Would different loss weights help?

Loss weights set the training importance of energy and force errors. Balancing them improved the 128-record force-MSE fraction from 0.97844 to 0.64119.

For force-only diagnosis, we set the energy-loss weight to zero in the energy-derived model. This differs from the later direct model with energy supervision.

Energy fitting became worse. A force-only objective gave force fraction 0.56994, but still failed the force gate.

Without energy supervision, an overall energy shift is unconstrained. A large absolute energy error in that control needs this caveat.

**THE DECISION**

We kept the force-energy trade-off and local parameter-gradient findings. We had not identified one cause or a permanent model limit.

Coverage, symmetry-floor, parameter-gradient and fixed-loss experiments [3]. Each ratio uses its stated immutable initial error.

## Page 16: Longer runs, wider models and error spikes

We next changed training time and internal model size. A wider network carries more internal feature channels at each atom.

A learning rate sets the size of parameter changes. Comparing width and learning rate together cannot isolate either effect.

| Lever | Measured force-MSE fraction | What it supports |
| --- | --- | --- |
| Continue balanced loss to 6,000 updates | 0.45319 | More updates improved the endpoint, but the joint gate still failed. |
| Continue force-only to 6,000 updates | 0.42357 | The force gate still failed. |
| Corrected width 32 / rate 0.001, 6,000 updates | 0.38981 | This completed setting improved force fit but did not pass the joint gate. |
| Width 32 / rate 0.0003 | 0.46571 | Reference for the next width comparison. |
| Width 64 / rate 0.0003 | 0.56395 | More parameters were not uniformly better here. |
| Late continuation to 10,000 updates | Constant rate: 0.32470; late decay: 0.34456 | Reducing the rate late did not beat constant continuation. |

These are training squared-error fractions, not the validation MAE scores used later. Smaller fractions mean less of the starting error remains.

The best historical fraction, 0.32470, came from selected endpoints across many diagnostics. It was not a measured irreducible floor.

### What did the energy spikes mean?

An energy spike is a temporary increase in energy error. Most of two sampled spikes came from an error in the common energy offset.

The common offset changes the overall energy level, not the coordinate slopes by itself. Small mean energy error also does not guarantee correct energy differences.

At both sampled spike peaks, energy-loss and force-loss parameter gradients pointed in similar directions. The snapshots did not identify the spike cause.

Saved Adam-step checks examined local optimizer proposals without adding training updates. They did not show a long-term optimizer cause.

Corrected width/rate evidence, capacity, late-rate and spike diagnostics [3]-[4]. Failed execution and reporting corrections remain recorded.

## Page 17: Offsets, force isolation and an unfinished attempt

### Remove one simple source of energy error

A global energy offset adds the same number to every prediction. It does not change forces because the coordinate slopes stay unchanged.

We reset one training-derived energy bias before training and after each update. A bias is an adjustable constant in the model output.

This offset-calibration trial reduced endpoint force MSE by 3.16% against its historical control. But a separate energy-shape measure worsened by 24.42%.

That measure was the centered-energy variance on the fitted anchor records. It removes the mean offset and examines the remaining energy-error spread.

Anchor records were fixed training examples, not an independent test set. The result was mixed and the joint gate still failed.

### Remove the energy-loss contribution to parameter updates

We kept the same bias calibration but set the energy-loss gradient coefficient to zero. This isolated one contribution to the optimizer update.

We still used energy labels to calibrate the bias. This was not an energy-free training experiment.

At 6,000 updates, force fraction 0.52744 was worse than the calibrated control's 0.45101. Endpoint force MSE worsened by 16.95%.

Many intermediate observations looked better, but the declared endpoint was worse. We did not select a convenient intermediate success afterward.

### The richer angular model did not finish

Model degree is a setting for internal directional patterns. A higher degree can represent richer patterns without guaranteeing better learning.

The degree-two attempt reached 5,765 of 6,000 updates before its fixed time cap. Its latest scheduled save was at 5,700.

**DO NOT CALL THIS A COMPLETED FAILURE**

The attempt has no registered final endpoint and no scientific gate verdict. We closed it without a retry and kept the incomplete evidence.

Offset-calibration, force-isolation and degree-two closeout [4]. All fractions and comparisons keep their original training-only boundaries.

## Page 18: Three different meanings of saturation

The word saturation can hide different claims. We must name the specific observation instead of treating every obstacle as a final limit.

| Observation | What it means | What it does not mean |
| --- | --- | --- |
| Early training plateau | Our small E3x model kept high training error under tested recipes and budgets. | The data or all energy-derived models are impossible to learn. |
| Zero-recovery floor | Every evaluated checkpoint recovered zero of ten original branch targets. | Every method always fails, or one method recovers better than another. |
| Crowded research area | The professor advised that energy-derived force-field research already has much prior work. | Our experiments measured a universal ceiling on energy-derived models. |

### What each result changed

The early plateau led us to change the forward gate and try a reference model used in prior research. We kept every earlier failed result.

MACE fit the same 128 labels. That showed one conservative model could learn them, while leaving the earlier E3x cause unresolved.

Full-data E3x later met our force-error rule. So the earlier small-subset difficulty was not the complete project story.

The 0/10 pilot led to controls at reference shapes and replays of individual movement traces. We examined model errors and the update procedure together.

The professor feedback led to a bounded representation comparison. It did not make the original recovery question disappear.

**A SENTENCE YOU CAN USE**

We changed the next experiment because different results exposed different limitations. We did not show that energy-derived forces had reached a universal limit.

The representation study asks a narrower prediction question. The original recovery question still needs a procedure that can distinguish meaningful outcomes.

Decision history and research-positioning limits [1], [3]-[14]. Prior-work context [15]-[21].

## Page 19: MACE B: fitting the same examples

MACE is a different atomic neural network from prior research. We trained it on the exact 128 examples that challenged the early E3x settings.

Run B is the name of this small-subset diagnostic. Its purpose was fitting, meaning matching examples used during training.

### The result uses a squared-error fraction

Zero-force MSE is the error from predicting zero force everywhere on these 128 examples. We divide MACE's force MSE by that number.

The primary final model gave fraction 0.001514599. Approximately 0.15146% of the zero-force squared error remained, a reduction of 99.8485%.

Its force RMSE was 4.619 meV/Å on the fitted subset. RMSE is the square root of the average squared component error.

The primary export uses EMA, an exponential moving average of trained parameters. This is a smoothed parameter copy maintained by the trainer.

| Run B fact | Meaning |
| --- | --- |
| 128 training records | Only these records changed the model weights. |
| 2,048 epochs; 51,200 optimizer updates | A different training exposure and optimization budget from the early E3x diagnostics. |
| Final epoch 2,047 is primary | Epoch numbering starts at zero. We kept the final model as the declared main diagnostic. |
| Selected epoch 399 is secondary | A separate checkpoint selected using weighted validation loss, not a replacement primary result. |

Primary validation equal-connection force MAE was 37.823 meV/Å. Validation contains other path groups, so this score asks a different question from subset fitting.

Running predictions on all 4,891 training-partition records did not mean the model trained on all of them. It trained on 128.

### What changed after this result?

We could no longer support a blanket claim that a conservative model could not fit these real labels. We proceeded to a full-data MACE reference.

**THE COMPARISON LIMIT**

MACE changed architecture, optimizer, budget and hardware. Its success did not isolate the reason our earlier E3x recipe struggled.

Accepted Run B fitting and reproduction evidence [5]. The synthetic teacher and real-data MACE fractions use different targets and denominators.

## Page 20: MACE Run A: the full-data reference

Run A is our MACE model trained on all 4,891 training records. A reference model here means a comparison model used in prior research, not the source of DFT labels.

It completed 2,048 epochs. We opened the saved models in a separate program on the local computer's general processor, called its CPU. Their validation metrics reproduced the records.

### Read the table before comparing the numbers

Force MAE averages component-error sizes. Equal-connection MAE gives each validation path group equal weight; pooled MAE gives every component equal weight.

Lower is better. All force-error values below use meV/Å, the smaller force unit introduced earlier.

The primary export uses EMA, an exponential moving average of trained parameters. This is a smoothed parameter copy maintained by the trainer.

| Validation measure | Primary final model | Secondary selected model |
| --- | --- | --- |
| Checkpoint epoch | 2,047 | 1,987 |
| Equal-connection force MAE | 5.348957 | 5.332516 |
| Pooled force MAE | 5.259833 | 5.242323 |
| Force RMSE | 26.410390 | 26.071354 |

The selected model minimizes weighted validation loss. The final model remains the declared primary reference; we do not switch because one column looks better.

The primary force MAE on training records was 0.575206 meV/Å. Validation error is larger, so fitting success is not the same as unseen-path accuracy.

Primary validation residual-energy MAE was 7.461842 meV per molecule. Residual energy uses the fixed training-derived offset, not an error per atom.

### How much training did it use?

Run A committed 2,002,944 optimizer updates and 10,014,720 record exposures. These counts exclude diagnostic and lost partial work.

**WHAT IT DOES AND DOES NOT SHOW**

MACE is a working low-error prediction reference on this split. We have not measured its recovery in the shared shape-search procedure.

Its architecture, training exposure and hardware differ from E3x. The score gap is not a controlled measure of architecture efficiency.

Accepted final and selected MACE Run A rescore [6]. Equal-connection, pooled and RMSE columns must stay distinct.

## Page 21: How we obtained the full-data E3x baseline

E3x supplies the geometry-aware network layers used in our small models. A baseline is the initial comparison model, here with energy-derived forces.

First, short 1,000-update energy and direct runs selected update zero as BEST. Their final force errors were worse than zero-force.

Their energy errors improved. So we did not treat energy improvement alone as useful force learning.

### Three recipes, fixed before the full comparison

A recipe specifies model size, optimizer settings and loss weights. All three candidates below completed 200,000 parameter updates.

Width counts internal feature channels. The learning rate controls update size. Balanced weights change energy's and force's contributions to the loss.

| Candidate | Width / rate | Loss weights | BEST force MAE |
| --- | --- | --- | --- |
| A | 32 / 0.001 | Equal normalized weights | 42.108 meV/Å |
| B | 64 / 0.003 | Equal normalized weights | 40.370 meV/Å |
| C | 64 / 0.003 | Balanced weights | 34.587 meV/Å |

These are equal-connection validation MAEs: smaller means more accurate force components. C passed the A1 threshold of 50 meV/Å.

C reduced error by 44.664% against zero-force MAE 62.504 meV/Å. But C's energy MAE was 126.990 meV per molecule, versus B's 28.178.

The force-selected checkpoints therefore show a force-energy trade-off. Comparing A with B also changes width and rate together.

### We repaired a checker without repeating the fits

A checkpoint checker rebuilt named saved values in a different order. It mistook the changed encoding for changed training state.

All 68 saved state entries had the same values, sizes and number types. We corrected the comparison and added rejection controls.

The weights and training results stayed unchanged. We kept the original failed checker outputs and the linked correction.

Short engineering fits and corrected, independently scored A/B/C baseline [7]. Scoring software failure and scientific failure remain separate.

## Page 22: The five model formulations and their errors

After selecting recipe C, we kept its settings for the main small-model comparison. We used two seeds, meaning two initialization and sampling choices.

We reused C seed zero and completed nine more fits. This gives ten final-recipe fits across five formulations.

All models use a graph of neighboring atoms, width 64, two information-exchange rounds and a 5 Å direct-neighbor cutoff.

The direct-neighbor cutoff means pairs farther apart do not exchange information through a direct edge. Connected multistep paths can still transmit information.

Each fit uses float64, Adam rate 0.003, batch four and 200,000 updates. Float64 stores numerical values with high precision.

### What the rows mean

Energy-derived means forces from energy slopes. Direct means forces from a vector output. Consistency adds the disagreement penalty with the listed lambda strength.

Each force column uses equal-connection validation MAE in meV/Å. Energy error uses meV per molecule at the same force-selected checkpoint.

| Formulation | Mean force MAE | Two-seed range | Mean energy MAE |
| --- | --- | --- | --- |
| Energy-derived | 34.893 | 34.587-35.198 | 128.725 |
| Ordinary direct | 38.186 | 37.291-39.082 | 76.684 |
| Consistency 0.1 | 38.024 | 36.773-39.275 | 87.006 |
| Consistency 1 | 34.683 | 33.723-35.643 | 112.341 |
| Consistency 10 | 39.725 | 39.060-40.390 | 161.662 |

The mean averages the two runs. The range only shows their smaller and larger scores; it is not a statistical confidence interval.

The energy model has 43,521 adjustable parameters. The direct and consistency models have 43,649 because they add a force-output head.

**HOW TO INTERPRET THE TABLE**

The lowest listed force mean is not proof of a generally superior method. There are two seeds and only five validation connections.

The proposal planned three seeds. The present evidence uses two, and the recipe was selected on the energy arm rather than tuned separately for each type.

Accepted controlled force/energy table and fixed configuration [8]. Every row uses the saved force-selected BEST model.

## Page 23: Similar force error, stronger agreement and cost

### Why match force error before interpreting recovery?

If one model predicts forces much more accurately, better recovery could result from accuracy rather than the agreement penalty itself.

Our matching rule therefore compares the mean direct error with each consistency mean. The permitted gap is at most 10% of the smaller mean.

The gap must also be no larger than each arm's seed-score span: its larger score minus its smaller score. Only lambda 0.1 passed.

Its mean gap from ordinary direct was 0.162 meV/Å. This rule makes a practical comparison. It does not show that the methods have equal performance in general.

### Why the small gap needs caution

BEST is the lowest of 401 validation checks. Late baseline scores varied by 2.2-3.1 meV/Å standard deviation across correlated checks.

Standard deviation measures their spread. These repeated checks share a model trajectory and data; they are not independent measurements of uncertainty.

The matching gap is much smaller than that training variation. We cannot interpret 0.162 meV/Å as a reliable general advantage.

In the nine subsequent fits, BEST was 0.1-7.4 meV/Å below LAST. The selected minimum is an optimistic score, not the final typical error.

### Agreement improved much more clearly

We measured normalized disagreement MSE between direct forces and the auxiliary energy slopes. Lower means those two predictions agree more closely.

For lambda 0.1, this score was 0.49-0.61 across seeds. Ordinary direct gave 116-379.

This score measures internal agreement, not error against reference forces. Two wrong outputs can still agree.

### The extra training work had a measured cost

Mean elapsed time for fitting and scoring was 1.985 times ordinary direct for lambda 0.1.

This is nearly double for those operations. It excludes earlier diagnostics, failures and unmeasured coordination work; it is not force-inference speed.

**THE SUPPORTED TAKEAWAY**

The penalty reduced disagreement and increased measured fit-plus-score time. It did not produce successful recovery in the original pilot.

Matching rule, trajectory-selection sensitivity, disagreement scores and bounded cost ratio [8]. No global-conservation result follows.

## Page 24: How we tried to find the target shape

Relaxation means repeatedly changing atomic positions using predicted forces until a stopping rule is met. It is a search procedure, not more model training.

Each checkpoint starts from the same ten stored branch arrangements. They come from the five validation connections and target six reference minima.

The reference minima are stored stable shapes. All six target classes appear in training, even though these starting paths were held out.

### Two conditions define success

First, the maximum absolute force component must be at most 0.01 eV/Å, or 10 meV/Å.

Maximum means the largest of all 63 component magnitudes. This asks whether predicted forces are small. Prediction error instead asks whether they agree with the reference forces.

Second, the final shape must be within 0.05 Å RMSD of its own branch reference.

RMSD is the root mean squared distance between corresponding atoms after alignment and permitted atom swaps. It measures the remaining structural difference.

A wrong reference match or a stopped wrong shape is not success. Small predicted force alone can be a false reassurance.

### How did a proposed movement get accepted?

The procedure multiplies predicted forces by a step scale to propose new positions. It can increase or reduce that scale based on force-direction agreement.

The scale converts predicted force into a proposed distance. Its units make force multiplied by scale give angstroms.

The scale starts at 0.2 Å²/eV and can reach 1.0. Each branch has a budget of 1,500 force attempts.

There is no energy-decrease requirement. The scale cap does not cap each atom's displacement when a predicted force is extremely large.

**IMPORTANT PROCEDURE LIMIT**

A large force multiplied by a permitted scale can produce a huge step. A scale limit is not an atomic-movement limit.

The no-relaxation control makes no movements. It asks whether the starting shape already passes the same reference-match condition.

Frozen relaxation rule, success thresholds and common starts [9]. These validation results did not use the main test partition.

## Page 25: What zero of ten actually tells us

Recovery means meeting both the force stopping condition and the correct reference-shape condition. We evaluated five specific saved checkpoints.

The table gives counts for the same ten starting branches per model. Wrong converged means the force rule passed but the structural target did not.

Not converged means the procedure did not reach its stopping condition within its permitted behavior and budget.

| Checkpoint | Correct / 10 | Wrong converged | Not converged |
| --- | --- | --- | --- |
| Energy C, seed 0 | 0 | 2 | 8 |
| Direct, seed 0 | 0 | 4 | 6 |
| Direct, seed 1 | 0 | 4 | 6 |
| Consistency 0.1, seed 0 | 0 | 6 | 4 |
| Consistency 0.1, seed 1 | 0 | 9 | 1 |

The energy C row uses seed zero of the selected energy-derived model. Direct and consistency 0.1 each use both declared seeds.

Every row has zero correct targets. The no-movement control also has zero, so the energy baseline fails the A2 recovery rule.

Across all model/branch outcomes, 25 stopped at wrong geometries and 25 did not converge. These are 50 paired outcomes on five connections.

We do not have 50 independent paths. The models repeatedly use the same ten branches, whose destination minima can also be shared.

### Why this does not settle the hypothesis

If every method scores zero, the outcome cannot show which recovers the correct shape more often. Even the energy-derived reference within E3x scored zero.

The result applies to these checkpoints, starting arrangements and movement procedure. It is not proof that direct forces always fail.

**THE HONEST RESULT**

The agreement penalty did not give recovery here. The broader question remains open because the pilot did not produce useful recovery separation.

We kept these original results. Later diagnostic controls examine causes; they do not replace this failed pilot with a different success rule.

Accepted original 50-row validation recovery study [9]. No MACE recovery or trained main-test result is included.

## Page 26: Do the models recognize stable target shapes?

Post-hoc means we designed these checks after seeing the recovery failure. They help interpretation but are not the original predeclared pilot.

### First, inspect forces without moving the atoms

At a stable reference minimum, DFT forces should be close to zero. A model that predicts a large push there may move away from the target.

For all 11 exact reference minima, every evaluated E3x model exceeded our force stopping threshold. Reference-label maximum forces stayed below 2.745 meV/Å.

A separate 155-evaluation diagnostic reproduced the 50 stored stopping flags and examined these reference and endpoint force values.

This is evidence at the exact stored coordinates. A nearby geometry might still pass both the force and shape conditions.

### Next, begin the movement procedure at the targets

The original ten starts lead to six distinct target minima. We started each model at those six targets and applied the same procedure.

| Model checkpoint | Correct target after reference start |
| --- | --- |
| Energy C, seed 0 | 4/6 |
| Direct, seed 0 | 1/6 |
| Direct, seed 1 | 0/6 |
| Consistency 0.1, seed 0 | 0/6 |
| Consistency 0.1, seed 1 | 0/6 |

This gives 30 reference-start relaxations: five checkpoints times six targets. They are separate from the 155 static force evaluations.

Energy C recovered four of six from these target starts. Each dual-head model recovered at most one.

Dual-head means a network with both an energy output and a direct-force output. The direct and consistency models are the dual-head models here.

**WHAT THIS ADDS**

The original starting paths are not the only concern. The models also behave differently near the reference shapes, without isolating a penalty-specific cause.

Static reference-force diagnosis [10] and separate reference-start relaxations [11]. Nearby valid solutions and causal explanations remain unresolved.

## Page 27: Why a zero force could mean a broken structure

Our graph links atoms only within a 5 Å direct-neighbor cutoff. A connected graph has a chain of links joining every atom.

A disconnected graph contains separate groups with no linking path. Those groups cannot exchange information through this message-passing network.

All 23 force-converged dual-head endpoints had disconnected graphs. They were not intact molecular conformations.

Seven endpoints had every atom isolated. In this model limit, each direct force becomes exactly zero even though the structure is wrong.

This zero comes from the missing neighbor information, not from a correct stable molecule.

The other disconnected endpoints can have internal fragment forces. We must not call all 23 cases automatically zero-force states.

### We replayed four stored movement histories

A trace records the sequence of force queries and atomic steps. A replay repeats it to see whether the saved endpoint is reproduced.

These four selected replays reproduced their endpoints. The jump is the largest accepted movement of one atom in a single step.

| Selected checkpoint type | Largest atom jump | Graph outcome |
| --- | --- | --- |
| Direct | 436.721 Å | Disconnected |
| Energy | 268.047 Å | Disconnected |
| Consistency | 0.086492 Å | Connected |
| Energy | 0.006083 Å | Connected |

A bond is normally only a few angstroms long. Jumps of hundreds of angstroms are enormous on a molecular scale.

Only two of the four traces separated. The other two stayed connected with jumps below 0.09 Å.

**CAUSAL LIMIT**

We found extreme steps and wrong model forces. These observations support a combined diagnosis; they do not show one cause for all failures.

A displacement-capped procedure would limit each atom's movement per step. We must declare and compare that new procedure separately before claiming improvement.

Qualified graph diagnosis [10] and selected exact trace replays [11]. The selected traces do not estimate how often every future path separates.

## Page 28: How the professor feedback changes the plan

The professor suggested graph models, a comparison of rotation behavior, long-range interactions and examination of other available models.

Advice identifies useful questions. It is not an experimental result from our dataset.

| Feedback | Meaning and our response |
| --- | --- |
| Use graph neural networks | We already make atoms graph nodes and exchange information along neighbor links. Our E3x and MACE models are graph networks. |
| Equivariant versus non-equivariant | Equivariant means force arrows rotate correctly when the molecule rotates. We must ask which architecture contrast the professor intends. |
| Long-range effects beyond 7 Å | These are interactions across larger separations. We have not shown that missing long-range physics caused our present failures. |
| Look at Matbench Discovery | This benchmark ranks materials models. It helps identify candidates, but a crystal-model rank does not establish isolated-aspirin performance. |
| Explore ORCHARD | It is a possible computing resource. We do not assume cluster access until the access and usable hardware are confirmed. |

Non-equivariant means the correct rotation rule is not built into the design. Such a model may need rotated examples to learn that behavior.

### Why the current extra study is narrower

A network can store only direction-free numbers internally or also carry directional features. We compared these choices while keeping forces energy-derived.

This does not remove correct rotation behavior from the output. It is not a model with symmetry versus one that learns symmetry from rotated examples.

Symmetry here means a physically appropriate response when we rotate, translate or relabel equivalent atoms. It is not a decorative geometric pattern.

**THE QUESTION TO SEND BACK**

Do you mean scalar versus vector internal features, or a model that must learn rotation behavior from rotated training examples?

Recorded faculty assessment and bounded representation protocol [12]. We have not sent a clarification message or received an answer through this guide.

## Page 29: Inside the network: scalars and vectors

A feature is internal information the network constructs. It is not an additional measured label supplied with the input.

A scalar feature is a single number without a direction, like a distance. A vector feature has directional components, like an arrow.

These are examples of behavior, not claims that every learned feature equals a specific measured distance or force.

### The three settings

E64, SP and S64 are short experiment labels. The table gives their actual meanings; they are not three different datasets.

| Label | Internal features | Width | Parameters |
| --- | --- | --- | --- |
| E64 | Degree 1; scalar and vector features | 64 | 43,521 |
| SP | Degree 0; scalar features | 88 | 43,473 |
| S64 | Degree 0; scalar features | 64 | 25,473 |

Degree zero carries scalar features. Degree one also permits vector features. Width counts internal feature channels at each atom.

Parameters are adjustable model numbers. SP width 88 matches E64's parameter count within 0.1103%; S64 instead keeps the same width.

Matching width and matching parameter count are different controls. Changing internal features also changes the computation, not just a single abstract property.

### Why every output still obeys rotation behavior

All three predict a scalar energy that stays unchanged when we rotate the molecule. This energy is called invariant.

We calculate forces from the negative coordinate gradient. Those forces rotate with the molecule, so they are equivariant.

Even the scalar-only model therefore has equivariant forces. We did not compare correct physical symmetry with no built-in symmetry.

Paired sampling means the same training-record sequence for corresponding seeds. We keep it the same even when the models have different feature shapes.

**FAIRNESS LIMIT**

We kept the data, seed-paired sampling, loss and update budget fixed. We selected the recipe on E64 and did not tune scalar recipes separately.

Verified parameter counts and fixed representation configuration [12]-[14]. Parameter count is capacity evidence, not predictive quality.

## Page 30: The completed representation results

All four new scalar fits completed 200,000 updates. We reused two completed E64 fits after identity and short reproduction checks.

A fit is one trained model. The suffix zero or one identifies its seed, the random initialization and sampling choice.

The table uses equal-connection validation force MAE in meV/Å. It averages force-component errors with each of five path groups weighted equally.

BEST is the selected minimum score, LAST is the endpoint, and Late20 averages the final 20 scheduled checks. Lower values are better.

| Arm / seed | BEST | LAST | Late20 mean |
| --- | --- | --- | --- |
| E64-0 | 34.587 | 37.791 | 39.008 |
| E64-1 | 35.198 | 39.340 | 39.029 |
| SP-0 | 41.046 | 52.522 | 46.276 |
| SP-1 | 41.669 | 43.647 | 45.743 |
| S64-0 | 40.264 | 49.632 | 44.062 |
| S64-1 | 38.820 | 42.195 | 42.471 |

Chart: BEST and Late20 means, with the two-seed ranges. Numeric values are in the table above.

Both scalar settings have higher error than E64 in each seed at BEST and Late20. This is the same direction in all four paired comparisons.

Completed six-row results and score identities [13]. The next page gives the paired arithmetic and limits.

## Page 31: What the representation gaps mean

E64 is the scalar-and-vector setting. SP is the scalar setting with a similar parameter count; S64 is the scalar setting with the same width.

The gap means scalar force error minus E64 force error. A positive gap means the scalar model made larger average component mistakes.

We first compare corresponding seeds, then average the two gaps. This is a descriptive pairing, not new independent path data.

| Scalar setting | Mean BEST gap | Mean Late20 gap |
| --- | --- | --- |
| SP | +6.465 meV/Å | +6.991 meV/Å |
| S64 | +4.649 meV/Å | +4.248 meV/Å |

For example, SP mean BEST is approximately 41.358 meV/Å and E64 mean BEST is 34.893. Their difference is approximately +6.465.

Late20 averages the final 20 validation checks, rather than selecting the single smallest error. It gives the same direction as BEST in both seeds.

Positive means the scalar error is larger. The last-20-check average supports the same direction as the selected best score.

**THE LIMITED CONCLUSION**

E64, with scalar and vector features, had lower error under this recipe. We cannot isolate those features alone as the cause.

Separate checks reproduced the saved summary arithmetic and the published file identities. The broader scientific interpretation still needs independent result review.

The four new scalar fits took 2.110 hours of summed training-process elapsed time. Scoring, preliminary checks and the publication interruption are additional.

Completed six-row saved results, endpoint scoring and paired summaries [13]. No new recovery or trained test result follows.

## Page 32: What earlier research already knew

A literature review examines earlier studies so we do not claim to invent an existing idea. We kept twelve targeted reading cards plus the MACE reference.

A reading card records the paper version, relevant result and its limits. This is a focused reading map, not a complete review of every paper.

| Earlier work | Plain meaning for this project |
| --- | --- |
| Landscape17 | A dataset and study showing why accurate individual predictions can still miss the correct molecular landscape or paths. |
| GDML, PaiNN and E3x | Earlier tools for force learning and geometric features. GDML builds conservative forces; PaiNN uses scalar/vector features; E3x supplies equivariant layers. |
| MACE | An established atomic graph architecture. We trained it on our split as an external predictive reference. |
| Forces are not Enough; smooth potentials | Evidence that a low stored-example error alone does not guarantee useful simulated behavior. |
| Bigi and Zhao studies | Earlier work on direct-force limitations and task-dependent optimization or transition-state searches. |
| Takamoto; When Forces Disagree | Earlier consistency-assisted training and comparisons between direct forces and energy slopes. |

### What is new in our work?

We do not claim new graph networks or a new general consistency loss. We examine force error, shape recovery and computing cost together.

Our setting holds out whole conformational connections and keeps paired starts. Our negative recovery result and its diagnosis are part of that evidence.

The scalar/vector study adds a bounded representation comparison. It does not answer the original recovery question on its own.

### Why not replace everything with a leaderboard model?

Matbench Discovery mainly evaluates crystals and materials. A high rank there does not show accuracy for one isolated aspirin molecule.

A foundation model is already trained on many external examples. We have not loaded one into this comparison or done compatibility checks.

Primary literature and exact versions [15]-[21]. The anonymous When Forces Disagree manuscript has no verified current acceptance status in our record.

## Page 33: What long-range physics would mean

Our graph uses direct edges only inside 5 Å. This is a local-neighbor design choice, not a claim that real forces stop at 5 Å.

Real electrostatic interactions are attractions and repulsions associated with electrical charge. Some effects can extend beyond a local neighbor distance.

Long-range modeling means adding a way to represent these more distant effects. Raising a local cutoff is not the same as modeling explicit electrostatics.

### What message passing can and cannot do

One message-passing round transfers information across a direct link. More rounds can transmit it along several connected links.

No number of such rounds creates a link between fully disconnected components. That is why a separated endpoint can be outside useful molecular behavior.

### What we have actually done

We screened material-model documentation and long-range paper abstracts. Latent Ewald Summation is one possible reading direction for nonlocal interactions.

We did not implement it, load external foundation-model weights or measure its effect on aspirin.

The professor's suggestion to examine effects beyond 7 Å is research advice. It is not a universal physical threshold or a diagnosed failure mechanism.

### Why defer it now?

Our current evidence still needs a MACE recovery control and a comparison of movement procedures. Adding a new physics model would change several factors again.

We should first know what the existing models and procedure can do. A later long-range experiment needs a separate question, control and budget.

**A SAFE MEETING ANSWER**

Long-range physics is an important possible extension. We have not shown that it caused our present force or recovery failures.

Faculty scope and screened primary sources [12], [21]. No current leaderboard order or universal best model is claimed.

## Page 34: What the computing and restart work accomplished

A saved checkpoint should let training continue after an interruption. To reproduce the same path, we also save optimizer and random-generator bookkeeping.

A checksum is a digital fingerprint calculated from file bytes. Matching fingerprints help show that the saved model and data have not changed.

This shows file identity, not scientific correctness. A perfectly saved model can still predict wrong forces.

### What went wrong during MACE operations

Cloud setup, package installation, formatting and storage-mount failures delayed training. A storage mount is the connection that makes remote Drive files accessible.

The original owner and device checks also made restarts difficult. We removed fixed pauses and added continuation paths with state checks while keeping evidence.

A GPU is a graphics processing unit used to accelerate numerical work. GB means gigabytes, a unit of memory capacity.

A copied-checkpoint diagnostic passed continuation checks on an A100 40 GB graphics processor. It has less memory than the initial 80 GB device.

A memory fit does not show the cheapest GPU. Graphics-processor cost depends on actual runtime and billing as well as available memory.

### Why the scalar study stayed on the local CPU

The CPU is this computer's general processor. The small-model workload already fit an existing CPU training path without cloud allocation.

A short E64 reproduction matched its saved validation history. A deliberately interrupted scalar control resumed to the same 2,000-update state within tolerance.

After SP-0 finished, a GitHub publication check stopped the queue even though the result had pushed successfully.

GitHub stores the team's code and result history. The original remote file bytes passed identity checks. We fixed publication and skipped completed SP-0 on restart.

The other three fits then finished. The four scalar training durations sum to 2.110 elapsed CPU hours, excluding preflight, scoring and the interruption.

**CURRENT COMPUTE ASSUMPTION**

Use local CPU or Colab until cluster access is usable. Colab compute units are billing credits, not a scientific measure of model quality.

Reproduction, full-state interruption and publication recovery [13]-[14]; MACE runtime history and cost limits [5]-[6].

## Page 35: What each teammate can do next

The proposal roles changed as the work developed. A task assignment is a responsibility, not proof that its owner already completed it.

| Owner | Current contribution or responsibility | Next concrete work |
| --- | --- | --- |
| Shedrack Eze | Prepared data, integrated training/models and completed the first MACE and current experiments. | Integrate the comparisons, keep the evidence and coordinate study and report decisions. |
| Olufunke Abodunrin | Authored the shared recovery evaluator, the program that moves atoms and classifies endpoints. | Connect real MACE force predictions to the runner/reporter and examine the search-method comparison. |
| Muhammed Idris | Owns MACE loading and independent reproduction. | Open the saved model, verify its identity/environment and reproduce retained reference predictions. |
| Emmanuel Aaron | Owns physical-consistency and cost analysis. | Build known-answer example fields, measure loop work and prediction time, and compare results by whole connection. |

### Idris: loading and parity, not repeated training

A loader opens the saved network and makes predictions. Admission checks that the model, units, atom order and software match the agreed identity.

Parity means the loader reproduces the stored predictions within the declared tolerance. A tolerance is the permitted small numerical difference.

Shedrack prepared a connector tested with synthetic examples. Idris's real checkpoint loading, admission, parity and packet access still need completion evidence.

### Aaron: physical and timing controls

A known-answer control uses a force rule we construct ourselves. We know its correct behavior before applying the measurement program.

Synchronized timing waits for the device to finish the requested calculations before stopping the clock. Otherwise, the measured time can be too small.

Closed-loop work asks how forces accumulate energy along a path that returns to its starting arrangement. Finite numerical checks do not show global conservation.

Inference timing measures prediction time. Compilation converts code for execution; warmup covers initial calls. These must be separate from steady prediction timing.

Each member needs meaningful code changes, recorded checks, analysis and a report section. A commit records an authored change in the shared history. Each member must approve the record of their own contribution before submission.

Current task lists, interfaces and contribution boundaries [22]. We do not credit planned tasks as completed work.

## Page 36: The next work and the report deadline

### First, use the strongest existing model as a control

Finish Idris's loader and prediction-parity checks. Then connect MACE to Olufunke's evaluator without retraining MACE.

Run the unchanged ten-start and six-reference procedure control. This asks whether the existing low-error reference behaves better under the original search method.

### Then compare a safer movement rule

Declare a procedure that limits each atom's displacement per step. Use the same saved models and starting arrangements for the comparison.

Label reused-validation diagnosis as post-hoc. We must not silently replace the original 0/10 results or tune on the main test.

### Complete the other comparison measures

Examine the bounded interpretation of the scalar/vector results. Finish closed-loop controls and prediction-time measurements on the same actual hardware.

Freeze model, search and analysis choices before any one-time trained main-test evaluation. Freeze means stop changing those choices for that final evaluation.

### Keep the course document separate from this guide

The midsemester draft is five pages in the course format. This longer guide teaches the project; it is not the submitted course report.

The saved rubric audit found the requested scientific content. We still need live Canvas instructions, team approval and grader access to the private repository.

The saved schedule lists 23 October for the midterm, 4 December for video and 9 December for the final report. Compare these dates with the live course instructions.

**DECISIONS FOR TODAY**

Agree on owners and target dates. Clarify the professor's intended symmetry comparison. Make sure members can open the saved-model packet and graders can open the repository. Agree on report contributions.

A proposed next checkpoint is loader/parity status and contribution approvals within 24 hours. The team must agree; this is not an existing promise.

Current next actions and course-release limits [12], [22]-[23]. Paracetamol, extra MACE seeds and long-range extensions remain deferred.

## Page 37: Questions a novice may ask

### What exactly does the model predict?

For one arrangement of atoms, it predicts an energy number and forces, the pushes and pulls on each atom. We compare these with reference-calculation answers.

### What does molecular shape mean?

It means the relative three-dimensional positions of the atoms. Rotating the whole molecule is not a new internal shape; moving one part can be.

### What are direct forces and learned energy?

Direct forces are arrows produced by the network's force output. Learned energy is the energy number the network predicts from examples.

Energy-derived forces instead come from how that predicted energy changes when we move each coordinate slightly. This calculation gives the energy slope.

### What does consistency training add?

It adds an error penalty when direct force arrows disagree with the predicted energy slopes. It encourages agreement, not guaranteed reference accuracy.

### What is a connection group?

One transition-state pass has two descent branches toward stable shapes. We keep all records from both branches together as one data group.

This limits leakage, where nearly identical neighboring records could otherwise appear on both sides of a training/evaluation split.

### Why not just give the model more examples?

More examples can help, but they do not correct every training recipe. Our early coverage trial did not solve fitting under its particular settings.

### What does zero of ten mean?

Starting from ten agreed shapes, no search met both conditions: small predicted force and a match to its correct reference shape.

Because every evaluated model scored zero, the pilot cannot show which recovers shapes better.

These answers repeat definitions deliberately so they work aloud without the earlier pages [1]-[14].

## Page 38: Questions about the findings and decisions

### Did you find that the data were saturated?

No. Our first small E3x network reached a training plateau under tested settings. A different model, MACE, later fit the same 128 labels.

### Why change the study if the original question matters?

The first search gave zero successes for every model. We must diagnose that shared failure before claiming one agreement rule improves recovery.

The new scalar/vector study asks whether internal directional features help force prediction. It adds evidence without answering the original recovery question.

### Did the agreement penalty work?

It greatly reduced disagreement between direct-force arrows and learned-energy slopes. It did not produce correct-shape recovery with the original procedure.

### Did the movement procedure cause everything?

We saw enormous accepted steps in selected traces. We also saw wrong force predictions at stable reference shapes.

Both model and procedure limitations matter. These checks do not isolate one cause for every failure.

### Are scalar-only models non-equivariant?

Equivariant means force arrows rotate when the molecule rotates. Our scalar-only networks predict invariant energy, then calculate forces from its slopes.

Those forces still rotate correctly. We changed internal features, not the built-in output rotation rule.

### Does E64 always beat scalar models?

E64 had lower force error in both seeds under our fixed recipe. We chose that recipe on E64 and did not tune scalar recipes separately.

The result does not show a universal ranking. A different recipe, dataset or budget could change the comparison.

### Why not use a top materials leaderboard model?

A leaderboard compares models on a particular task. Crystal performance does not automatically transfer to one aspirin molecule or our reference labels.

Descriptive results and causal/transfer limits [5]-[21]. No main-test or new recovery success is claimed.

## Page 39: A reminder of terms used in the text

The main text supplies these meanings where each concept is used. This page is a quick reminder during the meeting.

| Term | Meaning |
| --- | --- |
| Atom / molecule | Small unit of matter / a connected collection of atoms. |
| Geometry / conformation | The three-dimensional arrangement of the atoms; molecular shape. |
| Force / component | A push or pull / its amount along one coordinate direction. |
| DFT reference | Density-functional-theory calculation used to supply reference energies and forces. |
| Minimum / transition state | Locally stable shape / an unstable pass with descent paths toward minima. |
| Connection / branch | A whole transition-state group / one of its two descent directions. |
| Parameter / feature | Adjustable model number / internal numerical information about the surroundings. |
| Training / prediction | Changing parameters using examples / producing outputs with the current parameters. |
| Coordinate / parameter gradient | Energy slope versus atomic position / loss slope versus model parameters. |
| Direct / energy-derived force | Force-output arrow / force calculated from the negative energy slope. |
| Consistency penalty | Added training error when direct forces disagree with auxiliary energy slopes. |
| MAE / MSE / RMSE | Average absolute error / average squared error / square root of MSE. |
| RMSD | A structural-distance score after the declared alignment and permitted atom matching. |
| BEST / LAST / Late20 | Best selected validation checkpoint / final checkpoint / mean of the final 20 scheduled checks. |
| Seed / checkpoint | Random initialization and sampling choice / saved model and continuation state. |
| Invariant / equivariant | Energy stays unchanged on rotation / force arrows rotate with the molecule. |
| Admission / parity | Identity/environment checks / reproduction of retained predictions within a tolerance. |

This guide uses short sentences and technical terms under ASD-STE100 Issue 9 writing rules. It does not claim independent language-conformance certification.

Technical meanings are tied to this study. Illustrative examples and diagrams are labeled; they are not numerical evidence [24].

## Page 40: Where the project facts come from

A source of record is the stored document or result that supports a statement. The source package lists exact file paths and digital fingerprints.

The entries below refer to source groups. Old reports give the status at their original dates; the completed saved queue supplies the current fit status.

| Source | What it supports |
| --- | --- |
| [1] Proposal and claim audit | Original question, planned comparisons and scientific exclusions. |
| [2] Data and branch-reference manifests | Atom counts, units, connection split, matching correction and paracetamol block. |
| [3] Early learning and gradient reports | Initial controls, coverage, fixed losses, local gradients and budget checks. |
| [4] Corrected diagnostic and incomplete-run reports | Width/rate, capacity, offsets, force isolation and degree-two invalidity. |
| [5]-[6] Accepted MACE B and A reports | Small-subset fitting and full-data predictive reference. |
| [7] Accepted E3x baseline reports | Short negative runs, A/B/C results and scorer correction. |
| [8] Controlled-result acceptance and score records | Two-seed force/energy table, matching, disagreement and bounded cost. |
| [9] Original recovery report | Common starts, success rule and all 50 model/branch outcomes. |
| [10]-[11] Diagnosis and trace reports | 155 static evaluations, separate 30 reference-start relaxations, graph qualifiers and four replays. |
| [12]-[14] Faculty/protocol and representation records | Current scalar/vector design, six scored rows, reproduction and publication recovery. |
| [22] Team tasks and contribution ledger | Recorded work, ownership, interfaces and remaining checks. |
| [23] Five-page draft and saved-rubric audit | Course-report content and open release requirements. |

The repository is the shared code and evidence history. It is private, so teammates and graders need access.

This guide includes no model weights or numerical test arrays. It does not perform new fitting, inference or a new main-test evaluation.

Exact source paths, original/shared distinctions and SHA-256 fingerprints appear in SOURCES.md and sources.json.

## Page 41: The earlier papers and what to read next

A DOI is a permanent paper identifier. An arXiv or OpenReview identifier locates the particular manuscript; publication status can differ.

| Source | Paper group |
| --- | --- |
| [15] | Cărare et al. Global properties of the energy landscape. npj Computational Materials 12, 9 (2026). DOI: 10.1038/s41524-025-01878-x. |
| [16] | Chmiela et al., Science Advances (2017), DOI: 10.1126/sciadv.1603015. PaiNN: PMLR 139, schutt21a (2021). E3x: arXiv:2401.07595. MACE: arXiv:2206.07697. |
| [17] | Fu et al. Forces are not Enough, TMLR (2023), OpenReview A8pqQipwkt. Smooth and Expressive Interatomic Potentials, PMLR 267, fu25h (2025). |
| [18] | Bigi et al. The dark side of the forces, PMLR 267, bigi25a (2025). Zhao et al., DOI: 10.1002/advs.202506240. Unconstrained potentials: arXiv:2601.16195v3. |
| [19] | Takamoto et al. Physics-Informed Weakly Supervised Learning for Interatomic Potentials, PMLR 267 (2025). When Forces Disagree: OpenReview cQvBP4TZHe, saved manuscript. |
| [20] | Christensen and von Lilienfeld. On the role of gradients, arXiv:2007.09593; journal DOI: 10.1088/2632-2153/abba6f. |
| [21] | Matbench Discovery: DOI: 10.1038/s42256-025-01055-1 and official repository. Latent Ewald Summation: arXiv:2408.15165. Long-range electrostatics perspective: arXiv:2512.18029. |
| [24] | ASD Simplified Technical English, ASD-STE100 Issue 9 (January 2025). Official writing rules and dictionary. |

### Choose the reading for the question

For why shapes and paths matter, start with Landscape17. For direct-force limits in physical tasks, read The dark side of the forces.

For directional features and model construction, read PaiNN, E3x and MACE. For earlier consistency methods, read Takamoto and the saved When Forces Disagree manuscript.

For later long-range work, start with Latent Ewald Summation. Our abstract screening is not an implemented result.

The saved When Forces Disagree PDF is anonymous. Its current acceptance status remains unknown in our record.

**AFTER THE MEETING**

Write down agreed owners, dates and scope changes. Keep measured results, possible explanations and proposed experiments as separate statements.

Exact primary URLs are in SOURCES.md. This package does not include the official STE standard or its dictionary.
