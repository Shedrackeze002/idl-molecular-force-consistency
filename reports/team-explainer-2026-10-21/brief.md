# Do Consistent Learned Forces Find the Right Molecular Shape?

Team and Professor Meeting Explainer

Snapshot: 2026-10-21T10:35:11+02:00

Language: ASD-STE100 Issue 9 writing rules; technical terms in the glossary; no independent language-conformance certification

## Page 1: PROJECT EXPLAINER

Do Consistent Learned Forces Find the Right Molecular Shape?

Landscape17 | CMU Introduction to Deep Learning | Group 9 | 21 October 2026

**THE MESSAGE TO REMEMBER**

We built and compared learned molecular force models. Better force agreement did not produce correct-shape recovery with our first procedure.

We now have a MACE reference with low validation error, a controlled E3x comparison and a completed scalar-versus-vector representation study.

E64 has lower error under our fixed training recipe. These results do not answer the original recovery question.

### Current position

| Workstream | Position at this document snapshot |
| --- | --- |
| Data and targets | Audited aspirin split and branch references are available. |
| MACE | Run B and full-data Run A are complete and independently scored. |
| Controlled E3x | Ten final-recipe fits cover five formulations and two seeds. |
| Shape recovery | Every evaluated checkpoint recovered 0/10 original branch targets. |
| Representation study | All four new scalar fits are complete. Saved results are independently scored. |
| Main test | We have not scored a trained model on the main test partition. |

### How to use this guide

Read pages 2, 9, 12, 16, 19 and 20 for the meeting overview. Read the other pages for technical questions.

Pages 21-22 contain questions and answers. Page 23 gives the meanings of technical terms. Pages 24-25 identify the evidence and literature.

Status source: live saved queue and publication receipts; accepted reports [1]-[14]. Scientific limits apply throughout.

## Page 2: The project in two minutes

### Use this opening

We study whether accurate molecular force predictions also lead atoms to the correct molecular shape.

Our original question compares ordinary direct forces with direct forces trained to agree with a learned energy.

We use the same molecular data, fixed connection groups, common starting geometries and explicit computing budgets.

The first small E3x setup did not fit the real labels well enough. We examined data, gradients, losses and model settings.

MACE later fit the same small subset. This result prevented us from calling the earlier plateau a fundamental data limit.

We then trained larger full-data E3x models. Consistency training reduced direct-force disagreement, but it did not improve measured recovery.

Every evaluated checkpoint recovered zero of ten targets. This includes the energy baseline, so the pilot cannot rank recovery quality.

Follow-up controls found large forces at reference minima, disconnected endpoints and extremely large accepted coordinate steps.

We now separate model limitations from procedure limitations. We also completed a scalar-versus-vector graph representation comparison.

E64 has lower validation force error for each seed at BEST and Late20. We tuned the recipe on E64 first.

**THE OPEN QUESTION**

We have not shown that consistency training improves correct-shape recovery. We have identified why force error alone cannot answer that question.

### The research sequence

Audit data → Fit models → Score forces → Follow forces → Match shapes

The first three stages now have measured results. The recovery procedure and the final test study still need work.

Evidence: original scope [1]; MACE [5]-[6]; controlled results [8]; recovery [9]-[11]; representation [13].

## Page 3: The physical problem

### A force model replaces repeated expensive calculations

A molecular geometry contains the positions of all atoms. A reference calculation gives an energy and one force vector to each atom.

Energy gives the predicted landscape. Force gives a local push or pull on each atom. Our relaxation procedure uses predicted forces to change atomic positions.

A minimum is a stable reference shape on that landscape. A transition state connects descent branches toward minima.

A small average force error can still give a wrong endpoint after many coordinate steps.

### Our models are already graph neural networks

Each atom is a graph node. Neighboring atoms exchange information through graph edges inside the 5 Å cutoff.

Message passing updates the atomic features. The model combines these features to predict energy, forces or both.

| Term | Meaning in this project |
| --- | --- |
| Invariant energy | The energy stays the same when we rotate or translate the molecule. |
| Equivariant force | The force vectors rotate with the molecule. Translation does not change their values. |
| Conservative force | One scalar energy generates the forces through its negative coordinate gradient. |
| Direct force | A vector head predicts forces without the condition that they equal an energy gradient. |
| Consistency training | A penalty encourages the direct forces to agree with the auxiliary energy gradient. |

### Keep these implications separate

A conservative model can still predict the wrong energy landscape. A direct-force fixed point need not be a stable energy minimum.

Small disagreement between two predictions does not show that either prediction is correct.

**PLAIN EXPLANATION**

The energy model predicts a landscape. The coordinate-update procedure uses its slope. A direct model predicts force arrows. We examine whether agreement helps.

Definitions and motivation: [1], [15]-[19]. Technical terms continue on page 23.

## Page 4: The data and correct targets

Landscape17 contains descent pathways and stationary structures from a common hybrid-DFT reference method.

We use aspirin as the main molecule. Each aspirin structure contains 21 atoms.

| Aspirin partition | Connections | Eligible records | Use |
| --- | --- | --- | --- |
| Training | 26 | 4,891 | Fit weights and normalization |
| Validation | 5 | 974 | Select settings and examine the pilot |
| Main test | 6 | 1,127 | Final trained evaluation remains unperformed |

A whole connection includes both descent branches. We keep both branches in the same partition.

The archive has 37 aspirin connections, 74 branches and 11 reference minima. The eligible-record table uses the frozen preparation policy.

The training partition contains all 11 reference minima. An unseen connection is therefore not automatically an unseen minimum.

Earlier model-free software fixtures parsed test coordinates and labels. They did not score trained models or select models or solver settings.

### We corrected the branch-to-reference interface

Filename endpoint pairs did not reliably identify the destination of each branch. Empirical endpoint classifications determine the correct target.

The positional interpretation gave the wrong target to 26 of 74 branches. We corrected the shared interface and kept the audit.

For example, connection 66 names the pair 6/6, but its two classified destinations are minima 6 and 9.

### Geometry and units need explicit rules

We use Å for positions, eV for energy and eV/Å for force. We do not mix rMD17 labels with these labels.

The audit traced source unit conventions. We did not independently recompute the DFT labels.

Structural matching includes alignment and permitted atom permutations. Fixed atom indices alone can give a wrong mismatch.

**WHY PARACETAMOL IS DEFERRED**

One branch has nearest-reference RMSD 0.06431 Å, above the frozen 0.05 Å rule. We did not relax the rule to admit it.

The unmatched branch does not show corrupt labels or a new minimum. The separate replication remains deferred.

Data audit, units, split and empirical branch references [2]. Paracetamol branch: 01_2/plus.

## Page 5: The three force formulations

| Formulation | Prediction used as force | Energy supervision |
| --- | --- | --- |
| Energy-derived E3x | Negative gradient of predicted energy | Yes |
| Ordinary direct E3x | Direct vector-head output | Yes, through an auxiliary energy head |
| Consistency-trained E3x | Direct vector-head output | Yes, plus the consistency penalty |
| MACE reference | Negative gradient of predicted energy | Yes; separate architecture and environment |

Ordinary direct prediction is not a force-only training experiment here. It also fits an auxiliary energy output.

### The objective has three possible terms

**OBJECTIVE**

Loss = energy weight × normalized energy MSE + force weight × normalized force MSE + lambda × normalized disagreement MSE.

Disagreement compares direct force with the negative gradient of the auxiliary energy. Ordinary direct prediction uses lambda = 0.

The consistency sweep uses lambda = 0.1, 1 and 10. The energy arm does not need a direct-force agreement penalty.

Training-derived scales normalize the errors. Physical energy and force outputs keep their original units and derivative relationship.

### The final common E3x recipe

| Setting | Value |
| --- | --- |
| Backbone | Width 64; degree 1; two message-passing interactions |
| Geometry | Eight radial functions; 5 Å cutoff |
| Numerics | Float64; plain Adam; learning rate 0.003; batch size 4 |
| Budget | 200,000 updates; 800,000 sampled frame exposures |
| Supervised weights | Energy 0.0145844; force 1.9854156 |
| Selection schedule | Validation at update 0 and every 500 updates |

The energy model has 43,521 parameters. Each dual-head model has 43,649 parameters.

We selected this recipe using the energy arm. We transferred it to the dual-head arms without separate tuning.

Shared-trainer definitions and full-budget results [7]-[8]. Exact coefficients remain in the saved configurations.

## Page 6: Scores and decision gates

### Prediction and shape recovery answer different questions

| Measure | Definition or use |
| --- | --- |
| Force MAE | Mean absolute Cartesian force-component error. The primary score gives each validation connection equal weight. |
| Pooled MAE | Mean over all force components. Connections with more records receive more weight. |
| Force RMSE | Square root of mean squared component error. Large errors receive more weight. |
| Energy MAE | Error in molecular residual energy, with the fixed training-derived offset. |
| Recovery | Force convergence and a symmetry-aware match to the branch reference. |
| Cost | Measured elapsed training, scoring or inference cost, with its boundary stated. |

One eV/Å equals 1,000 meV/Å. An MSE ratio is not an RMSE ratio.

### We changed the decision gate, not the old results

The original small-subset rule set a strict limit. Both energy and force MSE had to be below 10% of their initial values.

Repeated diagnostics did not meet that rule. The rule had become a barrier to the actual held-out-connection study.

The amendment kept those failures in the record. We then used a bounded full-data comparison with two separate adequacy checks.

| Gate | Rule | Observed outcome |
| --- | --- | --- |
| A1: force-error rule | Energy-baseline validation force MAE at most 50 meV/Å | C passes at 34.587 meV/Å |
| A2: recovery rule | At least 1/10 correct targets and more than the no-relaxation control | Fails: baseline 0/10; control 0/10 |

The zero-force validation MAE was already known when A1 was set. Validation also selected the recipe and BEST.

**INTERPRETATION LIMIT**

The amendment followed the observed plateau and preceded the controlled comparison. It did not retrospectively pass the failed learning checks.

Gate history [3]-[4]; used adequacy and corrected baseline [7]; recovery [9].

## Page 7: The early diagnostic trail

These experiments used training subsets and one seed. Their force fractions use MSE, not the later validation MAE metric.

| Experiment | Measured result | What we learned |
| --- | --- | --- |
| Initial random energy head | Only 3.50% lower force MSE than zero-force | A large improvement from a poor random start can mislead. |
| Zero energy readout | Force fraction 0.74724 after 2,000 updates | A cleaner starting point helped, but the old gate still failed. |
| Easy synthetic teacher | Force fraction 0.00971; energy fraction 0.08482 | The pipeline learned this deliberately easy, representable target. |
| Real labels, full batch 16 | Force fraction 0.71455 | Changing batch sampling alone did not pass the fitting gate. |
| More training coverage | Narrow 16: 0.71455; diverse 16: 0.89676; diverse 128: 0.97844 | More coverage did not solve fitting under this recipe and budget. |
| Frozen-body linear readout | Force-only and joint solves stayed above the gate | The readout alone could not close the gap for these fixed bodies. |

### We examined possible physical and optimization limits

The necessary net-force and torque constraints gave an error floor far below the failed gate.

Those constraints alone could not account for the gap. This did not show full model expressivity or label integrability.

At selected saved states, energy-loss and force-loss parameter gradients opposed each other. Initial scaled energy-loss gradients also dominated force-loss gradients.

These are local measurements. They do not reconstruct the historical Adam trajectory or show the cause of the plateau.

**THE LESSON**

A passing software control does not show molecular accuracy. A failed training setting does not show an impossible scientific task.

Initial controls, coverage, fixed-head solves and gradient diagnosis [3]. All earlier failures remain in the evidence log.

## Page 8: The early diagnostic trail

| Lever | Measured endpoint | Decision and limit |
| --- | --- | --- |
| Balanced supervised loss | 128-frame force fraction 0.64119 versus 0.97844 | Force fit improved; energy fit worsened; the joint gate failed. |
| Force-only objective | Force fraction 0.56994 at 2,000 updates | The force gate failed; the energy gauge was unconstrained. |
| Longer training | At 6,000 updates: balanced 0.45319; force-only 0.42357 | More updates helped, but did not show convergence. |
| Width and learning rate | Corrected width-32/rate-0.001 result: 0.38981 | The corrected matrix still failed the joint gates. |
| Larger width | At rate 0.0003: width 32 gave 0.46571; width 64 gave 0.56395 | More parameters were not uniformly better under these settings. |
| Late rate decay | Constant: 0.32470; late decay: 0.34456 at 10,000 updates | The best historical fraction was a selected endpoint, not an irreducible floor. |
| Energy-offset calibration | Force MSE 3.16% lower; anchor centered-energy variance 24.42% higher | Mixed result; small energy mean error did not show accurate energy differences. |
| Remove energy-loss gradient | Force fraction 0.52744 versus calibrated control 0.45101 | Registered endpoint force MSE worsened by 16.95%. |
| Degree-two attempt | Stopped at 5,765/6,000 updates; latest scheduled save at 5,700 | Incomplete run; no registered endpoint or scientific gate verdict. |

The local update calculations passed the saved Adam-step checks. These checks added no training updates and did not show a long-term optimization cause.

Common energy offsets accounted for most of two sampled energy spikes. At both saved peaks, the parameter gradients pointed in similar directions.

We kept the failed width/rate execution and reporting corrections. We did not replace them with an unrecorded success.

Detailed diagnostic sources [3]-[4]. We must not call the incomplete degree-two attempt a completed gate failure.

## Page 9: Saturation and changes in direction

| Phrase | Evidence we have | Permitted explanation |
| --- | --- | --- |
| Historical training plateau | The early small E3x recipe had high training force error across bounded diagnostics. | This recipe did not fit enough under the tested settings. |
| A fundamental data limit | MACE B fit the same 128 labels to a much smaller error. | We did not show a data-wide or conservative-model limit. |
| Zero-recovery floor | All five evaluated checkpoints recovered 0/10 original branch references. | This procedure cannot separate these models by successful recovery. |
| A saturated research area | The professor advised that energy-derived force-field research is crowded. Previous studies already used consistency training. | This is research-positioning advice, not an experimentally measured ceiling. |

### Each observation changed a different decision

The early plateau led to a bounded gate amendment and a MACE comparison with measured results.

MACE success justified continuing the molecular study. It did not identify the cause of the earlier E3x failure.

The full-data E3x result met the declared force-error rule. It still needed a separate recovery assessment.

The 0/10 pilot led to reference-start controls, graph diagnostics and exact trace replays.

The professor feedback led to a representation comparison and a request to clarify the intended meaning of non-equivariant.

**USE THIS SENTENCE**

We changed the next experiment because each result exposed a different limitation. We did not show that energy-derived forces had reached a universal limit.

The original recovery question remains open beyond the first procedure. The new representation study adds evidence; it does not replace that question silently.

Decision trail [1], [3]-[14]; prior-work context [15]-[20].

## Page 10: The MACE results

### Run B asked a small fitting question

Run B used the exact frozen 128-frame subset. It completed 2,048 epochs with 51,200 optimizer updates.

The primary final model had 0.001514599 of zero-force MSE. This equals a 99.8485% reduction on those fitted examples.

Its training-subset force RMSE was 4.619 meV/Å. The primary equal-connection MAE on validation was 37.823 meV/Å.

This showed that one conservative architecture could fit the same real labels. It did not isolate architecture, optimizer or budget effects.

### Run A shown the full-data prediction reference

Run A used 4,891 training records and completed 2,048 epochs. Independent CPU inference reproduced its saved validation metrics.

| Run A metric | Primary final EMA | Secondary selected checkpoint |
| --- | --- | --- |
| Checkpoint epoch | 2,047 | 1,987 |
| Validation force MAE, equal connection | 5.348957 meV/Å | 5.332516 meV/Å |
| Validation force MAE, pooled | 5.259833 meV/Å | 5.242323 meV/Å |
| Validation force RMSE | 26.410390 meV/Å | 26.071354 meV/Å |

The primary force MAE on training records was 0.575206 meV/Å. Validation residual-energy MAE was 7.461842 meV per molecule.

Run A used 2,002,944 committed optimizer updates and 10,014,720 training-frame exposures. Diagnostics and lost partial work are additional.

**WHAT WE CAN SAY**

MACE is a working predictive reference on this split. Its relaxation recovery remains unmeasured in our shared evaluator.

MACE and E3x have different architectures, optimizers, exposure budgets and hardware. Their score difference is not a controlled efficiency result.

Accepted MACE B and A reports [5]-[6]. Primary final and secondary selected checkpoints remain distinct.

## Page 11: The full-data E3x baseline

### The first short full-data fits were negative controls

Both 1,000-update energy and direct engineering fits selected update 0 as BEST. Their final force errors exceeded zero-force.

Their energy errors improved. This showed why energy improvement alone could not show a useful force baseline.

### We then froze three energy-baseline configurations

| Configuration | Width / learning rate | Loss weighting | BEST force MAE |
| --- | --- | --- | --- |
| A | 32 / 0.001 | Equal normalized weights | 42.108 meV/Å |
| B | 64 / 0.003 | Equal normalized weights | 40.370 meV/Å |
| C | 64 / 0.003 | Balanced weights | 34.587 meV/Å |

All three completed 200,000 updates. Independent inference reproduced their BEST checkpoint scores.

C reduced equal-connection force MAE by 44.664% against the 62.504 meV/Å zero-force reference. C passed A1.

At their force-selected checkpoints, C had energy MAE 126.990 meV per molecule. B had 28.178 meV per molecule.

The weighting change helped force error and worsened energy error in this seed. A versus B also changes width and learning rate.

### We corrected a scorer, without repeating the training

The first scorer rejected equivalent checkpoint states because dictionary order changed during serialization.

All 68 state leaves had equal values, shapes and dtypes. The repair used authenticated original payload bytes and stronger rejection controls.

The model weights and training records stayed unchanged. We kept the failed scorer outcomes and the linked correction.

**THE LESSON**

A software rejection and a scientific failure are different outcomes. We must show which one occurred before changing the model.

Engineering fits and accepted corrected baseline evidence [7]. Exact scorer identities and rejection tests remain in the audit record.

## Page 12: The controlled E3x results

The final-recipe study contains five formulations and two seeds. We reused C seed 0 and completed nine subsequent fits.

The proposal planned three seeds. We currently give results for two seeds per formulation. This is a stated reduction in scope.

| Formulation | Force MAE mean | Seed range | Energy MAE mean |
| --- | --- | --- | --- |
| Energy-derived | 34.893 | 34.587-35.198 | 128.725 |
| Ordinary direct | 38.186 | 37.291-39.082 | 76.684 |
| Consistency 0.1 | 38.024 | 36.773-39.275 | 87.006 |
| Consistency 1 | 34.683 | 33.723-35.643 | 112.341 |
| Consistency 10 | 39.725 | 39.060-40.390 | 161.662 |

Force values use meV/Å. Energy values use meV per molecule. Each row uses force-selected BEST checkpoints.

### Only consistency 0.1 met the declared matching rule

Its mean force-error difference from ordinary direct was 0.162 meV/Å. The gap was at most 10% of the smaller mean.

The gap also had to fit within each arm's seed range. This rule matched aggregate means, not every seed.

The matching rule does not show statistical equivalence.

Late baseline trajectories varied by 2.2-3.1 meV/Å standard deviation. BEST also sat below final errors in the subsequent fits.

The small matching gap is inside this training variation. The seed range is descriptive, not a confidence interval.

### Consistency improved agreement, with a cost

The normalized disagreement MSE was 0.49-0.61 for consistency 0.1. Ordinary direct gave 116-379 across its two seeds.

For consistency 0.1, mean elapsed time for fitting and scoring was 1.985 times ordinary direct. This excludes earlier diagnostics, failures and unmeasured coordination overhead.

The energy error was also higher for consistency 0.1. Better agreement alone did not show global conservativity or correct-shape recovery.

**THE SUPPORTED RESULT**

The penalty strongly reduced measured disagreement. It did not produce successful recovery in the original pilot.

Accepted controlled comparison, matching and bounded cost records [8]. Mean differences do not show population superiority.

## Page 13: The zero-recovery pilot

### The comparison used identical starts and targets

We used ten first stored branch frames from the five validation connections. These branches target six reference minima.

The training partition contains all six targets. The no-relaxation control matched zero of ten targets.

Success had two conditions. Maximum absolute force component had to be at most 0.01 eV/Å. Own-reference RMSD had to be at most 0.05 Å.

| Checkpoint | Correct / 10 | Wrong converged | Not converged |
| --- | --- | --- | --- |
| Energy C, seed 0 | 0 | 2 | 8 |
| Direct, seed 0 | 0 | 4 | 6 |
| Direct, seed 1 | 0 | 4 | 6 |
| Consistency 0.1, seed 0 | 0 | 6 | 4 |
| Consistency 0.1, seed 1 | 0 | 9 | 1 |

Across 50 model/branch outcomes, 25 met the force criterion at a wrong geometry. The other 25 did not converge.

These are paired observations on five connections. They are not 50 independent molecular pathways.

### The original procedure had an important limitation

It proposed a coordinate change from step scale times predicted force. The procedure increased the step scale and used force-direction backtracking.

The scale started at 0.2 Å²/eV and could reach 1.0. Each branch had 1,500 force attempts.

The procedure had no energy-decrease rule. It capped the scale, but not each atom displacement.

**A2 OUTCOME**

Every evaluated checkpoint recovered 0/10. The energy baseline failed A2, so this pilot cannot show comparative recovery superiority.

We kept the original procedure and outcomes. Later controls are post-hoc diagnosis, not a replacement pilot result.

Accepted recovery procedure and original 50 outcomes [9]. The main test and MACE recovery remain unmeasured.

## Page 14: The failure analysis

### Static reference-force checks and reference-start controls

Each model exceeded the force criterion at all 11 exact DFT minima. The reference-label maximum force remained below 2.745 meV/Å.

Nearby geometries could still satisfy both criteria. These checks do not show that no nearby solution exists.

| Starting from the six target references | Correct recovery |
| --- | --- |
| Energy C, seed 0 | 4/6 |
| Direct, seed 0 | 1/6 |
| Direct, seed 1 | 0/6 |
| Consistency 0.1, seed 0 | 0/6 |
| Consistency 0.1, seed 1 | 0/6 |

### Graph diagnostics qualified the convergence counts

All 23 force-converged dual-head endpoints had disconnected 5 Å neighbor graphs. They were not intact molecular conformations.

Seven had every atom isolated. The direct force becomes exactly zero in that limit.

Other disconnected endpoints could have internal fragment forces. We must not apply the automatic-zero explanation to all 23 endpoints.

### Four exact trace replays showed two different behaviors

| Selected trace | Largest accepted atom jump | Final graph |
| --- | --- | --- |
| Direct | 436.721 Å | Disconnected |
| Energy | 268.047 Å | Disconnected |
| Consistency | 0.086492 Å | Connected |
| Energy | 0.006083 Å | Connected |

The replays reproduced the saved endpoints. Only two of these four selected traces separated.

**CAUSAL LIMIT**

The evidence identifies coupled model and procedure limitations. It does not show that the solver alone caused every failure.

A displacement cap is a justified next control to design. We have not yet shown its recovery performance.

155-evaluation diagnostic, reference starts, graph analysis and paired traces [10]-[11]. Selected traces do not give population rates.

## Page 15: Professor feedback and the next study

| Professor feedback | Our response |
| --- | --- |
| Use a graph neural network | Our E3x and MACE models already use graphs and message passing. |
| Compare equivariant and non-equivariant models | We must clarify the intended symmetry contrast. We completed a narrower internal-feature comparison. |
| Examine long-range interactions beyond 7 Å | Keep this as a later question. We have not measured a long-range cause of the current failures. |
| Use Matbench Discovery for model ideas | Use it for model scouting, with domain and label checks. No leaderboard model has entered this study. |
| Explore ORCHARD compute | Treat access as unavailable until the access checks pass. Current small-model work uses the local CPU. |

### The completed study changes internal features

| Arm | Internal features | Width | Parameters |
| --- | --- | --- | --- |
| E64 | Degree 1; scalar and vector features | 64 | 43,521 |
| SP | Degree 0; scalar features | 88 | 43,473 |
| S64 | Degree 0; scalar features | 64 | 25,473 |

SP matches E64 parameter count within 0.1103%. S64 holds width constant instead. These controls answer different questions.

All arms predict invariant scalar energy. All forces come from its gradient, so all output force fields remain equivariant.

This is not a rotation-symmetry model versus an unconstrained rotation-learning model. It is also not graph versus nongraph.

We kept the data, seed-paired sample streams, loss, optimizer and 200,000-update budget fixed. We tuned the recipe on E64 first.

Professor assessment and reviewed representation protocol [12]-[14]. Clarification of non-equivariant remains open.

## Page 16: The latest completed results

All four new scalar fits completed 200,000 updates. We reused the two E64 fits after source, recipe and reproduction checks.

| Arm / seed | BEST | LAST | Late20 mean |
| --- | --- | --- | --- |
| E64-0 | 34.587 | 37.791 | 39.008 |
| E64-1 | 35.198 | 39.340 | 39.029 |
| SP-0 | 41.046 | 52.522 | 46.276 |
| SP-1 | 41.669 | 43.647 | 45.743 |
| S64-0 | 40.264 | 49.632 | 44.062 |
| S64-1 | 38.820 | 42.195 | 42.471 |

All values are equal-connection validation force MAE in meV/Å. Late20 averages checks from update 190,500 through 200,000.

Chart: BEST and Late20 means, with the two-seed ranges. Numeric values are in the table above.

SP minus E64 gives a two-seed mean difference of +6.465 meV/Å for BEST and +6.991 meV/Å for Late20.

S64 minus E64 gives +4.649 meV/Å for BEST and +4.248 meV/Å for Late20. A positive difference means lower error for E64.

Both scalar arms have higher error for each seed at BEST and Late20. The sign is consistent in this fixed-recipe comparison.

**CLAIM LIMIT**

Two seeds and five validation connections support a descriptive result. They do not isolate a causal symmetry effect or show best-in-family performance.

The four scalar training durations sum to 2.110 CPU elapsed hours. This excludes preflight checks, scoring and the publication interruption.

Saved-result arithmetic and publication identities passed independent checks. The broader scientific interpretation still needs independent result review.

Saved queue, independent BEST/LAST scores, paired summary and verified per-fit publications [13]. Full scientific interpretation remains bounded to this recipe.

## Page 17: Literature and the novelty boundary

We kept twelve targeted reading cards and the primary MACE reference. This is a focused literature map, not a systematic review.

| Literature group | What it contributes | What it does not show here |
| --- | --- | --- |
| Landscape17 [15] | A task setting for landscape errors and pathway recovery | Our study is not a full kinetic-network benchmark. |
| GDML, PaiNN, E3x, MACE [16] | GDML: conservative forces. PaiNN: scalar/vector features. E3x: equivariant layers. MACE: reference architecture. | Their published scores do not reproduce our split or recipe. |
| Forces are not Enough; smooth potentials [17] | Task behavior matters beyond stored-example force error | Their simulation criteria are not our recovery rule. |
| Bigi; Zhao; unconstrained potentials [18] | Direct-force limits depend on task, data and corrections | Direct forces do not fail universally. |
| Takamoto; When Forces Disagree [19] | Previous studies used consistency training and disagreement diagnostics | Our squared penalty is not a new general method. |
| Gradient-label study [20] | Data quality and force labels affect learning | A learning curve does not prescribe our update budget. |

### Our contribution is an empirical comparison

We compare error, computing cost and reference-shape recovery on held-out connections. We also preserve the negative outcomes and procedure diagnosis.

The scalar study adds a controlled internal-representation question. It does not show a new force-model family.

### Long-range and foundation-model work remains prospective

Matbench Discovery primarily evaluates crystal and materials tasks. Its ranks do not show isolated-aspirin accuracy.

We did not do checks of the current leaderboard order. No external foundation-model weights entered our experiments.

A larger cutoff is not the same as explicit long-range electrostatics. Two message-passing layers cannot connect disconnected graph components.

Latent Ewald Summation is a possible reading direction. We have not implemented it or shown that it accounts for our failures.

Primary literature [15]-[21]. The professor reference to 7 Å is exploratory advice, not a universal physical threshold.

## Page 18: Engineering lessons and compute

### Keep result checks separate from operating failures

Colab setup, environment creation, formatting and Drive-mount failures delayed MACE. We kept their logs and corrected the affected code.

The original owner and device restrictions also made resume unnecessarily difficult. Later notebooks removed fixed pauses and added continuation paths with state checks.

A100 80 GB memory use was small. A copied-checkpoint diagnostic admitted A100 40 GB continuation with saved training state.

That diagnostic did not show the cheapest GPU for every workload. We did not show a complete measured GPU cost ranking.

### The current small-model study uses a simpler CPU path

The existing trainer saves full state every 500 updates. The wrapper skips complete fits after identity checks and resumes from eligible complete saves.

A fresh E64 reproduction matched saved validation at updates 0, 500 and 1,000 within the declared tolerance.

An interrupted scalar control resumed to the same 2,000-update state as an uninterrupted control, within 1e-10.

These controls include parameters, optimizer state, random-generator state and validation history. They do not show new scientific model quality.

### The latest queue interruption was a publication issue

SP-0 finished and pushed successfully. An immediate GitHub PR-head assertion then stopped the queue before the next fit.

The original remote files passed identity checks. We restored the missing publication receipt and added bounded retries for these checks.

The queue skipped completed SP-0 and finished the other three fits. We did not repeat training to repair publication.

**COMPUTE ASSUMPTION**

Use local CPU or Colab until new cluster access passes its checks. ORCHARD and PSC are not necessary for the completed representation study.

The coordinator can replenish Colab units. The quoted purchasing envelope is planning information, not a balance from account checks or a complete billing total.

Preserve source, data, configuration and checkpoint identities. Publish outcomes from a separate checkout while a numerical run keeps its revision fixed.

Reproduction, interruption and publication receipts [13]-[14]; MACE operating trail and kept cost limits [5]-[6].

## Page 19: Team ownership and remaining work

Responsibilities changed after the proposal. Each member must approve the record of their own work and report section before submission.

| Member | Work and responsibilities | Next substantive deliverable |
| --- | --- | --- |
| Shedrack Eze | Data preparation, model/training integration, first MACE results and current experiments | Integrate the comparisons, preserve evidence and coordinate the report and study decisions. |
| Olufunke Abodunrin | Authored the shared recovery evaluator | Integrate the trained MACE adapter with the runner/reporter; examine the recovery method and document changes. |
| Muhammed Idris | Owns the remaining MACE loader and independent reproduction | Implement checkpoint loading, identity/environment admission and prediction parity against kept outputs. |
| Emmanuel Aaron | Owns physical-consistency and cost analysis | Implement analytic controls, closed-loop work, synchronized inference timing and connection-level comparative analysis. |

### Idris does not need to train MACE again

Run A and Run B are complete. The handover includes an artifact inventory with identity checks and reproducibility instructions.

Shedrack prepared a synthetic-tested calculator bridge. This does not complete Idris's real loader, admission or parity work.

Packet delivery and access still need checks. Olufunke owns integration after the loader produces the agreed outputs.

### Aaron has independent work that can start now

Use conservative and nonconservative analytic fields first. Then select the loop geometries, refinement method, units and timing boundaries before model evaluation.

Do not label finite sampled loop work as proof of global conservativity. Separate compilation, warmup and synchronized inference.

### Record real contributions

Each owner needs meaningful commits, verification evidence, analysis and a report section. Proposed ownership is not completed contribution credit.

Use the shared repository tasks. Do not repeat completed diagnosis only to create commit activity.

Current task files, contribution ledger and MACE handover [22]. Teammate completion beyond the recorded evidence remains unconfirmed.

## Page 20: The next actions

### Finish the midterm as a separate release task

The current course-format draft has five pages. The saved-rubric audit found the substantive requested material.

That audit does not show live Canvas compliance, submission, team approval or grader access.

The saved schedule gives 23 October for the midterm, 4 December for video and 9 December for the final report.

Compare those dates with the live course instructions. The repository remains private, with reviewed work on stacked pull requests.

### Prioritize the scientific work in this order

- Complete Idris's loader admission and prediction parity. Do not repeat MACE training for this task.
- Run the unchanged MACE ten-start and six-reference controls after the integration passes its checks.
- Freeze a displacement-capped solver proposal. Compare the same checkpoints and starts under a separately declared procedure.
- Finish the bounded interpretation of the completed representation results. Decide whether they belong in the midterm or later analysis.
- Measure closed-loop work and matched-hardware force-inference cost. Preserve unavailable and failed outcomes.
- Freeze model, solver and analysis choices before any one-time trained main-test evaluation.

### Defer scope that cannot answer the immediate question

Keep paracetamol, explicit long-range interactions, foundation-model benchmarking and extra MACE seeds outside the immediate critical path.

Any later addition needs a clear question, a budget and a separate protocol. A larger compute balance alone does not justify it.

**MEETING DECISIONS**

Agree on owners and dates. Ask which non-equivariant comparison the professor wants. Make sure the team has artifact access and graders have repository access.

Proposed next checkpoint: loader/parity status and report-owner approvals within 24 hours. The team must agree this target; it is not an existing commitment.

Saved-rubric audit and current draft [23]; team tasks [22]; representation and professor scope [12]-[14].

## Page 21: Questions from a professor

### 1. What is the research question?

Does agreement between direct forces and learned-energy gradients improve recovery of the correct reference shape at comparable force error and cost?

### 2. What have you actually completed?

We audited the data, trained MACE references, completed the controlled E3x fits, measured a recovery pilot and diagnosed its failures.

We also completed four scalar-feature fits and compared them with two completed E64 fits.

### 3. Did you find the dataset was saturated?

No. Our early E3x setting reached a training plateau. MACE fit the same 128 labels, so a fundamental data limit is unsupported.

### 4. Did the consistency penalty work?

It strongly reduced measured direct-force disagreement. It did not produce successful recovery in the original procedure.

### 5. Why is the original hypothesis still unresolved?

All evaluated checkpoints recovered 0/10, including the energy baseline. This pilot cannot rank successful recovery among the methods.

### 6. Why not select the lowest force MAE and stop?

Stored-example force error is a local prediction score. A sequence of coordinate changes can still give a wrong endpoint.

### 7. Is the new scalar model non-equivariant?

Its internal features are scalar. Its invariant energy still produces equivariant gradient forces. It does not remove built-in output symmetry.

**QUESTION TO ASK THE PROFESSOR**

Do you mean scalar versus vector internal features, or a model that learns rotation symmetry from rotated examples?

Answers use the completed evidence and stated limits [1]-[14]. The last question remains unanswered.

## Page 22: Technical and scope questions

### 8. Does MACE show that equivariance caused the improvement?

No. MACE also changes architecture, optimization, training exposure and hardware. It is an external prediction reference, not a controlled symmetry ablation.

### 9. Why does E64 have lower error in the scalar study?

Under the fixed recipe, both scalar arms have higher BEST and Late20 error in both seeds.

We cannot isolate the cause. We selected the recipe using E64 and did not tune scalar recipes separately.

### 10. Does 0/10 show that direct forces cannot work?

No. The energy baseline also scored 0/10. The result concerns these checkpoints, starts, targets and this procedure.

### 11. Did the solver cause every failure?

We observed extreme steps and graph separation in selected traces. Reference-start controls also showed model limitations.

These results support a coupled diagnosis. They do not isolate the solver as the sole cause.

### 12. Why are the targets not truly unseen shapes?

We held out whole connections, but the training partition contains all reference minima. This brief identifies the shared-minimum limitation.

### 13. Are long-range interactions the missing cause?

We have not measured that cause. A 5 Å cutoff limits direct edges, while message passing can transmit information along connected paths.

Long-range electrostatics is a separate model question. A materials leaderboard does not answer it for isolated aspirin.

### 14. Is the final report ready?

The midterm draft is ready for team examination after its recorded wording corrections. Release checks and the live course instructions remain open.

**SAFE CLOSING**

We can tell the team what failed, what improved and what is still unknown. Our next control separates reference-model quality from the relaxation procedure.

These answers preserve the validation-only and descriptive-result boundaries. No final scientific or submission claim follows.

## Page 23: Technical terms

| Term | Project meaning |
| --- | --- |
| MLIP | Machine-learning interatomic potential: a model of atomic energy or forces. |
| DFT | Density-functional theory: the reference electronic-structure calculation used for labels. |
| Geometry / conformation | A specific set of atomic positions; molecular shape. |
| Connection / branch | A transition-state connection has two descent branches. The split keeps them together. |
| Reference minimum | An archived stable DFT geometry used as the structural target. |
| Training / fit | Adjustment of model parameters using the training records and declared objective. |
| Validation | The partition used for settings, checkpoint selection and the pilot in this brief. |
| Test partition | The held-out partition reserved for the final frozen trained evaluation. |
| Checkpoint | A saved model or full training state, with its identity and configuration. |
| BEST / LAST | The first strict validation-force minimum / the final full-budget checkpoint. |
| Late20 | Arithmetic mean of the 20 scheduled checks from 190,500 through 200,000. |
| MAE / MSE / RMSE | Mean absolute error / mean squared error / square root of MSE. |
| RMSD | Root-mean-square displacement after the declared alignment and symmetry mapping. |
| Gradient | Coordinate gradient: derivative of scalar energy with respect to atomic positions. |
| Parameter gradient | Derivative of a loss with respect to model weights; used in the training diagnostics. |
| Direct / energy-derived force | Vector-head prediction / negative energy gradient. |
| Consistency residual | Normalized difference between direct force and auxiliary energy-derived force. |
| Training plateau | Limited improvement under particular tested settings and budget. No universal limit is implied. |
| Recovery floor | The observed zero-success result that prevents ranking recovery in this pilot. |
| Seed | A declared random initialization and sampling choice; it is not another independent connection. |
| Parity / admission | Prediction agreement against kept outputs / identity and environment checks before use. |

Symbols: E is energy, R is atomic position, F is force and lambda is the consistency-penalty strength.

The gradient-force relation is F = -grad_R E. This relation gives conservative forces. It does not show agreement with the reference energy.

Language policy: ASD-STE100 Issue 9 writing rules and technical terms in the glossary. No independent language-conformance certification is claimed [24].

## Page 24: Project claims and their evidence

The source map points to kept reports and manifests. Old status sentences give the status at their original dates; current status uses the saved queue.

| ID | Source of record | Use in this guide |
| --- | --- | --- |
| [1] | proposal.tex; CLAIM_AUDIT.md | Original question, planned controls, exclusions and novelty limits |
| [2] | Landscape17 source audit; unit and split manifests; empirical branch-reference manifest | Units, partitions, matching, corrected targets and paracetamol block |
| [3] | Force-learning, coverage, representation-gradient and loss-balance diagnostic reports | Early learning controls, force fractions and local-gradient limits |
| [4] | Corrected width/rate report; capacity, late-rate, offset and force-isolation findings; degree-two closeout | Bounded diagnostic trail, mixed outcomes and invalid attempt |
| [5] | reports/mace-runB-fit.md | 128-frame fitting result and its limited interpretation |
| [6] | reports/mace-runA-fit.md; independent production rescore | Full-data reference metrics and selected/final distinction |
| [7] | reports/e3x-first-engineering-fits.md; e3x-full-data-baseline.md | Short negative fits, accepted A/B/C and scorer correction |
| [8] | Controlled-fit result record; midsemester Table 1 source map | Five formulations, two seeds, matching, disagreement and bounded cost |
| [9] | E3X_VALIDATION_RECOVERY_2026-10-17.md | Original ten-start pilot, force/match rules and all 50 outcomes |
| [10] | Recovery-diagnosis report and review addendum | 155 static evaluations and graph qualifiers |
| [11] | E3X_PAIRED_TRACE_2026-10-18.md | Four exact endpoint replays, accepted atom jumps and 30 reference-start relaxations |
| [12] | Representation specification and professor-feedback assessment | New scientific contrast and advice limits |
| [13] | Saved representation queue, BEST/LAST scores, paired summary and outcome publications | All six rows, four new fits and saved arithmetic with independent checks |
| [14] | CPU reproduction, resume and publication-recovery receipts | Operational validity and preserved interruption |
| [22] | TEAM.md; docs/team/*_TASKS.md; MACE handover and contribution ledger | Owner assignments, interfaces and completion boundaries |
| [23] | Five-page midsemester report and saved-rubric audit | Draft content and remaining release checks |

The editable source package provides exact file paths and SHA-256 identities. No restricted test arrays or model checkpoints accompany this brief.

Repository: https://github.com/Shedrackeze002/idl-molecular-force-consistency . Snapshot base: PR 10, commit 3fa5969. The repository is private.

## Page 25: Primary literature for follow-up

| ID | Primary references |
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

- For the original scientific motivation, read Landscape17 and The dark side of the forces.
- For scalar/vector features, read PaiNN and E3x. Use MACE for the reference architecture.
- For consistency prior work, read Takamoto and the specific saved When Forces Disagree manuscript.
- For later long-range work, start with the LES papers. Do not present screened abstracts as completed implementations.

The kept When Forces Disagree PDF is an anonymous manuscript. We do not know its current acceptance status.

The evidence map keeps exact paper versions and access history. We do not transfer MD17, crystal or reaction-search scores to aspirin.

**AFTER THE MEETING**

Record agreed owners, dates and scope changes. Keep measured results, interpretations and proposed work as separate entries.

Official paper links and exact internal source paths accompany the editable Markdown source. This package does not include the official STE standard.
