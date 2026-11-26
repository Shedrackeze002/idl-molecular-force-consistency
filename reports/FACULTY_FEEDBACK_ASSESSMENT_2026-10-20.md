# Faculty feedback and project implications — 20 October 2026

This records feedback reported by Shedrack on 20 October from a professor meeting the previous day, provisionally 19 October. The professor's name and exact meeting record were not supplied. Advice and research priorities are not experimental findings or an approved project change. No model, protocol, owner, report or resource allocation is changed here.

## Feedback as reported

- Use a graph-based neural network.
- Consider long-range interactions beyond 7 Å as a longer-term direction.
- For the class project, consider equivariant versus nonequivariant models, or a conventional neural-network versus graph-network comparison.
- Consult Matbench Discovery for current model families; the professor described energy-derived force fields as a mature or saturated area.
- Investigate CMU ORCHARD and other clusters. Ask the PI first; the professor offered to help if access cannot be arranged.

The threshold and saturation assessment are the reported advice. They are not established universal cutoffs or a verified claim that a research field has no open problems.

## What the current project already does

The controlled E3x study already uses a graph neural network with equivariant message passing. Atoms supply nodes; pairwise displacement indices supply edges. The smooth 5 Å cutoff weights the local basis, and two message-passing interactions update scalar/vector features. The accepted recipe uses 64 features and maximum degree one. `models/backbone.py`, `models/energy.py`, `models/direct.py` and `configs/aspirin-e3x-v3-C.template.json` establish these facts in the shared implementation. The source-bound accepted results and report are authoritative; an older root checkout does not contain all later heads.

All three force formulations share this backbone. The existing comparison tests energy-derived, direct and consistency-trained force formulations. It does **not** compare graph against nongraph models or equivariant against nonequivariant architectures. MACE is an external predictive reference, not a controlled architecture comparison.

A graph network is a kind of neural network. Therefore a proposed neural-network-versus-GNN comparison must define a nongraph baseline, such as an explicitly specified coordinate MLP; the labels alone do not define mutually exclusive model classes.

## Scope recommendation, not an adopted experiment

For the October 23 midterm, retain the completed experiments and their negative recovery outcome. Do not recast them as an architecture ablation. The report remains unchanged by this memo. A short prospective direction could be added later through normal report review if the team chooses it; no new result should be invented to accommodate the feedback.

For a possible class-project extension, prefer assessing **scalar/invariant graph features versus scalar-plus-vector equivariant graph features**, keeping energy-derived forces in both arms initially. This separates an architecture question from the direct-versus-gradient force question. Important terminology: the gradient forces of a rotation-invariant scalar energy are themselves rotation-equivariant. This would compare internal feature representations, not an equivariant force output against a necessarily nonequivariant one. If the professor intended genuinely nonequivariant predictions, clarify that intent before choosing the baseline.

This is one candidate, not a training plan. Before adoption, document the hypothesis, parameter/capacity and compute matching, data splits, force formulation, validation selection, rotation tests, seed/budget choices and acceptance criteria. A fixed-budget versus tuned-each-arm comparison answers different questions. The current direct head explicitly requires degree-one features: simply setting its `max_degree=0` is not a valid nonequivariant direct-force baseline. Existing recipe selection was energy-only and is not automatically a fair hyperparameter search for a new architecture.

Model accuracy and symmetry checks can be compared even if recovery remains at zero. A claim about recovery benefit would additionally require a useful, separately reviewed procedure control. Existing MACE loader/parity and unchanged solver controls remain relevant. Do not multiply new architectures by every penalty, seed and cutoff before a feasible comparison is defined. Do not reassign Aaron's or Idris's work without coordination.

## Long-range direction and literature

Current 5 Å local edges do not mean every dependence beyond 5 Å vanishes: information can pass indirectly along connected multi-hop paths. This is not equivalent to explicit electrostatics or dispersion at arbitrary separation. The advice's 7 Å is an exploratory scale, not a universal physical boundary. Increasing the neighbor cutoff changes local support and cost; by itself it is not a controlled test of explicit long-range physics.

Our disconnected-endpoint observation establishes a limitation of the local model/procedure combination. It does not establish that missing forces beyond 7 Å caused the failures or that adding electrostatics would repair them. The trace also exposes extreme accepted steps, so solver limitations must remain separate. A long-range study needs suitable larger/separated systems, reference labels and distance-dependent tests; the existing single-molecule pathway set has not been shown sufficient to identify that mechanism.

Two useful starting sources are [Latent Ewald Summation](https://arxiv.org/abs/2408.15165) (Cheng; journal version [npj Computational Materials 11, 80 (2025)](https://www.nature.com/articles/s41524-025-01577-7)) and [the LES perspective](https://arxiv.org/abs/2512.18029) (Kim and Cheng). The first demonstrates limits of local models on charged/polar dimers and other systems, including with message passing. The second discusses adding electrostatics using energy/force labels. These support a literature direction, not a validated extension to our aspirin models. Abstract-level screening only; no paper implementation or measured result is claimed.

The professor's saturation comment is useful novelty advice, but energy-derived forces remain a relevant control. Our proposal is not to claim the gradient-force construction as new. The [eSEN paper](https://arxiv.org/abs/2502.12147) instead studies whether predictive error translates to downstream physical quality, a question adjacent to ours. Its abstract alone does not establish our novelty or transfer its materials results to aspirin.

## Matbench Discovery: model scouting, with domain checks

The [official benchmark repository](https://github.com/janosh/matbench-discovery) describes tasks around crystal stability and other materials-simulation properties. Its warning against treating ranking as universal model endorsement matters here. The requested website and `/models` page were unavailable through web retrieval (403/unavailable); current official repository documentation was checked instead. **No exact current leaderboard order or universal best model was established.**

Candidates to read about, not an installation or scoring shortlist:

| Family/source | What to learn | Restriction for this study |
|---|---|---|
| [eSEN](https://arxiv.org/abs/2502.12147) | Smoothness, energy conservation and downstream evaluation | Materials-trained results do not establish molecular transfer or a clean architecture control |
| [Orb family documentation](https://github.com/orbital-materials/orb-models/blob/main/MODELS.md) | Both conservative/direct variants; separate molecular OrbMol versions and explicit-electrostatics variants | Exact version, training domain, label theory, charge/spin, licensing and training overlap need checking; no weights admitted |
| Existing MACE reference | Reuse established predictive evidence and complete independent loader/parity | Avoid duplicate training; recovery remains unmeasured |

Orb's documentation distinguishes bulk-crystal models from molecular models. For aspirin, molecular-domain documentation is worth inspecting rather than blindly choosing a crystal leaderboard winner. Even a suitable pretrained model has unmatched training data/capacity and possible overlap; it cannot isolate equivariance as a causal factor. Element coverage alone is not enough. Inspect exact license/availability, labels and energy gauge, allowed geometry/charge/spin, data overlap and compute cost before proposing admission. No additional foundation-model inference is approved or launched by this memo.

## ORCHARD access and next concrete request

CMU's [research-computing office](https://www.cmu.edu/rcd/) confirms ORCHARD is a Google Cloud research resource. The [FAQ](https://www.cmu.edu/rcd/faq.html) lists H100 GPUs and ACCESS-ID onboarding, with no existing ACCESS award required just to create the ID. ORCHARD access is not automatically conferred by the team's PSC request.

The [current access page](https://www.cmu.edu/rcd/request.html) asks the PI to request the group. Bootstrap allows one GPU for 30 minutes, with possible interruption; General allows one GPU for 12 hours after experience is demonstrated. Approval is not guaranteed. A 25 GB home quota and separately arranged storage mean full checkpoint histories need an explicit storage plan. Pilot compute terms do not establish zero storage/API cost or guaranteed queue availability; confirm current costs with RCD.

Next access step: Shedrack asks the PI about sponsoring the ORCHARD request, with a concise project summary, prior GPU execution evidence, requested single-GPU job needs and checkpoint/storage estimate. If sponsorship/access cannot be arranged, follow up with the professor as offered. Existing ACCESS ID can be included after verification. No application, message, account change, package install or remote job was performed. Do not make the October 23 report depend on provisioning.

## Decision and ownership boundaries

Recommended sequence: complete the midterm release checks; retain the current scientific question while assessing a bounded representation comparison; pursue ORCHARD sponsorship in parallel; keep explicit long-range modeling a separate feasibility/literature direction. These are recommendations awaiting the coordinator/team's choice, not approved experiments or claimed teammate acceptance. No new implementation or launch plan is frozen. Any adopted experiment must receive the usual two independent technical reviews and a reviewed budget before execution.


## Coordinator update — Colab-only planning, 20 October 2026

The coordinator confirmed that all GPU feasibility and experiment plans must assume Colab alone until ORCHARD access is actually granted. ORCHARD discussions proceed separately; neither ORCHARD nor pending PSC/other cluster capacity is a dependency for completing the project. A subsequent clarification makes current unit balance a replenishment detail rather than a reason to restrict the study: up to $100 is available to buy 1,000 additional CU at the coordinator's quoted $10/100 CU. This is available purchasing capacity, not a verified account balance or independent retail-price check. Estimate consumption and arrange top-ups as needed. The relevant constraints are Colab hardware availability, memory, session behavior and reliable checkpoint resumption. No new experiment, purchase or resource allocation is performed by this update.
