# Literature foundation — selective primary reading

This is a working foundation for the later literature review, not a finished systematic review or proof of novelty. Twelve primary papers were read in targeted sections. Published papers, versioned author manuscripts and the user-provided anonymous submission are distinguished below. Local PDFs and failed downloads are indexed in [download-manifest.json](download-manifest.json); [registry.json](registry.json) retains versions, read sections, access status and SHA256 identities.

Our research question remains whether consistency training improves reference stable-shape recovery on unseen whole transition-state connections, controlling architecture, force accuracy and cost. Current energy-derived training diagnostics are engineering prerequisites; none tests this main hypothesis yet.

## Claim map

| Paper-facing point | Primary evidence | Boundary for our study |
|---|---|---|
| Point errors need physical-task evaluation | LIT001,002,003,004,009 | Evaluate paired reference-minimum recovery; MD or reactive TS success is a different task. |
| Consistency-assisted training already exists | LIT011,012 | Define the exact penalty and evaluation distinction; do not claim invention of disagreement or consistency regularization. |
| Disagreement is an imperfect diagnostic | LIT012 | Retain consensus failures, calibration limits and DFT/reference outcomes. |
| Direct models can still be useful | LIT002,010 | Report task, data scale, accuracy and corrections; avoid categorical failure claims. |
| Data support and optimizer budget differ | LIT005,007,008 | Our fixed continuation changes updates only. It cannot prove convergence or insufficient capacity. |
| Architecture controls need provenance | LIT005,006 | E3x16features is not a PaiNN replication; document actual layers, weights, precision and schedule. |

## Reading cards

### LIT001 — Global properties of the energy landscape: a testing and training arena for machine learned potentials

Vlad Cărare ; Fabian L. Thiemann ; Joe D. Morrow ; David J. Wales ; Edward O. Pyzer-Knapp ; Luke Dicks. npj Computational Materials12:9 (2026), DOI10.1038/s41524-025-01878-x; user-provided publisher PDF; earlier targeted reading used arXiv2508.16425v1. [Primary source](https://www.nature.com/articles/s41524-025-01878-x).
Read: publisher PDF pp1,6–7: abstract/introduction, Fig4/5/6 mapping and topology, discussion opening; title/year/authors verified visually. Local access: saved_pdf_user_provided.

Source finding: Pointwise accuracy does not ensure recovery of the reference landscape. Pathway augmentation helps, while spurious minima remain in the models studied.

Our inference/action: Use its stationary points and branches to define whole-connection splits and reference-shape recovery. Preserve failures by branch and distinguish not found by a search from demonstrated absence.

Limits: Our128frame fitting diagnosis is smaller and is not its full training or kinetic-transition-network benchmark. Missing-transition-state headlines cannot be transferred to our models.

Saved version: [landscape17-published.pdf](papers/landscape17-published.pdf); SHA256 `f43b0df0e86039acff71f71c79b44c5107cab1ef4880d8d766b6a32f9faf85f7`. Targeted local reading; previous web/version provenance remains in the registry and intake snapshots.

### LIT002 — The dark side of the forces: assessing non-conservative force models for atomistic machine learning

Filippo Bigi ; Marcel F. Langer ; Michele Ceriotti. ICML2025, PMLR267:4384–4414. [Primary source](https://proceedings.mlr.press/v267/bigi25a.html).
Read: published PDF and author HTML: nonconservative-force analysis, geometry optimization, MD and tandem conservative corrections. Local access: saved_pdf.

Source finding: Nonconservative models can have physical and optimization failures beyond their point errors. Conservative corrections and multiple-time-stepping offer task-dependent alternatives.

Our inference/action: Motivate paired direct/conservative task evaluation and track inference cost alongside relaxation failures.

Limits: Neither a guarantee of universal direct-force failure nor evidence that our current energy-derived baseline has the same failure. Molecular shape recovery differs from MD stability.

Saved version: [bigi2025.pdf](papers/bigi2025.pdf); SHA256 `d221817447096c8bc85048f2666ee62223514d6491f0319c0ad6735e8fe71bf4`.

### LIT003 — Harnessing Machine Learning to Enhance Transition State Search with Interatomic Potentials and Generative Models

Qiyuan Zhao ; Yunhong Han ; Duo Zhang ; Jiaxu Wang ; Peichen Zhong ; Taoyong Cui ; Bangchen Yin ; Yirui Cao ; Haojun Jia ; Chenru Duan. Advanced Science12(34):e06240 (2025), DOI10.1002/advs.202506240; user-provided publisher PDF. [Primary source](https://pmc.ncbi.nlm.nih.gov/articles/PMC12442704/).
Read: publisher PDF pp1,5–6: abstract and §2.3 direct/autograd/autograd* force schemes, Fig3 and DFT-Hessian follow-up; earlier primary HTML reading retained. Local access: saved_pdf_user_provided.

Source finding: End-to-end transition-state search is not captured adequately by energy/force errors alone. Direct and energy-gradient force variants differ in Hessian behavior and search outcomes.

Our inference/action: Define a physical success criterion before comparing scalar errors; retain the distinction between intended endpoint recovery and numerical optimizer termination.

Limits: Reactive TS/GSM/Hessian/IRC search is not our conformational minimum-recovery task. Its DFT-Hessian intervention does not validate our planned consistency penalty.

Saved version: [zhao2025-published.pdf](papers/zhao2025-published.pdf); SHA256 `13aca4483cf27ed036a0e3fa12f6025cb4b643c6e7cf24c008b5356d1171af37`. Targeted local reading; previous web/version provenance remains in the registry and intake snapshots.

### LIT004 — Learning Smooth and Expressive Interatomic Potentials for Physical Property Prediction

Xiang Fu ; Brandon M. Wood ; Luis Barroso-Luque ; Daniel S. Levine ; Meng Gao ; Misko Dzamba ; C. Lawrence Zitnick. ICML2025, PMLR267:17875–17893. [Primary source](https://proceedings.mlr.press/v267/fu25h.html).
Read: published PDF and author HTML: conservative-force/smoothness discussion, cutoffs, physical-property evaluation. Local access: saved_pdf.

Source finding: Energy-derived forces alone do not ensure practical finite-step energy conservation. Smoothness and choices such as cutoffs matter for physical predictions.

Our inference/action: Keep spatial smoothness and numerical integration controls in the eventual physical evaluation.

Limits: Our finite-difference step failures concern directions in parameter space. They do not establish a nonsmooth atomic potential, a bad cutoff, or the causes investigated here.

Saved version: [esen2025.pdf](papers/esen2025.pdf); SHA256 `4a1cbd7f05f68846f221ac2e07acb41c04123f4db8c6f030f83d56a750afa2c7`.

### LIT005 — Equivariant message passing for the prediction of tensorial properties and molecular spectra

Kristof T. Schütt ; Oliver T. Unke ; Michael Gastegger. ICML2021, PMLR139:9377–9388. [Primary source](https://proceedings.mlr.press/v139/schutt21a.html).
Read: published PDF: equivariant model and §5 implementation/training settings, MD17 table context. Local access: saved_pdf.

Source finding: PaiNN uses equivariant scalar/vector representations. The reported setup includes128features, Adam and validation-plateau learning-rate reduction.

Our inference/action: A separately reviewed schedule/capacity study could follow the fixed-budget diagnostic if needed.

Limits: Our16feature E3x model with fixed learning rate is not a PaiNN replication. Original MD17 results cannot be substituted for rMD17 or whole-connection Landscape17 performance.

Saved version: [painn2021.pdf](papers/painn2021.pdf); SHA256 `dde25433fd28ff0cc34664c57b070cfcdec662d90f03bb164f4c455b0b638e13`.

### LIT006 — E3x: E(3)-Equivariant Deep Learning Made Easy

Oliver T. Unke ; Hartmut Maennel. arXiv2401.07595v3,11November2024. [Primary source](https://arxiv.org/abs/2401.07595).
Read: author PDF: representation conventions, equivariant layers and framework description. Local access: saved_pdf.

Source finding: E3x supplies equivariant representations and building blocks for molecular models.

Our inference/action: Cite the framework underpinning the controlled architecture and document our actual layer/feature configuration.

Limits: Framework availability and equivariance do not establish sufficient model capacity, optimizer convergence, numerical smoothness of every configuration, or physical-task performance.

Saved version: [e3x2024.pdf](papers/e3x2024.pdf); SHA256 `c198a25a29e0c0001c8d9bc5c72124d816813f81bf91fa9e17be456016557c4d`.

### LIT007 — On the role of gradients for machine learning of molecular energies and forces

Anders S. Christensen ; O. Anatole von Lilienfeld. arXiv2007.09593v1,19July2020; journal DOI10.1088/2632-2153/abba6f, journal metadata not used here. [Primary source](https://arxiv.org/abs/2007.09593).
Read: author PDF: introduction, same-molecule versus composition learning, learning curves, §4.3.1 revised MD17 and sampling protocols. Local access: saved_pdf.

Source finding: The value of force labels depends on the learning task. Revised MD17 addresses numerical data quality; kernel learning curves separate molecule-specific learning from composition transfer.

Our inference/action: Distinguish data-support questions from iteration-budget questions and preserve dataset versions and units.

Limits: Kernel sample-count learning curves do not determine neural Adam update budgets. Random trajectory subsampling differs from whole-connection held-out pathways.

Saved version: [gradients2020.pdf](papers/gradients2020.pdf); SHA256 `83fb601349f18d80542188f4494ec39cd91c2903775d8a290c813950c1e96e21`.

### LIT008 — Machine learning of accurate energy-conserving molecular force fields

Stefan Chmiela ; Alexandre Tkatchenko ; Huziel E. Sauceda ; Igor Poltavsky ; Kristof T. Schütt ; Klaus-Robert Müller. Science Advances3(5):e1603015,5May2017, DOI10.1126/sciadv.1603015; user-provided publisher PDF (6 article pages + publisher cover); earlier author reading used arXiv1611.04678v4. [Primary source](https://www.science.org/doi/10.1126/sciadv.1603015).
Read: publisher PDF pp1,4: conservation versus reference accuracy/topology warning; conservative kernel, matched naive-force comparison and sample-count results. Local access: saved_pdf_user_provided.

Source finding: Gradient-domain force learning constructs conservative molecular force fields and an associated energy potential.

Our inference/action: Use as foundational conservative-model context. Force labels constrain energy derivatives, leaving an additive energy gauge to handle explicitly.

Limits: Kernel training and trajectory data differ from our neural128frame continuation. Force-only energy offset behavior is not by itself proof of a bad force field; our fixed raw joint gate still must be reported unchanged.

Saved version: [gdml2017-published.pdf](papers/gdml2017-published.pdf); SHA256 `1001f6e5eeb57533c2772d9318cf3e07d8fd9c67c524abe2af63bea9eee804b7`. Targeted local reading; previous web/version provenance remains in the registry and intake snapshots.

### LIT009 — Forces are not Enough: Benchmark and Critical Evaluation for Machine Learning Force Fields with Molecular Simulations

Xiang Fu ; Zhenghao Wu ; Wujie Wang ; Tian Xie ; Sinan Keten ; Rafael Gomez-Bombarelli ; Tommi Jaakkola. user-provided PDF labelled Published in TMLR (04/2023),31pages, OpenReviewA8pqQipwkt; PDF metadata creation7August2024 is not the publication date; earlier author reading used arXiv2210.07237v2,26August2023. [Primary source](https://openreview.net/forum?id=A8pqQipwkt).
Read: user-provided published-labelled PDF pp1,6–7: abstract and §5 simulation observables/stability Eq1–5, stable-trajectory conditioning; earlier web author §4–5 pp1–7 reading retained. Local access: saved_pdf_user_provided.

Source finding: Static force accuracy does not align reliably with simulation stability or the structural metrics used by this benchmark.

Our inference/action: Keep an application-specific outcome alongside force errors and retain terminated simulations or relaxations as failures.

Limits: Its MD benchmark is background for evaluation design, not an experimental result for our geometry task or a requirement to expand the project into NVE evaluation.

Saved version: [forces2023-user.pdf](papers/forces2023-user.pdf); SHA256 `bc3b3771d17a69e456daea2e4a5e407beff110d24e7420409659f1807e93c595`. Targeted local reading; previous web/version provenance remains in the registry and intake snapshots.

### LIT010 — Pushing the limits of unconstrained machine-learned interatomic potentials

Filippo Bigi ; Paolo Pegolo ; Arslan Mazitov ; Jonathan Schmidt ; Michele Ceriotti. arXiv2601.16195v3,4June2026 (PDF internally dated5June); preprint. [Primary source](https://arxiv.org/abs/2601.16195v3).
Read: current author PDF/HTML: scaled unconstrained models, static workflows, conservative corrections and phonon limitations. Local access: saved_pdf.

Source finding: Large-data unconstrained models can be useful in static workflows when sufficiently accurate and appropriately corrected; important task-dependent constraints remain.

Our inference/action: Include as counterevidence to blanket claims that direct forces necessarily fail. Ask when consistency helps at matched accuracy and cost.

Limits: Large pretrained material models differ from our small fitted molecular diagnostic. The paper neither resolves our learning gate nor eliminates the need for conservative controls. Cite five v3 authors rather than the earlier four-author version.

Saved version: [unconstrained2026.pdf](papers/unconstrained2026.pdf); SHA256 `a52b92a3988497252b7ce3e1bd200477781a09e69df3175ab7e793638c43e157`.

### LIT011 — Physics-Informed Weakly Supervised Learning For Interatomic Potentials

Makoto Takamoto ; Viktor Zaverkin ; Mathias Niepert. ICML2025, PMLR267:58282–58310; published PDF preferred to saved arXivv2,5May2025. [Primary source](https://proceedings.mlr.press/v267/takamoto25a.html).
Read: published PDF: §4.1–4.4, Eq2–5, perturbations and weak-supervision training. Local access: saved_pdf.

Source finding: PITC compares perturbed energy with force-based Taylor estimates. PISC imposes agreement of energy estimates along spatial paths. Both augment supervised training.

Our inference/action: Physics/consistency-assisted training is established prior work. Map our exact proposed loss against these equations and retain perturbation and supervision differences.

Limits: These Taylor/spatial weak-label losses are not identical to a direct-force versus autograd-force squared penalty. Neither their reported gains nor perturbation sizes transfer automatically to Landscape17.

Saved version: [takamoto2025-published.pdf](papers/takamoto2025-published.pdf); SHA256 `dce2edbec058c0dec1df9a744c33d8420d40bdcfdb858cad00ec361a379a8945`.

### LIT012 — When Forces Disagree: A Data-Free Fast Diagnostic from Internal Consistency of Direct-Force Neural Network Potentials

Anonymous. user-provided anonymous manuscript labelled Under review as a conference paper at ICLR2026; acceptance/current forum status unverified. [Primary source](https://openreview.net/forum?id=cQvBP4TZHe).
Read: user PDF: §3, §4.2–4.4, §5 limitations and appendix fine-tuning examples; main text PDFpp3–9. Local access: saved_pdf.

Source finding: Force Delta is the RMS disagreement of direct and energy-gradient forces. The manuscript relates it to errors/stability and fine-tunes direct plus conservative pathways on adversarially selected structures. It reports consensus failures and says disagreement is uncalibrated.

Our inference/action: Very close prior work: no claim to invent force disagreement or conservative-pathway fine-tuning. A possible distinct contribution is held-out whole-connection reference-minimum recovery at matched accuracy/cost; this remains untested.

Limits: Pretrained Orb/EquiformerV2 systems and MD drift differ from our controlled molecular comparison. §4.4 always also fine-tunes the direct pathway; do not paraphrase it as conservative-only training. Small disagreement cannot certify correctness.

Saved version: [when-forces-disagree-user.pdf](papers/when-forces-disagree-user.pdf); SHA256 `711000270c2975933fa452c3b011c8133c718758a0500ce3a3ecd9ea16aaefe3`.

## Decisions and open work

1. Keep the twice-reviewed4000additional-update continuation fixed. No paper prescribes that budget or certifies convergence; no literature-driven schedule/model change is silently added.
2. Before direct-force implementation, compare the exact planned consistency objective against PITC/PISC and the supplied force-disagreement fine-tuning protocol. A separate plan requires two reviews.
3. Predeclare paired relaxation success, endpoint matching, physical failure categories, matched force-error bands and train/inference cost. Report inconsistent and consistent models that succeed or fail.
4. Do not use agreement as a correctness certificate. Preserve reference error, consensus failures and whole-connection leakage controls.
5. Expand citation coverage and verify final publication/status metadata when writing the manuscript. Treat tentative contribution boundaries as hypotheses.

## Access follow-up — resolved

All four previously missing PDFs were provided by the user on 30 September 2026, matched by title/authors/DOI or forum identity, successfully parsed, visually checked on identification/relevant pages and copied byte-for-byte into `papers/`. All 12 registered papers now have a checksum-verified local PDF. Earlier failed-download attempts remain in the manifest and append-only evidence log; the exact prior reading map and metadata are preserved in [the intake record](intake/user-pdfs-2026-09-30/intake-manifest.json).

Landscape17 is now cited as its 2026 publisher edition, retaining the earlier 2025 preprint reading history. GDML and Zhao are publisher PDFs; Forces Are Not Enough is the user-provided 31-page PDF labelled published in TMLR04/2023. Its 2024 file-creation metadata is not treated as a new publication year or a version equivalence certificate. Separate supplementary files for the publisher articles were not part of this four-PDF request. No access item remains open for these four article PDFs.

## Execution-status update

The fixed continuation is now completed exactly once: all execution checks pass, all original learning gates fail. No paper-driven schedule/model/loss change was added. See [measured outcomes](../../convergence-budget-diagnostics.md) and [final independent review](../../../docs/reviews/2026-09-30-convergence-budget-final-review.md). The earlier prospective decision above records the pre-run commitment, not unfinished work. Next intervention and novelty assessment remain separate open work.


## Local-source method notes — LIT-LOCAL-003

- **GDML, publisher p1/p4:** Energy conservation by construction does not ensure reference-trajectory accuracy; the introduction already warns about spurious critical points. Its conservative-versus-naive-force kernel comparison is prior consistency evidence. Our inference: evaluate reference outcomes alongside mathematical consistency; this is not a new general claim or evidence of our own model's behavior.
- **Zhao2025, §2.3/p5–6/Fig3:** Direct/autograd/autograd* schemes differ in Hessian behavior and intended-TS success. DFT-level TS optimization improves the direct schemes, and direct forces can perform best in that corrected workflow. Our inference: report the whole workflow and correction cost, avoiding blanket direct-force failure. Reactive TS search is distinct from our conformational minimum recovery; no DFT-Hessian intervention is being added.
- **Forces Are Not Enough, §5/p6–7:** Observable errors are computed over stable trajectory portions; stability/failure is measured separately. Our inference: later evaluation should retain failure rates and denominators alongside any conditional quality statistic, avoiding a success-only summary. Its MD stability thresholds are task-specific and are not adopted as our relaxation criteria.

These notes refine literature interpretation only. Future implementation or experiment changes require a new plan and two independent reviews. Current failed learning gates and saved numerical results are unchanged.
