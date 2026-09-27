# Consistent learned forces and molecular shape recovery

This repository is the working home for a **semester-long research project** in 11-785 Introduction to Deep Learning (Fall 2026). The proposal is its first milestone. The goal is to complete a rigorous empirical study and develop the findings into a publishable paper.

## Research question

Neural networks can estimate the forces that move a molecule's atoms. Directly predicted forces may be efficient, but they can disagree with the model's predicted energy. This project asks whether training those forces to agree more closely with energy helps a molecule reach the reference stable shape when the route to that shape was not seen during training.

The planned study uses one electronic-structure reference from [Landscape17](https://doi.org/10.6084/m9.figshare.29949230.v1). Both descent branches of each held-out transition-state connection will stay in one evaluation group. An energy-derived model, an ordinary direct-force model, and a consistency-trained direct-force model will share a controlled equivariant architecture; a same-split MACE model provides an established comparison. Evaluation will consider force error, recovery of verified reference shapes, physical consistency, training cost, and inference speed. Aspirin is the primary molecule, with separately trained paracetamol models as a second check.

## Project roadmap

1. **Foundation and proposal:** review the literature, define the claim boundary, inspect the data, and submit the course proposal. This is the current stage.
2. **Data and baseline:** audit Landscape17, freeze whole-connection train/validation/test splits and exact reference-branch relaxation starts, and verify an energy-derived baseline.
3. **Model comparisons:** train the direct-force and consistency-trained variants under matched protocols, then add the same-split MACE reference.
4. **Evaluation and analysis:** run paired relaxations, physical-consistency checks, accuracy and compute comparisons, uncertainty analysis, and paracetamol replication.
5. **Reporting and paper:** prepare the midterm and final course reports, release reproducible code and results, and develop a manuscript from the completed evidence.

The plan may be refined as pilot measurements and mentor feedback reveal what is feasible. No model has been trained yet; the repository contains no experimental results.

## Repository contents

- [Current proposal PDF](proposal.pdf) and [LaTeX source](proposal.tex)
- [Research field guide PDF](Research_Field_Guide.pdf) and [editable guide](FIELD_GUIDE.md)
- [Claim-by-claim evidence audit](CLAIM_AUDIT.md)
- [Landscape17 raw pathway sample](data/landscape17_sample/README.md), with source attribution and checksums
- `neurips_2019.sty`, the course-provided style used to compile the proposal

Training code, frozen split manifests, configuration files, evaluation scripts, and measured results will be added as those stages begin. The small committed data sample is for exploration; the [full Landscape17 archive](https://ndownloader.figshare.com/files/57341180) is needed for the proposed experiments.
