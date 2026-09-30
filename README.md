# Consistent learned forces and molecular shape recovery

This repository is the working home for a **semester-long research project** in 11-785 Introduction to Deep Learning (Fall 2026). The proposal is its first milestone. The goal is to complete a rigorous empirical study and develop the findings into a publishable paper.

## Research question

Neural networks can estimate the forces that move a molecule's atoms. Directly predicted forces may be efficient, but they can disagree with the model's predicted energy. This project asks whether training those forces to agree more closely with energy helps a molecule reach the reference stable shape when the route to that shape was not seen during training.

The planned study uses one electronic-structure reference from [Landscape17](https://doi.org/10.6084/m9.figshare.29949230.v1). Both descent branches of each held-out transition-state connection will stay in one evaluation group. An energy-derived model, an ordinary direct-force model, and a consistency-trained direct-force model will share a controlled equivariant architecture; a same-split MACE model provides an established comparison. Evaluation will consider force error, recovery of verified reference shapes, physical consistency, training cost, and inference speed. Aspirin is the primary molecule, with separately trained paracetamol models as a second check.

## Project roadmap

1. **Foundation and proposal:** review the literature, define the claim boundary, inspect the data, and submit the course proposal. Proposal submitted.
2. **Data and baseline:** audit Landscape17, freeze whole-connection train/validation/test splits and exact reference-branch relaxation starts, and verify an energy-derived baseline.
3. **Model comparisons:** train the direct-force and consistency-trained variants under matched protocols, then add the same-split MACE reference.
4. **Evaluation and analysis:** run paired relaxations, physical-consistency checks, accuracy and compute comparisons, uncertainty analysis, and paracetamol replication.
5. **Reporting and paper:** prepare the midterm and final course reports, release reproducible code and results, and develop a manuscript from the completed evidence.

The plan may be refined as pilot measurements and mentor feedback reveal what is feasible. Current stage: audited data and an implemented energy-derived baseline. Both original tiny-subset learning attempts failed the predeclared force criterion. Reviewed follow-up diagnostics learned a deliberately representable teacher but also failed the real-label full-batch force criterion; the full pilot remains pending. The reviewed coverage comparison (original16, diverse16 and diverse128) completed all fixed budgets but also failed both force gates. These are engineering diagnostics, not measured support for the research hypothesis.

## Repository contents

- [Submitted proposal PDF](Group9.pdf) and [LaTeX source](proposal.tex)
- [Research field guide PDF](Research_Field_Guide.pdf) and [editable guide](FIELD_GUIDE.md)
- [Claim-by-claim evidence audit](CLAIM_AUDIT.md)
- [Landscape17 raw pathway sample](data/landscape17_sample/README.md), with source attribution and checksums
- `neurips_2019.sty`, the course-provided style used to compile the proposal

The Python package, 246 verification tests, frozen aspirin manifests and reviewed configurations are implemented. See [run instructions](docs/run-baseline.md), [baseline status](reports/baseline-pilot.md), [physics checks](reports/baseline-verification.md), [force-learning diagnosis](reports/force-learning-diagnostics.md), [coverage diagnostics](reports/data-coverage-diagnostics.md) and the append-only [paper evidence log](reports/PAPER_EVIDENCE_LOG.md). Aspirin preparation passes; paracetamol remains blocked by one unmatched branch. Every new implementation plan requires two independent reviews under [AGENTS.md](AGENTS.md). The small committed data sample is for exploration; the [full Landscape17 archive](https://ndownloader.figshare.com/files/57341180) is needed for the proposed experiments.

Latest completed milestone: the twice-reviewed [fixed continuation-budget diagnostic](reports/convergence-budget-diagnostics.md) continued replay, balanced and force-only from their actual saved Adam states at 2000 updates to 6000. All execution checks passed, but all original learning gates still fail; force error continued changing over the terminal window, so convergence is not established. The full pilot remains pending. [Independent final review](docs/reviews/2026-09-30-convergence-budget-final-review.md), [execution record](reports/convergence-budget-execution-record.md) and a selective [12-paper literature foundation](reports/literature/2026-09-30/READING_MAP.md) preserve methods, unsuccessful trials, prior-work boundaries and open questions.
