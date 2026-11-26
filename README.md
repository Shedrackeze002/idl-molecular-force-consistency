# Can Consistent Forces Recover Molecular Shapes?

Group 9, 11-785 Introduction to Deep Learning, Carnegie Mellon University: Shedrack Eze, Olufunke Abodunrin, Emmanuel Aaron and Muhammed Idris. Updated 26 November 2026 with the final results.

## Project overview

We study whether training a direct-force model to agree with its own energy gradient gives forces that lead a geometry search back to the correct stable shape of a molecule. We use aspirin (21 atoms) from Landscape17, where each connection joins a transition state to two minima through two downhill branches, and we split its 37 connections at random into 26 for training, 5 for validation and 6 for testing (4,891, 974 and 1,127 records) so that both branches of a connection always stay together. All E3x models share one backbone, and we compare an energy-derived model, a direct force head and consistency training at strengths 0.1, 1 and 10, with a larger MACE model as an energy-derived reference. A search counts as a recovery when its largest force component falls to 0.01 eV/Å or below and the final shape lies within 0.05 Å of the target minimum after symmetry-aware alignment.

## Results

Consistency training at strength 0.1 matched direct prediction on validation force error, at 38.0 against 38.2 meV/Å, and it cut the disagreement between the two force outputs by more than two orders of magnitude for about twice the training time. Strength 1 gave the lowest mean E3x force error at 34.7 meV/Å, while MACE reached 5.3 meV/Å.

Our original force-following search recovered none of the ten validation starts for any model we tested, including runs with per-atom movement caps and, for the energy-derived models, an energy-decrease check. We traced this to the search itself, since its step size kept doubling without an energy check and 213 of its 250 primary runs used the whole budget of 1,500 force requests. With the standard L-BFGS optimizer the same MACE model recovered 7 of 10 starts at a stopping level of 0.01 eV/Å and 9 of 10 at 0.001 eV/Å, and BFGS gave 8 and 9 of 10.

With L-BFGS and FIRE the energy-derived E3x model converged, but its minima sit 0.06 to 0.33 Å from the reference minima, so it recovered none of the targets under the 0.05 Å rule. None of the direct or consistency models converged, even when started at a reference minimum, and with L-BFGS 95 of their 96 runs broke the molecule apart. This means that agreement on the training data was not enough to make the direct forces usable for finding minima, so the clearest next step is to apply the consistency penalty on the structures a search actually visits.

## Limitations

These results come from two seeds per E3x setting, a single MACE seed and one molecule, so they are descriptive. The five validation connections served both for checkpoint selection and for the recovery tests, and all six target minima also appear in the training connections, so the recovery test measures how well a model reaches known destinations. The six test connections were never scored.

## Code and evidence

- Fixed snapshots of the [E3x results](https://github.com/Shedrackeze002/idl-molecular-force-consistency/tree/435656659557af5e10e5a80d5f4abe0c2c66919a/reports/evidence/2026-11-12), the [MACE results](https://github.com/Shedrackeze002/idl-molecular-force-consistency/tree/527e1ddb8ce8406eb93382aeb25110607d0546a9/reports/evidence/2026-11-11) and the [energy-check runs](https://github.com/Shedrackeze002/idl-molecular-force-consistency/blob/24d8dc9b9cc0838173099e2b4b91d51aca672e16/reports/evidence/2026-11-12/energy-decrease-recovery.json).
- The [research snapshot at 3fa5969](https://github.com/Shedrackeze002/idl-molecular-force-consistency/tree/3fa5969e5a0c2fdeaebc76a05bace20799715203) with the shared E3x training code and the recovery evaluator.
- [Baseline run commands](docs/run-baseline.md) and [reproduction and checkpoint access](docs/reproduce-oct15-results.md).
- The [midterm report](reports/midterm/group9_Midterm_IDL.pdf) with its [source and verification](reports/midterm/README.md), the [experiment index](reports/PAPER_EVIDENCE_LOG.md) and the [project status log](docs/team/PROJECT_STATUS.md).
- The L-BFGS, BFGS and FIRE comparison scripts and their outputs are included with the final report's source files.

Older reports are stored with [Git LFS](docs/team/LFS_SETUP.md), and model checkpoints and the original datasets are kept outside this repository.

## Team contributions

Shedrack Eze prepared the data, integrated the models and training code, ran the reported experiments including the optimizer comparison, and computed the reported disagreement and cost figures. Emmanuel Aaron contributed a [physical-consistency evaluator](https://github.com/Shedrackeze002/idl-molecular-force-consistency/pull/27) for closed-loop work with finite-difference force checks, inference timing and 13 tests. Olufunke Abodunrin authored the [recovery evaluator](https://github.com/Shedrackeze002/idl-molecular-force-consistency/pull/4) and submitted a [Stage 2 reporting tool](https://github.com/Shedrackeze002/idl-molecular-force-consistency/pull/20). Muhammed Idris submitted the [Windows/WSL loader and parity code](https://github.com/Shedrackeze002/idl-molecular-force-consistency/pull/18) and its [report and integration handoff](https://github.com/Shedrackeze002/idl-molecular-force-consistency/pull/26). The [contribution ledger](docs/team/CONTRIBUTIONS.md) keeps the full history.
