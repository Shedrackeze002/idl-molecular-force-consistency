# Landscape17 molecular force consistency

- **Current midterm:** [PDF](https://github.com/Shedrackeze002/idl-molecular-force-consistency/blob/midsemester-report-draft/reports/midterm/group9_Midterm_IDL.pdf) and [source with verification](https://github.com/Shedrackeze002/idl-molecular-force-consistency/tree/midsemester-report-draft/reports/midterm).
- **Research snapshot:** [code and evidence at 3fa5969](https://github.com/Shedrackeze002/idl-molecular-force-consistency/tree/3fa5969e5a0c2fdeaebc76a05bace20799715203).
- The status text below describes the 15 October checkpoint; current report work is on `midsemester-report-draft`.

Updated 15 October 2026. Group 9 studies whether energy–force consistency improves reference-shape recovery on unseen Landscape17 connections after accounting for force error and cost. The proposal is submitted; the main controlled comparison remains pending.

The aspirin data pipeline and evaluator are established. MACE Run A completed2,048 epochs and independent scoring verifies validation pooled force MAE5.26 meV/Å/RMSE26.41 meV/Å. Shared E3x energy/direct/consistency training paths are implemented. Both first 1,000-update energy/direct engineering fits completed technical controls, but neither improved selected force validation: both BEST checkpoints remain update 0. These negative findings are retained.

Start with [current status](docs/team/PROJECT_STATUS.md), [team workflow](TEAM.md), [handoff](docs/team/PROJECT_HANDOFF.md), [run commands](docs/run-baseline.md) and [reproduction/access](docs/reproduce-oct15-results.md). Results: [MACE](reports/mace-runA-fit.md), [E3x](reports/e3x-first-engineering-fits.md), [evidence/attempts](reports/evidence/2026-10-15/README.md), [full shared experiment index](reports/PAPER_EVIDENCE_LOG.md).

Aspirin split4,891 / 974 / 1,127 records across 26/5/6 connections is frozen. Paracetamol remains blocked at01_2/plus. No main test or trained shape-recovery result is claimed. Next all-arm timing/commonbudget before three scientific baseline configurations and the12-run controlled study.

Historical sources, failures and original provenance remain retained; compact published evidence normalizes machine paths with explicit original/published hashes. Model/data/runtime archives still need separately verified access. Existing historical reports use [Git LFS](docs/team/LFS_SETUP.md); this update adds no new LFS objects. [Contribution ledger](docs/team/CONTRIBUTIONS.md) records authorship and remaining work.
