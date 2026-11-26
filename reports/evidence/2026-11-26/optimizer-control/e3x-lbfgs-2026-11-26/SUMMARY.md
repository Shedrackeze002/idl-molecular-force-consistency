# L-BFGS shape-recovery control: seven E3x checkpoints (26 Nov 2026)

Only the search changes: ASE L-BFGS (defaults, no line search) replaces the custom force-following search. Everything else stays as in the report: the same ten original validation starts (first stored frame of each branch), the six known-minimum controls, a 1,500 force-call budget, the team's checkpoint loaders with parity checks, the team's matcher (`classify_outcome`, endpoint tolerance 0.05 Å), and the report's success rule (max force component ≤ 0.01 eV/Å and a match to the branch target). Each run goes to fmax 0.01 eV/Å (ASE per-atom norm), then continues the same optimizer to fmax 0.001 within the same 1,500 calls.

**Result: L-BFGS recovers 0/10 original starts and keeps 0/6 known minima for all seven E3x checkpoints, at both thresholds.** MACE Run A, the validation run, reproduces the earlier MACE L-BFGS result in `../mace_lbfgs_bfgs_results.jsonl` exactly.

| Checkpoint | Recovered at 0.01 | Recovered at 0.001 | Known minima kept (0.01 / 0.001) | Force calls (min-max) | Median own-minimum RMSD from DFT minimum, Å (0.01 / 0.001) |
|---|---|---|---|---|---|
| energy-0 | 0/10 | 0/10 | 0/6 / 0/6 | 48-1500 | 0.184 / 0.311 |
| direct-0 | 0/10 | 0/10 | 0/6 / 0/6 | 1500-1500 | n/a (all 6 diverged) |
| consistency-0.1-0 | 0/10 | 0/10 | 0/6 / 0/6 | 1500-1500 | n/a (all 6 diverged) |
| direct-1 | 0/10 | 0/10 | 0/6 / 0/6 | 1500-1500 | n/a (all 6 diverged) |
| consistency-0.1-1 | 0/10 | 0/10 | 0/6 / 0/6 | 1500-1500 | n/a (all 6 diverged) |
| consistency-1-0 | 0/10 | 0/10 | 0/6 / 0/6 | 1500-1500 | n/a (all 6 diverged) |
| consistency-1-1 | 0/10 | 0/10 | 0/6 / 0/6 | 1500-1500 | n/a (all 6 diverged) |
| MACE Run A (validation) | 7/10 | 9/10 | 6/6 / 6/6 | 1-211 | 0.000 / 0.000 |

## What the numbers mean

- **energy-0 converges, but its own minima are not the DFT minima.** 15 of 16 runs reach fmax 0.001. Six original starts end in their target's basin, 0.068 to 0.44 Å from the target, so none is within 0.05 Å; the closest is 13/minus at 0.068 Å at fmax 0.001. At fmax 0.001, three starts (18/minus, 66/minus, 66/plus) end nearest a different minimum, 1.2 to 1.5 Å from their target. From the known minima the model's own minimum is 0.056 to 0.33 Å away at fmax 0.01. At fmax 0.001, min6 and min9 move on into other conformer basins (1.25 and 1.16 Å from their start). One run, 18_2/minus, diverged: the molecule broke apart and the force reached 6,953 eV/Å.
- **The custom search's known-minimum matches for energy-0 are early stops.** The saved combined-search endpoints for energy-0 (cap 0.05 with the energy-decrease check) matched min1, min3, min6 and min7. Each stopped with a max force of 0.0094 to 0.0100 eV/Å, right at the threshold. The L-BFGS endpoint at fmax 0.01 has a lower model energy in every one of those cases, by 2.5 to 7.7 meV. Across all 15 comparable starts the L-BFGS endpoint is lower in energy than the custom-search endpoint (`diagnostics/diag_energy.json`). Energy was read only for this check, after the runs had finished.
- **The six direct and consistency checkpoints diverge under L-BFGS.** All 96 runs use the full 1,500 calls. 95 end with the molecule broken apart, with max forces of 19 to 4,230 eV/Å, and the one endpoint that stays connected is not converged either. This includes the runs that start at the DFT minima; in the four traced controls the initial max force was only 0.047 to 0.071 eV/Å. A step-by-step trace (`diagnostics/trace_known_minima.log`) shows the same pattern each time: a slow drift, then forces that grow, then many curvature updates with negative y·s. That happened in 41 to 104 of 299 steps for direct-0 and consistency-1-0, against 0 for energy-0. L-BFGS assumes the forces come from an energy, and these force heads do not satisfy that. For these six checkpoints, the L-BFGS result shows that L-BFGS fails on non-conservative forces. It does not measure their shape accuracy.
- **MACE validation agrees with the earlier MACE L-BFGS run.** The team's matcher gives 7/10 at 0.01 and 9/10 at 0.001, with 13/minus the only miss at 0.001. The force-call counts match that run on all 20 endpoints, and the RMSDs agree to within 3.4e-6 Å (`mace-runA-validation/cloud-comparison.json`).

## Checks

- Parity was verified for every checkpoint before any run. For E3x it covered 974 validation records, with a maximum force difference of 4.3e-14 to 5.7e-14 eV/Å. E3x identities equal the accepted A0 replay (`A0_RESULT.json`, SHA-256 3a64d4f1…). For MACE it was the team's three-frame force and energy check against the authenticated snapshot, with a maximum difference of 0.
- No model energy reached the optimizer. The calculator returns 0.0 energy for every arm, line search is off, and ASE asked for energy only for its own log lines. Every such request was served from the cached 0.0; none reached the model.
- The team's matcher and `geom.sym_rmsd` agree on the recovered flag for all 256 endpoints (224 E3x and 32 MACE). The team matcher reports an RMSD only for a match; `sym_rmsd_to_target_A` gives the distance for every endpoint.
- Reserved test data was never read. The loaders read only train, validation and validation-minima files, plus the five validation connections and the minima from the archive. The MACE binding records `test_branch_members_read: 0`.
- No force evaluation returned a non-finite value, and no run exceeded 1,500 calls.

## Setup and problems

- **Environment.** The 19 Nov bound trees in `/private/tmp/g9-combined-bound` had been partly deleted by macOS temporary-file cleanup. They were rebuilt in a fresh temporary folder (`/private/tmp/group9-optimizer-run`) the same way they were first built: `git archive` of publisher commits e709d0d (E3x) and 0e1c537 (MACE), `git init`/`fetch --depth=1`/`update-ref`/`read-tree` so the source checks pass, and the combined-search implementation files copied in (their SHA-256 values match the audited `combined-summary-e3x.json`).
- **Run settings.** E3x used `Group9/.venv`, cwd `e3x`, `PYTHONPATH=e3x:e3x/src:e3x/scripts:mace/scripts`. MACE used `Group9/baselines/mace/.venv`, cwd `mace`, `PYTHONPATH=mace:mace/src:mace/scripts:e3x`. Both set `PYTHONDONTWRITEBYTECODE=1` and unset the thread variables. The exact paths are in each `environment.json`. Script SHA-256: c80bda26….
- **First MACE attempt failed.** The calculator imported `evaluation.adapter` before the MACE loader ran, and the loader refuses a module that is already loaded, so that import was moved to after the checkpoint loads. The failed attempt is kept in `failed-attempt-01-mace-preloaded-module/` and holds no rows.
- **Known-minimum scoring.** All known-minimum controls, MACE included, are scored with `classify_outcome`, the same path the combined search used for E3x. The report scored the MACE controls with `score_reference_control`.
- Nothing was installed, and no existing file was changed.

## Files

`e3x/` and `mace-runA-validation/` hold `rows.jsonl` (one row per endpoint, both thresholds, endpoint positions included), one folder per checkpoint with its `parity.json`, and `environment.json`. The per-start `NN.json` files written during the run repeat the rows in `rows.jsonl` and are not included in this copy. Also: `summary.json` (per checkpoint and threshold: counts, categories, force-call range, own-minimum RMSDs, wall time), `TABLE.md`, `mace-runA-validation/cloud-comparison.json`, the run logs `e3x.log` and `mace-runA-validation.log`, and `diagnostics/` (energy comparison and curvature trace, with their scripts).
