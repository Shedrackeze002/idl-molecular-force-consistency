# FIRE shape-recovery control: seven E3x checkpoints (26 Nov 2026)

This run repeats the L-BFGS control (`../e3x-lbfgs-2026-11-26/`) with `ase.optimize.FIRE` in place of `LBFGS`. FIRE runs at ASE defaults (dt 0.1, maxstep 0.2, dtmax 1.0, Nmin 5, finc 1.1, fdec 0.5, astart 0.1, fa 0.99) with `downhill_check` off, so the step never uses energy. FIRE was picked because it does not assume the forces come from an energy. Everything else is unchanged: the same eight models (MACE Run A first, then the seven E3x checkpoints), the same 16 starts, fmax 0.01 then the same optimizer continued to 0.001 within 1,500 force calls in total, the same non-finite stop, the team's matcher and the report's success rule, and the `geom.sym_rmsd` cross-check.

**Result: FIRE recovers 0/10 original starts for all seven E3x checkpoints, at both thresholds. No direct or consistency checkpoint converges from any known minimum.** energy-0 keeps 4/6 known minima at fmax 0.01 and 0/6 at 0.001. MACE Run A recovers 0/10 at 0.01 and 5/10 at 0.001.

| Checkpoint | Recovered at 0.01 | Recovered at 0.001 | Known minima kept (0.01 / 0.001) | Force calls (min-max) | Median own-minimum RMSD from DFT minimum, A (0.01 / 0.001) | Endpoint connected at 5 A, fraction of runs (0.01 / 0.001) |
|---|---|---|---|---|---|---|
| energy-0 | 0/10 | 0/10 | 4/6 / 0/6 | 26-1500 | 0.028 / 0.153 | 1.00 / 1.00 |
| direct-0 | 0/10 | 0/10 | 0/6 / 0/6 | 1500-1500 | n/a / n/a | 0.06 / 0.06 |
| consistency-0.1-0 | 0/10 | 0/10 | 0/6 / 0/6 | 1500-1500 | n/a / n/a | 0.00 / 0.00 |
| direct-1 | 0/10 | 0/10 | 0/6 / 0/6 | 1500-1500 | n/a / n/a | 0.00 / 0.00 |
| consistency-0.1-1 | 0/10 | 0/10 | 0/6 / 0/6 | 1316-1500 | n/a / n/a | 0.31 / 0.31 |
| consistency-1-0 | 0/10 | 0/10 | 0/6 / 0/6 | 1500-1500 | n/a / n/a | 0.19 / 0.19 |
| consistency-1-1 | 0/10 | 0/10 | 0/6 / 0/6 | 1500-1500 | n/a / n/a | 0.69 / 0.69 |
| MACE Run A (validation) | 0/10 | 5/10 | 6/6 / 6/6 | 1-1500 | 0.000 / 0.000 | 1.00 / 1.00 |

"n/a" means no known-minimum endpoint had the same bond graph as its DFT minimum, so no symmetry-aware RMSD exists.

## What the numbers mean

- **The direct and consistency checkpoints have no minimum near the DFT minima under either search.** All 96 runs, the 36 known-minimum controls included, end without converging. 95 use the full 1,500 calls; the remaining one stopped at the 15,000-step cap after 1,316 calls. None reaches the report's 0.01 eV/Å rule. At the end, the known-minimum endpoints sit 0.8 to 45 Å from their DFT start (fixed-atom-order RMSD; `diagnostics/control_drift.json`), and every one has a changed bond graph. consistency-1-1 stays the most intact: 4 of 6 controls connected, 1.3 to 4.1 Å from the DFT minimum, max force 1.05 eV/Å or more. Then come consistency-0.1-1 (2 of 6 connected, 0.8 to 6.6 Å) and consistency-1-0 (0 of 6 connected, 5.2 to 30 Å). The two direct models and consistency-0.1-0 end 4.8 to 45 Å away with nothing connected. Under L-BFGS the same controls ended 3.7 to 23 Å away.
- **FIRE accelerates on the direct models.** 30 of 32 direct-0 and direct-1 runs end at the maximum time step (dt 1.0), with only 10 to 108 velocity resets in 1,500 steps; consistency-0.1-0 is similar (15 of 16 at dt 1.0). Velocity and force stay aligned, so FIRE keeps speeding up, which is what a force field that keeps doing positive work along a path would produce. The seed-1 consistency checkpoints (consistency-0.1-1 and consistency-1-1) show the opposite: many resets (up to 1,076), dt collapsing to about 1e-10, and up to 13,685 steps that move the atoms too little to need a force call. That is FIRE stalling rather than descending.
- **energy-0: the 4/6 at 0.01 is an early stop.** FIRE keeps min1, min3, min6 and min7 at fmax 0.01 after 26 to 94 calls, at 0.009 to 0.032 Å, the same four the custom search kept. Continuing to 0.001 moves every control 0.08 to 0.25 Å from the DFT minimum (median 0.153 Å), so 0/6 are kept. Together with L-BFGS (median 0.31 Å at 0.001), this shows energy-0's own minima sit at least about 0.07 Å from the DFT minima. They match only if the search stops near the start. The closest original start is 13/minus at 0.069 Å.
- **MACE recovers less with FIRE than with L-BFGS** (L-BFGS: 7/10 and 9/10). This is FIRE being slower, not a loading problem. At fmax 0.01, FIRE stops earlier on the way in: 7 endpoints converge 0.05 to 0.27 Å from target, and 3 runs are still unconverged at 1,500 calls. At 0.001, 7 runs use the full budget and 4 of them end unconverged. One recovery is borderline: 18_2/minus matches at 0.0490 Å at the budget limit. MACE keeps 6/6 known minima, since its forces at the DFT minima already meet fmax 0.01.
- **Continuing FIRE can leave a converged point.** FIRE keeps its velocity between the two stages. For energy-0 78/minus and MACE 18/plus, the endpoint converged at 0.01 but ended unconverged at the budget during the 0.001 stage (max force 0.027 and 0.137 eV/Å).

## Change from the L-BFGS script (announced before the runs)

After many velocity resets FIRE's dt can become so small that a step changes the positions by less than ASE's 1e-15 Å tolerance. Such a step reuses the cached forces and needs no force call. The budget therefore stays 1,500 force calls, and total optimizer steps are capped at 15,000 per run as a safety stop. The run is driven in chunks that can never exceed the remaining calls. This mattered for 16 runs, all in consistency-0.1-1 and consistency-1-1. Only one hit the step cap: consistency-0.1-1, 78/minus, after 1,316 calls. Each row records `fire_velocity_resets` (counted as steps where dt fell, since only a reset lowers dt), `fire_final_dt`, `fire_steps_without_force_call` and `step_cap_reached`.

## Checks

- Parity was verified for every checkpoint before its runs: the E3x maximum force differences were 4.3e-14 to 5.7e-14 eV/Å on 974 records, and the MACE three-frame check was exact.
- No model energy reached FIRE. Energy is a constant 0.0 for every arm, `downhill_check` is off, and ASE's log requests were served from that cache.
- The team matcher and `geom.sym_rmsd` agree on the recovered flag for all 256 endpoints. There were no non-finite stops, and no run used more than 1,500 calls.
- Environment, loaders and bound trees are the same as the L-BFGS run (see `../e3x-lbfgs-2026-11-26/SUMMARY.md`). The reserved test partition was never read, nothing was installed, and no existing file was changed.

## Files

`e3x/` and `mace-runA-validation/` hold `rows.jsonl`, one folder per checkpoint with its `parity.json`, and `environment.json`. The per-start `NN.json` files written during the run repeat the rows in `rows.jsonl` and are not included in this copy. Also: `summary.json` (including reset statistics and connected counts), `TABLE.md`, the run logs, and `diagnostics/control_drift.{py,json}`. Script: `../e3x_fire.py` (`diff ../e3x_lbfgs.py ../e3x_fire.py` shows the optimizer swap, the reset counter, the stall-safe budget loop and the extra column).
