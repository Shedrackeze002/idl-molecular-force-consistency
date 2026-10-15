# MACE Run A: full-data reference result

Run A completed 2,048 epochs on 4,891 training structures. Independent CPU predictions reproduce the saved unrounded validation force metrics for both exported checkpoints; two independent result reviews approve the saved evidence. The predeclared reference rule identifies a **working MACE reference on this split**.

| Checkpoint | Validation force MAE, pooled (meV/Å) | Validation force MAE, equal connection (meV/Å) | Validation force RMSE (meV/Å) |
|---|---:|---:|---:|
| Final EMA, epoch 2047 — primary | 5.259833 | 5.348957 | 26.410390 |
| Selected by weighted validation loss, epoch 1987 — secondary | 5.242323 | 5.332516 | 26.071354 |

Validation contains 974 structures across five held-out connections. The zero-force equal-connection MAE is 62.504390 meV/Å. The final model's full-training force MAE is 0.575206 meV/Å and RMSE 0.861789 meV/Å. On the frozen diverse128 training subset, its force MSE / zero-force MSE fraction is 0.000049498918, below the declared 0.1 fitting threshold. Both validation checkpoint MAEs are below 0.0500 eV/Å and outside the inconclusive band; checkpoint choice does not change the reference reading.

The final model's validation residual-energy MAE is 7.461842 meV per molecule, and the training value 2.730468 meV per molecule. Energies use the fixed offset derived from training data for this molecular composition. This is distinct from per-atom RMSE in notebook logs.

Run B is already complete: 128 training frames, 2,048 epochs, final validation force RMSE 74.558482 meV/Å. Run A's corresponding value is 26.410390 meV/Å. This is a descriptive contrast: different update and exposure budgets prevent a causal conclusion about data size alone. Run A contains 2,002,944 optimizer/EMA updates and 10,014,720 committed training exposures; those counts exclude diagnostics and lost partial work.

Limits: one seed; validation influenced scheduling and selected-checkpoint choice. No test evaluation, relaxation/endpoint recovery, controlled E3x comparison, multi-seed robustness or complete billing history is established. The proposal's energy-derived/direct/consistency study remains necessary. Full-data E3x adequacy cannot be inferred from MACE's score.

Exact metrics, saved predictions, strict log checks and source identities are in [the result evidence](evidence/2026-10-15/README.md). Two independent scientific/engineering saved-result reviews accepted the result. Model checkpoints and original source-bound scoring package are retained separately; see [reproduction and access](../docs/reproduce-oct15-results.md). Scoring ran once on14October; result acceptance15October2026.
