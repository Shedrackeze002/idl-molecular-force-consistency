# MACE Run B: aspirin fitting diagnostic

Original result verified 8 October 2026; status links updated 15 October.

The primary final-epoch EMA model passes the prespecified force-fitting gate: diverse128 force-MSE fraction **0.001514599**, below the strict **0.1** threshold. This is **99.8485%** reduction relative to the zero-force baseline.

Run B trained on 128 frozen examples. Both final epoch 2047 and validation-selected epoch 399 pass CPU float64 S1 against their matching full-precision GPU validation MAE/RMSE at relative tolerance 1e-6 and absolute tolerance zero. Exact selected-model weights match its retained checkpoint.

| Model | Evaluation subset | Records | Force MSE / zero-force MSE | Force RMSE (meV/Å) | Force MAE (meV/Å) | Residual energy MAE (eV) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| final | train_diverse128 | 128 | 0.001514599 | 4.618607 | 3.110217 | 0.069726528 |
| final | train_narrow16 | 16 | 0.001684644 | 4.131761 | 3.104594 | 0.078107083 |
| final | train | 4891 | 0.201958466 | 57.022446 | 28.601917 | 0.083052312 |
| final | validation | 974 | 0.255517049 | 74.558482 | 38.498391 | 0.097719111 |
| selected | train_diverse128 | 128 | 0.007658614 | 10.385737 | 6.921634 | 0.079859176 |
| selected | train_narrow16 | 16 | 0.011967127 | 11.012248 | 7.445092 | 0.048241472 |
| selected | train | 4891 | 0.180557368 | 53.916590 | 27.999068 | 0.093934464 |
| selected | validation | 974 | 0.240334447 | 72.309461 | 35.787598 | 0.114184952 |

Primary diverse128 force MSE is 2.1331533342034656e-05 eV²/Å²; denominator 0.014083948218070219 eV²/Å². Statistics pool all 8,064 force components; this is an MSE fraction, not an RMSE ratio. All 16 narrow anchors belong to diverse128. The 4,891-record partition is inference reporting, not the model training set.

Validation: **not the reference (model trained on 128 frames)**. Equal-connection force MAE is 0.037823398798184735 eV/Å for final EMA, and 0.035200500854205834 eV/Å for the selected model. Validation groups/counts remain 13:165, 18:217, 18_2:227, 66:199, 78:166.

Both models are preserved; final epoch remains primary despite better selected-model validation force error. Validation informed the scheduler and secondary selection. This is one seed and a force-fitting diagnostic. Energy MAE is reported separately; this result does not declare a joint energy/force pilot gate passed. No test data were read.

The earlier E3x diverse128 arms reported MSE fractions 0.978 (coverage) and 0.3247 (late-rate constant), on the same frozen IDs and denominator. Budgets, architecture, optimization and hardware differ: those arms used 2,000/10,000 updates; MACE used 51,200 optimizer/EMA updates and 256,000 batch-frame exposures, 2,048 epochs, float64, seed 0. These observations do not isolate the cause of earlier E3x failures or establish setup specificity. The frozen reference protocol and original source-bound reports are retained separately.

Training launch-to-completion check took 4.0578 hours; local CPU scoring took 913.112 seconds (15.219 minutes), with one thread and one-structure ASE predictions. No new training, GPU allocation or numerical retry occurred. All archive, source, input, prediction and report hashes are bound in the execution records.

Residual energies use the frozen training offset −17647.258992926312 eV and zero per-element E0s; absolute reconstruction is reported separately in JSON. Single composition prevents per-element offset identification.

Run A has now completed independent scoring. Its full-data result is reported separately; the support-size contrast is descriptive because update and exposure budgets differ. Actual total billing and historical server-release status are not established by this diagnostic.

Original scoring JSON, artifact inventories, source packages and independent saved-result reviews are retained separately. Historical failed audits and implementation checks remain preserved; artifact access must be verified before reproduction.

This earlier diagnostic remains a 128-frame result. Run A is now independently scored; see [Run A](mace-runA-fit.md). Original Run B checkpoint/scoring artifacts require separate access; this report does not supply them.
