# First shared-trainer E3x engineering fits

Both energy-derived and direct-force models completed the planned 1,000 updates on 15 October 2026. Fresh processes reproduced LAST and BEST checkpoint predictions. **Neither fit improved validation force MAE over its untrained zero-force starting point; both selected update 0 as BEST.** This establishes a functioning training/checkpoint path, not an adequate learned baseline. Two independent saved-result reviews accepted the execution and this negative validation finding.

| Model | Final equal-connection force MAE (meV/Å) | Selected update | Parameters | Total trainer time (s) | Mean synchronized update (ms) |
|---|---:|---:|---:|---:|---:|
| energy | 70.467673 | 0 | 4737 | 32.98 | 3.761 |
| direct | 66.222396 | 0 | 4769 | 23.56 | 1.721 |

The common initial and selected equal-connection validation force MAE is **62.504390 meV/Å**. Validation covers all 974 structures in five connections. Model selection used only equal-connection validation force MAE; a lower final direct error than final energy error does not establish superiority, matching or recovery performance. The selected states remain untrained.

Validation energy MAE did improve: from 201.950559 meV per molecule initially to 130.332573 for the energy-derived model and 106.343408 for the direct model. Energy improvement alongside worse force MAE is a useful observation to preserve. It does not establish the cause of the force failure or predict the outcome of the declared balanced-weight candidate.

Both used the same frozen trainer revision, seed 0, full 4,891-record training support, batch 4, Adam learning rate 0.001, width 16, degree 1, two interactions, eight radial bases, 5 Å cutoff, float64 and equal supervised weights. The direct model adds a polar force head and 32 parameters. Each sampled 4,000 training-frame exposures with replacement (about 0.818 effective passes), plus one discarded four-frame warmup update. These are short engineering runs outside the three scientific candidates and 12-run comparison. No training-support coverage claim follows from exposure counts.

Timing is specific to this local CPU and width-16 recipe. Total trainer time includes initialization, compilation, warmup, validation, training, checkpoint writes and fresh-process reloads. The uninstrumented remainder is labelled unattributed; it is not a separate measurement of checkpoint cost. No consistency or wider-model timing was inferred from these runs.

Next: measure energy/direct/positive-consistency costs at widths 32 and 64, then freeze the common update budget and complete resource/deadline/reserve ledger before scientific candidates A/B/C. Preserve the declared balanced-weight candidate C and failed small-data history. Do not extend these engineering fits or select a recipe from them. A1/A2, two-seed matching, relaxation and test evaluation remain pending.

Exact source/config/input hashes, loss and validation histories, checkpoint receipts and timing summaries are in [the result evidence](evidence/2026-10-15/README.md). Independent E3x arithmetic used saved group sums/counts and checkpoint controls; full per-record validation prediction arrays were not retained. This differs from MACE's independent per-record rescore.
