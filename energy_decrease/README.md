# Energy-decrease validation comparison

The comparison uses the original uncapped recovery search and one additional acceptance condition. The energy at a trial must be strictly lower than at the latest accepted geometry. The archived paired outcomes are in [the result record](../reports/evidence/2026-11-12/energy-decrease-recovery.json).

The Python sources implement the solver, paired runner, saved-output audit and model-free tests. Run the model-free tests from this checkout with the E3x project environment and `PYTHONPATH=.:src:scripts python -m pytest -q energy_decrease/test_relax.py`.

The learned runs require the authenticated validation data, original raw outcome records and admitted checkpoints. E3x uses the frozen Stage 2 source at `e709d0d4514f450300e1eb99155df04c455ee9ca`. MACE uses its admitted uncapped recovery driver at `0e1c5374bf50fa3399e4e1e6d9ee9be7d56c5cbb` in the original float64 CPU environment, with the Stage 2 solver and follow-up modules on the import path. The runner accepts the retained admission and historical-record digests as arguments. Each run writes to a new directory. Both 16-case replays must be verified before executing the energy-decrease conditions.
