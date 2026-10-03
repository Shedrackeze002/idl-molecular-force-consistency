# Running the baseline

Use Python 3.12 and the committed environment lock. Run commands from the repository root. The delivered test set and historical reproduction dependencies are described in [the reproduction note](REPRODUCTION.md). Large reports require [Git LFS](team/LFS_SETUP.md).

```sh
uv sync --frozen --group dev
uv run --frozen pytest -q
uv run --frozen python -c 'from pathlib import Path; from force_consistency.data.source import fetch_archive; print(fetch_archive(Path("data/Landscape17.zip")))'
uv run --frozen force-consistency audit --archive data/Landscape17.zip
uv run --frozen force-consistency prepare --molecule aspirin --policy configs/preparation.json
```

Preparation reads the verified source archive, checks chemical graph and endpoint mappings, duplicate screens and source-grounded units, and exports arrays only after every selected branch passes. Nonempty outputs are protected. Choose another output directory for a repeat. Training rechecks manifest and array identities; test arrays remain inaccessible to fitting and selection.

Aspirin has 4,891 unique training records, 974 primary validation records and 1,127 test records across 26/5/6 whole connections. All 11 reference minima are training-seen, so this is a held-out-path/transition-state split and minimum diagnostics do not establish unseen-minimum generalization. Paracetamol remains blocked at `01_2/plus`.

## Training prerequisites

The original random-readout and zero-readout learning checks both failed. Later full-batch, coverage, loss-balance, continuation, width/rate, late-rate, capacity and offset-profile diagnostics also failed their applicable real-label joint learning gate. Read [the experiment index](../reports/PAPER_EVIDENCE_LOG.md) before planning a new run.

The CLI and frozen configurations provide historical reference behavior. Run and checkpoint directories are protected; do not overwrite them. Source-bound diagnostic configurations require separately retained original launch packages, and their guards fail closed in this export. A new trial needs a new plan, source identity, output directory and launch verification. The full pilot remains pending.

The energy-derived model uses float64 physical energy residuals in eV and forces in eV/Å. Training-only standard deviations scale losses, never physical derivatives. Validation is exhaustive at zero, each scheduled interval and the endpoint, including partial batches. Equal-connection force MAE selects checkpoints; pooled MAE is reported separately. Legacy checkpoints without selection controls remain inference-readable and cannot resume fitting.

## GPU verification still required

CMU, Colab and Kaggle are available options, but this project has not verified a GPU launch on them. The locked optional Linux CUDA 12 environment is installed with:

```sh
uv sync --frozen --group dev --extra gpu
uv run --frozen --extra gpu python -c 'import jax; print(jax.devices("gpu"))'
uv run --frozen --extra gpu pytest -q
```

Check the actual allocation against the [official JAX installation requirements](https://docs.jax.dev/en/latest/installation.html). Record Python, lock, driver, CUDA, device, precision and output-storage identity. Run physical and checkpoint checks before using GPU measurements as research evidence. Scheduler/account settings depend on the real CMU allocation. CPU timings do not imply GPU cost, and cross-platform bitwise resume continuity remains unverified.
