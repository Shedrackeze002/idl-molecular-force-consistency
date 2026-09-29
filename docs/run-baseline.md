# Running the Landscape17 baseline

Run commands from the repository root `Codex`. Proposal submission is complete. Aspirin passes preparation; paracetamol preparation is blocked by endpoint `01_2/plus`. Both original learning checks and the reviewed real-label full-batch control failed their unchanged force criteria. Read the current report before launching a pilot.

## CPU environment

Use Python3.12 and uv. The repository lock fixes the core environment; generated archives, arrays and checkpoints remain ignored.

```sh
uv sync --frozen --group dev
uv run --frozen pytest -q
uv run --frozen python -c 'from pathlib import Path; from force_consistency.data.source import fetch_archive; print(fetch_archive(Path("data/Landscape17.zip")))'
uv run --frozen force-consistency audit --archive data/Landscape17.zip
uv run --frozen force-consistency prepare --molecule aspirin --policy configs/preparation.json
```

Preparation reads the verified source archive directly, computes chemical graph mappings and duplicate screens, checks source-grounded units and gates every selected branch before exporting arrays. The archive checksum, frozen policy, source references, exact split IDs, memberships, physical units and canonical array digests enter the success manifest. Existing nonempty outputs are protected: choose another `--output` to rerun. Training rechecks manifests and array identities and loads only training, primary validation and separate minimum diagnostics. Test arrays remain inaccessible to fitting and selection.

The frozen 26/5/6 aspirin assignment has 4891 unique training records (including11 shared reference minima),974 primary validation records and1127 test records. All11 minima are seen in training, so validation-minimum diagnostics are empty; this is a held-out-path/TS split. Raw group count does not establish independence. Named endpoint pairs are retained as metadata; verified geometric classes determine minimum membership. See DATA-007/008 in `reports/PAPER_EVIDENCE_LOG.md`.

## Learning and pilot

```sh
uv run --frozen force-consistency train --config configs/aspirin-baseline-learning-check.json --backend cpu
uv run --frozen force-consistency summarize-run --run runs/aspirin-learning-check-seed0
```

The original config preserves the failed first attempt. Never overwrite its directory. A prospective initialization amendment has separate configuration and provenance; its status is recorded in `docs/superpowers/plans/2026-09-29-learning-gate-amendment.md`. The full pilot command is authorized only after the reviewed learning gate passes:

```sh
uv run --frozen force-consistency train --config configs/aspirin-baseline-pilot.json --backend cpu
```

Adam1e-3, batch4, seed0. Training selects a connection uniformly and then one of its eligible path/TS/minimum references uniformly; shared minima are stored once but may be sampled through several training groups. The learning check instead samples its frozen16 frames. Float64 energy residuals remain in eV, physical forces in eV/A. Training-only standard deviations scale losses, never derivatives. Validation is exhaustive at0/every100/final updates, including partial batches. Checkpoint selection averages force-component MAE equally across connection groups; exact ties keep the earlier checkpoint. Pooled MAE is also saved. Minimum diagnostics never select checkpoints.

`runs/<name>` retains config, code/lock/source/split/policy/unit identities, seed/device/precision, full primary validation IDs and history, losses, atomic last/best checkpoints, and a new-process prediction verification. An unavailable requested GPU fails explicitly. Nonfinite training stops with a failure summary. Compilation, warmup, synchronized steady updates, total validation and elapsed time are separate measurements. Device memory may be unavailable and is labelled accordingly. CPU elapsed time is not GPU-hours. CPU optimizer/sampling and validation-selection resume continuity are tested, with numeric tolerance1e-10; cross-platform/GPU bitwise continuity is unverified. Legacy checkpoints lacking historical selection controls remain readable for inference but cannot resume fitting. Learning-check resume requires an immutable original acceptance baseline and is disabled under the reviewed amendment.

## NVIDIA environment and portable launch

GPU portability remains unverified on CMU, Colab and Kaggle: no device, allocation, account or scheduler details have been supplied. Use a Linux NVIDIA allocation that meets the driver/device requirements in the [official JAX installation guide](https://docs.jax.dev/en/latest/installation.html). CUDA12 wheels support more older devices than CUDA13; the locked optional extra uses JAX0.9.2 CUDA12 and adds its platform-specific runtime dependencies without changing the core versions. The guide currently specifies Linux driver>=525 for CUDA12; verify against the actual allocation before installing. Do not disable JAX compatibility checks.

```sh
uv sync --frozen --group dev --extra gpu
uv run --frozen --extra gpu python -c 'import jax; print(jax.devices("gpu"))'
uv run --frozen --extra gpu pytest -q
uv run --frozen --extra gpu force-consistency train --config configs/aspirin-baseline-pilot.json --backend gpu
```

Use a fresh run directory in the copied JSON config, and carry the frozen source/manifest IDs. Keep Python3.12, the same lock and precision. Colab/Kaggle need a Python3.12 environment, archive access and persistent output storage; the launcher uses the package CLI, not notebook-specific state. CMU batch submission also needs the real partition/account/time/GPU settings, which are intentionally not invented. Record driver/CUDA/device metadata and rerun physical checks before treating a GPU result as verified. Later inference-speed comparisons must use the same recorded GPU setting; this CPU engineering check is not a scientific speed comparison.

The architecture is adapted from the [official E3x force-field example](https://e3x.readthedocs.io/stable/examples/md17_ethanol.html), with retained vector features and head-local scalar readout. Its MD17 splits and label conventions are not used here.

## Reviewed force-learning diagnostics (30 September)

The saved diagnostic run completed validly: its easy synthetic teacher passed at300updates, but original-label fullbatch16 fitting failed the force gate at2000updates. Read [the diagnostic report](../reports/force-learning-diagnostics.md) and [compact evidence](../reports/force-learning-diagnostics.json). The full pilot remains pending. Frozen-feature readout solutions diagnose two fixed bodies; they are not network capacity bounds or production checkpoints.

```sh
uv run --frozen force-consistency diagnose-learning --config configs/aspirin-force-diagnostics.json
```

This invocation is a reproducibility reference; the completed output directory is protected. A future trial requires a new prospective reviewed protocol and separate directory, rather than silently rerunning or tuning this config. The controller validates historical checkpoint/config/subset hashes and current prepared identity, opens train.npz only, records outcomes in the evidence log, and blocks real-label fitting if the teacher fails. Its `completed` status describes diagnostic execution, not learning acceptance. Control summaries contain synthetic-versus-DFT label type, own-initial MSE thresholds, full-subset histories, input identities, final checkpoints and new-process verification. Teacher energy offset is synthetic0, with original loss scales. It is not a DFT energy prediction.

The diagnostic figure can be regenerated from saved outputs with `python scripts/summarize_force_diagnostics.py` using Matplotlib (generator version recorded in the compact evidence). The original random-head run has no saved intermediate full-subset history, so only its endpoints are plotted. Existing minibatch losses are not substituted for missing full-subset metrics.
