# Reproduce the representation closeout

This package audits cached validation predictions and metadata. It executes no learned model. The private checkpoints, histories, cached predictions and prepared validation arrays remain required; their relative paths and identities are in input-inventory.json. No original numerical payload is included here.

From a checkout containing the supplied source, use an environment with NumPy and msgpack:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/close_representation_study.py \
  --repository "$PWD" --retained-root /path/to/retained-study-artifacts \
  --inventory reports/representation-closeout-2026-10-26/input-inventory.json \
  --output /path/to/new-closeout-output
```

The output path must be new. Inputs are authenticated before decoding and rehashed afterward. Model payloads remain opaque; the original scoring module is hashed rather than imported.

Render PNG/PDF figures from the verified analysis JSON using Matplotlib and a writable cache:

```bash
MPLCONFIGDIR=/path/to/writable-cache python scripts/plot_representation_closeout.py \
  --analysis /path/to/new-closeout-output/analysis.json \
  --output /path/to/new-figure-directory
```

The figure directory must also be new. Run tests from the repository root with PYTHONPATH=.:src:scripts. The new test_representation_closeout.py provides independent arithmetic/refusal checks; original representation analysis/protocol tests provide synthetic regressions. The latter may import model frameworks for synthetic fixtures, while the production closeout auditor does not.

Preserve the report's fixed-recipe, two-seed, validation-selection and timing limitations. Reproduction of saved arithmetic does not establish model portability, inference speed or recovery.
