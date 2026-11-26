# E3x timing and adequacy tools

These helpers use the unchanged shared trainer. The scientific matrix is A (width 32, learning rate 0.001, equal weights), B (64, 0.003, equal weights) and C (64, 0.003, preserved balanced weights). Candidate JSON files ending in `.template.json` are explanatory templates and cannot be passed to the trainer.

Run the explicitly named helper tests with the locked Python environment and `src`/`scripts` on `PYTHONPATH`:

```sh
python -m pytest tests/test_e3x_phase_tools.py
```

The probe runner requires a saved combined-review approval containing the exact source/helper/dependency and dataset pins. It runs six isolated cases and one synthetic I/O fixture, preserving worker logs and failure outcomes. It does not fit a scientific candidate. Every ordinary update measurement includes sampling, batching, finite checks and bookkeeping; compiled-step time is a separate diagnostic.

```sh
python scripts/benchmark_e3x_budget.py --data /absolute/prepared/aspirin --output /absolute/new/probes --approval /absolute/probe-approval.json
python scripts/estimate_e3x_budget.py /absolute/new/probes/result.json /absolute/new/budget.json --aux-seconds ACTUAL_AND_RESERVED_AUXILIARY_SECONDS
```

The budget is the largest feasible multiple of 500 between 50,000 and 200,000 updates under the adopted 24-hour phase/12-hour baseline window, each including 25% reserve. There is no automatic smaller budget if 50,000 is infeasible. It counts all twelve reserved scientific runs, repeated validation/history/checkpoint writes, scoring and auxiliary costs. Six cases represent the scalar-weight/strength variants; exact cost invariance is not established.

Filled C/B/A launch records need two reviewer diff checks against the measured rule before launch. The launcher verifies source/data/configuration and initial-state identities, refuses existing candidate outputs, attempts all three serially, saves consumed failures, and scores BEST in a separate process. An incomplete run or BEST at update zero cannot pass A1. No automatic retry or extra configuration is allowed.

The independent scorer saves all validation predictions and IDs, recomputes pooled and equal-connection metrics, and independently cross-checks zero-force error against update zero. A1 is evaluated once after all three consumed slots. Checkpoint selection uses validation force error; no test arrays or recovery results select a candidate. Original engineering fits remain historical negative evidence and are not repeated.

Source, helpers, dependency versions and CPU/thread settings are pinned for cost comparability. A different execution environment requires new reviewed cost evidence, not a GPU-name whitelist. No remote resources are allocated by these helpers.
