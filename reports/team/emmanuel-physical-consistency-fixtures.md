# Emmanuel physical-consistency analytic fixture result

Date: 26 November 2026. Status: implementation and synthetic controls completed; trained-model evaluation not started.

## Implemented

- A versioned JSON result schema with explicit units, numerical settings, work counts, unavailable values and structured failures.
- Central-difference comparison of supplied forces with `-grad(E)`.
- Trapezoidal closed-loop work for energy-bearing and force-only arms.
- Uniform per-segment loop refinement for quadrature-sensitivity controls while retaining the trapezoidal rule.
- A paired energy/force callable adapter for analytic controls and future checkpoint adapters.
- Warmup-separated, synchronization-aware inference timing.
- Explicit failure routing for nonfinite derived metrics and refusal of empty timing batches.
- A deterministic fixture runner and focused tests.

## Synthetic results

The harmonic conservative fixture produced maximum energy/force disagreement `1.0002e-13 eV/angstrom` and closed-loop work `0 eV`. The rotational nonconservative force-only fixture produced signed closed-loop work `4 eV`. Its energy-gradient metric is explicitly `unavailable`; no energy was invented. A nonlinear conservative control produced `0.5 eV` of coarse trapezoidal loop work and `0.0078125 eV` after eight uniform subdivisions per declared segment, demonstrating why quadrature refinement is required before interpreting nonzero sampled work. The saved machine-readable record is `reports/team/emmanuel-physical-consistency-fixtures.json`.

The timing record is a local smoke measurement only. It is not a model comparison and must not be reported as a GPU, MACE or E3x runtime result.

## Verification

Command:

```text
python -m pytest tests/test_shape_recovery_outcomes.py tests/test_shape_recovery_relax.py tests/test_shape_recovery_report.py tests/test_shape_recovery_runner.py tests/test_shape_recovery_starts.py tests/test_physical_consistency.py -q
```

Result: `203 passed, 9 skipped`. The skips require the separately retained Landscape17 archive. The focused physical-consistency file passes all 13 tests. The environment used Python 3.14, while the repository declares Python 3.12; the evaluator and adapter suite do not import the unavailable JAX training stack.

## Remaining production work

Before evaluating trained models, obtain the required two reviews, bind reviewed E3x/MACE checkpoint adapters, verify checkpoint identities, freeze validation geometries and loops, and run every arm with the same hardware, precision, batch size, warmup and synchronization. Connection-level uncertainty and plots require actual model results and remain pending. Main-test data stays closed until model selection and evaluation policy are frozen.
