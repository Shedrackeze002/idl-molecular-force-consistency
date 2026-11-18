# MACE reproducibility contribution — report draft

Muhammed Idris (`imhd866`), 18 November 2026. Methods and implementation evidence
are ready for team review. Numerical reproduction has not been run or accepted.

## Report-ready methods

We use the previously trained MACE Run A model as an external molecular-force
reference. It predicts a scalar molecular energy and obtains forces from its
negative positional gradient. Its equivariant graph representation accounts for
how molecular geometry changes under rotations. The reference was trained and
originally scored by Shedrack Eze; this contribution implements the separate
Windows/WSL loading, admission and independent reproduction workflow. It does
not introduce another training run or transfer credit for the original model.

The primary artifact is the final EMA model at epoch index 2047 after 2,048
epochs. The loader authenticates the released packet manifest and all eleven
members, then checks the final model identity. An independently provisioned
external launcher checks the approved revision and raw source bytes before
project execution. The approved environment must match the complete installed
package inventory and pinned sources. The launcher executes a private verified
source copy; direct bootstrap dispatch is refused. These implementation controls
are submitted in [PR #18](https://github.com/Shedrackeze002/idl-molecular-force-consistency/pull/18)
and remain subject to review and external provisioning.

The prospective validation attempt uses exactly 974 structures, 21 atoms per
structure and 61,362 force components. Predictions, extxyz structures and
connection memberships are joined by exact record ID, preserving the original
atomic order and the distinct connection label `18_2`. The connection counts are
13:165, 18:217, 18_2:227, 66:199 and 78:166. No alignment, permutation, fitted
correction or energy re-centering is part of reproduction. Coordinates are in
angstrom, forces in eV/angstrom, and energies in residual molecular eV.
`REF_energy` is already residual; the historical absolute-energy offset must
not be applied again.

The proposed calculator recipe is CPU, one Torch thread, float64, disabled
cueq/oeq acceleration and no compilation. One construction and 974 force plus
974 energy accessor requests are allowed. Accessor requests are not necessarily
model-forward counts because an ASE calculator may cache or jointly calculate
properties. The prospective pass condition is a maximum absolute discrepancy
of at most 1e-10 for both energy and force components, with relative tolerance
zero. The supervisor bounds setup, admission, construction, prediction and
persistence to 1,800 seconds, with 256 MiB of attempt output and 16 MiB per log.
Errors, incomplete output and timeouts are retained rather than retried silently.

A separate saved-output auditor verifies the admission chain, counts, model
snapshot, prediction digests and per-record/global differences without loading
the model. It recomputes pooled force-component MAE, force-component RMSE,
equal-connection force MAE and residual molecular-energy MAE. Score reproduction
requires all four unrounded differences to be at most 1e-10 in their respective
units. Prediction parity and score reproduction are separate verdicts.

## Verified implementation evidence

The latest recorded nine-file WSL suite returned **253 passed, 3 expected skips,
45 warnings and 19 subtests**, exit 0. It tested implementation commit
`dd2f3637c440e7a38364bcd944091d6a8a9a534c`; publication head is
`4eaeb7b96f330ebdfaf44c78d8baea13b998b140`. The skips concern unavailable private
packet/archive inputs; the warnings are the existing ASE/NumPy deprecation.
[Captured evidence](evidence/2026-11-18/trusted-dispatch-review/review-regression-suite.json).
These synthetic/fake-constructor checks demonstrate exercised engineering
behavior, not numerical fidelity of the trained checkpoint.

The existing shared bridge was also checked separately for the delivery handoff:
11 analytic-calculator tests passed. Its author remains Shedrack. Its successful
test does not imply acceptance by Olufunke or a successful MACE inference run.
[Handoff check evidence](evidence/2026-11-18/idris-delivery/bridge-check.json).

## Numerical comparison — awaiting authorization and execution

| Quantity | Original final-validation value | Idris WSL result | Absolute difference |
| --- | ---: | --- | --- |
| Pooled force-component MAE, eV/angstrom | 0.00525983286381519 | Not measured | Not available |
| Force-component RMSE, eV/angstrom | 0.02641038976153195 | Not measured | Not available |
| Equal-connection force MAE, eV/angstrom | 0.005348956868537277 | Not measured | Not available |
| Residual molecular-energy MAE, eV/molecule | 0.0074618421825989 | Not measured | Not available |
| Maximum absolute prediction-energy difference, eV | Acceptance bound: 1e-10 | Not measured | Not available |
| Maximum absolute prediction-force difference, eV/angstrom | Acceptance bound: 1e-10 | Not measured | Not available |

The four reference values above are Shedrack's original results, not this
contribution's measurements. Source: `models.final.validation` in the
[public report](evidence/2026-10-15/mace-runA/report.json), canonical Git blob
SHA-256 `85f70bb8865c23df86fa94303471f77bd76b03083fb184f6dbf513874b22c998`.
This public-report identity is distinct from the original private report identity.

Prediction verdict: **NOT RUN — authorization pending**. Score verdict:
**NOT RUN — authorization pending**. No attempt was made, so this is neither a
failed-parity result nor an incomplete numerical attempt.

After the approved attempt, replace only the missing measurement cells using
authenticated saved outputs and the independent audit. Record the reviewed
implementation, approved WSL/policy/launch identities, exact counts, maxima,
elapsed/resource status, and both independent saved-result reviews. A failure
or timeout must remain visible and must not be converted into a successful result.

## Interpretation and ownership limits

Even accepted reproduction would establish fidelity of this adapter and scoring
recipe on fixed validation inputs. It would not establish test generalization,
new training results, molecular recovery, multi-seed robustness or GPU throughput.
The original model used one seed and validation influenced scheduling/checkpoint
selection. MACE differs from the E3x models in architecture and training exposure;
it is not a controlled causal comparison of force consistency.

Olufunke owns evaluator/runner integration and its acceptance. Aaron owns later
physical-consistency and matched-hardware cost analysis; no inference timing,
memory benchmark or compute cost is reported here. The next required decisions
and delivery records are in the [delivery tracker](../docs/team/IDRIS_MACE_DELIVERY.md)
and [integration handoff](../docs/team/IDRIS_MACE_INTEGRATION_HANDOFF.md).
