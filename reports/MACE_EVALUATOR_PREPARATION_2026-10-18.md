# MACE evaluator preparation — 18 October 2026

The shared calculator bridge is implemented and accepted after two independent
technical reviews. This is an engineering result using synthetic ASE
calculators. Real MACE checkpoint loading, parity and recovery are pending.

The bridge preserves the frozen evaluator interface, atom order, nonperiodic
Å coordinates, float64 forces in eV/Å, and scalar energy in the supplied
calculator gauge. It validates the entire batch before calculation and
preserves declared numerical refusals while propagating operational errors.
It introduces no force projection, scaling, retry, subprocess service or
change to the solver. Provenance tags are isolated copies and explicitly
unverified; the future checkpoint loader must bind actual model bytes.

Verification: the same 11 synthetic tests pass in both existing environments
without dependency changes. Focused regressions pass 113 tests and 19
subtests; an additional runner/outcome/reporter/E3x-adapter suite passes 132
with one expected skip. The first suite emits 44 NumPy deprecation warnings
from existing code. All 54 inventoried frozen scientific/evaluator/scorer
files remain byte-identical. Routine tests used an unavailable archive
sentinel, with no archive payload reads or trained-model calls.

The initial missing-module RED test is retained. Review then exposed ASE's
acceptance of dummy element X (Z=0); a constructor test reproduced it before
the bridge was fixed to require real elements. A reviewer command initially
omitted PYTHONPATH; its corrected invocation passed. No scientific outcome
was discarded or recomputed as a result of these engineering corrections.

Metadata-only verification checked the primary Run A final EMA model bytes
and the accepted score-report hash. No model was deserialized. The
[contract](../docs/mace-evaluator-contract.md) specifies exact identities,
units and deferred checks; the [verification receipt](evidence/2026-10-18/mace-bridge-preparation.json)
records source hashes and scope.

Idris retains the checkpoint loader, model/environment admission and
independent reproduction; Olufunke retains runner/reporter integration.
After the midterm draft, verify real-model parity before MACE's unchanged
procedure control on ten validation starts and six own-reference controls.
Only then decide whether to revise the solver. This preparation establishes
no MACE recovery benefit, hardware portability, throughput or causal model
comparison, and it does not authorize test scoring or new training.
