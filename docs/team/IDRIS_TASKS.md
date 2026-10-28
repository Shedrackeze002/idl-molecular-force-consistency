# Muhammed Idris: reproduction and compute

**28 October access update:** [Download the verified MACE packet](https://github.com/Shedrackeze002/idl-molecular-force-consistency/releases/download/mace-idris-packet-v2-2026-10-28/mace-idris-handoff-2026-10-18-v2.tar.gz) and follow the [detailed handover](IDRIS_MACE_HANDOVER_2026-10-18.md). The [release](https://github.com/Shedrackeze002/idl-molecular-force-consistency/releases/tag/mace-idris-packet-v2-2026-10-28) includes checksums, current instructions and dataset attribution. Confirm download and verification; public availability alone does not establish receipt.

The [accepted 26 October specification](../research/MACE_LOADER_PARITY_SPEC_2026-10-26.md) governs the current attempt. Use the existing accepted macOS arm64/Python 3.12.13 environment; a second environment, another platform or dependency setup requires a separately reviewed amendment. Implementation reviews and bounded launch authorization precede real inference.

## Historical 15–18 October assignment context

The older second-environment, portability, deferred-parity and private-access directions below are superseded by the current specification and public release instructions. They remain as assignment history, rather than permission to start a different attempt.

Updated 15 October 2026. Shedrack completed the first MACE reference, including Run B and full-data Run A training/scoring; no duplicate exporter or diagnostic search is needed.

1. Verify exact artifact/source/storage access and independently reproduce saved MACE scores on a second environment. Start with accepted Run A; Run B remains a separate diagnostic.
2. Own the checkpoint/inference wrapper; agree atom order, Z,Å, (N,3) forces in eV/Å,optional energy and identities with Olufunke. She owns integration.
3. Coordinate actual hardware portability/throughput with Emmanuel; preserve setup/compile/warmup/inference distinctions.
4. Additional MACE seeds remain planned and need separate reviewed feasibility/launch records.
5. Write reproducibility/compute and MACE methods/results, distinguishing independently authored work from Shedrack's first reference.

Own authored commits and evidence are required. These are remaining assignments, not verified completion. [Status](PROJECT_STATUS.md) · [access/reproduction](../reproduce-oct15-results.md).

18 October clarification: Shedrack prepared a synthetic-tested shared ASE
calculator bridge. This does not implement or reassign Idris's MACE
checkpoint loader, model identity/environment admission or independent
reproduction. Olufunke retains runner/reporter integration. See the
[shared contract](../mace-evaluator-contract.md) for the final-primary model
identity and deferred parity checks. Real MACE inference/recovery follows
the midterm draft; no additional seeds or repeated training are needed for
this control.


The [detailed handover](IDRIS_MACE_HANDOVER_2026-10-18.md), updated 28 October, gives public release access, the accepted packet digest, current checkout instructions and the governing specification. It does not claim the loader or reproduction is complete. Please also review the MACE section and your contribution statement in the [midsemester draft](../../reports/midterm/README.md).
