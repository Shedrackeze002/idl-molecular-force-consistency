# Muhammed Idris: reproduction and compute

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


The [18 October actionable handover](IDRIS_MACE_HANDOVER_2026-10-18.md) now gives checkout instructions, private packet access and digest, source/model identities, units/gauge, row mapping and completion evidence. It does not claim the loader or reproduction is complete. Please also review the MACE section and your contribution statement in the [midsemester draft](../../reports/midterm/README.md).
