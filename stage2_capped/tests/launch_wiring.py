"""Run the production dependencies of the A0 and stage workers on the real accepted files, with an arm that returns the
saved forces (no model, no checkpoint is loaded). Used as a subprocess by `test_launch_entry.py`.

    python -m stage2_capped.tests.launch_wiring a0 VIOLATION_FOLDER
    python -m stage2_capped.tests.launch_wiring stage VIOLATION_FOLDER

The audit guard of the launch (deny by default over the whole data tree) is enabled before anything is read, so a read of
any file under `Group9/data/` other than the five allowed ones ends this process with exit code 97 before the file is
opened. The last line printed is one JSON object; the test reads it.
"""
import json
import sys

import numpy as np

from evaluation.adapter import callable_arm
from stage2_capped import admission, launch_a0, launch_checks, launch_stage, replay_evidence, stage0_compare


def species_of(partition):
    from ase.data import chemical_symbols
    return tuple(chemical_symbols[int(v)] for v in np.asarray(partition.atomic_numbers[0], dtype=np.int64))


def saved_forces_arm(partition, saved, name):
    """An arm whose forces are the saved forces of the record it is asked about (looked up by its positions)."""
    index = {partition.positions[i].tobytes(): i for i in range(len(partition.positions))}

    def forces(array):
        array = np.asarray(array, dtype=np.float64)
        if array.ndim == 2:
            return saved[index[array.tobytes()]]
        return np.stack([saved[index[row.tobytes()]] for row in array])
    arm = callable_arm(forces, species_of(partition), name)
    arm["checkpoint_identity"] = {"wiring": name}
    return arm


def a0(group9):
    deps = launch_a0.production_deps()
    items = deps.load_bindings(group9)
    data_directory = deps.data_directory(group9)
    partition = deps.load_partition(data_directory)
    item = items[0]
    saved = deps.saved_forces(item["score_directory"])
    arm = saved_forces_arm(partition, saved, item["id"])
    returned = deps.replay(arm, item["score_directory"], data_directory, launch_a0._split())
    evidence = replay_evidence.collect(arm, partition.positions, saved)
    with np.load(f"{item['score_directory']}/best-validation.npz", allow_pickle=False) as archive:
        raw_forces = archive["forces"].copy()
    return {"models": [i["id"] for i in items], "records": len(partition.record_ids),
            "first_five": [str(r) for r in list(partition.record_ids)[:5]], "groups": list(deps.groups),
            "replay": {k: returned[k] for k in ("status", "record_count", "single_record_ids", "rtol", "atol")},
            "saved_forces_are_the_forces_array": bool(np.array_equal(saved, raw_forces)),
            "evidence_batch4_bit_identical": evidence["batch4"]["bit_identical_records"],
            "evidence_batch1_bit_identical": evidence["batch1"]["bit_identical_records"]}


def stage(group9):
    deps = launch_stage.production_deps()
    hashes = stage0_compare.saved_file_hashes(group9)
    report = admission.admit_all(group9)
    inputs = deps.inputs_loader(report["launch_records_sha256"], group9)
    saved = deps.saved_loader(hashes, group9)
    bad = dict(hashes)
    key = sorted(bad)[0]
    bad[key] = ("0" if bad[key][0] != "0" else "1") + bad[key][1:]
    try:
        deps.saved_loader(bad, group9)
        refused = None
    except Exception as error:  # noqa: BLE001 - the type is what the test reports
        refused = type(error).__name__
    return {"starts": len(inputs.starts), "reference_ids": len(inputs.reference_ids),
            "saved_tables": [len(t) for t in saved], "changed_hash_refused_with": refused}


def main(argv):
    mode, violations = argv
    group9 = admission.group9_root()
    tree, allowed = launch_checks.audit_scope(group9)
    guard = launch_checks.AuditGuard(tree, allowed, violations, violation_name="AUDIT_VIOLATION-wiring").enable()
    out = {"a0": a0, "stage": stage}[mode](group9)
    guard.disable()
    out["audit"] = {k: v for k, v in guard.report().items() if k != "opened_data_files_elsewhere"}
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
