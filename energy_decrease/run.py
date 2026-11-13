"""Replay and compare two admitted energy-derived checkpoints on validation."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    temporary = path.with_suffix(".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    if path.exists():
        raise FileExistsError(path)
    temporary.replace(path)


def setup_e3x(args):
    from e3x_phase_common import SPLIT
    from run_validation_recovery import adapter_replay
    from stage2_capped import admission, driver
    from stage2_capped.launch_arms import real_arm_provider

    report = admission.admit_all(args.group9)
    accepted = read(args.admission)
    if not accepted["verdict"]["passed"] or report["identities"] != accepted["admission"]["identities"]:
        raise ValueError("E3x checkpoint identities differ from the accepted replay")
    inputs = driver.load_study_inputs(report["launch_records_sha256"], args.group9)
    items = admission.load_bindings(args.group9)
    item = next(row for row in items if row["id"] == "energy-0")
    directory = admission.data_directory(args.group9)
    arm = real_arm_provider([item], directory, admission.PinPolicy())("energy-0")
    parity = adapter_replay(arm, item["score_directory"], directory, SPLIT)
    return inputs, arm, parity


def setup_mace(args):
    from check_mace_parity import TOLERANCE, validate_inputs
    from mace_loader import MODEL_BYTES, MODEL_SHA256, _construct_calculator
    from run_mace_recovery_controls import preflight, verify_record

    record = verify_record(args.admission, args.admission_sha256)
    bridge, starts, refs, policy, binding, manifest, classification, procedure, seen = preflight(record)
    from stage2_capped.driver import StudyInputs

    snapshot = Path(record["parity_attempt"]) / "private" / "final_epoch.model"
    if snapshot.stat().st_size != MODEL_BYTES or sha(snapshot) != MODEL_SHA256:
        raise ValueError("MACE checkpoint differs from the authenticated snapshot")
    species = next(iter(starts.values()))["species"]
    identity = {"model_sha256": MODEL_SHA256, "model_bytes": MODEL_BYTES, "epoch_index": 2047}
    arm = bridge(_construct_calculator(snapshot), species, arm_id="mace-runA-final-ema",
                 checkpoint_identity=identity)
    saved = validate_inputs(Path(record["packet"]))
    maxima = [0.0, 0.0]
    selected = (0, len(saved["record_ids"]) // 2, len(saved["record_ids"]) - 1)
    for index in selected:
        key = saved["record_ids"][index]
        frame = saved["frames"][key]
        force, _ = arm["forces"](frame["positions"], frame["species"], key)
        energy, _ = arm["energy"](frame["positions"], frame["species"], key)
        maxima[0] = max(maxima[0], float(np.max(np.abs(force - saved["expected_forces"][index]))))
        maxima[1] = max(maxima[1], abs(energy - saved["expected_energy"][index]))
    if max(maxima) > TOLERANCE:
        raise ValueError(f"MACE three-frame prediction parity failed: {maxima}")
    parity = {"status": "verified", "record_count": len(selected),
              "maximum_absolute_force_difference": maxima[0],
              "maximum_absolute_energy_difference": maxima[1], "atol": TOLERANCE,
              "validation_binding": binding, "checkpoint_identity": identity}
    return StudyInputs(starts, refs, policy, manifest, classification, seen, procedure), arm, parity


def schedule(inputs):
    from evaluation.runner import _own_reference
    from run_mace_recovery_controls import BRANCHES, REFERENCE_IDS

    if set(inputs.starts) != BRANCHES or set(inputs.reference_ids) != set(REFERENCE_IDS):
        raise ValueError("Unexpected validation starts")
    for key, start in sorted(inputs.starts.items()):
        if start["frame_index"] != 0:
            raise ValueError(f"{key} is not the first stored frame")
        _, own = _own_reference(key, inputs.classification, inputs.refs)
        yield "original", key, own, start
    for key in REFERENCE_IDS:
        yield "reference", key, inputs.refs[key], None


def historical_rows(args):
    data = read(args.historical)
    expected = {}
    if args.model == "e3x":
        for row in data["result_rows"]:
            if row["checkpoint_id"] != "energy-0":
                continue
            kind, label = row["start_kind"], row["start_id"]
            with (args.historical_raw / row["trace_file"]).open("rb") as stream:
                if sha(args.historical_raw / row["trace_file"]) != row["trace_sha256"]:
                    raise ValueError("Saved E3x trace hash differs")
                trace = np.load(stream, allow_pickle=False)
            last = np.flatnonzero(trace["accepted"] == 1)
            final = trace["positions"][last[-1] if len(last) else 0]
            expected[kind, label] = dict(row, final_positions=final)
    else:
        for kind, count in (("original", 10), ("reference", 6)):
            for i in range(count):
                name = f"{kind}-{i:02d}.json"
                file = args.historical_raw / name
                if sha(file) != data["retained_artifact_sha256"][name]:
                    raise ValueError("Saved MACE endpoint hash differs")
                saved = read(file)
                if kind == "original":
                    expected[kind, saved["branch_key"]] = dict(
                        saved, outcome_category=saved["category"],
                        final_positions=saved["_audit_record"]["final_positions"])
                else:
                    score, result = saved["score"], saved["relaxation"]
                    expected[kind, Path(score["reference_id"]).stem.replace("coordsmin_", "min")] = dict(
                        score, outcome_category="matched_reference",
                        converged=result["converged"], budget_exhausted=result["budget_exhausted"],
                        final_positions=result["final_positions"])
    if len(expected) != 16:
        raise ValueError("Wrong number of historical controls")
    return expected


def run_case(model, mode, inputs, arm, task, output, index):
    from energy_decrease.relax import relax_energy
    from evaluation.outcomes import classify_outcome
    from evaluation.runner import _arm_force_supplier, _final_record
    from scipy.sparse.csgraph import connected_components
    from trace_validation_recovery import score_reference_control

    kind, key, own, start = task
    record_id = key if start is not None else key + "/reference-control"
    positions = start["positions"] if start is not None else own.positions
    force = _arm_force_supplier(arm, own.species, record_id)

    def energy(coordinates):
        value, info = arm["energy"](coordinates, species=own.species, record_id=record_id)
        if info["status"] != "ok":
            raise ValueError(f"Unexpected energy status for {record_id}")
        return value

    trace = []
    began = time.perf_counter()
    raw = relax_energy(positions, force, inputs.procedure, arm=arm, recorder=trace.append,
                       energy_fn=energy if mode == "energy" else None)
    elapsed = time.perf_counter() - began
    final_pos = raw["final_positions"]
    reason = raw["failure_reason"]
    if reason in ("invalid_energy", "nonfinite_energy"):
        category, rmsd = "energy_evaluation_failure", None
    elif start is None and model == "mace":
        scored = score_reference_control(raw, own, inputs.policy)
        category = ("matched_reference" if scored["operational_force_converged_own_match"]
                    else "unmatched_converged" if raw["converged"] else "nonconverged")
        rmsd = scored["own_alignment"]["rmsd_A"]
    else:
        final = (_final_record(key, start, final_pos) if start is not None
                 else replace(own, record_id=own.record_id + "/trace-final",
                              positions=final_pos, forces=np.zeros_like(own.forces)))
        scored = classify_outcome(
            final, own, inputs.refs, inputs.policy,
            budget_exhausted=raw["budget_exhausted"], converged=raw["converged"],
            forces_finite=raw["forces_finite"], stop_reason=reason,
            seen_minimum=inputs.seen[key] if start is not None else True)
        category, rmsd = scored["category"], scored["rmsd_A"]
    label = key if start is not None else Path(key).stem.replace("coordsmin_", "min")
    pairs = np.linalg.norm(final_pos[:, None] - final_pos[None, :], axis=-1)
    tracefile = output / f"{index:02d}.npz"
    with tracefile.open("xb") as stream:
        np.savez_compressed(
            stream, positions=np.stack([item["positions"] for item in trace]),
            accepted=np.array([item["accepted"] is True for item in trace]),
            max_force=np.array([np.nan if item["max_force_component"] is None
                                else item["max_force_component"] for item in trace]),
            force_stop=np.array([item["converged_here"] for item in trace]),
        )
    energy_record = raw.get("energy_record", {"requests": 0, "trials": []})
    row = {
        "start_id": label, "start_kind": kind, "outcome_category": category,
        "target_minimum": Path(own.record_id).stem.removeprefix("coordsmin_"),
        "recovered": category == "matched_reference", "stop_reason": reason,
        "rmsd_A": rmsd, "endpoint_connected": connected_components(
            pairs < 5.0, directed=False, return_labels=False) == 1,
        "final_positions": final_pos.tolist(), "elapsed_s": elapsed,
        "trace": tracefile.name, "trace_sha256": sha(tracefile),
        "energy": energy_record, "refusal_details": raw["refusal_details"],
        "final_max_force_component_eV_per_A": raw["final_max_force_component_eV_per_A"],
        **{key: raw[key] for key in ("budget_exhausted", "converged", "step_count",
                                     "force_evaluation_count")},
    }
    save(output / f"{index:02d}.json", row)
    return row


def compare_baseline(row, old):
    fields = ("outcome_category", "stop_reason", "budget_exhausted",
              "converged", "step_count", "force_evaluation_count")
    differences = [field for field in fields if row[field] != old[field]]
    maximum = float(np.max(np.abs(np.asarray(row["final_positions"]) - old["final_positions"])))
    if differences or maximum > 1e-8:
        raise ValueError(f"{row['start_id']}: replay fields differ {differences}, endpoint max {maximum} Å")
    return maximum


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("e3x", "mace"), required=True)
    parser.add_argument("--mode", choices=("replay", "energy"), required=True)
    for name in ("group9", "admission", "historical", "historical-raw", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--admission-sha256", required=True)
    parser.add_argument("--historical-sha256", required=True)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()

    args.output.mkdir()
    if sha(args.admission) != args.admission_sha256 or sha(args.historical) != args.historical_sha256:
        raise ValueError("Published historical record or admission digest differs")
    if args.mode == "energy":
        replay = read(args.replay)
        if (replay["model"], replay["mode"], replay["status"], len(replay["rows"])) != (
                args.model, "replay", "complete", 16):
            raise ValueError("Energy mode requires the matching completed replay")
        if (replay["historical_sha256"] != args.historical_sha256
                or replay["admission_sha256"] != args.admission_sha256
                or replay["relaxer_sha256"] != sha(Path(__file__).with_name("relax.py"))):
            raise ValueError("Energy mode inputs differ from its replay")
    expected = historical_rows(args)
    inputs, arm, parity = (setup_e3x(args) if args.model == "e3x" else setup_mace(args))
    save(args.output / "preflight.json", parity)
    print(f"{args.model} {args.mode}: checkpoint parity verified", flush=True)
    rows, endpoint_differences = [], []
    for index, task in enumerate(schedule(inputs)):
        row = run_case(args.model, args.mode, inputs, arm, task, args.output, index)
        if args.mode == "replay":
            endpoint_differences.append(compare_baseline(
                row, expected[row["start_kind"], row["start_id"]]))
        rows.append(row)
        print(f"{len(rows)}/16 {row['start_id']}: {row['outcome_category']} "
              f"({row['force_evaluation_count']} force, {row['energy']['requests']} energy)", flush=True)
    if len(rows) != 16:
        raise ValueError("Incomplete schedule")
    save(args.output / "summary.json", {
        "status": "complete", "mode": args.mode, "model": args.model, "rows": rows,
        "historical_sha256": args.historical_sha256, "admission_sha256": args.admission_sha256,
        "relaxer_sha256": sha(Path(__file__).with_name("relax.py")),
        "max_replay_endpoint_difference_A": max(endpoint_differences, default=None),
    })


if __name__ == "__main__":
    main()
