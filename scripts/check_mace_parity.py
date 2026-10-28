"""Validate the fixed MACE inputs and run one admitted prediction comparison."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from ase.io import read

EXPECTED_RECORDS = 974
EXPECTED_ATOMS = 21
EXPECTED_CONNECTION_COUNTS = {"13": 165, "18": 217, "18_2": 227, "66": 199, "78": 166}
TOLERANCE = 1e-10


class InputValidationError(ValueError):
    pass


def _ids(values: Any, label: str) -> list[str]:
    result = [str(value) for value in np.asarray(values).tolist()]
    if not result or any(not value for value in result) or len(set(result)) != len(result):
        raise InputValidationError(f"{label} IDs must be unique and nonempty")
    return result


def _load_memberships(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("memberships", payload)
    if isinstance(rows, dict):
        result = {}
        for record_id, value in rows.items():
            if isinstance(value, dict) and "groups" in value:
                groups = value["groups"]
                if not isinstance(groups, list) or len(groups) != 1:
                    raise InputValidationError("each validation membership must have exactly one group")
                connection = groups[0]
            else:
                connection = value.get("connection", value.get("connection_id")) if isinstance(value, dict) else value
            result[str(record_id)] = str(connection)
        return result
    if isinstance(rows, list):
        return {str(row["record_id"]): str(row.get("connection", row.get("connection_id"))) for row in rows}
    raise InputValidationError("membership sidecar has an unsupported schema")


def validate_inputs(packet_root: Path, *, expected_records: int = EXPECTED_RECORDS) -> dict[str, Any]:
    root = Path(packet_root)
    prediction_path = root / "predictions/final-valid-predictions.npz"
    try:
        archive = np.load(prediction_path, allow_pickle=False)
        if set(archive.files) != {"record_ids", "energy", "forces"}:
            raise InputValidationError("saved predictions must contain exactly record_ids, energy and forces")
        record_ids = _ids(archive["record_ids"], "prediction")
        energy = archive["energy"]
        forces = archive["forces"]
    except ValueError as exc:
        if isinstance(exc, InputValidationError):
            raise
        raise InputValidationError("saved predictions refused (including object/pickle arrays)") from exc
    if len(record_ids) != expected_records:
        raise InputValidationError(f"expected {expected_records} prediction records")
    expected_force_shape = (expected_records, EXPECTED_ATOMS, 3)
    if energy.dtype != np.float64 or energy.shape != (expected_records,) or not np.isfinite(energy).all():
        raise InputValidationError("saved energy must be finite float64 with fixed shape")
    if forces.dtype != np.float64 or forces.shape != expected_force_shape or not np.isfinite(forces).all():
        raise InputValidationError("saved forces must be finite float64 with fixed shape")

    frames = read(root / "data/valid.extxyz", index=":")
    if len(frames) != expected_records:
        raise InputValidationError(f"expected {expected_records} extxyz frames")
    by_id: dict[str, dict[str, Any]] = {}
    species_order: tuple[str, ...] | None = None
    for atoms in frames:
        record_id = atoms.info.get("record_id")
        if not isinstance(record_id, str) or not record_id or record_id in by_id:
            raise InputValidationError("extxyz IDs must be unique and nonempty")
        positions = np.asarray(atoms.positions, dtype=np.float64)
        species = tuple(atoms.get_chemical_symbols())
        if len(species) != EXPECTED_ATOMS or positions.shape != (EXPECTED_ATOMS, 3):
            raise InputValidationError("each frame must contain 21 atoms")
        if not np.isfinite(positions).all() or bool(np.any(atoms.pbc)):
            raise InputValidationError("frames must be finite and nonperiodic")
        if species_order is None:
            species_order = species
        elif species != species_order:
            raise InputValidationError("atom-symbol order differs across frames")
        by_id[record_id] = {"positions": positions.copy(), "species": species,
                            "ref_energy": atoms.info.get("REF_energy"),
                            "ref_forces": np.asarray(atoms.arrays.get("REF_forces"), dtype=np.float64)}

    memberships = _load_memberships(root / "data/valid.memberships.json")
    required = set(record_ids)
    if set(by_id) != required or set(memberships) != required:
        raise InputValidationError("record ID sets differ across inputs")
    counts = {key: 0 for key in EXPECTED_CONNECTION_COUNTS}
    for value in memberships.values():
        if value not in counts:
            raise InputValidationError(f"unexpected validation connection: {value}")
        counts[value] += 1
    if expected_records == EXPECTED_RECORDS and counts != EXPECTED_CONNECTION_COUNTS:
        raise InputValidationError("validation connection counts differ from the fixed contract")
    order = np.array([record_ids.index(record_id) for record_id in record_ids])
    return {"record_ids": record_ids, "frames": by_id, "memberships": memberships,
            "species": species_order, "expected_energy": energy[order].copy(),
            "expected_forces": forces[order].copy(), "connection_counts": counts}


def compare_predictions(actual: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    actual_ids = _ids(actual["record_ids"], "actual")
    expected_ids = _ids(expected["record_ids"], "expected")
    if set(actual_ids) != set(expected_ids):
        raise InputValidationError("actual and expected record ID sets differ")
    index = {record_id: i for i, record_id in enumerate(actual_ids)}
    take = [index[record_id] for record_id in expected_ids]
    actual_energy = np.asarray(actual["energy"])[take]
    actual_forces = np.asarray(actual["forces"])[take]
    expected_energy = np.asarray(expected["energy"])
    expected_forces = np.asarray(expected["forces"])
    if actual_energy.dtype != np.float64 or actual_forces.dtype != np.float64:
        raise InputValidationError("actual predictions must be float64")
    if actual_energy.shape != expected_energy.shape or actual_forces.shape != expected_forces.shape:
        raise InputValidationError("prediction shapes differ")
    if not np.isfinite(actual_energy).all() or not np.isfinite(actual_forces).all():
        raise InputValidationError("actual predictions are nonfinite")
    energy_diff = np.abs(actual_energy - expected_energy)
    force_diff = np.abs(actual_forces - expected_forces)
    max_energy = float(np.max(energy_diff, initial=0.0))
    max_force = float(np.max(force_diff, initial=0.0))
    passed = max_energy <= TOLERANCE and max_force <= TOLERANCE
    return {"status": "PARITY_PASS" if passed else "PARITY_FAIL",
            "record_count": len(expected_ids), "force_component_count": int(force_diff.size),
            "max_abs_energy_eV": max_energy,
            "max_abs_force_eV_per_angstrom": max_force,
            "per_record_max_energy_eV": energy_diff.tolist(),
            "per_record_max_force_eV_per_angstrom": force_diff.reshape(len(expected_ids), -1).max(axis=1).tolist(),
            "rtol": 0.0, "atol": TOLERANCE}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def run_parity(packet_root: Path, attempt_root: Path) -> dict[str, Any]:
    from mace_loader import load_admitted_arm, verify_packet

    started = time.monotonic()
    attempt = Path(attempt_root)
    packet = verify_packet(Path(packet_root))
    inputs = validate_inputs(Path(packet_root))
    arm, admission = load_admitted_arm(Path(packet_root), attempt)
    energies: list[float] = []
    forces: list[np.ndarray] = []
    completed: list[str] = []
    status: dict[str, Any]
    try:
        for record_id in inputs["record_ids"]:
            frame = inputs["frames"][record_id]
            predicted_force, _ = arm["forces"](frame["positions"], frame["species"], record_id)
            predicted_energy, _ = arm["energy"](frame["positions"], frame["species"], record_id)
            forces.append(np.asarray(predicted_force, dtype=np.float64))
            energies.append(float(predicted_energy))
            completed.append(record_id)
            np.savez(attempt / "predictions-partial.npz", record_ids=np.array(completed),
                     energy=np.array(energies, dtype=np.float64), forces=np.array(forces, dtype=np.float64))
        actual = {"record_ids": completed, "energy": np.array(energies, dtype=np.float64),
                  "forces": np.array(forces, dtype=np.float64)}
        expected = {"record_ids": inputs["record_ids"], "energy": inputs["expected_energy"],
                    "forces": inputs["expected_forces"]}
        comparison = compare_predictions(actual, expected)
        np.savez(attempt / "predictions.npz", **actual)
        status = {**comparison, "force_requests": len(completed), "energy_requests": len(completed),
                  "construction_count": 1, "elapsed_seconds": time.monotonic() - started,
                  "packet": packet, "admission_receipt_status": admission["status"]}
    except Exception as exc:
        status = {"status": "INCOMPLETE", "error_type": type(exc).__name__, "error": str(exc),
                  "completed_records": len(completed), "record_ids": completed,
                  "force_requests": len(completed) + (1 if len(forces) > len(energies) else 0),
                  "energy_requests": len(energies), "elapsed_seconds": time.monotonic() - started}
    _atomic_json(attempt / "parity-status.json", status)
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
    args = parser.parse_args()
    result = run_parity(args.packet_root, args.attempt_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") in {"PARITY_PASS", "PARITY_FAIL"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
