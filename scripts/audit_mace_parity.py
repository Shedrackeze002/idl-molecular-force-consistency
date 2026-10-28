"""Independent saved-output audit; deliberately imports neither Torch nor MACE."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from check_mace_parity import TOLERANCE, compare_predictions, validate_inputs

REFERENCE_SCORES = {
    "pooled_force_mae": 0.00525983286381519,
    "force_component_rmse": 0.02641038976153195,
    "equal_connection_force_mae": 0.005348956868537277,
    "residual_energy_mae": 0.0074618421825989,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_saved_parity(packet_root: Path, attempt_root: Path) -> dict:
    packet_root, attempt_root = Path(packet_root), Path(attempt_root)
    status = json.loads((attempt_root / "parity-status.json").read_text(encoding="utf-8"))
    if status.get("status") not in {"PARITY_PASS", "PARITY_FAIL"}:
        raise ValueError("partial or operational worker result cannot be audited as parity")
    inputs = validate_inputs(packet_root)
    with np.load(attempt_root / "predictions.npz", allow_pickle=False) as saved:
        if set(saved.files) != {"record_ids", "energy", "forces"}:
            raise ValueError("saved output fields differ from contract")
        actual = {key: saved[key].copy() for key in saved.files}
    expected = {"record_ids": inputs["record_ids"], "energy": inputs["expected_energy"],
                "forces": inputs["expected_forces"]}
    parity = compare_predictions(actual, expected)
    if parity["status"] != status["status"] or parity["record_count"] != status.get("record_count"):
        raise ValueError("worker parity claims do not match independently recomputed outputs")

    frame_ids = actual["record_ids"].astype(str).tolist()
    force_by_id = {record_id: actual["forces"][i] for i, record_id in enumerate(frame_ids)}
    energy_by_id = {record_id: actual["energy"][i] for i, record_id in enumerate(frame_ids)}
    force_errors, energy_errors = [], []
    connection_abs: dict[str, list[np.ndarray]] = {key: [] for key in inputs["connection_counts"]}
    for record_id in inputs["record_ids"]:
        frame = inputs["frames"][record_id]
        ref_forces = np.asarray(frame["ref_forces"], dtype=np.float64)
        ref_energy = np.float64(frame["ref_energy"])
        difference = np.asarray(force_by_id[record_id], dtype=np.float64) - ref_forces
        force_errors.append(difference.reshape(-1))
        connection_abs[inputs["memberships"][record_id]].append(np.abs(difference).reshape(-1))
        energy_errors.append(abs(float(energy_by_id[record_id]) - float(ref_energy)))
    pooled = np.concatenate(force_errors)
    scores = {
        "pooled_force_mae": float(np.mean(np.abs(pooled), dtype=np.float64)),
        "force_component_rmse": float(np.sqrt(np.mean(pooled * pooled, dtype=np.float64))),
        "equal_connection_force_mae": float(np.mean([
            np.mean(np.concatenate(connection_abs[key]), dtype=np.float64)
            for key in sorted(connection_abs)], dtype=np.float64)),
        "residual_energy_mae": float(np.mean(np.asarray(energy_errors, dtype=np.float64), dtype=np.float64)),
    }
    differences = {key: abs(scores[key] - REFERENCE_SCORES[key]) for key in scores}
    result = {"status": "AUDIT_COMPLETE", "parity": parity,
              "prediction_file_sha256": _sha256(attempt_root / "predictions.npz"),
              "scores": scores, "score_reference": REFERENCE_SCORES,
              "score_abs_differences": differences,
              "score_status": "SCORE_REPRODUCED" if all(value <= TOLERANCE for value in differences.values())
                              else "SCORE_NOT_REPRODUCED"}
    (attempt_root / "independent-audit.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--attempt-root", type=Path, required=True)
    args = parser.parse_args()
    result = audit_saved_parity(args.packet_root, args.attempt_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
