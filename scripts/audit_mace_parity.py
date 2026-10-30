"""Independent saved-output audit; deliberately imports neither Torch nor MACE."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from check_mace_parity import EXPECTED_ATOMS, TOLERANCE, validate_inputs

REFERENCE_SCORES = {
    "pooled_force_mae": 0.00525983286381519,
    "force_component_rmse": 0.02641038976153195,
    "equal_connection_force_mae": 0.005348956868537277,
    "residual_energy_mae": 0.0074618421825989,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validated_admission_chain(attempt_root: Path, status: dict, packet: dict) -> dict:
    admission_path = attempt_root / "admission-receipt.json"
    if not admission_path.is_file() or status.get("admission_receipt_sha256") != _sha256(admission_path):
        raise ValueError("worker admission receipt identity is missing or mismatched")
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    if admission.get("status") != "ADMITTED":
        raise ValueError("saved model admission is not ADMITTED")
    launch_path = attempt_root / "launch-receipt.json"
    if not launch_path.is_file():
        raise ValueError("approved launch receipt is missing")
    launch = json.loads(launch_path.read_text(encoding="utf-8"))
    if launch.get("status") not in {"APPROVED", "ACCEPTED"} or admission.get("launch_receipt") != launch:
        raise ValueError("saved launch approval is invalid or not bound to admission")
    admission_name = launch.get("wsl_admission_receipt")
    if not isinstance(admission_name, str) or Path(admission_name).name != admission_name:
        raise ValueError("launch does not name a local WSL admission receipt")
    wsl_path = attempt_root / admission_name
    if (not wsl_path.is_file() or launch.get("wsl_admission_sha256") != _sha256(wsl_path)):
        raise ValueError("launch is not bound to the saved WSL admission receipt")
    wsl = json.loads(wsl_path.read_text(encoding="utf-8"))
    if wsl.get("status") not in {"APPROVED", "ACCEPTED"} or admission.get("wsl_admission") != wsl:
        raise ValueError("saved WSL admission is invalid or not bound to model admission")
    identities = wsl.get("identities", {})
    commit = identities.get("implementation_commit")
    files = identities.get("implementation_files")
    from mace_loader import REQUIRED_IMPLEMENTATION_FILES
    if (not isinstance(commit, str) or len(commit) != 40 or launch.get("implementation_commit") != commit
            or not isinstance(files, dict) or set(files) != REQUIRED_IMPLEMENTATION_FILES):
        raise ValueError("reviewed implementation identity is incomplete")
    live = admission.get("live_identity_comparison", {})
    if (not isinstance(live.get("platform"), dict) or not live.get("packages")
            or len(live.get("sources", [])) != 8):
        raise ValueError("recorded runtime/source comparison is incomplete")
    snapshot = admission.get("model_snapshot", {})
    expected_snapshot = (attempt_root / "private" / "final_epoch.model").resolve()
    if Path(snapshot.get("snapshot_path", "")).resolve() != expected_snapshot:
        raise ValueError("admission is not bound to the private constructor snapshot")
    if (snapshot.get("sha256") != packet["model_sha256"]
            or admission.get("post_construction_sha256") != packet["model_sha256"]):
        raise ValueError("constructor snapshot identity differs from authenticated packet")
    return admission


def _independent_parity(actual: dict, expected: dict) -> dict:
    actual_ids = [str(value) for value in np.asarray(actual["record_ids"]).tolist()]
    expected_ids = [str(value) for value in np.asarray(expected["record_ids"]).tolist()]
    if (not actual_ids or len(set(actual_ids)) != len(actual_ids)
            or set(actual_ids) != set(expected_ids)):
        raise ValueError("saved record IDs differ from validated inputs")
    ae, af = np.asarray(actual["energy"]), np.asarray(actual["forces"])
    ee, ef = np.asarray(expected["energy"]), np.asarray(expected["forces"])
    if ae.shape != (len(actual_ids),) or af.shape != (len(actual_ids), EXPECTED_ATOMS, 3):
        raise ValueError("saved prediction arrays have invalid raw shapes")
    if ee.shape != (len(expected_ids),) or ef.shape != (len(expected_ids), EXPECTED_ATOMS, 3):
        raise ValueError("validated expected arrays have invalid raw shapes")
    if ae.dtype != np.float64 or af.dtype != np.float64 or not np.isfinite(ae).all() or not np.isfinite(af).all():
        raise ValueError("saved predictions must be finite float64")
    take = [{key: i for i, key in enumerate(actual_ids)}[key] for key in expected_ids]
    energy_diff = np.abs(ae[take] - ee)
    force_diff = np.abs(af[take] - ef)
    max_energy = float(np.max(energy_diff, initial=0.0))
    max_force = float(np.max(force_diff, initial=0.0))
    return {"status": "PARITY_PASS" if max_energy <= TOLERANCE and max_force <= TOLERANCE else "PARITY_FAIL",
            "record_count": len(expected_ids), "force_component_count": int(force_diff.size),
            "max_abs_energy_eV": max_energy, "max_abs_force_eV_per_angstrom": max_force,
            "per_record_max_energy_eV": energy_diff.tolist(),
            "per_record_max_force_eV_per_angstrom": force_diff.reshape(len(expected_ids), -1).max(axis=1).tolist(),
            "rtol": 0.0, "atol": TOLERANCE}


def audit_saved_parity(packet_root: Path, attempt_root: Path) -> dict:
    packet_root, attempt_root = Path(packet_root), Path(attempt_root)
    status = json.loads((attempt_root / "parity-status.json").read_text(encoding="utf-8"))
    if status.get("status") not in {"PARITY_PASS", "PARITY_FAIL"}:
        raise ValueError("partial or operational worker result cannot be audited as parity")
    supervisor = json.loads((attempt_root / "supervisor-status.json").read_text(encoding="utf-8"))
    if supervisor.get("status") != "WORKER_COMPLETED" or supervisor.get("exit_code") != 0:
        raise ValueError("operational supervisor result is not a successful bounded attempt")
    if float(supervisor.get("elapsed_seconds", float("inf"))) > float(supervisor.get("timeout_seconds", 1800)):
        raise ValueError("supervisor elapsed time exceeds the reviewed bound")
    from mace_loader import verify_packet
    packet = verify_packet(packet_root)
    inputs = validate_inputs(packet_root)
    expected_count = len(inputs["record_ids"])
    if (status.get("construction_count") != 1 or status.get("force_requests") != expected_count
            or status.get("energy_requests") != expected_count):
        raise ValueError("construction/accessor counts differ from the admitted contract")
    recorded_packet = status.get("packet", {})
    if (recorded_packet.get("manifest_sha256") != packet["manifest_sha256"]
            or recorded_packet.get("model_sha256") != packet["model_sha256"]):
        raise ValueError("worker packet identity differs from independently authenticated packet")
    _validated_admission_chain(attempt_root, status, packet)
    snapshot = status.get("final_model_snapshot", {})
    snapshot_path = Path(snapshot.get("path", ""))
    if (snapshot_path.resolve() != (attempt_root / "private" / "final_epoch.model").resolve()
            or not snapshot_path.is_file() or snapshot.get("sha256") != _sha256(snapshot_path)
            or snapshot.get("bytes") != snapshot_path.stat().st_size
            or snapshot.get("sha256") != packet["model_sha256"]):
        raise ValueError("final constructor-bound snapshot identity is invalid")
    with np.load(attempt_root / "predictions.npz", allow_pickle=False) as saved:
        if set(saved.files) != {"record_ids", "energy", "forces"}:
            raise ValueError("saved output fields differ from contract")
        actual = {key: saved[key].copy() for key in saved.files}
    prediction_sha256 = _sha256(attempt_root / "predictions.npz")
    if status.get("prediction_file_sha256") != prediction_sha256:
        raise ValueError("saved prediction bytes differ from the worker receipt")
    expected = {"record_ids": inputs["record_ids"], "energy": inputs["expected_energy"],
                "forces": inputs["expected_forces"]}
    parity = _independent_parity(actual, expected)
    numerical_claims = ["status", "record_count", "force_component_count",
                        "max_abs_energy_eV", "max_abs_force_eV_per_angstrom",
                        "per_record_max_energy_eV", "per_record_max_force_eV_per_angstrom"]
    if any(parity[key] != status.get(key) for key in numerical_claims):
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
              "prediction_file_sha256": prediction_sha256,
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
