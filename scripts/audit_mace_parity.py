"""Independent saved-output audit; deliberately imports neither Torch nor MACE."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import stat
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


def _validated_admission_chain(
    attempt_root: Path, status: dict, packet: dict, species: tuple[str, ...], supervisor: dict
) -> dict:
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
    from mace_loader import (FROZEN_SOURCE_FILES, REQUIRED_IMPLEMENTATION_FILES,
                             _approved_identity_inventory, _hex_digest,
                             _validate_launch_binding, _validate_supervisor_binding)
    if (not isinstance(commit, str) or len(commit) != 40
            or any(character not in "0123456789abcdef" for character in commit.lower())
            or launch.get("implementation_commit") != commit
            or not isinstance(files, dict) or set(files) != REQUIRED_IMPLEMENTATION_FILES
            or any(not isinstance(receipt, dict)
                   or not isinstance(receipt.get("bytes"), int) or receipt["bytes"] < 0
                   or not _hex_digest(receipt.get("sha256")) for receipt in files.values())):
        raise ValueError("reviewed implementation identity is incomplete")
    try:
        approved = _approved_identity_inventory(wsl)
    except RuntimeError as exc:
        raise ValueError(f"saved WSL approval identity is invalid: {exc}") from exc
    live = admission.get("live_identity_comparison", {})
    if live.get("platform") != approved["platform"] or live.get("packages") != approved["packages"]:
        raise ValueError("recorded runtime/package comparison differs from approval")
    if live.get("framework_argv0") != approved["framework_argv0"]:
        raise ValueError("recorded framework argument comparison differs from approval")
    if live.get("frozen_source_files") != FROZEN_SOURCE_FILES:
        raise ValueError("recorded frozen sources differ from the accepted baseline")
    recorded_sources = live.get("sources")
    if not isinstance(recorded_sources, list) or len(recorded_sources) != len(approved["sources"]):
        raise ValueError("recorded source comparison is incomplete")
    by_name = {}
    for item in recorded_sources:
        if (not isinstance(item, dict) or item.get("name") in by_name
                or not isinstance(item.get("path"), str) or not item["path"]):
            raise ValueError("recorded source comparison is malformed")
        by_name[item["name"]] = item
    if set(by_name) != set(approved["sources"]):
        raise ValueError("recorded source identities differ from approval")
    for name, item in by_name.items():
        if item.get("sha256") != approved["sources"][name]["actual_sha256"]:
            raise ValueError("recorded source digest differs from approval")
    try:
        execution = _validate_launch_binding(launch, attempt_root)
        if live.get("launch_identity") != execution:
            raise ValueError("recorded launch identity differs from approval")
        _validate_supervisor_binding(supervisor, execution, _sha256(launch_path), running=False)
    except RuntimeError as exc:
        raise ValueError(f"saved launch binding is invalid: {exc}") from exc
    if execution["attempt_root"] != str(attempt_root.resolve()):
        raise ValueError("approved attempt root differs from the audited directory")
    if tuple(execution["species"]) != species:
        raise ValueError("approved species order differs from the validated inputs")
    constructor = admission.get("constructor")
    expected_constructor = {"device": "cpu", "torch_threads": 1,
                            "default_dtype": "float64", "enable_cueq": False,
                            "enable_oeq": False, "compile_mode": None}
    if constructor != expected_constructor:
        raise ValueError("recorded constructor differs from the approved CPU recipe")
    if admission.get("packet") != packet:
        raise ValueError("model admission packet identity differs from authenticated packet")
    snapshot = admission.get("model_snapshot", {})
    expected_snapshot = attempt_root / "private" / "final_epoch.model"
    if snapshot.get("snapshot_path") != str(expected_snapshot):
        raise ValueError("admission is not bound to the private constructor snapshot")
    if (snapshot.get("model_sha256") != packet["model_sha256"]
            or snapshot.get("model_bytes") != packet.get("model_bytes")
            or snapshot.get("epoch_index") != 2047
            or admission.get("post_construction_sha256") != packet["model_sha256"]):
        raise ValueError("constructor snapshot identity differs from authenticated packet")
    return admission


def _attempt_file_sizes(root: Path) -> dict[str, int]:
    files = {}
    for path in root.iterdir():
        entry = path.lstat()
        if path.name == "private" and stat.S_ISDIR(entry.st_mode):
            for member in path.iterdir():
                leaf = member.lstat()
                if not stat.S_ISREG(leaf.st_mode):
                    raise ValueError(f"private snapshot contains a symlink or nonregular member: {member.name}")
                files[f"private/{member.name}"] = leaf.st_size
        elif stat.S_ISREG(entry.st_mode):
            files[path.name] = entry.st_size
        else:
            raise ValueError(f"attempt contains a symlink or nonregular member: {path.name}")
    return files


def _verify_retained_execution(root: Path, sizes: dict[str, int], supervisor: dict,
                               supervisor_bytes: bytes, launch: dict) -> None:
    from mace_loader import LAUNCH_RESOURCES, _launch_approval_refs
    expected = {ref["path"] for ref in _launch_approval_refs(launch)} | {
        "launch-receipt.json", "supervisor-status.json", "stdout.log", "stderr.log",
        "admission-receipt.json", "parity-status.json", "predictions.npz",
        "predictions-partial.npz", "private/final_epoch.model",
    }
    execution_names = sizes.keys() - {"independent-audit.json"}
    if execution_names != expected:
        raise ValueError("saved attempt file inventory differs from the authorized outputs")
    for name, counter in (("stdout.log", "stdout_bytes"), ("stderr.log", "stderr_bytes")):
        if sizes[name] != supervisor[counter] or sizes[name] > LAUNCH_RESOURCES["log_limit_bytes"]:
            raise ValueError(f"actual {name} size differs from the saved count or exceeds the log limit")
    total = sum(sizes[name] for name in execution_names)
    if total != supervisor["output_bytes"] or total > LAUNCH_RESOURCES["output_limit_bytes"]:
        raise ValueError("actual output tree size differs from the saved count or exceeds the limit")

    observation = root.with_name(root.name + ".supervisor-observation.json")
    if (observation.is_symlink() or not observation.is_file()
            or observation.stat().st_size > LAUNCH_RESOURCES["log_limit_bytes"]):
        raise ValueError("missing or invalid supervisor publication observation")
    observed = json.loads(observation.read_bytes())
    if not isinstance(observed, dict):
        raise ValueError("invalid supervisor publication observation")
    publication = observed.pop("publication", None)
    if (observed != supervisor or not isinstance(publication, dict)
            or set(publication) != {"receipt_sha256", "elapsed_seconds"}
            or publication["receipt_sha256"] != hashlib.sha256(supervisor_bytes).hexdigest()
            or "interrupt_signal" in supervisor):
        raise ValueError("supervisor publication observation differs from the saved receipt")
    elapsed = publication["elapsed_seconds"]
    if (type(elapsed) not in (int, float) or not math.isfinite(elapsed)
            or not supervisor["elapsed_seconds"] <= elapsed <= LAUNCH_RESOURCES["timeout_seconds"]):
        raise ValueError("supervisor publication time exceeds the reviewed bound or contradicts the receipt")


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


def audit_saved_parity(packet_root: Path, attempt_root: Path, *, persist: bool = True) -> dict:
    packet_root, attempt_root = Path(packet_root), Path(attempt_root).resolve()
    sizes = _attempt_file_sizes(attempt_root)
    status = json.loads((attempt_root / "parity-status.json").read_text(encoding="utf-8"))
    if status.get("status") not in {"PARITY_PASS", "PARITY_FAIL"}:
        raise ValueError("partial or operational worker result cannot be audited as parity")
    supervisor_bytes = (attempt_root / "supervisor-status.json").read_bytes()
    supervisor = json.loads(supervisor_bytes)
    if supervisor.get("status") != "WORKER_COMPLETED" or supervisor.get("exit_code") != 0:
        raise ValueError("operational supervisor result is not a successful bounded attempt")
    elapsed = supervisor.get("elapsed_seconds")
    if (type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed < 0
            or elapsed > 1800):
        raise ValueError("supervisor elapsed time is invalid or exceeds the reviewed bound")
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
    admission = _validated_admission_chain(attempt_root, status, packet, tuple(inputs["species"]), supervisor)
    _verify_retained_execution(attempt_root, sizes, supervisor, supervisor_bytes, admission["launch_receipt"])
    snapshot = status.get("final_model_snapshot", {})
    snapshot_path = Path(snapshot.get("path", ""))
    if (snapshot.get("path") != str(attempt_root / "private" / "final_epoch.model")
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
    if persist:
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
