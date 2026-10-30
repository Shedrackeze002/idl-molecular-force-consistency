import json
import hashlib

import numpy as np
import pytest

import audit_mace_parity as module


def fixture_inputs():
    ids = ["a"]
    zeros = np.zeros((1, 21, 3), dtype=np.float64)
    return {"record_ids": ids, "expected_energy": np.zeros(1, dtype=np.float64),
            "expected_forces": zeros, "memberships": {"a": "13"},
            "connection_counts": {"13": 1},
            "frames": {"a": {"ref_energy": 0.0, "ref_forces": zeros[0]}}}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_attempt(root, status="PARITY_PASS"):
    root.mkdir()
    np.savez(root / "predictions.npz", record_ids=np.array(["a"]),
             energy=np.zeros(1, dtype=np.float64), forces=np.zeros((1, 21, 3), dtype=np.float64))
    private = root / "private"
    private.mkdir()
    snapshot = private / "final_epoch.model"
    snapshot.write_bytes(b"model")
    packet = {"manifest_sha256": "manifest", "model_sha256": digest(snapshot)}
    from mace_loader import REQUIRED_IMPLEMENTATION_FILES
    wsl = {"status": "APPROVED", "identities": {
        "implementation_commit": "a" * 40,
        "implementation_files": {name: {"sha256": "b" * 64}
                                 for name in REQUIRED_IMPLEMENTATION_FILES}}}
    wsl_path = root / "wsl.json"
    wsl_path.write_text(json.dumps(wsl))
    launch = {"status": "APPROVED", "implementation_commit": "a" * 40,
              "wsl_admission_receipt": "wsl.json", "wsl_admission_sha256": digest(wsl_path)}
    (root / "launch-receipt.json").write_text(json.dumps(launch))
    admission = root / "admission-receipt.json"
    admission.write_text(json.dumps({"status": "ADMITTED", "launch_receipt": launch,
        "wsl_admission": wsl, "live_identity_comparison": {
            "platform": {"python": "3"}, "packages": {"torch": "1"},
            "sources": [{"sha256": str(i)} for i in range(8)]},
        "model_snapshot": {"snapshot_path": str(snapshot), "sha256": digest(snapshot)},
        "post_construction_sha256": digest(snapshot)}))
    parity = {"status": status, "record_count": 1, "force_component_count": 63,
              "max_abs_energy_eV": 0.0, "max_abs_force_eV_per_angstrom": 0.0,
              "per_record_max_energy_eV": [0.0],
              "per_record_max_force_eV_per_angstrom": [0.0],
              "construction_count": 1, "force_requests": 1, "energy_requests": 1,
              "packet": packet, "admission_receipt_sha256": digest(admission),
              "prediction_file_sha256": digest(root / "predictions.npz"),
              "final_model_snapshot": {"path": str(snapshot), "bytes": snapshot.stat().st_size,
                                       "sha256": digest(snapshot)}}
    (root / "parity-status.json").write_text(json.dumps(parity))
    (root / "supervisor-status.json").write_text(json.dumps(
        {"status": "WORKER_COMPLETED", "exit_code": 0, "elapsed_seconds": 1,
         "timeout_seconds": 1800}))
    return packet


def test_audit_recomputes_predictions_and_scores_without_model_import(tmp_path, monkeypatch):
    attempt = tmp_path / "attempt"
    packet = write_attempt(attempt)
    monkeypatch.setattr(module, "validate_inputs", lambda root: fixture_inputs())
    monkeypatch.setattr(__import__("mace_loader"), "verify_packet", lambda root: packet)
    result = module.audit_saved_parity(tmp_path, attempt)
    assert result["parity"]["status"] == "PARITY_PASS"
    assert result["score_status"] == "SCORE_NOT_REPRODUCED"


def test_partial_worker_result_is_refused(tmp_path):
    attempt = tmp_path / "partial"
    write_attempt(attempt, "INCOMPLETE")
    with pytest.raises(ValueError, match="partial"):
        module.audit_saved_parity(tmp_path, attempt)


def test_invented_success_flag_is_refused(tmp_path, monkeypatch):
    attempt = tmp_path / "altered"
    packet = write_attempt(attempt, "PARITY_FAIL")
    monkeypatch.setattr(module, "validate_inputs", lambda root: fixture_inputs())
    monkeypatch.setattr(__import__("mace_loader"), "verify_packet", lambda root: packet)
    with pytest.raises(ValueError, match="claims"):
        module.audit_saved_parity(tmp_path, attempt)


def test_operationally_invalid_timeout_is_refused(tmp_path, monkeypatch):
    attempt = tmp_path / "timeout"
    packet = write_attempt(attempt)
    (attempt / "supervisor-status.json").write_text(json.dumps(
        {"status": "TIMEOUT", "exit_code": -9, "elapsed_seconds": 1801,
         "timeout_seconds": 1800}))
    monkeypatch.setattr(module, "validate_inputs", lambda root: fixture_inputs())
    monkeypatch.setattr(__import__("mace_loader"), "verify_packet", lambda root: packet)
    with pytest.raises(ValueError, match="operational supervisor"):
        module.audit_saved_parity(tmp_path, attempt)


def test_refused_admission_is_not_accepted_by_hash_only(tmp_path, monkeypatch):
    attempt = tmp_path / "refused"
    packet = write_attempt(attempt)
    admission = json.loads((attempt / "admission-receipt.json").read_text())
    admission["status"] = "REFUSED"
    (attempt / "admission-receipt.json").write_text(json.dumps(admission))
    parity = json.loads((attempt / "parity-status.json").read_text())
    parity["admission_receipt_sha256"] = digest(attempt / "admission-receipt.json")
    (attempt / "parity-status.json").write_text(json.dumps(parity))
    monkeypatch.setattr(module, "validate_inputs", lambda root: fixture_inputs())
    monkeypatch.setattr(__import__("mace_loader"), "verify_packet", lambda root: packet)
    with pytest.raises(ValueError, match="not ADMITTED"):
        module.audit_saved_parity(tmp_path, attempt)


def test_prediction_bytes_are_bound_to_worker_receipt(tmp_path, monkeypatch):
    attempt = tmp_path / "changed"
    packet = write_attempt(attempt)
    with np.load(attempt / "predictions.npz") as saved:
        np.savez(attempt / "predictions.npz", record_ids=saved["record_ids"],
                 energy=np.array([1e-12]), forces=saved["forces"])
    monkeypatch.setattr(module, "validate_inputs", lambda root: fixture_inputs())
    monkeypatch.setattr(__import__("mace_loader"), "verify_packet", lambda root: packet)
    with pytest.raises(ValueError, match="prediction bytes"):
        module.audit_saved_parity(tmp_path, attempt)


def test_per_record_claims_are_recomputed(tmp_path, monkeypatch):
    attempt = tmp_path / "per-record"
    packet = write_attempt(attempt)
    parity = json.loads((attempt / "parity-status.json").read_text())
    parity["per_record_max_force_eV_per_angstrom"] = [5e-11]
    (attempt / "parity-status.json").write_text(json.dumps(parity))
    monkeypatch.setattr(module, "validate_inputs", lambda root: fixture_inputs())
    monkeypatch.setattr(__import__("mace_loader"), "verify_packet", lambda root: packet)
    with pytest.raises(ValueError, match="claims"):
        module.audit_saved_parity(tmp_path, attempt)
