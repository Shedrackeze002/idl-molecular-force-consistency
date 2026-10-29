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
    admission = root / "admission-receipt.json"
    admission.write_text("{}")
    snapshot = root / "snapshot.model"
    snapshot.write_bytes(b"model")
    packet = {"manifest_sha256": "manifest", "model_sha256": digest(snapshot)}
    parity = {"status": status, "record_count": 1, "force_component_count": 63,
              "max_abs_energy_eV": 0.0, "max_abs_force_eV_per_angstrom": 0.0,
              "construction_count": 1, "force_requests": 1, "energy_requests": 1,
              "packet": packet, "admission_receipt_sha256": digest(admission),
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
