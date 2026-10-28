import json

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


def write_attempt(root, status="PARITY_PASS"):
    root.mkdir()
    np.savez(root / "predictions.npz", record_ids=np.array(["a"]),
             energy=np.zeros(1, dtype=np.float64), forces=np.zeros((1, 21, 3), dtype=np.float64))
    (root / "parity-status.json").write_text(json.dumps({"status": status, "record_count": 1}))


def test_audit_recomputes_predictions_and_scores_without_model_import(tmp_path, monkeypatch):
    attempt = tmp_path / "attempt"
    write_attempt(attempt)
    monkeypatch.setattr(module, "validate_inputs", lambda root: fixture_inputs())
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
    write_attempt(attempt, "PARITY_FAIL")
    monkeypatch.setattr(module, "validate_inputs", lambda root: fixture_inputs())
    with pytest.raises(ValueError, match="claims"):
        module.audit_saved_parity(tmp_path, attempt)
