import hashlib
import json
import os
from pathlib import Path

import pytest
from ase.calculators.calculator import Calculator


class FakeCalculator(Calculator):
    implemented_properties = ["energy", "forces"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def packet(tmp_path, monkeypatch):
    import mace_loader as module
    root = tmp_path / "packet"
    members = []
    for index in range(10):
        path = root / f"data/member-{index}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"member-{index}".encode())
        members.append({"path": str(path.relative_to(root)).replace("\\", "/"),
                        "bytes": path.stat().st_size, "sha256": sha(path)})
    model = root / module.MODEL_RELATIVE
    model.parent.mkdir(parents=True)
    model.write_bytes(b"synthetic-model")
    members.append({"path": str(module.MODEL_RELATIVE).replace("\\", "/"),
                    "bytes": model.stat().st_size, "sha256": sha(model)})
    manifest = root / module.MANIFEST_RELATIVE
    manifest.write_text(json.dumps({"files": members}), encoding="utf-8")
    monkeypatch.setattr(module, "MANIFEST_SHA256", sha(manifest))
    monkeypatch.setattr(module, "MODEL_SHA256", sha(model))
    monkeypatch.setattr(module, "MODEL_BYTES", model.stat().st_size)
    return root


def attempt(tmp_path):
    root = tmp_path / "attempt"
    root.mkdir()
    admission = root / "wsl-admission.json"
    admission.write_text(json.dumps({"status": "APPROVED"}), encoding="utf-8")
    launch = {"status": "APPROVED", "species": ["H", "H"],
              "wsl_admission_receipt": admission.name,
              "wsl_admission_sha256": sha(admission)}
    (root / "launch-receipt.json").write_text(json.dumps(launch), encoding="utf-8")
    return root


def test_rejects_tampered_manifest_before_constructor(tmp_path, monkeypatch):
    import mace_loader as module
    root = packet(tmp_path, monkeypatch)
    (root / "MANIFEST.json").write_text("{}", encoding="utf-8")
    calls = []
    with pytest.raises(module.AdmissionError, match="manifest SHA"):
        module._load_admitted_arm(root, attempt(tmp_path), constructor=lambda path: calls.append(path))
    assert calls == []


def test_rejects_symlink_or_wrong_model_before_constructor(tmp_path, monkeypatch):
    import mace_loader as module
    root = packet(tmp_path, monkeypatch)
    (root / module.MODEL_RELATIVE).write_bytes(b"tampered")
    calls = []
    with pytest.raises(module.AdmissionError):
        module._load_admitted_arm(root, attempt(tmp_path), constructor=lambda path: calls.append(path))
    assert calls == []


def test_environment_refusal_precedes_constructor(tmp_path, monkeypatch):
    import mace_loader as module
    root = packet(tmp_path, monkeypatch)
    target = attempt(tmp_path)
    receipt = target / "wsl-admission.json"
    receipt.write_text(json.dumps({"status": "PROPOSED"}), encoding="utf-8")
    launch = json.loads((target / "launch-receipt.json").read_text())
    launch["wsl_admission_sha256"] = sha(receipt)
    (target / "launch-receipt.json").write_text(json.dumps(launch))
    calls = []
    with pytest.raises(module.AdmissionError, match="not approved"):
        module._load_admitted_arm(root, target, constructor=lambda path: calls.append(path))
    assert calls == []


def test_constructs_from_verified_snapshot_path(tmp_path, monkeypatch):
    import mace_loader as module
    root = packet(tmp_path, monkeypatch)
    target = attempt(tmp_path)
    calls = []
    arm, receipt = module._load_admitted_arm(
        root, target, constructor=lambda path: calls.append(path) or FakeCalculator())
    assert calls == [(target / "private/final_epoch.model").resolve()]
    assert sha(calls[0]) == module.MODEL_SHA256
    assert receipt["constructor"]["default_dtype"] == "float64"
    assert arm["checkpoint_identity_verified"] is False


def test_verified_packet_identity_when_available():
    import mace_loader as module
    packet_root = os.environ.get("MACE_PACKET_ROOT")
    if not packet_root:
        pytest.skip("private packet intentionally unavailable")
    result = module.verify_packet(Path(packet_root))
    assert result["status"] == "PACKET_VERIFIED"
    assert result["member_count"] == 11
    assert result["model_sha256"] == module.MODEL_SHA256
