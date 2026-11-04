import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest
from ase.calculators.calculator import Calculator


class FakeCalculator(Calculator):
    implemented_properties = ["energy", "forces"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_admission(module):
    packages = {name: "1" for name in module.CORE_PACKAGES}
    sources = []
    for name, (distribution, relative_path) in module.REQUIRED_SOURCES.items():
        sources.append({"name": name, "distribution": distribution,
                        "relative_path": relative_path,
                        "actual_sha256": "a" * 64, "original_sha256": "a" * 64,
                        "matches_original": True})
    return {"platform": {"python": "3.12.13", "platform": "Linux-test",
                         "architecture": "x86_64", "glibc": "2.35"},
            "installed_packages": [f"{name}=={version}" for name, version in packages.items()],
            "installed_package_versions": packages,
            "authenticated_source_comparison": sources}


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
    monkeypatch.setattr(module, "_live_admission_check", lambda admission: {"synthetic": True})
    return root


def attempt(tmp_path):
    import mace_loader as module
    root = tmp_path / "attempt"
    root.mkdir()
    repo = Path(module.__file__).resolve().parent.parent
    files = {name: {"bytes": (repo / name).stat().st_size, "sha256": sha(repo / name)}
             for name in module.REQUIRED_IMPLEMENTATION_FILES}
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                            text=True, capture_output=True).stdout.strip()
    admission = root / "wsl-admission.json"
    admission.write_text(json.dumps({"status": "APPROVED", "identities": {
        "implementation_commit": commit, "implementation_files": files}}), encoding="utf-8")
    launch = {"status": "APPROVED", "species": ["H"] * 21,
              "implementation_commit": commit,
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
        module._load_admitted_arm(root, attempt(tmp_path), constructor=lambda path: calls.append(path),
                                  expected_species=("H",) * 21)
    assert calls == []


def test_rejects_symlink_or_wrong_model_before_constructor(tmp_path, monkeypatch):
    import mace_loader as module
    root = packet(tmp_path, monkeypatch)
    (root / module.MODEL_RELATIVE).write_bytes(b"tampered")
    calls = []
    with pytest.raises(module.AdmissionError):
        module._load_admitted_arm(root, attempt(tmp_path), constructor=lambda path: calls.append(path),
                                  expected_species=("H",) * 21)
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
        module._load_admitted_arm(root, target, constructor=lambda path: calls.append(path),
                                  expected_species=("H",) * 21)
    assert calls == []


def test_constructs_from_verified_snapshot_path(tmp_path, monkeypatch):
    import mace_loader as module
    root = packet(tmp_path, monkeypatch)
    target = attempt(tmp_path)
    calls = []
    arm, receipt = module._load_admitted_arm(
        root, target, constructor=lambda path: calls.append(path) or FakeCalculator(),
        expected_species=("H",) * 21)
    assert calls == [(target / "private/final_epoch.model").resolve()]
    assert sha(calls[0]) == module.MODEL_SHA256
    assert receipt["constructor"]["default_dtype"] == "float64"
    assert arm["checkpoint_identity_verified"] is False


@pytest.mark.parametrize("species", [["Xx"] * 21, ["X"] * 21, ["H"] * 20,
                                     ["H"] * 20 + ["C"]])
def test_invalid_or_reordered_species_refused_before_constructor(tmp_path, monkeypatch, species):
    import mace_loader as module
    root = packet(tmp_path, monkeypatch)
    target = attempt(tmp_path)
    launch_path = target / "launch-receipt.json"
    launch = json.loads(launch_path.read_text())
    launch["species"] = species
    launch_path.write_text(json.dumps(launch))
    calls = []
    with pytest.raises(module.AdmissionError, match="species|element"):
        module._load_admitted_arm(root, target, constructor=lambda path: calls.append(path),
                                  expected_species=("H",) * 21)
    assert calls == []


def test_incomplete_live_identity_receipt_refused():
    import mace_loader as module
    with pytest.raises(module.AdmissionError, match="complete live identity"):
        module._live_admission_check({"status": "APPROVED"})


@pytest.mark.parametrize("damage", ["commit", "files"])
def test_incomplete_implementation_identity_refused_before_constructor(tmp_path, monkeypatch, damage):
    import mace_loader as module
    root = packet(tmp_path, monkeypatch)
    target = attempt(tmp_path)
    admission_path = target / "wsl-admission.json"
    admission = json.loads(admission_path.read_text())
    if damage == "commit":
        admission["identities"]["implementation_commit"] = "not-a-commit"
    else:
        admission["identities"]["implementation_files"] = {}
    admission_path.write_text(json.dumps(admission))
    launch_path = target / "launch-receipt.json"
    launch = json.loads(launch_path.read_text())
    launch["implementation_commit"] = admission["identities"]["implementation_commit"]
    launch["wsl_admission_sha256"] = sha(admission_path)
    launch_path.write_text(json.dumps(launch))
    calls = []
    with pytest.raises(module.AdmissionError, match="implementation|commit"):
        module._load_admitted_arm(root, target, constructor=lambda path: calls.append(path),
                                  expected_species=("H",) * 21)
    assert calls == []


@pytest.mark.parametrize("field,value", [
    ("installed_packages", []), ("authenticated_source_comparison", []),
    ("installed_package_versions", {}),
])
def test_empty_live_identity_inventories_refused(field, value):
    import mace_loader as module
    receipt = {"platform": {"python": "3", "platform": "Linux",
                            "architecture": "x86_64", "glibc": "2"},
               "installed_packages": ["ase==1"],
               "authenticated_source_comparison": [{}] * 8,
               "installed_package_versions": {"ase": "1"}}
    receipt[field] = value
    with pytest.raises(module.AdmissionError, match="identity|inventor"):
        module._live_admission_check(receipt)


@pytest.mark.parametrize("damage", ["duplicate-source", "substitute-source",
                                     "omitted-pin", "inconsistent-packages"])
def test_canonical_admission_inventory_refuses_substitution_or_contradiction(damage):
    import mace_loader as module
    receipt = canonical_admission(module)
    if damage == "duplicate-source":
        receipt["authenticated_source_comparison"][-1] = dict(
            receipt["authenticated_source_comparison"][0])
    elif damage == "substitute-source":
        receipt["authenticated_source_comparison"][0]["name"] = "unrelated.module"
    elif damage == "omitted-pin":
        receipt["installed_packages"] = receipt["installed_packages"][:-1]
    else:
        name = next(iter(module.CORE_PACKAGES))
        receipt["installed_package_versions"][name] = "2"
    with pytest.raises(module.AdmissionError, match="source|package|inventory"):
        module._approved_identity_inventory(receipt)


def test_verified_packet_identity_when_available():
    import mace_loader as module
    packet_root = os.environ.get("MACE_PACKET_ROOT")
    if not packet_root:
        pytest.skip("private packet intentionally unavailable")
    result = module.verify_packet(Path(packet_root))
    assert result["status"] == "PACKET_VERIFIED"
    assert result["member_count"] == 11
    assert result["model_sha256"] == module.MODEL_SHA256
