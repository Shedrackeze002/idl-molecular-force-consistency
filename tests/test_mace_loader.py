import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import types
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
    monkeypatch.setattr(module, "_validate_import_origins", lambda live: None)
    return root


def attempt(tmp_path, packet_root=None):
    import mace_loader as module
    root = tmp_path / "attempt"
    root.mkdir()
    repo = Path(module.__file__).resolve().parent.parent
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                            text=True, capture_output=True).stdout.strip()
    files = {name: {"bytes": (repo / name).stat().st_size, "sha256": sha(repo / name),
                    "git_blob": subprocess.run(
                        ["git", "rev-parse", f"{commit}:{name}"], cwd=repo, check=True,
                        text=True, capture_output=True).stdout.strip()}
             for name in module.REQUIRED_IMPLEMENTATION_FILES}
    admission = root / "wsl-admission.json"
    frozen = {name: {"bytes": (repo / name).stat().st_size, "sha256": sha(repo / name),
                     "baseline_commit": module.SOURCE_BASELINE, "git_blob": blob}
              for name, blob in module.FROZEN_DEPENDENCY_BLOBS.items()}
    admission.write_text(json.dumps({"status": "APPROVED", "identities": {
        "implementation_commit": commit, "implementation_files": files,
        "frozen_dependencies": frozen}}), encoding="utf-8")
    packet_root = Path(packet_root or (tmp_path / "packet")).resolve()
    interpreter = str(Path(sys.executable).resolve())
    worker = str(Path(module.__file__).with_name("check_mace_parity.py").resolve())
    launch = {"status": "APPROVED", "species": ["H"] * 21,
              "implementation_commit": commit,
              "wsl_admission_receipt": admission.name,
              "wsl_admission_sha256": sha(admission),
              "python_executable": interpreter,
              "packet_root": str(packet_root), "attempt_root": str(root.resolve()),
              "command": [interpreter, worker, "--packet-root", str(packet_root),
                          "--attempt-root", str(root.resolve())],
              "resource_limits": module.RESOURCE_LIMITS,
              "source_baseline": module.SOURCE_BASELINE,
              "implementation_reviews": [
                  {"status": "RESOLVED", "implementation_commit": commit,
                   "review_id": "review-1", "reviewer": "reviewer-one",
                   "url": "https://example.test/review/1"},
                  {"status": "RESOLVED", "implementation_commit": commit,
                   "review_id": "review-2", "reviewer": "reviewer-two",
                   "url": "https://example.test/review/2"}]}
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
    with pytest.raises(module.AdmissionError, match="identity|inventor|version-map"):
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


def test_noncore_version_pin_without_declaration_is_refused():
    import mace_loader as module
    receipt = canonical_admission(module)
    receipt["installed_package_versions"]["approved-extra"] = "9"
    with pytest.raises(module.AdmissionError, match="declaration.*version-map"):
        module._approved_identity_inventory(receipt)


@pytest.mark.parametrize("relative", ["scripts/mace_reference_adapter.py", "evaluation/adapter.py"])
def test_frozen_dependency_drift_refused_before_constructor(tmp_path, monkeypatch, relative):
    import mace_loader as module
    root = packet(tmp_path, monkeypatch)
    target = attempt(tmp_path, root)
    target_path = Path(module.__file__).resolve().parent.parent.joinpath(relative).resolve()
    original = Path.read_bytes

    def changed(self):
        value = original(self)
        return value + b"synthetic raw edit" if self.resolve() == target_path else value

    monkeypatch.setattr(Path, "read_bytes", changed)
    calls = []
    with pytest.raises(module.AdmissionError, match="frozen bridge/evaluator"):
        module._load_admitted_arm(root, target, constructor=lambda path: calls.append(path),
                                  expected_species=("H",) * 21)
    assert calls == []


def test_implementation_drift_with_refreshed_receipt_is_refused_before_constructor(tmp_path, monkeypatch):
    import mace_loader as module
    root = packet(tmp_path, monkeypatch)
    target = attempt(tmp_path, root)
    relative = sorted(module.REQUIRED_IMPLEMENTATION_FILES)[0]
    admission_path = target / "wsl-admission.json"
    admission = json.loads(admission_path.read_text())
    admission["identities"]["implementation_files"][relative].update(
        bytes=1, sha256="0" * 64)
    admission_path.write_text(json.dumps(admission))
    launch_path = target / "launch-receipt.json"
    launch = json.loads(launch_path.read_text())
    launch["wsl_admission_sha256"] = sha(admission_path)
    launch_path.write_text(json.dumps(launch))
    target_path = Path(module.__file__).resolve().parent.parent.joinpath(relative).resolve()
    original = Path.read_bytes

    def changed(self):
        value = original(self)
        return value + b"synthetic raw edit" if self.resolve() == target_path else value

    monkeypatch.setattr(Path, "read_bytes", changed)
    calls = []
    with pytest.raises(module.AdmissionError, match="execution source"):
        module._load_admitted_arm(root, target, constructor=lambda path: calls.append(path),
                                  expected_species=("H",) * 21)
    assert calls == []


def test_raw_commit_blob_bypasses_misleading_clean_filter(tmp_path):
    import mace_loader as module
    repo = tmp_path / "filter-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Synthetic"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "synthetic@example.test"], cwd=repo, check=True)
    subprocess.run(["git", "config", "filter.pin.clean", "sed s/edited/original/g"],
                   cwd=repo, check=True)
    (repo / ".gitattributes").write_text("guarded.py filter=pin\n")
    guarded = repo / "guarded.py"
    guarded.write_text("value = 'original'\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "baseline"], cwd=repo, check=True)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, check=True,
                              text=True, capture_output=True).stdout.strip()
    guarded.write_text("value = 'edited'\n")
    filtered = subprocess.run(["git", "hash-object", "--path=guarded.py", "guarded.py"],
                              cwd=repo, check=True, text=True, capture_output=True).stdout.strip()
    assert filtered == module._git_blob_id(repo, "guarded.py", revision)
    assert guarded.read_bytes() != module._git_blob_bytes(repo, "guarded.py", revision)


def test_extra_live_distribution_refused_before_constructor(tmp_path, monkeypatch):
    import mace_loader as module
    actual_check = module._live_admission_check
    root = packet(tmp_path, monkeypatch)
    target = attempt(tmp_path, root)
    real_distributions = list(importlib.metadata.distributions())
    versions = {}
    for distribution in real_distributions:
        name = distribution.metadata.get("Name")
        if name:
            versions[name.lower().replace("_", "-")] = distribution.version
    approved = canonical_admission(module)
    approved["platform"] = {"python": platform.python_version(), "platform": platform.platform(),
                            "architecture": platform.machine(), "glibc": platform.libc_ver()[1]}
    approved["installed_packages"] = [f"{name}=={version}" for name, version in versions.items()]
    approved["installed_package_versions"] = versions
    admission_path = target / "wsl-admission.json"
    admission = json.loads(admission_path.read_text())
    admission.update(approved)
    admission_path.write_text(json.dumps(admission))
    launch_path = target / "launch-receipt.json"
    launch = json.loads(launch_path.read_text())
    launch["wsl_admission_sha256"] = sha(admission_path)
    launch_path.write_text(json.dumps(launch))

    class ExtraDistribution:
        metadata = {"Name": "synthetic-extra"}
        version = "1.0"

    monkeypatch.setattr(module, "_live_admission_check", actual_check)
    monkeypatch.setattr(module.importlib.metadata, "distributions",
                        lambda: [*real_distributions, ExtraDistribution()])
    calls = []
    with pytest.raises(module.AdmissionError, match="complete live package inventory"):
        module._load_admitted_arm(root, target, constructor=lambda path: calls.append(path),
                                  expected_species=("H",) * 21)
    assert calls == []


def test_shadow_mace_import_refused_before_constructor(tmp_path, monkeypatch):
    import mace_loader as module
    actual_validate = module._validate_import_origins
    root = packet(tmp_path, monkeypatch)
    target = attempt(tmp_path, root)
    launch = json.loads((target / "launch-receipt.json").read_text())
    admission = json.loads((target / "wsl-admission.json").read_text())
    live = {"distribution_roots": {
        "mace-torch": str(Path(importlib.metadata.distribution("mace-torch").locate_file("")).resolve()),
        "torch": str(Path(importlib.metadata.distribution("torch").locate_file("")).resolve()),
    }}
    shadow = tmp_path / "shadow"
    (shadow / "mace").mkdir(parents=True)
    (shadow / "mace/__init__.py").write_text("SHADOW = True\n")
    monkeypatch.syspath_prepend(str(shadow))
    monkeypatch.delitem(sys.modules, "mace", raising=False)
    monkeypatch.setattr(module, "_validate_import_origins", actual_validate)
    monkeypatch.setattr(module, "_approved_launch",
                        lambda attempt_root, packet_root: (launch, admission, live))
    calls = []
    with pytest.raises(module.AdmissionError, match="import origin differs"):
        module._load_admitted_arm(root, target, constructor=lambda path: calls.append(path),
                                  expected_species=("H",) * 21)
    assert calls == []


def test_mace_line_endings_preserve_existing_lfs_mappings():
    attributes = Path(__file__).resolve().parent.parent.joinpath(".gitattributes").read_text()
    assert "reports/width-rate-replacement-diagnostics.json filter=lfs diff=lfs merge=lfs -text" in attributes
    assert ("reports/execution/width-rate-replacement-2026-10-01/corrected-report/"
            "width-rate-replacement-diagnostics.json filter=lfs diff=lfs merge=lfs -text") in attributes


@pytest.mark.parametrize("package", ["mace", "mace.calculators"])
def test_same_root_package_redirect_refused_before_constructor(tmp_path, monkeypatch, package):
    import mace_loader as module
    actual_validate = module._validate_import_origins
    root = packet(tmp_path, monkeypatch)
    target = attempt(tmp_path, root)
    launch = json.loads((target / "launch-receipt.json").read_text())
    admission = json.loads((target / "wsl-admission.json").read_text())
    base = Path(importlib.metadata.distribution("mace-torch").locate_file("")).resolve()
    fake = types.ModuleType(package)
    fake.__file__ = str(base.joinpath(*package.split("."), "__init__.py"))
    fake.__path__ = [str(tmp_path / "external")]
    monkeypatch.setitem(sys.modules, package, fake)
    live = {"distribution_roots": {"mace-torch": str(base), "torch": str(base)}}
    monkeypatch.setattr(module, "_approved_launch", lambda *args: (launch, admission, live))
    monkeypatch.setattr(module, "_validate_import_origins", actual_validate)
    calls = []
    with pytest.raises(module.AdmissionError, match="package search path"):
        module._load_admitted_arm(root, target, constructor=lambda p: calls.append(p),
                                  expected_species=("H",) * 21)
    assert calls == []


@pytest.mark.parametrize("shadow", ["numpy.py", "ase/io/__init__.py"])
def test_worker_refuses_shadow_before_dependency_executes(tmp_path, shadow):
    marker = tmp_path / "executed"
    path = tmp_path / shadow
    path.parent.mkdir(parents=True, exist_ok=True)
    if shadow.startswith("ase/"):
        (tmp_path / "ase/__init__.py").write_text("")
    path.write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    worker = Path(__file__).resolve().parent.parent / "scripts/check_mace_parity.py"
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    result = subprocess.run([sys.executable, str(worker), "--help"], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "import origin differs" in result.stderr
    assert not marker.exists()


@pytest.mark.parametrize("duplicate", ["review_id", "reviewer", "url"])
def test_duplicate_review_identity_is_refused_before_constructor(tmp_path, monkeypatch, duplicate):
    import mace_loader as module
    root = packet(tmp_path, monkeypatch)
    target = attempt(tmp_path, root)
    launch_path = target / "launch-receipt.json"
    launch = json.loads(launch_path.read_text())
    launch["implementation_reviews"][1][duplicate] = launch["implementation_reviews"][0][duplicate]
    launch_path.write_text(json.dumps(launch))
    calls = []
    with pytest.raises(module.AdmissionError, match="not distinct"):
        module._load_admitted_arm(root, target, constructor=lambda path: calls.append(path),
                                  expected_species=("H",) * 21)
    assert calls == []


@pytest.mark.parametrize("field,value", [
    ("python_executable", "/missing/python"),
    ("command", ["wrong"]),
    ("attempt_root", "/wrong/output"),
    ("resource_limits", {"seconds": 1}),
    ("implementation_reviews", []),
])
def test_incomplete_launch_binding_refused_before_constructor(tmp_path, monkeypatch, field, value):
    import mace_loader as module
    root = packet(tmp_path, monkeypatch)
    target = attempt(tmp_path, root)
    launch_path = target / "launch-receipt.json"
    launch = json.loads(launch_path.read_text())
    launch[field] = value
    launch_path.write_text(json.dumps(launch))
    calls = []
    with pytest.raises(module.AdmissionError, match="launch"):
        module._load_admitted_arm(root, target, constructor=lambda path: calls.append(path),
                                  expected_species=("H",) * 21)
    assert calls == []


def test_verified_packet_identity_when_available():
    import mace_loader as module
    packet_root = os.environ.get("MACE_PACKET_ROOT")
    if not packet_root:
        pytest.skip("private packet intentionally unavailable")
    result = module.verify_packet(Path(packet_root))
    assert result["status"] == "PACKET_VERIFIED"
    assert result["member_count"] == 11
    assert result["model_sha256"] == module.MODEL_SHA256
