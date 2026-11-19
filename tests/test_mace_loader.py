import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest
from ase.calculators.calculator import Calculator
from mace_test_runtime import load_arm_with_fake_constructor


class FakeCalculator(Calculator):
    implemented_properties = ["energy", "forces"]


def bootstrap_probe(python, bootstrap, entry, *args):
    # Library-level probe only. Production dispatch belongs to the external
    # launcher, which verifies source bytes before invoking this API.
    dispatch = ("import importlib.util,sys,runpy; path=sys.argv.pop(1); "
                "spec=importlib.util.spec_from_file_location('probe',path); "
                "module=importlib.util.module_from_spec(spec); "
                "spec.loader.exec_module(module); module.configure_imports(); "
                "from pathlib import Path; entry=sys.argv[2]; "
                "target=Path(path).with_name({'worker':'check_mace_parity.py',"
                "'supervisor':'supervise_mace_parity.py','audit':'audit_mace_parity.py'}[entry]); "
                "sys.argv=[str(target),*sys.argv[3:]]; runpy.run_path(str(target),run_name='__main__')")
    return [str(python), "-I", "-S", "-c", dispatch, str(bootstrap), "--entry", entry, *args]


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
            "import_initializers": {name: "a" * 64 for name in module.IMPORT_INITIALIZERS},
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
    worker = str((tmp_path / "trusted-launcher.py").resolve())
    policy = str((tmp_path / "policy.json").resolve())
    launch = {"status": "APPROVED", "species": ["H"] * 21,
              "implementation_commit": commit,
              "wsl_admission_receipt": admission.name,
              "wsl_admission_sha256": sha(admission),
              "python_executable": interpreter,
              "trusted_launcher": worker, "trusted_policy": policy,
              "packet_root": str(packet_root), "attempt_root": str(root.resolve()),
              "command": [interpreter, "-I", "-S", worker, "--entry", "supervisor", "--policy", policy],
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
        load_arm_with_fake_constructor(root, attempt(tmp_path), constructor=lambda path: calls.append(path),
                                  expected_species=("H",) * 21)
    assert calls == []


def test_rejects_symlink_or_wrong_model_before_constructor(tmp_path, monkeypatch):
    import mace_loader as module
    root = packet(tmp_path, monkeypatch)
    (root / module.MODEL_RELATIVE).write_bytes(b"tampered")
    calls = []
    with pytest.raises(module.AdmissionError):
        load_arm_with_fake_constructor(root, attempt(tmp_path), constructor=lambda path: calls.append(path),
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
        load_arm_with_fake_constructor(root, target, constructor=lambda path: calls.append(path),
                                  expected_species=("H",) * 21)
    assert calls == []


def test_constructs_from_verified_snapshot_path(tmp_path, monkeypatch):
    import mace_loader as module
    root = packet(tmp_path, monkeypatch)
    target = attempt(tmp_path)
    calls = []
    arm, receipt = load_arm_with_fake_constructor(
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
        load_arm_with_fake_constructor(root, target, constructor=lambda path: calls.append(path),
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
        load_arm_with_fake_constructor(root, target, constructor=lambda path: calls.append(path),
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
        load_arm_with_fake_constructor(root, target, constructor=lambda path: calls.append(path),
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
        load_arm_with_fake_constructor(root, target, constructor=lambda path: calls.append(path),
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
        load_arm_with_fake_constructor(root, target, constructor=lambda path: calls.append(path),
                                  expected_species=("H",) * 21)
    assert calls == []


def test_shadow_mace_import_refused_before_constructor(tmp_path, monkeypatch):
    import mace_loader as module
    actual_validate = module._validate_import_origins
    monkeypatch.setattr(sys, "meta_path", [module.importlib.machinery.BuiltinImporter,
                        module.importlib.machinery.FrozenImporter, module.importlib.machinery.PathFinder])
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
        load_arm_with_fake_constructor(root, target, constructor=lambda path: calls.append(path),
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
    monkeypatch.setattr(sys, "meta_path", [module.importlib.machinery.BuiltinImporter,
                        module.importlib.machinery.FrozenImporter, module.importlib.machinery.PathFinder])
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
        load_arm_with_fake_constructor(root, target, constructor=lambda p: calls.append(p),
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
    assert "external-trusted-launcher" in result.stderr
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
        load_arm_with_fake_constructor(root, target, constructor=lambda path: calls.append(path),
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
        load_arm_with_fake_constructor(root, target, constructor=lambda path: calls.append(path),
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


def test_isolated_bootstrap_blocks_startup_hook_and_dependency_shadows(tmp_path):
    marker = tmp_path / "executed"
    payload = f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
    (tmp_path / "sitecustomize.py").write_text(payload)
    (tmp_path / "numpy.py").write_text(payload)
    (tmp_path / "ase").mkdir()
    (tmp_path / "ase/__init__.py").write_text(payload)
    bootstrap = Path(__file__).resolve().parent.parent / "scripts/mace_bootstrap.py"
    env = dict(os.environ, PYTHONPATH=str(tmp_path), PYTHONUSERBASE=str(tmp_path))
    for entry in ("supervisor", "worker", "audit"):
        result = subprocess.run(bootstrap_probe(sys.executable, bootstrap, entry, "--help"), cwd=tmp_path,
                                env=env, text=True, capture_output=True, timeout=30)
        if entry == "worker":
            assert result.returncode != 0
            assert "Direct checker execution is refused" in result.stderr
        else:
            assert result.returncode == 0, result.stderr
        assert not marker.exists()


@pytest.mark.parametrize("shadow_name", ["mace_loader", "mace_reference_adapter", "evaluation.adapter"])
def test_bootstrap_project_modules_win_over_loose_site_modules(tmp_path, shadow_name):
    """Exercise real isolated interpreter startup in an otherwise empty venv."""
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)],
                   check=True, capture_output=True, timeout=60)
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    packages = (venv / "Lib/site-packages" if os.name == "nt" else
                venv / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
    scripts = tmp_path / "project/scripts"
    scripts.mkdir(parents=True)
    bootstrap = scripts / "mace_bootstrap.py"
    shutil.copyfile(Path(__file__).resolve().parent.parent / "scripts/mace_bootstrap.py", bootstrap)
    marker = tmp_path / "shadow-executed"
    loose = packages / f"{shadow_name}.py"
    reviewed = scripts / f"{shadow_name}.py"
    if shadow_name == "evaluation.adapter":
        (packages / "evaluation").mkdir()
        loose = packages / "evaluation/__init__.py"
        reviewed = scripts.parent / "evaluation/adapter.py"
        reviewed.parent.mkdir()
    loose.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    reviewed.write_text("REVIEWED = True\n")
    probe = (f"import {shadow_name} as selected\nfrom pathlib import Path\n"
             f"assert Path(selected.__file__).resolve() == Path({str(reviewed)!r}).resolve()\n"
             "assert selected.REVIEWED\n")
    for name in ("supervise_mace_parity.py", "check_mace_parity.py", "audit_mace_parity.py"):
        (scripts / name).write_text(probe)
    for entry in ("supervisor", "worker", "audit"):
        result = subprocess.run(bootstrap_probe(python, bootstrap, entry),
                                cwd=tmp_path, text=True, capture_output=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert not marker.exists()


@pytest.mark.parametrize("dependency", ["e3nn", "numpy", "ase", "torch", "scipy"])
def test_bootstrap_never_searches_project_directories_for_dependencies(tmp_path, dependency):
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(venv)],
                   check=True, capture_output=True, timeout=60)
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    packages = (venv / "Lib/site-packages" if os.name == "nt" else
                venv / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    bootstrap = scripts / "mace_bootstrap.py"
    shutil.copyfile(Path(__file__).resolve().parent.parent / "scripts/mace_bootstrap.py", bootstrap)
    marker = tmp_path / "project-shadow-executed"
    payload = f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
    for directory in (project, scripts, project / "src"):
        directory.mkdir(exist_ok=True)
        (directory / f"{dependency}.py").write_text(payload)
    installed = packages / f"{dependency}.py"
    installed.write_text("INSTALLED = True\n")
    probe = (f"import {dependency} as selected\nfrom pathlib import Path\n"
             f"assert Path(selected.__file__).resolve() == Path({str(installed)!r}).resolve()\n"
             "assert selected.INSTALLED\n")
    for name in ("supervise_mace_parity.py", "check_mace_parity.py", "audit_mace_parity.py"):
        (scripts / name).write_text(probe)
    for entry in ("supervisor", "worker", "audit"):
        result = subprocess.run(bootstrap_probe(python, bootstrap, entry),
                                cwd=project, env=dict(os.environ, PYTHONPATH=str(project)),
                                text=True, capture_output=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert not marker.exists()


def test_custom_meta_finder_refused_without_resolving_constructor(tmp_path, monkeypatch):
    import mace_loader as module
    calls = []
    class SpoofFinder:
        def find_spec(self, *args):
            calls.append(args)
            raise AssertionError("custom finder must never run")
    monkeypatch.setattr(sys, "meta_path", [SpoofFinder(), *sys.meta_path])
    with pytest.raises(module.AdmissionError, match="unexpected Python import finder"):
        module._construct_calculator(tmp_path / "unused.model")
    assert calls == []


def test_direct_loader_checks_ase_before_importing_data(tmp_path, monkeypatch):
    import mace_loader as module
    actual_validate = module._validate_import_origins
    root = packet(tmp_path, monkeypatch)
    target = attempt(tmp_path, root)
    launch = json.loads((target / "launch-receipt.json").read_text())
    admission = json.loads((target / "wsl-admission.json").read_text())
    live = {"distribution_roots": {name: str(Path(importlib.metadata.distribution(name).locate_file("")).resolve())
                                    for name in ("mace-torch", "torch")}}
    shadow = tmp_path / "shadow"
    (shadow / "ase").mkdir(parents=True)
    marker = tmp_path / "executed"
    (shadow / "ase/__init__.py").write_text(f"from pathlib import Path\nPath({str(marker)!r}).touch()\n")
    (shadow / "ase/data.py").write_text("raise AssertionError('shadow data executed')\n")
    for name in list(sys.modules):
        if name == "ase" or name.startswith("ase."):
            monkeypatch.delitem(sys.modules, name)
    monkeypatch.syspath_prepend(str(shadow))
    monkeypatch.setattr(sys, "meta_path", [module.importlib.machinery.BuiltinImporter,
                        module.importlib.machinery.FrozenImporter, module.importlib.machinery.PathFinder])
    monkeypatch.setattr(module, "_validate_import_origins", actual_validate)
    monkeypatch.setattr(module, "_approved_launch", lambda *args: (launch, admission, live))
    calls = []
    with pytest.raises(module.AdmissionError, match="import origin differs"):
        load_arm_with_fake_constructor(root, target, constructor=lambda p: calls.append(p),
                                  expected_species=("H",) * 21)
    assert not marker.exists()
    assert calls == []
