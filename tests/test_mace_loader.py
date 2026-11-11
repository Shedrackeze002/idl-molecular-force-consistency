import hashlib
import json
import os
import importlib
import py_compile
import shutil
import sys
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



def isolate_frozen_imports(monkeypatch):
    """Restore collected modules after a test's fresh-worker simulation."""
    for name in ("evaluation", "evaluation.adapter", "mace_reference_adapter"):
        monkeypatch.setitem(sys.modules, name, None)
        monkeypatch.delitem(sys.modules, name)


def accepted_frozen_inventory():
    return {
        "evaluation/__init__.py": {"bytes": 220, "sha256": "919da2d40e815006ac70cd1e8862b78db10ac72ed0a32ee8dba375b813ccf18e"},
        "evaluation/adapter.py": {"bytes": 14497, "sha256": "640a4afc31e971fa33876cb0b6a022416accd0a3c348a597bbebe62483ba65c5"},
        "scripts/mace_reference_adapter.py": {"bytes": 4694, "sha256": "c0aa26e964150817b382b0b4e3f9da7ef6dec1eaa94b0c275ebfd109d37c9759"},
    }


def scratch_execution_tree(tmp_path, monkeypatch):
    import mace_loader as module
    original = Path(module.__file__).resolve().parent.parent
    scratch = tmp_path / "execution"
    for name in module.REQUIRED_IMPLEMENTATION_FILES | set(accepted_frozen_inventory()):
        target = scratch / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original / name, target)
    # Read-only Git identity lookup remains the original worktree's HEAD.
    git_dir = subprocess.check_output(
        ["git", "-C", str(original), "rev-parse", "--absolute-git-dir"], text=True,
    ).strip()
    (scratch / ".git").write_text(f"gitdir: {git_dir}\n", encoding="utf-8")
    monkeypatch.setattr(module, "__file__", str(scratch / "scripts/mace_loader.py"))
    monkeypatch.syspath_prepend(str(scratch))
    monkeypatch.syspath_prepend(str(scratch / "scripts"))
    return scratch

def packet(tmp_path, monkeypatch):
    import mace_loader as module
    isolate_frozen_imports(monkeypatch)
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



RESOURCE_FIXTURE = {"timeout_seconds": 1800, "cleanup_seconds": 10,
                    "output_limit_bytes": 268435456, "log_limit_bytes": 16777216}
EXECUTION_FIXTURE_FIELDS = ("implementation_commit", "python_executable", "python_sha256",
                            "command", "packet_root", "attempt_root", "resources",
                            "wsl_admission_sha256", "species")


def write_launch_contract(root, packet_root, launch, *, output=None):
    """Construct independent synthetic trusted approval records, never a real approval."""
    import mace_loader as module
    output = (output or root).resolve()
    launch.update(launch_schema=1, python_executable=sys.executable,
                  python_sha256=sha(Path(sys.executable)),
                  packet_root=str(packet_root.resolve()), attempt_root=str(output),
                  resources=dict(RESOURCE_FIXTURE))
    launch["command"] = [sys.executable,
        str(Path(module.__file__).resolve().with_name("check_mace_parity.py")),
        "--packet-root", launch["packet_root"], "--attempt-root", launch["attempt_root"]]
    execution = {key: launch[key] for key in EXECUTION_FIXTURE_FIELDS}
    (root / "launch-plan.json").write_text(json.dumps(execution))
    launch["plan"] = {"path": "launch-plan.json", "sha256": sha(root / "launch-plan.json")}
    launch["reviews"] = []
    for index in (1, 2):
        name = f"implementation-review-{index}.json"
        (root / name).write_text(json.dumps({"status": "APPROVED", "reviewer_id": f"reviewer-{index}",
            "implementation_commit": launch["implementation_commit"],
            "plan_sha256": launch["plan"]["sha256"], "unresolved_findings": 0}))
        launch["reviews"].append({"path": name, "sha256": sha(root / name)})
    (root / "launch-receipt.json").write_text(json.dumps(launch))
    running = {"status": "RUNNING", "synthetic_command": False, "fresh_output": True,
               "command": launch["command"], "packet_root": launch["packet_root"],
               "attempt_root": launch["attempt_root"],
               "launch_receipt_sha256": sha(root / "launch-receipt.json"), **RESOURCE_FIXTURE}
    (root / "supervisor-status.json").write_text(json.dumps(running))
    return execution


def bind_worker_argv(monkeypatch, launch):
    monkeypatch.setattr(sys, "argv", launch["command"][1:])
    monkeypatch.setattr(sys, "orig_argv", list(launch["command"]))

def attempt(tmp_path, monkeypatch):
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
    write_launch_contract(root, tmp_path / "packet", launch)
    bind_worker_argv(monkeypatch, launch)
    return root


def test_rejects_tampered_manifest_before_constructor(tmp_path, monkeypatch):
    import mace_loader as module
    root = packet(tmp_path, monkeypatch)
    (root / "MANIFEST.json").write_text("{}", encoding="utf-8")
    calls = []
    with pytest.raises(module.AdmissionError, match="manifest SHA"):
        module._load_admitted_arm(root, attempt(tmp_path, monkeypatch), constructor=lambda path: calls.append(path),
                                  expected_species=("H",) * 21)
    assert calls == []


def test_rejects_symlink_or_wrong_model_before_constructor(tmp_path, monkeypatch):
    import mace_loader as module
    root = packet(tmp_path, monkeypatch)
    (root / module.MODEL_RELATIVE).write_bytes(b"tampered")
    calls = []
    with pytest.raises(module.AdmissionError):
        module._load_admitted_arm(root, attempt(tmp_path, monkeypatch), constructor=lambda path: calls.append(path),
                                  expected_species=("H",) * 21)
    assert calls == []


def test_environment_refusal_precedes_constructor(tmp_path, monkeypatch):
    import mace_loader as module
    root = packet(tmp_path, monkeypatch)
    target = attempt(tmp_path, monkeypatch)
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
    target = attempt(tmp_path, monkeypatch)
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
    target = attempt(tmp_path, monkeypatch)
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
    target = attempt(tmp_path, monkeypatch)
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


def test_rejects_version_map_only_noncore_pin_before_constructor(tmp_path, monkeypatch):
    import mace_loader as module
    real_live_check = module._live_admission_check
    root = packet(tmp_path, monkeypatch)
    target = attempt(tmp_path, monkeypatch)
    approved = json.loads((target / "wsl-admission.json").read_text())
    approved.update(canonical_admission(module))
    installed = tmp_path / "installed"
    for item in approved["authenticated_source_comparison"]:
        path = installed / item["relative_path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"synthetic-installed-source")
        item.update(actual_sha256=sha(path), original_sha256=sha(path))
    approved["installed_package_versions"]["extra-package"] = "2"
    admission_path = target / "wsl-admission.json"
    admission_path.write_text(json.dumps(approved))
    launch = json.loads((target / "launch-receipt.json").read_text())
    launch["wsl_admission_sha256"] = sha(admission_path)
    (target / "launch-receipt.json").write_text(json.dumps(launch))
    monkeypatch.setattr(module, "_live_admission_check", real_live_check)
    for name, value in (("python_version", "3.12.13"), ("platform", "Linux-test"),
                        ("machine", "x86_64")):
        monkeypatch.setattr(module.platform, name, lambda value=value: value)
    monkeypatch.setattr(module.platform, "libc_ver", lambda: ("glibc", "2.35"))
    monkeypatch.setattr(module.importlib.metadata, "version", lambda name: "1")
    class InstalledDistribution:
        def locate_file(self, relative):
            return installed / relative
    monkeypatch.setattr(module.importlib.metadata, "distribution", lambda name: InstalledDistribution())
    calls = []
    with pytest.raises(module.AdmissionError, match="package|inventor"):
        module._load_admitted_arm(
            root, target, constructor=lambda path: calls.append(path) or FakeCalculator(),
            expected_species=("H",) * 21)
    assert calls == []


@pytest.mark.parametrize("key,value", [
    ("extra-package", "1"), ("Torch_EMA", "1"), (7, "1"),
    ("", "1"), ("not a package", "1"), ("extra-package", ""),
    ("extra-package", None),
])
def test_version_map_rejects_unlisted_duplicate_or_malformed_pins(key, value):
    import mace_loader as module
    receipt = canonical_admission(module)
    receipt["installed_package_versions"][key] = value
    with pytest.raises(module.AdmissionError, match="package|inventor"):
        module._approved_identity_inventory(receipt)


def test_complete_package_inventory_normalizes_both_sides():
    import mace_loader as module
    receipt = canonical_admission(module)
    receipt["installed_packages"].append("Example_Pkg==2")
    receipt["installed_package_versions"]["EXAMPLE-PKG"] = "2"
    result = module._approved_identity_inventory(receipt)
    assert result["packages"] == {**{name: "1" for name in module.CORE_PACKAGES},
                                  "example-pkg": "2"}


def test_declared_noncore_pin_requires_matching_map_version():
    import mace_loader as module
    receipt = canonical_admission(module)
    receipt["installed_packages"].append("extra-package==2")
    receipt["installed_package_versions"]["extra-package"] = "1"
    with pytest.raises(module.AdmissionError, match="package|inventor"):
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


@pytest.mark.parametrize("relative", list(accepted_frozen_inventory()))
def test_changed_frozen_source_refused_before_constructor(tmp_path, monkeypatch, relative):
    import mace_loader as module
    root = packet(tmp_path, monkeypatch)
    scratch = scratch_execution_tree(tmp_path, monkeypatch)
    path = scratch / relative
    data = path.read_bytes()
    # Equal-size valid Python edit: source identity, not only length, matters.
    path.write_bytes(data.replace(b"a", b"b", 1))
    calls = []
    with pytest.raises(module.AdmissionError, match="frozen"):
        module._load_admitted_arm(root, attempt(tmp_path, monkeypatch),
            constructor=lambda path: calls.append(path) or FakeCalculator(),
            expected_species=("H",) * 21)
    assert calls == []


@pytest.mark.parametrize("damage", ["missing", "symlink-file", "symlink-parent"])
def test_nonregular_frozen_source_refused_before_constructor(tmp_path, monkeypatch, damage):
    import mace_loader as module
    root = packet(tmp_path, monkeypatch)
    scratch = scratch_execution_tree(tmp_path, monkeypatch)
    path = scratch / "evaluation/adapter.py"
    if damage == "symlink-parent":
        directory = scratch / "evaluation"
        directory.rename(scratch / "other-evaluation")
        directory.symlink_to(scratch / "other-evaluation", target_is_directory=True)
    else:
        data = path.read_bytes()
        path.unlink()
        if damage == "symlink-file":
            other = scratch / "other.py"
            other.write_bytes(data)
            path.symlink_to(other)
    calls = []
    with pytest.raises(module.AdmissionError, match="frozen"):
        module._load_admitted_arm(root, attempt(tmp_path, monkeypatch),
            constructor=lambda path: calls.append(path) or FakeCalculator(),
            expected_species=("H",) * 21)
    assert calls == []


@pytest.mark.parametrize("name", ["evaluation", "evaluation.adapter", "mace_reference_adapter"])
def test_preloaded_frozen_module_refused_before_constructor(tmp_path, monkeypatch, name):
    import mace_loader as module
    root = packet(tmp_path, monkeypatch)
    scratch_execution_tree(tmp_path, monkeypatch)
    # Real standard imports have correct origins but unverified execution history.
    importlib.import_module(name)
    calls = []
    with pytest.raises(module.AdmissionError, match="preloaded|fresh"):
        module._load_admitted_arm(root, attempt(tmp_path, monkeypatch),
            constructor=lambda path: calls.append(path) or FakeCalculator(),
            expected_species=("H",) * 21)
    assert calls == []


def poison_bridge_cache(scratch):
    path = scratch / "scripts/mace_reference_adapter.py"
    original, stamp = path.read_bytes(), path.stat()
    poisoned = original.replace(b"checkpoint_identity_verified=False)",
                                b"checkpoint_identity_verified=True )")
    assert poisoned != original and len(poisoned) == len(original)
    path.write_bytes(poisoned)
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    py_compile.compile(str(path), doraise=True)
    path.write_bytes(original)
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    # Prove that the normal source loader would read different cached code.
    loader = importlib.machinery.SourceFileLoader("mace_reference_adapter", str(path))
    assert loader.get_code("mace_reference_adapter") != compile(
        original, str(path), "exec", dont_inherit=True)


@pytest.mark.parametrize("preloaded", [False, True])
def test_poisoned_bytecode_never_supplies_admitted_bridge(tmp_path, monkeypatch, preloaded):
    import mace_loader as module
    root = packet(tmp_path, monkeypatch)
    scratch = scratch_execution_tree(tmp_path, monkeypatch)
    poison_bridge_cache(scratch)
    if preloaded:
        importlib.import_module("mace_reference_adapter")
    calls = []
    def construct(path):
        calls.append(path)
        return FakeCalculator()
    if preloaded:
        with pytest.raises(module.AdmissionError, match="preloaded|fresh"):
            module._load_admitted_arm(root, attempt(tmp_path, monkeypatch), constructor=construct,
                                      expected_species=("H",) * 21)
        assert calls == []
    else:
        arm, receipt = module._load_admitted_arm(
            root, attempt(tmp_path, monkeypatch), constructor=construct, expected_species=("H",) * 21)
        assert len(calls) == 1
        assert arm["checkpoint_identity_verified"] is False
        assert receipt["live_identity_comparison"]["frozen_source_files"] == accepted_frozen_inventory()


def test_clean_frozen_bridge_bound_before_constructor(tmp_path, monkeypatch):
    import mace_loader as module
    root = packet(tmp_path, monkeypatch)
    scratch_execution_tree(tmp_path, monkeypatch)
    calls = []
    def construct(path):
        # A later fresh import would observe this substituted module instead.
        from types import ModuleType
        replacement = ModuleType("mace_reference_adapter")
        replacement.calculator_arm = lambda *args, **kwargs: {"substituted": True}
        monkeypatch.setitem(sys.modules, "mace_reference_adapter", replacement)
        calls.append(path)
        return FakeCalculator()
    arm, receipt = module._load_admitted_arm(
        root, attempt(tmp_path, monkeypatch), constructor=construct, expected_species=("H",) * 21)
    assert len(calls) == 1
    assert arm["checkpoint_identity_verified"] is False
    assert receipt["live_identity_comparison"]["frozen_source_files"] == accepted_frozen_inventory()


def test_shadow_bridge_refused_without_execution_and_imports_rolled_back(tmp_path, monkeypatch):
    import mace_loader as module
    root = packet(tmp_path, monkeypatch)
    scratch_execution_tree(tmp_path, monkeypatch)
    shadow = tmp_path / "shadow"
    shadow.mkdir()
    marker = tmp_path / "shadow-executed"
    (shadow / "mace_reference_adapter.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('executed')\n"
        "def calculator_arm(*args, **kwargs): return {'untrusted': True}\n")
    monkeypatch.syspath_prepend(str(shadow))
    calls = []
    with pytest.raises(module.AdmissionError, match="frozen|origin"):
        module._load_admitted_arm(root, attempt(tmp_path, monkeypatch),
            constructor=lambda path: calls.append(path) or FakeCalculator(),
            expected_species=("H",) * 21)
    assert calls == []
    assert not marker.exists()
    assert all(name not in sys.modules for name in (
        "evaluation", "evaluation.adapter", "mace_reference_adapter"))


@pytest.mark.parametrize("damage", ["missing-command", "executable", "executable-sha", "command", "output", "packet", "resources", "plan-hash", "review-hash", "duplicate-review", "review-pending", "review-findings", "review-plan", "plan-command", "reserved-path", "prior-output", "supervisor-command", "supervisor-resources", "supervisor-synthetic", "supervisor-missing-flag", "argv", "orig-argv-flag", "orig-argv-X"])
def test_launch_binding_refuses_before_constructor(tmp_path, monkeypatch, damage):
    import mace_loader as module
    root = packet(tmp_path, monkeypatch)
    target = attempt(tmp_path, monkeypatch)
    lp = target / "launch-receipt.json"
    launch = json.loads(lp.read_text())
    if damage == "missing-command": del launch["command"]
    elif damage == "executable": launch["python_executable"] = "/not/the/interpreter"
    elif damage == "executable-sha": launch["python_sha256"] = "0" * 64
    elif damage == "command": launch["command"][1] = "/different/check_mace_parity.py"
    elif damage == "output": launch["attempt_root"] = "/other/attempt"
    elif damage == "packet": launch["packet_root"] = "/other/packet"
    elif damage == "resources": launch["resources"]["timeout_seconds"] = 3600
    elif damage == "plan-hash": launch["plan"]["sha256"] = "0" * 64
    elif damage == "review-hash": launch["reviews"][0]["sha256"] = "0" * 64
    elif damage == "duplicate-review": launch["reviews"][1] = launch["reviews"][0]
    elif damage == "reserved-path": launch["reviews"][0]["path"] = "stdout.log"
    elif damage.startswith("review-"):
        ref = launch["reviews"][0]; path = target / ref["path"]
        review = json.loads(path.read_text())
        key, value = {"review-pending": ("status", "PROPOSED"), "review-findings": ("unresolved_findings", 1), "review-plan": ("plan_sha256", "0" * 64)}[damage]
        review[key] = value; path.write_text(json.dumps(review)); ref["sha256"] = sha(path)
    elif damage == "plan-command":
        path = target / launch["plan"]["path"]; plan = json.loads(path.read_text())
        plan["command"][1] = "/different/check_mace_parity.py"; path.write_text(json.dumps(plan)); launch["plan"]["sha256"] = sha(path)
    elif damage == "prior-output": (target / "predictions.npz").write_bytes(b"stale")
    elif damage == "argv": monkeypatch.setattr(sys, "argv", ["different.py"])
    elif damage in {"orig-argv-flag", "orig-argv-X"}:
        flags = ["-O"] if damage == "orig-argv-flag" else ["-X", "dev"]
        monkeypatch.setattr(sys, "orig_argv", [sys.executable, *flags, *launch["command"][1:]])
    lp.write_text(json.dumps(launch))
    sp = target / "supervisor-status.json"; supervisor = json.loads(sp.read_text())
    supervisor["launch_receipt_sha256"] = sha(lp)
    if damage == "supervisor-command": supervisor["command"] = ["different"]
    elif damage == "supervisor-resources": supervisor["log_limit_bytes"] = 1
    elif damage == "supervisor-synthetic": supervisor["synthetic_command"] = True
    elif damage == "supervisor-missing-flag": del supervisor["synthetic_command"]
    sp.write_text(json.dumps(supervisor))
    calls = []
    with pytest.raises(module.AdmissionError, match="launch|review|plan|supervis|invocation|attempt"):
        module._load_admitted_arm(root, target, constructor=lambda path: calls.append(path) or FakeCalculator(), expected_species=("H",) * 21)
    assert calls == []


def test_approved_venv_symlink_identity_is_preserved(tmp_path, monkeypatch):
    import mace_loader as module
    root = packet(tmp_path, monkeypatch)
    alias = tmp_path / "venv/bin/python"; alias.parent.mkdir(parents=True)
    alias.symlink_to(sys.executable)
    monkeypatch.setattr(sys, "executable", str(alias))
    target = attempt(tmp_path, monkeypatch)
    calls = []
    arm, receipt = module._load_admitted_arm(root, target, constructor=lambda path: calls.append(path) or FakeCalculator(), expected_species=("H",) * 21)
    assert len(calls) == 1
    assert receipt["live_identity_comparison"]["launch_identity"]["python_executable"] == str(alias)


def refresh_execution_approvals(root, launch):
    """Keep nested synthetic approvals coherent so live checks must catch mismatch."""
    execution = {key: launch[key] for key in EXECUTION_FIXTURE_FIELDS}
    plan_path = root / launch["plan"]["path"]
    plan_path.write_text(json.dumps(execution))
    launch["plan"]["sha256"] = sha(plan_path)
    for ref in launch["reviews"]:
        path = root / ref["path"]
        review = json.loads(path.read_text())
        review["plan_sha256"] = launch["plan"]["sha256"]
        path.write_text(json.dumps(review))
        ref["sha256"] = sha(path)
    lp = root / "launch-receipt.json"
    lp.write_text(json.dumps(launch))
    sp = root / "supervisor-status.json"
    supervisor = json.loads(sp.read_text())
    supervisor.update(command=launch["command"], packet_root=launch["packet_root"],
                      attempt_root=launch["attempt_root"], launch_receipt_sha256=sha(lp))
    sp.write_text(json.dumps(supervisor))


@pytest.mark.parametrize("damage", ["executable", "executable-sha", "packet", "attempt", "worker"])
def test_coherent_approvals_cannot_replace_live_launch_context(tmp_path, monkeypatch, damage):
    import mace_loader as module
    root = packet(tmp_path, monkeypatch)
    target = attempt(tmp_path, monkeypatch)
    launch = json.loads((target / "launch-receipt.json").read_text())
    if damage == "executable":
        launch["python_executable"] = launch["command"][0] = "/different/python"
    elif damage == "executable-sha": launch["python_sha256"] = "0" * 64
    elif damage == "packet": launch["packet_root"] = launch["command"][3] = "/different/packet"
    elif damage == "attempt": launch["attempt_root"] = launch["command"][5] = "/different/output"
    else: launch["command"][1] = "/different/check_mace_parity.py"
    refresh_execution_approvals(target, launch)
    calls = []
    with pytest.raises(module.AdmissionError, match="live invocation"):
        module._load_admitted_arm(root, target,
            constructor=lambda path: calls.append(path) or FakeCalculator(), expected_species=("H",) * 21)
    assert calls == []


@pytest.mark.parametrize("field", ["launch_schema", "python_executable", "python_sha256", "packet_root", "attempt_root", "resources", "plan", "reviews"])
def test_incomplete_launch_contract_cannot_construct(tmp_path, monkeypatch, field):
    import mace_loader as module
    root = packet(tmp_path, monkeypatch)
    target = attempt(tmp_path, monkeypatch)
    path = target / "launch-receipt.json"; launch = json.loads(path.read_text())
    del launch[field]; path.write_text(json.dumps(launch))
    calls = []
    with pytest.raises(module.AdmissionError, match="launch"):
        module._load_admitted_arm(root, target,
            constructor=lambda path: calls.append(path) or FakeCalculator(), expected_species=("H",) * 21)
    assert calls == []



@pytest.mark.parametrize("name", ["parity-status.json.tmp", "predictions.npz.tmp.npz", "predictions-partial.npz.tmp.npz"])
def test_worker_temporary_names_cannot_overwrite_approved_reviews(tmp_path, monkeypatch, name):
    import mace_loader as module
    root = packet(tmp_path, monkeypatch)
    target = attempt(tmp_path, monkeypatch)
    lp = target / "launch-receipt.json"; launch = json.loads(lp.read_text())
    ref = launch["reviews"][0]; (target / ref["path"]).rename(target / name); ref["path"] = name
    lp.write_text(json.dumps(launch))
    sp = target / "supervisor-status.json"; supervisor = json.loads(sp.read_text())
    supervisor["launch_receipt_sha256"] = sha(lp); sp.write_text(json.dumps(supervisor))
    calls = []
    with pytest.raises(module.AdmissionError, match="reserved|launch"):
        module._load_admitted_arm(root, target,
            constructor=lambda path: calls.append(path) or FakeCalculator(), expected_species=("H",) * 21)
    assert calls == []
    assert (target / name).is_file()
