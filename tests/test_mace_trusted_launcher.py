"""Real subprocess trust-boundary tests; synthetic Git trees, no model inputs."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

SOURCE = Path(__file__).resolve().parents[1] / "trust/mace_trusted_launcher.py"
spec = importlib.util.spec_from_file_location("trusted_launcher_tests", SOURCE)
trusted = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trusted)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


@pytest.fixture
def boundary(tmp_path):
    root = tmp_path / "mutable-checkout"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "Synthetic test")
    git(root, "config", "user.email", "synthetic@example.test")
    git(root, "config", "core.autocrlf", "false")
    for relative in (*trusted.IMPLEMENTATION_FILES, *trusted.FROZEN_FILES):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# synthetic reviewed source\n")
    shutil.copyfile(SOURCE, root / "trust/mace_trusted_launcher.py")
    marker = tmp_path / "project-executed"
    # Exercise bootstrap -> supervisor -> worker, with each entry observable.
    (root / "scripts/mace_bootstrap.py").write_text(
        "from pathlib import Path\nimport runpy\n"
        "def main():\n"
        f"    Path({str(marker)!r}).write_text('bootstrap')\n"
        "    runpy.run_path(str(Path(__file__).with_name('supervise_mace_parity.py')),run_name='__main__')\n")
    (root / "scripts/supervise_mace_parity.py").write_text(
        "from pathlib import Path\nimport runpy\n"
        "runpy.run_path(str(Path(__file__).with_name('check_mace_parity.py')),run_name='__main__')\n")
    (root / "scripts/check_mace_parity.py").write_text(
        "from pathlib import Path\n" + f"Path({str(marker)!r}).write_text('worker')\n")
    git(root, "add", ".")
    git(root, "commit", "-qm", "synthetic approved implementation")
    commit = git(root, "rev-parse", "HEAD")
    control = tmp_path / "external-trust"
    control.mkdir()
    launcher = control / "mace_trusted_launcher.py"
    shutil.copyfile(SOURCE, launcher)
    execution = tmp_path / "verified-execution"
    packet = tmp_path / "packet"
    attempt = tmp_path / "attempt"
    admission = control / "admission.json"
    admission.write_text(json.dumps({"status": "APPROVED", "identities": {
        "implementation_commit": commit}}))
    launch = control / "launch.json"
    policy = control / "policy.json"
    launch.write_text(json.dumps({"status": "APPROVED", "implementation_commit": commit,
        "command": [sys.executable, "-I", "-S", str(launcher),
                    "--entry", "worker", "--policy", str(policy)],
        "trusted_launcher": str(launcher), "trusted_policy": str(policy),
        "packet_root": str(packet), "attempt_root": str(attempt),
        "wsl_admission_receipt": "admission.json", "wsl_admission_sha256": sha(admission)}))
    policy.write_text(json.dumps({"schema": 1, "status": "APPROVED",
        "checkout_root": str(root), "execution_root": str(execution),
        "implementation_commit": commit, "baseline_commit": commit, "python_executable": sys.executable,
        "trusted_launcher_sha256": sha(launcher),
        "launch_receipt": str(launch), "launch_sha256": sha(launch),
        "wsl_admission_receipt": str(admission), "wsl_admission_sha256": sha(admission)}))
    return dict(root=root, control=control, launcher=launcher, policy=policy,
                policy_hash=sha(policy), execution=execution, marker=marker, commit=commit)


def run(case, entry="supervisor", cwd=None):
    return subprocess.run([sys.executable, "-I", "-S", str(case["launcher"]),
        "--policy", str(case["policy"]), "--policy-sha256", case["policy_hash"], "--entry", entry],
        cwd=cwd, text=True, capture_output=True, timeout=90)


@pytest.mark.parametrize("relative", ["scripts/mace_bootstrap.py", "scripts/check_mace_parity.py",
    "scripts/supervise_mace_parity.py", "scripts/mace_loader.py", "scripts/mace_reference_adapter.py"])
def test_tampering_refused_before_any_project_code_starts(boundary, relative):
    path = boundary["root"] / relative
    path.write_text("from pathlib import Path\n" +
                    f"Path({str(boundary['marker'])!r}).write_text('TAMPERED')\n")
    result = run(boundary)
    assert result.returncode == 2
    assert "raw source differs before project startup" in result.stderr
    assert not boundary["marker"].exists()
    assert not boundary["execution"].exists()


def test_verified_sources_execute_from_separate_snapshot(boundary):
    result = run(boundary)
    assert result.returncode == 0, result.stderr
    assert boundary["marker"].read_text() == "worker"
    snapshot = boundary["execution"]
    assert git(snapshot, "rev-parse", "HEAD") == boundary["commit"]
    assert (snapshot / "scripts/mace_bootstrap.py").read_bytes() == (
        boundary["root"] / "scripts/mace_bootstrap.py").read_bytes()
    # Mutating the original checkout cannot affect the already verified copy.
    (boundary["root"] / "scripts/check_mace_parity.py").write_text("raise RuntimeError('changed later')\n")
    assert "changed later" not in (snapshot / "scripts/check_mace_parity.py").read_text()


@pytest.mark.parametrize("damage", ["commit", "policy", "launch", "admission", "launcher-label"])
def test_external_anchor_and_receipt_mismatches_refuse(boundary, damage):
    if damage == "commit":
        git(boundary["root"], "commit", "--allow-empty", "-qm", "unreviewed revision")
    elif damage in ("launch", "admission"):
        (boundary["control"] / (damage + ".json")).write_text("{}")
    else:
        policy = json.loads(boundary["policy"].read_text())
        policy["trusted_launcher_sha256"] = "0" * 64
        boundary["policy"].write_text(json.dumps(policy))
        if damage == "launcher-label":
            boundary["policy_hash"] = sha(boundary["policy"])
    result = run(boundary)
    assert result.returncode == 2, result.stderr
    assert not boundary["marker"].exists()


def test_launcher_inside_checkout_is_not_a_trust_anchor(boundary):
    boundary["launcher"] = boundary["root"] / "trust/mace_trusted_launcher.py"
    result = run(boundary)
    assert result.returncode == 2
    assert "outside the mutable checkout" in result.stderr
    assert not boundary["marker"].exists()


def test_existing_snapshot_refused_without_overwriting(boundary):
    boundary["execution"].mkdir()
    sentinel = boundary["execution"] / "preserve"
    sentinel.write_text("user content")
    result = run(boundary)
    assert result.returncode == 2
    assert sentinel.read_text() == "user content"
    assert not boundary["marker"].exists()


def test_startup_hook_does_not_execute_before_trust_gate(boundary, tmp_path, monkeypatch):
    (tmp_path / "sitecustomize.py").write_text(
        "from pathlib import Path\n" + f"Path({str(boundary['marker'])!r}).touch()\n")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    (boundary["root"] / "scripts/check_mace_parity.py").write_text("# modified")
    result = run(boundary, cwd=tmp_path)
    assert result.returncode == 2
    assert not boundary["marker"].exists()


@pytest.mark.parametrize("relative", ["scripts/mace_bootstrap.py", "scripts/check_mace_parity.py"])
def test_audit_rejects_tampered_execution_snapshot_before_startup(boundary, relative):
    assert run(boundary).returncode == 0
    boundary["marker"].unlink()
    path = boundary["execution"] / relative
    path.chmod(0o644)
    path.write_text("from pathlib import Path\n" + f"Path({str(boundary['marker'])!r}).touch()\n")
    result = run(boundary, entry="audit")
    assert result.returncode == 2
    assert not boundary["marker"].exists()


def test_external_and_worker_source_inventory_cannot_drift():
    import mace_loader
    assert set(trusted.IMPLEMENTATION_FILES) == mace_loader.REQUIRED_IMPLEMENTATION_FILES
    assert set(trusted.FROZEN_FILES) == set(mace_loader.FROZEN_DEPENDENCY_BLOBS)


@pytest.mark.parametrize("entry,filename", [("worker", "check_mace_parity.py"),
    ("supervisor", "supervise_mace_parity.py"), ("audit", "audit_mace_parity.py")])
def test_direct_bootstrap_refuses_before_tampered_entry_executes(tmp_path, entry, filename):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    bootstrap = scripts / "mace_bootstrap.py"
    shutil.copyfile(SOURCE.parent.parent / "scripts/mace_bootstrap.py", bootstrap)
    marker = tmp_path / "entry-executed"
    (scripts / filename).write_text("from pathlib import Path\n" +
                                    f"Path({str(marker)!r}).touch()\nraise SystemExit(17)\n")
    result = subprocess.run([sys.executable, "-I", "-S", str(bootstrap),
                             "--entry", entry, "--help"], cwd=tmp_path,
                            text=True, capture_output=True, timeout=30)
    assert result.returncode != 0
    assert "Direct bootstrap invocation is refused" in result.stderr
    assert not marker.exists()


def test_worker_external_entry_rechecks_snapshot(boundary):
    assert run(boundary).returncode == 0
    boundary["marker"].unlink()
    path = boundary["execution"] / "scripts/check_mace_parity.py"
    path.chmod(0o644)
    path.write_text("from pathlib import Path\n" + f"Path({str(boundary['marker'])!r}).touch()\n")
    result = run(boundary, entry="worker")
    assert result.returncode == 2
    assert not boundary["marker"].exists()


def test_verified_worker_uses_inherited_independent_policy_digest(boundary):
    assert run(boundary).returncode == 0
    boundary["marker"].unlink()
    command = json.loads((boundary["control"] / "launch.json").read_text())["command"]
    result = subprocess.run(command, env=dict(os.environ, MACE_TRUST_POLICY_SHA256=boundary["policy_hash"]),
                            text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stderr
    assert boundary["marker"].read_text() == "worker"
