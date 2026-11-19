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
def boundary(tmp_path, request):
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
    # Use the real bootstrap, supervisor and launch validator. Only the worker
    # is synthetic: the complete production handoff runs without any model.
    shutil.copyfile(SOURCE.parent.parent / "scripts/mace_bootstrap.py", root / "scripts/mace_bootstrap.py")
    for name in ("supervise_mace_parity.py", "mace_loader.py"):
        shutil.copyfile(SOURCE.parent.parent / "scripts" / name, root / "scripts" / name)
    variant = getattr(request, "param", {})
    supervisor = root / "scripts/supervise_mace_parity.py"
    for before, after in variant.get("supervisor_edits", []):
        supervisor.write_text(supervisor.read_text().replace(before, after))
    (root / "scripts/check_mace_parity.py").write_text(
        "from pathlib import Path\nimport os,json,sys\n"
        f"attempt=Path({str(tmp_path / 'attempt')!r})\n"
        "assert json.loads((attempt/'supervisor-status.json').read_text())['status']=='RUNNING'\n"
        "assert (attempt/'stdout.log').exists() and (attempt/'stderr.log').exists()\n"
        "assert os.getsid(0)==os.getpid() and os.getppid()!=os.getpid()\n"
        "assert sys.flags.isolated and sys.flags.no_site\n"
        + variant.get("worker", f"Path({str(marker)!r}).write_text('worker')\n"))
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
                    "--entry", "supervisor", "--policy", str(policy)],
        "python_executable": sys.executable,
        "resource_limits": {"seconds": 1800, "attempt_bytes": 268435456,
                            "per_log_bytes": 16777216, "cleanup_seconds": 10},
        "source_baseline": "3fa5969e5a0c2fdeaebc76a05bace20799715203",
        "implementation_reviews": [{"status": "RESOLVED", "implementation_commit": commit,
            "review_id": str(i), "reviewer": "synthetic-" + str(i),
            "url": "https://example.test/review/" + str(i)} for i in (1, 2)],
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
                policy_hash=sha(policy), execution=execution, marker=marker, commit=commit,
                attempt=attempt)


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
    status = json.loads((boundary["attempt"] / "supervisor-status.json").read_text())
    assert status["status"] == "WORKER_COMPLETED"
    assert status["exit_code"] == 0
    assert status["timeout_seconds"] == 1800
    assert (boundary["attempt"] / "stdout.log").exists()
    assert (boundary["attempt"] / "stderr.log").exists()
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


def test_direct_worker_refuses_even_with_staged_snapshot(boundary):
    assert run(boundary).returncode == 0
    boundary["marker"].unlink()
    path = boundary["execution"] / "scripts/check_mace_parity.py"
    path.chmod(0o644)
    path.write_text("from pathlib import Path\n" + f"Path({str(boundary['marker'])!r}).touch()\n")
    result = run(boundary, entry="worker")
    assert result.returncode == 2
    assert "invalid choice: 'worker'" in result.stderr
    assert not boundary["marker"].exists()


def test_direct_worker_refuses_without_supervisor_status(boundary):
    assert run(boundary).returncode == 0
    boundary["marker"].unlink()
    status = boundary["attempt"] / "supervisor-status.json"
    status.unlink()
    command = json.loads((boundary["control"] / "launch.json").read_text())["command"]
    command[command.index("--entry") + 1] = "worker"
    result = subprocess.run(command, env=dict(os.environ, MACE_TRUST_POLICY_SHA256=boundary["policy_hash"]),
                            text=True, capture_output=True, timeout=90)
    assert result.returncode == 2
    assert "invalid choice: 'worker'" in result.stderr
    assert not boundary["marker"].exists()
    assert not status.exists()


def test_direct_worker_refuses_before_any_snapshot_or_project_startup(boundary):
    result = run(boundary, entry="worker")
    assert result.returncode == 2
    assert "invalid choice: 'worker'" in result.stderr
    assert not boundary["execution"].exists()
    assert not boundary["attempt"].exists()
    assert not boundary["marker"].exists()


def test_prepare_api_cannot_select_worker(boundary):
    with pytest.raises(ValueError, match="private to the bounded supervisor"):
        trusted.prepare(boundary["policy"], boundary["policy_hash"], "worker")
    assert not boundary["execution"].exists()
    assert not boundary["marker"].exists()


@pytest.mark.parametrize("boundary,expected", [
    ({"worker": "import time; time.sleep(30)\n",
      "supervisor_edits": [("WALL_SECONDS = 1800", "WALL_SECONDS = 2")]}, "TIMEOUT"),
    ({"worker": "print('x'*10000)\n",
      "supervisor_edits": [("LOG_BYTES = 16 * 1024 * 1024", "LOG_BYTES = 100")]}, "LOG_OVERFLOW"),
    ({"worker": "(attempt/'large-output').write_bytes(b'x'*10000)\n",
      "supervisor_edits": [("OUTPUT_BYTES = 256 * 1024 * 1024", "OUTPUT_BYTES = 5000")]}, "OUTPUT_OVERFLOW"),
], indirect=["boundary"])
def test_real_supervisor_bounds_its_private_worker(boundary, expected):
    result = run(boundary)
    assert result.returncode == 2, result.stderr
    status = json.loads((boundary["attempt"] / "supervisor-status.json").read_text())
    assert status["status"] == expected
    assert not boundary["marker"].exists()


@pytest.mark.parametrize("filename,entry", [("check_mace_parity.py", "supervisor"),
    ("supervise_mace_parity.py", "supervisor"), ("audit_mace_parity.py", "audit")])
def test_direct_entry_refusal_points_to_supported_external_launcher(filename, entry):
    result = subprocess.run([sys.executable, str(SOURCE.parent.parent / "scripts" / filename), "--help"],
                            text=True, capture_output=True, timeout=30)
    assert result.returncode != 0
    assert "<external-trusted-launcher>" in result.stderr
    assert "--entry " + entry in result.stderr
    assert "mace_bootstrap.py --entry" not in result.stderr


@pytest.mark.parametrize("entry,filename", [("worker", "check_mace_parity.py"),
    ("supervisor", "supervise_mace_parity.py"), ("audit", "audit_mace_parity.py")])
def test_import_and_call_main_refuses_tampered_entry(tmp_path, entry, filename):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    bootstrap = scripts / "mace_bootstrap.py"
    shutil.copyfile(SOURCE.parent.parent / "scripts/mace_bootstrap.py", bootstrap)
    marker = tmp_path / "entry-executed"
    (scripts / filename).write_text("from pathlib import Path\n" +
                                    f"Path({str(marker)!r}).touch()\n")
    # Reproduce the removed BOOTSTRAP_DISPATCH import-and-call route verbatim.
    dispatch = ("import importlib.util,sys; path=sys.argv.pop(1); "
                "spec=importlib.util.spec_from_file_location('verified_mace_bootstrap',path); "
                "module=importlib.util.module_from_spec(spec); "
                "spec.loader.exec_module(module); module.main()")
    result = subprocess.run([sys.executable, "-I", "-S", "-c", dispatch, str(bootstrap),
                             "--entry", entry, "--help"], text=True, capture_output=True,
                            env=dict(os.environ, MACE_TRUST_POLICY_SHA256="f" * 64), timeout=30)
    assert result.returncode != 0
    assert "Bootstrap dispatch API is refused" in result.stderr
    assert not marker.exists()
