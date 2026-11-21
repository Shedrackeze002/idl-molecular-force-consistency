"""Real subprocess trust-boundary tests; synthetic Git trees, no model inputs."""
import hashlib
import ast
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import textwrap

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
    runtime = root / "trust/mace_trusted_launcher.py"
    for before, after in variant.get("supervisor_edits", []):
        # Runtime moved one lexical level into protected main(). Synthetic
        # source edits remain pinned by the fixture's immutable Git commit.
        runtime.write_text(runtime.read_text().replace(before, after.replace("\n", "\n    ")))
    for before, after in variant.get("launcher_edits", []):
        assert before in runtime.read_text()
        runtime.write_text(runtime.read_text().replace(before, after))
    worker_body = (
        "from pathlib import Path\nimport os,json,sys\n"
        f"attempt=Path({str(tmp_path / 'attempt')!r})\n"
        "assert json.loads((attempt/'supervisor-status.json').read_text())['status']=='RUNNING'\n"
        "assert (attempt/'stdout.log').exists() and (attempt/'stderr.log').exists()\n"
        "assert os.getsid(0)==os.getpid() and os.getppid()!=os.getpid()\n"
        "assert sys.flags.isolated and sys.flags.no_site\n"
        f"check_liveness(Path({str(tmp_path / 'packet')!r}),attempt)\n"
        + variant.get("worker", f"Path({str(marker)!r}).write_text('worker')\n")
        + "return {'status':'PARITY_PASS'}\n")
    (root / "scripts/check_mace_parity.py").write_text(
        "def _run_parity(packet_root,attempt_root,*,load_arm,check_liveness):\n" + textwrap.indent(worker_body, "    "))
    if variant.get("real_worker"):
        worker_source = (SOURCE.parent.parent / "scripts/check_mace_parity.py").read_text()
        tree = ast.parse(worker_source)
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "_run_parity":
                node.body = ast.parse(worker_body).body
        (root / "scripts/check_mace_parity.py").write_text(ast.unparse(tree) + "\n")
    integration = None
    if variant.get("integration"):
        from mace_test_runtime import inert_packet_sources
        integration = inert_packet_sources(root,tmp_path/"packet",tmp_path/"attempt")
        # Import immutable baseline objects, not a mutable working-tree bridge.
        trusted.stage_git_objects(SOURCE.parents[1],root,git(SOURCE.parents[1],"rev-parse","HEAD"),
                                  "3fa5969e5a0c2fdeaebc76a05bace20799715203")
    git(root, "add", ".")
    git(root, "commit", "-qm", "synthetic approved implementation")
    commit = git(root, "rev-parse", "HEAD")
    control = tmp_path / "external-trust"
    control.mkdir()
    launcher = control / "mace_trusted_launcher.py"
    shutil.copyfile(runtime, launcher)
    execution = tmp_path / "verified-execution"
    packet = tmp_path / "packet"
    attempt = tmp_path / "attempt"
    admission = control / "admission.json"
    admission_data={"status":"APPROVED","identities":{"implementation_commit":commit}}
    if integration is not None:
        from mace_test_runtime import inert_admission
        admission_data=inert_admission(root,commit)
    admission.write_text(json.dumps(admission_data))
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
        "species": ["H"]*21,
        "packet_root": str(packet), "attempt_root": str(attempt),
        "wsl_admission_receipt": "admission.json", "wsl_admission_sha256": sha(admission)}))
    policy.write_text(json.dumps({"schema": 1, "status": "APPROVED",
        "checkout_root": str(root), "execution_root": str(execution),
        "implementation_commit": commit,
        "baseline_commit": "3fa5969e5a0c2fdeaebc76a05bace20799715203" if integration is not None else commit,
        "python_executable": sys.executable,
        "trusted_launcher_sha256": sha(launcher),
        "launch_receipt": str(launch), "launch_sha256": sha(launch),
        "wsl_admission_receipt": str(admission), "wsl_admission_sha256": sha(admission)}))
    return dict(root=root, control=control, launcher=launcher, policy=policy,
                policy_hash=sha(policy), execution=execution, marker=marker, commit=commit,
                attempt=attempt, integration=integration)


def run(case, entry="supervisor", cwd=None):
    return subprocess.run([sys.executable, "-I", "-S", str(case["launcher"]),
        "--policy", str(case["policy"]), "--policy-sha256", case["policy_hash"], "--entry", entry],
        cwd=cwd, text=True, capture_output=True, timeout=90)


@pytest.mark.parametrize("boundary", [{"integration":True}], indirect=True)
def test_protected_launcher_runs_actual_parity_through_private_loader(boundary):
    import numpy as np
    assert (boundary["root"]/"scripts/check_mace_parity.py").read_bytes()==(
        SOURCE.parents[1]/"scripts/check_mace_parity.py").read_bytes()
    result=run(boundary)
    assert result.returncode==0, (result.stdout,result.stderr,
        (boundary["attempt"]/"stderr.log").read_text() if (boundary["attempt"]/"stderr.log").exists() else "")
    attempt=boundary["attempt"]
    supervisor=json.loads((attempt/"supervisor-status.json").read_text())
    parity=json.loads((attempt/"parity-status.json").read_text())
    admission=json.loads((attempt/"admission-receipt.json").read_text())
    assert supervisor["status"]=="WORKER_COMPLETED" and supervisor["exit_code"]==0
    assert parity["status"]=="PARITY_PASS"
    assert parity["construction_count"]==1
    assert parity["record_count"]==parity["persisted_records"]==974
    assert parity["force_requests"]==parity["energy_requests"]==974
    assert parity["max_abs_energy_eV"]==parity["max_abs_force_eV_per_angstrom"]==0
    events=[json.loads(line) for line in (attempt/"inert-events.jsonl").read_text().splitlines()]
    assert len(events)==1949 and events[0]["kind"]=="construction"
    assert {event["pid"] for event in events}=={events[0]["pid"]}
    assert [event["kind"] for event in events[1:]]==["forces","energy"]*974
    expected=boundary["integration"]
    assert [event["index"] for event in events[1::2]]==expected["indices"]
    assert [event["index"] for event in events[2::2]]==expected["indices"]
    for filename in ("predictions.npz","predictions-partial.npz"):
        with np.load(attempt/filename,allow_pickle=False) as saved:
            assert saved["record_ids"].tolist()==expected["record_ids"]
            np.testing.assert_array_equal(saved["energy"],expected["energy"])
            np.testing.assert_array_equal(saved["forces"],expected["forces"])
    assert parity["prediction_file_sha256"]==sha(attempt/"predictions.npz")
    assert parity["admission_receipt_sha256"]==sha(attempt/"admission-receipt.json")
    snapshot=attempt/"private/final_epoch.model"
    assert snapshot.read_bytes()==b"INERT FIXTURE - NOT A SERIALIZED MODEL\n"
    assert admission["model_snapshot"]["model_sha256"]==sha(snapshot)
    assert admission["post_construction_sha256"]==sha(snapshot)
    assert parity["final_model_snapshot"]["sha256"]==sha(snapshot)


@pytest.mark.parametrize("boundary", [{"supervisor_edits":[
    ("WALL_SECONDS = 1800","WALL_SECONDS = .2"),
    ],"launcher_edits":[
    ("sources = verify_sources(checkout, commit)","time.sleep(1.25)\n        sources = verify_sources(checkout, commit)")]}],indirect=True)
def test_setup_delay_is_interrupted_at_shared_deadline_and_retained(boundary):
    started=time.monotonic()
    result=run(boundary)
    elapsed=time.monotonic()-started
    assert result.returncode==2
    assert "TRUST_TIMEOUT" in result.stderr
    assert elapsed<.85, (elapsed,result.stdout,result.stderr)
    status=json.loads((boundary["attempt"]/"supervisor-status.json").read_text())
    assert status["status"]=="TIMEOUT" and status["failure_stage"]=="trusted_setup"
    assert status["worker_started"] is False and status["exit_code"] is None
    assert status["elapsed_seconds"]<.7
    assert (boundary["attempt"]/"stdout.log").read_bytes()==b""
    assert not boundary["marker"].exists()
    assert not boundary["execution"].exists()


@pytest.mark.parametrize("boundary", [{"supervisor_edits":[("WALL_SECONDS = 1800","WALL_SECONDS = .2")],
    "launcher_edits":[("policy = pinned_json(policy_path, policy_sha256)",
        "time.sleep(1.25)\n    policy = pinned_json(policy_path, policy_sha256)")]}],indirect=True)
def test_timeout_before_authentication_emits_evidence_without_unverified_writes(boundary):
    started=time.monotonic()
    result=run(boundary)
    assert time.monotonic()-started<.85
    assert result.returncode==2 and "TRUST_TIMEOUT" in result.stderr
    status=json.loads(result.stdout)
    assert status["status"]=="TIMEOUT" and status["worker_started"] is False
    assert not boundary["attempt"].exists() and not boundary["execution"].exists()


@pytest.mark.parametrize("boundary", [{"supervisor_edits":[("WALL_SECONDS = 1800","WALL_SECONDS = .2")],
    "launcher_edits":[("def git(checkout, *args):",'''def git(checkout, *args):
    if args == ("rev-parse", "HEAD"):
        return subprocess.check_output([sys.executable,"-I","-S","-c",
            "import os,sys,time;open(sys.argv[1],'w').write(str(os.getpid()));time.sleep(1.25)",
            str(checkout/"setup-git.pid")])''')]}],indirect=True)
def test_setup_deadline_interrupts_and_reaps_synchronous_git_wait(boundary):
    started=time.monotonic()
    result=run(boundary)
    assert time.monotonic()-started<.85
    assert result.returncode==2 and "TRUST_TIMEOUT" in result.stderr
    status=json.loads((boundary["attempt"]/"supervisor-status.json").read_text())
    assert status["status"]=="TIMEOUT" and status["worker_started"] is False
    pid=int((boundary["root"]/"setup-git.pid").read_text())
    with pytest.raises(ProcessLookupError): os.kill(pid,0)
    assert not boundary["marker"].exists()


@pytest.mark.parametrize("boundary", [{"supervisor_edits":[("WALL_SECONDS = 1800","WALL_SECONDS = 2")],
    "launcher_edits":[('namespace["configure_imports"]()',
        '(execution/"imports-entered").touch()\n            time.sleep(3.25)\n            namespace["configure_imports"]()')]}],indirect=True)
def test_import_configuration_uses_the_same_initial_deadline(boundary):
    started=time.monotonic()
    result=run(boundary)
    assert time.monotonic()-started<2.75
    assert result.returncode==2 and "TRUST_TIMEOUT" in result.stderr
    assert (boundary["execution"]/"imports-entered").exists()
    status=json.loads((boundary["attempt"]/"supervisor-status.json").read_text())
    assert status["status"]=="TIMEOUT" and status["worker_started"] is False
    assert not boundary["marker"].exists()


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


@pytest.mark.parametrize("flags", [[], ["-I", "-S"]])
def test_checker_cli_requires_capability_before_dependency_imports(tmp_path, flags):
    worker = SOURCE.parent.parent / "scripts/check_mace_parity.py"
    result = subprocess.run([sys.executable, *flags, str(worker),
        "--packet-root", str(tmp_path / "packet"), "--attempt-root", str(tmp_path / "attempt")],
        capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert "Direct checker execution is refused" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    assert not (tmp_path / "attempt").exists()


@pytest.mark.parametrize("forgery", ["none", "noop", "fake-object"])
def test_isolated_import_and_call_main_refuses_before_synthetic_parity(tmp_path, forgery):
    marker = tmp_path / "unbounded-parity"
    bootstrap = SOURCE.parent.parent / "scripts/mace_bootstrap.py"
    probe = ("import importlib.util,sys; from pathlib import Path; "
             "spec=importlib.util.spec_from_file_location('probe',sys.argv.pop(1)); "
             "module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); "
             "module.configure_imports(); import mace_loader as loader; "
             + ({"none": "", "noop": "loader._worker_capability=lambda *args: None; ",
                 "fake-object": "loader._worker_capability=object(); "}[forgery]) +
             "import check_mace_parity as worker; "
             f"worker.run_parity=lambda *args: (Path({str(marker)!r}).touch(),{{'status':'PARITY_PASS'}})[1]; "
             f"loader.load_admitted_arm=lambda *args,**kwargs: Path({str(marker)!r}).touch(); "
             f"loader._construct_calculator=lambda *args: Path({str(marker)!r}).touch(); "
             "worker.main()")
    result = subprocess.run([sys.executable, "-I", "-S", "-c", probe, str(bootstrap),
        "--packet-root", str(tmp_path), "--attempt-root", str(tmp_path / "attempt")],
        text=True, capture_output=True, timeout=30,
        env=dict(os.environ, MACE_TRUST_STARTED=str(time.monotonic()), MACE_TRUST_POLICY_SHA256="a" * 64))
    assert result.returncode != 0
    assert "Direct checker main() is refused" in result.stderr
    assert not marker.exists()


@pytest.mark.parametrize("entry", ["main", "supervise", "_supervise"])
def test_forged_supervisor_callback_with_unapproved_receipts_never_runs(tmp_path, entry):
    packet, attempt = tmp_path / "packet", tmp_path / "attempt"
    launcher, policy = tmp_path / "launcher.py", tmp_path / "policy.json"
    admission = tmp_path / "wsl.json"
    admission.write_text(json.dumps({"status": "NOT_APPROVED", "identities": {"implementation_commit": "a"*40}}))
    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps({"status": "NOT_APPROVED", "implementation_commit": "a"*40,
        "python_executable": sys.executable, "trusted_launcher": str(launcher), "trusted_policy": str(policy),
        "packet_root": str(packet), "attempt_root": str(attempt),
        "command": [sys.executable,"-I","-S",str(launcher),"--entry","supervisor","--policy",str(policy)],
        "wsl_admission_receipt": admission.name, "wsl_admission_sha256": "0"*64,
        "source_baseline": "3fa5969e5a0c2fdeaebc76a05bace20799715203",
        "resource_limits": {"seconds":1800,"attempt_bytes":268435456,"per_log_bytes":16777216,"cleanup_seconds":10},
        "implementation_reviews": [{"status":"RESOLVED","implementation_commit":"a"*40,
            "review_id":str(i),"reviewer":"fake-"+str(i),"url":"https://example.test/"+str(i)} for i in (1,2)]}))
    marker = tmp_path / "forged-callback-ran"
    probe = f'''
import importlib.util,sys
from pathlib import Path
bootstrap,supervisor,entry=sys.argv[1:4]
sys.argv=[supervisor,*sys.argv[4:]]
spec=importlib.util.spec_from_file_location('bootstrap',bootstrap)
bootstrap_module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap_module)
bootstrap_module.configure_imports()
spec=importlib.util.spec_from_file_location('supervisor',supervisor)
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
class FakeProcess:
    pid=2147483647
    returncode=0
    def poll(self): return 0
    def wait(self,timeout=None): return 0
def forged(*args,**kwargs):
    Path({str(marker)!r}).touch()
    return FakeProcess()
module.__trusted_worker_spawn__=forged
if entry=='main':
    raise SystemExit(module.main())
getattr(module,entry)(Path({str(packet)!r}),Path({str(attempt)!r}),worker_spawn=forged,
    launch_receipt=Path({str(launch)!r}),wsl_admission_receipt=Path({str(admission)!r}))
'''
    result = subprocess.run([sys.executable,"-I","-S","-c",probe,
        str(SOURCE.parent.parent/"scripts/mace_bootstrap.py"),
        str(SOURCE.parent.parent/"scripts/supervise_mace_parity.py"),entry,
        "--packet-root",str(packet),"--output",str(attempt),"--launch-receipt",str(launch),
        "--wsl-admission-receipt",str(admission)], capture_output=True,text=True,timeout=30,
        start_new_session=True)
    assert result.returncode != 0
    assert "refused" in result.stderr
    assert not marker.exists()
    assert not attempt.exists()


def test_direct_internal_loader_refuses_injected_constructor_before_any_model_access(tmp_path):
    marker = tmp_path / "constructor-invoked"
    probe = f'''
import importlib.util,sys,hashlib
from pathlib import Path
spec=importlib.util.spec_from_file_location('bootstrap',sys.argv[1])
bootstrap=importlib.util.module_from_spec(spec)
spec.loader.exec_module(bootstrap)
bootstrap.configure_imports()
import mace_loader as loader
root=Path({str(tmp_path)!r})
packet=root/'packet'
(packet/'model').mkdir(parents=True)
fake=packet/'model/final_epoch.model'
fake.write_bytes(b'synthetic non-model bytes')
loader.MODEL_BYTES=fake.stat().st_size
loader.MODEL_SHA256=hashlib.sha256(fake.read_bytes()).hexdigest()
loader.verify_packet=lambda *args: {{'status':'PACKET_VERIFIED'}}
loader._approved_launch=lambda *args: ({{'species':['H']*21}},{{'status':'APPROVED'}},{{}})
loader._validate_import_origins=lambda *args: None
loader.require_supervised_worker=lambda *args: None
loader._worker_capability=lambda *args: None
def constructor(*args):
    Path({str(marker)!r}).touch()
    raise RuntimeError('injected constructor invoked outside supervisor')
loader._load_admitted_arm(packet,root/'attempt',constructor=constructor,expected_species=('H',)*21)
'''
    result = subprocess.run([sys.executable,"-I","-S","-c",probe,
        str(SOURCE.parent.parent/"scripts/mace_bootstrap.py")],capture_output=True,text=True,timeout=30)
    assert result.returncode != 0
    assert "Direct project loading is refused" in result.stderr
    assert not marker.exists()
    assert not (tmp_path/"attempt").exists()


@pytest.mark.parametrize("boundary", [{"real_worker": True}], indirect=True)
def test_verified_private_driver_runs_real_checker_library(boundary):
    result = run(boundary)
    assert result.returncode == 0, result.stderr
    assert boundary["marker"].read_text() == "worker"
    status = json.loads((boundary["attempt"] / "supervisor-status.json").read_text())
    assert status["status"] == "WORKER_COMPLETED"


@pytest.mark.parametrize("boundary", [{"worker": "check_liveness(attempt,attempt)\n"}], indirect=True)
def test_private_driver_refuses_different_approved_paths(boundary):
    result = run(boundary)
    assert result.returncode == 2
    assert not boundary["marker"].exists()
    assert "does not authorize these paths" in (boundary["attempt"] / "stderr.log").read_text()


@pytest.mark.parametrize("boundary", [{"worker":
    "pid=os.fork()\n"
    "if pid==0:\n"
    "    try: check_liveness()\n"
    "    except ValueError: os._exit(0)\n"
    "    os._exit(1)\n"
    "assert os.waitstatus_to_exitcode(os.waitpid(pid,0)[1])==0\n"
    "(attempt/'fork-capability-refused').touch()\n"}], indirect=True)
def test_private_driver_liveness_is_bound_to_its_child_process(boundary):
    result = run(boundary)
    assert result.returncode == 0, result.stderr
    assert (boundary["attempt"] / "fork-capability-refused").exists()


@pytest.mark.parametrize("entry", ["parity", "loader", "constructor"])
def test_public_execution_apis_refuse_before_model_access(tmp_path, monkeypatch, entry):
    import check_mace_parity as worker
    import mace_loader as loader
    monkeypatch.setattr(loader, "_worker_capability", lambda *args: None, raising=False)
    monkeypatch.setattr(loader, "require_supervised_worker", lambda *args: None)
    monkeypatch.setattr(loader, "verify_packet", lambda *args: pytest.fail("packet/model access before authority"))
    with pytest.raises(RuntimeError, match="Direct project"):
        if entry == "parity":
            worker.run_parity(tmp_path, tmp_path / "attempt")
        elif entry == "loader":
            loader.load_admitted_arm(tmp_path, tmp_path / "attempt", expected_species=("H",) * 21)
        else:
            # Pytest's own finders are unrelated to this production test.
            monkeypatch.setattr(loader, "_require_standard_finders", lambda: None)
            loader._construct_calculator(tmp_path / "unused.model")


@pytest.mark.parametrize("boundary", [{"supervisor_edits": [
    ('status_path.write_text(json.dumps(receipt, indent=2) + "\\n")',
     'status_path.write_text(json.dumps(receipt, indent=2) + "\\n")\n'
     '        os.kill(os.getpid(), signal.SIGTERM)')]}], indirect=True)
def test_trusted_launch_interrupted_at_initial_receipt_never_starts_worker(boundary):
    result = run(boundary)
    assert result.returncode == 2, result.stderr
    status = json.loads((boundary["attempt"] / "supervisor-status.json").read_text())
    assert status["status"] == "INTERRUPTED"
    assert status["failure_stage"] == "setup"
    assert status["signal"] == signal.SIGTERM
    assert status["exit_code"] is None
    assert not boundary["marker"].exists()


def wait_for_file(path, process, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return
        assert process.poll() is None, process.communicate()
        time.sleep(.02)
    pytest.fail("synthetic worker did not reach startup barrier")


@pytest.mark.parametrize("boundary", [{"worker":
    "(attempt/'worker.pid').write_text(str(os.getpid()))\n"
    "(attempt/'ready').touch()\nimport time; time.sleep(1.5)\n"
    "(attempt/'escaped-completion').touch()\n"}], indirect=True)
@pytest.mark.parametrize("signum", [signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT])
def test_parent_interruption_reaps_actual_handoff_worker(boundary, signum):
    process = subprocess.Popen([sys.executable, "-I", "-S", str(boundary["launcher"]),
        "--policy", str(boundary["policy"]), "--policy-sha256", boundary["policy_hash"],
        "--entry", "supervisor"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        wait_for_file(boundary["attempt"] / "ready", process)
        pid = int((boundary["attempt"] / "worker.pid").read_text())
        os.kill(process.pid, signum)
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 2, (stdout, stderr)
        status = json.loads((boundary["attempt"] / "supervisor-status.json").read_text())
        assert status["status"] == "INTERRUPTED"
        assert status["signal"] == signum
        assert status["exit_code"] is not None
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        # Observe beyond the reviewer's delayed marker window.
        time.sleep(1.6)
        assert not (boundary["attempt"] / "escaped-completion").exists()
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=15)


@pytest.mark.parametrize("boundary", [{"worker": "import time; time.sleep(30)\n",
    "supervisor_edits": [("process = worker_spawn(stdout, stderr)",
        "process = worker_spawn(stdout, stderr)\n"
        "                    (output/'spawn-owned').write_text(str(process.pid))\n"
        "                    time.sleep(.3)")]}], indirect=True)
def test_interruption_during_spawn_waits_for_child_ownership(boundary):
    process = subprocess.Popen([sys.executable, "-I", "-S", str(boundary["launcher"]),
        "--policy", str(boundary["policy"]), "--policy-sha256", boundary["policy_hash"],
        "--entry", "supervisor"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        barrier = boundary["attempt"] / "spawn-owned"
        wait_for_file(barrier, process)
        pid = int(barrier.read_text())
        process.terminate()
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 2, (stdout, stderr)
        status = json.loads((boundary["attempt"] / "supervisor-status.json").read_text())
        assert status["status"] == "INTERRUPTED"
        assert status["signal"] == signal.SIGTERM
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=15)


@pytest.mark.parametrize("boundary", [{"worker":
    "import subprocess,time\n"
    "child_code=\"import signal,time; from pathlib import Path; signal.signal(signal.SIGTERM,signal.SIG_IGN); Path(\"+repr(str(attempt/'child-ready'))+\").touch(); time.sleep(30)\"\n"
    "child=subprocess.Popen([sys.executable,'-c',child_code])\n"
    "(attempt/'child.pid').write_text(str(child.pid))\n"
    "(attempt/'worker.pid').write_text(str(os.getpid()))\n"
    "while not (attempt/'child-ready').exists(): time.sleep(.01)\n"
    "(attempt/'ready').touch()\ntime.sleep(30)\n"}], indirect=True)
def test_repeated_parent_signals_still_kill_term_ignoring_descendant(boundary):
    process = subprocess.Popen([sys.executable, "-I", "-S", str(boundary["launcher"]),
        "--policy", str(boundary["policy"]), "--policy-sha256", boundary["policy_hash"],
        "--entry", "supervisor"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        wait_for_file(boundary["attempt"] / "ready", process)
        pids = [int((boundary["attempt"] / name).read_text()) for name in ("worker.pid", "child.pid")]
        process.terminate()
        time.sleep(.1)
        os.kill(process.pid, signal.SIGINT)
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 2, (stdout, stderr)
        status = json.loads((boundary["attempt"] / "supervisor-status.json").read_text())
        assert status["status"] == "INTERRUPTED"
        assert status["signal"] == signal.SIGTERM
        for pid in pids:
            with pytest.raises(ProcessLookupError):
                os.kill(pid, 0)
    finally:
        if process.poll() is None:
            process.terminate()
            process.communicate(timeout=15)


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
