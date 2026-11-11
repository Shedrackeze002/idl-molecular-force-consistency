import json
import os
import shutil
import signal
import subprocess
import sys
import threading
from types import SimpleNamespace
import time

import pytest

import supervise_mace_parity as module


def command(code):
    return [sys.executable, "-c", code]


def test_refuses_existing_output(tmp_path):
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(FileExistsError):
        module.supervise(tmp_path, output, command=command("pass"))


def test_production_launch_requires_both_reviewed_receipts(tmp_path):
    result = module.supervise(tmp_path, tmp_path / "attempt")
    assert result["status"] == "SETUP_FAILED"
    assert "requires reviewed" in result["error"]


def test_records_child_failure(tmp_path):
    result = module.supervise(tmp_path, tmp_path / "attempt", command=command("raise SystemExit(7)"))
    assert result["status"] == "WORKER_FAILED"
    assert result["exit_code"] == 7


def test_timeout_is_incomplete_and_preserved(tmp_path):
    result = module.supervise(tmp_path, tmp_path / "timeout", timeout=.05,
                              command=command("import time; time.sleep(5)"))
    assert result["status"] == "TIMEOUT"
    assert json.loads((tmp_path / "timeout/supervisor-status.json").read_text())["status"] == "TIMEOUT"


def test_log_overflow_never_completes(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "LOG_BYTES", 100)
    result = module.supervise(tmp_path, tmp_path / "overflow", command=command("print('x'*10000)"))
    assert result["status"] in {"LOG_OVERFLOW", "WORKER_FAILED"}
    assert result["status"] != "WORKER_COMPLETED"


def test_final_receipt_is_included_in_output_bound(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "OUTPUT_BYTES", 350)
    result = module.supervise(tmp_path, tmp_path / "bounded", command=command("pass"))
    assert result["status"] == "OUTPUT_OVERFLOW"
    assert result["output_bytes"] >= 350


def test_setup_and_launch_delay_counts_toward_timeout(tmp_path, monkeypatch):
    original = module.subprocess.Popen
    def delayed(*args, **kwargs):
        time.sleep(.06)
        return original(*args, **kwargs)
    monkeypatch.setattr(module.subprocess, "Popen", delayed)
    result = module.supervise(tmp_path, tmp_path / "delayed", timeout=.05, command=command("pass"))
    assert result["status"] == "TIMEOUT"


def test_spawn_failure_is_persisted(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("synthetic spawn failure")
    monkeypatch.setattr(module.subprocess, "Popen", fail)
    output = tmp_path / "spawn"
    result = module.supervise(tmp_path, output, command=command("pass"))
    assert result["status"] == "SPAWN_FAILED"
    assert json.loads((output / "supervisor-status.json").read_text())["error_type"] == "OSError"


def test_receipt_copy_failure_is_persisted(tmp_path):
    launch = tmp_path / "launch.json"
    launch.write_text(json.dumps({"wsl_admission_receipt": "missing.json"}))
    output = tmp_path / "copy"
    result = module.supervise(tmp_path, output, launch_receipt=launch,
                              wsl_admission_receipt=tmp_path / "absent.json")
    assert result["status"] == "SETUP_FAILED"
    assert json.loads((output / "supervisor-status.json").read_text())["failure_stage"] == "setup"


def test_last_status_write_delay_counts_toward_timeout(tmp_path, monkeypatch):
    original = module.Path.write_text
    def delayed(self, data, *args, **kwargs):
        if self.name == "supervisor-status.json.tmp" and json.loads(data)["status"] == "WORKER_COMPLETED":
            time.sleep(.06)
        return original(self, data, *args, **kwargs)
    monkeypatch.setattr(module.Path, "write_text", delayed)
    result = module.supervise(tmp_path, tmp_path / "last-write", timeout=.05,
                              command=command("pass"))
    assert result["status"] == "TIMEOUT"


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group regression")
def test_term_ignoring_child_is_killed_after_leader_exits(tmp_path):
    pidfile = tmp_path / "child.pid"
    code = ("import pathlib,signal,subprocess,sys,time; "
            f"p=subprocess.Popen([sys.executable,'-c','import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)']); "
            f"pathlib.Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(30)")
    result = module.supervise(tmp_path, tmp_path / "group-timeout", timeout=1.0, command=command(code))
    assert result["status"] == "TIMEOUT"
    child_pid = int(pidfile.read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(child_pid, 0)


@pytest.mark.parametrize("damage", ["command", "executable", "resources", "review", "actual-resource-type", "supervisor-status.json.tmp", "parity-status.json.tmp", "predictions.npz.tmp.npz", "predictions-partial.npz.tmp.npz"])
def test_production_binding_refuses_before_spawn(tmp_path, monkeypatch, damage):
    from test_mace_loader import write_launch_contract, sha
    approvals=tmp_path / "approvals"; approvals.mkdir()
    output=tmp_path / "output"
    admission=approvals / "wsl.json"; admission.write_text('{}')
    launch={"status":"APPROVED", "species":["H"]*21, "implementation_commit":"a"*40,
            "wsl_admission_receipt":"wsl.json", "wsl_admission_sha256":sha(admission)}
    write_launch_contract(approvals,tmp_path,launch,output=output)
    if damage == "command": launch["command"][1]="/different/check_mace_parity.py"
    elif damage == "executable": launch["python_executable"]="/not/the/interpreter"
    elif damage == "resources": launch["resources"]["timeout_seconds"]=3600
    elif damage == "review": launch["reviews"][0]["sha256"]="0"*64
    elif damage != "actual-resource-type":
        ref=launch["reviews"][0]; (approvals / ref["path"]).rename(approvals / damage); ref["path"]=damage
    lp=approvals / "launch-receipt.json"; lp.write_text(json.dumps(launch))
    calls=[]
    def forbidden(*args,**kwargs):
        calls.append(args)
        raise OSError("spawn must not occur")
    monkeypatch.setattr(module.subprocess,"Popen",forbidden)
    result=module.supervise(tmp_path,output,timeout=1800.0 if damage == "actual-resource-type" else 1800,launch_receipt=lp,wsl_admission_receipt=admission)
    assert result["status"] == "SETUP_FAILED"
    assert calls == []



def test_valid_production_launch_records_actual_binding(tmp_path, monkeypatch):
    from test_mace_loader import write_launch_contract, sha
    approvals = tmp_path / "approvals"; approvals.mkdir()
    output = tmp_path / "output"
    admission = approvals / "wsl.json"; admission.write_text('{}')
    launch = {"status": "APPROVED", "species": ["H"] * 21, "implementation_commit": "a" * 40,
              "wsl_admission_receipt": "wsl.json", "wsl_admission_sha256": sha(admission)}
    write_launch_contract(approvals, tmp_path, launch, output=output)
    class CompletedProcess:
        returncode = 0
        def poll(self): return 0
    def spawn(argv, **kwargs):
        running = json.loads((output / "supervisor-status.json").read_text())
        assert running["status"] == "RUNNING"
        assert running["command"] == argv == launch["command"]
        assert running["launch_receipt_sha256"] == sha(approvals / "launch-receipt.json")
        assert running["fresh_output"] is True
        assert running["synthetic_command"] is False
        return CompletedProcess()
    monkeypatch.setattr(module.subprocess, "Popen", spawn)
    monkeypatch.setattr(module, "_group_alive", lambda process: False)
    result = module.supervise(tmp_path, output,
        launch_receipt=approvals / "launch-receipt.json", wsl_admission_receipt=admission)
    assert result["status"] == "WORKER_COMPLETED"
    assert result["command"] == launch["command"]
    assert all((output / ref["path"]).is_file() for ref in [launch["plan"], *launch["reviews"]])


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group regression")
@pytest.mark.parametrize("signal_number", [signal.SIGINT, signal.SIGTERM])
def test_cli_interrupt_terminates_owned_worker(tmp_path, signal_number):
    pid_path = tmp_path / "worker.pid"
    worker = command(
        f"import os,time; from pathlib import Path; "
        f"Path({str(pid_path)!r}).write_text(str(os.getpid())); time.sleep(30)"
    )
    output = tmp_path / "attempt"
    bootstrap = (
        "from functools import partial; import supervise_mace_parity as m; "
        f"m.supervise = partial(m.supervise, command={worker!r}); "
        "raise SystemExit(m.main())"
    )
    supervisor = subprocess.Popen(
        command(bootstrap) + [
            "--packet-root", str(tmp_path), "--output", str(output),
            "--launch-receipt", "unused", "--wsl-admission-receipt", "unused",
        ],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    worker_pid = None
    try:
        deadline = time.monotonic() + 5
        while not pid_path.exists() or not pid_path.read_text():
            if supervisor.poll() is not None or time.monotonic() > deadline:
                pytest.fail("synthetic worker did not report its PID")
            time.sleep(0.01)
        worker_pid = int(pid_path.read_text())
        supervisor.send_signal(signal_number)
        assert supervisor.wait(timeout=5) == 2
        status = json.loads((output / "supervisor-status.json").read_text())
        assert status["status"] == "INTERRUPTED"
        assert status["interrupt_signal"] == signal_number
        with pytest.raises(ProcessLookupError):
            os.kill(worker_pid, 0)
        worker_pid = None
    finally:
        if worker_pid is not None:
            try:
                os.killpg(worker_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if supervisor.poll() is None:
            supervisor.kill()
        supervisor.wait(timeout=5)


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal regression")
@pytest.mark.parametrize("cli", [True, False], ids=["cli", "library"])
def test_interrupt_during_final_status_write(tmp_path, monkeypatch, cli):
    output = tmp_path / "attempt"
    original_supervise = module.supervise
    monkeypatch.setattr(
        module, "supervise",
        lambda *args, **kwargs: original_supervise(*args, command=command("pass"), **kwargs),
    )
    monkeypatch.setattr(sys, "argv", [
        "supervise_mace_parity.py", "--packet-root", str(tmp_path), "--output", str(output),
        "--launch-receipt", "unused", "--wsl-admission-receipt", "unused",
    ])
    original_write = module.Path.write_text
    saw_final_write = False

    def interrupt_final_write(path, data, *args, **kwargs):
        nonlocal saw_final_write
        if path.name == "supervisor-status.json.tmp" and not saw_final_write:
            receipt = json.loads(data)
            if receipt["status"] == "WORKER_COMPLETED" and "output_bytes" in receipt:
                saw_final_write = True
                if cli:
                    signal.raise_signal(signal.SIGINT)
                else:
                    raise KeyboardInterrupt
        return original_write(path, data, *args, **kwargs)

    monkeypatch.setattr(module.Path, "write_text", interrupt_final_write)
    try:
        if cli:
            assert module.main() == 2
        else:
            assert module.supervise(tmp_path, output)["status"] == "INTERRUPTED"
    except KeyboardInterrupt:
        pytest.fail("supervisor lost terminal evidence when interrupted")
    assert saw_final_write
    status = json.loads((output / "supervisor-status.json").read_text())
    assert status["status"] == "INTERRUPTED"
    assert status["interrupt_signal"] == signal.SIGINT


@pytest.mark.parametrize("fault", [None, "deadline", "replace"])
def test_saved_audit_rejects_failed_completion_publication(tmp_path, monkeypatch, fault):
    import mace_loader as loader
    from test_mace_parity_audit import module as auditor, fixture_inputs, write_attempt

    attempt = tmp_path / "attempt"
    packet = write_attempt(attempt)
    fixture = attempt.rename(tmp_path / "fixture")
    status_path = attempt / "supervisor-status.json"
    elapsed = 0
    fail_persistence = False
    original_write = module.Path.write_text
    original_replace = module.Path.replace

    def complete_worker(*args, **kwargs):
        for name in ("predictions.npz", "admission-receipt.json", "parity-status.json"):
            shutil.copyfile(fixture / name, attempt / name)
        shutil.copytree(fixture / "private", attempt / "private")
        return SimpleNamespace(poll=lambda: 0, returncode=0)

    def simulate_write(path, data, *args, **kwargs):
        nonlocal elapsed, fail_persistence
        if path.parent == attempt and path.name.startswith(status_path.name):
            if fail_persistence:
                raise OSError("simulated persistence failure")
            receipt = json.loads(data)
            if fault == "deadline" and receipt["status"] == "WORKER_COMPLETED" and "output_bytes" in receipt:
                elapsed = 1801
                fail_persistence = True
        return original_write(path, data, *args, **kwargs)

    def simulate_replace(path, destination):
        nonlocal fail_persistence
        if fault == "replace" and destination == status_path:
            fail_persistence = True
            raise OSError("simulated persistence failure")
        return original_replace(path, destination)

    monkeypatch.setattr(module.subprocess, "Popen", complete_worker)
    monkeypatch.setattr(module, "_group_alive", lambda process: False)
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: elapsed, time=lambda: 0))
    monkeypatch.setattr(module.Path, "write_text", simulate_write)
    monkeypatch.setattr(module.Path, "replace", simulate_replace)
    monkeypatch.setattr(loader, "verify_packet", lambda path: packet)
    monkeypatch.setattr(auditor, "validate_inputs", lambda path: fixture_inputs())
    launch = {
        "launch_receipt": fixture / "launch-receipt.json",
        "wsl_admission_receipt": fixture / "wsl.json",
    }
    if fault is None:
        result = module.supervise(tmp_path, attempt, **launch)
        assert result == json.loads(status_path.read_text())
        assert result["output_bytes"] == module._tree_bytes(attempt)
        assert auditor.audit_saved_parity(tmp_path, attempt)["parity"]["status"] == "PARITY_PASS"
    else:
        with pytest.raises(OSError, match="simulated persistence failure"):
            module.supervise(tmp_path, attempt, **launch)
        with pytest.raises(ValueError, match="operational supervisor"):
            auditor.audit_saved_parity(tmp_path, attempt)
        assert json.loads(status_path.read_text())["status"] == "RUNNING"


@pytest.mark.skipif(os.name == "nt", reason="POSIX process-group regression")
def test_direct_call_finishes_cleanup_after_sigint(tmp_path, monkeypatch):
    marker = tmp_path / "term-received"
    worker = command(
        "import signal,time; from pathlib import Path; "
        f"signal.signal(signal.SIGTERM, lambda *_: Path({str(marker)!r}).touch()); "
        "time.sleep(20)"
    )
    processes = []
    original_popen = module.subprocess.Popen
    done = threading.Event()

    def remember_process(*args, **kwargs):
        process = original_popen(*args, **kwargs)
        processes.append(process)
        return process

    def interrupt_after_term():
        while not marker.exists():
            if done.wait(0.01):
                return
        if not done.is_set():
            os.kill(os.getpid(), signal.SIGINT)

    monkeypatch.setattr(module.subprocess, "Popen", remember_process)
    original_handler = signal.getsignal(signal.SIGINT)
    sender = threading.Thread(target=interrupt_after_term, daemon=True)
    sender.start()
    output = tmp_path / "attempt"
    try:
        result = module.supervise(tmp_path, output, timeout=0.5, command=worker)
        assert len(processes) == 1 and marker.exists()
        assert processes[0].poll() is not None
        assert result["status"] == "TIMEOUT"
        assert result["interrupt_signal"] == signal.SIGINT
        assert result == json.loads((output / "supervisor-status.json").read_text())
        assert signal.getsignal(signal.SIGINT) == original_handler
    finally:
        done.set()
        sender.join(timeout=2)
        for process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=2)


@pytest.mark.parametrize("phase", ["before", "after"])
def test_completion_staging_overlap_counts_toward_budget(tmp_path, monkeypatch, phase):
    monkeypatch.setattr(module.subprocess, "Popen",
                        lambda *args, **kwargs: SimpleNamespace(poll=lambda: 0, returncode=0))
    monkeypatch.setattr(module, "_group_alive", lambda process: False)
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: 0, time=lambda: 0))
    control = module.supervise(tmp_path, tmp_path / "control", command=command("pass"))
    assert control["status"] == "WORKER_COMPLETED"
    output = tmp_path / "bounded"
    original_write = module.Path.write_text
    staged_success = False

    def measure_overlap(path, data, *args, **kwargs):
        nonlocal staged_success
        count = original_write(path, data, *args, **kwargs)
        if path.parent == output and path.name == "supervisor-status.json.tmp":
            receipt = json.loads(data)
            if receipt["status"] == "WORKER_COMPLETED":
                staged_success = True
                actual = module._tree_bytes(output)
                assert receipt["output_bytes"] < actual
                monkeypatch.setattr(module, "OUTPUT_BYTES", actual - 1)
        return count

    monkeypatch.setattr(module.Path, "write_text", measure_overlap)
    if phase == "before":
        monkeypatch.setattr(module, "OUTPUT_BYTES", control["output_bytes"] + 1)
    result = module.supervise(tmp_path, output, command=command("pass"))
    assert result == json.loads((output / "supervisor-status.json").read_text())
    assert result["status"] == "OUTPUT_OVERFLOW"
    assert staged_success == (phase == "after")


@pytest.mark.skipif(os.name == "nt", reason="POSIX signal regression")
def test_signal_during_final_deadline_sample_refuses_completion(tmp_path, monkeypatch):
    monkeypatch.setattr(module.subprocess, "Popen",
                        lambda *args, **kwargs: SimpleNamespace(poll=lambda: 0, returncode=0))
    monkeypatch.setattr(module, "_group_alive", lambda process: False)
    original_write = module.Path.write_text
    staged_success = False
    signal_sent = False

    def track_staging(path, data, *args, **kwargs):
        nonlocal staged_success
        count = original_write(path, data, *args, **kwargs)
        if path.name == "supervisor-status.json.tmp":
            staged_success = json.loads(data)["status"] == "WORKER_COMPLETED"
        return count

    def signal_on_final_clock():
        nonlocal signal_sent
        if staged_success and not signal_sent:
            signal_sent = True
            signal.raise_signal(signal.SIGINT)
        return 0

    monkeypatch.setattr(module.Path, "write_text", track_staging)
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=signal_on_final_clock, time=lambda: 0))
    output = tmp_path / "attempt"
    result = module.supervise(tmp_path, output, command=command("pass"))
    assert signal_sent
    assert result == json.loads((output / "supervisor-status.json").read_text())
    assert result["status"] == "INTERRUPTED"
    assert result["interrupt_signal"] == signal.SIGINT
