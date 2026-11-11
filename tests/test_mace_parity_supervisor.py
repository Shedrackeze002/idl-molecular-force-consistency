import json
import os
import signal
import subprocess
import sys
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
    calls = {"count": 0}
    def delayed(self, *args, **kwargs):
        if self.name == "supervisor-status.json":
            calls["count"] += 1
            if calls["count"] == 3:
                time.sleep(.06)
        return original(self, *args, **kwargs)
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


@pytest.mark.parametrize("damage", ["command", "executable", "resources", "review", "actual-resource-type", "parity-status.json.tmp", "predictions.npz.tmp.npz", "predictions-partial.npz.tmp.npz"])
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
        if path.name == "supervisor-status.json" and not saw_final_write:
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
