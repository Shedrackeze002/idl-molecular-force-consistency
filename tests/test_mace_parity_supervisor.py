import json
import os
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
