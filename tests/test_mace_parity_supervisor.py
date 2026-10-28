import json
import sys

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
    with pytest.raises(ValueError, match="requires reviewed"):
        module.supervise(tmp_path, tmp_path / "attempt")


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
