import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

import run_mace_recovery_controls as controls
from run_mace_recovery_controls import BRANCHES, REFERENCE_IDS, summarize


def test_recovery_summary_keeps_historical_controls_separate():
    branches = {key: {"category": "nonconverged", "force_evaluation_count": 1500}
                for key in BRANCHES}
    refs = {key: {"operational_force_converged_own_match": True, "force_evaluation_count": 1}
            for key in REFERENCE_IDS}
    outcome = summarize(branches, refs, {"own_geometry_status": "verified", "own_geometry_count": 0})
    assert outcome["original_recovered"] == 0
    assert outcome["reference_own_matches"] == 6
    assert outcome["force_requests"] == 15006
    assert outcome["exceeds_no_relaxation"] is False
    branches.pop(next(iter(branches)))
    with pytest.raises(ValueError, match="schedule"):
        summarize(branches, refs, {"own_geometry_status": "verified", "own_geometry_count": 0})


def test_sigterm_reaps_owned_recovery_worker(tmp_path):
    output = tmp_path / "attempt"
    marker = tmp_path / "pid"
    payload = f"import os,time;open({str(marker)!r},'w').write(str(os.getpid()));time.sleep(30)"
    record = {"commit": "fixture", "output": str(output),
              "command": [sys.executable, "-c", payload]}
    approval = {"status": "APPROVED", "record_sha256": "fixture", "commit": "fixture",
                "reviews": ["engineering", "scientific"]}
    parent = subprocess.Popen([sys.executable, "-c",
        f"from run_mace_recovery_controls import supervise; supervise({record!r},'fixture',{approval!r})"],
        cwd=Path(__file__).resolve().parents[1], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    worker = None
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        worker = int(marker.read_text())
        parent.terminate()
        assert parent.wait(timeout=15) != 0
        terminal = json.loads((output / "terminal.json").read_bytes())
        assert terminal["complete"] is False
        assert terminal["interrupt_signal"] == signal.SIGTERM
        with pytest.raises(ProcessLookupError):
            os.kill(worker, 0)
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait()
        if worker is not None:
            try:
                os.killpg(worker, signal.SIGKILL)
            except ProcessLookupError:
                pass


def test_late_terminal_publication_refuses_completion(tmp_path, monkeypatch):
    output = tmp_path / "attempt"
    result = str(output / "result.json")
    command = [sys.executable, "-c", f"from pathlib import Path;Path({result!r}).write_text('{{}}')"]
    record = {"commit": "fixture", "output": str(output), "command": command}
    approval = {"status": "APPROVED", "record_sha256": "fixture", "commit": "fixture",
                "reviews": ["engineering", "scientific"]}
    clock = [0]
    original = controls.persist

    def delayed(path, value):
        original(path, value)
        if path.name == "terminal.json":
            clock[0] = controls.RESOURCES["wall_seconds"] + 1

    monkeypatch.setattr(controls.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(controls, "persist", delayed)
    with pytest.raises(RuntimeError, match="publication exceeded bounds"):
        controls.supervise(record, "fixture", approval)
    assert (output / "publication-refused.json").is_file()
