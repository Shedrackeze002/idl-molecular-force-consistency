"""M6: a full synthetic driver dry run. Each stage runs in its own supervised worker process
(launch_io.supervise, the frozen bounded_process) on the synthetic landscape with fixture arms; the
parent holds the stage ledger as a launch script will. No model is loaded.

Covers the plan's list: the whole A0 -> 0 -> 0b -> 1 -> 2 sequence, a timeout with partial rows kept,
a hard kill, a storage-guard trip, Stage 0 gating after a mismatch, and kill-then-new-launch."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from stage2_capped import admission, driver, launch_io
from stage2_capped.stage_gate import StageLedger, StageOrderError

WORKER = Path(__file__).with_name("dry_run_worker.py")
REPO = Path(__file__).resolve().parents[2]

# rows per stage: models x caps x (2 original starts + 2 reference starts)
EXPECTED_ROWS = {"0": 8, "0b": 4, "1": 16, "2": 8}


@pytest.fixture(autouse=True)
def absolute_environment(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(str(p) for p in (REPO, REPO / "src", REPO / "scripts")))
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")


def real_files():
    try:
        return (admission.group9_root() / admission.DATA_DIRECTORY).is_dir()
    except admission.AdmissionError:
        return False


class Launch:
    """What a launch script does: one root, one ledger, one supervised worker per stage."""

    def __init__(self, root, launch_id):
        self.root = launch_io.fresh_root(root, launch_id)
        self.launch_id, self.ledger, self.runs = launch_id, StageLedger(), 0

    def a0(self):
        self.ledger.start("A0")
        passed = True
        if real_files():          # the no-model admission checks, as launch A0 runs them
            report = admission.admit_all()
            passed = bool(report["replay_plan"]["records"] == 974)
        self.ledger.finish("A0", passed, replay_passed=passed)
        return passed

    def worker(self, stage, *extra, snapshot=None, numerical=240.0, finalization=10.0):
        self.runs += 1
        snapshot = self.ledger.snapshot() if snapshot is None else snapshot
        return launch_io.supervise(
            [sys.executable, str(WORKER), "--root", str(self.root), "--launch-id", self.launch_id,
             "--stage", stage, "--ledger", json.dumps(snapshot), *extra],
            self.root / f"worker-{stage}-{self.runs}.log", numerical, finalization)

    def stage(self, stage, *extra, **limits):
        snapshot = self.ledger.snapshot()                  # taken before the parent starts the stage
        self.ledger.start(stage)
        record = self.worker(stage, *extra, snapshot=snapshot, **limits)
        ok = record["returncode"] == 0 and not record["timed_out"]
        if not ok:                                         # a worker killed hard cannot mark its own stage
            launch_io.mark_incomplete_if_absent(self.root / "rows" / f"stage{stage}",
                                                f"worker did not finish ({record['returncode']})")
        self.ledger.finish(stage, ok)
        return record, ok

    def rows(self, stage):
        return sorted((self.root / "rows" / f"stage{stage}").glob("*/*/*.json"))

    def result(self, stage):
        return json.loads((self.root / f"worker-{stage}.result.json").read_text())


def tree_hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(root).rglob("*")) if p.is_file()}


def complete_launch(root, launch_id):
    launch = Launch(root, launch_id)
    assert launch.a0()
    for stage in ("0", "0b", "1", "2"):
        record, ok = launch.stage(stage)
        assert ok, (stage, record, (launch.root / f"worker-{stage}-{launch.runs}.log").read_text()[-600:])
    return launch


def test_a_complete_launch_runs_every_stage_through_supervised_workers(tmp_path):
    launch = complete_launch(tmp_path / "launch", "dry-complete")
    assert all(state == "complete" for state in launch.ledger.state.values()) and launch.ledger.a0_replay_passed
    for stage, expected in EXPECTED_ROWS.items():
        rows = launch.rows(stage)
        assert len(rows) == expected, stage
        directory = launch.root / "rows" / f"stage{stage}"
        assert (directory / "STAGE_COMPLETE.json").exists() and not (directory / "INCOMPLETE.json").exists()
        assert launch.result(stage)["ok"] is True
        assert launch.result(stage)["summary"]["runs"] == expected
    # The capped stages ran against the controls and wrote one directory per fixed cap label.
    assert {p.parent.name for p in launch.rows("1")} == {"c0p05", "c0p25"}
    assert {p.parent.name for p in launch.rows("0")} == {"cinf"}
    # No partial files are left anywhere.
    assert not list(launch.root.rglob("*.partial"))
    assert launch_io.total_size(launch.root) < launch_io.GIB


def test_a_completed_stage_cannot_be_run_again_in_the_same_root(tmp_path):
    launch = complete_launch(tmp_path / "launch", "dry-twice")
    before = {k: v for k, v in tree_hashes(launch.root / "rows").items()}
    record = launch.worker("0")                                  # the parent's ledger says stage 0 is complete
    assert record["returncode"] != 0 and "StageOrderError" in launch.result("0")["error"]
    assert "runs once" in launch.result("0")["error"]
    # A worker handed a ledger that still shows Stage 0 pending is stopped by the spent stage folder.
    pending = StageLedger()
    pending.start("A0")
    pending.finish("A0", True, replay_passed=True)
    record = launch.worker("0", snapshot=pending.snapshot())
    assert record["returncode"] != 0 and "FileExistsError" in launch.result("0")["error"]
    assert tree_hashes(launch.root / "rows") == before          # nothing was added or changed
    with pytest.raises(StageOrderError, match="runs once"):
        launch.ledger.start("0")


def test_an_overrunning_stage_is_terminated_keeps_its_rows_and_blocks_the_rest(tmp_path):
    launch = Launch(tmp_path / "launch", "dry-timeout")
    launch.a0()
    assert launch.stage("0")[1]
    record, ok = launch.stage("0b", "--stall-after-rows", "2", numerical=14.0, finalization=20.0)
    assert record["timed_out"] and not record["hard_kill"] and not ok
    directory = launch.root / "rows" / "stage0b"
    marker = json.loads((directory / "INCOMPLETE.json").read_text())      # the SIGTERM handler let it finalize
    assert marker["complete"] is False and "DeadlineReached" in marker["reason"]   # the parent did not overwrite it
    assert len(launch.rows("0b")) == len(marker["completed_runs"]) >= 2   # the rows before the stall are kept
    assert not (directory / "STAGE_COMPLETE.json").exists()
    assert launch.ledger.state["0b"] == "failed"
    assert launch.ledger.blockers("2") and any("no retry" in r for r in launch.ledger.blockers("0b"))
    with pytest.raises(StageOrderError):
        launch.ledger.start("2")


def test_a_worker_that_ignores_sigterm_is_killed_and_the_supervisor_marks_the_stage(tmp_path):
    launch = Launch(tmp_path / "launch", "dry-kill")
    launch.a0()
    assert launch.stage("0")[1]
    record, ok = launch.stage("0b", "--stall-after-rows", "2", "--ignore-sigterm", numerical=14.0, finalization=2.0)
    assert record["timed_out"] and record["hard_kill"] and not ok and record["returncode"] != 0
    assert len(launch.rows("0b")) >= 2                                   # completed rows survive a SIGKILL
    marker = json.loads((launch.root / "rows" / "stage0b" / "INCOMPLETE.json").read_text())
    assert marker["reason"].startswith("supervisor: ") and marker["failed_task"] is None   # the parent marked it
    assert len(marker["completed_runs"]) == len(launch.rows("0b"))                        # the finished rows, listed
    assert launch.ledger.state["0b"] == "failed"
    # A worker started by hand afterwards is refused: the control index will not read an INCOMPLETE stage.
    before = tree_hashes(launch.root / "rows")
    again = launch.worker("0b")
    error = launch.result("0b")["error"]
    assert again["returncode"] == 3 and "MissingControl" in error and "INCOMPLETE" in error
    assert tree_hashes(launch.root / "rows") == before


def test_kill_then_new_launch_leaves_the_old_root_untouched(tmp_path):
    first = Launch(tmp_path / "one", "dry-first")
    first.a0()
    assert first.stage("0")[1]
    first.stage("0b", "--stall-after-rows", "2", numerical=14.0, finalization=20.0)
    frozen = tree_hashes(first.root)
    with pytest.raises(FileExistsError):
        launch_io.fresh_root(first.root, "dry-first")                    # the old root is spent
    second = complete_launch(tmp_path / "two", "dry-second")             # a new reviewed launch: a new root
    assert all(state == "complete" for state in second.ledger.state.values())
    assert tree_hashes(first.root) == frozen
    assert first.root != second.root and not (second.root / "rows" / "stage0b" / "INCOMPLETE.json").exists()


def test_the_storage_guard_stops_a_worker_and_the_stage_is_marked(tmp_path):
    launch = Launch(tmp_path / "launch", "dry-storage")
    launch.a0()
    assert launch.stage("0")[1]
    trace = next((launch.root / "rows" / "stage0").glob("*/*/*.npy"))
    atoms = np.load(trace).dtype["positions"].shape[0]
    limit = launch_io.total_size(launch.root) + driver.run_reserve_bytes(atoms) + 2000
    record, ok = launch.stage("0b", "--storage-limit", str(limit))
    assert record["returncode"] == 3 and not ok
    marker = json.loads((launch.root / "rows" / "stage0b" / "INCOMPLETE.json").read_text())
    assert "StorageExceeded" in marker["reason"]
    assert 1 <= len(launch.rows("0b")) < EXPECTED_ROWS["0b"]
    assert launch.ledger.state["0b"] == "failed" and launch.ledger.blockers("2")


def test_a_stage_zero_mismatch_stops_and_the_single_frozen_rerun_finds_a_code_error(tmp_path):
    launch = Launch(tmp_path / "launch", "dry-mismatch")
    launch.a0()
    record, ok = launch.stage("0", "--perturb")
    assert record["returncode"] == 3 and not ok
    directory = launch.root / "rows" / "stage0"
    mismatch = json.loads((directory / "STAGE0_MISMATCH.json").read_text())
    assert mismatch["verdict"]["verdict"] == "frozen re-run required"
    assert len(launch.rows("0")) == mismatch["compared_before"] + 1       # fail fast: nothing after the mismatch
    rerun = json.loads((directory / "STAGE0_RERUN.json").read_text())
    assert rerun["verdict"] == "stop: code error" and rerun["proceed"] is False
    assert (directory / "INCOMPLETE.json").exists()
    assert launch.ledger.state["0"] == "failed"
    for later in ("0b", "1", "2"):
        assert launch.ledger.blockers(later)
        with pytest.raises(StageOrderError):
            launch.ledger.start(later)
