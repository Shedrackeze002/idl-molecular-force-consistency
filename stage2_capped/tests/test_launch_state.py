"""Plan v3 sections 3, 4 and 6 with amendments A1 to A7 and B1 to B5: the event log, the registry, the lock, root reuse,
markers and the completion decision, the A0 verdict and evidence, the post-A0 guard, violations and orphaned workers."""
import copy
import json
import os
import pathlib
import subprocess
import sys

import pytest

from stage2_capped import admission, driver, launch_state
from stage2_capped.launch_record import LaunchRefused
from stage2_capped.launch_state import EventLog, Lock, LockHeld, Registry
from stage2_capped.tests.launch_fixtures import (GROUPS, FIVE, REPO, make_a0_result, make_stage_directory,
                                                 needs_real_files)

SHA_A, SHA_B = "a" * 64, "b" * 64
RELAXER = "r" * 64


# ---- Event log ---------------------------------------------------------------------------------------

def test_the_event_log_numbers_events_and_chains_them(tmp_path):
    log = EventLog(tmp_path)
    assert log.files() == [] and log.head_sha256() is None
    first, first_sha = log.append("started", unit="A0")
    second, _ = log.append("exited", outcome="complete")
    assert [p.name for p in log.files()] == ["0001-started.json", "0002-exited.json"]
    assert json.loads(second.read_text())["previous_sha256"] == first_sha
    assert log.problems() == [] and log.head_sha256() is not None


def test_an_edited_or_removed_log_file_breaks_the_chain(tmp_path):
    log = EventLog(tmp_path)
    for name in ("a", "b", "c"):
        log.append(name)
    first = log.files()[0]
    first.write_text(first.read_text().replace('"n": 1', '"n": 1, "extra": 0'))
    assert log.problems() == ["0002-b.json: the chain is broken"]
    log.files()[1].unlink()
    assert any("0003-c.json" in p for p in log.problems())


def test_the_event_log_never_overwrites(tmp_path, monkeypatch):
    log = EventLog(tmp_path)
    log.append("one")
    monkeypatch.setattr(EventLog, "files", lambda self: [])      # a second writer that numbered the same event
    with pytest.raises(FileExistsError):
        log.append("one")
    assert not list(log.directory.glob("*.partial"))


# ---- Registry ----------------------------------------------------------------------------------------

def test_registry_events_are_written_once_with_attempt_numbers(tmp_path):
    registry = Registry(tmp_path / "registry")
    assert registry.next_attempt(SHA_A, "A0") == 1
    registry.write(SHA_A, "A0", 1, "started", launch_id="x")
    assert registry.next_attempt(SHA_A, "A0") == 2
    with pytest.raises(FileExistsError):
        registry.write(SHA_A, "A0", 1, "started", launch_id="x")
    registry.write(SHA_A, "A0", 1, "exited", outcome="complete", spent=True)       # the same attempt, another event
    names = sorted(e["file"] for e in registry.entries())
    assert names == [f"{SHA_A}-A0-1-exited.json", f"{SHA_A}-A0-1-started.json"]
    assert registry.next_attempt(SHA_A, "A0") == 2 and registry.next_attempt(SHA_B, "A0") == 1


def test_the_registry_names_its_event_files_and_says_whether_it_holds_one(tmp_path):
    registry = Registry(tmp_path / "registry")
    assert Registry.event_name(SHA_A, "A0", 1, "started") == f"{SHA_A}-A0-1-started.json"
    assert registry.holds(SHA_A, "A0", 1, "started") is False          # the folder does not exist yet
    registry.write(SHA_A, "A0", 1, "started", launch_id="x")
    assert registry.holds(SHA_A, "A0", 1, "started") is True
    assert registry.holds(SHA_A, "A0", 1, "exited") is False and registry.holds(SHA_A, "A0", 2, "started") is False
    assert registry.holds(SHA_B, "A0", 1, "started") is False
    assert [e["file"] for e in registry.entries()] == [Registry.event_name(SHA_A, "A0", 1, "started")]


@pytest.mark.parametrize("error, answer", [
    (FileNotFoundError(2, "gone"), False), (NotADirectoryError(20, "not a folder"), False),
    (PermissionError(13, "denied"), False),                  # a folder this process may not search took no link from it
    (OSError(5, "input/output error"), True), (OSError(116, "stale handle"), True)])
def test_holds_answers_true_when_the_stat_fails_in_a_way_that_cannot_show_the_file_is_absent(tmp_path, monkeypatch, error, answer):
    """Missing file, missing folder or a folder it may not search: not held. Any other stat failure (EIO, ESTALE): held,
    because a spent unit reported as spent is safe and one reported as a refusal is not (R3 finding 1)."""
    registry = Registry(tmp_path / "registry")

    def stat(self, *arguments, **keywords):
        raise error
    monkeypatch.setattr(pathlib.Path, "stat", stat)
    assert registry.holds(SHA_A, "A0", 1, "started") is answer


def test_the_registry_refuses_unknown_units_and_events(tmp_path):
    registry = Registry(tmp_path)
    with pytest.raises(ValueError):
        registry.write(SHA_A, "9", 1, "started")
    with pytest.raises(ValueError):
        registry.write(SHA_A, "A0", 1, "finished")


def test_started_without_exited_is_spent_and_a_refusal_is_not(tmp_path):
    registry = Registry(tmp_path)
    registry.write(SHA_A, "0", 1, "started")
    assert registry.is_spent(SHA_A, "0") and registry.spent_attempts(SHA_A, "0") == [1]
    registry.write(SHA_A, "0", 1, "exited", outcome="refused_before_start", spent=False)
    assert not registry.is_spent(SHA_A, "0")
    registry.write(SHA_A, "0", 2, "started")
    registry.write(SHA_A, "0", 2, "exited", outcome="failed", spent=True)
    assert registry.spent_attempts(SHA_A, "0") == [2] and registry.is_spent(SHA_A, "0")
    assert not registry.is_spent(SHA_A, "1") and not registry.is_spent(SHA_B, "0")


def test_reconciliation_only_confirms_spent_and_never_writes_a_refusal(tmp_path):
    registry = Registry(tmp_path)
    registry.write(SHA_A, "A0", 1, "started")
    written = registry.confirm_spent(SHA_A, "A0", {"root_holds": "nothing"})
    assert len(written) == 1
    exited = registry.attempts(SHA_A, "A0")[1]["exited"]["payload"]
    assert exited["spent"] is True and exited["outcome"] == "confirmed_spent_after_missing_exit"
    assert exited["evidence"] == {"root_holds": "nothing"}
    assert registry.confirm_spent(SHA_A, "A0", {}) == []                    # nothing left to confirm
    assert registry.is_spent(SHA_A, "A0")


def test_prior_dispatches_must_list_every_event_outside_the_pair(tmp_path):
    registry = Registry(tmp_path)
    own = registry.write(SHA_A, "A0", 1, "started")
    other = registry.write("c" * 64, "A0", 1, "started")
    pair = {SHA_A, SHA_B}
    assert registry.prior_problems(pair, [[other["file"], other["sha256"]]]) == []
    assert registry.prior_problems(pair, []) == [f"registry event {other['file']} is not listed in prior_dispatches"]
    listed_own = registry.prior_problems(pair, [[other["file"], other["sha256"]], [own["file"], own["sha256"]]])
    assert listed_own == [f"prior_dispatches lists {own['file']}, which the registry does not hold"]
    assert registry.prior_problems(pair, [[other["file"], other["sha256"]], ["c" * 64 + "-0-1-started.json", "0" * 64]])
    assert registry.prior_problems(pair, [[other["file"], "0" * 64]])           # a changed event file is not the listed one


def test_an_audit_violation_exit_in_the_registry_voids_the_pair_even_without_a_violation_file(tmp_path):
    registry = Registry(tmp_path)
    registry.write(SHA_A, "A0", 1, "started")
    registry.write(SHA_A, "A0", 1, "exited", outcome="audit_violation", spent=True)
    registry.write("c" * 64, "A0", 1, "started")
    registry.write("c" * 64, "A0", 1, "exited", outcome="audit_violation", spent=True)      # another pair's
    registry.write(SHA_B, "0", 1, "started")
    registry.write(SHA_B, "0", 1, "exited", outcome="failed", spent=True)                   # failed, not a violation
    assert registry.violation_events({SHA_A, SHA_B}) == [f"{SHA_A}-A0-1-exited.json"]
    assert registry.violation_events({"d" * 64}) == []


def test_a_deleted_registry_entry_is_found(tmp_path):
    registry = Registry(tmp_path)
    other = registry.write("c" * 64, "1", 1, "started")
    (tmp_path / other["file"]).unlink()
    assert registry.prior_problems({SHA_A, SHA_B}, [[other["file"], other["sha256"]]])


@pytest.mark.parametrize("events, expected", [
    ([], "0 spent"),
    ([(1, "started"), (1, "exited", "complete", True), (2, "started"), (2, "exited", "failed", True)], "2 spent"),
    ([(1, "started"), (1, "exited", "failed", True)], "did not end complete"),
    ([(1, "started"), (1, "exited", "refused_before_start", False), (2, "started"), (2, "exited", "complete", True)], None),
    ([(1, "started")], "did not end complete"),
])
def test_exactly_one_spent_a0_attempt_that_ended_with_a_result(tmp_path, events, expected):
    registry = Registry(tmp_path)
    for item in events:
        if item[1] == "started":
            registry.write(SHA_A, "A0", item[0], "started")
        else:
            registry.write(SHA_A, "A0", item[0], "exited", outcome=item[2], spent=item[3])
    problems = registry.a0_problems(SHA_A)
    assert (problems == []) if expected is None else any(expected in p for p in problems)


# ---- Lock ------------------------------------------------------------------------------------------------

def test_a_second_lock_on_the_same_root_is_refused_and_released_after_use(tmp_path):
    with Lock(tmp_path):
        with pytest.raises(LockHeld):
            with Lock(tmp_path):
                pass
    with Lock(tmp_path):
        pass
    assert (tmp_path / launch_state.LOCK_NAME).exists()


def test_the_lock_is_not_inherited_by_a_child_process(tmp_path):
    with Lock(tmp_path):
        done = subprocess.run([sys.executable, "-c", "import time; time.sleep(0.2)"])
        assert done.returncode == 0
    probe = subprocess.run(
        [sys.executable, "-c",
         "import fcntl,os,sys; fd=os.open(sys.argv[1],os.O_RDWR); fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB); print('free')",
         str(tmp_path / launch_state.LOCK_NAME)], capture_output=True)
    assert probe.stdout.strip() == b"free"


# ---- Root contents per unit (A1, B1) -------------------------------------------------------------------------

def touch(root, *relative):
    for name in relative:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")


FIRST_ROOT = ["LAUNCH.json", "LAUNCH_LOG/0001-started.json", ".launch.lock", "A0_REFUSED-1.json", "logs/worker-1.log",
              "worker-0.attempt-1.result.json", "WORKER_PID-0-1.json"]


@pytest.mark.parametrize("unit", ["A0", "0"])
def test_the_first_command_of_a_root_may_find_only_launch_and_attempt_files(tmp_path, unit):
    touch(tmp_path, *FIRST_ROOT)
    assert launch_state.root_contents_problems(tmp_path, unit) == []
    for stray in ("rows/stage0/STAGE_COMPLETE.json", "A0_RESULT.json", "A0_RESTORE_BEGUN.json", "notes.txt",
                  "worker-0.result.json", "STAGE0_RERUN.json"):
        touch(tmp_path, stray)
        assert launch_state.root_contents_problems(tmp_path, unit), stray
        (tmp_path / stray).unlink()


def test_a_later_stage_may_find_the_other_stages_but_not_its_own_artifacts(tmp_path):
    touch(tmp_path, *FIRST_ROOT, "rows/stage0/STAGE_COMPLETE.json", "rows/stage0/model0/cinf/row.json",
          "worker-0.result.json", "STAGE0_RERUN_ATTEMPTED.json")
    assert launch_state.root_contents_problems(tmp_path, "0b") == []
    assert launch_state.root_contents_problems(tmp_path, "1") == []
    assert launch_state.root_contents_problems(tmp_path, "0")                     # stage 0's own tree is present
    touch(tmp_path, "rows/stage0b/STAGE_COMPLETE.json", "worker-0b.result.json")
    assert launch_state.root_contents_problems(tmp_path, "1") == []
    own = launch_state.root_contents_problems(tmp_path, "0b")
    assert len(own) == 2 and any("stage0b" in p for p in own) and any("worker-0b.result.json" in p for p in own)


def test_a_stage_0b_restart_after_a_refusal_is_allowed_and_a_stage_0_tree_is_not_confused_with_0b(tmp_path):
    touch(tmp_path, *FIRST_ROOT, "rows/stage0/STAGE_COMPLETE.json", "worker-0b.attempt-1.result.json",
          "logs/worker-0b-1.log")
    assert launch_state.root_contents_problems(tmp_path, "0b") == []
    assert launch_state.root_contents_problems(tmp_path, "0")
    touch(tmp_path, "rows/stage0b/x.json")
    assert launch_state.root_contents_problems(tmp_path, "0b")


def test_an_unknown_file_blocks_a_later_stage_too(tmp_path):
    touch(tmp_path, *FIRST_ROOT, "rows/stage0/STAGE_COMPLETE.json", "extra/anything.json")
    assert any("extra/anything.json" in p for p in launch_state.root_contents_problems(tmp_path, "1"))


def test_the_spent_signal_is_the_same_test_for_parent_and_worker(tmp_path):
    assert not launch_state.spent_signal(tmp_path, "A0") and not launch_state.spent_signal(tmp_path, "1")
    touch(tmp_path, "A0_REFUSED-1.json", "worker-1.attempt-1.result.json")
    assert not launch_state.spent_signal(tmp_path, "A0") and not launch_state.spent_signal(tmp_path, "1")
    touch(tmp_path, "A0_RESTORE_BEGUN.json")
    assert launch_state.spent_signal(tmp_path, "A0")
    (tmp_path / "rows" / "stage1").mkdir(parents=True)
    assert launch_state.spent_signal(tmp_path, "1") and not launch_state.spent_signal(tmp_path, "2")
    other = tmp_path / "second"
    other.mkdir()
    touch(other, "A0_RESULT.json")
    assert launch_state.spent_signal(other, "A0")


# ---- Markers and the completion decision -----------------------------------------------------------------------------

RECORD = {"saved_file_hashes": {f"path{i}": f"{i:064x}" for i in range(110)}}


def valid_stage(tmp_path, stage, **kwargs):
    return make_stage_directory(tmp_path, stage, RECORD, "L-1", RELAXER, **kwargs)


@pytest.mark.parametrize("stage", ["0", "0b", "1", "2"])
def test_a_valid_stage_directory_has_no_marker_problems(tmp_path, stage):
    _, sha = valid_stage(tmp_path, stage)
    problems, found = launch_state.marker_problems(tmp_path, stage, RECORD, "L-1", RELAXER)
    assert problems == [] and found == sha


@pytest.mark.parametrize("kwargs, message", [
    ({"complete": False}, "does not say complete"),
    ({"custom": True}, "custom task list"),
    ({"runs": 79}, "row files on disk"),
    ({"force_evaluations": 120001}, "above the attempt ceiling"),
    ({"marker_overrides": {"runs": 79}}, "records 79 runs"),
    ({"marker_overrides": {"stage": "0b"}}, "different stage"),
    ({"marker_overrides": {"manifest_sha256": "0" * 64}}, "manifest_sha256"),
    ({"marker_overrides": {"force_evaluations": None}}, "force_evaluations"),
    ({"marker_overrides": {"complete": None}}, "does not say complete"),            # a missing key is not "complete"
    ({"marker_overrides": {"custom_tasks": None}}, "custom task list"),             # nor "not custom"
    ({"manifest_overrides": {"stage": "0b"}}, "different launch or stage"),
    ({"manifest_overrides": {"launch_id": "other"}}, "different launch"),
    ({"manifest_overrides": {"relaxer_sha256": "z" * 64}}, "different relaxer"),
    ({"manifest_overrides": {"stage0": {"verdict": {"proceed": False}, "compared": 80,
                                        "saved_file_hashes": RECORD["saved_file_hashes"]}}}, "does not proceed"),
    ({"manifest_overrides": {"stage0": {"verdict": {"proceed": True}, "compared": 79,
                                        "saved_file_hashes": RECORD["saved_file_hashes"]}}}, "does not proceed"),
    ({"manifest_overrides": {"stage0": {"verdict": {"proceed": True}, "compared": 80, "saved_file_hashes": {}}}},
     "saved-file hashes"),
])
def test_each_broken_stage_0_marker_or_manifest_is_a_problem(tmp_path, kwargs, message):
    valid_stage(tmp_path, "0", **kwargs)
    problems, _ = launch_state.marker_problems(tmp_path, "0", RECORD, "L-1", RELAXER)
    assert any(message in p for p in problems), problems


def test_a_marker_whose_force_evaluations_equal_the_attempt_ceiling_is_valid(tmp_path):
    valid_stage(tmp_path, "0", force_evaluations=120000)
    assert launch_state.marker_problems(tmp_path, "0", RECORD, "L-1", RELAXER)[0] == []


def test_an_incomplete_marker_beside_a_complete_one_is_a_problem(tmp_path):
    directory, _ = valid_stage(tmp_path, "0b")
    (directory / "INCOMPLETE.json").write_text("{}")
    problems, _ = launch_state.marker_problems(tmp_path, "0b", RECORD, "L-1", RELAXER)
    assert problems == ["INCOMPLETE.json exists"]


def test_a_missing_marker_or_manifest_or_junk_is_a_problem(tmp_path):
    directory, _ = valid_stage(tmp_path, "0b")
    (directory / "STAGE_COMPLETE.json").unlink()
    assert launch_state.marker_problems(tmp_path, "0b", RECORD, "L-1", RELAXER)[0]
    valid_stage(tmp_path / "second", "0b")
    (tmp_path / "second" / "rows" / "stage0b" / "STAGE_MANIFEST.json").write_text("not json")
    problems, _ = launch_state.marker_problems(tmp_path / "second", "0b", RECORD, "L-1", RELAXER)
    assert any("not JSON" in p or "manifest_sha256" in p for p in problems)


def decision(tmp_path, exit_record=None, audit=None, stage="0b"):
    audit = {"violations": 0, "opened_in_data_directory": ["processed/aspirin/train.npz"]} if audit is None else audit
    return launch_state.completion_problems(exit_record or {"returncode": 0, "timed_out": False}, tmp_path, stage, RECORD,
                                            "L-1", RELAXER, audit, ["processed/aspirin/train.npz"],
                                            ["processed/aspirin/train.npz", "Landscape17.zip"])


def test_a_clean_exit_with_a_valid_marker_and_audit_is_complete(tmp_path):
    valid_stage(tmp_path, "0b")
    assert decision(tmp_path) == []


@pytest.mark.parametrize("exit_record, message", [
    ({"returncode": 3, "timed_out": False}, "exited with code 3"),
    ({"returncode": 0, "timed_out": True}, "wall ceiling"),
    ({"returncode": -9, "timed_out": True}, "wall ceiling"),
])
def test_a_nonzero_exit_or_a_timeout_fails_even_with_a_valid_marker(tmp_path, exit_record, message):
    valid_stage(tmp_path, "0b")
    assert any(message in p for p in decision(tmp_path, exit_record))


def test_exit_zero_without_a_marker_fails(tmp_path):
    (tmp_path / "rows" / "stage0b").mkdir(parents=True)
    assert any("missing" in p for p in decision(tmp_path))


def test_an_audit_violation_or_a_missing_audit_log_fails_the_stage(tmp_path):
    valid_stage(tmp_path, "0b")
    assert decision(tmp_path, audit={"violations": 1, "opened_in_data_directory": ["processed/aspirin/train.npz"]})
    assert decision(tmp_path, audit={"violations": 0, "opened_in_data_directory": ["processed/aspirin/other.npz"]})
    assert decision(tmp_path, audit={})
    assert decision(tmp_path, audit={"violations": 0, "opened_in_data_directory": []})      # a hook that saw nothing (N04)


# ---- The ledger derived from evidence (A7) ------------------------------------------------------------------------

def record_for(launch_id="L-1"):
    return {**RECORD, "launch_id": launch_id}


REGISTRY_SHA = "d" * 64


def ledger_of(tmp_path, typed, *, a0_valid=True, complete=(), registry=None):
    """derive_ledger with a registry that holds a spent, complete dispatch for each stage in `complete`."""
    registry = registry or Registry(tmp_path / "ledger-registry")
    for stage in complete:
        registry.write(REGISTRY_SHA, stage, 1, "started")
        registry.write(REGISTRY_SHA, stage, 1, "exited", outcome="complete", spent=True)
    return launch_state.derive_ledger(tmp_path, record_for(), typed, a0_valid=a0_valid, relaxer_sha256=RELAXER,
                                      registry=registry, record_sha256=REGISTRY_SHA)


def test_the_ledger_is_rebuilt_from_markers_and_typed_hashes(tmp_path):
    _, sha0 = valid_stage(tmp_path, "0")
    ledger, notes = ledger_of(tmp_path, {"0": sha0}, complete=["0"])
    assert ledger.state == {"A0": "complete", "0": "complete", "0b": "pending", "1": "pending", "2": "pending"}
    assert ledger.a0_replay_passed and notes == {}
    assert ledger.blockers("0b") == [] and ledger.blockers("1") == [] and ledger.blockers("2")


def test_without_valid_a0_evidence_nothing_can_start(tmp_path):
    ledger, _ = ledger_of(tmp_path, {}, a0_valid=False)
    assert ledger.state["A0"] == "pending" and not ledger.a0_replay_passed and ledger.blockers("0")


def test_a_valid_marker_with_no_typed_hash_is_untyped_and_satisfies_nothing(tmp_path):
    valid_stage(tmp_path, "0")
    ledger, notes = ledger_of(tmp_path, {}, complete=["0"])
    assert ledger.state["0"] == "failed" and "untyped" in notes["0"][0]
    assert ledger.blockers("0") and ledger.blockers("1")                 # blocks its own start; satisfies no prerequisite


def test_a_typed_hash_that_differs_from_the_marker_is_refused(tmp_path):
    valid_stage(tmp_path, "0")
    with pytest.raises(LaunchRefused, match="typed hash for stage 0"):
        ledger_of(tmp_path, {"0": "0" * 64}, complete=["0"])


def test_a_hash_typed_for_a_stage_with_no_directory_is_refused(tmp_path):
    with pytest.raises(LaunchRefused, match="no directory"):
        ledger_of(tmp_path, {"1": "0" * 64})


def test_a_stage_directory_without_a_valid_marker_or_with_incomplete_is_failed(tmp_path):
    (tmp_path / "rows" / "stage1").mkdir(parents=True)
    directory, _ = valid_stage(tmp_path, "0b")
    (directory / "INCOMPLETE.json").write_text("{}")
    ledger, notes = ledger_of(tmp_path, {}, complete=["0b"])
    assert ledger.state["1"] == "failed" and ledger.state["0b"] == "failed" and set(notes) == {"1", "0b"}


def test_incomplete_wins_over_a_marker(tmp_path):
    directory, sha = valid_stage(tmp_path, "0")
    (directory / "INCOMPLETE.json").write_text("{}")
    ledger, _ = ledger_of(tmp_path, {"0": sha}, complete=["0"])
    assert ledger.state["0"] == "failed"


@pytest.mark.parametrize("events, expected", [
    ([], "no dispatch of unit 0"),
    ([(1, "started"), (1, "exited", "complete", True)], None),
    ([(1, "started"), (1, "exited", "failed", True)], "as failed, not complete"),
    ([(1, "started"), (1, "exited", "audit_violation", True)], "as audit_violation"),
    ([(1, "started"), (1, "exited", "confirmed_spent_after_missing_exit", True)], "as confirmed_spent_after_missing_exit"),
    ([(1, "started")], "started, with no exit seen"),
    ([(1, "started"), (1, "exited", "refused_before_start", False)], "no dispatch of unit 0"),
    ([(1, "started"), (1, "exited", "refused_before_start", False), (2, "started"), (2, "exited", "complete", True)], None),
])
def test_the_registry_says_whether_a_unit_completed_under_a_record(tmp_path, events, expected):
    registry = Registry(tmp_path)
    for item in events:
        if item[1] == "started":
            registry.write(SHA_A, "0", item[0], "started")
        else:
            registry.write(SHA_A, "0", item[0], "exited", outcome=item[2], spent=item[3])
    problems = registry.unit_problems(SHA_A, "0")
    assert (problems == []) if expected is None else (len(problems) >= 1 and any(expected in p for p in problems)), problems
    assert Registry(tmp_path).unit_problems("e" * 64, "0")                    # another record's dispatches are not this record's


@pytest.mark.parametrize("outcome, spent", [("failed", True), ("audit_violation", True),
                                            ("confirmed_spent_after_missing_exit", True)])
def test_a_stage_the_parent_called_failed_cannot_satisfy_a_prerequisite_whatever_its_files_say(tmp_path, outcome, spent):
    _, sha = valid_stage(tmp_path, "0")                                       # a valid marker, typed correctly
    registry = Registry(tmp_path / "failed-registry")
    registry.write(REGISTRY_SHA, "0", 1, "started")
    registry.write(REGISTRY_SHA, "0", 1, "exited", outcome=outcome, spent=spent)
    ledger, notes = ledger_of(tmp_path, {"0": sha}, registry=registry)
    assert ledger.state["0"] == "failed" and outcome in notes["0"][0]
    assert ledger.blockers("0b") and ledger.blockers("1")                      # it satisfies no prerequisite


def test_a_complete_looking_stage_with_no_registry_record_is_not_complete(tmp_path):
    _, sha = valid_stage(tmp_path, "0")
    ledger, notes = ledger_of(tmp_path, {"0": sha})                            # a hand-made directory: nobody dispatched it
    assert ledger.state["0"] == "failed" and "no dispatch" in notes["0"][0]


def test_a_started_stage_with_no_exit_in_the_registry_is_failed_even_with_a_marker(tmp_path):
    _, sha = valid_stage(tmp_path, "0")
    registry = Registry(tmp_path / "orphan-registry")
    registry.write(REGISTRY_SHA, "0", 1, "started")
    ledger, notes = ledger_of(tmp_path, {"0": sha}, registry=registry)
    assert ledger.state["0"] == "failed" and "no exit seen" in notes["0"][0]


# ---- The A0 verdict (B5, A6) ------------------------------------------------------------------------------------------

def test_a_complete_valid_a0_result_passes():
    passed, reasons = launch_state.a0_verdict(make_a0_result())
    assert passed and reasons == []


def mutate(change):
    result = make_a0_result()
    change(result)
    return launch_state.a0_verdict(result)


@pytest.mark.parametrize("name, change, message", [
    ("missing model", lambda r: r["models"].pop("direct-0"), "direct-0: no replay return"),
    ("failed model", lambda r: r["models"]["direct-0"].update(replay=None, failure="replay: boom"), "no replay return"),
    ("status", lambda r: r["models"]["energy-0"]["replay"].update(status="failed"), "status 'failed'"),
    ("records", lambda r: r["models"]["energy-0"]["replay"].update(record_count=973), "record_count"),
    ("rtol", lambda r: r["models"]["energy-0"]["replay"].update(rtol=1e-9), "tolerances"),
    ("atol", lambda r: r["models"]["energy-0"]["replay"].update(atol=1e-9), "tolerances"),
    ("single ids", lambda r: r["models"]["energy-0"]["replay"].update(single_record_ids=FIVE[::-1]), "first five"),
    ("no scope ids", lambda r: r["replay_scope"].update(first_five_record_ids=[]), "first five"),
    ("nan", lambda r: r["models"]["energy-0"]["replay"].update(maximum_absolute_force_difference=float("nan")), "not finite"),
    ("inf", lambda r: r["models"]["energy-0"]["replay"].update(maximum_absolute_force_difference=float("inf")), "not finite"),
    ("bool", lambda r: r["models"]["energy-0"]["replay"].update(maximum_absolute_force_difference=True), "not finite"),
    ("groups", lambda r: r["replay_scope"].update(groups=["13", "18"]), "five frozen connection groups"),
    ("A6 maximum", lambda r: r["models"]["energy-0"]["replay"].update(maximum_absolute_force_difference=5e-8), "exceeds atol"),
    ("A6 maximum just above atol + rtol x max", lambda r: r["models"]["energy-0"]["replay"].update(
        maximum_absolute_force_difference=4.2e-9), "exceeds atol"),                   # 1e-10 + 1e-10 x 40 = 4.1e-9
    ("A6 residual", lambda r: r["models"]["energy-0"]["evidence"]["batch4"].update(max_scaled_residual=1.5), "contradicts"),
    ("A6 residual NaN", lambda r: r["models"]["energy-0"]["evidence"]["batch4"].update(max_scaled_residual=float("nan")), "contradicts"),
    ("no single ids on either side", lambda r: ([m["replay"].pop("single_record_ids") for m in r["models"].values()],
                                                r["replay_scope"].pop("first_five_record_ids")), "first five"),
    ("six groups with a duplicate", lambda r: r["replay_scope"].update(groups=["13", "13", "18", "18_2", "66", "78"]),
     "five frozen connection groups"),
])
def test_each_broken_a0_entry_fails_the_verdict(name, change, message):
    passed, reasons = mutate(change)
    assert not passed and any(message in r for r in reasons), reasons


@pytest.mark.parametrize("field, value, message", [
    ("checkpoint_sha256", "9" * 64, "checkpoint_sha256"),
    ("score_sha256", "9" * 64, "score_sha256"),
    ("best_update", 7, "best_update"),
    ("arm", "other", "arm or seed"),
    ("seed", 9, "arm or seed"),
    ("dataset_identity", "other", "dataset_identity"),
    ("source_fingerprint", "other", "source_fingerprint"),
])
def test_one_mismatching_identity_field_fails_the_entry(field, value, message):
    passed, reasons = mutate(lambda r: r["models"]["consistency-0.1-0"]["replay"]["checkpoint_identity"].update({field: value}))
    assert not passed and any(message in x and x.startswith("consistency-0.1-0") for x in reasons), reasons


def test_a_difference_inside_atol_plus_rtol_times_the_largest_saved_force_passes():
    """A frozen replay that np.allclose accepts must not fail A0 (a false FAIL spends the record pair): with forces near 40
    the allowed maximum is 1e-10 + 1e-10 x 40, not 1e-10; the residual of exactly 1.0 is inside the tolerance too."""
    passed, reasons = mutate(lambda r: r["models"]["energy-0"]["replay"].update(maximum_absolute_force_difference=2e-9))
    assert passed, reasons
    passed, reasons = mutate(lambda r: (r["models"]["energy-0"]["replay"].update(maximum_absolute_force_difference=4.1e-9),
                                        r["models"]["energy-0"]["evidence"]["batch4"].update(max_scaled_residual=1.0)))
    assert passed, reasons


@pytest.mark.parametrize("field", ["checkpoint_sha256", "score_sha256", "best_update", "dataset_identity", "source_fingerprint"])
def test_a_field_missing_on_both_sides_is_not_a_match(field):
    """N10: `None != None` is False, so a damaged result with the field absent on both sides must still fail."""
    accepted_key = {"checkpoint_sha256": "best_msgpack_sha256", "score_sha256": "score_json_sha256", "best_update": "best_update"}

    def damage(result):
        result["models"]["energy-0"]["replay"]["checkpoint_identity"].pop(field)
        if field in accepted_key:
            result["admission"]["identities"]["energy-0"].pop(accepted_key[field])
        elif field == "dataset_identity":
            result["admission"]["dataset_pin"].pop("manifest_identity")
        else:
            result["admission"]["accepted_pin"].pop("code_fingerprint")
    passed, reasons = mutate(damage)
    assert not passed and any(field in r for r in reasons), reasons


def test_lambda_is_compared_for_consistency_arms_only():
    passed, reasons = mutate(lambda r: r["models"]["consistency-1-1"]["replay"]["checkpoint_identity"].update({"lambda": 0.5}))
    assert not passed and any("lambda" in x for x in reasons)
    passed, _ = mutate(lambda r: r["models"]["direct-0"]["replay"]["checkpoint_identity"].update({"lambda": 0.0}))
    assert passed                                                       # a direct arm's stored lambda is not pinned


def test_one_failing_model_fails_all_and_every_reason_is_listed():
    result = make_a0_result()
    result["models"]["energy-0"]["replay"]["status"] = "failed"
    result["models"]["consistency-1-1"]["replay"]["record_count"] = 1
    passed, reasons = launch_state.a0_verdict(result)
    assert not passed and len(reasons) == 2


# ---- A0 evidence for record B ------------------------------------------------------------------------------------------

def a0_setup(tmp_path, **overrides):
    root = tmp_path / "a0-root"
    root.mkdir()
    result = make_a0_result(record_a_sha256=SHA_A, launch_id="A0-launch", output_root=str(root), commit="c" * 40,
                            launch_records_sha256="f" * 64, runtime={"jax_backend": "cpu", "k": 1})
    result.update(overrides)
    data = launch_state.dumps(result)
    (root / "A0_RESULT.json").write_bytes(data)
    record_b = {"a0_output_root": str(root), "record_a_sha256": SHA_A, "a0_launch_id": "A0-launch", "commit": "c" * 40,
                "expected_launch_records_sha256": "f" * 64}
    registry = Registry(tmp_path / "registry")
    registry.write(SHA_A, "A0", 1, "started")
    registry.write(SHA_A, "A0", 1, "exited", outcome="complete", spent=True)
    from stage2_capped.launch_record import sha256_bytes
    return record_b, sha256_bytes(data), registry


def test_a_later_record_b_may_reuse_the_same_a0_when_record_a_and_the_commit_are_the_same(tmp_path):
    record_b, sha, registry = a0_setup(tmp_path)
    later = {**record_b, "ceilings": "changed in a later record B", "prior_dispatches": []}
    result, problems = launch_state.a0_evidence(later, sha, registry, current_runtime={"k": 1})
    assert problems == [] and result is not None


def test_a_later_record_b_naming_another_record_a_or_commit_cannot_reuse_it(tmp_path):
    record_b, sha, registry = a0_setup(tmp_path)
    for change in ({"record_a_sha256": "e" * 64}, {"commit": "e" * 40}):
        _, problems = launch_state.a0_evidence({**record_b, **change}, sha, registry, current_runtime={"k": 1})
        assert problems


def test_valid_a0_evidence_has_no_problems(tmp_path):
    record_b, sha, registry = a0_setup(tmp_path)
    result, problems = launch_state.a0_evidence(record_b, sha, registry, current_runtime={"k": 1})
    assert problems == [] and result["launch_id"] == "A0-launch"


@pytest.mark.parametrize("change, message", [
    ({"record_sha256": "d" * 64}, "record_sha256"),
    ({"launch_id": "other"}, "launch_id"),
    ({"output_root": "/elsewhere"}, "output_root"),
    ({"commit": "e" * 40}, "commit"),
])
def test_a0_evidence_from_another_record_root_launch_or_commit_is_refused(tmp_path, change, message):
    record_b, sha, registry = a0_setup(tmp_path, **change)
    _, problems = launch_state.a0_evidence(record_b, sha, registry, current_runtime={"k": 1})
    assert any(message in p for p in problems)


def test_a0_evidence_needs_the_typed_hash_the_registry_and_the_runtime(tmp_path):
    record_b, sha, registry = a0_setup(tmp_path)
    assert any("different SHA-256" in p for p in launch_state.a0_evidence(
        record_b, "0" * 64, registry, current_runtime={"k": 1})[1])
    assert any("runtime k" in p for p in launch_state.a0_evidence(record_b, sha, registry, current_runtime={"k": 2})[1])
    empty = Registry(tmp_path / "empty-registry")
    assert any("0 spent A0 attempts" in p for p in launch_state.a0_evidence(
        record_b, sha, empty, current_runtime={"k": 1})[1])
    assert any("does not exist" in p for p in launch_state.a0_evidence(
        {**record_b, "a0_output_root": str(tmp_path / "none")}, sha, registry, current_runtime={"k": 1})[1])


def test_a0_evidence_with_the_wrong_launch_records_or_a_failed_verdict_is_refused(tmp_path):
    record_b, sha, registry = a0_setup(tmp_path)
    assert any("launch records" in p for p in launch_state.a0_evidence(
        {**record_b, "expected_launch_records_sha256": "0" * 64}, sha, registry, current_runtime={"k": 1})[1])
    broken = make_a0_result(record_a_sha256=SHA_A, launch_id="A0-launch", output_root=record_b["a0_output_root"],
                            commit="c" * 40, launch_records_sha256="f" * 64, runtime={"jax_backend": "cpu", "k": 1})
    broken["models"]["energy-0"]["replay"]["status"] = "failed"
    from stage2_capped.launch_record import sha256_bytes
    data = launch_state.dumps(broken)
    path = tmp_path / "a0-root" / "A0_RESULT.json"
    path.unlink()
    path.write_bytes(data)
    _, problems = launch_state.a0_evidence(record_b, sha256_bytes(data), registry, current_runtime={"k": 1})
    assert any(p.startswith("A0 verdict:") for p in problems)


# ---- The post-A0 guard (A3, R2) -----------------------------------------------------------------------------------------

def test_the_guard_passes_at_the_historical_latency_and_uses_the_largest_mean(tmp_path):
    result = make_a0_result(latency=0.001)
    for stage, spec in driver.STAGE_SPECS.items():
        assert launch_state.guard_problems(result, stage, spec["wall_seconds"]) == [], stage
    result["models"]["direct-0"]["evidence"]["batch1_latency"]["mean_seconds"] = 0.01
    problems = launch_state.guard_problems(result, "1", 2100.0)
    assert len(problems) == 1 and "mean latency exceeds" in problems[0]
    assert launch_state.guard_problems(result, "0b", 300.0) == []        # direct-0 is not a model of stage 0b


def test_the_guard_uses_the_mean_and_the_p99_is_only_a_flag():
    result = make_a0_result(latency=0.001)
    for model in result["models"].values():
        model["evidence"]["batch1_latency"]["p99_seconds"] = 5.0       # a jittery tail
    assert launch_state.guard_problems(result, "1", 2100.0) == []
    flags = launch_state.guard_flags(result, "1", 2100.0)
    assert len(flags) == 1 and flags[0].startswith("flag:")
    assert launch_state.guard_flags(make_a0_result(latency=0.0005), "1", 2100.0) == []


@pytest.mark.parametrize("value", [None, 0, -1.0, float("nan"), float("inf"), True, "fast"])
def test_a_missing_or_unusable_latency_for_a_model_of_the_stage_is_a_refusal(value):
    result = make_a0_result()
    result["models"]["energy-0"]["evidence"]["batch1_latency"]["mean_seconds"] = value
    assert launch_state.guard_problems(result, "0", 600.0) == ["no usable batch-1 mean latency for energy-0 in the A0 result"]
    assert launch_state.guard_problems(result, "0b", 300.0) == []        # energy-0 is not used by stage 0b


def test_the_guard_summary_names_the_stage_its_models_the_latency_used_the_ceiling_and_the_verdict():
    result = make_a0_result(latency=0.001)
    result["models"]["direct-0"]["evidence"]["batch1_latency"]["mean_seconds"] = 0.002
    assert launch_state.guard_summary(result, "0", 600.0) == (
        "post-A0 guard, stage 0: models energy-0, direct-0, consistency-0.1-0, direct-1, consistency-0.1-1; "
        "mean batch-1 latency used 0.002 s (direct-0); ceiling 600 s; OK")
    assert launch_state.guard_summary(result, "0b", 300.0) == (
        "post-A0 guard, stage 0b: models consistency-1-0, consistency-1-1; "
        "mean batch-1 latency used 0.001 s (consistency-1-0); ceiling 300 s; OK")
    above = launch_state.guard_summary(result, "0", 1800.0)           # a ceiling above the spec's floor is the one printed
    assert above.endswith("ceiling 1800 s; OK") and "ceiling 600 s" not in above
    slow = launch_state.guard_summary(make_a0_result(latency=0.01), "1", 2100.0)
    assert slow.startswith("post-A0 guard, stage 1: models energy-0, direct-0,") and "ceiling 2100 s; REFUSED: stage 1: worst case " in slow
    assert "mean batch-1 latency used 0.01 s (energy-0)" in slow


def test_the_guard_summary_shows_a_missing_latency_as_a_refusal_that_names_the_model():
    result = make_a0_result()
    result["models"]["direct-1"]["evidence"] = {"error": "KeyError: the saved forces cannot be read"}
    assert launch_state.guard_summary(result, "0", 600.0) == (
        "post-A0 guard, stage 0: models energy-0, direct-0, consistency-0.1-0, direct-1, consistency-0.1-1; "
        "mean batch-1 latency used none (no usable mean for direct-1); ceiling 600 s; "
        "REFUSED: no usable batch-1 mean latency for direct-1 in the A0 result")
    assert launch_state.guard_summary(result, "2", 900.0).endswith("ceiling 900 s; OK")      # direct-1 is not a model of stage 2


def test_the_guard_formula_counts_runs_attempts_latency_twice_and_setup():
    result = make_a0_result(latency=0.0005)
    needed = 80 * 1500 * 0.0005 * 2 + 6.0 * 5
    assert launch_state.guard_problems(result, "0", needed + 0.001) == []
    assert launch_state.guard_problems(result, "0", needed - 0.001)
    assert launch_state.guard_problems(result, "0", needed) == []          # a worst case equal to the ceiling is not above it


# ---- The A0 completion decision ------------------------------------------------------------------------------------------

OK_EXIT, REQUIRED_FILES, ALLOWED_FILES = {"returncode": 0, "timed_out": False}, ["a", "b"], ["a", "b", "c"]


def a0_result_root(tmp_path, name, **fields):
    result = make_a0_result()
    result.update({"audit": {"violations": 0, "opened_in_data_directory": ["a", "b"]}, "fatal": None,
                   "module_problems_at_exit": []})
    result.update(fields)
    root = tmp_path / name
    root.mkdir()
    (root / "A0_RESULT.json").write_bytes(launch_state.dumps(result))
    return root


def test_the_a0_completion_decision_needs_exit_zero_a_passing_result_a_clean_audit_and_no_late_module_problem(tmp_path):
    assert launch_state.a0_completion_problems(OK_EXIT, a0_result_root(tmp_path, "good"), REQUIRED_FILES, ALLOWED_FILES) == []
    cases = [
        ("exit", dict(OK_EXIT, returncode=3), {}, "exited with code 3"),
        ("timeout", dict(OK_EXIT, timed_out=True), {}, "wall ceiling"),
        ("fatal", OK_EXIT, {"fatal": "other: boom"}, "fatal stop"),
        ("module", OK_EXIT, {"module_problems_at_exit": ["x differs"]}, "module differed"),
        ("empty audit", OK_EXIT, {"audit": {"violations": 0, "opened_in_data_directory": []}}, "must open"),
        ("one file short", OK_EXIT, {"audit": {"violations": 0, "opened_in_data_directory": ["a"]}}, "must open"),
        ("violation", OK_EXIT, {"audit": {"violations": 1, "opened_in_data_directory": ["a", "b"]}}, "violation"),
        ("extra file", OK_EXIT, {"audit": {"violations": 0, "opened_in_data_directory": ["a", "b", "z"]}}, "outside the allowed set"),
        ("verdict", OK_EXIT, {"models": {}}, "A0 verdict"),
    ]
    for name, exit_record, fields, message in cases:
        root = a0_result_root(tmp_path, name.replace(" ", "-"), **fields)
        problems = launch_state.a0_completion_problems(exit_record, root, REQUIRED_FILES, ALLOWED_FILES)
        assert any(message in p for p in problems), (name, problems)
    empty = tmp_path / "empty"
    empty.mkdir()
    assert launch_state.a0_completion_problems(OK_EXIT, empty, REQUIRED_FILES, ALLOWED_FILES) == ["A0_RESULT.json does not exist"]
    allowed_only = a0_result_root(tmp_path, "allowed-extra", audit={"violations": 0, "opened_in_data_directory": ["a", "b", "c"]})
    assert launch_state.a0_completion_problems(OK_EXIT, allowed_only, REQUIRED_FILES, ALLOWED_FILES) == []


# ---- Violations and workers ----------------------------------------------------------------------------------------------

def test_violation_files_void_either_record_of_the_pair(tmp_path):
    folder = tmp_path / "launch"
    assert launch_state.violation_files(folder, {SHA_A, SHA_B}) == []
    (folder / "violations").mkdir(parents=True)
    for name in (f"AUDIT_VIOLATION-{SHA_A}-A0-1.json", "AUDIT_VIOLATION-" + "c" * 64 + "-0-1.json", "unrelated.json"):
        (folder / "violations" / name).write_text("{}")
    assert launch_state.violation_files(folder, {SHA_A, SHA_B}) == [f"AUDIT_VIOLATION-{SHA_A}-A0-1.json"]
    assert launch_state.violation_files(folder, {"d" * 64}) == []


def test_process_start_and_worker_alive_use_the_start_time_and_the_group():
    mine = {"pid": os.getpid(), "start_time": launch_state.process_start(os.getpid())}
    assert mine["start_time"] and launch_state.worker_alive(mine)
    assert launch_state.process_start(2 ** 22 + 12345) is None
    gone = {"pid": 2 ** 22 + 12345, "start_time": "Thu Jan  1 00:00:00 1970"}
    assert not launch_state.worker_alive(gone)


def test_a_reused_pid_is_judged_by_its_process_group_not_by_the_pid(monkeypatch):
    """The start time differs (the pid was reused): the group probe decides. No group means gone, a live group means
    alive, and a group that exists but cannot be signalled (EPERM) is alive too (N13)."""
    record = {"pid": 12345, "start_time": "Thu Jan  1 00:00:00 1970"}
    monkeypatch.setattr(launch_state, "process_start", lambda pid: "Fri Jan  2 00:00:00 1970")

    def gone(pid, sig):
        raise ProcessLookupError

    def refused(pid, sig):
        raise PermissionError

    monkeypatch.setattr(os, "killpg", gone)
    assert launch_state.worker_alive(record) is False
    monkeypatch.setattr(os, "killpg", lambda pid, sig: None)
    assert launch_state.worker_alive(record) is True
    monkeypatch.setattr(os, "killpg", refused)
    assert launch_state.worker_alive(record) is True


def test_a_real_worker_is_alive_while_it_runs_and_gone_after_it_exits():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"], start_new_session=True)
    try:
        record = {"pid": child.pid, "start_time": launch_state.process_start(child.pid)}
        assert record["start_time"] and launch_state.worker_alive(record)
    finally:
        child.kill()
        child.wait()
    assert not launch_state.worker_alive(record)


def test_the_worker_writes_its_pid_file_first_and_only_once(tmp_path):
    path = launch_state.write_worker_pid(tmp_path, "0b", 2)
    payload = json.loads(path.read_text())
    assert path.name == "WORKER_PID-0b-2.json" and payload["pid"] == os.getpid() and payload["unit"] == "0b"
    assert payload["start_time"] == launch_state.process_start(os.getpid())
    with pytest.raises(FileExistsError):
        launch_state.write_worker_pid(tmp_path, "0b", 2)
