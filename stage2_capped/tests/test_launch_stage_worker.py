"""The stage worker's own preconditions (plan v3 section 6, S11) and where it gets its hashes from.

The control checks are a function of plain values, so they are tested directly with one defect at a time; the hash
sources are tested on the syntax tree, because the rule is about where a value comes from ("never computed by the
worker": the saved-file hashes and the launch-records hash come from the approved record, never from the disk)."""
import ast
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from stage2_capped import admission, driver, launch_io, launch_stage, launch_state
from stage2_capped.tests.launch_fixtures import make_stage_directory, needs_real_files, scratch, scratch_base  # noqa: F401
from stage2_capped.tests.launch_harness import Harness

PACKAGE = Path(__file__).resolve().parents[1]
RELAXER = "r" * 64
RECORD = {"launch_id": "L-1", "saved_file_hashes": {f"path{i}": f"{i:064x}" for i in range(110)}}


def identity(model_id):
    return {"checkpoint_sha256": f"{model_id}-sha", "best_update": 100, "arm": "x"}


# ---- control_problems: one defect at a time -----------------------------------------------------------------

def control(tmp_path, *, manifest_overrides=None, drop=None, typed=None, current=identity):
    """(problems) for stage 1 against a control root holding a valid stage 0, with the given defect."""
    tasks = driver.build_schedule("1", ["s1", "s2"], check_count=False)
    models = list(dict.fromkeys(t["model_id"] for t in tasks))
    arms = {m: {"checkpoint_identity": identity(m)} for m in models}
    overrides = {"arms": arms, **(manifest_overrides or {})}
    _, sha = make_stage_directory(tmp_path, "0", RECORD, "L-1", RELAXER, manifest_overrides=overrides)
    records = {(t["model_id"], t["kind"], driver.start_id_of(t)): True for t in tasks}
    if drop is not None:
        records.pop(sorted(records)[drop])
    context = SimpleNamespace(arm=lambda model_id: {"checkpoint_identity": current(model_id)})
    return launch_stage.control_problems(SimpleNamespace(records=records), tasks, tmp_path, "0",
                                         {"0": sha} if typed is None else typed, RECORD, context, RELAXER)


def test_a_consistent_control_has_no_problems(tmp_path):
    assert control(tmp_path) == []


def test_a_control_index_that_lacks_a_key_of_the_schedule_is_refused(tmp_path):
    problems = control(tmp_path, drop=3)
    caps = len(driver.STAGE_SPECS["1"]["caps"])                          # one control key serves every cap of the stage
    assert len(problems) == 1 and f"the control index lacks {caps} of this stage's control runs" in problems[0]


def test_a_control_marker_that_is_not_the_typed_one_is_refused(tmp_path):
    problems = control(tmp_path, typed={"0": "0" * 64})
    assert problems == ["the control root's stage 0 marker is not the typed one"]
    assert control(tmp_path / "second", typed={}) == ["the control root's stage 0 marker is not the typed one"]


@pytest.mark.parametrize("override", [{"launch_id": "another launch"}, {"relaxer_sha256": "z" * 64}])
def test_a_control_manifest_of_another_launch_or_relaxer_is_refused(tmp_path, override):
    problems = control(tmp_path, manifest_overrides=override)
    assert problems == ["the control manifest names a different launch or relaxer"]


def test_a_control_checkpoint_identity_that_differs_from_this_launchs_is_refused(tmp_path):
    def other(model_id):
        return {**identity(model_id), "checkpoint_sha256": "changed"} if model_id == "direct-1" else identity(model_id)
    problems = control(tmp_path, current=other)
    assert problems == ["direct-1: the control's checkpoint identity differs from this launch's"]


def test_a_manifest_without_the_arm_is_refused_for_that_model(tmp_path):
    problems = control(tmp_path, manifest_overrides={"arms": {}})
    assert len(problems) == len(dict.fromkeys(t["model_id"] for t in driver.build_schedule("1", ["s1"], check_count=False)))
    assert all("checkpoint identity differs" in p for p in problems)


# ---- Where the stage worker gets its hashes from (static) ----------------------------------------------------------

def tree_of(name):
    return ast.parse((PACKAGE / name).read_text())


def is_record_key(node, key):
    return (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "record"
            and isinstance(node.slice, ast.Constant) and node.slice.value == key)


def calls_named(tree, name):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and (
        (isinstance(n.func, ast.Attribute) and n.func.attr == name) or (isinstance(n.func, ast.Name) and n.func.id == name))]


def test_the_stage_worker_takes_the_saved_hashes_and_the_launch_records_hash_from_the_record():
    """Plan section 2 and decision log C21: the 110 saved-file hashes and the launch-records hash are never computed by
    the worker. Computed from the disk they would agree with the disk by construction and check nothing."""
    tree = tree_of("launch_stage.py")
    (saved_call,) = calls_named(tree, "saved_loader")                    # `deps.saved_loader(...)` (the definition is not a call)
    assert is_record_key(saved_call.args[0], "saved_file_hashes")
    (inputs_call,) = calls_named(tree, "inputs_loader")
    assert is_record_key(inputs_call.args[0], "expected_launch_records_sha256")
    (run_stage_call,) = calls_named(tree, "run_stage")
    (passed,) = [k.value for k in run_stage_call.keywords if k.arg == "saved_hashes"]
    assert isinstance(passed, ast.Name)                                   # `saved_hashes=saved_hashes`, assigned once:
    (assignment,) = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == passed.id for t in n.targets)]
    hashes = assignment.value
    assert isinstance(hashes, ast.IfExp) and is_record_key(hashes.body, "saved_file_hashes")
    assert isinstance(hashes.orelse, ast.Constant) and hashes.orelse.value is None


@pytest.mark.parametrize("name", ["launch_stage.py", "launch_a0.py"])
def test_no_worker_computes_a_hash_the_record_names(name):
    tree = tree_of(name)
    for forbidden in ("saved_file_hashes", "sha256_file"):
        assert not [c for c in calls_named(tree, forbidden)], (name, forbidden)
    for function in ast.walk(tree):                                       # the saved results are loaded by the injected loader only
        if isinstance(function, ast.FunctionDef) and function.name in ("run", "main", "control_problems"):
            assert not calls_named(function, "load_saved"), (name, function.name)


# ---- A stage worker started by hand ----------------------------------------------------------------------------------

def worker(h, unit, *extra, scenario=None):
    command = [sys.executable, "-m", "stage2_capped.tests.launch_test_worker", "--kind", "stage", "--unit", unit,
               "--record", str(h.path_b), "--approval", str(h.approval_b), "--record-sha256", h.sha_b, "--attempt", "1",
               "--group9-root", str(h.group9), *extra]
    return subprocess.run(command, cwd=h.scratch, env=h.environment(scenario), capture_output=True, text=True)


@needs_real_files
def test_a_stage_worker_started_by_hand_is_refused_and_spends_nothing(scratch, tmp_path):
    h = Harness(scratch, tmp_path)
    launch_io.fresh_root(h.root_b, "B-e2e")
    done = worker(h, "0b")
    assert done.returncode == 2, done.stdout + done.stderr
    refusal = json.loads((h.root_b / "worker-0b.attempt-1.result.json").read_text())
    assert refusal["label"] == "refused_before_start" and "no `started` event" in refusal["reason"]
    assert not (h.root_b / "worker-0b.result.json").exists() and not (h.root_b / "rows").exists()


@needs_real_files
def test_a_started_event_that_names_another_parent_does_not_start_a_stage_worker(scratch, tmp_path):
    h = Harness(scratch, tmp_path)
    launch_io.fresh_root(h.root_b, "B-e2e")
    launch_state.EventLog(h.root_b).append("started", unit="0b", attempt=1, parent_pid=1)       # not this process's parent
    done = worker(h, "0b")
    assert done.returncode == 2
    assert "names this process's parent" in json.loads((h.root_b / "worker-0b.attempt-1.result.json").read_text())["reason"]
    assert not (h.root_b / "rows").exists()
