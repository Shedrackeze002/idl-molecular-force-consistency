"""M4 engineering review fixes (F03 to F08, F12, F14 to F16): the driver's own refusals, the stage
manifest, the launch root, the control index and the study-input binding."""
import copy
import hashlib
import inspect
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from evaluation.adapter import callable_arm
from stage2_capped import admission, driver, launch_io, row_schema
from stage2_capped.stage_gate import Stage0Controller, StageLedger, StageOrderError
from stage2_capped.tests import fixtures as fx
from stage2_capped.tests.test_driver import (LAUNCH_RECORDS_SHA256, TARGET, frozen_saved, make_context,  # noqa: F401
                                             make_inputs, make_tasks, needs_real_files, run_test_stage,
                                             stage0_setup, study, world)

REPO = Path(driver.__file__).resolve().parents[1]


def ledger_through(*stages):
    ledger = StageLedger()
    for stage in ("A0",) + stages:
        ledger.start(stage)
        ledger.finish(stage, True, replay_passed=True if stage == "A0" else None)
    return ledger


def force_counter(context):
    """Counts every force call any arm of this context makes."""
    calls = []
    species = next(iter(context.inputs.starts.values()))["species"]
    force = fx.harmonic(TARGET, 4.0)
    context.arm_provider = lambda model_id: callable_arm(lambda p: (calls.append(model_id), force(p))[1],
                                                         species, model_id)
    return calls


# ---- F03: the caller cannot leave enforcement out -------------------------

def test_a_stage_needs_a_ledger_and_the_fixed_schedule_unless_a_test_flag_is_set(world, tmp_path):
    context = make_context(tmp_path, *world)
    tasks = make_tasks(context, "0b", ["consistency-1-0"], [None])
    with pytest.raises(driver.DriverError, match="StageLedger"):
        driver.run_stage("0b", context, tasks=tasks, allow_custom_tasks=True)
    with pytest.raises(driver.DriverError, match="tests only"):
        driver.run_stage("0b", context, ledger=ledger_through("0"), tasks=tasks)
    assert not (context.root / "rows").exists()                         # nothing was started or written


def test_the_fixed_schedule_is_checked_before_the_ledger_starts_the_stage(world, tmp_path):
    context = make_context(tmp_path, *world)                            # 2 starts: not the 10 of the study
    ledger = ledger_through("0")
    with pytest.raises(driver.DriverError, match="schedule has"):
        driver.run_stage("0b", context, ledger=ledger)
    assert ledger.state["0b"] == "pending" and not (context.root / "rows").exists()


def test_stage_zero_refuses_a_controller_that_expects_fewer_than_all_80_runs(world, tmp_path):
    context = make_context(tmp_path, *world)
    ledger = ledger_through()
    with pytest.raises(driver.DriverError, match="expects 5"):
        driver.run_stage("0", context, ledger=ledger, saved=({}, {}), controller=Stage0Controller(expected_runs=5))
    assert ledger.state["0"] == "pending" and not (context.root / "rows").exists()


def test_a_stage_that_completed_fewer_runs_than_its_spec_is_not_marked_complete(world, tmp_path, monkeypatch):
    context = make_context(tmp_path, *world)
    tasks = make_tasks(context, "0b", ["consistency-1-0"], [None])
    monkeypatch.setattr(driver, "build_schedule", lambda *a, **k: tasks)    # a short schedule slips through
    with pytest.raises(driver.DriverError, match=f"completed {len(tasks)} runs, expected 32"):
        driver.run_stage("0b", context, ledger=ledger_through("0"))
    stage = context.root / "rows" / "stage0b"
    assert (stage / "INCOMPLETE.json").exists() and not (stage / "STAGE_COMPLETE.json").exists()


# ---- F04: the stage manifest ----------------------------------------------

def test_a_finished_stage_writes_a_manifest_and_the_completion_record_names_its_hash(world, tmp_path):
    context = make_context(tmp_path, *world)
    inner = context.arm_provider

    def provider(model_id):
        arm = inner(model_id)
        arm["checkpoint_identity"] = {"checkpoint_sha256": hashlib.sha256(model_id.encode()).hexdigest()}
        return arm
    context.arm_provider = provider
    tasks = make_tasks(context, "0b", ["consistency-1-0", "consistency-1-1"], [None])
    summary = run_test_stage("0b", context, tasks=tasks)
    directory = context.root / "rows" / "stage0b"
    manifest_bytes = (directory / "STAGE_MANIFEST.json").read_bytes()
    assert summary["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest() == \
        json.loads((directory / "STAGE_COMPLETE.json").read_text())["manifest_sha256"]
    manifest = json.loads(manifest_bytes)
    assert manifest["stage"] == "0b" and manifest["launch_id"] == context.launch_id and manifest["runs"] == len(tasks)
    assert manifest["relaxer_sha256"] == hashlib.sha256((REPO / "stage2_capped" / "relax_capped.py").read_bytes()).hexdigest()
    assert set(manifest["arms"]) == {"consistency-1-0", "consistency-1-1"}
    assert manifest["arms"]["consistency-1-1"]["arm_id"] == "consistency-1-1"
    assert manifest["arms"]["consistency-1-1"]["checkpoint_identity"]["checkpoint_sha256"] == \
        hashlib.sha256(b"consistency-1-1").hexdigest()
    rows = [row_schema.loads_row(p.read_text()) for p in directory.glob("*/cinf/*.json")]
    def counts(values):
        found = {}
        for value in values:
            found["none" if value is None else value] = found.get("none" if value is None else value, 0) + 1
        return dict(sorted(found.items()))
    assert manifest["outcome_categories"] == counts(r["outcome_category"] for r in rows)
    assert manifest["stop_reasons"] == counts(r["stop_reason"] for r in rows)
    assert sum(manifest["outcome_categories"].values()) == len(rows)
    assert manifest["refused_trial_count"] == sum(r["refused_trial_count"] for r in rows)
    assert manifest["control"] == summary["control"] and "stage0" not in manifest


def test_the_stage_zero_manifest_holds_the_verdict_the_hashes_and_one_match_line_per_task(world, tmp_path):
    context, tasks, saved, controller = stage0_setup(world, tmp_path)
    hashes = {"reports/a/result.json": "0" * 64}
    run_test_stage("0", context, tasks=tasks, saved=saved, controller=controller, saved_hashes=hashes)
    manifest = json.loads((context.root / "rows" / "stage0" / "STAGE_MANIFEST.json").read_text())
    stage0 = manifest["stage0"]
    assert stage0["verdict"]["proceed"] is True and stage0["compared"] == len(tasks)
    assert stage0["saved_file_hashes"] == hashes
    assert [m["task"] for m in stage0["matches"]] == [driver.task_id(t) for t in tasks]
    assert all(m["matched"] is True for m in stage0["matches"])


# ---- F05: the launch root -------------------------------------------------

def test_a_context_needs_a_root_made_for_its_own_launch_id(world, tmp_path):
    land, manifest, classification = world
    inputs = make_inputs(land, manifest, classification)
    provider = lambda model_id: None  # noqa: E731
    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(ValueError, match="not made by fresh_root"):
        driver.Context(bare, "L-bare", inputs, provider)
    root = launch_io.fresh_root(tmp_path / "other", "L-other")
    with pytest.raises(ValueError, match="belongs to launch 'L-other'"):
        driver.Context(root, "L-mine", inputs, provider)


def test_a_context_refuses_a_root_inside_the_worktree(world, tmp_path, monkeypatch):
    land, manifest, classification = world
    inputs = make_inputs(land, manifest, classification)
    tree = tmp_path / "tree"
    tree.mkdir()
    monkeypatch.setattr(driver, "WORKTREE", tree)                       # a stand-in worktree in a temporary folder
    inside = launch_io.fresh_root(tree / "out", "L-in")
    with pytest.raises(driver.DriverError, match="inside the worktree"):
        driver.Context(inside, "L-in", inputs, lambda model_id: None)
    outside = launch_io.fresh_root(tmp_path / "outside", "L-out")        # a sibling of the worktree is fine
    assert driver.Context(outside, "L-out", inputs, lambda model_id: None).root == outside


def test_a_context_checks_free_space_for_the_storage_ceiling(world, tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(launch_io, "require_free", lambda path, required: seen.append((Path(path), required)))
    context = make_context(tmp_path, *world, storage_limit=12345)
    assert seen == [(context.root, 12345)]

    def refuse(path, required):
        raise RuntimeError("not enough free storage")
    monkeypatch.setattr(launch_io, "require_free", refuse)
    with pytest.raises(RuntimeError, match="free storage"):
        make_context(tmp_path, *world, name="second")


# ---- F06: the launch records and the procedure ----------------------------

@needs_real_files
def test_the_launch_records_must_have_the_hash_the_launch_record_names():
    with pytest.raises(admission.AdmissionError, match="launch records differ"):
        driver.load_study_inputs("0" * 64)
    with pytest.raises(TypeError):
        driver.load_study_inputs()                                       # the hash is required


@needs_real_files
def test_a_changed_evaluation_config_or_procedure_is_refused(tmp_path, monkeypatch):
    worktree = tmp_path / "tree"
    (worktree / "configs").mkdir(parents=True)
    config = json.loads((REPO / "configs" / "aspirin-shape-recovery.json").read_text())
    config["relaxation"]["initial_step_size"] = 0.25
    (worktree / "configs" / "aspirin-shape-recovery.json").write_text(json.dumps(config))
    with pytest.raises(admission.AdmissionError, match="evaluation block differs from the config"):
        driver.load_study_inputs(LAUNCH_RECORDS_SHA256, worktree=worktree)
    altered = {**driver.FROZEN_PROCEDURE, "max_backtracking_attempts_per_step": 41}
    monkeypatch.setattr(driver, "FROZEN_PROCEDURE", altered)
    with pytest.raises(admission.AdmissionError, match="six frozen values"):
        driver.load_study_inputs(LAUNCH_RECORDS_SHA256)


def test_the_frozen_procedure_constant_matches_the_budget_and_the_config_file():
    config = json.loads((REPO / "configs" / "aspirin-shape-recovery.json").read_text())
    assert driver.FROZEN_PROCEDURE == config["relaxation"]
    assert driver.MAX_FORCE_EVALUATIONS == driver.FROZEN_PROCEDURE["max_force_evaluations"] == 1500
    assert len(driver.FROZEN_PROCEDURE) == 6


# ---- F07: the control index -----------------------------------------------

@pytest.fixture
def finished_control(world, tmp_path):
    context = make_context(tmp_path, *world, stiffness=0.5)
    run_test_stage("0b", context, tasks=make_tasks(context, "0b", ["consistency-1-0"], [None]))
    return context


def test_a_complete_control_root_loads_and_is_refused_unless_custom_runs_are_allowed(finished_control):
    root = finished_control.root
    with pytest.raises(driver.MissingControl, match="not a complete run of the fixed schedule"):
        driver.ControlIndex.from_roots([root])
    assert len(driver.ControlIndex.from_roots([root], allow_custom=True).records) == 4


def test_a_control_root_without_its_completion_record_or_with_an_incomplete_marker_is_refused(finished_control):
    directory = finished_control.root / "rows" / "stage0b"
    complete = directory / "STAGE_COMPLETE.json"
    saved = complete.read_text()
    complete.unlink()
    with pytest.raises(driver.MissingControl, match="no STAGE_COMPLETE.json"):
        driver.ControlIndex.from_roots([finished_control.root], allow_custom=True)
    complete.write_text(saved)
    (directory / "INCOMPLETE.json").write_text("{}")
    with pytest.raises(driver.MissingControl, match="marked INCOMPLETE"):
        driver.ControlIndex.from_roots([finished_control.root], allow_custom=True)


def test_a_control_root_with_a_missing_row_a_foreign_relaxer_or_a_changed_trace_is_refused(finished_control, monkeypatch):
    root = finished_control.root
    rows = sorted((root / "rows" / "stage0b").glob("*/cinf/*.json"))
    with pytest.raises(driver.MissingControl, match="made by a different relaxer"):
        monkeypatch.setattr(driver, "relaxer_sha256", lambda: "f" * 64)
        driver.ControlIndex.from_roots([root], allow_custom=True)
    monkeypatch.undo()
    trace = root / row_schema.loads_row(rows[0].read_text())["trace_file"]
    original = trace.read_bytes()
    trace.write_bytes(original[:-1] + bytes([original[-1] ^ 1]))
    with pytest.raises(ValueError, match="SHA-256"):
        driver.ControlIndex.from_roots([root], allow_custom=True)
    trace.write_bytes(original)
    hidden = rows[0].with_suffix(".hidden")
    rows[0].rename(hidden)
    with pytest.raises(driver.MissingControl, match="control rows, the stage recorded"):
        driver.ControlIndex.from_roots([root], allow_custom=True)


def test_two_control_rows_for_one_start_are_refused(finished_control):
    with pytest.raises(driver.MissingControl, match="two control rows"):
        driver.ControlIndex.from_roots([finished_control.root, finished_control.root], allow_custom=True)


# ---- F08: every arm is built and checked before the first force call ------

def test_a_failing_second_arm_stops_the_stage_before_any_force_call_and_before_the_ledger_starts(world, tmp_path):
    context = make_context(tmp_path, *world)
    calls = force_counter(context)
    working = context.arm_provider

    def provider(model_id):
        if model_id == "consistency-1-1":
            raise RuntimeError("checkpoint missing")
        return working(model_id)
    context.arm_provider = provider
    tasks = make_tasks(context, "0b", ["consistency-1-0", "consistency-1-1"], [None])
    ledger = ledger_through("0")
    with pytest.raises(RuntimeError, match="checkpoint missing"):
        driver.run_stage("0b", context, ledger=ledger, tasks=tasks, allow_custom_tasks=True)
    assert calls == [] and ledger.state["0b"] == "pending" and not (context.root / "rows").exists()


def test_an_arm_that_is_not_ready_is_refused_before_any_force_call(world, tmp_path):
    context = make_context(tmp_path, *world)
    calls = force_counter(context)
    working = context.arm_provider

    def provider(model_id):
        arm = working(model_id)
        arm["status"] = "unavailable"
        return arm
    context.arm_provider = provider
    with pytest.raises(ValueError, match="not ready"):
        run_test_stage("0b", context, tasks=make_tasks(context, "0b", ["consistency-1-0"], [None]))
    assert calls == [] and not (context.root / "rows").exists()


# ---- F12, F14, F15, F16 ---------------------------------------------------

def test_the_incomplete_marker_lists_only_runs_that_passed_every_gate(world, tmp_path):
    context, tasks, saved, controller = stage0_setup(world, tmp_path)
    target = next(t for t in tasks if t["kind"] == "original" and t["model_id"] == "direct-0")
    key = (target["model_id"], target["key"])
    saved[0][key] = {**saved[0][key], "step_count": saved[0][key]["step_count"] + 1}
    with pytest.raises(driver.Stage0Stop):
        run_test_stage("0", context, tasks=tasks, saved=saved, controller=controller)
    marker = json.loads((context.root / "rows" / "stage0" / "INCOMPLETE.json").read_text())
    before = [driver.task_id(t) for t in tasks[:tasks.index(target)]]
    assert marker["failed_task"] == driver.task_id(target) and marker["completed_runs"] == before
    assert len(list((context.root / "rows" / "stage0").glob("*/cinf/*.json"))) == len(before) + 1   # its row is on disk


@pytest.mark.parametrize("failure", [ValueError("bad trace"), OverflowError("too large")])
def test_a_trace_that_cannot_be_built_stops_the_stage_with_a_labelled_error_naming_the_task(world, tmp_path,
                                                                                           monkeypatch, failure):
    context = make_context(tmp_path, *world)
    task = make_tasks(context, "0b", ["consistency-1-0"], [None])[0]

    def broken(records, n_atoms):
        raise failure
    monkeypatch.setattr(row_schema, "trace_from_records", broken)
    with pytest.raises(driver.DriverError, match=driver.task_id(task)) as error:
        driver.execute_task(context, task)
    assert "valid trace" in str(error.value) and isinstance(error.value.__cause__, type(failure))
    assert not list(context.root.rglob("*.npy")) and not list(context.root.rglob("original-*.json"))


@pytest.mark.parametrize("failure", [ValueError("bad row"), OverflowError("too large")])
def test_a_row_that_cannot_be_built_stops_the_stage_with_a_labelled_error_naming_the_task(world, tmp_path,
                                                                                         monkeypatch, failure):
    context = make_context(tmp_path, *world)
    task = make_tasks(context, "0b", ["consistency-1-0"], [None])[0]

    def broken(**kwargs):
        raise failure
    monkeypatch.setattr(driver, "make_result_row", broken)
    with pytest.raises(driver.DriverError, match=driver.task_id(task)) as error:
        driver.execute_task(context, task)
    assert isinstance(error.value.__cause__, type(failure))


def test_the_storage_reserve_comes_from_the_trace_record_size_at_the_budget(world, tmp_path):
    context = make_context(tmp_path, *world)
    atoms = np.asarray(next(iter(context.inputs.starts.values()))["positions"]).shape[-2]
    itemsize = row_schema.trace_dtype(atoms).itemsize
    assert context.reserve_bytes == driver.run_reserve_bytes(atoms) == 1500 * itemsize + driver.ROW_RESERVE_BYTES
    assert driver.run_reserve_bytes(atoms + 1) - driver.run_reserve_bytes(atoms) == 1500 * 24   # three floats an atom
    out = driver.execute_task(context, make_tasks(context, "0b", ["consistency-1-0"], [None])[0])
    assert len(out["records"]) * itemsize + 4096 <= context.reserve_bytes


def test_the_storage_reserve_uses_the_largest_molecule_among_the_starts(world, tmp_path):
    land, manifest, classification = world
    inputs = make_inputs(land, manifest, classification)
    small = np.asarray(next(iter(inputs.starts.values()))["positions"]).shape[-2]
    inputs.starts = {**inputs.starts, "aspirin/99/plus": {"positions": np.zeros((small + 4, 3))}}
    root = launch_io.fresh_root(tmp_path / "mixed", "L-mixed")
    context = driver.Context(root, "L-mixed", inputs, lambda model_id: None)
    assert context.reserve_bytes == driver.run_reserve_bytes(small + 4) > driver.run_reserve_bytes(small)


def test_the_context_no_longer_takes_a_relaxer_hash_and_rows_carry_the_hash_of_the_relaxer_file(world, tmp_path):
    assert "relaxer_hash" not in inspect.signature(driver.Context).parameters
    context = make_context(tmp_path, *world)
    with pytest.raises(TypeError):
        driver.Context(context.root, context.launch_id, context.inputs, context.arm_provider, relaxer_hash="a" * 64)
    out = driver.execute_task(context, make_tasks(context, "0b", ["consistency-1-0"], [None])[0])
    assert out["row"]["relaxer_sha256"] == hashlib.sha256(
        (REPO / "stage2_capped" / "relax_capped.py").read_bytes()).hexdigest() == driver.relaxer_sha256()


# ---- Round 2 of the review: N1, N2, N3, N12, R37 and the production path --------------------

def test_a_launch_stage_zero_needs_the_hashes_of_all_110_saved_files(world, tmp_path):
    context = make_context(tmp_path, *world)
    for bad in (None, {}, {f"f{i}": "0" * 64 for i in range(109)}, {f"f{i}": "short" for i in range(110)}, "x" * 110):
        ledger = ledger_through()
        with pytest.raises(driver.DriverError, match="SHA-256 of all 110 saved files"):
            driver.run_stage("0", context, ledger=ledger, saved=({}, {}), controller=Stage0Controller(),
                             saved_hashes=bad)
        assert ledger.state["0"] == "pending"
    good = {f"f{i}": "0" * 64 for i in range(driver.SAVED_FILE_COUNT)}
    with pytest.raises(driver.DriverError, match="schedule has"):       # past the hash check; the synthetic study is small
        driver.run_stage("0", context, ledger=ledger_through(), saved=({}, {}), controller=Stage0Controller(),
                         saved_hashes=good)
    assert driver.SAVED_FILE_COUNT == 110


@pytest.mark.parametrize("kind", ["original", "reference"])
def test_stage_zero_refuses_a_saved_set_that_misses_a_scheduled_task_before_any_force_call(world, tmp_path, kind):
    context, tasks, saved, controller = stage0_setup(world, tmp_path)
    calls = force_counter(context)
    context._arms.clear()
    table = saved[0] if kind == "original" else saved[1]
    target = next(t for t in tasks if t["kind"] == kind)
    del table[(target["model_id"], target["key"])]
    ledger = ledger_through()
    with pytest.raises(driver.DriverError, match=f"1 scheduled Stage 0 tasks have no saved {kind} result"):
        driver.run_stage("0", context, ledger=ledger, saved=saved, controller=controller, tasks=tasks,
                         allow_custom_tasks=True)
    assert calls == [] and ledger.state["0"] == "pending" and not (context.root / "rows").exists()


@pytest.mark.parametrize("identity", [{"best_update": np.float32(1.5)}, {"offset": float("nan")}])
def test_a_checkpoint_identity_that_is_not_plain_json_is_refused_before_any_force_call(world, tmp_path, identity):
    context = make_context(tmp_path, *world)
    calls = force_counter(context)
    inner = context.arm_provider

    def provider(model_id):
        arm = inner(model_id)
        arm["checkpoint_identity"] = identity                              # a numpy scalar or a NaN cannot be written
        return arm
    context.arm_provider = provider
    ledger = ledger_through("0")
    with pytest.raises(driver.DriverError, match="consistency-1-0: the checkpoint identity is not plain JSON"):
        driver.run_stage("0b", context, ledger=ledger, allow_custom_tasks=True,
                         tasks=make_tasks(context, "0b", ["consistency-1-0"], [None]))
    assert calls == [] and ledger.state["0b"] == "pending" and not (context.root / "rows").exists()


def test_a_spent_stage_folder_is_refused_before_the_ledger_starts_the_stage(world, tmp_path):
    context = make_context(tmp_path, *world)
    tasks = make_tasks(context, "0b", ["consistency-1-0"], [None])
    (context.root / "rows" / "stage0b").mkdir(parents=True)
    ledger = ledger_through("0")
    with pytest.raises(FileExistsError, match="a spent stage directory is never reused"):
        driver.run_stage("0b", context, ledger=ledger, tasks=tasks, allow_custom_tasks=True)
    assert ledger.state["0b"] == "pending"                      # not left "running" for a stage that never began


def test_a_stage_the_ledger_refuses_builds_no_arm_at_all(world, tmp_path):
    context = make_context(tmp_path, *world)
    built = []
    context.arm_provider = lambda model_id: built.append(model_id)
    tasks = make_tasks(context, "0b", ["consistency-1-0"], [None])
    with pytest.raises(StageOrderError, match="stage 0 is not complete"):
        driver.run_stage("0b", context, ledger=ledger_through(), tasks=tasks, allow_custom_tasks=True)
    assert built == [] and not (context.root / "rows").exists()


def test_the_manifest_counts_refused_trials(world, tmp_path):
    context = make_context(tmp_path, *world)
    species = next(iter(context.inputs.starts.values()))["species"]
    calls, force = [], fx.harmonic(TARGET, 4.0)

    def flaky(positions):
        calls.append(1)
        return np.full_like(positions, np.nan) if len(calls) == 3 else force(positions)
    context.arm_provider = lambda model_id: callable_arm(flaky, species, model_id)
    tasks = make_tasks(context, "0b", ["consistency-1-0"], [None])
    run_test_stage("0b", context, tasks=tasks)
    directory = context.root / "rows" / "stage0b"
    rows = [row_schema.loads_row(p.read_text()) for p in directory.glob("*/cinf/*.json")]
    refused = sum(r["refused_trial_count"] for r in rows)
    assert refused == 1 and any(r["stop_reason"] == "nonfinite_force" for r in rows)
    assert json.loads((directory / "STAGE_MANIFEST.json").read_text())["refused_trial_count"] == refused


@needs_real_files
def test_the_production_path_runs_the_fixed_stages_with_real_inputs_and_no_test_flag(study, tmp_path):
    """Real study inputs, zero-force arms, the fixed schedules, no allow_* flag. Stage 0 stops at its first task
    against the saved results (a zero-force arm cannot reproduce them); Stages 0b and 2 then complete."""
    species = next(iter(study.starts.values()))["species"]
    root = launch_io.fresh_root(tmp_path / "prod", "L-prod")
    context = driver.Context(root, "L-prod", study,
                             lambda model_id: callable_arm(lambda p: np.zeros_like(p), species, model_id))
    from stage2_capped import stage0_compare
    hashes = stage0_compare.saved_file_hashes()
    original, reference = stage0_compare.load_saved(expected_hashes=hashes)
    ledger = ledger_through()
    with pytest.raises(driver.Stage0Stop) as stop:
        driver.run_stage("0", context, ledger=ledger, saved=(original, reference), controller=Stage0Controller(),
                         saved_hashes=hashes)
    assert stop.value.verdict["verdict"] == "frozen re-run required" and ledger.state["0"] == "failed"
    assert (root / "rows" / "stage0" / "INCOMPLETE.json").exists()
    # A parent whose Stage 0 completed (a ledger snapshot), then the fixed lambda = 1 control and capped stages.
    later = StageLedger.from_snapshot({"state": {"A0": "complete", "0": "complete", "0b": "pending", "1": "pending",
                                                 "2": "pending"}, "a0_replay_passed": True, "events": []})
    control = driver.run_stage("0b", context, ledger=later)
    assert control["runs"] == 32 and control["complete"] is True and control["custom_tasks"] is False
    later.state["1"] = "complete"
    index = driver.ControlIndex.from_roots([root], stages=("0b",))
    capped = driver.run_stage("2", context, ledger=later, control_index=index)
    assert capped["runs"] == 128 and capped["complete"] is True
    labels = {p.parent.name for p in (root / "rows" / "stage2").glob("*/*/*.json")}
    assert labels == {"c0p05", "c0p10", "c0p25", "c1p00"}
    manifest = json.loads((root / "rows" / "stage2" / "STAGE_MANIFEST.json").read_text())
    assert manifest["runs"] == 128 and set(manifest["arms"]) == set(admission.LAMBDA1_IDS)
