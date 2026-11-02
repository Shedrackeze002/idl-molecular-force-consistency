"""M4: the driver, on synthetic inputs and fixture arms (no model), plus the real study
inputs with a zero-force arm when the saved Group9 files are present."""
import copy
import json
import time

import numpy as np
import pytest

from evaluation.adapter import callable_arm
from evaluation.start_set import build_start_set
from stage2_capped import admission, driver, launch_io, prefix_gate, row_schema, stage0_compare
from stage2_capped.stage_gate import Stage0Controller, Stage0Halted, StageLedger, StageOrderError
from stage2_capped.stage0_compare import first_difference
from stage2_capped.tests import fixtures as fx

TARGET = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.3, 0.0]])


def real_files():
    try:
        root = admission.group9_root()
        return (root / admission.DATA_DIRECTORY).is_dir() and (root / admission.RECOVERY_RECORDS).is_file()
    except admission.AdmissionError:
        return False


needs_real_files = pytest.mark.skipif(not real_files(), reason="saved Group9 files are absent")


# ---- Synthetic context ----------------------------------------------------

def make_inputs(land, manifest, classification, procedure=None):
    starts = build_start_set(manifest, land["records"], land["policy"], {"kind": "first_stored_frame"},
                             partitions=("validation",))
    refs = {r.record_id: r for r in land["records"] if r.kind == "minimum"}
    return driver.StudyInputs(starts, refs, land["policy"], manifest, classification,
                              {k: False for k in starts}, procedure or fx.procedure(max_force_evaluations=60),
                              reference_ids=tuple(sorted(refs)))


def make_context(tmp_path, land, manifest, classification, stiffness=4.0, name="launch", **kwargs):
    inputs = make_inputs(land, manifest, classification)
    species = next(iter(inputs.starts.values()))["species"]
    root = launch_io.fresh_root(tmp_path / name)
    provider = lambda model_id: callable_arm(fx.harmonic(TARGET, stiffness), species, model_id)  # noqa: E731
    return driver.Context(root, f"L-{name}", inputs, provider, **kwargs)


def make_tasks(context, stage, models, caps, keys=2):
    tasks = []
    for model in models:
        for cap in caps:
            for key in sorted(context.inputs.starts)[:keys]:
                tasks.append({"stage": stage, "model_id": model, "cap": cap, "kind": "original", "key": key})
            for ref in context.inputs.reference_ids:
                tasks.append({"stage": stage, "model_id": model, "cap": cap, "kind": "reference", "key": ref})
    return tasks


@pytest.fixture
def world(synthetic_landscape, frozen_synthetic_manifest, synthetic_endpoint_classification):
    return synthetic_landscape, frozen_synthetic_manifest, synthetic_endpoint_classification


def frozen_saved(context, tasks):
    """The 'saved' results: what the frozen evaluator and frozen relax produce for the tasks."""
    from evaluation.relax import relax
    from evaluation.runner import _arm_force_supplier, evaluate_starts
    i = context.inputs
    original, reference = {}, {}
    for task in tasks:
        arm = context.arm(task["model_id"])
        if task["kind"] == "original":
            row = evaluate_starts({task["key"]: i.starts[task["key"]]}, arm, i.procedure, i.classification,
                                  i.refs, i.policy, split_manifest=i.manifest, seen_minimum=i.seen)
            original[(task["model_id"], task["key"])] = row["branches"][task["key"]]
        else:
            own = i.refs[task["key"]]
            supplier = _arm_force_supplier(arm, own.species, own.record_id + "/reference-control")
            reference[(task["model_id"], task["key"])] = {"relaxation": relax(own.positions, supplier, i.procedure, arm=arm)}
    return original, reference


# ---- Schedule -------------------------------------------------------------

def test_the_fixed_schedules_have_the_spec_run_counts():
    keys = [f"aspirin/{g}/{b}" for g in ("13", "18", "18_2", "66", "78") for b in ("minus", "plus")]
    for stage, runs in (("0", 80), ("0b", 32), ("1", 320), ("2", 128)):
        tasks = driver.build_schedule(stage, keys)
        assert len(tasks) == runs == driver.STAGE_SPECS[stage]["runs"]
        assert len({driver.task_id(t) for t in tasks}) == runs
    assert sum(s["runs"] for s in driver.STAGE_SPECS.values()) == 560
    assert sum(s["attempt_ceiling"] for s in driver.STAGE_SPECS.values()) == 840_000
    assert {s: driver.STAGE_SPECS[s]["wall_seconds"] for s in driver.STAGE_SPECS} == {
        "0": 600.0, "0b": 300.0, "1": 2100.0, "2": 900.0}


def test_stage_zero_is_the_accepted_order_and_matches_the_comparator_schedule():
    keys = [f"aspirin/{g}/{b}" for g in ("13", "18", "18_2", "66", "78") for b in ("minus", "plus")]
    assert [driver.task_id(t) for t in driver.build_schedule("0", keys)] == stage0_compare.schedule(keys)


def test_capped_stages_run_each_start_at_every_cap_in_fixed_order():
    keys = [f"aspirin/{g}/{b}" for g in ("13", "18", "18_2", "66", "78") for b in ("minus", "plus")]
    tasks = driver.build_schedule("1", keys)
    assert [t["cap"] for t in tasks[:16]] == [0.05] * 16 and tasks[16]["cap"] == 0.10
    assert {t["cap"] for t in tasks} == {0.05, 0.10, 0.25, 1.0} and {t["model_id"] for t in tasks} == set(admission.PRIMARY_IDS)
    assert all(t["model_id"] in admission.LAMBDA1_IDS for t in driver.build_schedule("2", keys))
    with pytest.raises(driver.DriverError):
        driver.build_schedule("1", keys[:9])


# ---- Prevalidation --------------------------------------------------------

def test_prevalidation_returns_one_plan_per_start_and_runs_no_force_call(world):
    land, manifest, classification = world
    inputs, calls = make_inputs(land, manifest, classification), []
    species = next(iter(inputs.starts.values()))["species"]
    arm = callable_arm(lambda p: (calls.append(1), np.zeros_like(p))[1], species, "counted")
    plans, bound = driver.prevalidate(inputs.starts, arm, inputs.procedure, classification, inputs.refs,
                                      land["policy"], manifest)
    assert [p[0] for p in plans] == sorted(inputs.starts) and bound == "fixture" and not calls


def test_prevalidation_refuses_a_batched_start_before_any_force_call(world):
    land, manifest, classification = world
    inputs, calls = make_inputs(land, manifest, classification), []
    species = next(iter(inputs.starts.values()))["species"]
    arm = callable_arm(lambda p: (calls.append(1), np.zeros_like(p))[1], species, "counted")
    key = sorted(inputs.starts)[0]
    starts = dict(inputs.starts)
    starts[key] = {**starts[key], "positions": np.stack([starts[key]["positions"]] * 2)}
    with pytest.raises(ValueError):
        driver.prevalidate(starts, arm, inputs.procedure, classification, inputs.refs, land["policy"], manifest)
    assert not calls


def test_prevalidation_refuses_bad_species_and_bad_procedure(world):
    land, manifest, classification = world
    inputs = make_inputs(land, manifest, classification)
    arm = callable_arm(lambda p: np.zeros_like(p), ("C", "H", "O"), "a")
    bad_species = dict(inputs.starts)
    key = sorted(bad_species)[-1]
    bad_species[key] = {**bad_species[key], "species": ("H", "H", "O")}
    with pytest.raises(ValueError):
        driver.prevalidate(bad_species, arm, inputs.procedure, classification, inputs.refs, land["policy"], manifest)
    with pytest.raises(ValueError):
        driver.prevalidate(inputs.starts, arm, {**inputs.procedure, "max_force_evaluations": 0}, classification,
                           inputs.refs, land["policy"], manifest)


def test_a_batch_of_one_and_a_degenerate_start_are_refused_before_any_force_call(world):
    land, manifest, classification = world
    inputs, calls = make_inputs(land, manifest, classification), []
    species = next(iter(inputs.starts.values()))["species"]
    arm = callable_arm(lambda p: (calls.append(1), np.zeros_like(p))[1], species, "counted")
    key = sorted(inputs.starts)[0]
    position = inputs.starts[key]["positions"]

    def refused(positions, match):
        starts = dict(inputs.starts)
        starts[key] = {**starts[key], "positions": positions}
        with pytest.raises(ValueError, match=match):
            driver.prevalidate(starts, arm, inputs.procedure, classification, inputs.refs, land["policy"], manifest)

    refused(np.stack([position]), "batch size 1")                       # shape (1, N, 3): still a batch
    coincident = np.array(position, copy=True)
    coincident[1] = coincident[0]
    refused(coincident, "coincident")                                   # the frozen start check
    assert not calls


def test_prevalidation_refuses_starts_outside_the_validation_partition(world):
    land, manifest, classification = world
    inputs = make_inputs(land, manifest, classification)
    species = next(iter(inputs.starts.values()))["species"]
    arm = callable_arm(lambda p: np.zeros_like(p), species, "a")
    # The synthetic landscape's own held-out partition (no real data), as a start set.
    held_out = build_start_set(manifest, land["records"], land["policy"], {"kind": "first_stored_frame"},
                               partitions=("test",))
    with pytest.raises(ValueError, match="Only validation branches admitted"):
        driver.prevalidate(held_out, arm, inputs.procedure, classification, inputs.refs, land["policy"], manifest)


# ---- Row ingredients ------------------------------------------------------

def test_component_count_equals_the_frozen_trace_summary():
    from recovery_trace import graph_summary
    rng = np.random.default_rng(0)
    for trial in range(60):
        n = int(rng.integers(2, 12))
        p = rng.normal(scale=float(rng.choice([0.5, 2.0, 6.0])), size=(n, 3))
        assert driver.component_count(p) == graph_summary(p, np.zeros_like(p), 5.0)["interaction_component_count"]
    assert driver.component_count(np.array([[0, 0, 0], [20, 0, 0.0]])) == 2
    assert driver.component_count(np.array([[0, 0, 0], [4.999, 0, 0.0]])) == 1
    assert driver.component_count(np.array([[0, 0, 0], [5.0, 0, 0.0]])) == 2   # strictly below the cutoff


def test_the_first_disconnection_query_covers_rejected_trials_and_the_start():
    far = np.array([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]])
    near = np.zeros((2, 3))
    records = [{"attempt_index": 1, "positions": near}, {"attempt_index": 2, "positions": near},
               {"attempt_index": 3, "positions": far}, {"attempt_index": 4, "positions": near}]
    assert driver.connectivity(records, near) == (True, 3)
    assert driver.connectivity(records[:2], far) == (False, None)
    assert driver.connectivity([{"attempt_index": 1, "positions": far}], far) == (False, 1)


def test_accepted_step_fractions_count_only_accepted_trials():
    base = {"kind": "trial", "accepted": True}
    records = [{"kind": "initial", "accepted": None, "max_displacement_A": 0.0},
               {**base, "max_displacement_A": 0.1}, {**base, "max_displacement_A": 0.25},
               {**base, "max_displacement_A": 1.0}, {**base, "max_displacement_A": 1.5},
               {"kind": "trial", "accepted": False, "max_displacement_A": 9.0}]
    assert driver.accepted_step_fractions(records) == (2 / 4, 1 / 4)   # strictly above 0.25 and 1.0
    assert driver.accepted_step_fractions(records[:1]) == (0.0, 0.0)


def test_the_final_force_is_taken_at_the_last_accepted_state_not_the_last_attempt():
    records = [{"kind": "initial", "accepted": None, "max_force_component": 5.0},
               {"kind": "trial", "accepted": True, "max_force_component": 2.0},
               {"kind": "trial", "accepted": False, "max_force_component": 9.0},
               {"kind": "trial", "accepted": None, "max_force_component": None}]
    assert driver.final_max_force(records) == 2.0
    assert driver.final_max_force(records[:1]) == 5.0
    assert driver.final_max_force([{"kind": "initial", "accepted": None, "max_force_component": float("nan")}]) is None


# ---- One run --------------------------------------------------------------

def test_an_original_run_writes_a_valid_row_and_a_trace_that_matches_it(world, tmp_path):
    context = make_context(tmp_path, *world)
    task = make_tasks(context, "0b", ["direct-0"], [None])[0]
    out = driver.execute_task(context, task)
    row = out["row"]
    assert row_schema.row_problems(row) == [] and row["cap_label"] == "cinf" and row["start_kind"] == "original"
    assert row["stage"] == "0b" and row["checkpoint_id"] == "direct-0" and row["connection"] and row["branch"]
    assert row["own_minimum"] in (0, 1) and row["relaxer_sha256"] == context.relaxer_sha256
    trace = row_schema.verify_trace_link(row, context.root)
    assert len(trace) == row["force_evaluation_count"] == len(out["records"])
    assert json.loads(out["paths"]["row"].read_text()) == row
    fractions = driver.accepted_step_fractions(out["records"])
    assert row["accepted_step_fraction_gt_0p25"] == fractions[0] and row["accepted_step_fraction_gt_1p0"] == fractions[1]
    assert row["would_bind_at_1p0"] is prefix_gate.would_bind(out["records"], 1.0)
    assert row["final_max_force_component_eV_per_A"] == driver.final_max_force(out["records"])
    assert row["endpoint_connected"] is True and row["first_disconnection_attempt"] is None


def test_a_reference_run_matches_the_paired_trace_harness(world, tmp_path):
    from evaluation.outcomes import classify_outcome
    from evaluation.relax import relax
    from evaluation.runner import _arm_force_supplier
    context = make_context(tmp_path, *world)
    task = next(t for t in make_tasks(context, "0b", ["direct-0"], [None]) if t["kind"] == "reference")
    out = driver.execute_task(context, task)
    i, arm = context.inputs, context.arm("direct-0")
    own = i.refs[task["key"]]
    expected = relax(own.positions, _arm_force_supplier(arm, own.species, own.record_id + "/reference-control"),
                     i.procedure, arm=arm)
    assert first_difference(expected, {k: out["compared"][k] for k in expected}) is None
    row = out["row"]
    assert row["start_kind"] == "reference" and row["connection"] is None and row["branch"] is None
    assert row["own_minimum"] == stage0_compare.minimum_number(task["key"])
    assert row["start_id"] == f"min{row['own_minimum']}"
    assert row["final_max_force_component_eV_per_A"] == expected["final_max_force_component_eV_per_A"]


def test_the_original_run_equals_the_frozen_evaluator_at_no_cap(world, tmp_path):
    context = make_context(tmp_path, *world)
    task = make_tasks(context, "0b", ["direct-0"], [None])[0]
    out = driver.execute_task(context, task)
    saved = frozen_saved(context, [task])[0][("direct-0", task["key"])]
    assert stage0_compare.compare_original(saved, out["compared"])["problems"] == []


def test_a_capped_run_records_its_cap_and_never_moves_an_atom_further_than_it(world, tmp_path):
    context = make_context(tmp_path, *world, stiffness=40.0)
    control = driver.execute_task(context, make_tasks(context, "2", ["consistency-1-0"], [None])[0])["row"]
    capped = driver.execute_task(context, make_tasks(context, "2", ["consistency-1-0"], [0.05])[0])["row"]
    assert capped["cap_label"] == "c0p05" and capped["movement_cap_A"] == 0.05
    assert capped["capped_trial_count"] > 0 and capped["first_capped_attempt"] >= 2
    assert capped["max_accepted_displacement_A"] <= 0.05 * (1 + 1e-12)
    assert capped["capped_fraction"] == capped["capped_trial_count"] / capped["trial_count"]
    assert control["capped_trial_count"] == 0 and control["cap_label"] == "cinf" and control["movement_cap_A"] is None
    assert control["exploratory_label"] == row_schema.EXPLORATORY_LABEL == capped["exploratory_label"]
    assert capped["accepted_step_fraction_gt_0p25"] is None and control["accepted_step_fraction_gt_0p25"] is not None


# ---- Stages ---------------------------------------------------------------

def test_a_control_stage_completes_and_summarizes_binding(world, tmp_path):
    context = make_context(tmp_path, *world)
    tasks = make_tasks(context, "0b", ["consistency-1-0", "consistency-1-1"], [None])
    summary = driver.run_stage("0b", context, tasks=tasks)
    assert summary["runs"] == len(tasks) and summary["complete"] and summary["control"]["runs"] == len(tasks)
    assert summary["control"]["runs_where_c1p00_would_bind"] >= 0
    assert (context.root / "rows" / "stage0b" / "STAGE_COMPLETE.json").exists()
    assert not (context.root / "rows" / "stage0b" / "INCOMPLETE.json").exists()
    rows = sorted((context.root / "rows" / "stage0b").glob("*/cinf/*.json"))
    assert len(rows) == len(tasks) and all(row_schema.row_problems(row_schema.loads_row(p.read_text())) == [] for p in rows)


def test_a_capped_stage_runs_against_the_control_index_and_passes_the_prefix_gate(world, tmp_path):
    context = make_context(tmp_path, *world, stiffness=0.5)
    models = ["consistency-1-0"]
    driver.run_stage("0b", context, tasks=make_tasks(context, "0b", models, [None]))
    index = driver.ControlIndex.from_roots([context.root])
    capped_tasks = make_tasks(context, "2", models, [0.05, 0.25])
    summary = driver.run_stage("2", context, tasks=capped_tasks, control_index=index)
    assert summary["runs"] == len(capped_tasks)
    labels = {p.parent.name for p in (context.root / "rows" / "stage2").glob("*/*/*.json")}
    assert labels == {"c0p05", "c0p25"}


def test_the_prefix_gate_stops_a_run_whose_control_trace_disagrees(world, tmp_path):
    context = make_context(tmp_path, *world, stiffness=0.5)
    models = ["consistency-1-0"]
    driver.run_stage("0b", context, tasks=make_tasks(context, "0b", models, [None]))
    index = driver.ControlIndex.from_roots([context.root])
    capped = make_tasks(context, "2", models, [0.05])
    key = (models[0], capped[0]["kind"], driver.start_id_of(capped[0]))
    forged = copy.deepcopy(index.records[key])
    forged[0]["positions"] = forged[0]["positions"] + 1e-9
    index.records[key] = forged
    with pytest.raises(driver.PrefixGateFailure):
        driver.run_stage("2", context, tasks=capped, control_index=index)
    assert (context.root / "rows" / "stage2" / "INCOMPLETE.json").exists()


def test_a_missing_control_is_a_hard_stop(world, tmp_path):
    context = make_context(tmp_path, *world)
    with pytest.raises(driver.MissingControl):
        driver.run_stage("2", context, tasks=make_tasks(context, "2", ["consistency-1-0"], [0.05]),
                         control_index=driver.ControlIndex())


def test_capped_stages_and_stage_zero_refuse_to_start_without_their_inputs(world, tmp_path):
    context = make_context(tmp_path, *world)
    with pytest.raises(driver.DriverError, match="control index"):
        driver.run_stage("1", context)
    with pytest.raises(driver.DriverError, match="saved results"):
        driver.run_stage("0", context)
    assert not (context.root / "rows").exists()


def test_the_ledger_gates_a_stage_before_anything_is_written(world, tmp_path):
    context = make_context(tmp_path, *world)
    with pytest.raises(StageOrderError):
        driver.run_stage("0b", context, ledger=StageLedger(), tasks=make_tasks(context, "0b", ["consistency-1-0"], [None]))
    assert not (context.root / "rows").exists()


def test_the_ledger_records_completion_and_failure(world, tmp_path):
    ledger = StageLedger()
    ledger.start("A0"), ledger.finish("A0", True, replay_passed=True)
    ledger.start("0"), ledger.finish("0", True)
    context = make_context(tmp_path, *world)
    driver.run_stage("0b", context, ledger=ledger, tasks=make_tasks(context, "0b", ["consistency-1-0"], [None]))
    assert ledger.state["0b"] == "complete"
    failing = make_context(tmp_path, *world, name="second", storage_limit=10)
    ledger2 = StageLedger()
    ledger2.start("A0"), ledger2.finish("A0", True, replay_passed=True)
    ledger2.start("0"), ledger2.finish("0", True)
    with pytest.raises(launch_io.StorageExceeded):
        driver.run_stage("0b", failing, ledger=ledger2, tasks=make_tasks(failing, "0b", ["consistency-1-0"], [None]))
    assert ledger2.state["0b"] == "failed"


def test_the_attempt_ceiling_stops_a_stage_and_keeps_the_rows(world, tmp_path, monkeypatch):
    monkeypatch.setitem(driver.STAGE_SPECS, "0b", {**driver.STAGE_SPECS["0b"], "attempt_ceiling": 5})
    context = make_context(tmp_path, *world)
    with pytest.raises(driver.AttemptCeilingExceeded):
        driver.run_stage("0b", context, tasks=make_tasks(context, "0b", ["consistency-1-0"], [None]))
    stage = context.root / "rows" / "stage0b"
    rows = [row_schema.loads_row(p.read_text()) for p in stage.glob("*/cinf/*.json")]
    marker = json.loads((stage / "INCOMPLETE.json").read_text())
    counts = [r["force_evaluation_count"] for r in rows]
    # The run that crossed the ceiling is kept as evidence but not counted as completed.
    assert len(rows) == len(marker["completed_runs"]) + 1 and sum(counts) > 5
    assert marker["failed_task"] not in marker["completed_runs"]


def test_a_passed_deadline_stops_before_any_run(world, tmp_path):
    context = make_context(tmp_path, *world)
    with pytest.raises(driver.DeadlineReached):
        driver.run_stage("0b", context, deadline=time.perf_counter() - 1,
                         tasks=make_tasks(context, "0b", ["consistency-1-0"], [None]))
    assert not list((context.root / "rows" / "stage0b").glob("*/*/*.json"))
    assert (context.root / "rows" / "stage0b" / "INCOMPLETE.json").exists()


def test_the_storage_guard_stops_a_stage_after_the_runs_that_fit(world, tmp_path):
    context = make_context(tmp_path, *world, storage_limit=driver.RUN_RESERVE_BYTES + 100)
    with pytest.raises(launch_io.StorageExceeded):
        driver.run_stage("0b", context, tasks=make_tasks(context, "0b", ["consistency-1-0"], [None]))
    stage = context.root / "rows" / "stage0b"
    assert len(list(stage.glob("*/cinf/*.json"))) == 1 and (stage / "INCOMPLETE.json").exists()


def test_a_killed_launch_is_never_resumed_a_new_launch_gets_a_new_root(world, tmp_path):
    first = make_context(tmp_path, *world, name="one")
    tasks = make_tasks(first, "0b", ["consistency-1-0"], [None])
    with pytest.raises(driver.DeadlineReached):
        driver.run_stage("0b", first, deadline=time.perf_counter() - 1, tasks=tasks)
    before = sorted(p.name for p in first.root.rglob("*") if p.is_file())
    with pytest.raises(FileExistsError):                       # the root and its marker are spent
        launch_io.fresh_root(first.root)
    with pytest.raises(FileExistsError):
        driver.run_stage("0b", first, tasks=tasks)
    second = make_context(tmp_path, *world, name="two")
    driver.run_stage("0b", second, tasks=make_tasks(second, "0b", ["consistency-1-0"], [None]))
    assert sorted(p.name for p in first.root.rglob("*") if p.is_file()) == before


# ---- Stage 0 against saved results ---------------------------------------

def stage0_setup(world, tmp_path, models=("direct-0", "energy-0")):
    context = make_context(tmp_path, *world)
    tasks = make_tasks(context, "0", list(models), [None])
    saved = frozen_saved(context, tasks)
    return context, tasks, saved, Stage0Controller(expected_runs=len(tasks))


def test_stage_zero_passes_when_every_run_reproduces_the_saved_result(world, tmp_path):
    context, tasks, saved, controller = stage0_setup(world, tmp_path)
    summary = driver.run_stage("0", context, tasks=tasks, saved=saved, controller=controller)
    assert summary["runs"] == len(tasks) and controller.finish()["proceed"] is True


def test_stage_zero_stops_at_the_first_mismatch_and_writes_the_evidence(world, tmp_path):
    context, tasks, saved, controller = stage0_setup(world, tmp_path)
    target = next(t for t in tasks if t["kind"] == "original" and t["model_id"] == "direct-0")
    key = (target["model_id"], target["key"])
    saved[0][key] = {**saved[0][key], "step_count": saved[0][key]["step_count"] + 1}
    position = tasks.index(target)
    with pytest.raises(driver.Stage0Stop) as stop:
        driver.run_stage("0", context, tasks=tasks, saved=saved, controller=controller)
    assert stop.value.verdict["verdict"] == "frozen re-run required" and stop.value.task_id == driver.task_id(target)
    stage = context.root / "rows" / "stage0"
    assert len(list(stage.glob("*/cinf/*.json"))) == position + 1    # fail fast: nothing after the mismatch
    assert json.loads((stage / "STAGE0_MISMATCH.json").read_text())["task"] == driver.task_id(target)
    assert (stage / "INCOMPLETE.json").exists()
    with pytest.raises(Stage0Halted):
        controller.observe("anything", [])


def test_the_single_frozen_rerun_decides_runtime_drift_when_frozen_also_differs(world, tmp_path):
    context, tasks, saved, controller = stage0_setup(world, tmp_path)
    target = next(t for t in tasks if t["kind"] == "reference")
    key = (target["model_id"], target["key"])
    perturbed = copy.deepcopy(saved[1][key])
    perturbed["relaxation"]["step_count"] += 1
    saved[1][key] = perturbed
    with pytest.raises(driver.Stage0Stop):
        driver.run_stage("0", context, tasks=tasks, saved=saved, controller=controller)
    verdict = driver.run_frozen_rerun(context, target, saved, controller)
    assert verdict["verdict"] == "stop: runtime drift" and not verdict["proceed"]
    with pytest.raises(Stage0Halted):
        driver.run_frozen_rerun(context, target, saved, controller)       # only one re-run


@pytest.mark.parametrize("kind", ["original", "reference"])
def test_a_code_error_is_told_apart_from_runtime_drift(world, tmp_path, monkeypatch, kind):
    """The capped code differs from the saved result while the frozen relax matches it."""
    import stage2_capped.pass_copy as pass_copy
    context, tasks, saved, controller = stage0_setup(world, tmp_path)
    real = driver.relax_capped if kind == "reference" else pass_copy.relax_capped

    def broken(*args, **kwargs):
        result = real(*args, **kwargs)
        result["final_positions"] = np.asarray(result["final_positions"]) + 1e-12
        return result

    monkeypatch.setattr(driver if kind == "reference" else pass_copy, "relax_capped", broken)
    target = next(t for t in tasks if t["kind"] == kind)
    with pytest.raises(driver.Stage0Stop) as stop:
        driver.run_stage("0", context, tasks=tasks, saved=saved, controller=controller)
    assert stop.value.task_id == driver.task_id(target)
    verdict = driver.run_frozen_rerun(context, target, saved, controller)
    assert verdict["verdict"] == "stop: code error" and not verdict["proceed"]


def test_an_incomplete_stage_zero_is_not_a_pass(world, tmp_path):
    context, tasks, saved, _ = stage0_setup(world, tmp_path)
    controller = Stage0Controller(expected_runs=len(tasks) + 1)
    with pytest.raises(driver.Stage0Stop) as stop:
        driver.run_stage("0", context, tasks=tasks, saved=saved, controller=controller)
    assert stop.value.verdict["verdict"] == "stop: incomplete comparison"


# ---- Real study inputs (zero-force arm) ----------------------------------

@pytest.fixture(scope="module")
def study():
    return driver.load_study_inputs()


@needs_real_files
def test_the_real_study_inputs_load_and_bind_to_the_accepted_run(study):
    assert len(study.starts) == 10 and set(study.refs) >= set(stage0_compare.OWN_IDS)
    assert study.procedure["max_force_evaluations"] == 1500 and study.procedure["convergence_tolerance_eV_per_A"] == 0.01
    assert set(study.seen) == set(study.starts) and all(v is True for v in study.seen.values())  # all ten seen minima
    assert study.reference_ids == stage0_compare.OWN_IDS
    tasks = driver.build_schedule("0", sorted(study.starts), study.reference_ids)
    assert len(tasks) == 80 and tasks[0]["model_id"] == "energy-0"


@needs_real_files
def test_changed_frozen_inputs_or_evaluation_sources_are_refused(tmp_path, monkeypatch):
    real = driver._evaluation_pin
    monkeypatch.setattr(driver, "_evaluation_pin", lambda root: {**real(root), "evaluation/runner.py": "0" * 64})
    with pytest.raises(admission.AdmissionError, match="evaluation sources"):
        driver.load_study_inputs()


@needs_real_files
def test_real_starts_prevalidate_and_run_with_a_zero_force_arm(study, tmp_path):
    species = next(iter(study.starts.values()))["species"]
    root = launch_io.fresh_root(tmp_path / "real")
    provider = lambda model_id: callable_arm(lambda p: np.zeros_like(p), species, model_id)  # noqa: E731
    context = driver.Context(root, "L-real", study, provider)
    plans, bound = context.plans("direct-0")
    assert len(plans) == 10 and bound == "approved"
    original = driver.execute_task(context, {"stage": "0", "model_id": "direct-0", "cap": None,
                                             "kind": "original", "key": sorted(study.starts)[0]})["row"]
    assert original["force_evaluation_count"] == 1 and original["converged"] and original["connection"]
    own = study.classification[sorted(study.starts)[0]]["reference_class"]
    assert original["own_minimum"] == stage0_compare.minimum_number(own) and original["start_id"].startswith("aspirin/")
    reference = driver.execute_task(context, {"stage": "0", "model_id": "direct-0", "cap": None,
                                              "kind": "reference", "key": stage0_compare.OWN_IDS[2]})["row"]
    assert reference["own_minimum"] == 6 and reference["recovered"] is True and reference["step_count"] == 0
    assert row_schema.row_problems(original) == [] and row_schema.row_problems(reference) == []


@needs_real_files
def test_a_changed_frozen_input_is_refused(monkeypatch):
    real = admission.relocate
    group9 = admission.group9_root()
    other = group9 / ".local-workflow/e3x-budget-worktree/reports/landscape17-endpoints.json"

    def redirect(path, root):
        found = real(path, root)
        return other if "landscape17-aspirin-split" in str(found) else found

    monkeypatch.setattr(admission, "relocate", redirect)
    with pytest.raises(admission.AdmissionError, match="frozen input changed"):
        driver.load_study_inputs()


@needs_real_files
def test_a_changed_start_binding_is_refused(monkeypatch):
    import evaluation.validation_controls as controls
    real = controls.validation_inputs

    def altered(archive, manifest, classification):
        starts, refs, policy, binding = real(archive, manifest, classification)
        return starts, refs, policy, {**binding, "validation_payload_sha256": "0" * 64}

    monkeypatch.setattr(controls, "validation_inputs", altered)
    with pytest.raises(admission.AdmissionError, match="start"):
        driver.load_study_inputs()


@needs_real_files
def test_a_policy_that_differs_from_the_approved_one_is_refused_before_any_force_call(study):
    from dataclasses import replace
    from evaluation.runner import SplitBindingError
    species = next(iter(study.starts.values()))["species"]
    calls = []
    arm = callable_arm(lambda p: (calls.append(1), np.zeros_like(p))[1], species, "counted")
    first = next(f for f in study.policy.__dataclass_fields__ if isinstance(getattr(study.policy, f), float))
    other = replace(study.policy, **{first: getattr(study.policy, first) * 1.5})
    with pytest.raises(SplitBindingError):
        driver.prevalidate(study.starts, arm, study.procedure, study.classification, study.refs, other, study.manifest)
    with pytest.raises(SplitBindingError):
        driver.prevalidate(study.starts, arm, study.procedure, study.classification, study.refs, "not a policy",
                           study.manifest)
    assert not calls


def test_the_control_fields_are_assigned_to_the_right_columns():
    """Accepted steps of 0.5 and 2.0 A: the fraction above 0.25 is 1.0, above 1.0 is 0.5."""
    from types import SimpleNamespace
    position = np.zeros((3, 3))
    records = [
        {"attempt_index": 1, "kind": "initial", "accepted": None, "max_displacement_A": 0.0,
         "max_force_component": 1.0, "positions": position, "scale_s": 0.2, "force_norm_max_f": 1.0},
        {"attempt_index": 2, "kind": "trial", "accepted": True, "max_displacement_A": 0.5,
         "max_force_component": 0.8, "positions": position, "scale_s": 0.5, "force_norm_max_f": 1.0},
        {"attempt_index": 3, "kind": "trial", "accepted": True, "max_displacement_A": 2.0,
         "max_force_component": 0.6, "positions": position, "scale_s": 2.0, "force_norm_max_f": 1.0}]
    cap_record = {"trial_count": 2, "capped_trial_count": 0, "first_capped_attempt": None,
                  "accepted_trial_count": 2, "rejected_trial_count": 0, "refused_trial_count": 0,
                  "max_accepted_displacement_A": 2.0, "accepted_travel_A": 2.5, "attempted_travel_A": 2.5}
    task = {"stage": "0", "model_id": "direct-0", "cap": None, "kind": "original", "key": "aspirin/13/plus"}
    row = driver.make_result_row(
        context=SimpleNamespace(launch_id="L", relaxer_sha256="a" * 64), task=task, category="nonconverged",
        stop_reason="budget_exhausted", converged=False, budget_exhausted=True, step_count=2,
        force_evaluation_count=3, connection="13", branch="plus", own_minimum=1, cap_record=cap_record,
        records=records, final_positions=position, elapsed=0.1, trace_file="x.npy", trace_sha256="0" * 64)
    assert row["accepted_step_fraction_gt_0p25"] == 1.0 and row["accepted_step_fraction_gt_1p0"] == 0.5
    assert row["would_bind_at_1p0"] is True            # trial 3 asks for 2.0 A
    assert row["final_max_force_component_eV_per_A"] == 0.6 and row["max_accepted_displacement_A"] == 2.0
    assert row["endpoint_connected"] is True and row["first_disconnection_attempt"] is None


def test_a_control_run_from_far_away_reports_that_c1p00_would_bind(world, tmp_path):
    land, manifest, classification = world
    inputs = make_inputs(land, manifest, classification)
    species = next(iter(inputs.starts.values()))["species"]
    root = launch_io.fresh_root(tmp_path / "far")
    provider = lambda model_id: callable_arm(fx.harmonic(TARGET + 30.0, 4.0), species, model_id)  # noqa: E731
    context = driver.Context(root, "L-far", inputs, provider)
    out = driver.execute_task(context, make_tasks(context, "0b", ["consistency-1-0"], [None])[0])
    row = out["row"]
    assert row["would_bind_at_1p0"] is True and prefix_gate.would_bind(out["records"], 1.0)
    near = driver.execute_task(make_context(tmp_path, *world, name="near", stiffness=0.5),
                               make_tasks(make_context(tmp_path, *world, name="near2", stiffness=0.5),
                                          "0b", ["consistency-1-0"], [None])[0])["row"]
    assert near["would_bind_at_1p0"] is False
