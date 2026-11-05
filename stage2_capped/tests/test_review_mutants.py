"""Tests that kill the mutants the M4 mutation run left alive (review finding F09). Each test names
what the mutant changed. Synthetic inputs and fixture arms; no model."""
import copy
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

from evaluation.adapter import callable_arm
from stage2_capped import admission, diff_gate, driver, launch_io, prefix_gate, row_schema
from stage2_capped.tests import fixtures as fx
from stage2_capped.tests.test_driver import (TARGET, make_context, make_inputs, make_tasks, needs_real_files,
                                             run_test_stage, world)  # noqa: F401


# ---- Prefix gate in the driver: both capped stages, with a forged control (D15) -------------

@pytest.mark.parametrize("stage, models", [("1", ["direct-0"]), ("2", ["consistency-1-0"])])
def test_the_prefix_gate_runs_in_both_capped_stages(world, tmp_path, stage, models):
    context = make_context(tmp_path, *world, stiffness=0.5)
    run_test_stage("0b", context, tasks=make_tasks(context, "0b", models, [None]))   # controls for these models
    index = driver.ControlIndex.from_roots([context.root], allow_custom=True)
    capped = make_tasks(context, stage, models, [0.05])
    key = (models[0], capped[0]["kind"], driver.start_id_of(capped[0]))
    forged = copy.deepcopy(index.records[key])
    forged[0]["positions"] = forged[0]["positions"] + 1e-9
    index.records[key] = forged
    with pytest.raises(driver.PrefixGateFailure, match="c0p05|attempt|prefix|differ"):
        run_test_stage(stage, context, tasks=capped, control_index=index)
    assert (context.root / "rows" / f"stage{stage}" / "INCOMPLETE.json").exists()


# ---- MODEL_IDENTITY against the model ids and the launch records (D26 to D28) ---------------

def parse_model_id(model_id):
    match = re.fullmatch(r"(energy|direct)-(\d+)", model_id)
    if match:
        return match.group(1), int(match.group(2)), None
    match = re.fullmatch(r"consistency-(0\.1|1)-(\d+)", model_id)
    return "consistency", int(match.group(2)), float(match.group(1))


def test_the_model_identity_table_agrees_with_every_model_id():
    assert set(driver.MODEL_IDENTITY) == set(admission.ALL_IDS) and len(admission.ALL_IDS) == 7
    for model_id, identity in driver.MODEL_IDENTITY.items():
        assert identity == parse_model_id(model_id), model_id
        assert isinstance(identity[1], int) and not isinstance(identity[1], bool)
    assert {m: driver.MODEL_IDENTITY[m][2] for m in admission.LAMBDA1_IDS} == {m: 1.0 for m in admission.LAMBDA1_IDS}
    assert {driver.MODEL_IDENTITY[m][2] for m in admission.PRIMARY_IDS} == {None, 0.1}


@needs_real_files
def test_the_model_identity_table_agrees_with_the_recorded_launch_configs():
    records = json.loads((admission.group9_root() / admission.RECOVERY_RECORDS).read_text())["models"]
    assert set(records) == set(admission.PRIMARY_IDS)
    for model_id, entry in records.items():
        config = entry["launch_record"]["config"]
        arm, seed, lam = driver.MODEL_IDENTITY[model_id]
        assert config["arm"] == arm and config["seed"] == seed, model_id
        if arm == "consistency":
            assert config["consistency_strength"] == lam, model_id


def test_a_row_carries_the_arm_seed_and_lambda_of_its_model(world, tmp_path):
    context = make_context(tmp_path, *world)
    for model_id in ("consistency-1-1", "direct-0"):
        task = make_tasks(context, "0b", [model_id], [None])[0]
        row = driver.execute_task(context, task)["row"]
        arm, seed, lam = driver.MODEL_IDENTITY[model_id]
        assert (row["arm"], row["seed"], row["consistency_lambda"]) == (arm, seed, lam)
        assert (row["exploratory_label"] is not None) == (lam == 1.0)


# ---- Travel: accepted versus attempted (D30, D31) -------------------------------------------

def test_accepted_and_attempted_travel_are_the_sums_over_the_right_trials(world, tmp_path):
    context = make_context(tmp_path, *world, stiffness=40.0)           # stiff: some trial steps are rejected
    outs = [driver.execute_task(context, t) for t in make_tasks(context, "0b", ["consistency-1-0"], [None])]
    assert any(o["row"]["rejected_trial_count"] > 0 for o in outs)
    for out in outs:
        trials = [r for r in out["records"] if r["kind"] == "trial"]
        accepted = sum(r["max_displacement_A"] for r in trials if r["accepted"] is True)
        attempted = sum(r["max_displacement_A"] for r in trials if r["refused"] is None)
        row = out["row"]
        assert row["accepted_travel_A"] == pytest.approx(accepted, rel=1e-12, abs=1e-15)
        assert row["attempted_travel_A"] == pytest.approx(attempted, rel=1e-12, abs=1e-15)
        assert row["accepted_trial_count"] == sum(r["accepted"] is True for r in trials)
        assert row["rejected_trial_count"] == sum(r["accepted"] is False and r["refused"] is None for r in trials)
        if row["rejected_trial_count"]:
            assert row["attempted_travel_A"] > row["accepted_travel_A"]      # rejected steps add only to attempted
        else:
            assert row["attempted_travel_A"] == pytest.approx(row["accepted_travel_A"], rel=1e-12, abs=1e-15)


# ---- Reference classification and supplier arguments (D21 to D25) ---------------------------

def test_a_reference_run_scores_its_final_geometry_with_the_accepted_arguments(world, tmp_path, monkeypatch):
    import evaluation.outcomes as outcomes
    import evaluation.runner as runner
    context = make_context(tmp_path, *world)
    task = make_tasks(context, "0b", ["consistency-1-0"], [None])
    task = next(t for t in task if t["kind"] == "reference")
    own = context.inputs.refs[task["key"]]
    scored, suppliers = [], []
    real_score, real_supplier = outcomes.classify_outcome, runner._arm_force_supplier

    def spy_score(*args, **kwargs):
        scored.append((args, kwargs))
        return real_score(*args, **kwargs)

    def spy_supplier(*args, **kwargs):
        suppliers.append((args, kwargs))
        return real_supplier(*args, **kwargs)

    monkeypatch.setattr(outcomes, "classify_outcome", spy_score)
    monkeypatch.setattr(runner, "_arm_force_supplier", spy_supplier)
    result, outcome, records, _ = driver.run_reference(context, task)
    assert len(suppliers) == 1 and len(scored) == 1
    (arm_arg, species_arg, record_id), _ = suppliers[0]
    assert arm_arg is context.arm("consistency-1-0") and record_id == own.record_id + "/reference-control"
    assert np.array_equal(species_arg, own.species)
    (final, start, refs, policy), kwargs = scored[0]
    assert final.record_id == own.record_id + "/trace-final" and start is own
    assert np.array_equal(final.positions, result["final_positions"]) and not final.forces.any()
    assert refs is context.inputs.refs and policy is context.inputs.policy
    assert kwargs == {"budget_exhausted": result["budget_exhausted"], "converged": result["converged"],
                      "forces_finite": result["forces_finite"], "stop_reason": result["failure_reason"],
                      "seen_minimum": True}
    assert outcome["category"] == driver.classify_reference(context, task, result)["category"]


# ---- The arm provider is called once per model (D48) ----------------------------------------

def test_each_models_arm_is_built_once_for_a_whole_stage(world, tmp_path):
    context = make_context(tmp_path, *world)
    species = next(iter(context.inputs.starts.values()))["species"]
    built = []

    def provider(model_id):
        built.append(model_id)
        return callable_arm(fx.harmonic(TARGET, 4.0), species, model_id)

    context.arm_provider = provider
    run_test_stage("0b", context, tasks=make_tasks(context, "0b", ["consistency-1-0", "consistency-1-1"], [None]))
    assert sorted(built) == ["consistency-1-0", "consistency-1-1"]


# ---- Control summary and the c = 1.0 binding test (D54, D36) --------------------------------

def test_the_control_summary_counts_exactly_the_runs_where_c1p00_would_bind(world, tmp_path):
    far = make_context(tmp_path, *world, name="far")
    far.arm_provider = lambda model_id: callable_arm(
        fx.harmonic(TARGET + 30.0, 4.0), next(iter(far.inputs.starts.values()))["species"], model_id)
    near = make_context(tmp_path, *world, name="near", stiffness=0.5)
    tasks = make_tasks(far, "0b", ["consistency-1-0"], [None])
    binding = [driver.execute_task(far, t)["records"] for t in tasks[:2]]
    quiet = [driver.execute_task(near, t)["records"] for t in tasks[:1]]
    assert all(prefix_gate.would_bind(r, 1.0) for r in binding) and not prefix_gate.would_bind(quiet[0], 1.0)
    mixed = driver.control_summary({"a": binding[0], "b": binding[1], "c": quiet[0]})
    assert mixed["runs_where_c1p00_would_bind"] == 2 and mixed["c1p00_duplicates_control"] is False
    only_quiet = driver.control_summary({"c": quiet[0]})
    assert only_quiet["runs_where_c1p00_would_bind"] == 0 and only_quiet["c1p00_duplicates_control"] is True
    assert driver.control_summary({"a": binding[0]})["runs_where_c1p00_would_bind"] == 1


def mid_size_trace(step=0.5):
    """One accepted step of `step` A (default 0.5): a c = 0.25 cap would bind on it, a c = 1.0 cap would not."""
    position = np.zeros((3, 3))
    return [{"attempt_index": 1, "kind": "initial", "accepted": None, "max_displacement_A": 0.0,
             "max_force_component": 1.0, "positions": position, "scale_s": 0.2, "force_norm_max_f": 1.0},
            {"attempt_index": 2, "kind": "trial", "accepted": True, "max_displacement_A": step,
             "max_force_component": 0.8, "positions": position, "scale_s": step, "force_norm_max_f": 1.0}]


def test_the_c1p00_binding_count_uses_the_1p0_cap_not_a_smaller_one():
    mid = mid_size_trace()
    assert prefix_gate.would_bind(mid, 0.25) and not prefix_gate.would_bind(mid, 1.0)
    summary = driver.control_summary({"mid": mid})
    assert summary["runs_where_c1p00_would_bind"] == 0 and summary["c1p00_duplicates_control"] is True


def test_a_control_row_reports_binding_at_1p0_from_the_1p0_cap_not_a_smaller_one():
    from types import SimpleNamespace
    mid = mid_size_trace()
    cap_record = {"trial_count": 1, "capped_trial_count": 0, "first_capped_attempt": None, "accepted_trial_count": 1,
                  "rejected_trial_count": 0, "refused_trial_count": 0, "max_accepted_displacement_A": 0.5,
                  "accepted_travel_A": 0.5, "attempted_travel_A": 0.5}
    task = {"stage": "0", "model_id": "direct-0", "cap": None, "kind": "original", "key": "aspirin/13/plus"}
    row = driver.make_result_row(
        context=SimpleNamespace(launch_id="L", relaxer_sha256="a" * 64), task=task, category="unmatched_converged",
        stop_reason=None, converged=True, budget_exhausted=False, step_count=1, force_evaluation_count=2,
        connection="13", branch="plus", own_minimum=1, cap_record=cap_record, records=mid,
        final_positions=mid[-1]["positions"], elapsed=0.1, trace_file="x.npy", trace_sha256="0" * 64)
    assert row["would_bind_at_1p0"] is False and row_schema.row_problems(row) == []


def trial_record(index, s, f):
    return {"attempt_index": index, "kind": "trial", "scale_s": s, "force_norm_max_f": f}


def test_a_cap_binds_exactly_when_the_step_would_exceed_it_never_at_equality():
    assert prefix_gate.would_bind([trial_record(2, 0.5, 2.0)], 1.0) is False              # s f == c exactly
    assert prefix_gate.would_bind([trial_record(2, 0.5, 2.0000000000000004)], 1.0) is True  # one ulp over
    assert prefix_gate.would_bind([trial_record(2, 0.5, 1.9999999999999998)], 1.0) is False
    assert prefix_gate.would_bind([trial_record(2, 0.2, 1.0)], 0.25) is False             # 0.2 A < 0.25 A
    assert prefix_gate.would_bind([trial_record(2, 0.4, 1.0)], 0.25) is True
    initial = {"attempt_index": 1, "kind": "initial", "scale_s": 10.0, "force_norm_max_f": 10.0}
    assert prefix_gate.would_bind([initial], 0.05) is False                                # only trials can bind
    assert prefix_gate.first_binding_attempt([trial_record(2, 0.1, 1.0), trial_record(3, 0.9, 1.0)], 0.25) == 3


# ---- Connection partitions are validated before any force call (D47) ------------------------

def test_a_manifest_that_names_a_connection_twice_is_refused_before_any_force_call(world, tmp_path):
    land, manifest, classification = world
    inputs = make_inputs(land, manifest, classification)
    ambiguous = copy.deepcopy(manifest)
    ambiguous["groups"]["aspirin"]["train"] = list(ambiguous["groups"]["aspirin"]["validation"])
    species = next(iter(inputs.starts.values()))["species"]
    calls = []
    arm = callable_arm(lambda p: (calls.append(1), np.zeros_like(p))[1], species, "counted")
    from evaluation.runner import PartitionOverlapError
    with pytest.raises((PartitionOverlapError, ValueError), match="partitions|ambiguous|manifest|identity"):
        driver.prevalidate(inputs.starts, arm, inputs.procedure, inputs.classification, inputs.refs,
                           inputs.policy, ambiguous)
    assert calls == []


def test_prevalidate_calls_the_connection_partition_check(world, monkeypatch):
    import evaluation.runner as runner
    land, manifest, classification = world
    inputs = make_inputs(land, manifest, classification)
    species = next(iter(inputs.starts.values()))["species"]
    seen = []
    real = runner._validate_connection_partitions
    monkeypatch.setattr(runner, "_validate_connection_partitions", lambda *a: (seen.append(a), real(*a))[1])
    driver.prevalidate(inputs.starts, callable_arm(fx.harmonic(TARGET, 4.0), species, "arm"), inputs.procedure,
                       inputs.classification, inputs.refs, inputs.policy, manifest)
    assert len(seen) == 1 and seen[0][0] is inputs.starts and seen[0][1] is manifest


# ---- The storage reserve is checked before every run (D29) ----------------------------------

def test_a_run_is_refused_when_the_files_plus_the_reserve_would_pass_the_limit(world, tmp_path):
    for name, extra, expect_error in (("edge-ok", 0, False), ("edge-over", -1, True)):
        context = make_context(tmp_path, *world, name=name)
        task = make_tasks(context, "0b", ["consistency-1-0"], [None])[0]
        context.guard.limit = launch_io.total_size(context.root) + context.reserve_bytes + extra
        if expect_error:
            with pytest.raises(launch_io.StorageExceeded):
                driver.execute_task(context, task)
            assert not (context.root / "rows").exists()                   # nothing was written
        else:
            assert driver.execute_task(context, task)["row"]["start_kind"] in ("original", "reference")


# ---- The diff gate's trailing-newline rule (X06) --------------------------------------------

@pytest.mark.parametrize("frozen_text, derived_text", [("a\nb\n", "a\nb"), ("a\nb", "a\nb\n")])
def test_the_diff_gate_reports_a_trailing_newline_difference_by_name(tmp_path, frozen_text, derived_text):
    (tmp_path / "frozen.py").write_bytes(frozen_text.encode())
    (tmp_path / "derived.py").write_bytes(derived_text.encode())
    allowlist = {"targets": {"t": {"frozen_file": "frozen.py", "derived_file": "derived.py",
                                   "frozen_sha256": diff_gate.sha256_bytes(frozen_text.encode()),
                                   "allowed_categories": ["x"], "hunks": []}}}
    path = tmp_path / "allowlist.json"
    path.write_text(json.dumps(allowlist))
    problems = diff_gate.check_gate("t", allowlist_path=path, root=tmp_path)
    assert any("trailing newline differs" in p for p in problems)
    same = tmp_path / "same.py"
    same.write_bytes(frozen_text.encode())
    allowlist["targets"]["t"]["derived_file"] = "same.py"
    path.write_text(json.dumps(allowlist))
    assert diff_gate.check_gate("t", allowlist_path=path, root=tmp_path) == []


@pytest.mark.parametrize("step", [0.5, 0.95, 0.9999999999999999])
def test_the_1p0_threshold_is_exactly_1p0_in_the_summary_and_in_the_row(step):
    """An accepted step of 0.95 A binds a cap of 0.9 but not the c = 1.0 cap; a threshold of 0.9 would flag it."""
    from types import SimpleNamespace
    trace = mid_size_trace(step)
    assert prefix_gate.would_bind(trace, 0.9) == (step > 0.9) and not prefix_gate.would_bind(trace, 1.0)
    assert driver.control_summary({"t": trace})["runs_where_c1p00_would_bind"] == 0
    assert driver.accepted_step_fractions(trace) == (1.0, 0.0)
    cap_record = {"trial_count": 1, "capped_trial_count": 0, "first_capped_attempt": None, "accepted_trial_count": 1,
                  "rejected_trial_count": 0, "refused_trial_count": 0, "max_accepted_displacement_A": step,
                  "accepted_travel_A": step, "attempted_travel_A": step}
    task = {"stage": "0", "model_id": "direct-0", "cap": None, "kind": "original", "key": "aspirin/13/plus"}
    row = driver.make_result_row(
        context=SimpleNamespace(launch_id="L", relaxer_sha256="a" * 64), task=task, category="unmatched_converged",
        stop_reason=None, converged=True, budget_exhausted=False, step_count=1, force_evaluation_count=2,
        connection="13", branch="plus", own_minimum=1, cap_record=cap_record, records=trace,
        final_positions=trace[-1]["positions"], elapsed=0.1, trace_file="x.npy", trace_sha256="0" * 64)
    assert row["would_bind_at_1p0"] is False


def test_a_reference_run_that_runs_out_of_budget_is_scored_with_its_own_flags(world, tmp_path, monkeypatch):
    """The first spy test uses a converging run, where the constants a mutant might hard-code equal the real values.
    Here the budget runs out, so converged, budget_exhausted and stop_reason all differ from those constants."""
    import evaluation.outcomes as outcomes
    context = make_context(tmp_path, *world, stiffness=0.5)
    context.inputs.procedure = fx.procedure(max_force_evaluations=3)
    task = next(t for t in make_tasks(context, "0b", ["consistency-1-0"], [None]) if t["kind"] == "reference")
    scored, real = [], outcomes.classify_outcome
    monkeypatch.setattr(outcomes, "classify_outcome", lambda *a, **k: (scored.append(k), real(*a, **k))[1])
    result, outcome, _, _ = driver.run_reference(context, task)
    assert result["converged"] is False and result["budget_exhausted"] is True and result["forces_finite"] is True
    assert result["failure_reason"] == "budget_exhausted" and outcome["category"] == "nonconverged"
    assert scored == [{"budget_exhausted": True, "converged": False, "forces_finite": True,
                       "stop_reason": "budget_exhausted", "seen_minimum": True}]


def test_a_non_finite_reference_run_is_scored_as_not_finite(world, tmp_path, monkeypatch):
    import evaluation.outcomes as outcomes
    context = make_context(tmp_path, *world)
    species = next(iter(context.inputs.starts.values()))["species"]
    calls, force = [], fx.harmonic(TARGET, 4.0)
    context.arm_provider = lambda model_id: callable_arm(
        lambda p: (calls.append(1), np.full_like(p, np.nan) if len(calls) == 2 else force(p))[1], species, model_id)
    task = next(t for t in make_tasks(context, "0b", ["consistency-1-0"], [None]) if t["kind"] == "reference")
    scored, real = [], outcomes.classify_outcome
    monkeypatch.setattr(outcomes, "classify_outcome", lambda *a, **k: (scored.append(k), real(*a, **k))[1])
    result, outcome, _, _ = driver.run_reference(context, task)
    assert result["forces_finite"] is False and result["failure_reason"] == "nonfinite_force"
    assert scored[0]["forces_finite"] is False and scored[0]["stop_reason"] == "nonfinite_force"
    assert scored[0]["converged"] is False and outcome["category"] == "nonconverged"


def test_a_control_trace_that_is_shorter_than_its_row_is_refused(world, tmp_path):
    import hashlib
    context = make_context(tmp_path, *world, stiffness=0.5)
    run_test_stage("0b", context, tasks=make_tasks(context, "0b", ["consistency-1-0"], [None]))
    path = sorted((context.root / "rows" / "stage0b").glob("*/cinf/*.json"))[0]
    row = json.loads(path.read_text())
    trace_path = context.root / row["trace_file"]
    trace = row_schema.read_trace(trace_path)
    assert len(trace) == row["force_evaluation_count"] and len(trace) > 1
    np.save(str(trace_path), trace[:-1], allow_pickle=False)
    row["trace_sha256"] = hashlib.sha256(trace_path.read_bytes()).hexdigest()  # the hash agrees; the length does not
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="trace length differs"):
        driver.ControlIndex.from_roots([context.root], allow_custom=True)


def test_the_0p25_and_1p0_step_fractions_are_pinned_from_both_sides():
    """Accepted steps 0.25, 0.28, 1.0 and 1.05 A: above 0.25 are three (0.25 itself is not), above 1.0 is one."""
    position = np.zeros((3, 3))
    records = [{"attempt_index": 1, "kind": "initial", "accepted": None, "max_displacement_A": 0.0,
                "positions": position}]
    for index, step in enumerate((0.25, 0.28, 1.0, 1.05), start=2):
        records.append({"attempt_index": index, "kind": "trial", "accepted": True, "max_displacement_A": step,
                        "positions": position})
    records.append({"attempt_index": 6, "kind": "trial", "accepted": False, "max_displacement_A": 5.0,
                    "positions": position})                                  # a rejected trial is not counted
    assert driver.accepted_step_fractions(records) == (3 / 4, 1 / 4)
    assert driver.accepted_step_fractions(records[:1]) == (0.0, 0.0)


@pytest.mark.parametrize("step, binds", [(1.05, True), (1.0000000000000002, True), (1.0, False)])
def test_a_step_just_above_1p0_binds_the_c1p00_cap_in_the_row_the_summary_and_the_duplicate_flag(step, binds):
    from types import SimpleNamespace
    trace = mid_size_trace(step)
    assert prefix_gate.would_bind(trace, 1.0) is binds
    summary = driver.control_summary({"t": trace})
    assert summary["runs_where_c1p00_would_bind"] == int(binds) and summary["c1p00_duplicates_control"] is (not binds)
    cap_record = {"trial_count": 1, "capped_trial_count": 0, "first_capped_attempt": None, "accepted_trial_count": 1,
                  "rejected_trial_count": 0, "refused_trial_count": 0, "max_accepted_displacement_A": step,
                  "accepted_travel_A": step, "attempted_travel_A": step}
    task = {"stage": "0", "model_id": "direct-0", "cap": None, "kind": "original", "key": "aspirin/13/plus"}
    row = driver.make_result_row(
        context=SimpleNamespace(launch_id="L", relaxer_sha256="a" * 64), task=task, category="unmatched_converged",
        stop_reason=None, converged=True, budget_exhausted=False, step_count=1, force_evaluation_count=2,
        connection="13", branch="plus", own_minimum=1, cap_record=cap_record, records=trace,
        final_positions=trace[-1]["positions"], elapsed=0.1, trace_file="x.npy", trace_sha256="0" * 64)
    assert row["would_bind_at_1p0"] is binds
    assert row["accepted_step_fraction_gt_1p0"] == (1.0 if step > 1.0 else 0.0)
