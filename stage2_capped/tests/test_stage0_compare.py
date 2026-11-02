"""M4: Stage 0 loader and comparator. A self-compare of the real saved results
passes and every single-field perturbation fails; no model is loaded."""
import copy
import json

import numpy as np
import pytest

from stage2_capped import admission
from stage2_capped import stage0_compare as s0


def real_files():
    try:
        root = admission.group9_root()
        return (root / s0.RECOVERY_RUN / "pass-energy-0" / "raw-branches" / "000.json").is_file() and \
            (root / s0.PAIRED_TRACE_RUN / "04-energy-0" / "relaxation.json").is_file()
    except admission.AdmissionError:
        return False


needs_real_files = pytest.mark.skipif(not real_files(), reason="saved Group9 results are absent")


@pytest.fixture(scope="module")
def saved():
    return s0.load_saved_original(), s0.load_saved_reference()


def self_sets(saved):
    original, reference = saved
    return copy.deepcopy(original), {k: copy.deepcopy(v["relaxation"]) for k, v in reference.items()}


# ---- first_difference ----------------------------------------------------

def test_first_difference_is_bit_exact_and_type_strict():
    assert s0.first_difference({"a": [1.5, 2.5]}, {"a": [1.5, 2.5]}) is None
    assert s0.first_difference(0.0, -0.0) == "/"                          # sign of zero
    assert s0.first_difference(0.1, float(np.nextafter(0.1, 1))) == "/"   # one ulp
    assert s0.first_difference(True, 1) == "/" and s0.first_difference(1, 1.0) == "/"
    assert s0.first_difference([1, 2], [1, 2, 3]) == "/"
    assert s0.first_difference({"a": 1}, {"a": 1, "b": 2}) == "/"
    assert s0.first_difference({"a": [1, {"b": 2}]}, {"a": [1, {"b": 3}]}) == "/a/1/b"
    assert s0.first_difference(np.array([[1.0, 2.0]]), [[1.0, 2.0]]) is None
    assert s0.first_difference(np.float64(1.0), 1.0) is None
    assert s0.first_difference(None, None) is None and s0.first_difference("a", "b") == "/"
    assert s0.first_difference(float("nan"), float("nan")) is None        # same bits


def test_the_very_same_object_is_never_a_difference_even_if_it_cannot_be_compared():
    class Opaque:
        def __eq__(self, other):
            raise AssertionError("compared")
    obj = Opaque()
    assert s0.first_difference({"reference": obj}, {"reference": obj}) is None


def test_the_schedule_is_80_tasks_in_the_accepted_order():
    keys = [f"aspirin/{g}/{b}" for g in ("13", "18", "18_2", "66", "78") for b in ("plus", "minus")]
    tasks = s0.schedule(keys)
    assert len(tasks) == s0.EXPECTED_TOTAL == 80 and len(set(tasks)) == 80
    assert tasks[0] == "energy-0/original/aspirin/13/minus" and tasks[10] == "energy-0/reference/min1"
    assert tasks[16].startswith("direct-0/original/") and tasks[-1] == "consistency-0.1-1/reference/min9"
    with pytest.raises(s0.SavedResultError):
        s0.schedule(keys[:9])


def test_minimum_numbers_come_from_the_reference_ids():
    assert [s0.minimum_number(i) for i in s0.OWN_IDS] == [1, 3, 6, 7, 8, 9]
    with pytest.raises(s0.SavedResultError):
        s0.minimum_number("Landscape17/aspirin/Conformations/other.xyz")


# ---- Synthetic comparator behavior ---------------------------------------

def row(**changes):
    base = {"category": "nonconverged", "stop_reason": "budget_exhausted", "step_count": 700,
            "force_evaluation_count": 1500, "converged": False, "budget_exhausted": True,
            "forces_finite": True, "nonfinite_force": False, "status": "budget_exhausted",
            "rmsd_A": 0.5, "_audit_record": {"final_positions": [[0.1, -0.0, 2.0]]}}
    base.update(changes)
    return base


@pytest.mark.parametrize("field, value", [
    ("category", "matched_reference"), ("stop_reason", None), ("step_count", 701),
    ("force_evaluation_count", 1499), ("converged", True), ("budget_exhausted", False),
    ("forces_finite", False), ("nonfinite_force", True), ("status", "converged")])
def test_each_hard_stop_field_is_compared(field, value):
    outcome = s0.compare_original(row(), row(**{field: value}))
    assert outcome["problems"] and field in outcome["problems"][0]


def test_original_final_positions_are_compared_bitwise():
    assert s0.compare_original(row(), row())["problems"] == []
    other = row()
    other["_audit_record"] = {"final_positions": [[0.1, 0.0, 2.0]]}  # -0.0 versus 0.0
    assert s0.compare_original(row(), other)["problems"]
    ulp = row()
    ulp["_audit_record"] = {"final_positions": [[float(np.nextafter(0.1, 1)), -0.0, 2.0]]}
    assert s0.compare_original(row(), ulp)["problems"]


def test_rmsd_difference_is_informational_never_a_mismatch():
    outcome = s0.compare_original(row(), row(rmsd_A=0.5 + 1e-13))
    assert outcome["problems"] == [] and outcome["informational"]["rmsd_A_difference"] > 0


def relaxation(**changes):
    base = {"status": "budget_exhausted", "converged": False, "budget_exhausted": True,
            "forces_finite": True, "failure_reason": "budget_exhausted", "step_count": 3,
            "force_evaluation_count": 7, "final_positions": [[0.0, 1.0, 2.0]],
            "final_max_force_component_eV_per_A": 0.3, "refusal_details": [],
            "steps": [{"accepted_step_size": 0.2, "trial_count": 1}],
            "history": {"evaluation_count": 7, "force_maximums": [1.0, 0.5], "maximum_seen_eV_per_A": 1.0},
            "force_history": {"evaluation_count": 7, "force_maximums": [1.0, 0.5], "maximum_seen_eV_per_A": 1.0}}
    base.update(changes)
    return base


def test_reference_comparison_covers_every_saved_field_and_ignores_extra_new_fields():
    assert s0.compare_reference(relaxation(), relaxation())["problems"] == []
    assert s0.compare_reference(relaxation(), {**relaxation(), "cap_record": {"x": 1}})["problems"] == []
    for key, value in [("steps", [{"accepted_step_size": 0.2, "trial_count": 2}]),
                       ("force_history", {"evaluation_count": 7, "force_maximums": [1.0, 0.5000000000000001],
                                          "maximum_seen_eV_per_A": 1.0}),
                       ("history", {"evaluation_count": 8, "force_maximums": [1.0, 0.5], "maximum_seen_eV_per_A": 1.0}),
                       ("step_count", 4), ("final_positions", [[0.0, 1.0, 2.0000000000000004]]),
                       ("failure_reason", None), ("refusal_details", [{"x": 1}])]:
        assert s0.compare_reference(relaxation(), relaxation(**{key: value}))["problems"], key
    missing = relaxation()
    del missing["steps"]
    assert s0.compare_reference(relaxation(), missing)["problems"]


# ---- Loader refusals on fabricated trees ----------------------------------

def fabricate_tree(tmp_path, original_per_model=10):
    (tmp_path / "reports" / "execution").mkdir(parents=True)
    base = tmp_path / s0.RECOVERY_RUN
    keys = [f"aspirin/{g}/{b}" for g in ("13", "18", "18_2", "66", "78") for b in ("minus", "plus")]
    for model_id in s0.MODEL_ORDER:
        raw = base / f"pass-{model_id}" / "raw-branches"
        raw.mkdir(parents=True)
        for index, key in enumerate(keys[:original_per_model]):
            (raw / f"{index:03d}.json").write_text(json.dumps(row(branch_key=key)))
    return tmp_path, keys


def test_the_loader_accepts_a_complete_fabricated_set_and_refuses_damage(tmp_path):
    root, keys = fabricate_tree(tmp_path)
    assert len(s0.load_saved_original(root)) == 50
    (root / s0.RECOVERY_RUN / "pass-direct-0" / "raw-branches" / "009.json").unlink()
    with pytest.raises(s0.SavedResultError, match="direct-0: 9 saved rows, expected 10"):
        s0.load_saved_original(root)


def test_the_loader_refuses_duplicate_starts_and_unequal_start_sets(tmp_path):
    root, keys = fabricate_tree(tmp_path)
    path = root / s0.RECOVERY_RUN / "pass-energy-0" / "raw-branches" / "001.json"
    path.write_text(json.dumps(row(branch_key=keys[0])))  # now two rows for one start
    with pytest.raises(s0.SavedResultError, match="duplicate"):
        s0.load_saved_original(root)
    root2, keys2 = fabricate_tree(tmp_path / "second")
    path = root2 / s0.RECOVERY_RUN / "pass-energy-0" / "raw-branches" / "001.json"
    path.write_text(json.dumps(row(branch_key="aspirin/99/plus")))
    with pytest.raises(s0.SavedResultError, match="different starts"):
        s0.load_saved_original(root2)


def test_the_reference_loader_refuses_a_missing_or_extra_relaxation(tmp_path):
    (tmp_path / "reports" / "execution").mkdir(parents=True)
    base = tmp_path / s0.PAIRED_TRACE_RUN
    for index, (model_id, ref) in enumerate((m, r) for m in s0.MODEL_ORDER for r in s0.OWN_IDS):
        directory = base / f"{index:02d}-{model_id}"
        directory.mkdir(parents=True)
        (directory / "result.json").write_text(json.dumps(
            {"task": {"kind": "reference_control", "model_id": model_id, "reference_id": ref}}))
        (directory / "relaxation.json").write_text(json.dumps(relaxation()))
    assert len(s0.load_saved_reference(tmp_path)) == 30
    (base / "00-energy-0" / "result.json").write_text(json.dumps(
        {"task": {"kind": "replay", "model_id": "energy-0", "branch_key": "aspirin/18/plus"}}))
    with pytest.raises(s0.SavedResultError, match="differ from the 30 expected"):
        s0.load_saved_reference(tmp_path)
    # a duplicate (same model and reference in a second directory) is its own refusal
    (base / "00-energy-0" / "result.json").write_text(json.dumps(
        {"task": {"kind": "reference_control", "model_id": "energy-0", "reference_id": s0.OWN_IDS[0]}}))
    twin = base / "99-energy-0"
    twin.mkdir()
    (twin / "result.json").write_text(json.dumps(
        {"task": {"kind": "reference_control", "model_id": "energy-0", "reference_id": s0.OWN_IDS[0]}}))
    (twin / "relaxation.json").write_text(json.dumps(relaxation()))
    with pytest.raises(s0.SavedResultError, match="duplicate saved reference relaxation"):
        s0.load_saved_reference(tmp_path)


# ---- The real saved results ----------------------------------------------

@needs_real_files
def test_the_real_saved_results_load_as_50_and_30(saved):
    original, reference = saved
    assert len(original) == 50 and len(reference) == 30
    assert {m for (m, _) in original} == set(s0.MODEL_ORDER)
    assert {r for (_, r) in reference} == set(s0.OWN_IDS)
    keys = sorted({k for (_, k) in original})
    assert len(s0.schedule(keys)) == 80


@needs_real_files
def test_self_compare_of_the_real_results_passes(saved):
    new_original, new_reference = self_sets(saved)
    assert s0.compare_all(saved[0], saved[1], new_original, new_reference) == {}


@needs_real_files
@pytest.mark.parametrize("what", ["category", "stop_reason", "step_count", "force_evaluation_count",
                                  "converged", "budget_exhausted", "forces_finite", "final_positions",
                                  "one_ulp"])
def test_any_perturbed_original_field_of_the_real_results_fails(saved, what):
    new_original, new_reference = self_sets(saved)
    key = sorted(new_original)[7]
    target = new_original[key]
    if what == "final_positions":
        target["_audit_record"]["final_positions"][3][1] += 1e-9
    elif what == "one_ulp":
        value = target["_audit_record"]["final_positions"][0][0]
        target["_audit_record"]["final_positions"][0][0] = float(np.nextafter(value, np.inf))
    elif isinstance(target[what], bool):
        target[what] = not target[what]
    elif isinstance(target[what], int):
        target[what] += 1
    else:
        target[what] = "changed" if target[what] != "changed" else "other"
    mismatches = s0.compare_all(saved[0], saved[1], new_original, new_reference)
    assert list(mismatches) == [s0.original_task_id(*key)]


@needs_real_files
@pytest.mark.parametrize("what", ["steps", "force_maximums", "final_positions", "step_count",
                                  "force_evaluation_count", "failure_reason", "converged"])
def test_any_perturbed_reference_field_of_the_real_results_fails(saved, what):
    new_original, new_reference = self_sets(saved)
    key = sorted(new_reference)[11]
    target = new_reference[key]
    if what == "steps":
        target["steps"][0]["trial_count"] += 1
    elif what == "force_maximums":
        target["force_history"]["force_maximums"][5] += 1e-12
    elif what == "final_positions":
        target["final_positions"][0][0] = float(np.nextafter(target["final_positions"][0][0], np.inf))
    elif isinstance(target[what], bool):
        target[what] = not target[what]
    elif isinstance(target[what], int):
        target[what] += 1
    else:
        target[what] = "changed"
    mismatches = s0.compare_all(saved[0], saved[1], new_original, new_reference)
    assert list(mismatches) == [s0.reference_task_id(*key)]


@needs_real_files
def test_incomplete_new_results_are_refused(saved):
    new_original, new_reference = self_sets(saved)
    del new_original[sorted(new_original)[0]]
    with pytest.raises(s0.SavedResultError, match="same 80 tasks"):
        s0.compare_all(saved[0], saved[1], new_original, new_reference)


@needs_real_files
def test_the_saved_file_hashes_cover_all_80_results():
    hashes = s0.saved_file_hashes()
    assert len(hashes) == 50 + 30 * 2
    assert all(len(h) == 64 for h in hashes.values())
