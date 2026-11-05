"""M4: the copied second pass. A diff gate ties it to runner.py lines 811 to 861, and at
no cap it reproduces the frozen evaluate_starts rows field for field (no model)."""
import copy
import json

import numpy as np
import pytest

from evaluation.adapter import callable_arm
from evaluation.runner import (_bind_split_manifest, _connection_partition,
                               _manifest_synthetic_declaration, _own_reference, evaluate_starts)
from evaluation.start_set import build_start_set
from stage2_capped import diff_gate, pass_copy
from stage2_capped.stage0_compare import first_difference
from stage2_capped.tests import fixtures as fx

ALLOWLIST = diff_gate.ROOT / "stage2_capped/allowlist_pass.json"
FROZEN, DERIVED = "evaluation/runner.py", "stage2_capped/pass_copy.py"


# ---- Diff gate -----------------------------------------------------------

def test_diff_gate_passes_on_the_committed_files():
    diff_gate.assert_gate("pass_copy", allowlist_path=ALLOWLIST)


def test_the_allowlist_pins_the_runner_and_the_excerpt():
    entry = json.loads(ALLOWLIST.read_text())["targets"]["pass_copy"]
    frozen = (diff_gate.ROOT / FROZEN).read_bytes()
    assert entry["frozen_sha256"] == diff_gate.sha256_bytes(frozen)
    assert entry["frozen_line_range"] == [811, 861]
    excerpt = diff_gate.frozen_excerpt(frozen, entry["frozen_line_range"])
    assert entry["frozen_excerpt_sha256"] == diff_gate.sha256_bytes(excerpt)
    assert excerpt.decode().startswith("    for branch_key, start_entry, partition, _pool_key, own in plans:")
    assert excerpt.decode().rstrip().endswith("}")
    assert len(entry["hunks"]) == 4 and set(entry["allowed_categories"]) == {"module_header", "cap_pass"}


def _root_with(tmp_path, frozen_bytes, derived_bytes):
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "stage2_capped").mkdir()
    (tmp_path / FROZEN).write_bytes(frozen_bytes)
    (tmp_path / DERIVED).write_bytes(derived_bytes)
    return tmp_path


@pytest.mark.parametrize("old, new", [
    ('outcome = classify_outcome(', 'outcome = classify_outcome(  # tampered\n            None, '),
    ('"force_evaluation_count": relaxation["force_evaluation_count"],',
     '"force_evaluation_count": relaxation["step_count"],'),
    ('"status": relaxation["status"],', '"status": "converged",'),
    ('seen_minimum=flag,', 'seen_minimum=True,'),
    ('stop_reason=relaxation["failure_reason"],', 'stop_reason=None,'),
    ('movement_cap_A=movement_cap_A,', 'movement_cap_A=None,'),
])
def test_diff_gate_catches_each_kind_of_edit(tmp_path, old, new):
    text = (diff_gate.ROOT / DERIVED).read_text()
    assert text.count(old) == 1
    root = _root_with(tmp_path, (diff_gate.ROOT / FROZEN).read_bytes(), text.replace(old, new).encode())
    assert diff_gate.check_gate("pass_copy", allowlist_path=ALLOWLIST, root=root)


def test_diff_gate_catches_a_change_inside_the_frozen_excerpt_and_outside_it(tmp_path):
    frozen = (diff_gate.ROOT / FROZEN).read_text()
    derived = (diff_gate.ROOT / DERIVED).read_bytes()
    lines = frozen.split("\n")
    inside = list(lines)
    inside[820] = inside[820] + "  # edited"        # a line within 811..861
    outside = list(lines)
    outside[0] = outside[0] + " edited"             # a line outside it
    (tmp_path / "in").mkdir()
    (tmp_path / "out").mkdir()
    r_in = _root_with(tmp_path / "in", "\n".join(inside).encode(), derived)
    r_out = _root_with(tmp_path / "out", "\n".join(outside).encode(), derived)
    problems_in = diff_gate.check_gate("pass_copy", allowlist_path=ALLOWLIST, root=r_in)
    problems_out = diff_gate.check_gate("pass_copy", allowlist_path=ALLOWLIST, root=r_out)
    assert any("pinned SHA-256" in p for p in problems_in) and any("excerpt" in p for p in problems_in)
    assert any("pinned SHA-256" in p for p in problems_out)  # the whole file is pinned too


def test_the_line_range_must_lie_inside_the_frozen_file(tmp_path):
    allowlist = json.loads(ALLOWLIST.read_text())
    allowlist["targets"]["pass_copy"]["frozen_line_range"] = [811, 99999]
    path = tmp_path / "range.json"
    path.write_text(json.dumps(allowlist))
    problems = diff_gate.check_gate("pass_copy", allowlist_path=path)
    assert any("outside the frozen file" in p for p in problems)


def test_a_target_without_a_line_range_behaves_as_before():
    diff_gate.assert_gate("relax_capped")
    diff_gate.assert_gate("restore_fork", allowlist_path=diff_gate.ROOT / "stage2_capped/allowlist_restore.json")


# ---- Equivalence with the frozen pass ------------------------------------

def inputs(land, manifest, classification):
    starts = build_start_set(manifest, land["records"], land["policy"], {"kind": "first_stored_frame"},
                             partitions=("validation",))
    refs = {r.record_id: r for r in land["records"] if r.kind == "minimum"}
    species = next(iter(starts.values()))["species"]
    return starts, refs, species


def plans_for(starts, classification, refs, manifest):
    plans = []
    for key, entry in starts.items():
        partition = _connection_partition(key, manifest)
        pool_key, own = _own_reference(key, classification, refs)
        plans.append((key, entry, partition, pool_key, own))
    return plans


def arm_for(species, stiffness, name):
    target = np.array([[0.0, 0.0, 0.0], [1.1, 0.0, 0.0], [0.0, 1.3, 0.0]])
    return callable_arm(fx.harmonic(target, stiffness), species, name)


def stable(row):
    """The row without its cap record and wall-clock time. Shallow copies only: the
    audit record shares the reference object with the other row."""
    row = dict(row)
    row.pop("cap_record", None)
    audit = dict(row["_audit_record"])
    audit.pop("timing_seconds")
    row["_audit_record"] = audit
    return row


@pytest.mark.parametrize("stiffness", [0.5, 4.0, 40.0])
@pytest.mark.parametrize("cap", [None, float("inf"), 1e12])
def test_without_a_cap_the_copied_pass_reproduces_the_frozen_rows(
        synthetic_landscape, frozen_synthetic_manifest, synthetic_endpoint_classification, stiffness, cap):
    land, manifest, classification = synthetic_landscape, frozen_synthetic_manifest, synthetic_endpoint_classification
    starts, refs, species = inputs(land, manifest, classification)
    arm = arm_for(species, stiffness, f"harmonic-{stiffness}")
    procedure = fx.procedure(max_force_evaluations=80)
    frozen = evaluate_starts(starts, arm, procedure, classification, refs, land["policy"],
                             split_manifest=manifest, seen_minimum=False)
    bound = _bind_split_manifest(manifest, synthetic_declaration=_manifest_synthetic_declaration(manifest))
    rows = pass_copy.second_pass(plans_for(starts, classification, refs, manifest), arm, procedure, refs,
                                 land["policy"], False, bound, movement_cap_A=cap,
                                 recorder_for=lambda key: None)
    assert list(rows) == list(frozen["branches"])
    for key, frozen_row in frozen["branches"].items():
        assert first_difference(stable(frozen_row), stable(rows[key])) is None, key
        assert set(rows[key]) - set(frozen_row) == {"cap_record"}
        assert rows[key]["cap_record"]["capped_trial_count"] == 0


def test_a_binding_cap_changes_the_run_and_the_row_records_it(
        synthetic_landscape, frozen_synthetic_manifest, synthetic_endpoint_classification):
    land, manifest, classification = synthetic_landscape, frozen_synthetic_manifest, synthetic_endpoint_classification
    starts, refs, species = inputs(land, manifest, classification)
    arm = arm_for(species, 0.5, "soft")
    procedure = fx.procedure(max_force_evaluations=80)
    bound = _bind_split_manifest(manifest, synthetic_declaration=_manifest_synthetic_declaration(manifest))
    plans = plans_for(starts, classification, refs, manifest)
    control = pass_copy.second_pass(plans, arm, procedure, refs, land["policy"], False, bound,
                                    movement_cap_A=None, recorder_for=lambda key: None)
    capped = pass_copy.second_pass(plans, arm, procedure, refs, land["policy"], False, bound,
                                   movement_cap_A=0.01, recorder_for=lambda key: None)
    for key in control:
        record = capped[key]["cap_record"]
        assert record["movement_cap_A"] == 0.01 and record["capped_trial_count"] > 0
        assert record["max_accepted_displacement_A"] <= 0.01 * (1 + 1e-12)
        assert first_difference(control[key]["_audit_record"]["final_positions"],
                                capped[key]["_audit_record"]["final_positions"]) is not None


def test_the_recorder_is_asked_once_per_branch_and_sees_every_attempt(
        synthetic_landscape, frozen_synthetic_manifest, synthetic_endpoint_classification):
    land, manifest, classification = synthetic_landscape, frozen_synthetic_manifest, synthetic_endpoint_classification
    starts, refs, species = inputs(land, manifest, classification)
    arm = arm_for(species, 4.0, "stiff")
    bound = _bind_split_manifest(manifest, synthetic_declaration=_manifest_synthetic_declaration(manifest))
    asked, records = [], {}

    def recorder_for(key):
        asked.append(key)
        records[key] = []
        return records[key].append

    rows = pass_copy.second_pass(plans_for(starts, classification, refs, manifest), arm,
                                 fx.procedure(max_force_evaluations=80), refs, land["policy"], False, bound,
                                 movement_cap_A=0.05, recorder_for=recorder_for)
    assert asked == list(rows)
    for key, row in rows.items():
        assert len(records[key]) == row["force_evaluation_count"] == row["cap_record"]["trial_count"] + 1
        assert [r["attempt_index"] for r in records[key]] == list(range(1, len(records[key]) + 1))
