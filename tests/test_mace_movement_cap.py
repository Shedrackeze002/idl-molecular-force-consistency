import json
import os
from pathlib import Path

import numpy as np
import pytest

from mace_movement_cap import CAPS, load_solver, prepare_cap, trace_writer, verify_trace, verify_unbound


@pytest.mark.parametrize("cap", CAPS)
def test_rejected_query_does_not_move_displacement_anchor(cap):
    start = np.array([[0., 0., 0.], [0., 0., 1.]])
    reject = start + [cap, 0., 0.]
    final = start - [cap, 0., 0.]
    binding = {"scale_s": 2.0, "scale_effective": cap, "force_norm_max_f": 1.0}
    rows = [
        {"attempt_index": 1, "kind": "initial", "positions": start},
        {**binding, "attempt_index": 2, "kind": "trial", "positions": reject, "accepted": False,
         "max_displacement_A": cap, "capped": True},
        {**binding, "attempt_index": 3, "kind": "trial", "positions": final, "accepted": True,
         "max_displacement_A": cap, "capped": True},
    ]
    summary = verify_trace(rows, 3, cap, start, final)
    assert summary["capped_trial_count"] == 2
    assert summary["accepted_travel_A"] == pytest.approx(cap)
    assert summary["attempted_travel_A"] == pytest.approx(2 * cap)
    rows[2]["positions"] = start - [1.1 * cap, 0., 0.]
    with pytest.raises(ValueError, match="displacement"):
        verify_trace(rows, 3, cap, start, final)
    with pytest.raises(ValueError, match="initial coordinates"):
        verify_trace(rows, 3, cap, final, final)


def test_changed_solver_fails_before_executing(tmp_path):
    source = tmp_path.resolve() / "relax_capped.py"
    source.write_text("raise AssertionError('untrusted code executed')\n")
    with pytest.raises(ValueError, match="source digest"):
        load_solver({"inputs": {"capped_relaxer": {"path": str(source)}}})


def test_unbound_outcome_requires_exact_saved_endpoint():
    row = {"category": "matched_reference", "force_evaluation_count": 1}
    position = [[0., 0., 0.]]
    verify_unbound(row, row, tuple(row), position, position)
    with pytest.raises(ValueError, match="historical outcome or endpoint"):
        verify_unbound(row, row, tuple(row), position, [[1e-12, 0., 0.]])


def test_interrupted_trace_keeps_completed_force_rows(tmp_path):
    path = tmp_path / "trace.jsonl"
    row = {"attempt_index": 1, "positions": np.zeros((2, 3))}
    with pytest.raises(RuntimeError, match="interrupted"):
        with trace_writer(path) as (_, append):
            append(row)
            assert json.loads(path.read_text())["attempt_index"] == 1
            raise RuntimeError("interrupted")
    assert len(path.read_text().splitlines()) == 1
    with pytest.raises(FileExistsError):
        with trace_writer(path):
            pass


@pytest.mark.parametrize("cap", CAPS)
def test_original_branch_uses_reviewed_capped_pass(cap, tmp_path):
    candidate = os.environ.get("MACE_CAP_FIXTURE_RECORD")
    if not candidate:
        pytest.skip("requires retained Stage 2 source and MACE control artifacts")
    from evaluation.runner import _bind_split_manifest, _own_reference
    from evaluation.validation_controls import validation_inputs

    record = json.loads(Path(candidate).read_bytes())
    record["movement_cap_A"] = cap
    _, _, second_pass = prepare_cap(record)
    inputs = {name: Path(item["path"]) for name, item in record["inputs"].items()}
    manifest = json.loads(inputs["manifest"].read_bytes())
    classification = json.loads(inputs["classification"].read_bytes())
    procedure = json.loads(inputs["config"].read_bytes())["relaxation"]
    starts, refs, policy, _ = validation_inputs(inputs["archive"], manifest, classification)
    key = sorted(starts)[0]
    start = starts[key]
    pool, own = _own_reference(key, classification, refs)
    bound = _bind_split_manifest(manifest, synthetic_declaration=None)
    calls = []

    def force(positions, **_):
        calls.append(1)
        values = np.zeros_like(positions)
        values[0, 0, 0] = {1: 100., 2: -100.}.get(len(calls), 0.)
        return values, {"status": "ok"}

    arm = {"arm_id": "synthetic-force", "status": "ready", "species": start["species"], "forces": force}
    path = tmp_path / "trace.jsonl"
    with trace_writer(path) as (trace, append):
        row = second_pass([(key, start, "validation", pool, own)], arm, procedure, refs, policy,
                          {key: True}, bound, movement_cap_A=cap, recorder_for=lambda _: append)[key]
    saved = [json.loads(line) for line in path.read_text().splitlines()]
    diagnostics = verify_trace(saved, row["force_evaluation_count"], cap, start["positions"],
                               row["_audit_record"]["final_positions"])
    assert len(calls) == len(trace) == row["force_evaluation_count"] == 3
    assert row["cap_record"]["rejected_trial_count"] == 1
    assert diagnostics["capped_trial_count"] >= 1
