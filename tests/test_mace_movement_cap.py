import numpy as np
import pytest

from mace_movement_cap import CAPS, load_solver, verify_trace, verify_unbound


@pytest.mark.parametrize("cap", CAPS)
def test_rejected_query_does_not_move_displacement_anchor(cap):
    start = np.array([[0., 0., 0.], [0., 0., 1.]])
    reject = start + [cap, 0., 0.]
    final = start - [cap, 0., 0.]
    rows = [
        {"attempt_index": 1, "kind": "initial", "positions": start},
        {"attempt_index": 2, "kind": "trial", "positions": reject, "accepted": False,
         "max_displacement_A": cap, "capped": True},
        {"attempt_index": 3, "kind": "trial", "positions": final, "accepted": True,
         "max_displacement_A": cap, "capped": True},
    ]
    summary = verify_trace(rows, 3, cap, final)
    assert summary["capped_trial_count"] == 2
    assert summary["accepted_travel_A"] == pytest.approx(cap)
    assert summary["attempted_travel_A"] == pytest.approx(2 * cap)
    rows[2]["positions"] = start - [1.1 * cap, 0., 0.]
    with pytest.raises(ValueError, match="displacement"):
        verify_trace(rows, 3, cap, final)


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
