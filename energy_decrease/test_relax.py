import numpy as np
import pytest

from stage2_capped.driver import FROZEN_PROCEDURE
from stage2_capped.relax_capped import relax_capped
from energy_decrease.relax import relax_energy


START = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def force_to_zero_after_first_step(positions):
    result = np.zeros_like(positions)
    if positions[0, 0, 0] == 0:
        result[0, 0, 0] = 1.0
    return result


def test_uphill_low_force_trial_spends_last_request_without_converging():
    proc = dict(FROZEN_PROCEDURE, max_force_evaluations=2)
    result = relax_energy(START, force_to_zero_after_first_step, proc,
                          energy_fn=lambda positions: positions[0, 0])
    assert result["budget_exhausted"] and not result["converged"]
    assert result["force_evaluation_count"] == 2
    assert result["step_count"] == 0
    np.testing.assert_array_equal(result["final_positions"], START)
    assert result["energy_record"]["requests"] == 2
    assert result["energy_record"]["trials"][0]["decision"] == "uphill"


def test_downhill_low_force_trial_converges_on_last_request():
    proc = dict(FROZEN_PROCEDURE, max_force_evaluations=2)
    result = relax_energy(START, force_to_zero_after_first_step, proc,
                          energy_fn=lambda positions: -positions[0, 0])
    assert result["converged"] and not result["budget_exhausted"]
    assert result["step_count"] == 1
    assert result["energy_record"]["trials"][0]["decision"] == "accepted"


def test_equal_energy_rejects_and_cached_incumbent_is_reused():
    proc = dict(FROZEN_PROCEDURE, max_force_evaluations=5)
    seen = []

    def energy(positions):
        x = positions[0, 0]
        seen.append(x)
        return 0.0 if x >= 0.15 else -x

    result = relax_energy(START, force_to_zero_after_first_step, proc, energy_fn=energy)
    assert result["converged"]
    assert seen == [0.0, 0.2, 0.1]
    assert [t["decision"] for t in result["energy_record"]["trials"]] == ["equal", "accepted"]


def test_accepted_energy_replaces_incumbent():
    def force(positions):
        result = np.zeros_like(positions)
        result[0, 0, 0] = float(positions[0, 0, 0] < 0.4)
        return result

    proc = dict(FROZEN_PROCEDURE, max_force_evaluations=3)
    result = relax_energy(START, force, proc, energy_fn=lambda p: -p[0, 0])
    first, second = result["energy_record"]["trials"]
    assert result["step_count"] == 2
    assert first["trial_eV"] == second["incumbent_eV"]
    assert result["energy_record"]["requests"] == 3


def test_initial_force_stop_does_not_call_energy():
    def unexpected_energy(positions):
        pytest.fail("energy requested at an initially converged geometry")

    result = relax_energy(START, np.zeros_like, FROZEN_PROCEDURE, energy_fn=unexpected_energy)
    assert result["converged"]
    assert result["step_count"] == result["energy_record"]["requests"] == 0


@pytest.mark.parametrize("value,reason", [(float("nan"), "nonfinite_energy"),
                                           ([1.0], "invalid_energy"),
                                           ("1.0", "invalid_energy")])
def test_invalid_trial_energy_keeps_last_accepted_geometry(value, reason):
    result = relax_energy(START, force_to_zero_after_first_step, FROZEN_PROCEDURE,
                          energy_fn=lambda p: value if p[0, 0] else 0.0)
    assert not result["converged"]
    assert result["failure_reason"] == reason
    assert result["energy_record"]["requests"] == 2
    np.testing.assert_array_equal(result["final_positions"], START)


def test_disabled_mode_has_identical_result_and_trace():
    left, right = [], []
    actual = relax_energy(START, force_to_zero_after_first_step, FROZEN_PROCEDURE, recorder=left.append)
    saved = relax_capped(START, force_to_zero_after_first_step, FROZEN_PROCEDURE, recorder=right.append)
    np.testing.assert_array_equal(actual.pop("final_positions"), saved.pop("final_positions"))
    assert actual == saved
    assert len(left) == len(right)
    for x, y in zip(left, right):
        for key in x:
            np.testing.assert_equal(x[key], y[key])
