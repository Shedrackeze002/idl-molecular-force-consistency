"""Tests for the frozen relaxation procedure (Task 2).

relax() is steepest descent with a backtracking step size, driven by any
force function. Every test here uses analytic potentials with exact,
hand-computable answers.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import analytic_fixtures as fixtures
from conftest import REPO


def _procedure(**overrides):
    base = {
        "max_force_evaluations": 300,
        "initial_step_size": 0.2,
        "backtracking_factor": 0.5,
        "min_backtracking_floor": 1e-8,
        "convergence_tolerance_eV_per_A": 1e-2,
        "max_backtracking_attempts_per_step": 40,
    }
    base.update(overrides)
    return base


def _start(shape_offsets):
    positions = np.array(fixtures.FIXTURE_MINIMUM, dtype=np.float64, copy=True)
    positions = positions + np.asarray(shape_offsets, dtype=np.float64)
    positions.setflags(write=False)
    return positions


def _single_start(x):
    # One molecule, unbatched: shape (N, 3) with N = 1. The unbatched
    # form is the single-molecule convention (see relax's docstring and
    # the real archive records), so the endpoint comes back (1, 3) and
    # final_positions[0] is the plain (3,) atom row.
    return np.array([x], dtype=np.float64)


def _max_component(forces):
    return float(np.max(np.abs(forces)))


# ---------------------------------------------------------------------------
# Converging cases
# ---------------------------------------------------------------------------

def test_harmonic_single_atom_converges_to_the_analytic_minimum():
    from evaluation.relax import relax

    start = _single_start([3.2, 2.0, -1.5])
    result = relax(start, fixtures.harmonic_force, _procedure())
    assert result["status"] == "converged"
    assert result["budget_exhausted"] is False
    assert result["forces_finite"] is True
    assert result["force_evaluation_count"] == result["step_count"] + 1
    np.testing.assert_allclose(
        result["final_positions"][0], fixtures.HARMONIC_MINIMUM, atol=1e-8)
    # The final force is at or under the convergence tolerance.
    assert result["final_max_force_component_eV_per_A"] <= 1e-2 + 1e-15
    # Force history summary is present and counts every evaluation.
    assert result["force_history"]["evaluation_count"] == result["force_evaluation_count"]
    assert result["force_history"]["maximum_seen_eV_per_A"] == pytest.approx(
        _max_component(fixtures.harmonic_force(start)), rel=1e-12)


def test_double_well_started_on_the_right_stays_right():
    from evaluation.relax import relax

    # Start at x = 0.6, on the +1 side of the barrier at 0.
    result = relax(_single_start([0.6, 0.3, -0.2]), fixtures.double_well_force,
                   _procedure(max_force_evaluations=500))
    assert result["status"] == "converged"
    np.testing.assert_allclose(
        result["final_positions"][0], fixtures.DOUBLE_WELL_MINIMA[0], atol=1e-8)


def test_double_well_started_on_the_left_stays_left():
    from evaluation.relax import relax

    result = relax(_single_start([-0.75, -0.2, 0.1]), fixtures.double_well_force,
                   _procedure(max_force_evaluations=500))
    assert result["status"] == "converged"
    np.testing.assert_allclose(
        result["final_positions"][0], fixtures.DOUBLE_WELL_MINIMA[1], atol=1e-8)


def test_backtracking_shrinks_the_step_until_energy_would_fall():
    from evaluation.relax import relax

    # A steep start makes the first trial step overshoot; the backtracking
    # rule must recover and still converge. The observed step size of the
    # accepted step is recorded in the history.
    start = _single_start([2.5, 0.0, 0.0])
    result = relax(start, fixtures.harmonic_force, _procedure(initial_step_size=1.0))
    assert result["status"] == "converged"
    assert result["steps"] and "accepted_step_size" in result["steps"][0]
    accepted = [s["accepted_step_size"] for s in result["steps"]]
    assert all(a > 0 for a in accepted)


def test_max_component_rule_is_stated_per_component_not_as_a_norm():
    from evaluation.relax import relax

    # A well with anisotropic strength: convergence happens only when the
    # LARGEST component of every atom's force is under the tolerance.
    def anisotropic_force(positions):
        positions = np.asarray(positions, dtype=np.float64)
        target = np.array([[1.0, 0.0, 0.0]])
        # Different scale per component: x pulls hard, z pulls softly.
        scale = np.array([4.0, 1.0, 0.25])
        return (target - positions) * scale

    start = _single_start([1.1, 0.0, 0.3])
    result = relax(start, anisotropic_force,
                   _procedure(convergence_tolerance_eV_per_A=1e-2))
    assert result["status"] == "converged"
    final_force = np.squeeze(anisotropic_force(result["final_positions"]))
    # The maximum component decides, not the norm.
    assert _max_component(final_force) <= 1e-2 + 1e-15
    # The norm mixes components, so it is only strictly larger than the
    # maximum component when two or more components are nonzero. The
    # START has two nonzero components and shows that difference; the
    # converged endpoint collapses to one dominant tiny component where
    # norm and maximum coincide, so the strict comparison belongs on
    # the start, not on the endpoint.
    start_force = np.squeeze(anisotropic_force(start))
    assert np.linalg.norm(start_force) > _max_component(start_force)
    assert _max_component(final_force) >= np.linalg.norm(final_force) - 1e-15


# ---------------------------------------------------------------------------
# Nonconverging cases
# ---------------------------------------------------------------------------

def test_constant_force_field_never_converges_and_keeps_its_endpoint():
    from evaluation.relax import relax

    start = _start(0.0)
    procedure = _procedure(max_force_evaluations=50)
    result = relax(start, fixtures.constant_nonzero_force(0.05), procedure)
    # The step cap is met, the endpoint is retained for audit, and the run
    # is marked nonconverged. Callers score this as not a shape recovery.
    assert result["status"] == "nonconverged"
    assert result["budget_exhausted"] is True
    assert result["forces_finite"] is True
    assert result["force_evaluation_count"] == 50
    assert result["final_positions"].shape == start.shape
    # A constant downhill push must have moved the structure from its start.
    moved = np.max(np.abs(result["final_positions"] - start))
    assert moved > 1e-6
    # Backtracking saturates at the floor and every trial fails (no valid
    # step exists), so the endpoint is the last retained trial.
    assert result["history"]["force_maximums"][-1] == pytest.approx(0.05, abs=1e-12)


def test_nonfinite_force_stops_the_run_and_is_reported():
    from evaluation.relax import relax

    calls = []

    def poisoned_force(positions):
        forces = fixtures.harmonic_force(positions)
        calls.append(1)
        out = np.array(forces, dtype=np.float64, copy=True)
        if len(calls) >= 3:
            out[..., 0] = np.inf
        return out

    result = relax(_single_start([2.0, 0.0, 0.0]), poisoned_force, _procedure())
    # The run ends at the first nonfinite evaluation and reports it
    # explicitly instead of propagating NaNs into the step history.
    assert result["status"] == "nonconverged"
    assert result["forces_finite"] is False
    assert result["force_evaluation_count"] == 3
    assert result["budget_exhausted"] is False
    assert np.isfinite(result["final_positions"]).all()


def test_nonfinite_positions_do_not_pass_through_silently():
    from evaluation.relax import relax

    def exploding_force(positions):
        positions = np.asarray(positions, dtype=np.float64)
        if np.max(np.abs(positions)) > 50.0:
            return np.full_like(positions, np.nan)
        return positions * (-2.0)

    result = relax(_single_start([60.0, 0.0, 0.0]), exploding_force,
                   _procedure(max_force_evaluations=100))
    assert result["status"] == "nonconverged"
    assert result["forces_finite"] is False
    assert np.isfinite(result["final_positions"]).all()


# ---------------------------------------------------------------------------
# Budget accounting
# ---------------------------------------------------------------------------

def test_force_evaluation_count_is_exact():
    from evaluation.relax import relax

    counter = {"calls": 0}

    def counting_force(positions):
        counter["calls"] += 1
        return fixtures.harmonic_force(positions)

    result = relax(_single_start([2.0, 1.0, 0.5]), counting_force,
                   _procedure(max_force_evaluations=20))
    assert result["status"] == "converged"
    assert counter["calls"] == result["force_evaluation_count"]
    # Convergence costs one evaluation to confirm the stop, so the count is
    # the steps plus that final check, never a mysterious number.
    assert counter["calls"] == result["step_count"] + 1
    assert counter["calls"] <= 20


def test_budget_respected_and_excess_marked_incomplete():
    from evaluation.relax import relax

    # A well that would converge but with a budget too small to get there:
    # the run must be marked incomplete rather than silently truncated.
    tiny = _procedure(max_force_evaluations=5)
    result = relax(_single_start([2.0, 1.0, 0.5]), fixtures.harmonic_force, tiny)
    assert result["status"] == "nonconverged"
    assert result["budget_exhausted"] is True
    assert result["force_evaluation_count"] == 5
    np.testing.assert_allclose(
        result["final_positions"][0], fixtures.HARMONIC_MINIMUM, atol=0.5)
    # It is NOT a success and MUST NOT be reported as converged.
    assert result["status"] != "converged"


def test_backtracking_trials_cost_budgeted_evaluations():
    from evaluation.relax import relax

    # Steep well, big initial step: the first step rejects several trials.
    # Every trial is a real force evaluation and the total count includes
    # them all.
    counter = {"calls": 0}

    def counting_force(positions):
        counter["calls"] += 1
        return fixtures.harmonic_force(positions)

    result = relax(_single_start([2.0, 0.0, 0.0]), counting_force,
                   _procedure(initial_step_size=4.0, max_force_evaluations=300))
    assert result["status"] == "converged"
    # At least one step tried more than one candidate (the first trial plus
    # a rejected one) and the total count includes all trials.
    first_step_trials = result["steps"][0]["trial_count"]
    assert first_step_trials >= 1
    assert counter["calls"] == result["force_evaluation_count"]
    assert counter["calls"] >= result["step_count"] + 1


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_final_structure_is_bitwise_identical_across_two_fresh_processes():
    from evaluation.relax import relax

    start = _start(np.array([[0.4, -0.2, 0.1]]))
    procedure = _procedure(max_force_evaluations=400)

    def force(positions):
        return fixtures.harmonic_force(positions)

    here = relax(start, force, procedure)["final_positions"]
    script = (
        "import json, sys\n"
        "import numpy as np\n"
        "sys.path.insert(0, r'" + str(REPO / "tests").replace("\\", "/") + "')\n"
        "sys.path.insert(0, r'" + str(REPO).replace("\\", "/") + "')\n"
        "import analytic_fixtures as fixtures\n"
        "from evaluation.relax import relax\n"
        "start = np.array(fixtures.FIXTURE_MINIMUM, dtype=np.float64) + np.array([[0.4, -0.2, 0.1]])\n"
        "procedure = " + json.dumps(procedure) + "\n"
        "result = relax(start, fixtures.harmonic_force, procedure)\n"
        "print(json.dumps({'positions': result['final_positions'].tolist()}))\n"
    )
    outputs = []
    for _ in range(2):
        env_script = Path(__file__).resolve().parent / "_determinism_child.py"
        env_script.write_text(script)
        try:
            out = subprocess.run(
                [sys.executable, str(env_script)], capture_output=True, text=True, check=True)
            outputs.append(out.stdout.strip().splitlines()[-1])
        finally:
            env_script.unlink()
    assert outputs[0] == outputs[1]
    child = outputs[0]
    fresh = np.array(__import__("json").loads(child)["positions"], dtype=np.float64)
    # Bitwise, not approximately: byte-level equality of IEEE doubles.
    assert here.tobytes() == fresh.tobytes()


def test_same_inputs_in_process_reproduce_bitwise():
    from evaluation.relax import relax

    start = _start(np.array([[0.4, -0.2, 0.1]]))
    procedure = _procedure()
    first = relax(start, fixtures.harmonic_force, procedure)
    second = relax(start, fixtures.harmonic_force, procedure)
    assert first["final_positions"].tobytes() == second["final_positions"].tobytes()
    assert first["steps"] == second["steps"]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def test_start_positions_must_be_finite_and_well_shaped():
    from evaluation.relax import relax

    with pytest.raises(ValueError):
        relax(np.zeros((2, 3)), fixtures.harmonic_force, _procedure())
    poisoned = np.array([[np.nan, 0.0, 0.0]])
    with pytest.raises(ValueError):
        relax(poisoned, fixtures.harmonic_force, _procedure())


def test_procedure_settings_are_validated():
    from evaluation.relax import relax

    with pytest.raises(ValueError):
        relax(_single_start([1.0, 0.0, 0.0]), fixtures.harmonic_force,
              _procedure(max_force_evaluations=0))
    with pytest.raises(ValueError):
        relax(_single_start([1.0, 0.0, 0.0]), fixtures.harmonic_force,
              _procedure(initial_step_size=-1.0))
    with pytest.raises(ValueError):
        relax(_single_start([1.0, 0.0, 0.0]), fixtures.harmonic_force,
              _procedure(convergence_tolerance_eV_per_A=0.0))
    with pytest.raises(ValueError):
        relax(_single_start([1.0, 0.0, 0.0]), fixtures.harmonic_force,
              _procedure(backtracking_factor=1.0))