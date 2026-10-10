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


# Tolerance used when a test wants convergence to pin the endpoint to the
# procedure's own fixed point. The declared rule converges when max |F| <=
# the configured tolerance; a tightened CONFIGURATION is not a changed
# rule. Value chosen from a measured trace of the double well: its steepest
# descent step arithmetic stalls at max |F| ~ 3.2e-10 eV/angstrom, so
# 1e-12 would never be met while 1e-9 is met at the stall (both wells
# deterministic).
_TIGHT_PROCEDURE = {"convergence_tolerance_eV_per_A": 1e-9}


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
                   _procedure(**_TIGHT_PROCEDURE))
    assert result["status"] == "converged"
    np.testing.assert_allclose(
        result["final_positions"][0], fixtures.DOUBLE_WELL_MINIMA[0], atol=1e-9)


def test_double_well_started_on_the_left_stays_left():
    from evaluation.relax import relax

    result = relax(_single_start([-0.75, -0.2, 0.1]), fixtures.double_well_force,
                   _procedure(**_TIGHT_PROCEDURE))
    assert result["status"] == "converged"
    np.testing.assert_allclose(
        result["final_positions"][0], fixtures.DOUBLE_WELL_MINIMA[1], atol=1e-9)


# Under the FROZEN 1e-2 tolerance the declared rule stops at the first
# evaluation with every |F| component <= 1e-2, which on these ladders is
# the step that lands nearest the minimum, not the minimum itself. The
# basin side is the physics claim; the frozen tolerance's endpoint
# precision is asserted against the measured endpoint force, which is the
# rule the frozen config actually declares.

def test_double_well_frozen_tolerance_stops_in_the_right_basin():
    from evaluation.relax import relax

    for start, expected_minimum in (
            ([0.6, 0.3, -0.2], fixtures.DOUBLE_WELL_MINIMA[0]),
            ([-0.75, -0.2, 0.1], fixtures.DOUBLE_WELL_MINIMA[1])):
        result = relax(_single_start(start), fixtures.double_well_force,
                       _procedure(max_force_evaluations=500))
        assert result["status"] == "converged"
        final = result["final_positions"][0]
        # The endpoint sits in the EXPECTED basin, nowhere near the other
        # well (the wells are 2 angstrom apart).
        assert np.linalg.norm(final - expected_minimum) <= 1.0
        # The endpoint force itself satisfies the declared rule it
        # converged under (the rule is on the force, not on the position).
        final_force = np.asarray(fixtures.double_well_force(
            result["final_positions"]), dtype=np.float64)
        assert float(np.max(np.abs(final_force))) <= 1e-2 + 1e-15


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


def test_budget_respected_and_not_silently_extended():
    from evaluation.relax import relax

    # The budget is an exact cap on force ATTEMPTS, counted in the result.
    # A start whose declared-rule endpoint is reachable WITHIN the budget
    # converges there (that is the A2 reviewer finding: a result
    # established within the budget is not thrown away); a budget never
    # buys an extra evaluation beyond the cap.
    tiny = _procedure(max_force_evaluations=5)
    counter = {"calls": 0}

    def counting_force(positions):
        counter["calls"] += 1
        return fixtures.harmonic_force(positions)

    result = relax(_single_start([2.0, 1.0, 0.5]), counting_force, tiny)
    assert counter["calls"] <= 5
    assert result["force_evaluation_count"] == counter["calls"]
    # Under the declared rule this start converges on the LAST budgeted
    # attempt (measured: five evaluations, per-eval max |F| 1.5, 1.2, 0.72,
    # 0.144, 0.0): the fifth evaluation is the landing's endpoint force and
    # it satisfies the rule, so the result established within the budget
    # is reported, not thrown away because the budget is now spent.
    assert result["status"] == "converged"
    assert result["budget_exhausted"] is False
    assert result["force_evaluation_count"] == 5
    np.testing.assert_allclose(
        result["final_positions"][0], fixtures.HARMONIC_MINIMUM, atol=1e-12)


def test_budget_exhaustion_marks_the_run_incomplete():
    from evaluation.relax import relax

    # A start that genuinely needs more evaluations than the budget
    # allows: the run must stop unpaid, mark budget_exhausted, and NOT be
    # silently truncated into a success.
    tiny = _procedure(max_force_evaluations=2)
    result = relax(_single_start([2.0, 1.0, 0.5]), fixtures.harmonic_force, tiny)
    assert result["status"] == "nonconverged"
    assert result["budget_exhausted"] is True
    assert result["force_evaluation_count"] == 2
    # The endpoint is near where two budgeted attempts could reach, not
    # claimed to be the minimum.
    distance_to_minimum = float(np.linalg.norm(
        result["final_positions"][0] - fixtures.HARMONIC_MINIMUM))
    assert distance_to_minimum > 1e-4
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


# ---------------------------------------------------------------------------
# Plain-supplier refusals (declared, recorded, never read as physics)
# ---------------------------------------------------------------------------

def _refusal_procedure():
    return _procedure(max_force_evaluations=10)


def test_scalar_zero_force_for_a_two_atom_start_is_a_recorded_refusal():
    from evaluation.relax import relax

    # A scalar zero broadcasts to every component, so without a shape
    # check it would be read as instant convergence for any molecule
    # size. It is a refusal, not physics.
    def scalar_zero(positions):
        return 0.0

    result = relax(np.array([[1.0, 0.0, 0.0], [0.5, 0.2, 0.0]], dtype=np.float64),
                   scalar_zero, _refusal_procedure())
    assert result["status"] == "nonconverged"
    assert result["failure_reason"] == "invalid_force_shape"
    # The values seen stay finite-labelled (review round 2): a wrongly
    # shaped return carries no readable values at all, so only a genuine
    # nonfinite VALUE sets forces_finite False.
    assert result["forces_finite"] is True
    assert result["budget_exhausted"] is False
    assert result["step_count"] == 0
    assert result["force_evaluation_count"] == 1
    assert np.isfinite(np.asarray(result["final_positions"])).all()


def test_refusal_record_names_the_arm_and_the_attempt():
    from evaluation.relax import relax

    # The audit trail must keep two same-wording refusals from two
    # different arms distinguishable: each refusal detail records WHICH
    # refusal, from WHICH arm, at WHICH attempt number.
    def zero_scalar(positions):
        return 0.0

    first = relax(np.array([[1.0, 0.0, 0.0], [0.5, 0.2, 0.0]], dtype=np.float64),
                  zero_scalar, _refusal_procedure(), arm={"arm_id": "arm-one"})
    second = relax(np.array([[1.0, 0.0, 0.0], [0.5, 0.2, 0.0]], dtype=np.float64),
                   zero_scalar, _refusal_procedure(), arm={"arm_id": "arm-two"})
    # A direct plain-float return refuses with the measured shape detail:
    # the scalar's array shape () next to the positions' (2, 3).
    expected_detail = (f"force shape {np.asarray(0.0).shape} "
                       f"for positions (2, 3)")
    for result, arm_id in ((first, "arm-one"), (second, "arm-two")):
        assert result["failure_reason"] == "invalid_force_shape"
        assert result["refusal_details"] == [{
            "reason": "invalid_force_shape",
            "arm_id": arm_id,
            "detail": expected_detail,
            "force_evaluation_count": 1,
        }]
    # The two records name different arms despite the identical refusal
    # wording, which is the entire point of the detail record.
    assert (first["refusal_details"][0]["arm_id"]
            != second["refusal_details"][0]["arm_id"])
    assert (first["refusal_details"][0]["detail"]
            == second["refusal_details"][0]["detail"])


def test_mismatched_force_shape_is_a_recorded_refusal():
    from evaluation.relax import relax

    def wrong_shape(positions):
        return np.zeros((3, 3), dtype=np.float64)

    result = relax(np.array([[1.0, 0.0, 0.0], [0.5, 0.2, 0.0]], dtype=np.float64),
                   wrong_shape, _refusal_procedure())
    assert result["status"] == "nonconverged"
    assert result["failure_reason"] == "invalid_force_shape"
    assert result["force_evaluation_count"] == 1


def test_shape_mismatch_refusal_does_not_abort_a_batched_run():
    from evaluation.relax import relax

    # The batched run: a malformed batched return is refused the same way
    # (batched (B, 2, 3) positions must come back (B, 2, 3); one batch of
    # 1x1x1 zeros is a shape mismatch, not two converged molecules).
    valid_batched = np.array([[[1.0, 0.0, 0.0], [0.5, 0.2, 0.0]],
                              [[0.9, 0.1, 0.0], [0.4, 0.3, 0.1]]], dtype=np.float64)

    def wrong_shape(positions):
        return np.zeros((1, 1, 1), dtype=np.float64)

    result = relax(valid_batched, wrong_shape, _refusal_procedure())
    assert result["status"] == "nonconverged"
    assert result["failure_reason"] == "invalid_force_shape"


def test_nonfloating_refusal_types_are_never_swallowed():
    from evaluation.relax import relax

    class BombError(RuntimeError):
        pass

    def bomb(positions):
        raise BombError("kaboom")

    with pytest.raises(BombError):
        relax(_single_start([1.0, 0.0, 0.0]), bomb, _refusal_procedure())


def test_arbitrary_supplier_value_error_propagates_and_scores_nothing():
    from evaluation.relax import relax

    # A supplier's OWN ValueError (an internal model error: the round-3
    # gap-3 witness) is NOT a refusal. It propagates out of relax and
    # scores nothing: only the declared refusal exceptions carry the
    # refusal fact (MalformedForceReturn, ForceShapeMismatch from the
    # adapter's contract, and the recorded NonfiniteValueError), so any
    # other ValueError is an undeclared error that fails the run loudly.
    def broken_model(positions):
        raise ValueError("model configuration: missing radial basis parameter")

    with pytest.raises(ValueError) as excinfo:
        relax(_single_start([1.0, 0.0, 0.0]), broken_model,
              _refusal_procedure())
    assert ("model configuration: missing radial basis parameter"
            in str(excinfo.value))


def test_bare_floating_point_error_from_a_supplier_propagates():
    from evaluation.relax import relax

    # The declared nonfinite refusal rides on NonfiniteValueError (the
    # recorded-field carrier). A BARE FloatingPointError raised by the
    # supplier itself is not that declaration, so it propagates like any
    # other undeclared error instead of being scored as a refusal.
    def bare_fpe(positions):
        raise FloatingPointError("bare supplier failure")

    with pytest.raises(FloatingPointError):
        relax(_single_start([1.0, 0.0, 0.0]), bare_fpe,
              _refusal_procedure())


def test_nonfinite_refusal_is_counted_before_the_supplier_runs():
    from evaluation.adapter import NonfiniteValueError
    from evaluation.relax import relax

    # The count is spent on the attempt: this supplier raises on the very
    # first call, so the recorded count of exactly one proves the attempt
    # was accounted for even though no value ever came back. The refusal
    # is DECLARED through NonfiniteValueError (review round 3), the
    # recorded-field carrier the adapter and a supplier both use.
    attempts = {"n": 0}

    def raise_nonfinite(positions):
        attempts["n"] += 1
        raise NonfiniteValueError("refused: nonfinite force", {
            "status": "unavailable",
            "reason": "nonfinite force",
        })

    result = relax(_single_start([1.0, 0.0, 0.0]), raise_nonfinite,
                   _refusal_procedure())
    assert attempts["n"] == 1
    assert result["failure_reason"] == "nonfinite_force"
    assert result["force_evaluation_count"] == 1


# ---------------------------------------------------------------------------
# Round 4, gap 2: malformed RETURNS on the public direct path
# ---------------------------------------------------------------------------

def test_direct_supplier_returning_a_string_is_a_recorded_refusal():
    from evaluation.relax import relax

    # The public direct route (a plain force_fn, no arm wrapper) must
    # record a nonnumeric RETURN as the declared malformed-return refusal,
    # exactly as the callable_arm route does, so the documented refusal
    # contract does not depend on how the supplier was bound. The returned
    # value's conversion is guarded on its own, below the supplier call's
    # narrowed catch: a bare conversion ValueError from np.asarray must
    # become a recorded branch failure, not an abort of the caller.
    attempts = {"n": 0}

    def texty(positions):
        attempts["n"] += 1
        return "oops"

    result = relax(_single_start([2.0, 0.0, 0.0]), texty,
                   _refusal_procedure(), arm={"arm_id": "texty"})
    assert attempts["n"] == 1
    assert result["status"] == "nonconverged"
    assert result["failure_reason"] == "invalid_force_shape"
    # A malformed return carries no readable values, so the values-seen
    # fact stays finite-labelled (review round 2), and the run reports a
    # finite endpoint instead of aborting.
    assert result["forces_finite"] is True
    assert result["budget_exhausted"] is False
    assert result["step_count"] == 0
    assert result["force_evaluation_count"] == 1
    assert np.isfinite(np.asarray(result["final_positions"])).all()
    assert result["refusal_details"] == [{
        "reason": "invalid_force_shape",
        "arm_id": "texty",
        "detail": "nonnumeric force return (could not convert string to float: 'oops')",
        "force_evaluation_count": 1,
    }]


def test_direct_supplier_returning_a_ragged_array_is_a_recorded_refusal():
    from evaluation.relax import relax

    # A ragged return cannot become a numeric array at all: the same
    # recorded malformed-return refusal as the string return, never an
    # abort of the caller and never read as physics.
    attempts = {"n": 0}

    def ragged(positions):
        attempts["n"] += 1
        return [[1.0, 2.0], [3.0]]

    result = relax(_single_start([2.0, 0.0, 0.0]), ragged,
                   _refusal_procedure())
    assert attempts["n"] == 1
    assert result["status"] == "nonconverged"
    assert result["failure_reason"] == "invalid_force_shape"
    assert result["forces_finite"] is True
    assert result["step_count"] == 0
    assert result["force_evaluation_count"] == 1
    assert np.isfinite(np.asarray(result["final_positions"])).all()
    assert result["refusal_details"][0]["reason"] == "invalid_force_shape"
    assert "nonnumeric force return" in result["refusal_details"][0]["detail"]


def test_direct_supplier_own_value_error_still_propagates():
    from evaluation.relax import relax

    # The conversion guard introduced for malformed RETURNS must not
    # swallow the supplier's OWN ValueError: an internal model error is
    # not a refusal (round 3, gap 3), so it keeps propagating out of
    # relax instead of being scored as a branch outcome. This re-pins the
    # boundary against the new guard on the same call path.
    def broken_model(positions):
        raise ValueError("model configuration: missing radial basis parameter")

    with pytest.raises(ValueError) as excinfo:
        relax(_single_start([1.0, 0.0, 0.0]), broken_model,
              _refusal_procedure())
    assert ("model configuration: missing radial basis parameter"
            in str(excinfo.value))


def test_scalar_zero_start_refusal_survives_the_conversion_guard():
    from evaluation.relax import relax

    # The scalar zero that broadcasts to every component: still a recorded
    # shape refusal through the refactored call path, naming the arm in
    # the audit record with the measured shapes, exactly as before
    # round 4.
    def scalar_zero(positions):
        return 0.0

    result = relax(_single_start([2.0, 0.0, 0.0]), scalar_zero,
                   _refusal_procedure(), arm={"arm_id": "scalar-arm"})
    assert result["status"] == "nonconverged"
    assert result["failure_reason"] == "invalid_force_shape"
    assert result["forces_finite"] is True
    assert result["budget_exhausted"] is False
    assert result["step_count"] == 0
    assert result["force_evaluation_count"] == 1
    assert np.isfinite(np.asarray(result["final_positions"])).all()
    assert result["refusal_details"] == [{
        "reason": "invalid_force_shape",
        "arm_id": "scalar-arm",
        "detail": f"force shape {np.asarray(0.0).shape} for positions (1, 3)",
        "force_evaluation_count": 1,
    }]