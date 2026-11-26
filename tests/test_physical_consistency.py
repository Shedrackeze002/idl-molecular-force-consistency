import json

import numpy as np
import pytest

from analytic_fixtures import FIXTURE_MINIMUM
from evaluation.adapter import analytic_fixture_arm, callable_arm
from evaluation.paired_adapter import energy_force_arm
from evaluation.physical_consistency import (
    PhysicalConsistencyConfig,
    benchmark_inference,
    closed_loop_work,
    energy_force_disagreement,
    evaluate_physical_consistency,
)


def _translated_square(target):
    offsets = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
    ])
    return np.stack([target + offset for offset in offsets])


def test_harmonic_fixture_matches_energy_gradient_and_zero_loop_work():
    arm = analytic_fixture_arm()
    displaced = FIXTURE_MINIMUM + np.linspace(-0.2, 0.2, FIXTURE_MINIMUM.size).reshape(FIXTURE_MINIMUM.shape)
    disagreement = energy_force_disagreement(arm, displaced, record_id="harmonic")
    assert disagreement["status"] == "ok"
    assert disagreement["max_abs_eV_per_A"] < 1e-9
    assert disagreement["energy_evaluation_count"] == 2 * displaced.size

    work = closed_loop_work(arm, _translated_square(FIXTURE_MINIMUM), loop_id="square")
    assert work["status"] == "ok"
    assert work["absolute_work_eV"] < 1e-12
    assert work["force_call_count"] == 1
    assert work["force_configuration_count"] == 5


def test_rotational_force_only_fixture_has_nonzero_loop_work_and_no_energy_metric():
    def rotational(positions):
        positions = np.asarray(positions)
        output = np.zeros_like(positions)
        output[..., 0] = -positions[..., 1]
        output[..., 1] = positions[..., 0]
        return output

    arm = callable_arm(rotational, species=("C",), arm_id="rotational")
    square = np.array([
        [[1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0]],
        [[-1.0, 0.0, 0.0]],
        [[0.0, -1.0, 0.0]],
        [[1.0, 0.0, 0.0]],
    ])
    disagreement = energy_force_disagreement(arm, square[0], record_id="rotational")
    assert disagreement["status"] == "unavailable"
    assert "provides no energy" in disagreement["reason"]
    work = closed_loop_work(arm, square, loop_id="diamond")
    assert work["status"] == "ok"
    assert work["signed_work_eV"] == pytest.approx(4.0)


def test_open_loop_is_refused_instead_of_silently_closed():
    arm = analytic_fixture_arm()
    with pytest.raises(ValueError, match="loop is open"):
        closed_loop_work(arm, _translated_square(FIXTURE_MINIMUM)[:-1])


def test_nonfinite_supplier_failure_is_recorded():
    arm = callable_arm(lambda positions: np.full_like(positions, np.nan),
                       species=("C",), arm_id="nonfinite")
    loop = np.zeros((4, 1, 3))
    result = closed_loop_work(arm, loop, loop_id="bad")
    assert result["status"] == "failed"
    assert result["exception_type"] == "NonfiniteValueError"
    assert result["refusal"]["reason"] == "nonfinite force"


def test_nonfinite_disagreement_arithmetic_is_recorded_as_failed():
    def paired(positions):
        return np.zeros(np.asarray(positions).shape[:-2]), np.full_like(positions, 1e200)

    arm = energy_force_arm(paired, species=("C",), arm_id="overflowing-paired")
    result = energy_force_disagreement(arm, np.zeros((1, 3)), record_id="overflow")
    assert result["status"] == "failed"
    assert result["exception_type"] == "FloatingPointError"
    json.dumps(result, allow_nan=False)


def test_nonfinite_loop_arithmetic_is_recorded_as_failed():
    arm = callable_arm(
        lambda positions: np.full_like(positions, 1e308),
        species=("C",),
        arm_id="overflowing-loop",
    )
    loop = np.array([
        [[0.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0]],
        [[0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0]],
    ])
    result = closed_loop_work(arm, loop, loop_id="overflow")
    assert result["status"] == "failed"
    assert result["exception_type"] == "FloatingPointError"
    json.dumps(result, allow_nan=False)


def test_loop_quadrature_refinement_reduces_conservative_field_error():
    def nonlinear_conservative_force(positions):
        positions = np.asarray(positions)
        x = positions[..., 0]
        y = positions[..., 1]
        output = np.zeros_like(positions)
        output[..., 0] = -3.0 * x * x * y
        output[..., 1] = -(x ** 3)
        return output

    arm = callable_arm(
        nonlinear_conservative_force, species=("C",), arm_id="nonlinear-conservative")
    loop = np.array([
        [[0.0, 0.0, 0.0]],
        [[1.0, 0.0, 0.0]],
        [[1.0, 1.0, 0.0]],
        [[0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0]],
    ])
    coarse = closed_loop_work(arm, loop)
    refined = closed_loop_work(
        arm,
        loop,
        config=PhysicalConsistencyConfig(loop_quadrature_subdivisions=8),
    )
    assert coarse["absolute_work_eV"] == pytest.approx(0.5)
    assert refined["absolute_work_eV"] < coarse["absolute_work_eV"]
    assert refined["quadrature_subdivisions_per_declared_segment"] == 8
    assert refined["force_configuration_count"] == 33


def test_combined_report_is_json_safe_and_counts_states():
    arm = analytic_fixture_arm()
    report = evaluate_physical_consistency(
        arm,
        geometries={"displaced": FIXTURE_MINIMUM + 0.1},
        loops={"square": _translated_square(FIXTURE_MINIMUM)},
        config=PhysicalConsistencyConfig(finite_difference_step_A=1e-5),
    )
    assert report["schema_version"] == "physical-consistency/v1"
    assert report["summary"] == {
        "result_count": 2, "ok": 2, "unavailable": 0, "failed": 0}
    json.dumps(report, allow_nan=False)


def test_benchmark_separates_warmup_and_measured_calls():
    arm = analytic_fixture_arm()
    result = benchmark_inference(
        arm, (FIXTURE_MINIMUM + 0.1)[None, ...], warmup_calls=2, measured_calls=5)
    assert result["status"] == "ok"
    assert result["warmup_calls"] == 2
    assert result["measured_calls"] == 5
    assert len(result["sample_seconds"]) == 5
    assert result["median_seconds_per_structure"] >= 0


def test_benchmark_rejects_empty_batches():
    with pytest.raises(ValueError, match="shape"):
        benchmark_inference(analytic_fixture_arm(), np.empty((0, 1, 3)))


@pytest.mark.parametrize("field,value", [("warmup_calls", 1.5), ("measured_calls", True)])
def test_benchmark_rejects_noninteger_call_counts(field, value):
    kwargs = {"warmup_calls": 1, "measured_calls": 1, field: value}
    with pytest.raises(ValueError, match="integer"):
        benchmark_inference(analytic_fixture_arm(), FIXTURE_MINIMUM, **kwargs)


def test_energy_force_callable_adapter_validates_paired_shapes():
    target = np.array([[0.0, 0.0, 0.0]])

    def paired(positions):
        return 0.5 * np.sum(positions ** 2, axis=(-2, -1)), -positions

    arm = energy_force_arm(paired, species=("C",), arm_id="paired")
    result = energy_force_disagreement(arm, target + 0.2)
    assert result["status"] == "ok"
    assert result["max_abs_eV_per_A"] < 1e-10
