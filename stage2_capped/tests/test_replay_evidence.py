"""Plan v3 section 5 item 6: the replay evidence is measurement only. Fixture arms, no model."""
import numpy as np
import pytest

from evaluation.adapter import callable_arm
from stage2_capped import admission, replay_evidence
from stage2_capped.driver import DeadlineReached

SPECIES = ("H", "H", "H")
N_RECORDS = 11


def positions():
    rng = np.random.default_rng(3)
    return rng.normal(size=(N_RECORDS, 3, 3))


def force_fn(array):
    return -4.0 * array                                  # a fixed function of the positions, any batch shape


def arm(function=force_fn):
    return callable_arm(function, SPECIES, "fixture")


def saved_from(function=force_fn):
    return np.stack([function(p) for p in positions()])


def test_an_exact_arm_has_no_difference_and_every_record_is_bit_identical():
    evidence = replay_evidence.collect(arm(), positions(), saved_from(), clock=iter(range(10 ** 6)).__next__)
    for key in ("batch4", "batch1"):
        assert evidence[key]["max_abs_difference"] == 0.0 and evidence[key]["max_scaled_residual"] == 0.0
        assert evidence[key]["bit_identical_records"] == N_RECORDS == evidence[key]["records"]
    assert evidence["batch1_vs_batch4_max_abs"] == 0.0
    assert evidence["saved_force_abs_max"] == float(np.max(np.abs(saved_from())))


def test_the_batch_four_pass_uses_batches_of_four_and_a_last_batch_of_two_or_less():
    shapes = []

    def spy(array):
        shapes.append(array.shape)
        return -4.0 * array
    replay_evidence.collect(arm(spy), positions(), saved_from(), clock=iter(range(10 ** 6)).__next__)
    assert shapes[:3] == [(4, 3, 3), (4, 3, 3), (3, 3, 3)] and shapes[3:] == [(3, 3)] * N_RECORDS


def test_a_perturbation_is_located_and_scaled_against_the_frozen_tolerance():
    saved = saved_from()
    saved[7, 2, 1] += 3e-10                                 # outside 1e-10 + 1e-10 |s| for forces of order 1 to 10
    evidence = replay_evidence.collect(arm(), positions(), saved, clock=iter(range(10 ** 6)).__next__)
    four = evidence["batch4"]
    assert four["max_abs_location"] == {"record_index": 7, "atom": 2, "component": 1}
    assert four["max_abs_difference"] == pytest.approx(3e-10, rel=1e-3)
    assert four["max_scaled_residual"] > 1.0 and four["bit_identical_records"] == N_RECORDS - 1
    inside = saved_from()
    inside[7, 2, 1] += 1e-12
    assert replay_evidence.collect(arm(), positions(), inside, clock=iter(range(10 ** 6)).__next__)["batch4"]["max_scaled_residual"] < 1.0


def test_the_scaled_residual_is_the_largest_difference_over_atol_plus_rtol_times_the_saved_value():
    saved = np.array([[[10.0, 0.0, 0.0]]])
    forces = np.array([[[10.0 + 2e-9, 3e-10, 0.0]]])
    expected = max(2e-9 / (1e-10 + 1e-9), 3e-10 / 1e-10)
    assert replay_evidence.scaled_residual(forces, saved, 1e-10, 1e-10) == pytest.approx(expected)


def test_the_latency_is_summarised_from_the_batch_one_calls_only():
    """A clock with known steps: each batch-1 call takes the next duration, so the mean, median, p99 and maximum have exact
    values. The post-A0 guard uses `mean_seconds`, so a mean computed as the minimum (or the median) must fail here."""
    durations = [0.001, 0.002, 0.003, 0.010, 0.001, 0.001, 0.002, 0.004, 0.001, 0.005, 0.020]
    assert len(durations) == N_RECORDS
    ticks, now = [], 0.0
    for duration in durations:
        ticks += [now, now + duration]
        now += duration + 1.0
    feed = iter(ticks)
    evidence = replay_evidence.collect(arm(), positions(), saved_from(), clock=lambda: next(feed))
    latency, expected = evidence["batch1_latency"], np.array(durations)
    assert latency["count"] == N_RECORDS
    assert latency["mean_seconds"] == pytest.approx(expected.mean(), rel=1e-9) and latency["mean_seconds"] > latency["p50_seconds"]
    assert latency["p50_seconds"] == pytest.approx(np.percentile(expected, 50), rel=1e-9)
    assert latency["p99_seconds"] == pytest.approx(np.percentile(expected, 99), rel=1e-9) and latency["p99_seconds"] > latency["mean_seconds"]
    assert latency["max_seconds"] == pytest.approx(0.020, rel=1e-9)


def test_the_comparison_reports_the_signed_free_difference_the_rms_ratio_and_the_scaled_residual():
    saved = np.array([[[2.0, -4.0, 0.5]], [[3.0, 0.0, -2.0]]])
    forces = np.array([[[3.0, -4.0, 0.5]], [[3.0, 0.0, -5.0]]])
    result = replay_evidence.comparison(forces, saved, 1e-3, 1e-10)
    assert result["max_abs_difference"] == 3.0 and result["max_abs_location"] == {"record_index": 1, "atom": 0, "component": 2}
    assert result["bit_identical_records"] == 0 and result["records"] == 2
    rms_difference = np.sqrt(np.mean((forces - saved) ** 2))
    rms_saved = np.sqrt(np.mean(saved ** 2))
    assert result["rms_difference_over_rms_saved"] == pytest.approx(rms_difference / rms_saved, rel=1e-12)
    # the scaled residual divides by atol + rtol |saved| with the SAVED forces and the right tolerances (1e-3 vs 1e-10)
    # |saved| is not 1 anywhere that matters, so swapping rtol and atol in the denominator changes the value
    assert result["max_scaled_residual"] == pytest.approx(np.max(np.abs(forces - saved) / (1e-10 + 1e-3 * np.abs(saved))), rel=1e-12)
    assert result["max_scaled_residual"] == pytest.approx(3.0 / (1e-10 + 1e-3 * 2.0), rel=1e-12)


def test_the_batch_one_against_batch_four_difference_is_an_absolute_value():
    """Batch-1 forces below the batch-4 forces must give a positive maximum, not a negative one."""
    calls = []

    def lower_for_single_calls(array):
        calls.append(array.ndim)
        return -4.0 * array - (1e-6 if array.ndim == 2 else 0.0)
    evidence = replay_evidence.collect(arm(lower_for_single_calls), positions(), saved_from(),
                                       clock=iter(range(10 ** 6)).__next__)
    assert evidence["batch1_vs_batch4_max_abs"] == pytest.approx(1e-6, rel=1e-3) and evidence["batch1_vs_batch4_max_abs"] > 0


def test_a_failing_pass_is_recorded_as_an_error_and_never_raised():
    calls = []

    def flaky(array):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("model refused")
        return -4.0 * array
    evidence = replay_evidence.collect(arm(flaky), positions(), saved_from(), clock=iter(range(10 ** 6)).__next__)
    assert evidence["batch4"] == {"error": "RuntimeError: model refused"}
    assert "error" not in evidence["batch1"]                      # the other pass still ran
    assert "batch1_vs_batch4_max_abs" not in evidence


def test_a_non_finite_arm_is_an_error_not_a_crash():
    evidence = replay_evidence.collect(arm(lambda a: np.full_like(a, np.nan)), positions(), saved_from(),
                                       clock=iter(range(10 ** 6)).__next__)
    assert "error" in evidence["batch4"] and "error" in evidence["batch1"]      # the arm refuses non-finite forces
    assert evidence["batch1_latency"] == {"error": "no batch-1 latency was measured"}


def test_both_passes_failing_leaves_a_latency_error_and_no_cross_comparison():
    def broken(array):
        raise ValueError("boom")
    evidence = replay_evidence.collect(arm(broken), positions(), saved_from(), clock=iter(range(10 ** 6)).__next__)
    assert "error" in evidence["batch4"] and "error" in evidence["batch1"]
    assert evidence["batch1_latency"] == {"error": "no batch-1 latency was measured"}


def test_the_supervisors_deadline_is_not_swallowed():
    def stop(array):
        raise DeadlineReached("terminated by the supervisor")
    with pytest.raises(DeadlineReached):
        replay_evidence.collect(arm(stop), positions(), saved_from(), clock=iter(range(10 ** 6)).__next__)


def test_the_defaults_are_the_frozen_replay_settings():
    import inspect
    parameters = inspect.signature(replay_evidence.collect).parameters
    assert parameters["rtol"].default == parameters["atol"].default == admission.REPLAY_TOLERANCE == 1e-10
    assert parameters["batch"].default == admission.REPLAY_BATCH == 4
