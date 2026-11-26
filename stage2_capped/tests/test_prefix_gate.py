"""M1: prefix-equality gate. The control trace predicts each capped run's first
binding trial; the capped run equals the control up to it."""
import copy

import numpy as np
import pytest

from stage2_capped.prefix_gate import (check_prefix, duplicates_control,
                                       first_binding_attempt, records_equal,
                                       would_bind)
from stage2_capped.relax_capped import relax_capped
from stage2_capped.tests import fixtures as fx

STUDY_CAPS = [0.05, 0.10, 0.25, 1.0]


def trace(start, force, proc, cap):
    records = []
    result = relax_capped(start, force, proc, movement_cap_A=cap, recorder=records.append)
    return result, records


def fixture(name):
    start = fx.molecule(6, seed=1)
    target = fx.molecule(6, seed=2)
    table = {
        "far_start": (start * 30.0, fx.harmonic(target, 2.0)),
        "stiff": (start, fx.harmonic(target, 40.0)),
        "valley": (start, fx.anisotropic(target, 30.0, 0.3)),
        "pair": (start * 0.8, fx.pair_potential()),
        "soft_near": (target + 0.05 * start, fx.harmonic(target, 0.4)),
        # s*f grows as s doubles while f barely falls: binds late for c = 0.10, 0.25
        "soft_far": (target + 8.0 * start / np.linalg.norm(start, axis=1, keepdims=True),
                     fx.harmonic(target, 0.05)),
    }
    return table[name][0], table[name][1], fx.procedure()


@pytest.mark.parametrize("cap", STUDY_CAPS, ids=str)
@pytest.mark.parametrize("name", ["far_start", "stiff", "valley", "pair", "soft_near", "soft_far"])
def test_capped_run_follows_the_control_prediction(name, cap):
    _, control = trace(*fixture(name), None)
    _, capped = trace(*fixture(name), cap)
    report = check_prefix(control, capped, cap)
    assert report["problems"] == []
    if first_binding_attempt(control, cap) is None:
        assert report["status"] == "unbound"
    else:
        assert report["status"] == "binds"


def test_the_fixtures_exercise_bound_unbound_and_late_binding():
    seen = set()
    for name in ("far_start", "stiff", "valley", "pair", "soft_near", "soft_far"):
        _, control = trace(*fixture(name), None)
        for cap in STUDY_CAPS:
            k = first_binding_attempt(control, cap)
            seen.add("unbound" if k is None else ("first_trial" if k <= 2 else "late"))
    assert seen == {"unbound", "first_trial", "late"}


def test_unbound_capped_run_equals_the_control_entirely():
    start, force, proc = fixture("soft_near")
    control_result, control = trace(start, force, proc, None)
    assert not would_bind(control, 1.0)
    capped_result, capped = trace(*fixture("soft_near"), 1.0)
    assert len(control) == len(capped)
    assert all(records_equal(a, b) for a, b in zip(control, capped))
    assert fx.bits(fx.without_cap_record(capped_result)) == fx.bits(fx.without_cap_record(control_result))
    assert capped_result["cap_record"]["capped_trial_count"] == 0


def test_c1p00_duplicate_of_the_control_is_detected():
    traces = [trace(*fixture("soft_near"), None)[1]]
    assert duplicates_control(traces, 1.0)
    traces.append(trace(*fixture("far_start"), None)[1])
    assert not duplicates_control(traces, 1.0)
    assert not duplicates_control(traces, 0.05)


def test_late_binding_fixture_binds_at_the_predicted_attempts():
    _, control = trace(*fixture("soft_far"), None)
    ks = {cap: first_binding_attempt(control, cap) for cap in STUDY_CAPS}
    assert ks[0.05] == 2 and ks[0.10] == 3 and ks[0.25] == 4 and ks[1.0] is None
    for cap in STUDY_CAPS:
        _, capped = trace(*fixture("soft_far"), cap)
        report = check_prefix(control, capped, cap)
        assert report["problems"] == [] and report["first_binding_attempt"] == ks[cap]
        if ks[cap]:
            assert [r["capped"] for r in capped if r["kind"] == "trial"][ks[cap] - 2]
            assert not any(r["capped"] for r in capped[:ks[cap] - 1])


def test_prediction_uses_all_trials_including_rejected_ones():
    # A control run whose only large-step trial is a rejected one still counts.
    _, control = trace(*fixture("stiff"), None)
    rejected_first = next(r for r in control if r["kind"] == "trial" and not r["accepted"])
    cap = 0.5 * rejected_first["scale_s"] * rejected_first["force_norm_max_f"]
    k = first_binding_attempt(control, cap)
    assert k is not None and k <= rejected_first["attempt_index"]


def test_binding_attempt_is_where_s_times_f_first_exceeds_c():
    _, control = trace(*fixture("pair"), None)
    for cap in STUDY_CAPS:
        k = first_binding_attempt(control, cap)
        manual = next((r["attempt_index"] for r in control if r["kind"] == "trial"
                       and r["scale_s"] * r["force_norm_max_f"] > cap * (1 + 1e-9)), None)
        if manual is not None:
            assert k is not None and k <= manual


def test_gate_flags_a_capped_run_that_diverges_early():
    _, control = trace(*fixture("far_start"), None)
    _, capped = trace(*fixture("far_start"), 0.25)
    k = first_binding_attempt(control, 0.25)
    assert k is not None and k > 1
    tampered = copy.deepcopy(capped)
    tampered[0]["positions"][0, 0] += 1e-9
    assert check_prefix(control, tampered, 0.25)["problems"]


def test_gate_flags_a_wrong_cap_claim():
    _, control = trace(*fixture("far_start"), None)
    _, capped = trace(*fixture("far_start"), 1.0)
    # The run used c = 1.0; judging it against the 0.05 prediction must fail.
    assert check_prefix(control, capped, 1.0)["problems"] == []
    assert check_prefix(control, capped, 0.05)["problems"]


def test_gate_flags_a_cap_that_never_bound_but_reports_capped():
    _, control = trace(*fixture("soft_near"), None)
    _, capped = trace(*fixture("soft_near"), 1.0)
    forged = copy.deepcopy(capped)
    forged[-1]["capped"] = True
    assert check_prefix(control, forged, 1.0)["problems"]


def test_gate_flags_a_capped_run_that_stops_before_the_binding_attempt():
    _, control = trace(*fixture("far_start"), None)
    k = first_binding_attempt(control, 0.05)
    _, capped = trace(*fixture("far_start"), 0.05)
    assert check_prefix(control, capped[:k - 1], 0.05)["problems"]


def test_gate_flags_a_missing_candidate_change_at_binding():
    _, control = trace(*fixture("far_start"), None)
    _, capped = trace(*fixture("far_start"), 0.05)
    k = first_binding_attempt(control, 0.05)
    forged = copy.deepcopy(capped)
    forged[k - 1]["positions"] = control[k - 1]["positions"].copy()
    assert check_prefix(control, forged, 0.05)["problems"]


def test_records_equal_is_bitwise():
    a = {"x": 0.1, "p": np.zeros((2, 3))}
    b = {"x": 0.1, "p": np.zeros((2, 3))}
    assert records_equal(a, b)
    b["x"] = float(np.nextafter(0.1, 1.0))
    assert not records_equal(a, b)
    c = {"x": 0.1, "p": np.zeros((2, 3))}
    c["p"][0, 0] = -0.0
    assert not records_equal(a, c)
