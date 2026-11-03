"""M1a: row, trace and label-record schema. Synthetic data only."""
import json

import numpy as np
import pytest

from stage2_capped import row_schema as rs
from stage2_capped.relax_capped import relax_capped
from stage2_capped.tests import fixtures as fx

SHA = "ab" * 32


def sample_row(**overrides):
    row = {
        "launch_id": "launch-A", "stage": "1", "checkpoint_id": "direct-s0",
        "arm": "direct", "consistency_lambda": None, "seed": 0,
        "start_kind": "original", "start_id": "13/plus", "connection": "13",
        "branch": "plus", "own_minimum": 1, "cap_label": "c0p10",
        "movement_cap_A": 0.10, "relaxer_sha256": SHA, "exploratory_label": None,
        "outcome_category": "nonconverged", "recovered": False,
        "stop_reason": "budget_exhausted", "converged": False,
        "budget_exhausted": True, "step_count": 3, "force_evaluation_count": 8,
        "final_max_force_component_eV_per_A": 0.3,
        "max_accepted_displacement_A": 0.1, "endpoint_connected": True,
        "first_disconnection_attempt": None, "trial_count": 7,
        "capped_trial_count": 7, "capped_fraction": 1.0, "first_capped_attempt": 2,
        "accepted_trial_count": 3, "rejected_trial_count": 4,
        "refused_trial_count": 0, "accepted_travel_A": 0.3, "attempted_travel_A": 0.7,
        "elapsed_s": 0.02, "accepted_step_fraction_gt_0p25": None,
        "accepted_step_fraction_gt_1p0": None, "would_bind_at_1p0": None,
        "trace_file": "run0001.npy", "trace_sha256": SHA,
    }
    row.update(overrides)
    return row


def control_row(**overrides):
    fields = dict(stage="0", cap_label="cinf", movement_cap_A=None,
                  capped_trial_count=0, capped_fraction=0.0, first_capped_attempt=None,
                  accepted_step_fraction_gt_0p25=0.4, accepted_step_fraction_gt_1p0=0.1,
                  would_bind_at_1p0=True)
    fields.update(overrides)
    return sample_row(**fields)


# ---- caps -----------------------------------------------------------------

def test_cap_labels_round_trip():
    for value, label in rs.CAP_LABELS.items():
        assert rs.cap_label(value) == label
        assert rs.cap_value(label) == value
    assert rs.cap_label(float("inf")) == "cinf"
    assert rs.cap_label(0.1) == "c0p10" and rs.cap_label(1) == "c1p00"


@pytest.mark.parametrize("bad", [0.07, 0.0, -0.05, True, 2.0, float("nan")])
def test_non_study_caps_have_no_label(bad):
    with pytest.raises(ValueError):
        rs.cap_label(bad)
    with pytest.raises(ValueError):
        rs.cap_value("c0p07")


def test_vocabulary_matches_the_frozen_outcome_taxonomy_and_relaxer():
    from evaluation.outcomes import CATEGORY_NAMES
    assert rs.CATEGORY_NAMES == tuple(CATEGORY_NAMES)
    start = fx.molecule(5, seed=1)
    reasons = {relax_capped(start, force, fx.procedure())["failure_reason"]
               for force in [fx.harmonic(start * 0, 0.4), fx.constant(3.0),
                             fx.returns_nan(), fx.returns(0.0), fx.opposing()]}
    assert reasons <= set(rs.STOP_REASONS)


# ---- rows -----------------------------------------------------------------

def test_valid_rows_round_trip_through_json():
    for row in (sample_row(), control_row()):
        text = rs.dumps_row(row)
        assert rs.loads_row(text) == row
        assert "Infinity" not in text and "NaN" not in text
    assert json.loads(rs.dumps_row(control_row()))["movement_cap_A"] is None


def test_make_row_returns_a_copy_of_a_valid_row():
    row = rs.make_row(**sample_row())
    assert row == sample_row()


def test_lambda_one_row_needs_the_exploratory_label():
    base = dict(arm="consistency", checkpoint_id="consistency-1-s0")
    ok = sample_row(consistency_lambda=1.0, exploratory_label=rs.EXPLORATORY_LABEL, **base)
    assert rs.row_problems(ok) == []
    assert rs.row_problems(sample_row(consistency_lambda=1.0, **base))
    assert rs.row_problems(sample_row(consistency_lambda=1.0, exploratory_label="post-hoc",
                                      **base))
    assert rs.row_problems(sample_row(consistency_lambda=0.1,
                                      exploratory_label=rs.EXPLORATORY_LABEL, **base))
    assert rs.row_problems(sample_row(consistency_lambda=0.1, **base)) == []
    assert rs.row_problems(sample_row(consistency_lambda=10.0, **base))
    assert rs.row_problems(sample_row(consistency_lambda=0.1))   # direct arm with lambda


@pytest.mark.parametrize("change", [
    {"movement_cap_A": 0.25},                         # label and value disagree
    {"cap_label": "c0p07", "movement_cap_A": 0.07},
    {"relaxer_sha256": "xyz"},
    {"trace_sha256": SHA.upper()},
    {"recovered": True},                              # category says not recovered
    {"stop_reason": None},                            # converged must follow
    {"converged": True},
    {"budget_exhausted": False},
    {"trial_count": 8},                               # accepted+rejected+refused != trials
    {"step_count": 4},
    {"capped_trial_count": 9, "capped_fraction": 9 / 7},
    {"capped_fraction": 0.5},
    {"first_capped_attempt": None},
    {"capped_trial_count": 0, "capped_fraction": 0.0},
    {"accepted_travel_A": 0.9},                       # more than attempted
    {"elapsed_s": -1.0},
    {"stage": "3"}, {"arm": "mace"}, {"outcome_category": "recovered"},
    {"stop_reason": "unknown"},
    {"seed": True}, {"seed": -1}, {"own_minimum": 1.5},
    {"connection": "13", "start_kind": "reference"},
    {"accepted_step_fraction_gt_0p25": 0.1},          # control-only field on a capped row
    {"elapsed_s": float("inf")}, {"elapsed_s": float("nan")},
    {"accepted_travel_A": float("inf")},
    {"final_max_force_component_eV_per_A": float("inf")},
    {"launch_id": ""}, {"launch_id": None},
])
def test_invalid_rows_are_refused(change):
    assert rs.row_problems(sample_row(**change))
    with pytest.raises(ValueError):
        rs.make_row(**sample_row(**change))


def test_converged_follows_the_outcome_category_in_both_directions():
    """Only 'nonconverged' is an unconverged outcome (outcomes.classify_outcome); every other category is
    a converged run. Each row below agrees with its stop reason, so only the category invariant can fail."""
    converged_but_nonconverged = sample_row(stop_reason=None, converged=True, budget_exhausted=False)
    problems = rs.row_problems(converged_but_nonconverged)
    assert problems == ["converged must equal outcome_category != nonconverged"]
    unconverged_but_matched = sample_row(outcome_category="matched_reference", recovered=True)
    assert rs.row_problems(unconverged_but_matched) == ["converged must equal outcome_category != nonconverged"]
    for category in ("matched_reference", "unmatched_converged", "ambiguous", "unresolved"):
        row = sample_row(outcome_category=category, recovered=category == "matched_reference",
                         stop_reason=None, converged=True, budget_exhausted=False)
        assert rs.row_problems(row) == [], category
    assert rs.row_problems(sample_row()) == []                    # nonconverged, budget exhausted
    assert rs.row_problems(sample_row(stop_reason="no_progress", budget_exhausted=False)) == []


def test_control_rows_need_the_control_only_fields_and_no_capped_trials():
    assert rs.row_problems(control_row()) == []
    assert rs.row_problems(sample_row(cap_label="cinf", movement_cap_A=None))
    bad = control_row()
    bad["would_bind_at_1p0"] = None
    assert rs.row_problems(bad)
    assert rs.row_problems(control_row(capped_trial_count=1, capped_fraction=1 / 7,
                                       first_capped_attempt=2))


def test_missing_and_unknown_fields_are_refused():
    row = sample_row()
    del row["trace_sha256"]
    assert rs.row_problems(row)
    assert rs.row_problems(dict(sample_row(), extra=1))
    assert rs.row_problems("not a dict")


def test_zero_trial_rows_are_valid():
    row = sample_row(trial_count=0, capped_trial_count=0, capped_fraction=None,
                     first_capped_attempt=None, accepted_trial_count=0,
                     rejected_trial_count=0, step_count=0, accepted_travel_A=0.0,
                     attempted_travel_A=0.0, max_accepted_displacement_A=None,
                     force_evaluation_count=1, stop_reason=None, converged=True,
                     budget_exhausted=False, outcome_category="matched_reference",
                     recovered=True)
    assert rs.row_problems(row) == []


def test_dumps_and_loads_refuse_non_finite_numbers():
    with pytest.raises(ValueError):
        rs.dumps_row(sample_row(elapsed_s=float("inf")))
    text = rs.dumps_row(sample_row()).replace('"elapsed_s":0.02', '"elapsed_s":Infinity')
    with pytest.raises(ValueError):
        rs.loads_row(text)
    text = rs.dumps_row(sample_row()).replace('"elapsed_s":0.02', '"elapsed_s":NaN')
    with pytest.raises(ValueError):
        rs.loads_row(text)


def row_from_result(result, cap, **overrides):
    """Fill a row from a real relax_capped result: shows the schema covers the
    relaxer's own fields."""
    c = result["cap_record"]
    n = c["trial_count"]
    label = rs.cap_label(cap)
    row = sample_row(
        cap_label=label, movement_cap_A=rs.cap_value(label),
        outcome_category="matched_reference" if result["converged"] else "nonconverged",
        recovered=bool(result["converged"]), stop_reason=result["failure_reason"],
        converged=result["converged"], budget_exhausted=result["budget_exhausted"],
        step_count=result["step_count"],
        force_evaluation_count=result["force_evaluation_count"],
        final_max_force_component_eV_per_A=result["final_max_force_component_eV_per_A"],
        max_accepted_displacement_A=c["max_accepted_displacement_A"],
        trial_count=n, capped_trial_count=c["capped_trial_count"],
        capped_fraction=(c["capped_trial_count"] / n) if n else None,
        first_capped_attempt=c["first_capped_attempt"],
        accepted_trial_count=c["accepted_trial_count"],
        rejected_trial_count=c["rejected_trial_count"],
        refused_trial_count=c["refused_trial_count"],
        accepted_travel_A=c["accepted_travel_A"], attempted_travel_A=c["attempted_travel_A"],
        accepted_step_fraction_gt_0p25=0.0 if label == "cinf" else None,
        accepted_step_fraction_gt_1p0=0.0 if label == "cinf" else None,
        would_bind_at_1p0=False if label == "cinf" else None)
    row.update(overrides)
    return row


@pytest.mark.parametrize("cap", [None, 0.05, 0.10, 0.25, 1.0], ids=str)
@pytest.mark.parametrize("name", ["far_start", "stiff", "refused"])
def test_rows_built_from_real_relaxations_validate(name, cap):
    start = fx.molecule(6, seed=1)
    target = fx.molecule(6, seed=2)
    force = {"far_start": fx.harmonic(target, 2.0), "stiff": fx.harmonic(target, 40.0),
             "refused": fx.flaky(fx.harmonic(target, 0.4), 3, fx.nan_return)}[name]
    start = start * 30.0 if name == "far_start" else start
    result = relax_capped(start, force, fx.procedure(), movement_cap_A=cap)
    row = row_from_result(result, cap)
    if row["capped_trial_count"] == 0 and cap is not None:
        row["first_capped_attempt"] = None
    assert rs.row_problems(row) == [], rs.row_problems(row)
    assert rs.loads_row(rs.dumps_row(row)) == row


# ---- traces ---------------------------------------------------------------

def make_trace(n_atoms=6, cap=0.1, name="stiff"):
    start = fx.molecule(n_atoms, seed=1)
    target = fx.molecule(n_atoms, seed=2)
    force = fx.harmonic(target, 40.0 if name == "stiff" else 0.4)
    records = []
    result = relax_capped(start, force, fx.procedure(), movement_cap_A=cap,
                          recorder=records.append)
    return result, records


def test_trace_round_trips_bitwise():
    from stage2_capped.prefix_gate import records_equal
    for cap in (None, 0.05, 1.0):
        _, records = make_trace(cap=cap)
        trace = rs.trace_from_records(records, n_atoms=6)
        back = rs.records_from_trace(trace)
        assert len(back) == len(records)
        assert all(records_equal(a, b) for a, b in zip(records, back))
        assert records[0]["kind"] == "initial" and records[0]["scale_s"] is None


def test_trace_with_a_refused_attempt_round_trips():
    start = fx.molecule(5, seed=4)
    records = []
    relax_capped(start, fx.flaky(fx.harmonic(start * 0, 0.4), 3, fx.nan_return),
                 fx.procedure(), movement_cap_A=0.1, recorder=records.append)
    assert records[-1]["refused"] == "nonfinite_force" and records[-1]["max_force_component"] is None
    back = rs.records_from_trace(rs.trace_from_records(records, n_atoms=5))
    assert back[-1]["refused"] == "nonfinite_force" and back[-1]["accepted"] is False


def test_trace_file_write_read_and_link(tmp_path):
    result, records = make_trace()
    trace = rs.trace_from_records(records, n_atoms=6)
    digest = rs.write_trace(tmp_path / "run.npy", trace)
    row = sample_row(trace_file="run.npy", trace_sha256=digest,
                     force_evaluation_count=result["force_evaluation_count"])
    loaded = rs.verify_trace_link(row, tmp_path)
    assert loaded.tobytes() == trace.tobytes()
    assert not list(tmp_path.glob("*.partial"))
    with pytest.raises(FileExistsError):
        rs.write_trace(tmp_path / "run.npy", trace)


def test_trace_link_detects_tampering_and_length_mismatch(tmp_path):
    result, records = make_trace()
    digest = rs.write_trace(tmp_path / "run.npy", rs.trace_from_records(records, 6))
    data = bytearray((tmp_path / "run.npy").read_bytes())
    data[-1] ^= 0x01
    (tmp_path / "bad.npy").write_bytes(bytes(data))
    with pytest.raises(ValueError, match="SHA-256"):
        rs.read_trace(tmp_path / "bad.npy", digest)
    row = sample_row(trace_file="run.npy", trace_sha256=digest,
                     force_evaluation_count=result["force_evaluation_count"] + 1)
    with pytest.raises(ValueError, match="force_evaluation_count"):
        rs.verify_trace_link(row, tmp_path)


def test_trace_writing_is_deterministic(tmp_path):
    _, records = make_trace()
    trace = rs.trace_from_records(records, 6)
    assert rs.write_trace(tmp_path / "a.npy", trace) == rs.write_trace(tmp_path / "b.npy", trace)


def test_trace_validation_refuses_bad_arrays():
    _, records = make_trace()
    good = rs.trace_from_records(records, 6)
    for mutate in (
            lambda t: t["positions"].__setitem__((1, 0, 0), np.inf),
            lambda t: t["positions"].__setitem__((1, 0, 0), np.nan),
            lambda t: t["scale_s"].__setitem__(2, np.inf),
            lambda t: t["scale_effective"].__setitem__(2, np.nan),
            lambda t: t["scale_s"].__setitem__(0, 0.2),            # initial must be NaN
            lambda t: t["attempt_index"].__setitem__(2, 99),
            lambda t: t["kind"].__setitem__(2, 0),
            lambda t: t["max_force_component"].__setitem__(1, np.nan),
            lambda t: t["accepted"].__setitem__(0, 1),
            lambda t: t["capped"].__setitem__(0, 1)):
        broken = good.copy()
        mutate(broken)
        with pytest.raises(ValueError):
            rs.validate_trace(broken)
    with pytest.raises(ValueError):
        rs.validate_trace(np.zeros(3))


def test_binary_trace_fits_the_storage_ceiling_and_json_would_not():
    itemsize = rs.trace_dtype(21).itemsize                    # aspirin has 21 atoms
    assert itemsize == 561
    attempts = 840_000
    assert attempts * itemsize < 0.5e9                        # about 0.47 GB
    assert attempts * itemsize < 2 ** 30
    positions = np.random.default_rng(0).normal(size=(21, 3))
    record = {"attempt_index": 123456, "kind": "trial", "step_index": 12345,
              "trial_in_step": 2, "scale_s": 0.1234567890123456,
              "scale_effective": 0.0123456789012345, "force_norm_max_f": 12.345678901234567,
              "capped": True, "max_displacement_A": 0.05, "accepted": False,
              "refused": None, "converged_here": False,
              "max_force_component": 3.4567890123456789, "positions": positions.tolist()}
    json_bytes = len(json.dumps(record, separators=(",", ":")))
    assert json_bytes > 1500
    assert attempts * json_bytes > 2 ** 30                    # a full JSON record exceeds 1 GiB


# ---- label records --------------------------------------------------------

def label(**overrides):
    fields = dict(scope="cap", cap_label="c0p25", label="no direction (informative)",
                  interpretable_items=12, connections_covered=4, minima_covered=2,
                  adequacy_passed=True,
                  reference_start_counts={"direct": 3, "consistency": 2},
                  set_E=["c0p25"], not_evaluable=[], detail="")
    fields.update(overrides)
    return fields


def test_label_records_validate_and_serialize():
    assert rs.make_label_record(**label())["label"] in rs.LABELS
    sweep = rs.make_label_record(**label(scope="sweep", cap_label=None,
                                         label="uninformative (floor or gate)",
                                         set_E=[], not_evaluable=["c0p05"]))
    json.dumps(sweep, allow_nan=False)
    assert len(rs.LABELS) == 4


@pytest.mark.parametrize("change", [
    {"label": "significant effect"}, {"scope": "cap", "cap_label": None},
    {"scope": "sweep"}, {"scope": "other"}, {"cap_label": "cinf"},
    {"interpretable_items": 21}, {"connections_covered": 6}, {"minima_covered": -1},
    {"adequacy_passed": "yes"}, {"set_E": ["cinf"]}, {"set_E": ["c9"]},
    {"not_evaluable": "c0p05"}, {"reference_start_counts": {"direct": -1}},
    {"reference_start_counts": []}, {"interpretable_items": True},
])
def test_invalid_label_records_are_refused(change):
    with pytest.raises(ValueError):
        rs.make_label_record(**label(**change))


def test_label_record_needs_every_field():
    fields = label()
    del fields["detail"]
    with pytest.raises(ValueError):
        rs.make_label_record(**fields)
    with pytest.raises(ValueError):
        rs.make_label_record(**label(extra=1))
