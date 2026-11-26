"""Prefix-equality gate (spec section 7).

A cap binds at a trial exactly when the uncapped step is longer than the cap.
Before the first binding trial a capped run is the control run, so the control
trace predicts where each capped run must first differ:

- attempts before the first binding trial must be bitwise identical to the
  control's records;
- at the first binding trial the run state (step, trial number, s, f) is still
  the control's, the cap is recorded as binding, the applied scale is smaller,
  and the candidate differs;
- if no control trial would bind, the capped run must equal the control
  entirely (an unbound run), and for c = 1.0 the arm is a duplicate of the
  control and is reported as such.

Records are the dicts produced by relax_capped's recorder (or
row_schema.records_from_trace).
"""
import struct

import numpy as np

from stage2_capped.relax_capped import effective_scale

_STATE_FIELDS = ("kind", "step_index", "trial_in_step", "scale_s", "force_norm_max_f")


def _bits(value):
    if isinstance(value, np.ndarray):
        return ("array", value.shape, str(value.dtype), value.tobytes())
    if isinstance(value, float):
        return ("float", struct.pack("<d", value))
    if isinstance(value, dict):
        return tuple(sorted((k, _bits(v)) for k, v in value.items()))
    return ("value", value)


def records_equal(a, b):
    """Bitwise equality of two recorder records."""
    return _bits(a) == _bits(b)


def first_binding_attempt(control_records, movement_cap):
    """Attempt index of the first control trial that the cap would reduce.

    Uses the control trace's own s and f with the module's effective_scale, so
    the prediction is the same expression the capped run evaluates.
    """
    for record in control_records:
        if record["kind"] == "trial" and effective_scale(
                record["scale_s"], record["force_norm_max_f"],
                movement_cap) < record["scale_s"]:
            return record["attempt_index"]
    return None


def would_bind(control_records, movement_cap):
    return first_binding_attempt(control_records, movement_cap) is not None


def duplicates_control(control_traces, movement_cap):
    """True when no control run in the list would bind at this cap (the capped
    arm is then identical to the control)."""
    return not any(would_bind(records, movement_cap) for records in control_traces)


def trial_problems(records, movement_cap):
    """Every trial must apply exactly effective_scale(s, f, c), flag binding
    exactly when that reduces s, and stay within the cap."""
    problems = []
    for r in records:
        if r["kind"] != "trial":
            continue
        expected = effective_scale(r["scale_s"], r["force_norm_max_f"], movement_cap)
        if _bits(r["scale_effective"]) != _bits(expected):
            problems.append(f"attempt {r['attempt_index']}: applied scale is not min(s, c/f) for c={movement_cap}")
        if r["capped"] != (expected < r["scale_s"]):
            problems.append(f"attempt {r['attempt_index']}: capped flag disagrees with the cap rule")
        if movement_cap is not None and r["max_displacement_A"] > movement_cap * (1 + 1e-12):
            problems.append(f"attempt {r['attempt_index']}: displacement exceeds the cap")
    return problems[:5]


def check_prefix(control_records, capped_records, movement_cap):
    """Compare a capped run with its control. Returns a report dict with
    "status" ("unbound" or "binds"), "first_binding_attempt" and "problems"."""
    problems = trial_problems(capped_records, movement_cap)
    k = first_binding_attempt(control_records, movement_cap)
    if k is None:
        status = "unbound"
        if len(control_records) != len(capped_records):
            problems.append("unbound run has a different number of attempts than the control")
        for a, b in zip(control_records, capped_records):
            if not records_equal(a, b):
                problems.append(f"unbound run differs from the control at attempt {a['attempt_index']}")
                break
        if any(r["capped"] for r in capped_records):
            problems.append("unbound run records a capped trial")
        return {"status": status, "first_binding_attempt": None, "problems": problems}

    status = "binds"
    if len(capped_records) < k:
        problems.append(f"capped run ends before the predicted binding attempt {k}")
        return {"status": status, "first_binding_attempt": k, "problems": problems}
    for a, b in zip(control_records[:k - 1], capped_records[:k - 1]):
        if not records_equal(a, b):
            problems.append(f"capped run differs from the control before binding, at attempt {a['attempt_index']}")
            break
    control_k, capped_k = control_records[k - 1], capped_records[k - 1]
    for field in _STATE_FIELDS:
        if _bits(control_k[field]) != _bits(capped_k[field]):
            problems.append(f"state field {field} differs at the binding attempt {k}")
    if not capped_k["capped"]:
        problems.append(f"the cap is not recorded as binding at attempt {k}")
    if not capped_k["scale_effective"] < capped_k["scale_s"]:
        problems.append(f"applied scale was not reduced at attempt {k}")
    if np.array_equal(control_k["positions"], capped_k["positions"]):
        problems.append(f"candidate at attempt {k} equals the control's")
    earlier = [r["attempt_index"] for r in capped_records[:k - 1] if r["capped"]]
    if earlier:
        problems.append(f"capped trials before the predicted attempt: {earlier}")
    return {"status": status, "first_binding_attempt": k, "problems": problems}
