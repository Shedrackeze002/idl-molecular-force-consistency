"""Result-row, trace and label-record schema for the Stage 2 movement-cap study.

Defined before rules.py (plan M1a). Three pieces:

- Result rows: one JSON object per relaxation, validated field by field. Every
  row carries the cap label and the relaxer SHA-256 (spec section 5). The
  control cap (c = infinity) is stored as movement_cap_A null with the label
  "cinf"; infinity and NaN are never written (json allow_nan=False).
- Traces: one binary .npy structured array per relaxation, one record per force
  attempt (positions, scale, force norm, displacement, flags). Positions are
  binary because a full JSON record is about 1,600 bytes, so 840,000 attempts
  would take about 1.34 GB (1.25 GiB), over the 1 GiB storage ceiling (JSON
  positions alone are 1.07 GB, just under it). The binary record is 561 bytes,
  about 0.47 GB. The row carries the trace's SHA-256.
- Label records: the cap-level and sweep-level conclusion records of spec
  section 8, with the fields printed beside every label.
"""
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np

# ---- Caps ---------------------------------------------------------------

CAP_LABELS = {None: "cinf", 0.05: "c0p05", 0.10: "c0p10", 0.25: "c0p25", 1.0: "c1p00"}
CAP_VALUES = {label: value for value, label in CAP_LABELS.items()}
CONTROL_LABEL = "cinf"


def cap_label(movement_cap_A):
    """Label for a study cap. None or +inf is the control; other values must be
    exactly one of the study caps (0.05, 0.10, 0.25, 1.0)."""
    if movement_cap_A is None or movement_cap_A == float("inf"):
        return CONTROL_LABEL
    if isinstance(movement_cap_A, bool) or movement_cap_A not in CAP_LABELS:
        raise ValueError(f"movement cap {movement_cap_A!r} is not one of the study caps")
    return CAP_LABELS[movement_cap_A]


def cap_value(label):
    """Inverse of cap_label: None for the control, else the float cap."""
    if label not in CAP_VALUES:
        raise ValueError(f"unknown cap label {label!r}")
    return CAP_VALUES[label]


# ---- Vocabulary ---------------------------------------------------------

CATEGORY_NAMES = ("matched_reference", "unmatched_converged", "ambiguous",
                  "unresolved", "nonconverged")  # same tuple as evaluation/outcomes.py
STOP_REASONS = (None, "budget_exhausted", "nonfinite_force", "invalid_force_shape",
                "no_progress")
STAGES = ("0", "0b", "1", "2")
ARMS = ("energy", "direct", "consistency")
START_KINDS = ("original", "reference")
LABELS = ("direction stated", "single-cap observation",
          "no direction (informative)", "uninformative (floor or gate)")
EXPLORATORY_LABEL = "exploratory, post-hoc, not MAE-matched"
LAMBDAS = (0.1, 1.0)


# ---- Result rows --------------------------------------------------------

def _is_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value)


def _is_sha(value):
    return isinstance(value, str) and len(value) == 64 and \
        all(c in "0123456789abcdef" for c in value)


def _kind_ok(kind, value):
    if kind == "str":
        return isinstance(value, str) and value != ""
    if kind == "int":
        return _is_int(value) and value >= 0
    if kind == "float":
        return _is_real(value)
    if kind == "bool":
        return isinstance(value, bool)
    if kind == "sha256":
        return _is_sha(value)
    if kind.startswith("enum:"):
        return value in ENUMS[kind[5:]]
    raise AssertionError(kind)


ENUMS = {"stage": STAGES, "arm": ARMS, "start_kind": START_KINDS,
         "category": CATEGORY_NAMES, "stop": STOP_REASONS,
         "cap": tuple(CAP_VALUES)}

# name: (kind, nullable)
ROW_FIELDS = {
    "launch_id": ("str", False),
    "stage": ("enum:stage", False),
    "checkpoint_id": ("str", False),
    "arm": ("enum:arm", False),
    "consistency_lambda": ("float", True),
    "seed": ("int", False),
    "start_kind": ("enum:start_kind", False),
    "start_id": ("str", False),
    "connection": ("str", True),
    "branch": ("str", True),
    "own_minimum": ("int", False),
    "cap_label": ("enum:cap", False),
    "movement_cap_A": ("float", True),
    "relaxer_sha256": ("sha256", False),
    "exploratory_label": ("str", True),
    "outcome_category": ("enum:category", False),
    "recovered": ("bool", False),
    "stop_reason": ("enum:stop", True),
    "converged": ("bool", False),
    "budget_exhausted": ("bool", False),
    "step_count": ("int", False),
    "force_evaluation_count": ("int", False),
    "final_max_force_component_eV_per_A": ("float", True),
    "max_accepted_displacement_A": ("float", True),
    "endpoint_connected": ("bool", True),
    "first_disconnection_attempt": ("int", True),
    "trial_count": ("int", False),
    "capped_trial_count": ("int", False),
    "capped_fraction": ("float", True),
    "first_capped_attempt": ("int", True),
    "accepted_trial_count": ("int", False),
    "rejected_trial_count": ("int", False),
    "refused_trial_count": ("int", False),
    "accepted_travel_A": ("float", False),
    "attempted_travel_A": ("float", False),
    "elapsed_s": ("float", False),
    # Control (cinf) rows only: fractions of accepted steps above 0.25 and 1.0
    # angstrom, and whether this control run would bind at c = 1.0.
    "accepted_step_fraction_gt_0p25": ("float", True),
    "accepted_step_fraction_gt_1p0": ("float", True),
    "would_bind_at_1p0": ("bool", True),
    "trace_file": ("str", False),
    "trace_sha256": ("sha256", False),
}
_CONTROL_ONLY = ("accepted_step_fraction_gt_0p25", "accepted_step_fraction_gt_1p0",
                 "would_bind_at_1p0")


def row_problems(row):
    """List every way a row breaks the schema (empty means valid)."""
    if not isinstance(row, dict):
        return ["row is not an object"]
    problems = []
    missing = set(ROW_FIELDS) - set(row)
    extra = set(row) - set(ROW_FIELDS)
    if missing:
        problems.append(f"missing fields: {sorted(missing)}")
    if extra:
        problems.append(f"unknown fields: {sorted(extra)}")
    if problems:
        return problems
    for name, (kind, nullable) in ROW_FIELDS.items():
        value = row[name]
        if value is None:
            if not nullable:
                problems.append(f"{name} may not be null")
        elif not _kind_ok(kind, value):
            problems.append(f"{name} has an invalid value {value!r}")
    if problems:
        return problems

    # Cross-field rules.
    label = row["cap_label"]
    if row["movement_cap_A"] != CAP_VALUES[label]:
        problems.append("movement_cap_A does not match cap_label (control is null)")
    if row["arm"] == "consistency":
        if row["consistency_lambda"] not in LAMBDAS:
            problems.append("consistency rows need consistency_lambda 0.1 or 1.0")
    elif row["consistency_lambda"] is not None:
        problems.append("only consistency rows carry consistency_lambda")
    lam_one = row["consistency_lambda"] == 1.0
    if lam_one != (row["exploratory_label"] is not None):
        problems.append("exploratory_label is required exactly for lambda = 1 rows")
    if lam_one and row["exploratory_label"] != EXPLORATORY_LABEL:
        problems.append("exploratory_label has the wrong text")
    if row["recovered"] != (row["outcome_category"] == "matched_reference"):
        problems.append("recovered must equal outcome_category == matched_reference")
    if row["converged"] != (row["stop_reason"] is None):
        problems.append("converged must equal stop_reason is None")
    if row["converged"] != (row["outcome_category"] != "nonconverged"):
        problems.append("converged must equal outcome_category != nonconverged")
    if row["budget_exhausted"] != (row["stop_reason"] == "budget_exhausted"):
        problems.append("budget_exhausted must equal stop_reason == budget_exhausted")
    n = row["trial_count"]
    if row["accepted_trial_count"] + row["rejected_trial_count"] + \
            row["refused_trial_count"] != n:
        problems.append("accepted + rejected + refused trials must equal trial_count")
    if row["capped_trial_count"] > n:
        problems.append("capped_trial_count exceeds trial_count")
    if row["step_count"] != row["accepted_trial_count"]:
        problems.append("step_count must equal accepted_trial_count")
    if n == 0:
        if row["capped_fraction"] is not None:
            problems.append("capped_fraction must be null with no trials")
    elif row["capped_fraction"] is None or \
            abs(row["capped_fraction"] - row["capped_trial_count"] / n) > 1e-12:
        problems.append("capped_fraction must equal capped_trial_count / trial_count")
    if (row["capped_trial_count"] == 0) != (row["first_capped_attempt"] is None):
        problems.append("first_capped_attempt is null exactly when no trial was capped")
    if label == CONTROL_LABEL and row["capped_trial_count"] != 0:
        problems.append("a control row cannot have capped trials")
    for name in _CONTROL_ONLY:
        if (row[name] is None) == (label == CONTROL_LABEL):
            problems.append(f"{name} is set exactly for control rows")
    if row["start_kind"] == "reference" and row["connection"] is not None:
        problems.append("reference starts carry no connection")
    for name in ("accepted_travel_A", "attempted_travel_A", "elapsed_s"):
        if row[name] < 0:
            problems.append(f"{name} is negative")
    if row["accepted_travel_A"] > row["attempted_travel_A"] * (1 + 1e-9) + 1e-12:
        problems.append("accepted travel exceeds attempted travel")
    return problems


def make_row(**fields):
    """Build a row, refusing anything that breaks the schema."""
    problems = row_problems(fields)
    if problems:
        raise ValueError("invalid result row: " + "; ".join(problems))
    return dict(fields)


def dumps_row(row):
    """Canonical JSON text. Raises on a schema problem, inf or NaN."""
    problems = row_problems(row)
    if problems:
        raise ValueError("invalid result row: " + "; ".join(problems))
    return json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)


def loads_row(text):
    """Parse and validate one JSON row (rejects the non-standard inf/NaN tokens)."""
    def refuse(token):
        raise ValueError(f"non-finite JSON token {token}")
    row = json.loads(text, parse_constant=refuse)
    problems = row_problems(row)
    if problems:
        raise ValueError("invalid result row: " + "; ".join(problems))
    return row


# ---- Traces -------------------------------------------------------------

KIND_CODES = {"initial": 0, "trial": 1}
REFUSED_CODES = {None: 0, "nonfinite_force": 1, "invalid_force_shape": 2}
_FLOAT_FIELDS = ("scale_s", "scale_effective", "force_norm_max_f",
                 "max_displacement_A", "max_force_component")


def trace_dtype(n_atoms):
    """One record per force attempt. Missing values in float fields are NaN."""
    return np.dtype([("attempt_index", "<i4"), ("kind", "i1"), ("step_index", "<i4"),
                     ("trial_in_step", "<i4"), ("scale_s", "<f8"),
                     ("scale_effective", "<f8"), ("force_norm_max_f", "<f8"),
                     ("max_displacement_A", "<f8"), ("max_force_component", "<f8"),
                     ("capped", "i1"), ("accepted", "i1"), ("refused", "i1"),
                     ("converged_here", "i1"), ("positions", "<f8", (n_atoms, 3))])


def trace_from_records(records, n_atoms):
    """Pack recorder dicts (from relax_capped) into the binary trace array."""
    out = np.zeros(len(records), dtype=trace_dtype(n_atoms))
    for i, r in enumerate(records):
        out["attempt_index"][i] = r["attempt_index"]
        out["kind"][i] = KIND_CODES[r["kind"]]
        out["step_index"][i] = r["step_index"]
        out["trial_in_step"][i] = r["trial_in_step"]
        for name in _FLOAT_FIELDS:
            out[name][i] = np.nan if r[name] is None else r[name]
        out["capped"][i] = int(bool(r["capped"]))
        out["accepted"][i] = -1 if r["accepted"] is None else int(bool(r["accepted"]))
        out["refused"][i] = REFUSED_CODES[r["refused"]]
        out["converged_here"][i] = int(bool(r["converged_here"]))
        out["positions"][i] = r["positions"]
    validate_trace(out)
    return out


def records_from_trace(trace):
    """Inverse of trace_from_records (floats and positions come back bitwise)."""
    kind_names = {v: k for k, v in KIND_CODES.items()}
    refused_names = {v: k for k, v in REFUSED_CODES.items()}
    records = []
    for row in trace:
        record = {"attempt_index": int(row["attempt_index"]),
                  "kind": kind_names[int(row["kind"])],
                  "step_index": int(row["step_index"]),
                  "trial_in_step": int(row["trial_in_step"])}
        for name in _FLOAT_FIELDS:
            value = float(row[name])
            record[name] = None if math.isnan(value) else value
        record["capped"] = bool(row["capped"])
        record["accepted"] = None if row["accepted"] < 0 else bool(row["accepted"])
        record["refused"] = refused_names[int(row["refused"])]
        record["converged_here"] = bool(row["converged_here"])
        record["positions"] = np.array(row["positions"], dtype=np.float64)
        records.append(record)
    return records


def validate_trace(trace):
    """Refuse infinities, NaN in the wrong place, and a broken attempt sequence."""
    if trace.dtype.names is None or "positions" not in trace.dtype.names:
        raise ValueError("not a trace array")
    if not np.isfinite(trace["positions"]).all():
        raise ValueError("trace positions must be finite")
    for name in _FLOAT_FIELDS:
        if np.isinf(trace[name]).any():
            raise ValueError(f"trace field {name} contains an infinity")
    if len(trace) and not np.array_equal(trace["attempt_index"],
                                         np.arange(1, len(trace) + 1)):
        raise ValueError("attempt_index must run 1..n with no gaps")
    initial = trace["kind"] == 0
    if len(trace) and not (trace["kind"][0] == 0 and not initial[1:].any()):
        raise ValueError("only the first attempt may be the initial evaluation")
    for name in ("scale_s", "scale_effective", "force_norm_max_f", "max_displacement_A"):
        if not np.array_equal(np.isnan(trace[name]), initial):
            raise ValueError(f"{name} must be NaN exactly on the initial attempt")
    if not np.array_equal(np.isnan(trace["max_force_component"]), trace["refused"] != 0):
        raise ValueError("max_force_component must be NaN exactly on refused attempts")
    if not np.array_equal(trace["accepted"] < 0, initial):
        raise ValueError("accepted is -1 exactly on the initial attempt")
    if (trace["capped"][initial] != 0).any():
        raise ValueError("the initial attempt cannot be capped")


def write_trace(path, trace):
    """Write a trace as .npy without overwriting; returns its SHA-256."""
    validate_trace(trace)
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + ".partial")
    with open(temporary, "xb") as handle:
        np.save(handle, trace, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_trace(path, expected_sha256=None):
    data = Path(path).read_bytes()
    if expected_sha256 is not None and hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f"trace {path} does not match its recorded SHA-256")
    import io
    trace = np.load(io.BytesIO(data), allow_pickle=False)
    validate_trace(trace)
    return trace


def verify_trace_link(row, directory):
    """The row's trace file exists, matches its SHA-256 and has one record per
    force evaluation."""
    trace = read_trace(Path(directory) / row["trace_file"], row["trace_sha256"])
    if len(trace) != row["force_evaluation_count"]:
        raise ValueError("trace length differs from force_evaluation_count")
    return trace


# ---- Label records (spec section 8) -------------------------------------

LABEL_FIELDS = ("scope", "cap_label", "label", "interpretable_items",
                "connections_covered", "minima_covered", "adequacy_passed",
                "reference_start_counts", "set_E", "not_evaluable", "detail")


def make_label_record(**fields):
    """A cap-level or sweep-level conclusion with the fields printed beside it."""
    missing = set(LABEL_FIELDS) - set(fields)
    extra = set(fields) - set(LABEL_FIELDS)
    if missing or extra:
        raise ValueError(f"label record fields: missing {sorted(missing)}, unknown {sorted(extra)}")
    if fields["scope"] not in ("cap", "sweep"):
        raise ValueError("scope must be 'cap' or 'sweep'")
    if (fields["scope"] == "cap") != (fields["cap_label"] in CAP_VALUES and
                                      fields["cap_label"] != CONTROL_LABEL):
        raise ValueError("cap records need a capped cap_label; sweep records need null")
    if fields["label"] not in LABELS:
        raise ValueError("label must be one of the four spec labels")
    for name, top in (("interpretable_items", 20), ("connections_covered", 5)):
        if not (_is_int(fields[name]) and 0 <= fields[name] <= top):
            raise ValueError(f"{name} must be an integer in 0..{top}")
    if not (_is_int(fields["minima_covered"]) and fields["minima_covered"] >= 0):
        raise ValueError("minima_covered must be a non-negative integer")
    if fields["adequacy_passed"] is not None and not isinstance(fields["adequacy_passed"], bool):
        raise ValueError("adequacy_passed must be a bool or null")
    for name in ("set_E", "not_evaluable"):
        values = fields[name]
        if not isinstance(values, list) or any(v not in CAP_VALUES or v == CONTROL_LABEL
                                               for v in values):
            raise ValueError(f"{name} must list capped cap labels")
    counts = fields["reference_start_counts"]
    if not isinstance(counts, dict) or any(not _is_int(v) or v < 0 for v in counts.values()):
        raise ValueError("reference_start_counts must map names to counts")
    json.dumps(fields, allow_nan=False)
    return dict(fields)
