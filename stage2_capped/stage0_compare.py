"""Stage 0 loader and comparator (spec v4 section 7). No model, no force call.

Stage 0 reruns the accepted 80 controls (five checkpoints x (10 original starts +
6 reference starts)) with the capped code at c = infinity and requires the saved
results back exactly. This module reads the saved results and compares.

Saved inputs, all read-only:
- 50 original-start rows: runs/aspirin-e3x-validation-recovery-2026-10-17/
  pass-<model>/raw-branches/NNN.json (one row per start).
- 30 reference-start relaxations: runs/aspirin-e3x-paired-trace-2026-10-18/
  NN-<model>/relaxation.json with result.json naming the task. (The other four
  tasks of that run are replays of original starts and are not used here.)

Comparison is bit-exact, with no tolerance. Original rows are compared on the
spec's list: category, stop reason, step count, force-evaluation count, flags and
bitwise final positions. Reference relaxations are compared on every field the
saved file holds (a superset of the spec's list, which names steps and
force_maximums). Alignment-derived numbers (rmsd_A) depend on a numerical
decomposition, so they are reported as informational differences, never as
mismatches.
"""
import hashlib
import json
import re
import struct
from pathlib import Path

import numpy as np

from stage2_capped import admission

RECOVERY_RUN = "runs/aspirin-e3x-validation-recovery-2026-10-17"
PAIRED_TRACE_RUN = "runs/aspirin-e3x-paired-trace-2026-10-18"
MODEL_ORDER = admission.PRIMARY_IDS
OWN_MINIMA = (1, 3, 6, 7, 8, 9)
OWN_IDS = tuple(f"Landscape17/aspirin/Conformations/coordsmin_{i}.xyz" for i in OWN_MINIMA)
ORIGINAL_PER_MODEL = 10
EXPECTED_ORIGINAL = len(MODEL_ORDER) * ORIGINAL_PER_MODEL      # 50
EXPECTED_REFERENCE = len(MODEL_ORDER) * len(OWN_IDS)           # 30
EXPECTED_TOTAL = EXPECTED_ORIGINAL + EXPECTED_REFERENCE        # 80

# Hard-stop fields of an original-start row (spec section 7).
ORIGINAL_FIELDS = ("category", "stop_reason", "step_count", "force_evaluation_count",
                   "converged", "budget_exhausted", "forces_finite", "nonfinite_force", "status")


class SavedResultError(ValueError):
    """The saved results are missing, duplicated or malformed."""


def minimum_number(reference_id):
    match = re.fullmatch(r"Landscape17/aspirin/Conformations/coordsmin_(\d+)\.xyz", reference_id)
    if not match:
        raise SavedResultError(f"not a minimum reference id: {reference_id!r}")
    return int(match.group(1))


def original_task_id(model_id, branch_key):
    return f"{model_id}/original/{branch_key}"


def reference_task_id(model_id, reference_id):
    return f"{model_id}/reference/min{minimum_number(reference_id)}"


def schedule(branch_keys):
    """The 80 Stage 0 task ids in the accepted run's order: per model, the ten
    original starts sorted by branch key, then the six references in minimum order."""
    keys = sorted(branch_keys)
    if len(keys) != ORIGINAL_PER_MODEL:
        raise SavedResultError(f"expected {ORIGINAL_PER_MODEL} branch keys, got {len(keys)}")
    tasks = []
    for model_id in MODEL_ORDER:
        tasks += [original_task_id(model_id, key) for key in keys]
        tasks += [reference_task_id(model_id, ref) for ref in OWN_IDS]
    return tasks


# ---- Loading -------------------------------------------------------------

def _read(path):
    return json.loads(Path(path).read_text())


def load_saved_original(root=None):
    """{(model_id, branch_key): saved row} for the 50 accepted original-start runs."""
    base = admission.group9_root(root) / RECOVERY_RUN
    saved = {}
    for model_id in MODEL_ORDER:
        files = sorted((base / f"pass-{model_id}" / "raw-branches").glob("*.json"))
        if len(files) != ORIGINAL_PER_MODEL:
            raise SavedResultError(f"{model_id}: {len(files)} saved rows, expected {ORIGINAL_PER_MODEL}")
        for path in files:
            row = _read(path)
            key = (model_id, row["branch_key"])
            if key in saved:
                raise SavedResultError(f"duplicate saved row for {key}")
            saved[key] = row
        keys = {k for (m, k) in saved if m == model_id}
        if len(keys) != ORIGINAL_PER_MODEL:
            raise SavedResultError(f"{model_id}: saved rows do not cover ten distinct starts")
    if len(saved) != EXPECTED_ORIGINAL:
        raise SavedResultError(f"{len(saved)} saved original rows, expected {EXPECTED_ORIGINAL}")
    branch_sets = {frozenset(k for (m, k) in saved if m == model_id) for model_id in MODEL_ORDER}
    if len(branch_sets) != 1:
        raise SavedResultError("the five models were run on different starts")
    return saved


def load_saved_reference(root=None):
    """{(model_id, reference_id): {"relaxation", "result", "directory"}} for the 30
    accepted reference-start relaxations."""
    base = admission.group9_root(root) / PAIRED_TRACE_RUN
    saved = {}
    for directory in sorted(base.glob("[0-9][0-9]-*")):
        result = _read(directory / "result.json")
        task = result["task"]
        if task["kind"] != "reference_control":
            continue
        key = (task["model_id"], task["reference_id"])
        if key in saved:
            raise SavedResultError(f"duplicate saved reference relaxation for {key}")
        saved[key] = {"relaxation": _read(directory / "relaxation.json"),
                      "result": result, "directory": directory.name}
    expected = {(m, r) for m in MODEL_ORDER for r in OWN_IDS}
    if set(saved) != expected:
        raise SavedResultError(f"saved reference relaxations differ from the 30 expected "
                               f"(missing {len(expected - set(saved))}, extra {len(set(saved) - expected)})")
    return saved


def saved_file_hashes(root=None):
    """SHA-256 of every saved file Stage 0 compares against (for the launch record)."""
    root = admission.group9_root(root)
    hashes = {}
    for model_id in MODEL_ORDER:
        for path in sorted((root / RECOVERY_RUN / f"pass-{model_id}" / "raw-branches").glob("*.json")):
            hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    for key, item in sorted(load_saved_reference(root).items()):
        for name in ("relaxation.json", "result.json"):
            path = root / PAIRED_TRACE_RUN / item["directory"] / name
            hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


# ---- Bit-exact comparison ------------------------------------------------

def _bits(value):
    return struct.pack("<d", value)


def first_difference(saved, new, path=""):
    """The path of the first bit-level difference between two nested values, or None.

    Floats compare by their 64-bit pattern (so -0.0 differs from 0.0 and a one-ulp
    change is a difference). bool and int are different types. Arrays compare as
    their lists. dict keys must match exactly."""
    if saved is new:  # the very same object (for example a shared reference record)
        return None
    if isinstance(saved, np.ndarray):
        saved = saved.tolist()
    if isinstance(new, np.ndarray):
        new = new.tolist()
    if isinstance(saved, np.generic):
        saved = saved.item()
    if isinstance(new, np.generic):
        new = new.item()
    if isinstance(saved, bool) or isinstance(new, bool):
        return None if (type(saved) is bool and type(new) is bool and saved == new) else (path or "/")
    if isinstance(saved, float) or isinstance(new, float):
        if not (isinstance(saved, float) and isinstance(new, float)):
            return path or "/"
        return None if _bits(saved) == _bits(new) else (path or "/")
    if isinstance(saved, dict) or isinstance(new, dict):
        if not (isinstance(saved, dict) and isinstance(new, dict)) or set(saved) != set(new):
            return path or "/"
        for key in saved:
            found = first_difference(saved[key], new[key], f"{path}/{key}")
            if found is not None:
                return found
        return None
    if isinstance(saved, (list, tuple)) or isinstance(new, (list, tuple)):
        if not (isinstance(saved, (list, tuple)) and isinstance(new, (list, tuple))) or len(saved) != len(new):
            return path or "/"
        for index, (a, b) in enumerate(zip(saved, new)):
            found = first_difference(a, b, f"{path}/{index}")
            if found is not None:
                return found
        return None
    return None if (type(saved) is type(new) and saved == new) else (path or "/")


def compare_original(saved_row, new_row):
    """Problems (empty means an exact match) between a saved and a new original-start
    row, plus the informational alignment difference."""
    problems = []
    for field in ORIGINAL_FIELDS:
        if field not in new_row or field not in saved_row:
            problems.append(f"{field}: missing")
        elif first_difference(saved_row[field], new_row[field]) is not None:
            problems.append(f"{field}: saved {saved_row[field]!r}, new {new_row[field]!r}")
    saved_positions = saved_row["_audit_record"]["final_positions"]
    new_positions = new_row["_audit_record"]["final_positions"]
    found = first_difference(saved_positions, new_positions)
    if found is not None:
        problems.append(f"final_positions differ at {found}")
    informational = {}
    a, b = saved_row.get("rmsd_A"), new_row.get("rmsd_A")
    if isinstance(a, float) and isinstance(b, float):
        informational["rmsd_A_difference"] = abs(a - b)
    return {"problems": problems, "informational": informational}


def compare_reference(saved_relaxation, new_result):
    """Problems between a saved relaxation.json and a new relaxation result. Every
    field the saved file holds must be bit-identical (steps and force_maximums
    included); fields only the new result has (the cap record) are ignored."""
    problems = []
    for key in saved_relaxation:
        if key not in new_result:
            problems.append(f"{key}: missing from the new result")
            continue
        found = first_difference(saved_relaxation[key], new_result[key], f"/{key}")
        if found is not None:
            problems.append(f"{key} differs at {found}")
    return {"problems": problems, "informational": {}}


def compare_all(saved_original, saved_reference, new_original, new_reference):
    """Compare complete sets. Returns {task_id: problems} for every task that differs
    (empty dict means all 80 match), and raises if a set is incomplete."""
    if set(new_original) != set(saved_original) or set(new_reference) != set(saved_reference):
        raise SavedResultError("the new results do not cover the same 80 tasks")
    mismatches = {}
    for key, saved in saved_original.items():
        outcome = compare_original(saved, new_original[key])
        if outcome["problems"]:
            mismatches[original_task_id(*key)] = outcome["problems"]
    for key, saved in saved_reference.items():
        outcome = compare_reference(saved["relaxation"], new_reference[key])
        if outcome["problems"]:
            mismatches[reference_task_id(*key)] = outcome["problems"]
    return mismatches
