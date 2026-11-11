"""Bind the reviewed Stage 2 solver and audit one MACE cap unit."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np

from mace_loader import _sha256


CAPS = (0.05, 0.10, 0.25, 1.00)
SOURCE_PINS = {
    "capped_relaxer": ("relax_capped", "9ee0dfd0fa8254f8e501e5d0f8d626ee9bbc03c18551049d725a6d7b340fb187"),
    "capped_pass": ("pass_copy", "14421489bb2f34db5e17f54ee8f7f6257d3fdd99f64f1055531a211f7dc5412c"),
}


def load_solver(record):
    """Execute the source buffers reviewed at Stage 2 commit e709d0d."""
    sources = {}
    for key, (name, expected) in SOURCE_PINS.items():
        path = Path(record["inputs"][key]["path"])
        if path.resolve() != path or not path.is_file():
            raise ValueError(f"Stage 2 {name} source path differs")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Stage 2 {name} source digest differs")
        sources[f"stage2_capped.{name}"] = path, data
    names = ("stage2_capped", *sources)
    if any(name in sys.modules for name in names):
        raise ValueError("Stage 2 solver modules were already imported")
    package = ModuleType(names[0])
    package.__path__ = []
    sys.modules[names[0]] = package
    try:
        for name, (path, data) in sources.items():
            module = ModuleType(name)
            module.__file__ = str(path)
            module.__package__ = names[0]
            sys.modules[name] = module
            exec(compile(data, str(path), "exec"), vars(module))
        return (sys.modules["stage2_capped.relax_capped"].relax_capped,
                sys.modules["stage2_capped.pass_copy"].second_pass)
    finally:
        for name in names:
            sys.modules.pop(name, None)


def prepare_cap(record):
    cap = record["movement_cap_A"]
    if type(cap) not in (float, int) or cap not in CAPS:
        raise ValueError("Movement cap differs from the prespecified values")
    root = Path(record["uncapped_root"]).resolve()
    output = Path(record["output"]).resolve()
    if root == output or root in output.parents or output in root.parents:
        raise ValueError("Capped output overlaps the uncapped control")
    evidence = json.loads(Path(record["inputs"]["uncapped_evidence"]["path"]).read_bytes())
    audit = Path(record["inputs"]["uncapped_audit"]["path"])
    if evidence["status"] != "COMPLETE_AUDITED" or _sha256(audit) != evidence["independent_audit_sha256"]:
        raise ValueError("Uncapped audit identity differs")
    for name, expected in evidence["retained_artifact_sha256"].items():
        file = root / name
        if file.parent != root or file.is_symlink() or _sha256(file) != expected:
            raise ValueError(f"Uncapped control artifact differs: {name}")
    return root, *load_solver(record)


@contextmanager
def trace_writer(path):
    """Keep completed force attempts even when a start stops mid-run."""
    attempts = []
    with path.open("x", encoding="utf-8") as stream:
        def append(row):
            value = {**row, "positions": row["positions"].tolist()}
            stream.write(json.dumps(value, allow_nan=False, separators=(",", ":")) + "\n")
            stream.flush()
            attempts.append(row)

        yield attempts, append


def verify_trace(attempts, count, cap, start, final):
    """Check each trial against the latest accepted positions."""
    if not attempts or len(attempts) != count or attempts[0]["kind"] != "initial":
        raise ValueError("Capped trace length or initial row differs")
    from scipy.sparse.csgraph import connected_components

    accepted = np.asarray(attempts[0]["positions"])
    if not np.array_equal(accepted, start):
        raise ValueError("Capped trace initial coordinates differ from the authenticated start")
    first_disconnected = None
    trial_moves, accepted_moves = [], []
    capped_trials = 0
    for i, row in enumerate(attempts, 1):
        positions = np.asarray(row["positions"])
        if (row["attempt_index"] != i or positions.shape != accepted.shape
                or not np.isfinite(positions).all()):
            raise ValueError("Invalid capped attempt identity or positions")
        if i > 1:
            scale, effective, norm = (row[key] for key in ("scale_s", "scale_effective", "force_norm_max_f"))
            if (not all(np.isfinite(x) and x > 0 for x in (scale, effective, norm))
                    or effective != min(scale, cap / norm)
                    or row["capped"] is not (effective < scale)):
                raise ValueError("Capped trial binding flag or scale differs")
            move = float(np.linalg.norm(positions - accepted, axis=-1).max())
            if (row["kind"] != "trial" or move > cap * (1 + 1e-12)
                    or not np.isclose(move, row["max_displacement_A"], rtol=0, atol=1e-12)):
                raise ValueError("Capped trial displacement differs")
            trial_moves.append(move)
            capped_trials += int(row["capped"])
            if row["accepted"] is True:
                accepted = positions
                accepted_moves.append(move)
        if first_disconnected is None:
            adjacency = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1) < 5
            if connected_components(adjacency, directed=False, return_labels=False) > 1:
                first_disconnected = i
    if not np.array_equal(accepted, final):
        raise ValueError("Capped endpoint differs from last accepted coordinates")
    adjacency = np.linalg.norm(accepted[:, None] - accepted[None, :], axis=-1) < 5
    return {"capped_trial_count": capped_trials,
            "accepted_travel_A": sum(accepted_moves), "attempted_travel_A": sum(trial_moves),
            "max_accepted_displacement_A": max(accepted_moves, default=None),
            "max_trial_displacement_A": max(trial_moves, default=None),
            "first_disconnected_attempt": first_disconnected,
            "endpoint_connected": connected_components(adjacency, directed=False, return_labels=False) == 1}


def verify_unbound(current, historical, fields, final, historical_final):
    if (any(current[name] != historical[name] for name in fields)
            or not np.array_equal(final, historical_final)):
        raise ValueError("Cap never bound but the historical outcome or endpoint changed")
