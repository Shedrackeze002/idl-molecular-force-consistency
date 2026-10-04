"""Render the shape-recovery fixture diagnostics and freeze the start-set hash.

This script is the reporter for the shape-recovery evaluator. It does two
jobs, both verifiable from their printed output:

1. It runs a four-branch analytic fixture slice through the whole
   evaluator (adapter arm, relaxation, outcome taxonomy, report counts)
   and renders reports/shape-recovery-fixture-diagnostics.json and .md.
   It evaluates NO trained model: every force comes from the analytic
   fixture arm, and every branch is labelled partition "validation".

2. It freezes the real aspirin start-set fingerprint to
   reports/execution/shape-recovery-2026-10-03/start-set-hash.json, the
   file tests/test_shape_recovery_starts.py
   ::test_real_start_set_fingerprint_is_stable reads back. The digests it
   writes cover the branch keys, the chosen record ids, the frame indices
   and the start positions, computed with sha256 over
   json.dumps(payload, sort_keys=True).

Run it from the repository root with the evaluation package importable
(repo root plus src/ on the path) and LANDSCAPE17_ARCHIVE pointing at the
frozen archive. The archive digest is asserted, never assumed.
"""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for entry in (str(REPO), str(REPO / "src"), str(REPO / "tests")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from force_consistency.data.types import FrameRecord, PreparationPolicy

from evaluation.adapter import analytic_fixture_arm
from evaluation.outcomes import classify_outcome
from evaluation.relax import relax
from evaluation.reporter import build_report, freeze_start_set_hash, render
from evaluation.start_set import build_start_set
import analytic_fixtures as fixtures


# The frozen archive digest, asserted before anything reads it.
ARCHIVE_SHA256 = "3870ecfd6a5404c6654efba22f003f645e2f4dffffad1b5a22770d3437b3027b"

# The frozen relaxation procedure, identical to
# configs/aspirin-shape-recovery.json's relaxation block.
PROCEDURE = {
    "max_force_evaluations": 1500,
    "initial_step_size": 0.2,
    "backtracking_factor": 0.5,
    "min_backtracking_floor": 1e-08,
    "convergence_tolerance_eV_per_A": 0.01,
    "max_backtracking_attempts_per_step": 40,
}


def _fixture_records():
    """The fixture reference records: the true minimum and a distinct one."""
    minimum = np.array(fixtures.FIXTURE_MINIMUM, dtype=np.float64)
    other = minimum + np.array([2.0, 0.0, 0.0])
    records = {}
    for name, positions in (("fixture-minimum", minimum), ("fixture-other", other)):
        frozen = np.array(positions, dtype=np.float64, copy=True)
        frozen.setflags(write=False)
        records[name] = FrameRecord(
            record_id=name, molecule="analytic", kind="minimum", source_path=name,
            connection_id=None, branch=None, frame_index=None,
            species=fixtures.FIXTURE_SPECIES, positions=frozen, energy=0.0,
            forces=np.zeros_like(frozen), source_sha256="0" * 64,
        )
    return records


def _final_record(record_id, positions):
    positions = np.array(positions, dtype=np.float64, copy=True)
    positions.setflags(write=False)
    return FrameRecord(
        record_id=record_id, molecule="analytic", kind="path", source_path=record_id,
        connection_id=None, branch=None, frame_index=None,
        species=fixtures.FIXTURE_SPECIES, positions=positions, energy=0.0,
        forces=np.zeros_like(positions), source_sha256="0" * 64,
    )


def _procedure(**overrides):
    base = dict(PROCEDURE)
    base.update(overrides)
    return base


def run_fixture_slice(policy):
    """Relax four fixture branches and classify every endpoint."""
    arm = analytic_fixture_arm()
    records = _fixture_records()
    minimum = records["fixture-minimum"]
    # The wrong-shape branch's force field pulls toward a GENUINELY
    # distinct fixed point: atom-specific offsets, not a rigid shift. A
    # rigidly shifted copy of the minimum is the same equivalence class
    # under the symmetry-aware comparison and must come out matched;
    # per-atom offsets change pairwise distances, so the endpoint lands
    # outside the branch's reference class.
    per_atom = np.where(np.arange(len(fixtures.FIXTURE_MINIMUM))[:, None] % 2 == 0,
                        0.6, -0.6) * np.array([1.0, 0.0, 0.0])
    shifted = fixtures.fixture_molecule_potential(offset=per_atom)

    def arm_force_fn(positions):
        out, info = arm["forces"](positions, species=fixtures.FIXTURE_SPECIES,
                                  record_id="fixture-slice")
        assert info["status"] == "ok"
        return out

    branches = [
        # (branch key, start, force_fn, own reference, procedure)
        ("analytic/recover/plus", np.array(minimum.positions) + 0.5,
         arm_force_fn, minimum, _procedure()),
        ("analytic/recover/minus", np.array(minimum.positions) + [-0.4, 0.3, -0.2],
         arm_force_fn, minimum, _procedure()),
        ("analytic/wrongshape/plus", np.array(minimum.positions) + np.array([2.0, 0.3, 0.0]),
         lambda p: np.asarray(shifted(p), dtype=np.float64), minimum, _procedure()),
        ("analytic/wrongshape/minus", np.array(minimum.positions) + 0.5,
         arm_force_fn, minimum, _procedure(max_force_evaluations=2)),
    ]

    references = dict(records)
    evaluated = {}
    for key, start, force_fn, own, procedure in branches:
        result = relax(start, force_fn, procedure)
        final = _final_record(key + "/final", result["final_positions"])
        outcome = classify_outcome(
            final, own, references, policy,
            budget_exhausted=result["budget_exhausted"],
            converged=result["converged"],
            forces_finite=result["forces_finite"],
            seen_minimum=True,
        )
        print(f"{key}: status={result['status']} category={outcome['category']} "
              f"evals={result['force_evaluation_count']}")
        evaluated[key] = {
            "category": outcome["category"],
            "seen_minimum": outcome["seen_minimum"],
            "partition": "validation",
            "budget_exhausted": outcome["budget_exhausted"],
            "converged": outcome["converged"],
        }
    return evaluated


def freeze_real_start_set(policy):
    """Build the real start set and freeze its fingerprint."""
    archive = Path(os.environ["LANDSCAPE17_ARCHIVE"])
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if digest != ARCHIVE_SHA256:
        raise SystemExit(f"archive digest mismatch: {digest} != {ARCHIVE_SHA256}")
    from force_consistency.data.xyz import read_records

    records = read_records(archive, molecules=("aspirin",))
    manifest = json.loads(
        (REPO / "manifests" / "landscape17-aspirin-split.json").read_text())
    start_set = build_start_set(manifest, records, policy,
                                {"kind": "first_stored_frame"})
    out_dir = REPO / "reports" / "execution" / "shape-recovery-2026-10-03"
    path = freeze_start_set_hash(start_set, out_dir)
    recorded = json.loads(path.read_text())
    print(f"start-set hash frozen: {path}")
    print(f"  start_record_id_map_sha256 = {recorded['start_record_id_map_sha256']}")
    print(f"  start_set_payload_sha256   = {recorded['start_set_payload_sha256']}")
    print(f"  branches covered           = {recorded['branch_count']}")
    return path


def main():
    archive = os.environ.get("LANDSCAPE17_ARCHIVE")
    if not archive:
        raise SystemExit("Set LANDSCAPE17_ARCHIVE to the frozen Landscape17.zip first")
    policy = PreparationPolicy()

    evaluated = run_fixture_slice(policy)
    report, diagnostics = build_report(evaluated)
    out = render(report, diagnostics, REPO / "reports",
                 stem="shape-recovery-fixture-diagnostics")
    print(f"fixture diagnostics rendered: {out['json']} and {out['md']}")
    print("category totals:", json.dumps(report["category_totals"], sort_keys=True))

    hash_path = freeze_real_start_set(policy)
    print("done:", out["json"], out["md"], hash_path)


if __name__ == "__main__":
    main()