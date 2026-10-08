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
import math
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
from evaluation.reporter import build_report, freeze_start_set_hash, render, _plain_json_field
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

# The procedure the saved audit must show as the configured baseline when a
# branch runs under a synthetic override (review round 2): the configured
# value is recorded next to the effective one so the difference is
# auditable directly from the report.
CONFIGURED_PROCEDURE = dict(PROCEDURE)


def _arm_identity(arm, supplier):
    """The arm record naming the branch's ACTUAL force supplier.

    kind/arm_id are the adapter identity (constant across this slice), and
    force_supplier is what actually produced the branch's forces. A branch
    that drives a modified potential records its own supplier here instead
    of passing under the standard arm's name (review round 2).
    """
    return {
        "kind": arm["kind"],
        "arm_id": arm["arm_id"],
        "force_supplier": supplier,
    }


def _procedure_identity(procedure, overrides):
    """The effective procedure a branch ran under, with overrides labelled.

    procedure: the dict the branch actually relaxed with. overrides: the
    key names the branch passed to _procedure(). Every overridden key gets
    a synthetic_override entry: the effective value next to the configured
    value, labelled as synthetic so a reader can never mistake the
    branch-local cap for the configured procedure (review round 2).
    """
    identity = {"effective": dict(procedure), "configured": dict(CONFIGURED_PROCEDURE)}
    if overrides:
        identity["overrides"] = {
            key: {
                "effective": procedure[key],
                "configured": CONFIGURED_PROCEDURE[key],
                "label": "synthetic override",
            }
            for key in overrides
        }
    return identity


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

    # (branch key, start, force_fn, own reference, procedure,
    #  supplier label, procedure-override keys). Every branch's audit
    # entry must name the supplier that ACTUALLY produced its forces and
    # the procedure it ACTUALLY ran under (review round 2): the audit
    # follows the branch, not the other way round.
    branches = [
        ("analytic/recover/plus", np.array(minimum.positions) + 0.5,
         arm_force_fn, minimum, _procedure(), "analytic-fixture-harmonic", ()),
        ("analytic/recover/minus", np.array(minimum.positions) + [-0.4, 0.3, -0.2],
         arm_force_fn, minimum, _procedure(), "analytic-fixture-harmonic", ()),
        ("analytic/wrongshape/plus", np.array(minimum.positions) + np.array([2.0, 0.3, 0.0]),
         lambda p: np.asarray(shifted(p), dtype=np.float64), minimum,
         _procedure(),
         "analytic-fixture-harmonic-shifted-per-atom-potential", ()),
        ("analytic/wrongshape/minus", np.array(minimum.positions) + 0.5,
         arm_force_fn, minimum, _procedure(max_force_evaluations=2),
         "analytic-fixture-harmonic", ("max_force_evaluations",)),
    ]

    references = dict(records)
    evaluated = {}
    for key, start, force_fn, own, procedure, supplier, overrides in branches:
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
        # The auditable branch record (review round 2): the endpoint,
        # matching and relaxation evidence of THIS branch, serialized as
        # plain data by the reporter. The reference is the branch's own
        # minimum record; the start identity names the displaced fixture
        # structure; the failure facts and exact force-call and step
        # counts come from the relaxation result itself.
        evaluated[key] = {
            "category": outcome["category"],
            "seen_minimum": outcome["seen_minimum"],
            "partition": "validation",
            "budget_exhausted": outcome["budget_exhausted"],
            "converged": outcome["converged"],
            "_audit_record": {
                "reference": own,
                "start": {"branch_key": f"{key}/start", "kind": "fixture-displacement"},
                # The supplier that ACTUALLY produced this branch's
                # forces, not a blanket analytic-fixture label.
                "arm": _arm_identity(arm, supplier),
                # The procedure this branch ACTUALLY ran under, with any
                # branch-level override recorded and labelled synthetic
                # next to the configured value.
                "procedure": _procedure_identity(procedure, overrides),
                "final_positions": result["final_positions"],
                "rmsd_A": outcome["rmsd_A"],
                "permutation": outcome["permutation"],
                "rotation": outcome["rotation"],
                "accepted_classes": outcome["accepted_classes"],
                "failure": {
                    "failed": result["status"] == "nonconverged",
                    "failure_reason": result["failure_reason"],
                    "budget_exhausted": result["budget_exhausted"],
                    "converged": result["converged"],
                    "forces_finite": result["forces_finite"],
                },
                "force_evaluation_count": result["force_evaluation_count"],
                "step_count": result["step_count"],
                # Per-branch timing (review item 2) is recorded on the
                # EVALUATION path, where the runner measures each branch.
                # This fixture artifact is required to regenerate byte for
                # byte, and a wall-clock value would break exactly that, so
                # the field is present and explicitly null here instead of
                # silently absent. The reason is stated in the payload's
                # timing note and in the rendered report.
                "timing_seconds": None,
            },
        }
    return evaluated


def run_nonrigid_positive_fixture(policy):
    """One positive fixture: outside the tolerance before relax, matched after.

    The start is a NONRIGID per-atom displacement of the fixture minimum
    (every row moves by a different vector, so pairwise distances change
    and no rigid motion maps it onto the reference). Its pre-relaxation
    alignment is MEASURED with align_records and must come out both
    finite (graph-isomorphic to the reference: not the graph_mismatch
    sentinel) and outside the 0.05 angstrom match tolerance; the branch
    relaxes under the CONFIGURED procedure and the endpoint must come out
    matched_reference with the measured rmsd inside the tolerance (review
    round 2). Both facts are printed and returned as plain data for the
    saved diagnostics.
    """
    from force_consistency.data.geometry import align_records

    arm = analytic_fixture_arm()
    records = _fixture_records()
    minimum = records["fixture-minimum"]
    # The per-atom displacement is constrained by the reference chemical
    # graph, which has exactly two edges, C0-H6 and C1-H6 (each 1.166 A
    # against a 1.284 A cutoff): carbons move at most ~0.13 outward in x
    # so both C-H6 distances stay in edge range, H4-H7 sits at exactly
    # the 0.4 A graph_min_distance so those two hydrogens are pushed
    # apart past the (0.4, 0.744) H-H edge band instead of into it (H4
    # toward H6's z but not inside the band against H5/H6/H7), and the
    # isolated O/H rows move freely. Every row is a different vector, so
    # the displacement is genuinely nonrigid; probe-measured pre RMSD is
    # 0.157 A, above the 0.05 tolerance, and the graph stays isomorphic
    # so the RMSD is a real measurement, never the inf sentinel.
    nonrigid = np.array([
        [0.10, 0.0, 0.0], [-0.10, 0.0, 0.0],
        [0.0, 0.20, 0.0], [0.0, -0.20, 0.0],
        [0.0, 0.0, -0.20], [0.0, 0.0, 0.10],
        [0.0, 0.0, -0.08], [0.0, 0.0, 0.20],
    ], dtype=np.float64)
    start = np.array(minimum.positions, dtype=np.float64) + nonrigid

    def arm_force_fn(positions):
        out, info = arm["forces"](positions, species=fixtures.FIXTURE_SPECIES,
                                  record_id="nonrigid-positive-slice")
        assert info["status"] == "ok"
        return out

    procedure = _procedure()
    pre = align_records(_final_record("nonrigid-positive/start", start),
                        minimum, policy)
    pre_rmsd = float(pre.rmsd)
    # The fixture's own claim, enforced where it is produced: the start
    # sits outside the tolerance BY MEASUREMENT. A graph_mismatch (inf
    # sentinel) or any non-finite pre RMSD means the displacement broke
    # the chemical graph and the number is not a measurement: the
    # fixture is misbuilt, no diagnostics are written from it.
    if not math.isfinite(pre_rmsd) or pre.search_complete is False:
        raise SystemExit(
            f"nonrigid positive fixture is misbuilt: align_records did not "
            f"measure the start (reason={pre.reason!r}, rmsd={pre_rmsd!r}); "
            f"the displacement must keep the chemical graph isomorphic")
    if pre_rmsd <= 0.05:
        raise SystemExit(
            f"nonrigid positive fixture is misbuilt: pre-relaxation rmsd "
            f"{pre_rmsd:.6f} is not outside the 0.05 tolerance")
    result = relax(start, arm_force_fn, procedure)
    final = _final_record("analytic/nonrigid/positive/final",
                          result["final_positions"])
    outcome = classify_outcome(
        final, minimum, records, policy,
        budget_exhausted=result["budget_exhausted"],
        converged=result["converged"],
        forces_finite=result["forces_finite"],
        seen_minimum=True,
    )
    # The fixture's own claim, enforced where it is produced: the relaxed
    # endpoint is matched and inside the tolerance.
    if outcome["category"] != "matched_reference" or outcome["rmsd_A"] > 0.05:
        raise SystemExit(
            f"nonrigid positive fixture did not recover: category "
            f"{outcome['category']!r}, post-relaxation rmsd "
            f"{outcome['rmsd_A']:.6f}")
    print(f"nonrigid-positive: pre-relaxation rmsd_A {pre_rmsd:.6f} is "
          f"OUTSIDE the 0.05 angstrom match tolerance")
    print(f"nonrigid-positive: after relaxation the outcome is "
          f"{outcome['category']} with rmsd_A {float(outcome['rmsd_A']):.6f}")
    audit = _plain_json_field({
        "reference": minimum,
        "start": {"branch_key": "analytic/nonrigid/positive/start",
                  "kind": "nonrigid-per-atom-displacement",
                  "positions": start},
        "arm": _arm_identity(arm, "analytic-fixture-harmonic"),
        "procedure": _procedure_identity(procedure, ()),
        "final_positions": result["final_positions"],
        "rmsd_A": outcome["rmsd_A"],
        "permutation": outcome["permutation"],
        "rotation": outcome["rotation"],
        "force_evaluation_count": result["force_evaluation_count"],
        "step_count": result["step_count"],
    })
    return {
        "pre_relaxation_rmsd_A": pre_rmsd,
        "tolerance_A": 0.05,
        "outcome": {
            "category": outcome["category"],
            "rmsd_A": float(outcome["rmsd_A"]),
        },
        "audit_record": audit,
    }


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
    # The nonrigid positive fixture (review round 2): one branch that
    # starts outside the 0.05 angstrom match tolerance and becomes
    # matched after relaxation under the configured procedure. Its
    # measured pre/post evidence is recorded in the diagnostics payload.
    nonrigid = run_nonrigid_positive_fixture(policy)
    report, diagnostics = build_report(evaluated)
    diagnostics = {"per_branch": diagnostics["per_branch"],
                   "nonrigid_positive_fixture": nonrigid,
                   "timing_note": (
                       "Per-branch timing is recorded on the evaluation path, "
                       "where the runner measures each branch. This fixture "
                       "artifact is required to regenerate byte for byte, so "
                       "its rows carry timing_seconds as null rather than a "
                       "wall-clock value that would differ between runs.")}
    out = render(report, diagnostics, REPO / "reports",
                 stem="shape-recovery-fixture-diagnostics")
    print(f"fixture diagnostics rendered: {out['json']} and {out['md']}")
    print("category totals:", json.dumps(report["category_totals"], sort_keys=True))

    hash_path = freeze_real_start_set(policy)
    print("done:", out["json"], out["md"], hash_path)


if __name__ == "__main__":
    main()