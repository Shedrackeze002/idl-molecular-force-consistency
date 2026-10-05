"""Tests for the arm adapter and the shape-recovery reporter (Task 3).

No trained checkpoint exists and none is reachable. The adapter is built
against an analytic fixture arm and a callable arm, both fully testable.
The checkpoint path must report an explicit unavailable status and never
invent a weights format.
"""
import json
import copy
import numpy as np
import pytest

import analytic_fixtures as fixtures
from conftest import REPO


# ---------------------------------------------------------------------------
# Arm adapter: analytic fixture arm
# ---------------------------------------------------------------------------

def test_analytic_arm_reports_full_contract():
    from evaluation.adapter import analytic_fixture_arm

    arm = analytic_fixture_arm()
    assert arm["kind"] == "analytic_fixture"
    assert arm["status"] == "ready"
    assert arm["provides_energy"] is True
    assert arm["arm_id"].startswith("analytic-fixture-")
    probe = arm["probe"]()
    assert probe["shape"] == (1, 8, 3)
    assert probe["units"] == {"positions": "angstrom", "forces": "eV/angstrom", "energy": "eV"}
    assert probe["force_order"] == "forces on the returned positions"
    assert probe["max_abs_force_component_eV_per_A"] >= 0.0


def test_analytic_arm_forces_are_exact_at_known_positions():
    from evaluation.adapter import analytic_fixture_arm

    arm = analytic_fixture_arm()
    positions = np.array(fixtures.FIXTURE_MINIMUM, dtype=np.float64)
    forces, info = arm["forces"](positions, species=fixtures.FIXTURE_SPECIES, record_id="probe/minimum")
    assert info["status"] == "ok"
    np.testing.assert_allclose(forces, 0.0, atol=1e-14)
    energy, info = arm["energy"](positions, species=fixtures.FIXTURE_SPECIES, record_id="probe/minimum")
    assert info["status"] == "ok"
    assert energy == pytest.approx(0.0, abs=1e-14)


def test_analytic_arm_preserves_atom_order_and_species_encoding():
    from evaluation.adapter import analytic_fixture_arm

    arm = analytic_fixture_arm()
    rng_positions = np.array(fixtures.FIXTURE_MINIMUM, dtype=np.float64) + 0.1
    forces, info = arm["forces"](rng_positions, species=fixtures.FIXTURE_SPECIES, record_id="probe/order")
    assert info["status"] == "ok"
    assert info["record_id"] == "probe/order"
    assert info["species"] == fixtures.FIXTURE_SPECIES
    # The arm never reorders atoms: row i of the output belongs to row i of
    # the input, and the species encoding is echoed back unchanged.
    assert info["atom_order_preserved"] is True
    expected = fixtures.FIXTURE_WELL_STRENGTH * (-0.1) * np.ones((8, 3))
    np.testing.assert_allclose(forces, expected, atol=1e-12)


# ---------------------------------------------------------------------------
# Arm adapter: callable arm wrapping an arbitrary force function
# ---------------------------------------------------------------------------

def test_callable_arm_wraps_any_force_function():
    from evaluation.adapter import callable_arm

    arm = callable_arm(lambda p: fixtures.harmonic_force(p), species=("H",), arm_id="callable-harmonic")
    assert arm["kind"] == "callable"
    assert arm["provides_energy"] is False
    # The wrapper rejects shape mismatches instead of broadcasting silently.
    with pytest.raises(ValueError):
        arm["forces"](np.zeros((2, 3)), species=("H", "H"), record_id="probe/bad")


def test_callable_arm_shape_mismatch_is_rejected():
    from evaluation.adapter import callable_arm

    arm = callable_arm(lambda p: np.zeros_like(p), species=("H", "H", "O"), arm_id="callable-zero")
    with pytest.raises(ValueError):
        arm["forces"](np.zeros((1, 4, 3)), species=("H", "H", "O"), record_id="probe/mismatch")
    forces, info = arm["forces"](np.zeros((1, 3, 3)), species=("H", "H", "O"), record_id="probe/ok")
    assert info["status"] == "ok"
    np.testing.assert_allclose(forces, 0.0)


# ---------------------------------------------------------------------------
# Checkpoint adapter: explicit unavailability, never a guessed format
# ---------------------------------------------------------------------------

def test_checkpoint_arm_returns_explicit_unavailable(tmp_path):
    from evaluation.adapter import arm_from_checkpoint

    arm = arm_from_checkpoint(tmp_path / "missing.npz", contract={"species": tuple(fixtures.FIXTURE_SPECIES)})
    assert arm["kind"] == "checkpoint"
    assert arm["status"] == "unavailable"
    # The refusal is recorded, not silent, and no weights format is guessed.
    assert arm["refusal_reason"]
    assert "not reachable" in arm["refusal_reason"] or "not found" in arm["refusal_reason"]
    assert arm["provides_energy"] is False
    with pytest.raises(LookupError):
        arm["forces"](np.zeros((1, 8, 3)), species=fixtures.FIXTURE_SPECIES, record_id="probe/x")


def test_checkpoint_arm_refuses_when_path_is_present_but_unusable(tmp_path):
    from evaluation.adapter import arm_from_checkpoint

    fake = tmp_path / "checkpoint.npz"
    np.savez(fake, nota_weights_array=np.arange(3))
    arm = arm_from_checkpoint(fake, contract={"species": tuple(fixtures.FIXTURE_SPECIES)})
    assert arm["status"] == "unavailable"
    assert arm["refusal_reason"]
    with pytest.raises(LookupError):
        arm["forces"](np.zeros((1, 8, 3)), species=fixtures.FIXTURE_SPECIES, record_id="probe/x")


# ---------------------------------------------------------------------------
# Energy availability discipline
# ---------------------------------------------------------------------------

def test_direct_force_arm_accepted_for_shape_recovery_refused_for_energy_metrics():
    from evaluation.adapter import callable_arm, accepted_for_shape_recovery, accepted_for_energy_metrics

    arm = callable_arm(lambda p: np.zeros_like(p), species=("H",), arm_id="direct-force")
    assert accepted_for_shape_recovery(arm) is True
    allowed, refusal = accepted_for_energy_metrics(arm)
    assert allowed is False
    # The refusal is recorded with a reason, and the reason is honest about
    # what the arm cannot provide.
    assert refusal and "energy" in refusal


def test_refusal_is_a_recorded_field_not_an_exception_only():
    from evaluation.adapter import callable_arm, accepted_for_energy_metrics

    arm = callable_arm(lambda p: np.zeros_like(p), species=("H",), arm_id="direct-force-2")
    allowed, refusal = accepted_for_energy_metrics(arm)
    assert allowed is False and isinstance(refusal, str) and refusal


def test_nonfinite_forces_are_reported_unavailable_never_zeroed():
    from evaluation.adapter import callable_arm
    from evaluation.relax import relax

    def nan_force(positions):
        out = np.zeros_like(positions)
        out[0, 0, 0] = np.nan
        return out

    arm = callable_arm(nan_force, species=("H", "H"), arm_id="nan-arm")
    with pytest.raises(FloatingPointError):
        arm["forces"](np.zeros((1, 2, 3)), species=("H", "H"), record_id="probe/nan")


def test_relax_through_the_adapter_records_the_nonfinite_refusal():
    # The integration path from the review: the adapter raises
    # NonfiniteValueError (a FloatingPointError) for a NaN force, and
    # relax routes that declared refusal into a recorded branch failure
    # instead of aborting the whole run. The force attempt is counted
    # BEFORE the supplier runs, so the refusal is accounted for exactly
    # like a returned value would be.
    from evaluation.adapter import callable_arm
    from evaluation.relax import relax

    def nan_force(positions):
        out = np.zeros(np.asarray(positions).shape, dtype=np.float64)
        out[..., 0] = np.nan
        return out

    arm = callable_arm(nan_force, species=("H", "H"), arm_id="nan-arm")
    result = relax(
        np.array([[1.0, 0.0, 0.0], [0.5, 0.2, 0.0]], dtype=np.float64),
        lambda p: arm["forces"](p, species=("H", "H"),
                                record_id="nan-integration")[0],
        {"max_force_evaluations": 10, "initial_step_size": 0.2,
         "backtracking_factor": 0.5, "min_backtracking_floor": 1e-8,
         "convergence_tolerance_eV_per_A": 1e-2,
         "max_backtracking_attempts_per_step": 40},
    )
    assert result["status"] == "nonconverged"
    assert result["forces_finite"] is False
    assert result["failure_reason"] == "nonfinite_force"
    assert result["budget_exhausted"] is False
    assert result["converged"] is False
    assert result["step_count"] == 0
    # Exactly one attempt, spent even though the supplier raised.
    assert result["force_evaluation_count"] == 1
    # The endpoint stays finite for the audit trail.
    assert np.isfinite(np.asarray(result["final_positions"])).all()


def test_adapter_round_trip_positions_species_order(tmp_path):
    from evaluation.adapter import analytic_fixture_arm
    from evaluation.relax import relax

    arm = analytic_fixture_arm()
    # Round trip: start from a known displaced structure, relax with the
    # adapter's own force callable, and confirm the identity, species and
    # atom order survive all the way into the returned record.
    start = np.array(fixtures.FIXTURE_MINIMUM, dtype=np.float64) + 0.5
    start.setflags(write=False)
    procedure = {
        "max_force_evaluations": 300, "initial_step_size": 0.2,
        "backtracking_factor": 0.5, "min_backtracking_floor": 1e-8,
        "convergence_tolerance_eV_per_A": 1e-2, "max_backtracking_attempts_per_step": 40,
    }
    result = relax(start, lambda p: arm["forces"](p, species=fixtures.FIXTURE_SPECIES, record_id="rt")[0],
                   procedure)
    assert result["status"] == "converged"
    np.testing.assert_allclose(result["final_positions"], fixtures.FIXTURE_MINIMUM, atol=1e-8)


# ---------------------------------------------------------------------------
# Reporter: build_report and render
# ---------------------------------------------------------------------------

def _outcome(branch, category, seen_minimum, **extra):
    from evaluation.outcomes import CATEGORY_NAMES

    assert category in CATEGORY_NAMES
    connection_id = branch.rsplit("/", 1)[0].rsplit("/", 1)[-1]
    base = {
        "branch_key": branch, "category": category, "seen_minimum": seen_minimum,
        "partition": "validation" if connection_id in ("01", "10") else "test",
        "budget_exhausted": False,
    }
    # Only nonconverged is compatible with converged=False; every other
    # category is a verdict on a reached endpoint (review round 2).
    base["converged"] = category != "nonconverged"
    base.update(extra)
    return base


def _sample_evaluated():
    return {
        "aspirin/01/plus": _outcome("aspirin/01/plus", "matched_reference", True),
        "aspirin/01/minus": _outcome("aspirin/01/minus", "nonconverged", True),
        "aspirin/10/plus": _outcome("aspirin/10/plus", "ambiguous", False),
        "aspirin/10/minus": _outcome("aspirin/10/minus", "unresolved", False),
    }


def test_reporter_counts_categories_and_denominators_sum(tmp_path):
    from evaluation.reporter import build_report

    evaluated = _sample_evaluated()
    report, diagnostics = build_report(evaluated)
    assert report["branch_count"] == 4
    # Per-connection denominators sum to the evaluated branch count.
    total = sum(c["denominator"] for c in report["per_connection"])
    assert total == 4
    conn01 = next(c for c in report["per_connection"] if c["connection_id"] == "01")
    assert conn01["denominator"] == 2
    assert conn01["counts"]["matched_reference"] == 1
    assert conn01["counts"]["nonconverged"] == 1
    assert conn01["denominators"]["shape_recovery_total"] == 2
    assert conn01["rates"]["matched_reference_rate"] == pytest.approx(0.5)


def test_reporter_rejects_forged_or_missing_outcomes(tmp_path):
    from evaluation.reporter import build_report

    evaluated = _sample_evaluated()
    forged = copy.deepcopy(evaluated)
    del forged["aspirin/10/minus"]
    with pytest.raises(ValueError):
        build_report(forged)

    forged2 = copy.deepcopy(evaluated)
    forged2["aspirin/01/plus"]["category"] = "made_up_category"
    with pytest.raises(ValueError):
        build_report(forged2)

    forged3 = copy.deepcopy(evaluated)
    # seen_minimum silently dropped must be rejected, not defaulted.
    del forged3["aspirin/01/plus"]["seen_minimum"]
    with pytest.raises(ValueError):
        build_report(forged3)


def test_reporter_preserves_seen_minimum_per_branch():
    from evaluation.reporter import build_report

    evaluated = _sample_evaluated()
    report, diagnostics = build_report(evaluated)
    per_branch = {(b["molecule"], b["connection_id"], b["branch"]): b for b in diagnostics["per_branch"]}
    assert per_branch[("aspirin", "01", "plus")]["seen_minimum"] is True
    assert per_branch[("aspirin", "10", "plus")]["seen_minimum"] is False
    assert per_branch[("aspirin", "10", "minus")]["seen_minimum"] is False


def test_reporter_train_partition_branch_is_excluded_from_held_out_pools():
    from evaluation.reporter import build_report

    # An outcome whose connection is not held out cannot enter a held-out
    # pool; it stays visible as a counted branch but leaves the held-out
    # denominators. The partition is a connection-level fact (review
    # round 2), so the exclusion applies to the whole pair, not one arm.
    evaluated = _sample_evaluated()
    evaluated["aspirin/10/plus"]["partition"] = "train"
    evaluated["aspirin/10/minus"]["partition"] = "train"
    report, diagnostics = build_report(evaluated)
    conn10 = next(c for c in report["per_connection"] if c["connection_id"] == "10")
    # Both branches excluded: the connection leaves the held-out pools
    # entirely, dropping to the honest unavailable state.
    assert conn10["denominator"] == 0
    assert conn10["availability"] == "unavailable"
    assert conn10["rates"]["matched_reference_rate"] is None
    # It is still counted as evaluated branches, just not held out.
    assert report["branch_count"] == 4


def test_reporter_zero_denominator_connection_reported_unavailable():
    from evaluation.reporter import build_report

    # A connection appearing only with a nonfinite/unresolved or empty set:
    # build a partition where one connection has ALL branches unresolved.
    # It still has denominator 2. A true zero denominator arises when a
    # connection key exists in the config but no outcome is provided. The
    # reporter must list it as unavailable, not as zero percent.
    from evaluation.reporter import build_report

    report, diagnostics = build_report({}, expected_connections=["aspirin/44/plus"])
    conn = next(c for c in report["per_connection"] if c["connection_id"] == "44")
    assert conn["denominator"] == 0
    assert conn["availability"] == "unavailable"
    assert conn["rates"]["matched_reference_rate"] is None


def test_render_writes_json_and_md_pair(tmp_path):
    from evaluation.reporter import build_report, render

    evaluated = _sample_evaluated()
    report, diagnostics = build_report(evaluated, expected_connections=["aspirin/01/plus", "aspirin/01/minus",
                                                                       "aspirin/10/plus", "aspirin/10/minus"])
    written = render(report, diagnostics, tmp_path, stem="shape-recovery-fixture-diagnostics")
    json_path = tmp_path / "shape-recovery-fixture-diagnostics.json"
    md_path = tmp_path / "shape-recovery-fixture-diagnostics.md"
    assert json_path.exists() and md_path.exists()
    loaded = json.loads(json_path.read_text())
    assert loaded["report"]["branch_count"] == 4
    # The md is a readable rendering of the same data, not prose fiction.
    text = md_path.read_text()
    assert "matched_reference" in text
    assert str(report["branch_count"]) in text
    assert written == {"json": str(json_path), "md": str(md_path)}


def test_render_includes_seven_category_disclaimer(tmp_path):
    from evaluation.reporter import build_report, render

    evaluated = _sample_evaluated()
    report, diagnostics = build_report(evaluated)
    render(report, diagnostics, tmp_path, stem="probe")
    text = (tmp_path / "probe.md").read_text()
    assert "not a minimum" in text
    assert "fixture" in text.lower()
# ---------------------------------------------------------------------------
# Review round 2: consistency validation, true convergence count, audit
# record, partition labels (Task B: B1, B2, B3, B4)
# ---------------------------------------------------------------------------

def test_reporter_rejects_matched_reference_that_did_not_converge():
    # B1 acceptance: exactly the contradictory pair from the review, a
    # branch labelled matched_reference with converged=False and
    # budget_exhausted=True, must be REJECTED instead of reporting a
    # matched-reference rate of 1.0.
    from evaluation.reporter import build_report

    evaluated = _sample_evaluated()
    forged = copy.deepcopy(evaluated)
    forged["aspirin/01/plus"]["category"] = "matched_reference"
    forged["aspirin/01/plus"]["converged"] = False
    forged["aspirin/01/plus"]["budget_exhausted"] = True
    with pytest.raises(ValueError) as excinfo:
        build_report(forged)
    message = str(excinfo.value)
    assert "aspirin/01/plus" in message
    assert "contradicts" in message


def test_reporter_legitimate_category_fact_combinations_pass():
    # B1 acceptance, the constructive half: every combination the
    # taxonomy actually produces still passes validation, including the
    # ones the old presence-only check let through.
    from evaluation.reporter import build_report

    evaluated = _sample_evaluated()
    for key, outcome in evaluated.items():
        outcome["converged"] = outcome["category"] in (
            "matched_reference", "unmatched_converged", "ambiguous", "unresolved")
        outcome["budget_exhausted"] = outcome["category"] == "nonconverged"
    report, diagnostics = build_report(evaluated)
    assert report["branch_count"] == 4

    # nonconverged on budget without convergence stays legitimate, and
    # nonconverged WITHOUT budget exhaustion (no_progress, a refused
    # force) stays legitimate too: the only fact nonconverged requires
    # is converged=False.
    evaluated["aspirin/01/minus"].update(
        category="nonconverged", converged=False, budget_exhausted=False)
    report, diagnostics = build_report(evaluated)
    conn = next(c for c in report["per_connection"]
                if c["connection_id"] == "01")
    assert conn["counts"]["nonconverged"] == 1

    # unresolved stays legitimate on a reached, under-budget endpoint:
    # the comparison failed, not the relaxation, and the facts must say
    # the relaxation reached its endpoint (converged=True,
    # budget_exhausted=False).
    evaluated["aspirin/10/minus"].update(
        category="unresolved", converged=True, budget_exhausted=False)
    report, diagnostics = build_report(evaluated)


def test_reporter_rejects_non_boolean_convergence_facts():
    # Facts must be actual booleans, not anything truthy: a string or a
    # number would pass the old presence-only validation.
    from evaluation.reporter import build_report

    evaluated = _sample_evaluated()
    forged = copy.deepcopy(evaluated)
    forged["aspirin/01/plus"]["converged"] = "true"
    with pytest.raises(ValueError) as excinfo:
        build_report(forged)
    assert "converged must be a boolean" in str(excinfo.value)

    forged2 = copy.deepcopy(evaluated)
    forged2["aspirin/01/plus"]["budget_exhausted"] = 1
    with pytest.raises(ValueError) as excinfo:
        build_report(forged2)
    assert "budget_exhausted must be a boolean" in str(excinfo.value)

    forged3 = copy.deepcopy(evaluated)
    forged3["aspirin/01/plus"]["partition"] = "holdout"
    with pytest.raises(ValueError) as excinfo:
        build_report(forged3)
    assert "unknown partition" in str(excinfo.value)


def test_reporter_nonconverged_converged_true_is_rejected():
    # The mirror contradiction: a converged branch reported as
    # nonconverged erases a real recovery, so it is refused too.
    from evaluation.reporter import build_report

    evaluated = _sample_evaluated()
    forged = copy.deepcopy(evaluated)
    forged["aspirin/01/plus"]["category"] = "nonconverged"
    forged["aspirin/01/plus"]["converged"] = True
    forged["aspirin/01/plus"]["budget_exhausted"] = False
    with pytest.raises(ValueError) as excinfo:
        build_report(forged)
    message = str(excinfo.value)
    assert "aspirin/01/plus" in message and "nonconverged contradicts" in message


def test_reporter_counts_true_convergence_not_the_denominator():
    # B2 acceptance: the converged column is the counted branches that
    # met the force rule. A connection carrying one converged branch
    # (plus, unmatched_converged) and one nonconverged branch (minus):
    # the count must be 1 while the rate denominator stays 2.
    from evaluation.reporter import build_report

    evaluated = _sample_evaluated()
    report, _diagnostics = build_report(evaluated)
    conn01 = next(c for c in report["per_connection"]
                  if c["connection_id"] == "01")
    assert conn01["converged_count"] == 1
    assert conn01["denominator"] == 2
    assert conn01["denominators"]["shape_recovery_total"] == 2

    conn10 = next(c for c in report["per_connection"]
                  if c["connection_id"] == "10")
    # one ambiguous (converged) plus one unresolved (converged, the
    # endpoint was reached): both meet the force rule.
    assert conn10["converged_count"] == 2 and conn10["denominator"] == 2


def test_reporter_audit_record_round_trips_json_and_md(tmp_path):
    # B3 acceptance: the caller's endpoint/matching and relaxation
    # evidence survives into the JSON (json.loads round trip, no loss,
    # including the positions as a plain float list) and into the
    # Markdown as a per-branch audit row.
    from evaluation.reporter import build_report, render

    reference_positions = [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.3, 0.0]]

    class _Reference:
        record_id = "m0"
        molecule = "aspirin"
        kind = "minimum"
        source_path = "m0.xyz"
        connection_id = None
        branch = None
        frame_index = None
        species = ("C", "H", "O")
        positions = reference_positions

    evaluated = {}
    for key, category, record in (
            ("aspirin/01/plus", "matched_reference", _Reference()),
            ("aspirin/01/minus", "nonconverged", _Reference())):
        outcome = _sample_evaluated()[key]
        outcome["category"] = category
        outcome["_audit_record"] = {
            "reference": record,
            "start": {"record_id": "start-frame-0", "frame_index": 0},
            "arm": {"kind": "analytic_fixture",
                    "arm_id": "analytic-fixture-8at-harmonic"},
            "final_positions": _Reference.positions,
            "rmsd_A": 0.004,
            "permutation": [0, 1, 2],
            "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "failure": {"status": "converged", "failure_reason": None},
            "force_evaluation_count": 17,
            "step_count": 3,
        }
        evaluated[key] = outcome

    report, diagnostics = build_report(evaluated)
    assert report["branch_count"] == 2
    plus = diagnostics["per_branch"][0]
    assert plus["reference"]["record_id"] == "m0"
    assert plus["reference"]["species"] == ["C", "H", "O"]
    assert plus["arm"]["arm_id"] == "analytic-fixture-8at-harmonic"
    assert plus["final_positions"] == reference_positions
    assert plus["rmsd_A"] == pytest.approx(0.004)
    assert plus["permutation"] == [0, 1, 2]
    assert plus["failure"]["failure_reason"] is None
    assert plus["force_evaluation_count"] == 17 and plus["step_count"] == 3

    render(report, diagnostics, tmp_path, stem="audit-probe")
    loaded = json.loads((tmp_path / "audit-probe.json").read_text())
    branch_rows = loaded["diagnostics"]["per_branch"]
    by_key = {row["branch_key"]: row for row in branch_rows}
    audited = by_key["aspirin/01/plus"]
    # The round trip keeps every retained field intact in the file.
    assert audited["reference"]["record_id"] == "m0"
    assert audited["reference"]["positions"] == reference_positions
    assert audited["final_positions"] == reference_positions
    assert audited["arm"]["arm_id"] == "analytic-fixture-8at-harmonic"
    assert audited["force_evaluation_count"] == 17
    assert audited["step_count"] == 3

    text = (tmp_path / "audit-probe.md").read_text()
    for fragment in ("branch_key", "reference", "analytic-fixture-8at-harmonic",
                     "start-frame-0", '"failure_reason": null',
                     "force_evaluation_count", "step_count"):
        assert fragment in text, fragment


def test_reporter_audit_record_fields_may_be_omitted_and_never_stringified(tmp_path):
    # Partial audit records keep only what the caller supplied; an
    # audit value with no understood structure is refused (never
    # str()ed into an opaque string column).
    from evaluation.reporter import build_report, render

    evaluated = _sample_evaluated()
    evaluated["aspirin/01/plus"]["_audit_record"] = {
        "rmsd_A": 0.01, "arm": {"kind": "callable", "arm_id": "bowl"},
    }
    report, diagnostics = build_report(evaluated)
    plus = next(row for row in diagnostics["per_branch"]
                if row["branch_key"] == "aspirin/01/plus")
    assert plus["rmsd_A"] == pytest.approx(0.01)
    assert "reference" not in plus and "final_positions" not in plus
    # Rendering a partially audited report still works; the absent
    # fields show as blank cells, not invented facts.
    written = render(report, diagnostics, tmp_path, stem="audit-partial")
    assert (tmp_path / "audit-partial.md").exists()

    class _Opaque:
        def __repr__(self):
            return "<opaque object>"

    evaluated["aspirin/01/minus"]["_audit_record"] = {"arm": _Opaque()}
    with pytest.raises(ValueError) as excinfo:
        build_report(evaluated)
    assert "not plain-JSON representable" in str(excinfo.value)
    assert "arm" in str(excinfo.value)


def test_reporter_rejects_split_partitions_within_one_connection():
    # B4 acceptance: opposite branches of one connection carrying
    # different partition labels is misdescription and is refused,
    # naming both branch keys.
    from evaluation.reporter import build_report

    evaluated = _sample_evaluated()
    evaluated["aspirin/01/minus"]["partition"] = "test"
    with pytest.raises(ValueError) as excinfo:
        build_report(evaluated)
    message = str(excinfo.value)
    assert "different partition labels" in message
    assert "aspirin/01/minus" in message and "aspirin/01/plus" in message


def test_reporter_labels_connections_and_splits_partition_totals():
    # B4 acceptance: each connection row carries its partition, and the
    # totals no longer pool validation and test: two validation
    # recoveries and two test nonconvergences must read as separate
    # per-partition counts, not as merged totals.
    from evaluation.reporter import build_report

    evaluated = {
        "aspirin/01/plus": _outcome("aspirin/01/plus", "matched_reference", True),
        "aspirin/01/minus": _outcome("aspirin/01/minus", "matched_reference", True),
        "aspirin/00/plus": _outcome("aspirin/00/plus", "nonconverged", False),
        "aspirin/00/minus": _outcome("aspirin/00/minus", "nonconverged", False),
    }
    report, diagnostics = build_report(evaluated)
    connections = {c["connection_id"]: c for c in report["per_connection"]}
    assert connections["01"]["partition"] == "validation"
    assert connections["00"]["partition"] == "test"

    totals = report["category_totals"]
    assert totals["validation"]["matched_reference"] == 2
    assert totals["test"]["nonconverged"] == 2
    assert totals["held_out"]["matched_reference"] == 2
    assert totals["held_out"]["nonconverged"] == 2
    assert report["held_out_branch_counts"] == {"validation": 2, "test": 2}

    # An explicitly train-partition connection stays visible but stays
    # out of every held-out pool and total.
    train_evaluated = {
        "aspirin/01/plus": _outcome("aspirin/01/plus", "matched_reference", True),
        "aspirin/01/minus": _outcome("aspirin/01/minus", "nonconverged", True),
    }
    for key, value in train_evaluated.items():
        value["partition"] = "train"
    report, diagnostics = build_report(train_evaluated)
    assert report["held_out_branch_count"] == 0
    assert report["held_out_branch_counts"] == {"validation": 0, "test": 0}
    assert all(count == 0 for count in
               report["category_totals"]["validation"].values())
    assert all(count == 0 for count in
               report["category_totals"]["test"].values())
