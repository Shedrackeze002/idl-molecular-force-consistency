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
    # No audit record reached this report, so the provenance says the
    # arm is not recorded instead of claiming an arm it has no evidence
    # for (review item 2: the rendered provenance must not name an
    # analytic fixture arm when another arm ran).
    assert "not recorded" in text.lower()
    assert "analytic fixture" not in text.lower()
    assert "fixture arm" not in text.lower()
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


def test_reporter_rejects_failure_audit_contradicting_the_summary():
    # Review item 5 acceptance: a branch whose summary contradicts its
    # own force-failure facts is refused on all three surfaces the
    # review names. Surface 1 is the review's own reproduction: a
    # matched-reference summary kept unchanged while the nested
    # _audit_record failure section is switched to failed=True,
    # failure_reason nonfinite_force, forces_finite False, converged
    # False, exactly what the reporter accepted before this guard.
    from evaluation.reporter import build_report

    evaluated = _sample_evaluated()
    forged = copy.deepcopy(evaluated)
    forged["aspirin/01/plus"]["_audit_record"] = {
        "failure": {"failed": True, "failure_reason": "nonfinite_force",
                    "budget_exhausted": False, "converged": False,
                    "forces_finite": False},
    }
    with pytest.raises(ValueError) as excinfo:
        build_report(forged)
    message = str(excinfo.value)
    assert "aspirin/01/plus" in message
    assert "contradictory failure audit" in message
    assert "failed=True" in message

    # Surface 2: force-finiteness versus the stop facts the branch
    # outcome itself carries (the runner copies relaxation /
    # classify_outcome facts onto every branch). A converged shape
    # verdict with forces_finite=False is the nonfinite-force
    # contradiction; a stop reason on a shape verdict claims the run
    # stopped before the rule was met, contradicting the verdict scored
    # on a reached endpoint.
    forged2 = copy.deepcopy(evaluated)
    forged2["aspirin/01/plus"]["forces_finite"] = False
    with pytest.raises(ValueError) as excinfo:
        build_report(forged2)
    message = str(excinfo.value)
    assert "aspirin/01/plus" in message
    assert "forces_finite" in message

    forged3 = copy.deepcopy(evaluated)
    forged3["aspirin/01/minus"].update(
        category="nonconverged", converged=False, budget_exhausted=False,
        stop_reason="no_progress")
    report, diagnostics = build_report(forged3)
    assert report["branch_count"] == 4

    forged4 = copy.deepcopy(evaluated)
    forged4["aspirin/10/plus"]["stop_reason"] = "no_progress"
    with pytest.raises(ValueError) as excinfo:
        build_report(forged4)
    message = str(excinfo.value)
    assert "aspirin/10/plus" in message
    assert "stop_reason" in message

    # Surface 3: the nested audit section against itself, isolated from
    # surface 2 by sitting on a branch whose summary the section would
    # otherwise agree with (nonconverged, budget not exhausted): a
    # section recording failed=True with no failure reason (or
    # failed=False carrying a declared failure reason) is self
    # contradictory no matter what the summary says.
    forged5 = copy.deepcopy(evaluated)
    forged5["aspirin/01/minus"]["_audit_record"] = {
        "failure": {"failed": True, "failure_reason": None,
                    "budget_exhausted": False, "converged": False,
                    "forces_finite": False},
    }
    with pytest.raises(ValueError) as excinfo:
        build_report(forged5)
    message = str(excinfo.value)
    assert "aspirin/01/minus" in message
    assert "failure_reason=None" in message

    forged6 = copy.deepcopy(evaluated)
    forged6["aspirin/01/plus"]["_audit_record"] = {
        "failure": {"failed": False, "failure_reason": "nonfinite_force"},
    }
    with pytest.raises(ValueError) as excinfo:
        build_report(forged6)
    message = str(excinfo.value)
    assert "aspirin/01/plus" in message
    assert "failed=False" in message
    assert "nonfinite_force" in message

    # The honest fixture slice stays untouched: the reviewer's probe
    # builds its contradiction from the unmodified fixture row, so the
    # guard must leave that row valid and reject only the doctored one.
    honest = _sample_evaluated()
    honest["aspirin/01/plus"]["_audit_record"] = {
        "failure": {"failed": False, "failure_reason": None,
                    "budget_exhausted": False, "converged": True,
                    "forces_finite": True},
    }
    report, diagnostics = build_report(honest)
    plus = next(row for row in diagnostics["per_branch"]
                if row["branch_key"] == "aspirin/01/plus")
    assert plus["failure"]["failed"] is False
    # The unmodified sample counts stay what they always were.
    assert report["category_totals"]["held_out"]["matched_reference"] == 1
    assert report["branch_count"] == 4


# ---------------------------------------------------------------------------
# Review round 3, gap 1 (card B): reconcile the failure vocabulary the
# RUNNER emits (stop_reason / failure_reason / nonfinite_force, and no
# "failed" key) against the summary, before any counting. The
# regressions run on evaluate_starts output, as the reviewer asked.
# ---------------------------------------------------------------------------

def test_runner_nested_failure_reason_on_matched_summary_is_refused():
    # Witness a: a matched runner row whose ONLY change is the nested
    # failure_reason reading 'nonfinite_force'. The runner's audit
    # section has no "failed" key, so the reason itself must be
    # reconciled with the summary: a declared stop reason on a branch
    # counted as a reached endpoint is refused before anything is
    # counted.
    from evaluation.reporter import build_report

    branches = _recovery_runner_result()["branches"]
    forged = copy.deepcopy(branches)
    target = next(key for key, value in forged.items()
                  if value["category"] == "matched_reference")
    forged[target]["_audit_record"]["failure"]["failure_reason"] = \
        "nonfinite_force"
    with pytest.raises(ValueError) as excinfo:
        build_report(forged)
    message = str(excinfo.value)
    assert target in message
    assert "contradictory failure audit" in message
    assert "nonfinite_force" in message
    assert "matched_reference" in message


def test_runner_nested_stop_reason_and_nonfinite_flag_on_matched_refused():
    # Witness b: nested stop_reason='nonfinite_force' together with
    # nonfinite_force=True on a matched runner row, saved with
    # forces_finite=True unchanged. The effective stop reason
    # contradicts the reached-endpoint summary and the nonfinite flag
    # contradicts the converged summary; the record is refused as it
    # stands, not repaired.
    from evaluation.reporter import build_report

    branches = _recovery_runner_result()["branches"]
    forged = copy.deepcopy(branches)
    target = next(key for key, value in forged.items()
                  if value["category"] == "matched_reference")
    nested = forged[target]["_audit_record"]["failure"]
    nested["stop_reason"] = "nonfinite_force"
    nested["nonfinite_force"] = True
    with pytest.raises(ValueError) as excinfo:
        build_report(forged)
    message = str(excinfo.value)
    assert target in message
    assert "contradictory failure audit" in message
    assert "nonfinite_force" in message


def test_fixture_slice_failure_reason_witness_stays_refused():
    # Regression 3: the fixture-slice witness refused before this
    # round stays refused. The fixture audit carries failed=False, so
    # surface 3 (unfailed with a declared reason) refuses it exactly
    # as before; the runner-vocabulary checks must not have weakened
    # that path, and the unmutated fixture slice still builds.
    from force_consistency.data.types import PreparationPolicy

    from evaluation.reporter import build_report

    module = _load_report_script()
    evaluated = module.run_fixture_slice(PreparationPolicy())
    report, _diagnostics = build_report(evaluated)
    assert report["branch_count"] == 4

    target = next(key for key, value in evaluated.items()
                  if value["category"] == "matched_reference")
    evaluated[target]["_audit_record"]["failure"]["failure_reason"] = \
        "nonfinite_force"
    with pytest.raises(ValueError) as excinfo:
        build_report(evaluated)
    message = str(excinfo.value)
    assert target in message
    assert "failed=False" in message
    assert "nonfinite_force" in message


def test_honest_runner_slice_reports_its_categories_unchanged():
    # Regression 4: the reconciliation refuses contradictions without
    # refusing valid records. An honest evaluate_starts slice over the
    # recovery landscape passes every new check and reports the same
    # categories it always did: two matched_reference recoveries,
    # nothing else, nothing dropped.
    from evaluation.reporter import build_report

    branches = _recovery_runner_result()["branches"]
    report, diagnostics = build_report(branches)
    assert report["branch_count"] == 2
    assert {row["category"] for row in diagnostics["per_branch"]} == \
        {"matched_reference"}
    totals = report["category_totals"]
    assert totals["held_out"]["matched_reference"] == 2
    assert totals["held_out"]["nonconverged"] == 0
    assert totals["validation"]["matched_reference"] == 2
    assert totals["test"]["matched_reference"] == 0
    # The nested runner facts arrive in the report unmodified: the
    # reconciliation never normalises a valid record either.
    row = diagnostics["per_branch"][0]
    assert row["failure"]["stop_reason"] is None
    assert row["failure"]["failure_reason"] is None
    assert row["failure"]["forces_finite"] is True
    assert row["failure"]["nonfinite_force"] is False


# ---------------------------------------------------------------------------
# Review item 2 (card B1): the runner's per-branch audit record survives
# runner -> build_report -> render, and the rendered provenance names the
# arm that ACTUALLY produced the numbers.
# ---------------------------------------------------------------------------

def _recovery_runner_result(connection_id="80", arm=None):
    """One real evaluate_starts run over the recovery landscape.

    The composition is the probe's: the self-contained landscape from
    the runner test module, built explicitly here because pytest
    fixtures do not cross test-module boundaries. arm defaults to the
    bowl arm; the refused-branch regression passes its own.
    """
    import copy as _copy

    import test_shape_recovery_runner as runner_support
    from evaluation.runner import evaluate_starts

    landscape = runner_support.RecoveryLandscape()
    manifest = _copy.deepcopy(landscape.manifest)
    classification = runner_support._recovery_classification(landscape)
    references = landscape.references()
    procedure = runner_support._runner_procedure()
    if arm is None:
        arm = runner_support._bowl_arm_for(
            runner_support.recovery_minimum(0))
    full_start_set = runner_support._recovery_start_set(landscape, manifest)
    start_set = {key: entry for key, entry in full_start_set.items()
                 if key.split("/")[1] == connection_id}
    return evaluate_starts(
        start_set, arm, procedure, classification, references,
        landscape.policy(), split_manifest=manifest,
        seen_minimum=True,
    )


def _assert_audit_facts_present(row):
    """The twelve evidence facts, asserted on one saved branch row.

    Shared by the successful and the refused end-to-end regression: a
    row that survives the runner and the reporter must carry the arm
    identity, the start identity, the branch's own reference identity,
    the endpoint coordinates, the RMSD and mapping facts, the
    structured failure facts, the exact force-call and step counts,
    the per-branch wall-clock timing, the partition identity and the
    effective settings.
    """
    assert row["arm"]["kind"] == "callable"
    assert row["arm"]["arm_id"] == "recovery-bowl"
    assert row["arm"]["status"] == "ready"
    assert row["start"]["record_id"].endswith("frame_0")
    assert row["reference"]["record_id"].startswith("recovery/minimum_")
    assert row["final_positions"] and len(row["final_positions"]) == 3
    assert row["rmsd_A"] is not None
    assert row["permutation"] is not None
    assert row["rotation"] is not None
    assert "failure" in row and "stop_reason" in row["failure"]
    assert isinstance(row["force_evaluation_count"], int)
    assert row["force_evaluation_count"] > 0
    assert isinstance(row["step_count"], int)
    assert isinstance(row["timing_seconds"], float) and row["timing_seconds"] >= 0.0
    assert row["partition"] in ("train", "validation", "test")
    assert row["procedure"]["effective"]["max_force_evaluations"] == 300


def test_runner_audit_record_survives_to_saved_report(tmp_path):
    # End-to-end successful branch: evaluate_starts -> build_report ->
    # render, then read the SAVED JSON back and assert the evidence
    # chain arrived in full (review item 2).
    import json as _json

    from evaluation.reporter import build_report, render

    result = _recovery_runner_result()
    report, diagnostics = build_report(result["branches"])
    out = render(report, diagnostics, tmp_path,
                 stem="audit-chain-success")
    saved = _json.loads((tmp_path / "audit-chain-success.json").read_text())

    rows_by_key = {row["branch_key"]: row
                   for row in saved["diagnostics"]["per_branch"]}
    assert sorted(rows_by_key) == ["aspirin/80/minus", "aspirin/80/plus"]
    for _key, row in rows_by_key.items():
        assert row["category"] == "matched_reference"
        _assert_audit_facts_present(row)
    # The provenance names the arm that ACTUALLY produced the numbers.
    assert saved["report"]["provenance"]["source"] == "audit_record"
    assert (saved["report"]["provenance"]["arm"]["arm_id"]
            == "recovery-bowl")
    text = (tmp_path / "audit-chain-success.md").read_text()
    assert "recovery-bowl" in text
    assert "analytic fixture" not in text


def test_runner_refused_branch_carries_audit_facts_to_report(tmp_path):
    # End-to-end refused branch: a branch whose force supplier refuses
    # every call (wrong shape, the declared invalid_force_shape route)
    # still carries its structured failure facts, the refusing arm's
    # identity, the counts the failed attempts spent, its own timing
    # and the endpoint it stopped on, all the way into the saved
    # report. A refused run is audited, not erased.
    import json as _json

    from conftest import SPECIES
    from evaluation.adapter import callable_arm
    from evaluation.reporter import build_report, render

    def wrong_shape(positions):
        return np.zeros((1, 1, 3), dtype=np.float64)

    arm = callable_arm(wrong_shape, species=SPECIES,
                       arm_id="refusing-shape-arm")
    result = _recovery_runner_result(arm=arm)
    branches = result["branches"]
    assert branches, "the run must return branch outcomes to report"
    for row in branches.values():
        assert row["category"] == "nonconverged"
        assert row["converged"] is False
    report, diagnostics = build_report(branches)
    out = render(report, diagnostics, tmp_path,
                 stem="audit-chain-refused")
    saved = _json.loads((tmp_path / "audit-chain-refused.json").read_text())
    rows_by_key = {row["branch_key"]: row
                   for row in saved["diagnostics"]["per_branch"]}
    for _key, row in rows_by_key.items():
        assert row["arm"]["arm_id"] == "refusing-shape-arm"
        # The structured failure record: the declared reason, the
        # per-refusal trail naming WHICH arm refused, and the counts
        # the refused attempts spent.
        failure = row["failure"]
        assert failure["stop_reason"] == "invalid_force_shape"
        assert failure["refusal_details"], "the refusal trail must not be empty"
        assert failure["refusal_details"][0]["arm_id"] == "refusing-shape-arm"
        assert row["force_evaluation_count"] > 0
        # Timing and settings survive on the refused branch too.
        assert isinstance(row["timing_seconds"], float)
        assert row["procedure"]["effective"]["max_force_evaluations"] == 300
        # The refused branch still carries its own reference identity and
        # the endpoint the run stopped on (never NaN), so the seed-to-
        # endpoint evidence survives the refusal too.
        assert row["reference"]["record_id"]
        assert row["final_positions"]
    assert saved["report"]["provenance"]["arm"]["arm_id"] == "refusing-shape-arm"


def test_reporter_caller_supplied_audit_record_survives_rendering(tmp_path):
    # A caller-built record (no runner involved) survives build_report
    # and the JSON round trip field for field, including the names this
    # card's chain relies on, and the provenance names the record's arm.
    import json as _json

    from evaluation.reporter import build_report, render

    record = {
        "arm": {"kind": "callable", "arm_id": "caller-arm",
                "status": "ready", "provides_energy": False},
        "start": {"record_id": "caller/start/frame_0", "frame_index": 3},
        "reference": {"record_id": "recovery/minimum_0"},
        "final_positions": [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 1.3, 0.0]],
        "rmsd_A": 0.012,
        "permutation": [0, 1, 2],
        "rotation": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "failure": {"stop_reason": None, "failure_reason": None,
                    "refusal_details": []},
        "force_evaluation_count": 9,
        "step_count": 2,
        "timing_seconds": 0.25,
        "procedure": {"effective": {"max_force_evaluations": 300}},
    }
    evaluated = {
        "aspirin/01/plus": _outcome("aspirin/01/plus", "matched_reference", True),
        "aspirin/01/minus": _outcome("aspirin/01/minus", "nonconverged", True),
    }
    evaluated["aspirin/01/plus"]["_audit_record"] = dict(record)
    report, diagnostics = build_report(evaluated)
    plus = next(row for row in diagnostics["per_branch"]
                if row["branch_key"] == "aspirin/01/plus")
    for name, expected in record.items():
        assert plus[name] == expected, name
    render(report, diagnostics, tmp_path, stem="caller-record")
    saved = _json.loads((tmp_path / "caller-record.json").read_text())
    row = next(r for r in saved["diagnostics"]["per_branch"]
               if r["branch_key"] == "aspirin/01/plus")
    for name, expected in record.items():
        assert row[name] == expected, name
    assert saved["report"]["provenance"]["arm"]["arm_id"] == "caller-arm"
    text = (tmp_path / "caller-record.md").read_text()
    assert "caller-arm" in text


# ---------------------------------------------------------------------------
# Review round 2 card C1: the fixture diagnostics audit must record the
# ACTUAL force supplier and the EFFECTIVE procedure per branch (with the
# synthetic 2-call override labelled against the configured 1,500), and a
# nonrigid positive fixture must start outside the 0.05 angstrom match
# tolerance and become matched after relaxation.
# ---------------------------------------------------------------------------

def _load_report_script():
    # The reporter script lives under scripts/ and is only imported
    # from the command line; loading it by file path keeps the frozen
    # pyproject paths untouched.
    import importlib.util
    path = REPO / "scripts" / "report_shape_recovery.py"
    spec = importlib.util.spec_from_file_location(
        "report_shape_recovery_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fixture_report_records_supplier_effective_procedure_and_nonrigid_positive():
    from evaluation.reporter import build_report
    from force_consistency.data.types import PreparationPolicy

    module = _load_report_script()
    evaluated = module.run_fixture_slice(PreparationPolicy())
    report, diagnostics = build_report(evaluated)
    rows = {row["branch_key"]: row for row in diagnostics["per_branch"]}

    # The wrong-shape plus branch forces through a SHIFTED per-atom
    # potential, so its audit entry must name that supplier, not pass
    # under the standard harmonic arm's identity.
    wrong_arm = rows["analytic/wrongshape/plus"]["arm"]
    assert wrong_arm["force_supplier"] == \
        "analytic-fixture-harmonic-shifted-per-atom-potential"
    assert wrong_arm["arm_id"] == "analytic-fixture-8at-harmonic"

    # The three branches that really use the standard arm keep saying so.
    for key in ("analytic/recover/plus", "analytic/recover/minus",
                "analytic/wrongshape/minus"):
        assert rows[key]["arm"]["force_supplier"] == "analytic-fixture-harmonic"

    # Every branch carries the EFFECTIVE procedure next to the configured
    # one; an unoverridden branch shows both identical and no overrides.
    for key, row in rows.items():
        procedure = row["procedure"]
        assert procedure["effective"] == procedure["configured"] or \
            "overrides" in procedure
    standard = rows["analytic/recover/plus"]["procedure"]
    assert standard["configured"]["max_force_evaluations"] == 1500
    assert standard["effective"]["max_force_evaluations"] == 1500
    assert "overrides" not in standard

    # The 2-call cap override on wrong-shape minus is recorded and
    # LABELLED synthetic, with the configured 1,500 visible beside it.
    minus_procedure = rows["analytic/wrongshape/minus"]["procedure"]
    override = minus_procedure["overrides"]["max_force_evaluations"]
    assert override["effective"] == 2
    assert override["configured"] == 1500
    assert override["label"] == "synthetic override"
    assert minus_procedure["effective"]["max_force_evaluations"] == 2
    assert minus_procedure["configured"]["max_force_evaluations"] == 1500
    assert report["branch_count"] == len(rows) == 4

    # The nonrigid positive fixture: a NONRIGID per-atom displacement that
    # starts OUTSIDE the 0.05 angstrom match tolerance and becomes
    # matched under the configured procedure. Both facts are printed.
    nonrigid = module.run_nonrigid_positive_fixture(PreparationPolicy())
    assert nonrigid["pre_relaxation_rmsd_A"] > 0.05
    assert nonrigid["tolerance_A"] == 0.05
    assert nonrigid["outcome"]["category"] == "matched_reference"
    assert nonrigid["outcome"]["rmsd_A"] <= 0.05


# ---------------------------------------------------------------------------
# Review round 4, card B (gap 1): the two surfaces must also agree on a
# FAILED branch's repeated failure facts, and a declared reason must sit
# beside facts it can describe. The witnesses mutate GENUINE failed
# evaluate_starts rows over the recovery landscape — the all-NaN arm
# (summary stop_reason nonfinite_force) and the one-call budget fixture
# (summary stop_reason budget_exhausted). Before this round all four
# mutations were ACCEPTED with the branch still counted nonconverged.
# ---------------------------------------------------------------------------

def _recovery_failed_runner_rows():
    """Genuine failed runner rows: the all-NaN arm and the one-call budget.

    Both runs use the self-contained recovery landscape from the runner
    test module, so every witness mutates real runner output instead of
    a caller-built dict. The composition is _recovery_runner_result's,
    built explicitly because the two runs need different arms and
    different procedures: the NaN arm refuses its first evaluation and
    stops on nonfinite_force; the bowl arm under max_force_evaluations=1
    spends its only attempt on the start force (finite, above the rule)
    and stops on budget_exhausted.
    """
    import test_shape_recovery_runner as runner_support
    from conftest import SPECIES
    from evaluation.adapter import callable_arm
    from evaluation.runner import evaluate_starts

    landscape = runner_support.RecoveryLandscape()
    manifest = copy.deepcopy(landscape.manifest)
    classification = runner_support._recovery_classification(landscape)
    references = landscape.references()
    procedure = runner_support._runner_procedure()
    start_set = {key: entry for key, entry in
                 runner_support._recovery_start_set(landscape, manifest).items()
                 if key.split("/")[1] == "80"}

    def nan_force(positions):
        array = np.zeros(np.asarray(positions).shape, dtype=np.float64)
        array[..., 0] = np.nan
        return array

    nan_arm = callable_arm(nan_force, species=SPECIES, arm_id="nan-recovery")
    nan_branches = evaluate_starts(
        start_set, nan_arm, procedure, classification, references,
        landscape.policy(), split_manifest=manifest, seen_minimum=True,
    )["branches"]
    bowl_arm = runner_support._bowl_arm_for(runner_support.recovery_minimum(0))
    budget_branches = evaluate_starts(
        start_set, bowl_arm,
        runner_support._runner_procedure(max_force_evaluations=1),
        classification, references, landscape.policy(),
        split_manifest=manifest, seen_minimum=True,
    )["branches"]
    return nan_branches, budget_branches


def test_failed_branch_repeated_facts_between_surfaces_are_reconciled():
    # The reviewer's four witnesses, all against genuine failed runner
    # rows. Where both surfaces state the same failure fact they must
    # agree; a reason must sit beside facts it can describe. Each
    # mutated row is refused with error naming the two surfaces and
    # the facts that disagree.
    from evaluation.reporter import build_report

    nan_branches, budget_branches = _recovery_failed_runner_rows()
    # The witnesses' genuine facts, asserted so a fixture change here
    # fails loudly instead of muting the witnesses below: the all-NaN
    # arm stops on nonfinite_force with nonfinite values; the one-call
    # budget stops on budget_exhausted with finite values and no
    # refusal at all.
    for row in nan_branches.values():
        assert row["category"] == "nonconverged"
        assert row["stop_reason"] == "nonfinite_force"
        assert row["nonfinite_force"] is True
        assert row["forces_finite"] is False
        assert row["budget_exhausted"] is False
        nested = row["_audit_record"]["failure"]
        assert nested["nonfinite_force"] is True
        assert nested["forces_finite"] is False
        assert nested["refusal_details"]
    for row in budget_branches.values():
        assert row["category"] == "nonconverged"
        assert row["stop_reason"] == "budget_exhausted"
        assert row["nonfinite_force"] is False
        assert row["forces_finite"] is True
        assert row["budget_exhausted"] is True
        assert row["_audit_record"]["failure"]["refusal_details"] == []

    # MUTATION a: nested nonfinite_force flipped to False while the
    # summary states nonfinite_force=True.
    forged_a = copy.deepcopy(nan_branches)
    key = sorted(forged_a)[0]
    forged_a[key]["_audit_record"]["failure"]["nonfinite_force"] = False
    with pytest.raises(ValueError) as excinfo:
        build_report(forged_a)
    message = str(excinfo.value)
    assert key in message
    assert "contradictory failure audit" in message
    assert "nonfinite_force=True" in message
    assert "nonfinite_force=False" in message
    assert "summary" in message and "audit" in message

    # MUTATION b: both nested reasons switched to 'invalid_force_shape'
    # while the summary states the stop reason 'nonfinite_force'.
    forged_b = copy.deepcopy(nan_branches)
    key = sorted(forged_b)[0]
    nested = forged_b[key]["_audit_record"]["failure"]
    nested["stop_reason"] = "invalid_force_shape"
    nested["failure_reason"] = "invalid_force_shape"
    with pytest.raises(ValueError) as excinfo:
        build_report(forged_b)
    message = str(excinfo.value)
    assert key in message
    assert "contradictory failure audit" in message
    assert "'nonfinite_force'" in message
    assert "'invalid_force_shape'" in message

    # MUTATION c: nested reason 'no_progress' against a summary that
    # states the stop reason 'budget_exhausted'.
    forged_c = copy.deepcopy(budget_branches)
    key = sorted(forged_c)[0]
    nested = forged_c[key]["_audit_record"]["failure"]
    nested["stop_reason"] = "no_progress"
    nested["failure_reason"] = "no_progress"
    with pytest.raises(ValueError) as excinfo:
        build_report(forged_c)
    message = str(excinfo.value)
    assert key in message
    assert "contradictory failure audit" in message
    assert "'budget_exhausted'" in message
    assert "'no_progress'" in message

    # MUTATION d: nested forces_finite flipped to False while the
    # summary states forces_finite=True.
    forged_d = copy.deepcopy(budget_branches)
    key = sorted(forged_d)[0]
    forged_d[key]["_audit_record"]["failure"]["forces_finite"] = False
    with pytest.raises(ValueError) as excinfo:
        build_report(forged_d)
    message = str(excinfo.value)
    assert key in message
    assert "contradictory failure audit" in message
    assert "forces_finite=True" in message
    assert "forces_finite=False" in message


def test_failed_branch_honest_and_omitted_optional_keys_still_pass():
    # The constructive half: the unmutated rows are still ACCEPTED and
    # the category totals are unchanged (every branch nonconverged,
    # counted under its own partition), and an honest row whose audit
    # simply omits an optional nested fact is accepted too: what one
    # surface does not state is not a contradiction, and no default is
    # invented for it.
    from evaluation.reporter import build_report

    nan_branches, budget_branches = _recovery_failed_runner_rows()
    for branches in (nan_branches, budget_branches):
        report, diagnostics = build_report(branches)
        assert report["branch_count"] == 2
        totals = report["category_totals"]["held_out"]
        assert totals["nonconverged"] == 2
        assert totals["matched_reference"] == 0
        assert report["held_out_branch_counts"] == {"validation": 2, "test": 0}
        assert {row["category"] for row in diagnostics["per_branch"]} == \
            {"nonconverged"}

    # The audit drops the optional budget_exhausted key on one row: the
    # summary still states it, the audit does not, and the row
    # validates with the same totals. (The runner's nested section
    # never states "converged" either — the unmutated runs above
    # already exercise that absent-key acceptance.)
    pruned = copy.deepcopy(budget_branches)
    key = sorted(pruned)[0]
    del pruned[key]["_audit_record"]["failure"]["budget_exhausted"]
    report, diagnostics = build_report(pruned)
    assert report["branch_count"] == 2
    assert report["category_totals"]["held_out"]["nonconverged"] == 2

    # The audit drops the optional finiteness keys on one row of the
    # all-NaN run: still accepted, still counted nonconverged.
    pruned = copy.deepcopy(nan_branches)
    key = sorted(pruned)[0]
    nested = pruned[key]["_audit_record"]["failure"]
    del nested["nonfinite_force"]
    del nested["forces_finite"]
    report, diagnostics = build_report(pruned)
    assert report["branch_count"] == 2
    assert report["category_totals"]["held_out"]["nonconverged"] == 2


# ---------------------------------------------------------------------------
# Review round 4, card D (residual A1): a stated reason must sit beside
# stated facts it can describe, on each surface separately. A shape
# refusal means the returned value never became a numeric array, so no
# nonfinite value can have been seen; declaring 'invalid_force_shape'
# while STATING nonfinite_force=True beside it is refused on both the
# summary surface and the nested audit section. Before this round both
# A1 shapes were accepted, because the finiteness fact was left
# unstated while the nonfinite fact was stated.
# ---------------------------------------------------------------------------

def test_a1_summary_reason_invalid_force_shape_with_stated_nonfinite_force_is_refused():
    # The summary A1 shape: failure_reason 'invalid_force_shape' with
    # nonfinite_force=True stated beside it, forces_finite unstated
    # and no audit record. The refusal names the surface, the reason
    # and the contradicting stated fact.
    from evaluation.reporter import build_report

    forged = _outcome("aspirin/01/plus", "nonconverged", True,
                      failure_reason="invalid_force_shape",
                      nonfinite_force=True)
    honest = _outcome("aspirin/01/minus", "nonconverged", True)
    with pytest.raises(ValueError) as excinfo:
        build_report({"aspirin/01/plus": forged,
                      "aspirin/01/minus": honest})
    message = str(excinfo.value)
    assert "aspirin/01/plus" in message
    assert "summary surface" in message
    assert "'invalid_force_shape'" in message
    assert "nonfinite_force=True" in message
    assert ("contradicting stated fact(s): nonfinite_force=True"
            in message)


def test_a1_nested_reason_invalid_force_shape_with_stated_nonfinite_force_is_refused():
    # The nested A1 shape: the audit section declares only stop_reason
    # 'invalid_force_shape' with nonfinite_force=True beside it, while
    # the summary states no reason and no finiteness fact. The nested
    # surface's own facts cannot describe its own reason, and the
    # refusal says so.
    from evaluation.reporter import build_report

    forged = _outcome("aspirin/01/plus", "nonconverged", True)
    forged["_audit_record"] = {
        "arm": {"kind": "callable", "arm_id": "forging-arm"},
        "failure": {"stop_reason": "invalid_force_shape",
                    "nonfinite_force": True},
    }
    honest = _outcome("aspirin/01/minus", "nonconverged", True)
    with pytest.raises(ValueError) as excinfo:
        build_report({"aspirin/01/plus": forged,
                      "aspirin/01/minus": honest})
    message = str(excinfo.value)
    assert "aspirin/01/plus" in message
    assert "contradictory failure audit" in message
    assert "'invalid_force_shape'" in message
    assert "nonfinite_force=True" in message
    assert ("contradicting stated fact(s): nonfinite_force=True"
            in message)


# ---------------------------------------------------------------------------
# Review round 5, card E: a budget stop beside a STATED forces_finite=False
# is refused on each surface separately. The relaxation refuses a run on
# its first nonfinite force and the budget-exhaustion assignment is only
# reachable while no failure reason is set (evaluation/relax.py: the
# refusal return and the exhaustion branch are mutually exclusive), so a
# run that stopped on the budget can never have seen nonfinite values.
# Only STATED facts are checked: an absent finiteness key on either
# surface stays "not stated" and is never defaulted into the refusal.
# ---------------------------------------------------------------------------

def test_budget_row_both_surfaces_stating_forces_finite_false_is_refused():
    # The reviewer's reproduction: forces_finite flipped to False on BOTH
    # surfaces of an honest budget row. The summary surface's
    # reason-vs-facts refusal fires first and names the surface, the
    # stated reason and the exact contradicting stated fact.
    from evaluation.reporter import build_report

    _, budget_rows = _recovery_failed_runner_rows()
    forged = copy.deepcopy(budget_rows)
    key = sorted(forged)[0]
    forged[key]["forces_finite"] = False
    forged[key]["_audit_record"]["failure"]["forces_finite"] = False
    with pytest.raises(ValueError) as excinfo:
        build_report(forged)
    message = str(excinfo.value)
    assert key in message
    assert "summary surface" in message
    assert "'budget_exhausted'" in message
    assert "forces_finite=False" in message
    assert ("contradicting stated fact(s): forces_finite=False"
            in message)


def test_budget_row_summary_only_forces_finite_false_is_refused():
    # Summary-only shape: forces_finite=False stated on the summary alone,
    # the audit stating neither finiteness nor nonfinite. Still refused:
    # the stated False cannot describe the budget reason on its own
    # surface, and the audit's silence does not repair it.
    from evaluation.reporter import build_report

    _, budget_rows = _recovery_failed_runner_rows()
    forged = copy.deepcopy(budget_rows)
    key = sorted(forged)[0]
    forged[key]["forces_finite"] = False
    nested = forged[key]["_audit_record"]["failure"]
    del nested["forces_finite"]
    del nested["nonfinite_force"]
    with pytest.raises(ValueError) as excinfo:
        build_report(forged)
    message = str(excinfo.value)
    assert key in message
    assert "summary surface" in message
    assert "'budget_exhausted'" in message
    assert "forces_finite=False" in message
    assert ("contradicting stated fact(s): forces_finite=False"
            in message)


def test_budget_row_audit_only_forces_finite_false_is_refused():
    # Audit-only shape: the nested section states forces_finite=False
    # while the summary states no finiteness fact. The nested surface's
    # own facts cannot describe its own budget reason: refused on the
    # audit's own evidence, with the audit surface named.
    from evaluation.reporter import build_report

    _, budget_rows = _recovery_failed_runner_rows()
    forged = copy.deepcopy(budget_rows)
    key = sorted(forged)[0]
    del forged[key]["forces_finite"]
    forged[key]["_audit_record"]["failure"]["forces_finite"] = False
    with pytest.raises(ValueError) as excinfo:
        build_report(forged)
    message = str(excinfo.value)
    assert key in message
    assert "contradictory failure audit" in message
    assert "audit surface" in message
    assert "'budget_exhausted'" in message
    assert "forces_finite=False" in message
    assert ("contradicting stated fact(s): forces_finite=False"
            in message)


def test_budget_row_absent_finiteness_keys_still_accepted():
    # The optional-key half of the rule: with the finiteness keys absent
    # on BOTH surfaces, the budget row still validates and the totals are
    # unchanged — "not stated" is never defaulted into the new refusal.
    from evaluation.reporter import build_report

    _, budget_rows = _recovery_failed_runner_rows()
    pruned = copy.deepcopy(budget_rows)
    key = sorted(pruned)[0]
    del pruned[key]["forces_finite"]
    nested = pruned[key]["_audit_record"]["failure"]
    del nested["forces_finite"]
    del nested["nonfinite_force"]
    report, diagnostics = build_report(pruned)
    assert report["branch_count"] == 2
    assert report["category_totals"]["held_out"]["nonconverged"] == 2
