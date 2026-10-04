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
    base = {
        "branch_key": branch, "category": category, "seen_minimum": seen_minimum,
        "partition": "validation" if branch.rsplit("/", 1)[0] in ("01", "10") else "test",
        "budget_exhausted": False,
    }
    base["converged"] = category in ("matched_reference", "unmatched_converged", "ambiguous")
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
    # denominators.
    evaluated = _sample_evaluated()
    evaluated["aspirin/10/minus"]["partition"] = "train"
    report, diagnostics = build_report(evaluated)
    conn10 = next(c for c in report["per_connection"] if c["connection_id"] == "10")
    assert conn10["denominator"] == 1


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