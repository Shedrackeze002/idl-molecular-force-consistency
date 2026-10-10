"""Tests for the five-category outcome taxonomy (Task 1 taxonomy half).

The taxonomy reuses classify_endpoint from the frozen matching module for
every structural comparison. Five categories, never collapsed:

matched_reference, unmatched_converged, ambiguous, unresolved, nonconverged.
"""
import copy
import numpy as np

from force_consistency.data.types import PreparationPolicy

from conftest import build_minima_pool


def _final_at(positions_source, record_id, landscape):
    from force_consistency.data.types import FrameRecord

    template = landscape["records"][0]
    positions = np.array(positions_source, dtype=np.float64)
    positions.setflags(write=False)
    return FrameRecord(
        record_id=record_id, molecule=template.molecule, kind="path",
        source_path=record_id, connection_id="00", branch="plus", frame_index=0,
        species=template.species, positions=positions, energy=0.0,
        forces=np.zeros_like(positions), source_sha256="0" * 64,
    )


def _policy(landscape):
    return landscape["policy"]


# ---------------------------------------------------------------------------
# matched_reference
# ---------------------------------------------------------------------------

def test_final_at_each_minimum_matches_only_that_class(synthetic_landscape):
    from evaluation.outcomes import classify_outcome

    landscape = synthetic_landscape
    references = build_minima_pool(landscape)
    for name, record in references.items():
        final = _final_at(record.positions, f"synthetic/final/at/{name}", landscape)
        own = references["m0"] if name == "m0" else references["m1"]
        result = classify_outcome(
            final, own, references, _policy(landscape),
            budget_exhausted=False, converged=True, forces_finite=True, seen_minimum=True,
        )
        assert result["category"] == "matched_reference"
        assert result["matching"]["status"] == "matched"
        # The class representative is the reference pool key of the matched
        # minimum; the accepted classes list carries that one key.
        assert result["matching"]["reference_class"] == name
        assert result["matching"]["accepted_classes"] == [name]
        # The seen_minimum flag is carried through on every result.
        assert result["seen_minimum"] is True
        assert result["converged"] is True and result["budget_exhausted"] is False
        assert result["rmsd_A"] is not None and result["rmsd_A"] <= 1e-12
        assert result["permutation"] is not None and result["rotation"] is not None


def test_final_at_the_other_minimum_is_unmatched_converged(synthetic_landscape):
    from evaluation.outcomes import classify_outcome

    landscape = synthetic_landscape
    references = build_minima_pool(landscape)
    final = _final_at(references["m1"].positions, "synthetic/final/at/m1", landscape)
    result = classify_outcome(
        final, references["m0"], references, _policy(landscape),
        budget_exhausted=False, converged=True, forces_finite=True, seen_minimum=False,
    )
    # Converged to a fixed point that is not the branch's own reference:
    # a wrong shape, not a failure to converge.
    assert result["category"] == "unmatched_converged"
    assert result["matching"]["status"] == "unmatched"
    assert result["matching"]["reference_class"] is None
    assert result["seen_minimum"] is False


# ---------------------------------------------------------------------------
# ambiguous
# ---------------------------------------------------------------------------

def test_midway_structure_between_minima_is_ambiguous(synthetic_landscape):
    from evaluation.outcomes import classify_outcome

    landscape = synthetic_landscape
    references = build_minima_pool(landscape)
    midpoint = 0.5 * (references["m0"].positions + references["m1"].positions)
    final = _final_at(midpoint, "synthetic/final/midway", landscape)
    result = classify_outcome(
        final, references["m0"], references, _policy(landscape),
        budget_exhausted=False, converged=True, forces_finite=True, seen_minimum=True,
    )
    assert result["category"] == "ambiguous"
    assert result["matching"]["status"] == "ambiguous"
    assert len(result["matching"]["accepted_classes"]) == 2


# ---------------------------------------------------------------------------
# unresolved
# ---------------------------------------------------------------------------

def test_nonfinite_final_structure_is_unresolved_and_retained(synthetic_landscape):
    from evaluation.outcomes import classify_outcome

    landscape = synthetic_landscape
    references = build_minima_pool(landscape)
    poisoned = np.array(references["m0"].positions, dtype=np.float64, copy=True)
    poisoned[0, 0] = np.nan
    final = _final_at(poisoned, "synthetic/final/nan", landscape)
    result = classify_outcome(
        final, references["m0"], references, _policy(landscape),
        budget_exhausted=False, converged=True, forces_finite=True, seen_minimum=True,
    )
    assert result["category"] == "unresolved"
    assert result["seen_minimum"] is True
    # The nonfinite structure itself is retained for audit, not discarded.
    assert result["final_positions"] is final.positions


def test_search_limit_incompleteness_is_unresolved():
    from dataclasses import replace

    from evaluation.outcomes import classify_outcome
    from force_consistency.data.types import FrameRecord

    # A frame with a symmetric chemical graph: the square planar C4 unit has
    # four equal species-respecting graph mappings. A policy whose search
    # limit is 1 cannot finish enumerating them, so the comparison is
    # honestly unresolved instead of silently partial.
    square = 0.7 * np.array([[1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0]], dtype=np.float64)
    own = FrameRecord("ref-square", "aspirin", "minimum", "ref-square", None, None, None,
                      ("C", "C", "C", "C"), square, 0.0, np.zeros((4, 3)), "0" * 64)
    rotated = square @ np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    rotated.setflags(write=False)
    final = FrameRecord("final-square", "aspirin", "path", "final-square", "00", "plus", 0,
                        ("C", "C", "C", "C"), rotated, 0.0, np.zeros((4, 3)), "0" * 64)
    references = {"ref-square": own}
    tight = replace(PreparationPolicy(), search_limit=1)
    result = classify_outcome(
        final, own, references, tight,
        budget_exhausted=False, converged=True, forces_finite=True, seen_minimum=True,
    )
    assert result["category"] == "unresolved"
    # The same comparison under a generous limit matches cleanly, which
    # proves the unresolved verdict came from the limit, not geometry.
    generous = replace(PreparationPolicy(), search_limit=100000)
    ok = classify_outcome(
        final, own, references, generous,
        budget_exhausted=False, converged=True, forces_finite=True, seen_minimum=True,
    )
    assert ok["category"] == "matched_reference"


# ---------------------------------------------------------------------------
# nonconverged
# ---------------------------------------------------------------------------

def test_budget_exhausted_run_is_never_counted_as_a_wrong_shape(synthetic_landscape):
    from evaluation.outcomes import classify_outcome

    landscape = synthetic_landscape
    references = build_minima_pool(landscape)
    # Final structure sits exactly at the WRONG minimum, but the relaxation
    # never met the force rule. Even a converged-shaped endpoint must not be
    # counted against the arm.
    final = _final_at(references["m1"].positions, "synthetic/final/wrong-shape", landscape)
    result = classify_outcome(
        final, references["m0"], references, _policy(landscape),
        budget_exhausted=True, converged=False, forces_finite=True, seen_minimum=True,
    )
    assert result["category"] == "nonconverged"
    assert result["seen_minimum"] is True


def test_nonfinite_force_terminates_as_nonconverged(synthetic_landscape):
    from evaluation.outcomes import classify_outcome

    landscape = synthetic_landscape
    references = build_minima_pool(landscape)
    final = _final_at(references["m1"].positions, "synthetic/final/nonfinite-force", landscape)
    result = classify_outcome(
        final, references["m0"], references, _policy(landscape),
        budget_exhausted=False, converged=False, forces_finite=False, seen_minimum=True,
    )
    # A nonfinite force ends the run; the outcome is nonconverged, never a
    # shape verdict, and never unresolved.
    assert result["category"] == "nonconverged"
    assert result["nonfinite_force"] is True


def test_nonconverged_takes_priority_over_every_comparison_verdict(synthetic_landscape):
    from evaluation.outcomes import classify_outcome

    landscape = synthetic_landscape
    references = build_minima_pool(landscape)
    # A nonconverged run whose final happens to sit at the own reference
    # still reports nonconverged.
    final = _final_at(references["m0"].positions, "synthetic/final/lucky-endpoint", landscape)
    lucky = classify_outcome(
        final, references["m0"], references, _policy(landscape),
        budget_exhausted=True, converged=False, forces_finite=True, seen_minimum=True,
    )
    assert lucky["category"] == "nonconverged"
    # A nonconverged run near the midway point likewise never becomes
    # ambiguous: the comparison is not even attempted.
    midpoint = 0.5 * (references["m0"].positions + references["m1"].positions)
    midway = classify_outcome(
        _final_at(midpoint, "synthetic/final/midway-nc", landscape), references["m0"],
        references, _policy(landscape),
        budget_exhausted=True, converged=False, forces_finite=True, seen_minimum=True,
    )
    assert midway["category"] == "nonconverged"
    assert "matching" not in midway or midway.get("matching") is None


# ---------------------------------------------------------------------------
# result contract
# ---------------------------------------------------------------------------

def test_stalled_run_with_unused_budget_keeps_supplied_termination_facts(synthetic_landscape):
    from evaluation.outcomes import classify_outcome

    landscape = synthetic_landscape
    references = build_minima_pool(landscape)
    # A run that stops above the force rule on a stuck structure after 41
    # of 100 budgeted evaluations. The category stays nonconverged either
    # way, but the TERMINATION FACTS must come out exactly as supplied:
    # overwriting budget_exhausted to True, or dropping the stop reason,
    # misdescribes how the run ended.
    final = _final_at(references["m1"].positions, "synthetic/final/stalled", landscape)
    result = classify_outcome(
        final, references["m0"], references, _policy(landscape),
        budget_exhausted=False, converged=False, forces_finite=True,
        seen_minimum=True, stop_reason="no_progress",
    )
    assert result["category"] == "nonconverged"
    assert result["budget_exhausted"] is False
    assert result["converged"] is False
    assert result["forces_finite"] is True
    assert result["stop_reason"] == "no_progress"


def test_budget_exhausted_stop_reason_is_carried_through(synthetic_landscape):
    from evaluation.outcomes import classify_outcome

    landscape = synthetic_landscape
    references = build_minima_pool(landscape)
    final = _final_at(references["m1"].positions, "synthetic/final/budget", landscape)
    result = classify_outcome(
        final, references["m0"], references, _policy(landscape),
        budget_exhausted=True, converged=False, forces_finite=True,
        seen_minimum=True, stop_reason="budget_exhausted",
    )
    assert result["category"] == "nonconverged"
    assert result["budget_exhausted"] is True
    assert result["stop_reason"] == "budget_exhausted"


def test_nonfinite_forced_stop_reason_matches_the_refusal(synthetic_landscape):
    from evaluation.outcomes import classify_outcome

    landscape = synthetic_landscape
    references = build_minima_pool(landscape)
    final = _final_at(references["m1"].positions, "synthetic/final/nf-reason", landscape)
    result = classify_outcome(
        final, references["m0"], references, _policy(landscape),
        budget_exhausted=False, converged=False, forces_finite=True,
        seen_minimum=True, nonfinite_force=True,
    )
    assert result["category"] == "nonconverged"
    assert result["nonfinite_force"] is True
    assert result["stop_reason"] == "nonfinite_force"


def test_forged_stop_reason_is_rejected(synthetic_landscape):
    import pytest

    from evaluation.outcomes import classify_outcome

    landscape = synthetic_landscape
    references = build_minima_pool(landscape)
    final = _final_at(references["m0"].positions, "synthetic/final/forged", landscape)
    with pytest.raises(ValueError):
        classify_outcome(
            final, references["m0"], references, _policy(landscape),
            budget_exhausted=False, converged=False, forces_finite=True,
            seen_minimum=True, stop_reason="vibes",
        )


def test_unsupplied_stopped_run_defaults_to_the_honest_no_progress_reason(
        synthetic_landscape):
    # Compatibility: a caller that supplies the bare pre-review interface
    # (no stop_reason) still gets an honest termination record, not a
    # silently dropped fact.
    from evaluation.outcomes import classify_outcome

    landscape = synthetic_landscape
    references = build_minima_pool(landscape)
    final = _final_at(references["m0"].positions, "synthetic/final/bare", landscape)
    result = classify_outcome(
        final, references["m0"], references, _policy(landscape),
        budget_exhausted=False, converged=False, forces_finite=True,
        seen_minimum=True,
    )
    assert result["category"] == "nonconverged"
    assert result["budget_exhausted"] is False
    assert result["stop_reason"] == "no_progress"


def test_result_carries_exactly_one_category_and_measured_fields(synthetic_landscape):
    from evaluation.outcomes import CATEGORY_NAMES, classify_outcome

    landscape = synthetic_landscape
    references = build_minima_pool(landscape)
    final = _final_at(references["m0"].positions, "synthetic/final/contract", landscape)
    result = classify_outcome(
        final, references["m0"], references, _policy(landscape),
        budget_exhausted=False, converged=True, forces_finite=True, seen_minimum=True,
    )
    assert sorted(CATEGORY_NAMES) == sorted([
        "matched_reference", "unmatched_converged", "ambiguous", "unresolved", "nonconverged"])
    assert result["category"] in CATEGORY_NAMES
    # Exactly one category flag is true.
    flags = [result["is_" + name] for name in CATEGORY_NAMES]
    assert sum(bool(flag) for flag in flags) == 1
    assert copy.deepcopy(result) is not None