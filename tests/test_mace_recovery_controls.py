import pytest

from run_mace_recovery_controls import BRANCHES, REFERENCE_IDS, summarize


def test_recovery_summary_keeps_historical_controls_separate():
    branches = {key: {"category": "nonconverged", "force_evaluation_count": 1500}
                for key in BRANCHES}
    refs = {key: {"operational_force_converged_own_match": True, "force_evaluation_count": 1}
            for key in REFERENCE_IDS}
    outcome = summarize(branches, refs, {"own_geometry_status": "verified", "own_geometry_count": 0})
    assert outcome["original_recovered"] == 0
    assert outcome["reference_own_matches"] == 6
    assert outcome["force_requests"] == 15006
    assert outcome["exceeds_no_relaxation"] is False
    branches.pop(next(iter(branches)))
    with pytest.raises(ValueError, match="schedule"):
        summarize(branches, refs, {"own_geometry_status": "verified", "own_geometry_count": 0})
