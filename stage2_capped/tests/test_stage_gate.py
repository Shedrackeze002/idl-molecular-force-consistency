"""M4: stage gating and the Stage 0 mismatch state machine (pure, no model)."""
import itertools

import pytest

from stage2_capped import rules
from stage2_capped.stage_gate import (STAGE_ORDER, StageLedger, StageOrderError, Stage0Controller,
                                      Stage0Halted)


def run(ledger, stage, ok=True, **kwargs):
    ledger.start(stage)
    ledger.finish(stage, ok, **kwargs)


# ---- Stage ledger --------------------------------------------------------

def test_the_happy_path_runs_in_the_spec_order():
    ledger = StageLedger()
    run(ledger, "A0", replay_passed=True)
    for stage in ("0", "0b", "1", "2"):
        run(ledger, stage)
    assert all(state == "complete" for state in ledger.state.values())


@pytest.mark.parametrize("stage, needed", [("0", "A0"), ("0b", "0"), ("1", "0"), ("2", "1")])
def test_a_stage_cannot_start_before_its_prerequisites(stage, needed):
    ledger = StageLedger()
    with pytest.raises(StageOrderError, match=f"stage {needed}"):
        ledger.start(stage)


def test_stage_two_needs_stage_one_stage_0b_and_the_a0_replay():
    for missing in ("1", "0b"):
        ledger = StageLedger()
        run(ledger, "A0", replay_passed=True)
        run(ledger, "0")
        for stage in ("0b", "1"):
            if stage != missing:
                run(ledger, stage)
        with pytest.raises(StageOrderError, match=f"stage {missing}"):
            ledger.start("2")
    ledger = StageLedger()
    ledger.start("A0")
    ledger.finish("A0", True, replay_passed=False)  # admission ran but the replay did not pass
    assert ledger.state["A0"] == "failed" and not ledger.a0_replay_passed


def test_a_failed_a0_blocks_everything_and_a_failed_stage_zero_blocks_the_rest():
    ledger = StageLedger()
    run(ledger, "A0", ok=False, replay_passed=True)
    assert ledger.state["A0"] == "failed"
    for stage in ("0", "0b", "1", "2"):
        assert ledger.blockers(stage)
    ledger = StageLedger()
    run(ledger, "A0", replay_passed=True)
    run(ledger, "0", ok=False)
    for stage in ("0b", "1", "2"):
        assert ledger.blockers(stage)


def test_stage_one_does_not_need_stage_0b():
    ledger = StageLedger()
    run(ledger, "A0", replay_passed=True)
    run(ledger, "0")
    run(ledger, "0b", ok=False)
    assert ledger.blockers("1") == []          # the spec's table: Stage 1 needs Stage 0 only
    assert ledger.blockers("2")                # but Stage 2 needs 0b complete


def test_no_stage_runs_twice_and_none_resumes():
    ledger = StageLedger()
    run(ledger, "A0", replay_passed=True)
    with pytest.raises(StageOrderError, match="runs once"):
        ledger.start("A0")
    run(ledger, "0", ok=False)
    with pytest.raises(StageOrderError, match="no retry or resume"):
        ledger.start("0")


def test_only_one_stage_runs_at_a_time_and_finish_needs_a_running_stage():
    ledger = StageLedger()
    ledger.start("A0")
    with pytest.raises(StageOrderError, match="still running"):
        ledger.start("0")
    with pytest.raises(StageOrderError, match="not running"):
        ledger.finish("0", True)
    with pytest.raises(StageOrderError):
        ledger.start("nonsense")


def test_the_ledger_snapshot_records_every_transition():
    ledger = StageLedger()
    run(ledger, "A0", replay_passed=True)
    snapshot = ledger.snapshot()
    assert snapshot["events"] == [["A0", "started"], ["A0", "complete"]] and snapshot["a0_replay_passed"]


def test_exhaustively_no_order_violation_is_possible():
    """Over every sequence of up to four start attempts with success or failure, a stage
    starts only if its prerequisites completed and it has not run before."""
    for sequence in itertools.product(STAGE_ORDER, repeat=4):
        for outcomes in itertools.product((True, False), repeat=4):
            ledger = StageLedger()
            for stage, ok in zip(sequence, outcomes):
                complete_before = {s for s, st in ledger.state.items() if st == "complete"}
                was_pending = ledger.state[stage] == "pending"
                try:
                    ledger.start(stage)
                except StageOrderError:
                    assert not was_pending or ledger.blockers(stage)
                    continue
                assert was_pending
                assert set(__import__("stage2_capped.stage_gate", fromlist=["x"]).PREREQUISITES[stage]) <= complete_before
                ledger.finish(stage, ok, replay_passed=True)


# ---- Stage 0 controller --------------------------------------------------

def test_a_clean_stage_zero_passes_only_after_all_80_runs():
    controller = Stage0Controller()
    for index in range(79):
        assert controller.observe(f"t{index}", []) is None
    assert controller.finish()["verdict"] == "stop: incomplete comparison"
    controller = Stage0Controller()
    for index in range(80):
        controller.observe(f"t{index}", [])
    verdict = controller.finish()
    assert verdict["verdict"] == "pass" and verdict["proceed"] is True


def test_the_first_mismatch_halts_and_asks_for_the_single_frozen_rerun():
    controller = Stage0Controller()
    for index in range(12):
        controller.observe(f"t{index}", [])
    verdict = controller.observe("t12", ["step_count: saved 1, new 2"])
    assert verdict["verdict"] == "frozen re-run required" and not verdict["proceed"]
    assert controller.compared == 12 and controller.mismatches == ["t12"] and controller.halted
    with pytest.raises(Stage0Halted):
        controller.observe("t13", [])                 # fail fast: nothing continues


@pytest.mark.parametrize("matches, expected", [(False, "stop: runtime drift"), (True, "stop: code error")])
def test_the_rerun_outcome_decides_drift_or_code_error_and_both_stop(matches, expected):
    controller = Stage0Controller()
    controller.observe("t0", ["differs"])
    controller.authorize_rerun("t0")
    verdict = controller.record_frozen_rerun("t0", matches)
    assert verdict["verdict"] == expected and not verdict["proceed"]
    assert controller.finish()["verdict"] == expected


def test_only_one_rerun_and_only_of_the_first_mismatching_task():
    controller = Stage0Controller()
    with pytest.raises(Stage0Halted, match="no mismatch"):
        controller.authorize_rerun("t0")
    controller.observe("t0", ["differs"])
    with pytest.raises(Stage0Halted, match="first mismatching task"):
        controller.authorize_rerun("t5")
    controller.authorize_rerun("t0")
    controller.record_frozen_rerun("t0", False)
    with pytest.raises(Stage0Halted, match="has been used"):
        controller.authorize_rerun("t0")
    with pytest.raises(Stage0Halted, match="has been used"):
        controller.record_frozen_rerun("t0", False)


def test_a_result_cannot_be_recorded_without_authorization():
    controller = Stage0Controller()
    controller.observe("t0", ["differs"])
    with pytest.raises(Stage0Halted, match="not authorized"):
        controller.record_frozen_rerun("t0", True)
    assert controller.rerun_done is False


def test_an_authorized_rerun_that_crashes_stays_spent():
    controller = Stage0Controller()
    controller.observe("t0", ["differs"])
    controller.authorize_rerun("t0")                 # the model call then fails; nothing is recorded
    with pytest.raises(Stage0Halted, match="has been used"):
        controller.authorize_rerun("t0")


def test_authorization_needs_a_halted_stage():
    controller = Stage0Controller()
    controller.observe("t0", [])
    with pytest.raises(Stage0Halted, match="no mismatch"):
        controller.authorize_rerun("t0")


def test_no_tolerance_a_single_bit_problem_is_a_mismatch():
    controller = Stage0Controller()
    assert controller.observe("t0", ["final_positions differ at /0/1"])["proceed"] is False


def test_stage_zero_verdicts_are_the_rules_verdicts():
    controller = Stage0Controller()
    controller.observe("t0", ["x"])
    assert controller.verdict == rules.r_control(0, ["t0"])


def test_stage_two_still_needs_the_replay_flag_if_the_ledger_is_set_up_by_hand():
    """Defense in depth: A0 complete without a passed replay cannot be reached through
    finish(), but the check on stage 2 does not rely on that."""
    ledger = StageLedger()
    ledger.state.update({"A0": "complete", "0": "complete", "0b": "complete", "1": "complete"})
    assert ledger.a0_replay_passed is False
    assert any("replay has not passed" in reason for reason in ledger.blockers("2"))
    ledger.a0_replay_passed = True
    assert ledger.blockers("2") == []


def test_a_worker_rebuilds_the_parents_ledger_and_the_same_start_rules_hold():
    parent = StageLedger()
    parent.start("A0")
    parent.finish("A0", True, replay_passed=True)
    snapshot = parent.snapshot()
    worker = StageLedger.from_snapshot(snapshot)
    assert worker.snapshot() == snapshot and worker.a0_replay_passed
    worker.start("0")                                              # allowed: A0 is complete
    with pytest.raises(StageOrderError):
        StageLedger.from_snapshot(snapshot).start("1")             # Stage 1 needs Stage 0
    with pytest.raises(StageOrderError):
        StageLedger().start("0")                                   # a fresh ledger knows nothing of A0


@pytest.mark.parametrize("bad", [{"state": {"A0": "pending"}, "a0_replay_passed": False, "events": []},
                                 {"state": {s: "bogus" for s in STAGE_ORDER}, "a0_replay_passed": False, "events": []}])
def test_a_malformed_ledger_snapshot_is_refused(bad):
    with pytest.raises(ValueError, match="not a ledger snapshot"):
        StageLedger.from_snapshot(bad)
