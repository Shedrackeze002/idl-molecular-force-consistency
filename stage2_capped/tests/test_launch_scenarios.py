"""Plan v3 sections 4 to 7 with amendments A1 to A7, B1 to B9: the parent and workers under failure. Every test runs the
real parent as a subprocess from a committed scratch tree, with fixture arms; the study inputs and saved results are
the real accepted files."""
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from stage2_capped import admission, driver, launch_checks, launch_io, launch_record, launch_state
from stage2_capped.tests.launch_fixtures import needs_real_files, scratch, scratch_base  # noqa: F401
from stage2_capped.tests.launch_harness import Harness

pytestmark = needs_real_files


@pytest.fixture
def h(scratch, tmp_path):
    return Harness(scratch, tmp_path)


def read(path):
    return json.loads(path.read_text())


def unit_events(h, sha, unit):
    return {p.name.split("-")[-1][:-5] + p.name.split("-")[-2]: read(p) for p in sorted(h.registry.glob(f"{sha}-{unit}-*"))}


def exited(h, sha, unit, attempt):
    return read(h.registry / f"{sha}-{unit}-{attempt}-exited.json")


def no_spend(h, sha, unit):
    assert not list(h.registry.glob(f"{sha}-{unit}-*")) if h.registry.exists() else True


# ---- check and refusals that spend nothing ----------------------------------------------------------------------

def test_check_passes_without_writing_and_refuses_the_wrong_record(h):
    done = h.run(*h.a0_arguments("check"))
    assert done.returncode == 0 and "nothing was written" in done.stdout
    assert not h.root_a.exists() and not h.registry.exists() and not (h.folder / "violations").exists()
    wrong = h.run(*h.stage_arguments("0", a0="0" * 64, record=(h.path_a, h.approval_a, h.sha_a), check=True))
    assert wrong.returncode == 2 and "needs record B" in wrong.stdout
    assert not h.root_a.exists() and not h.root_b.exists()


def test_a_dirty_tree_is_refused_before_anything_is_created(h):
    (h.scratch / "pyproject.toml").write_text((h.scratch / "pyproject.toml").read_text() + "\n# edited after the commit\n")
    done = h.run(*h.a0_arguments())
    assert done.returncode == 2 and "not clean" in done.stdout
    assert not h.root_a.exists() and not h.registry.exists()


def test_a_registry_event_the_record_does_not_list_is_refused(h):
    h.registry.mkdir(parents=True)
    (h.registry / ("c" * 64 + "-0-1-started.json")).write_text("{}")
    done = h.run(*h.a0_arguments())
    assert done.returncode == 2 and "not listed in prior_dispatches" in done.stdout and not h.root_a.exists()


def test_the_post_a0_guard_refuses_a_stage_the_measured_latency_cannot_fit_and_spends_nothing(h):
    first = h.run(*h.a0_arguments(), scenario={"slow_force_seconds": 0.004})
    assert first.returncode == 0, first.stdout + first.stderr
    done = h.run(*h.stage_arguments("0", a0=h.a0_result_sha()))
    assert done.returncode == 2 and "mean latency exceeds" in done.stdout
    assert not h.root_b.exists() and not list(h.registry.glob(f"{h.sha_b}-*"))


# ---- A0 outcomes --------------------------------------------------------------------------------------------------

def test_one_failing_model_fails_a0_but_all_seven_are_replayed_and_measured(h):
    done = h.run(*h.a0_arguments(), scenario={"perturb": {"direct-0": 1e-9}})
    assert done.returncode == 1 and "A0 verdict: FAIL" in done.stdout
    result = read(h.root_a / "A0_RESULT.json")
    assert result["verdict"]["passed"] is False and set(result["models"]) == set(admission.ALL_IDS)
    assert result["models"]["direct-0"]["failure"].startswith("replay:") and "replay" not in result["models"]["direct-0"]
    assert result["models"]["direct-0"]["evidence"]["batch4"]["max_scaled_residual"] > 1.0
    assert all(result["models"][m]["replay"]["status"] == "verified" for m in admission.ALL_IDS if m != "direct-0")
    assert exited(h, h.sha_a, "A0", 1)["outcome"] == "failed" and exited(h, h.sha_a, "A0", 1)["spent"] is True
    again = h.run(*h.a0_arguments())
    assert again.returncode == 2 and "already spent" in again.stdout
    stage = h.run(*h.stage_arguments("0", a0=h.a0_result_sha()))
    assert stage.returncode == 2 and "A0 verdict" in stage.stdout and not h.root_b.exists()


def test_a_restore_refusal_for_one_model_is_labelled_and_the_rest_are_still_replayed(h):
    done = h.run(*h.a0_arguments(), scenario={"restore_fail": "energy-0"})
    assert done.returncode == 1
    result = read(h.root_a / "A0_RESULT.json")
    assert result["models"]["energy-0"]["failure"].startswith("restore:")
    assert all(result["models"][m]["replay"]["status"] == "verified" for m in admission.ALL_IDS if m != "energy-0")
    assert "evidence" not in result["models"]["energy-0"]                    # no arm was built


def test_a_model_that_raises_is_a_replay_failure_with_evidence_errors_not_a_crash(h):
    done = h.run(*h.a0_arguments(), scenario={"raise": ["direct-1"]})
    assert done.returncode == 1
    entry = read(h.root_a / "A0_RESULT.json")["models"]["direct-1"]
    assert entry["failure"].startswith("replay:") and "error" in entry["evidence"]["batch4"] and "error" in entry["evidence"]["batch1"]


def test_an_a0_refusal_before_the_spent_point_spends_nothing_and_a_restart_reuses_the_root(h):
    first = h.run(*h.a0_arguments(), scenario={"fail_precondition_attempts": [1]})
    assert first.returncode == 2
    assert (h.root_a / "A0_REFUSED-1.json").is_file() and not (h.root_a / "A0_RESTORE_BEGUN.json").exists()
    assert "precondition failed" in read(h.root_a / "A0_REFUSED-1.json")["reason"]
    refused = exited(h, h.sha_a, "A0", 1)
    assert refused["outcome"] == "refused_before_start" and refused["spent"] is False
    second = h.run(*h.a0_arguments())
    assert second.returncode == 0, second.stdout + second.stderr
    assert exited(h, h.sha_a, "A0", 2)["outcome"] == "complete" and (h.root_a / "A0_RESULT.json").is_file()
    assert {p.name for p in h.root_a.glob("WORKER_PID-A0-*.json")} == {"WORKER_PID-A0-1.json", "WORKER_PID-A0-2.json"}
    assert launch_state.EventLog(h.root_a).problems() == [] and len(launch_state.EventLog(h.root_a).files()) == 4
    assert launch_state.Registry(h.registry).a0_problems(h.sha_a) == []


def test_a_worker_started_by_hand_is_refused_and_spends_nothing(h):
    launch_io.fresh_root(h.root_a, "A-e2e")
    command = [sys.executable, "-m", "stage2_capped.tests.launch_test_worker", "--kind", "a0", "--record", str(h.path_a),
               "--approval", str(h.approval_a), "--record-sha256", h.sha_a, "--attempt", "1", "--group9-root", str(h.group9)]
    done = subprocess.run(command, cwd=h.scratch, env=h.environment(), capture_output=True, text=True)
    assert done.returncode == 2, done.stdout + done.stderr
    assert "no `started` event" in read(h.root_a / "A0_REFUSED-1.json")["reason"]
    assert not (h.root_a / "A0_RESTORE_BEGUN.json").exists() and not (h.root_a / "A0_RESULT.json").exists()


# ---- Audit violations (B2) -----------------------------------------------------------------------------------------

def test_a_violation_in_a_worker_precondition_voids_the_unit_and_the_record_pair(h):
    done = h.run(*h.a0_arguments(), scenario={"open_forbidden": True})
    assert done.returncode == 1 and "audit violation" in done.stdout
    record = exited(h, h.sha_a, "A0", 1)
    assert record["outcome"] == "audit_violation" and record["spent"] is True
    assert (h.folder / "violations" / f"AUDIT_VIOLATION-{h.sha_a}-A0-1.json").is_file()
    assert not (h.root_a / "A0_RESULT.json").exists()                            # it never got past its preconditions
    for argv in (h.a0_arguments("check"), h.stage_arguments("0", a0="0" * 64, check=True)):
        again = h.run(*argv)
        assert again.returncode == 2 and "voids this record pair" in again.stdout


def test_a_violation_exit_recorded_in_the_registry_voids_the_pair_when_its_file_was_never_written(h):
    """B2: the fixed exit code voids the unit and the pair. If the violation file could not be written the registry still
    says `audit_violation`, and every later command of either record refuses."""
    registry = launch_state.Registry(h.registry)
    registry.write(h.sha_a, "A0", 1, "started", launch_id="A-e2e")
    registry.write(h.sha_a, "A0", 1, "exited", outcome="audit_violation", spent=True)
    assert not (h.folder / "violations").exists()
    for argv in (h.a0_arguments("check"), h.stage_arguments("0", a0="0" * 64, check=True)):
        again = h.run(*argv)
        assert again.returncode == 2 and "voids this record pair" in again.stdout


def test_an_unusable_group9_root_is_a_refusal_with_exit_2_not_a_traceback(h, tmp_path):
    """Exit 1 means `spent and not complete`; a command that fails before it checks anything must not exit 1."""
    done = h.run("a0", "--record", str(h.path_a), "--approval", str(h.approval_a), "--record-sha256", h.sha_a,
                 "--group9-root", str(tmp_path / "nowhere"))
    assert done.returncode == 2 and "is not the Group9 root" in done.stdout and "Traceback" not in done.stderr
    assert not h.root_a.exists() and not h.registry.exists()


def test_a_violation_in_the_parents_own_checks_exits_with_the_fixed_code_and_voids_the_pair(h):
    done = h.run(*h.a0_arguments("check"), scenario={"open_forbidden_parent": True})
    assert done.returncode == launch_checks.AUDIT_EXIT_CODE
    assert (h.folder / "violations" / f"AUDIT_VIOLATION-{h.sha_a}-A0-parent.json").is_file()
    assert not h.root_a.exists() and not list(h.registry.glob("*")) if h.registry.exists() else not h.root_a.exists()
    assert "voids this record pair" in h.run(*h.a0_arguments("check")).stdout
    assert "voids this record pair" in h.run(*h.stage_arguments("0", a0="0" * 64, check=True)).stdout


# ---- Stages ---------------------------------------------------------------------------------------------------------

def test_a_stage_0_mismatch_runs_the_single_frozen_rerun_once_and_spends_the_stage(h):
    first = h.run(*h.a0_arguments())
    assert first.returncode == 0
    a0 = h.a0_result_sha()
    done = h.run(*h.stage_arguments("0", a0=a0), scenario={"mismatch": True})
    assert done.returncode == 1 and "STAGE 0 INCOMPLETE, not a result" in done.stdout
    assert (h.root_b / "STAGE0_RERUN_ATTEMPTED.json").is_file() and (h.root_b / "STAGE0_RERUN.json").is_file()
    directory = h.root_b / "rows" / "stage0"
    assert (directory / "INCOMPLETE.json").is_file() and (directory / "STAGE0_MISMATCH.json").is_file()
    assert read(h.root_b / "worker-0.result.json")["error"].startswith("Stage0Stop")
    assert exited(h, h.sha_b, "0", 1)["outcome"] == "failed"
    again = h.run(*h.stage_arguments("0", a0=a0))
    assert again.returncode == 2 and "already spent" in again.stdout
    later = h.run(*h.stage_arguments("0b", a0=a0))
    assert later.returncode == 2 and "typed marker hash of stage 0" in later.stdout


def test_a_killed_stage_is_marked_incomplete_by_the_parent_and_a_failed_0b_does_not_block_stage_1(h):
    h.run_a0_and_stage0()
    a0, stage0 = h.a0_result_sha(), h.marker_sha("0")
    killed = h.run(*h.stage_arguments("0b", a0=a0, stage0=stage0), scenario={"kill_in_force": True})
    assert killed.returncode == 1 and "STAGE 0b INCOMPLETE, not a result" in killed.stdout
    directory = h.root_b / "rows" / "stage0b"
    assert (directory / "INCOMPLETE.json").is_file() and "worker did not finish" in read(directory / "INCOMPLETE.json")["reason"]
    assert exited(h, h.sha_b, "0b", 1)["exit"]["returncode"] == -9
    assert h.run(*h.stage_arguments("0b", a0=a0, stage0=stage0)).returncode == 2         # spent
    one = h.run(*h.stage_arguments("1", a0=a0, stage0=stage0))
    assert one.returncode == 0, one.stdout + one.stderr                                    # a failed 0b does not block stage 1
    assert read(h.root_b / "rows" / "stage1" / "STAGE_COMPLETE.json")["runs"] == 320
    two = h.run(*h.stage_arguments("2", a0=a0, stage1=h.marker_sha("1"), stage0b=stage0))
    assert two.returncode == 2                                                            # stage 2 needs a complete 0b


def test_an_unexpected_exception_after_the_spent_point_is_exit_4_with_a_result_file(h):
    h.run_a0_and_stage0()
    a0, stage0 = h.a0_result_sha(), h.marker_sha("0")
    done = h.run(*h.stage_arguments("0b", a0=a0, stage0=stage0), scenario={"unexpected_in_run": True})
    assert done.returncode == 1 and "STAGE 0b INCOMPLETE, not a result" in done.stdout
    assert exited(h, h.sha_b, "0b", 1)["exit"]["returncode"] == 4
    assert read(h.root_b / "worker-0b.result.json")["error"].startswith("KeyError") and (h.root_b / "rows" / "stage0b" / "INCOMPLETE.json").is_file()


def test_an_unexpected_exception_in_a0_outside_the_per_model_handlers_is_exit_4_with_a_failing_result(h):
    done = h.run(*h.a0_arguments(), scenario={"items_raise": True})
    assert done.returncode == 1
    result = read(h.root_a / "A0_RESULT.json")
    assert result["fatal"].startswith("other: RuntimeError") and result["verdict"]["passed"] is False
    assert list(result["models"]) == [admission.ALL_IDS[0]]                    # the loop stopped where its iterator failed
    assert exited(h, h.sha_a, "A0", 1)["exit"]["returncode"] == 4 and exited(h, h.sha_a, "A0", 1)["outcome"] == "failed"


def test_an_evidence_failure_never_stops_the_other_models_and_never_fails_a0_but_the_guard_refuses_the_stage(h):
    """N07 b and plan section 5 items 4 and 6: the saved forces of one model cannot be read. Its replay still counts, the
    other six models are still replayed and measured, the verdict is the frozen replay's, and the missing latency
    stops a stage that needs that model before anything is spent."""
    done = h.run(*h.a0_arguments(), scenario={"saved_forces_raise_for": ["direct-1"]})
    assert done.returncode == 0, done.stdout + done.stderr
    result = read(h.root_a / "A0_RESULT.json")
    assert set(result["models"]) == set(admission.ALL_IDS) and result["fatal"] is None and result["verdict"]["passed"] is True
    broken = result["models"]["direct-1"]
    assert broken["replay"]["status"] == "verified" and list(broken["evidence"]) == ["error"]
    assert broken["evidence"]["error"].startswith("KeyError") and "saved_force_abs_max" not in broken
    assert all("batch1_latency" in result["models"][m]["evidence"] for m in admission.ALL_IDS if m != "direct-1")
    stage = h.run(*h.stage_arguments("0", a0=h.a0_result_sha()))
    assert stage.returncode == 2 and "no usable batch-1 mean latency for direct-1" in stage.stdout
    assert not h.root_b.exists() and not list(h.registry.glob(f"{h.sha_b}-*"))


def test_an_a0_root_holding_a_stray_file_is_refused_before_a_worker_starts(h):
    launch_io.fresh_root(h.root_a, "A-e2e")
    (h.root_a / "notes.txt").write_text("a file this launch did not write\n")
    done = h.run(*h.a0_arguments())
    assert done.returncode == 2 and "notes.txt" in done.stdout
    assert not h.registry.exists() and not (h.root_a / "LAUNCH_LOG").exists() and not (h.root_a / "logs").exists()


def test_a0_restores_and_replays_the_models_in_the_admission_order(h):
    assert h.run(*h.a0_arguments()).returncode == 0
    assert (h.fixture_dir / "provider-order.txt").read_text().split() == list(admission.ALL_IDS)


@pytest.mark.parametrize("unit", ["A0", "0"])
def test_a_late_sigterm_during_finalization_does_not_degrade_the_result(h, unit):
    """The supervisor's SIGTERM can arrive while the worker is writing its result. Finalization ignores it, so the result
    is the ordinary one: no `finalization` fatal in A0, no `post_run_error` in a stage, exit code 0."""
    if unit == "A0":
        done = h.run(*h.a0_arguments(), scenario={"sigterm_at_exit": True})
        assert done.returncode == 0, done.stdout[-500:]
        result = read(h.root_a / "A0_RESULT.json")
        assert result["fatal"] is None and result["verdict"]["passed"] is True
    else:
        assert h.run(*h.a0_arguments()).returncode == 0
        done = h.run(*h.stage_arguments("0", a0=h.a0_result_sha()), scenario={"sigterm_at_exit": True})
        assert done.returncode == 0, done.stdout[-500:]
        result = read(h.root_b / "worker-0.result.json")
        assert result["ok"] is True and "post_run_error" not in result


def test_the_diff_gates_are_computed_before_the_spent_point_so_a_failure_there_spends_nothing(h):
    """N07 a: the worker's own gate computation is its second call (the first is the context check). Run after the spent
    write, an error there spent A0 and left no result file."""
    done = h.run(*h.a0_arguments(), scenario={"gates_raise_on_call": 2})
    assert done.returncode == 2, done.stdout
    assert not (h.root_a / "A0_RESTORE_BEGUN.json").exists() and not (h.root_a / "A0_RESULT.json").exists()
    assert "diff gates could not be computed" in read(h.root_a / "A0_REFUSED-1.json")["reason"]
    refused = exited(h, h.sha_a, "A0", 1)
    assert refused["outcome"] == "refused_before_start" and refused["spent"] is False
    again = h.run(*h.a0_arguments())                                                       # nothing was spent: it restarts
    assert again.returncode == 0, again.stdout + again.stderr


def test_a_stage_0b_restart_after_a_refusal_is_allowed_under_the_same_record(h):
    h.run_a0_and_stage0()
    a0, stage0 = h.a0_result_sha(), h.marker_sha("0")
    refused = h.run(*h.stage_arguments("0b", a0=a0, stage0=stage0), scenario={"fail_precondition_attempts": [1]})
    assert refused.returncode == 2
    assert (h.root_b / "worker-0b.attempt-1.result.json").is_file() and not (h.root_b / "rows" / "stage0b").exists()
    assert exited(h, h.sha_b, "0b", 1)["spent"] is False
    second = h.run(*h.stage_arguments("0b", a0=a0, stage0=stage0))
    assert second.returncode == 0, second.stdout + second.stderr
    assert exited(h, h.sha_b, "0b", 2)["outcome"] == "complete" and (h.root_b / "worker-0b.result.json").is_file()
    assert launch_state.EventLog(h.root_b).problems() == []


def test_typed_hashes_are_required_and_recomputed_and_nothing_is_spent_by_a_wrong_one(h):
    h.run_a0_and_stage0()
    a0, stage0 = h.a0_result_sha(), h.marker_sha("0")
    before = {p.name for p in h.registry.glob(f"{h.sha_b}-*")}
    cases = [(h.stage_arguments("0b", a0=a0), "typed marker hash of stage 0"),
             (h.stage_arguments("0b", a0=a0, stage0="0" * 64), "typed hash for stage 0"),
             (h.stage_arguments("0b", stage0=stage0), "--a0-result-sha256"),
             (h.stage_arguments("0b", a0="0" * 64, stage0=stage0), "different SHA-256"),
             (h.stage_arguments("2", a0=a0, stage1="0" * 64, stage0b=stage0), "a hash was typed for stage"),
             (h.stage_arguments("1", a0=a0), "typed marker hash of stage 0")]
    for argv, message in cases:
        done = h.run(*argv)
        assert done.returncode == 2 and message in done.stdout, (argv[:2], done.stdout)
    assert {p.name for p in h.registry.glob(f"{h.sha_b}-*")} == before


def test_an_a0_result_of_another_record_is_refused_even_with_its_own_hash(h):
    h.run_a0_and_stage0()
    result = read(h.root_a / "A0_RESULT.json")
    other = h.tmp / "other-a0"
    other.mkdir()
    result["record_sha256"] = "d" * 64
    data = launch_state.dumps(result)
    (h.root_a / "A0_RESULT.json").rename(h.root_a / "A0_RESULT.moved")
    (h.root_a / "A0_RESULT.json").write_bytes(data)
    done = h.run(*h.stage_arguments("0b", a0=launch_record.sha256_bytes(data), stage0=h.marker_sha("0")))
    assert done.returncode == 2 and "record_sha256" in done.stdout


# ---- Reconciliation, the lock and orphaned workers (A2, B4) ------------------------------------------------------------

def test_a_second_parent_is_refused_while_the_first_holds_the_roots_lock_and_spends_nothing(h):
    launch_io.fresh_root(h.root_a, "A-e2e")
    with launch_state.Lock(h.root_a):
        done = h.run(*h.a0_arguments())
    assert done.returncode == 2 and "another launch command holds" in done.stdout
    assert not list(h.registry.glob("*")) if h.registry.exists() else True
    assert not (h.root_a / "LAUNCH_LOG").exists()


def orphan(h, unit, *, stage_directory):
    from stage2_capped.launch import reconcile
    launch_io.fresh_root(h.root_b, "B-e2e")
    registry = launch_state.Registry(h.registry)
    registry.write(h.sha_b, unit, 1, "started", launch_id="B-e2e")
    if stage_directory:
        (h.root_b / "rows" / f"stage{unit}" / "model" / "cinf").mkdir(parents=True)
        (h.root_b / "rows" / f"stage{unit}" / "model" / "cinf" / "row.json").write_text("{}")
    return reconcile, registry


def test_a_dead_orphaned_worker_is_confirmed_spent_and_its_stage_marked_incomplete(h):
    reconcile, registry = orphan(h, "1", stage_directory=True)
    (h.root_b / "WORKER_PID-1-1.json").write_text(json.dumps({"pid": 2 ** 22 + 99, "start_time": "gone"}))
    assert reconcile(registry, h.root_b, h.sha_b, h.record_b) == []
    exited_event = exited(h, h.sha_b, "1", 1)
    assert exited_event["outcome"] == "confirmed_spent_after_missing_exit" and exited_event["spent"] is True
    assert (h.root_b / "rows" / "stage1" / "INCOMPLETE.json").is_file()
    assert registry.is_spent(h.sha_b, "1")


def test_a_live_orphaned_worker_blocks_the_next_command_and_nothing_is_confirmed_or_marked(h):
    reconcile, registry = orphan(h, "1", stage_directory=True)
    (h.root_b / "WORKER_PID-1-1.json").write_text(json.dumps(
        {"pid": os.getpid(), "start_time": launch_state.process_start(os.getpid())}))
    problems = reconcile(registry, h.root_b, h.sha_b, h.record_b)
    assert problems == ["the worker of 1 attempt 1 may still be running"]
    assert not (h.root_b / "rows" / "stage1" / "INCOMPLETE.json").exists()
    assert registry.attempts(h.sha_b, "1")[1]["exited"] is None


def test_an_orphaned_start_with_no_pid_file_and_no_stage_is_still_confirmed_as_spent_never_as_a_refusal(h):
    reconcile, registry = orphan(h, "0b", stage_directory=False)
    assert reconcile(registry, h.root_b, h.sha_b, h.record_b) == []
    assert exited(h, h.sha_b, "0b", 1)["spent"] is True and not (h.root_b / "rows").exists()


# ---- The lock, the registry and where a spent unit is remembered (N01, N02, N03; plan section 4) ------------------------

def test_a_second_parent_writes_nothing_while_another_holds_the_lock(h):
    """N01: a parent that cannot take the root's lock must not reconcile, confirm or mark anything. Here a `started`
    event has no pid file yet (the first parent's worker is still starting), exactly what a parent that reconciled
    before locking would wrongly confirm as spent after a missing exit."""
    launch_io.fresh_root(h.root_a, "A-e2e")
    launch_state.Registry(h.registry).write(h.sha_a, "A0", 1, "started", launch_id="A-e2e")
    before = sorted(p.name for p in h.registry.glob("*"))
    with launch_state.Lock(h.root_a):
        second = h.run(*h.a0_arguments())
    assert second.returncode == 2 and "another launch command holds" in second.stdout
    assert sorted(p.name for p in h.registry.glob("*")) == before
    assert not (h.root_a / "LAUNCH_LOG").exists()


def test_a_spent_unit_stays_spent_when_its_root_is_moved_away_and_when_the_record_is_copied(h, tmp_path):
    """N02 and plan section 10, 'a new root does not hide a spent unit': the registry, not the root, remembers a spent
    unit, and the registry is where the approved record says, not beside whichever copy of the record is run."""
    first = h.run(*h.a0_arguments(), scenario={"perturb": {"direct-0": 1e-9}})
    assert first.returncode == 1                                                          # A0 spent and failed
    h.root_a.rename(tmp_path / "outA-moved")
    names = sorted(p.name for p in h.registry.glob("*"))
    again = h.run(*h.a0_arguments())
    assert again.returncode == 2 and "already spent" in again.stdout
    assert sorted(p.name for p in h.registry.glob("*")) == names and not h.root_a.exists()
    other = tmp_path / "other-launch" / "records"
    other.mkdir(parents=True)
    for source in (h.path_a, h.approval_a):
        shutil.copy2(source, other / source.name)
    copied = h.run("a0", "--record", str(other / h.path_a.name), "--approval", str(other / h.approval_a.name),
                   "--record-sha256", h.sha_a, "--group9-root", str(h.group9))
    assert copied.returncode == 2 and "names the launch folder" in copied.stdout, copied.stdout[-400:]
    assert not (tmp_path / "other-launch" / "registry").exists() and not h.root_a.exists()


def test_a_stage_that_ends_failed_after_writing_its_marker_cannot_satisfy_the_next_stage(h):
    """N03 and the exit-time module check (M61, M146): the worker finishes its marker, then a project module is found
    to come from outside the worktree. The stage is FAILED in the registry, and stage 0b, which needs stage 0, is refused
    even though the marker is valid and its hash is typed correctly."""
    first = h.run(*h.a0_arguments())
    assert first.returncode == 0, first.stdout + first.stderr
    a0 = h.a0_result_sha()
    done = h.run(*h.stage_arguments("0", a0=a0), scenario={"module_outside_at_exit": True})
    assert done.returncode == 1 and "outside the worktree" not in done.stderr
    result = read(h.root_b / "worker-0.result.json")
    assert result["ok"] is False and result["module_problems_at_exit"] and "outside the worktree" in result["module_problems_at_exit"][0]
    assert exited(h, h.sha_b, "0", 1)["exit"]["returncode"] == 3 and exited(h, h.sha_b, "0", 1)["outcome"] == "failed"
    assert read(h.root_b / "rows" / "stage0" / "STAGE_COMPLETE.json")["complete"] is True      # the marker alone looks fine
    later = h.run(*h.stage_arguments("0b", a0=a0, stage0=h.marker_sha("0")))
    assert later.returncode == 2 and "stage 0 is not complete (failed)" in later.stdout
    assert not (h.root_b / "rows" / "stage0b").exists()


def test_a_stage_whose_result_cannot_be_completed_still_leaves_its_result_file(h):
    """N07 c: after the marker, the exit-time inventory raises. The result file is still written, naming the error, the
    exit code is nonzero and the parent calls the stage failed."""
    first = h.run(*h.a0_arguments())
    assert first.returncode == 0, first.stdout + first.stderr
    done = h.run(*h.stage_arguments("0", a0=h.a0_result_sha()), scenario={"raise_at_exit": True})
    assert done.returncode == 1
    result = read(h.root_b / "worker-0.result.json")
    assert result["ok"] is False and "module inventory could not be computed" in result["post_run_error"]
    assert exited(h, h.sha_b, "0", 1)["exit"]["returncode"] == 4 and exited(h, h.sha_b, "0", 1)["outcome"] == "failed"
    assert read(h.root_b / "rows" / "stage0" / "STAGE_COMPLETE.json")["complete"] is True


def test_an_a0_whose_finalization_fails_still_leaves_a_failing_result_file(h):
    """The A0 counterpart of N07 c: the exit-time inventory raises after the replay. The result is a minimal failing one,
    never a missing file, and the exit code is nonzero."""
    done = h.run(*h.a0_arguments(), scenario={"raise_at_exit": True})
    assert done.returncode == 1
    result = read(h.root_a / "A0_RESULT.json")
    assert result["fatal"].startswith("finalization: OSError") and result["verdict"]["passed"] is False
    assert exited(h, h.sha_a, "A0", 1)["exit"]["returncode"] == 4 and exited(h, h.sha_a, "A0", 1)["outcome"] == "failed"


def test_a_module_that_differs_at_exit_fails_a0_even_though_the_replay_passed(h):
    """M60: the exit-time module inventory of the A0 worker."""
    done = h.run(*h.a0_arguments(), scenario={"module_outside_at_exit": True})
    assert done.returncode == 1
    result = read(h.root_a / "A0_RESULT.json")
    assert result["module_problems_at_exit"] and result["verdict"]["passed"] is False
    assert all(result["models"][m]["replay"]["status"] == "verified" for m in admission.ALL_IDS)
    assert exited(h, h.sha_a, "A0", 1)["exit"]["returncode"] == 3 and exited(h, h.sha_a, "A0", 1)["outcome"] == "failed"


@pytest.mark.parametrize("unit", ["A0", "0"])
def test_the_parent_repeats_the_tree_check_after_the_worker_exits(h, unit):
    """M46: a file appears in the worktree during the run. The worker's own checks do not see it (it is no module); the
    parent's repeat of the tree check does, and calls the unit failed although the worker reported success."""
    if unit == "A0":
        done = h.run(*h.a0_arguments(), scenario={"dirty_worktree": True})
        assert read(h.root_a / "A0_RESULT.json")["verdict"]["passed"] is True
        assert exited(h, h.sha_a, "A0", 1)["exit"]["returncode"] == 0
        record_sha = h.sha_a
    else:
        assert h.run(*h.a0_arguments()).returncode == 0
        done = h.run(*h.stage_arguments("0", a0=h.a0_result_sha()), scenario={"dirty_worktree": True})
        assert read(h.root_b / "worker-0.result.json")["ok"] is True
        record_sha = h.sha_b
    assert done.returncode == 1 and "the working tree is not clean" in done.stdout, done.stdout[-500:]
    assert exited(h, record_sha, unit, 1)["outcome"] == "failed" and exited(h, record_sha, unit, 1)["spent"] is True


def test_a_stage_whose_admission_changed_since_a0_is_refused_before_anything_is_spent(h):
    """M148: the worker compares the identities `admit_all` returns now with the ones the A0 result holds."""
    assert h.run(*h.a0_arguments()).returncode == 0
    done = h.run(*h.stage_arguments("0", a0=h.a0_result_sha()), scenario={"drift_stage_identity": True})
    assert done.returncode == 2, done.stdout
    refusal = read(h.root_b / "worker-0.attempt-1.result.json")
    assert refusal["label"] == "refused_before_start" and "admission now differs" in refusal["reason"]
    assert not (h.root_b / "rows").exists() and exited(h, h.sha_b, "0", 1)["spent"] is False


def test_the_completion_decision_requires_the_audit_files_for_a0_and_for_a_stage(scratch, tmp_path):
    """The parent's `decide` hands the completion rules the files each kind of unit must have opened: the four dataset
    files for A0, all five for a stage. A worker whose hook recorded nothing (or too little) is not complete."""
    from types import SimpleNamespace
    from stage2_capped import driver, launch
    from stage2_capped.tests.launch_fixtures import make_a0_result, make_stage_directory
    h = Harness(scratch, tmp_path)
    tree, allowed = launch_checks.audit_scope(h.group9)
    four, five = launch_checks.required_relative(tree, allowed, "A0"), launch_checks.allowed_relative(tree, allowed)
    ok = {"returncode": 0, "timed_out": False}

    def a0_case(name, opened):
        root = tmp_path / name
        root.mkdir()
        result = make_a0_result()
        result.update({"audit": {"violations": 0, "opened_in_data_directory": opened}, "fatal": None, "module_problems_at_exit": []})
        (root / "A0_RESTORE_BEGUN.json").write_text("{}")
        (root / "A0_RESULT.json").write_bytes(launch_state.dumps(result))
        args = SimpleNamespace(record_sha256=h.sha_a, group9_root=str(h.group9))
        return launch.decide("A0", args, h.record_a, root, h.folder, ok, h.scratch)

    def stage_case(name, opened):
        root = tmp_path / name
        make_stage_directory(root, "0b", h.record_b, "B-e2e", driver.relaxer_sha256())
        (root / "worker-0b.result.json").write_text(json.dumps({"audit": {"violations": 0, "opened_in_data_directory": opened}}))
        args = SimpleNamespace(record_sha256=h.sha_b, group9_root=str(h.group9))
        return launch.decide("0b", args, h.record_b, root, h.folder, ok, h.scratch)

    assert a0_case("a0-four", four)[:2] == (True, "complete") and a0_case("a0-five", five)[:2] == (True, "complete")
    assert a0_case("a0-none", [])[:2] == (False, "failed") and a0_case("a0-three", four[:3])[:2] == (False, "failed")
    assert stage_case("stage-five", five)[:2] == (True, "complete")
    assert stage_case("stage-none", [])[:2] == (False, "failed") and stage_case("stage-four", four)[:2] == (False, "failed")


# ---- Approval, ceilings and the guard at launch level --------------------------------------------------------------------

@pytest.mark.parametrize("damage", ["draft", "missing"])
def test_a_launch_without_a_valid_approval_is_refused_and_creates_nothing(h, damage):
    if damage == "missing":
        h.approval_a.unlink()
    else:
        body = read(h.approval_a)
        body["status"] = "draft"
        h.approval_a.write_text(json.dumps(body))
    done = h.run(*h.a0_arguments())
    assert done.returncode == 2 and "approval" in done.stdout
    assert not h.root_a.exists() and not h.registry.exists()


def test_the_supervisor_and_the_guard_use_the_ceilings_of_the_record_not_the_spec_floor(scratch, tmp_path):
    """M77 and M72: the record raised the A0 ceiling to 400 s and stage 0's to 700 s (both inside floor and cap). The
    parent must supervise with those, and test the guard against the stage's ceiling, not A0's. The guard is also asked
    for every stage when A0 ends (its printed summary), each with that stage's ceiling."""
    from stage2_capped import launch_record
    ceilings = launch_record.expected_ceilings()
    ceilings["a0_wall_seconds"], ceilings["stages"]["0"]["wall_seconds"] = 400.0, 700.0
    h = Harness(scratch, tmp_path, ceilings=ceilings)
    spy = tmp_path / "spy.jsonl"
    assert h.run(*h.a0_arguments(), scenario={"spy_file": str(spy)}).returncode == 0
    done = h.run(*h.stage_arguments("0", a0=h.a0_result_sha()), scenario={"spy_file": str(spy)})
    assert done.returncode == 0, done.stdout + done.stderr
    lines = [json.loads(line) for line in spy.read_text().splitlines()]
    printed_at_a0_end = [{"what": "guard", "stage": stage, "ceiling": ceiling}          # the guard shown for every stage when A0 ends
                         for stage, ceiling in (("0", 700.0), ("0b", 300.0), ("1", 2100.0), ("2", 900.0))]
    assert lines == [{"what": "supervise", "wall": 400.0}, *printed_at_a0_end,
                     {"what": "guard", "stage": "0", "ceiling": 700.0}, {"what": "supervise", "wall": 700.0}]


def test_the_started_events_name_the_files_the_parents_checks_opened_and_the_ledger_it_derived(scratch, tmp_path):
    h = Harness(scratch, tmp_path)
    assert h.run(*h.a0_arguments()).returncode == 0
    tree, allowed = launch_checks.audit_scope(h.group9)
    four = launch_checks.required_relative(tree, allowed, "A0")
    started = read(h.registry / f"{h.sha_a}-A0-1-started.json")
    assert started["model_free_check_opened"]["opened_in_data_directory"] == four
    assert started["model_free_check_opened"]["violations"] == 0 and started["ledger_snapshot"] is None
    first_log = read(launch_state.EventLog(h.root_a).files()[0])
    assert first_log["model_free_check_opened"] == started["model_free_check_opened"]


# ---- Reconciliation through the real parent (A2, B4) ------------------------------------------------------------------------

def stage_orphan(h, pid_record):
    """Record B's root with a started stage 1 that never exited, its stage directory and a worker pid file."""
    launch_io.fresh_root(h.root_b, "B-e2e")
    launch_state.Registry(h.registry).write(h.sha_b, "1", 1, "started", launch_id="B-e2e")
    row = h.root_b / "rows" / "stage1" / "model" / "cinf" / "row.json"
    row.parent.mkdir(parents=True)
    row.write_text("{}")
    if pid_record is not None:
        (h.root_b / "WORKER_PID-1-1.json").write_text(json.dumps(pid_record))
    return h.run(*h.stage_arguments("1", a0="0" * 64, stage0="0" * 64))


def test_the_next_command_confirms_a_dead_orphan_as_spent_and_marks_its_stage_incomplete(h):
    """R01: the confirmation is part of the command, not only a function: even a command that is going to be refused
    (here the A0 evidence is missing) reconciles the orphan under the lock first."""
    done = stage_orphan(h, {"pid": 2 ** 22 + 77, "start_time": "gone"})
    assert done.returncode == 2 and "already spent" in done.stdout
    event = exited(h, h.sha_b, "1", 1)
    assert event["outcome"] == "confirmed_spent_after_missing_exit" and event["spent"] is True
    assert "supervisor" in read(h.root_b / "rows" / "stage1" / "INCOMPLETE.json")["reason"]


def test_the_next_command_is_refused_while_an_orphaned_worker_may_still_be_running(h):
    """R02: a live worker blocks the command and nothing is confirmed or marked."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
    try:
        done = stage_orphan(h, {"pid": child.pid, "start_time": launch_state.process_start(child.pid)})
    finally:
        child.kill()
        child.wait()
    assert done.returncode == 2 and "the worker of 1 attempt 1 may still be running" in done.stdout
    assert not (h.registry / f"{h.sha_b}-1-1-exited.json").exists()
    assert not (h.root_b / "rows" / "stage1" / "INCOMPLETE.json").exists()


# ---- Workers started by hand, SIGTERM at the ceiling (the real handlers) -------------------------------------------------------

def a0_worker_command(h, kind, *extra):
    return [sys.executable, "-m", "stage2_capped.tests.launch_test_worker", "--kind", kind, "--record",
            str(h.path_a if kind == "a0" else h.path_b), "--approval", str(h.approval_a if kind == "a0" else h.approval_b),
            "--record-sha256", h.sha_a if kind == "a0" else h.sha_b, "--attempt", "1", "--group9-root", str(h.group9), *extra]


def wait_for(path, seconds=240):
    deadline = time.time() + seconds
    while not path.exists():
        assert time.time() < deadline, f"{path} did not appear"
        time.sleep(0.1)


def terminate_and_collect(process):
    try:
        process.send_signal(signal.SIGTERM)
        code = process.wait(timeout=90)
    finally:
        if process.poll() is None:
            process.kill()
    return code, process.stdout.read()


def test_a_started_event_that_names_another_parent_does_not_start_an_a0_worker(h):
    """M99: the `started` event must name this process's parent."""
    launch_io.fresh_root(h.root_a, "A-e2e")
    launch_state.EventLog(h.root_a).append("started", unit="A0", attempt=1, parent_pid=1)
    done = subprocess.run(a0_worker_command(h, "a0"), cwd=h.scratch, env=h.environment(), capture_output=True, text=True)
    assert done.returncode == 2
    assert "names this process's parent" in read(h.root_a / "A0_REFUSED-1.json")["reason"]
    assert not (h.root_a / "A0_RESTORE_BEGUN.json").exists()


def test_a_sigterm_stops_an_a0_worker_with_a_labelled_result_that_is_written(h):
    """W05, W07: the A0 worker turns the supervisor's SIGTERM into a labelled stop and still writes its result."""
    launch_io.fresh_root(h.root_a, "A-e2e")
    launch_state.EventLog(h.root_a).append("started", unit="A0", attempt=1, parent_pid=os.getpid())
    process = subprocess.Popen(a0_worker_command(h, "a0"), cwd=h.scratch, env=h.environment({"sleep_in_provider": 300}),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    wait_for(h.root_a / "A0_RESTORE_BEGUN.json")
    code, output = terminate_and_collect(process)
    assert code == 3, output
    result = read(h.root_a / "A0_RESULT.json")
    assert result["fatal"] == "deadline: terminated by the supervisor" and result["verdict"]["passed"] is False
    assert list(result["models"]) == [admission.ALL_IDS[0]] and result["models"][admission.ALL_IDS[0]]["failure"].startswith("deadline")
    assert result["audit"]["violations"] == 0


def test_a_sigterm_stops_a_stage_worker_with_a_labelled_result_and_an_incomplete_marker(h):
    """W06, W12, T01: the same for a stage worker. The marker says INCOMPLETE and the result file is written."""
    assert h.run(*h.a0_arguments()).returncode == 0
    launch_io.fresh_root(h.root_b, "B-e2e")
    launch_state.EventLog(h.root_b).append("started", unit="0", attempt=1, parent_pid=os.getpid())
    directory = h.root_b / "rows" / "stage0"
    scenario = {"sleep_in_stage_force": 300, "stage_directory": str(directory)}
    process = subprocess.Popen(a0_worker_command(h, "stage", "--unit", "0", "--a0-result-sha256", h.a0_result_sha()),
                               cwd=h.scratch, env=h.environment(scenario), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True)
    wait_for(directory)
    time.sleep(1.0)
    code, output = terminate_and_collect(process)
    assert code == 3, output
    result = read(h.root_b / "worker-0.result.json")
    assert result["ok"] is False and result["error"].startswith("DeadlineReached") and result["audit"]["violations"] == 0
    assert (directory / "INCOMPLETE.json").is_file()


def test_the_supervisor_stops_a_slow_a0_at_its_wall_ceiling_and_the_unit_is_failed_with_its_result(h):
    """The whole path: the real supervisor sends SIGTERM at the ceiling (here 12 s instead of 300 s), the worker's handler
    labels the stop and writes its result, and the parent calls the unit failed because it timed out."""
    done = h.run(*h.a0_arguments(), scenario={"sleep_in_provider": 300, "wall_override": 12, "finalization": 30})
    assert done.returncode == 1, done.stdout[-600:]
    record = exited(h, h.sha_a, "A0", 1)
    assert record["outcome"] == "failed" and record["exit"]["timed_out"] is True and record["exit"]["returncode"] == 3
    assert read(h.root_a / "A0_RESULT.json")["fatal"].startswith("deadline")
    assert "stopped at the wall ceiling" in done.stdout


# ---- Second fix pass: exit 1 only after the `started` event (R2-N01) -----------------------------------------------------------

@pytest.mark.parametrize("head", [["check", "a0"], ["a0"]])
def test_a_missing_record_file_is_a_refusal_with_exit_2_not_a_traceback(h, tmp_path, head):
    """Exit 1 means `spent`; a typo in --record spends nothing. The real command line, in a subprocess."""
    done = h.run_production(*head, "--record", str(tmp_path / "nowhere" / "records" / "RECORD_A.json"), "--approval",
                            str(h.approval_a), "--record-sha256", h.sha_a, "--group9-root", str(h.group9))
    assert done.returncode == 2 and "refused: FileNotFoundError" in done.stdout and "Traceback" not in done.stderr
    assert not h.root_a.exists() and not h.registry.exists()


def test_saved_files_that_are_missing_are_a_refusal_with_exit_2_not_a_traceback(h, tmp_path):
    mirror = tmp_path / "g9"
    (mirror / "reports" / "execution").mkdir(parents=True)
    destination = mirror / admission.RECOVERY_RECORDS
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(h.group9 / admission.RECOVERY_RECORDS, destination)             # the launch records are there, the saved results are not
    done = h.run_production("check", "a0", "--record", str(h.path_a), "--approval", str(h.approval_a),
                            "--record-sha256", h.sha_a, "--group9-root", str(mirror))
    assert done.returncode == 2 and done.stdout.startswith("refused: ") and "Traceback" not in done.stderr, done.stdout + done.stderr[-400:]
    assert not h.root_a.exists() and not h.registry.exists()


def test_an_a0_result_that_is_not_json_is_a_refusal_with_exit_2_not_a_traceback(h):
    h.root_a.mkdir()
    garbage = b"this is not json\n"
    (h.root_a / "A0_RESULT.json").write_bytes(garbage)
    done = h.run_production("check", "stage", "0", "--record", str(h.path_b), "--approval", str(h.approval_b),
                            "--record-sha256", h.sha_b, "--group9-root", str(h.group9),
                            "--a0-result-sha256", launch_record.sha256_bytes(garbage))
    assert done.returncode == 2 and "refused: JSONDecodeError" in done.stdout and "Traceback" not in done.stderr
    assert not h.root_b.exists() and not h.registry.exists()


def test_a_root_made_by_another_parent_meanwhile_is_a_refusal_with_exit_2(h):
    """The `fresh_root` race: this parent saw no root, another created it before this one did."""
    done = h.run(*h.a0_arguments(), scenario={"before_lock": {"create_root": True}})
    assert done.returncode == 2 and "refused: FileExistsError" in done.stdout and "Traceback" not in done.stderr
    assert not h.registry.exists() and sorted(p.name for p in h.root_a.iterdir()) == ["LAUNCH.json"]


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_a_lock_file_that_cannot_be_opened_is_a_refusal_with_exit_2_not_a_traceback(h):
    launch_io.fresh_root(h.root_a, "A-e2e")
    h.root_a.chmod(0o555)
    try:
        done = h.run(*h.a0_arguments())
    finally:
        h.root_a.chmod(0o755)
    assert done.returncode == 2 and "refused: PermissionError" in done.stdout and "Traceback" not in done.stderr
    assert not h.registry.exists() and sorted(p.name for p in h.root_a.iterdir()) == ["LAUNCH.json"]


def test_any_other_exception_in_the_checks_before_the_started_event_is_a_refusal_with_exit_2(h):
    """A worker pid file that is not JSON makes reconciliation raise, under the lock, before anything is written."""
    launch_io.fresh_root(h.root_b, "B-e2e")
    launch_state.Registry(h.registry).write(h.sha_b, "1", 1, "started", launch_id="B-e2e")
    (h.root_b / "WORKER_PID-1-1.json").write_text("not json {")
    before = sorted(p.name for p in h.registry.glob("*"))
    done = h.run(*h.stage_arguments("1", a0="0" * 64, stage0="0" * 64))
    assert done.returncode == 2 and "refused: JSONDecodeError" in done.stdout and "Traceback" not in done.stderr
    assert sorted(p.name for p in h.registry.glob("*")) == before


@pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
def test_a_registry_that_cannot_be_written_is_a_refusal_because_the_started_event_never_got_there(h):
    """The `started` event goes to the root's log first and to the registry second; the registry is the authority and the
    flag that makes a failure exit 1 is set only when the registry holds the event. A registry that refuses the write
    before linking its file spends nothing: exit 2, no worker, and the same record starts again once the registry can be
    written (the case where the file is linked and the write then raises is the next test)."""
    h.registry.mkdir(parents=True)
    h.registry.chmod(0o555)
    try:
        done = h.run(*h.a0_arguments())
    finally:
        h.registry.chmod(0o755)
    assert done.returncode == 2 and "refused: PermissionError" in done.stdout and "Traceback" not in done.stderr
    assert list(h.registry.glob("*")) == [] and not list(h.root_a.glob("WORKER_PID-*"))
    assert not (h.fixture_dir / "provider-order.txt").exists()                           # no worker ran
    again = h.run(*h.a0_arguments())
    assert again.returncode == 0, again.stdout + again.stderr


def test_a_registry_write_that_links_its_file_and_then_raises_is_a_spent_unit_not_a_refusal(h):
    """R3-N01. The registry write of the `started` event links its file and then raises. The registry holds `started`
    (a start with no exit counts as spent), so the command may not report a refusal that spent nothing: exit 1, no
    `refused:` line, no worker, and the next command says the unit is spent."""
    done = h.run(*h.a0_arguments(), scenario={"registry_raises_after_started_write": True})
    assert (h.registry / f"{h.sha_a}-A0-1-started.json").exists(), "the scenario did not leave the event in the registry"
    assert done.returncode == 1 and "refused:" not in done.stdout and "Traceback" in done.stderr
    assert not list(h.root_a.glob("WORKER_PID-*")) and not (h.fixture_dir / "provider-order.txt").exists()   # no worker ran
    again = h.run(*h.a0_arguments())
    assert again.returncode == 2 and "unit A0 is already spent under this record" in again.stdout


def test_a_worker_that_cannot_be_run_is_a_refusal_that_spends_nothing_and_a_restart_works(h):
    """The `started` event is written, then the supervisor cannot start the worker at all: no spent point was reached, so
    the exit is 2, the registry says `refused_before_start` and the same record starts again."""
    done = h.run(*h.a0_arguments(), scenario={"supervise_raises": True})
    assert done.returncode == 2 and "the worker could not be run: OSError" in done.stdout and "Traceback" not in done.stderr
    refused = exited(h, h.sha_a, "A0", 1)
    assert refused["outcome"] == "refused_before_start" and refused["spent"] is False
    assert "scenario: the worker could not be started" in refused["exit"]["supervisor_error"]
    again = h.run(*h.a0_arguments())
    assert again.returncode == 0, again.stdout + again.stderr
    assert exited(h, h.sha_a, "A0", 2)["outcome"] == "complete"


def test_an_exception_after_the_started_event_is_still_a_traceback_with_exit_1_never_a_refusal(h):
    """After `started` the unit is spent, whatever happens to the parent: here its own printing fails after the exit was
    recorded. The exit stays 1 and the registry stays as the decision left it."""
    done = h.run(*h.a0_arguments(), scenario={"print_a0_raises": True})
    assert done.returncode == 1 and "RuntimeError: scenario: printing failed" in done.stderr and "refused:" not in done.stdout
    assert exited(h, h.sha_a, "A0", 1)["outcome"] == "complete" and exited(h, h.sha_a, "A0", 1)["spent"] is True


# ---- Second fix pass: the lock does not make the checks fresh by itself (R2-N02) --------------------------------------------------

def test_a_parent_whose_checks_went_stale_before_it_got_the_lock_refuses_a_unit_spent_meanwhile(h):
    """The parent gathers its checks against a root and a registry that are clean; before it has the lock, another parent
    runs the unit. Without the second check under the lock this parent would start attempt 2 of a spent unit."""
    launch_io.fresh_root(h.root_a, "A-e2e")
    done = h.run(*h.a0_arguments(), scenario={"before_lock": {"spent_unit": "A0"}})
    assert done.returncode == 2 and "unit A0 is already spent under this record" in done.stdout
    assert sorted(p.name for p in h.registry.glob("*")) == [f"{h.sha_a}-A0-1-exited.json", f"{h.sha_a}-A0-1-started.json"]
    assert not (h.root_a / "LAUNCH_LOG").exists() and not (h.root_a / "logs").exists()
    assert not list(h.root_a.glob("WORKER_PID-*")) and not (h.root_a / "A0_RESTORE_BEGUN.json").exists()


def test_a_parent_whose_checks_went_stale_before_it_got_the_lock_refuses_a_root_that_changed_meanwhile(h):
    launch_io.fresh_root(h.root_a, "A-e2e")
    done = h.run(*h.a0_arguments(), scenario={"before_lock": {"root_file": "A0_RESTORE_BEGUN.json"}})
    assert done.returncode == 2 and "A0_RESTORE_BEGUN.json" in done.stdout and "may not find" in done.stdout
    assert not h.registry.exists() and not (h.root_a / "LAUNCH_LOG").exists() and not (h.root_a / "logs").exists()
    assert not list(h.root_a.glob("WORKER_PID-*"))


# ---- Second fix pass: the start-of-run module check (R2-N04, M62) ------------------------------------------------------------------

@pytest.mark.parametrize("unit", ["A0", "0"])
def test_a_worker_that_finds_a_module_outside_the_worktree_at_start_refuses_before_the_spent_point(h, unit):
    if unit == "A0":
        done = h.run(*h.a0_arguments(), scenario={"module_outside_at_start": True})
        refusal, spent_signal, sha = h.root_a / "A0_REFUSED-1.json", h.root_a / "A0_RESTORE_BEGUN.json", h.sha_a
    else:
        assert h.run(*h.a0_arguments()).returncode == 0
        done = h.run(*h.stage_arguments("0", a0=h.a0_result_sha()), scenario={"module_outside_at_start": True})
        refusal, spent_signal, sha = h.root_b / "worker-0.attempt-1.result.json", h.root_b / "rows", h.sha_b
    assert done.returncode == 2, done.stdout[-600:]
    assert "outside the worktree" in read(refusal)["reason"] and not spent_signal.exists()
    assert exited(h, sha, unit, 1)["outcome"] == "refused_before_start" and exited(h, sha, unit, 1)["spent"] is False


def test_an_a0_started_with_src_from_another_checkout_is_refused_before_the_spent_point(h, tmp_path):
    """M62 with a real PYTHONPATH mistake. A second clean checkout of the same commit that supplies everything would pass
    (the worktree is derived from the imported `driver`); a mixed path does not: the project modules come from two trees."""
    other = tmp_path / "other"
    subprocess.run(["git", "clone", "-q", str(h.scratch), str(other)], check=True, capture_output=True)
    mixed = os.pathsep.join([str(h.scratch), str(other / "src"), str(h.scratch / "scripts")])
    original = h.environment

    def environment(scenario=None):
        env = original(scenario)
        env["PYTHONPATH"] = mixed
        return env
    h.environment = environment
    done = h.run(*h.a0_arguments())
    assert done.returncode == 2, done.stdout[-600:] + done.stderr[-600:]
    reason = read(h.root_a / "A0_REFUSED-1.json")["reason"]
    assert "outside the worktree" in reason and str(other) in reason
    assert not (h.root_a / "A0_RESTORE_BEGUN.json").exists() and exited(h, h.sha_a, "A0", 1)["spent"] is False


# ---- Second fix pass: the post-A0 guard is printed for every stage when A0 ends ---------------------------------------------------------

GUARD_LINE = re.compile(r"^  post-A0 guard, stage (\S+): models ([^;]+); mean batch-1 latency used (.+); ceiling (\d+) s; (OK|REFUSED: .+)$")


def guard_lines(output):
    return {m.group(1): m for m in (GUARD_LINE.match(line) for line in output.splitlines()) if m}


def test_a_passing_a0_prints_the_guard_for_every_stage_with_its_models_latency_and_ceiling(h):
    done = h.run(*h.a0_arguments())
    assert done.returncode == 0 and "post-A0 guard (record A's ceilings; a stage command applies record B's):" in done.stdout
    lines = guard_lines(done.stdout)
    assert list(lines) == list(driver.STAGE_SPECS)
    for stage, spec in driver.STAGE_SPECS.items():
        line = lines[stage]
        assert line.group(2) == ", ".join(spec["models"]) and int(line.group(4)) == spec["wall_seconds"]
        assert line.group(5) == "OK" and re.fullmatch(r"[0-9.e-]+ s \(([^)]+)\)", line.group(3))
        assert re.fullmatch(r"[0-9.e-]+ s \(([^)]+)\)", line.group(3)).group(1) in spec["models"]


def test_an_a0_with_a_missing_latency_prints_the_stages_that_need_that_model_as_refused(h):
    done = h.run(*h.a0_arguments(), scenario={"saved_forces_raise_for": ["direct-1"]})
    assert done.returncode == 0, done.stdout + done.stderr                      # the frozen replay passed: A0 passes (N07 b)
    lines = guard_lines(done.stdout)
    assert list(lines) == list(driver.STAGE_SPECS)
    for stage, spec in driver.STAGE_SPECS.items():
        if "direct-1" in spec["models"]:
            assert lines[stage].group(3) == "none (no usable mean for direct-1)", stage
            assert lines[stage].group(5) == "REFUSED: no usable batch-1 mean latency for direct-1 in the A0 result"
        else:
            assert lines[stage].group(5) == "OK", stage


def test_an_a0_that_fails_prints_that_the_guard_does_not_apply(h):
    done = h.run(*h.a0_arguments(), scenario={"perturb": {"direct-0": 1e-9}})
    assert done.returncode == 1 and "post-A0 guard: not applicable, A0 did not complete" in done.stdout
    assert not guard_lines(done.stdout)


# ---- Second fix pass: tests the mutation check found missing (X04, X10, X60) -----------------------------------------------------------

def test_a_root_this_launch_did_not_make_is_refused_without_a_lock_file_going_into_it(h):
    """X04: a directory with no LAUNCH.json for this launch id is refused before the lock, so no `.launch.lock` and no log
    file lands in a folder that is not ours."""
    h.root_a.mkdir()
    (h.root_a / "notes.txt").write_text("a folder this launch did not make\n")
    done = h.run(*h.a0_arguments())
    assert done.returncode == 2 and "LAUNCH.json" in done.stdout
    assert [p.name for p in h.root_a.iterdir()] == ["notes.txt"] and not h.registry.exists()


def test_the_next_command_confirms_a_dead_orphaned_a0_as_spent(h):
    """X10: reconciliation covers A0 as it covers a stage. The orphan's worker is gone, so the unit is confirmed spent and
    the command is refused as already spent."""
    launch_io.fresh_root(h.root_a, "A-e2e")
    launch_state.Registry(h.registry).write(h.sha_a, "A0", 1, "started", launch_id="A-e2e")
    (h.root_a / "WORKER_PID-A0-1.json").write_text(json.dumps({"pid": 2 ** 22 + 55, "start_time": "gone"}))
    done = h.run(*h.a0_arguments())
    assert done.returncode == 2 and "already spent" in done.stdout
    event = exited(h, h.sha_a, "A0", 1)
    assert event["outcome"] == "confirmed_spent_after_missing_exit" and event["spent"] is True
    assert not (h.root_a / "rows").exists()


def test_a_live_orphaned_a0_worker_blocks_the_next_command_and_nothing_is_confirmed(h):
    launch_io.fresh_root(h.root_a, "A-e2e")
    launch_state.Registry(h.registry).write(h.sha_a, "A0", 1, "started", launch_id="A-e2e")
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"], start_new_session=True)
    try:
        (h.root_a / "WORKER_PID-A0-1.json").write_text(json.dumps(
            {"pid": child.pid, "start_time": launch_state.process_start(child.pid)}))
        done = h.run(*h.a0_arguments())
    finally:
        child.kill()
        child.wait()
    assert done.returncode == 2 and "the worker of A0 attempt 1 may still be running" in done.stdout
    assert not (h.registry / f"{h.sha_a}-A0-1-exited.json").exists()


def test_a_stop_during_the_evidence_step_is_a_labelled_deadline_and_no_later_model_runs(h):
    """X60: SIGTERM arrives while the first model's evidence is being collected (the saved forces are being read). The stop
    is labelled, the result is written, and the remaining models are not replayed."""
    launch_io.fresh_root(h.root_a, "A-e2e")
    launch_state.EventLog(h.root_a).append("started", unit="A0", attempt=1, parent_pid=os.getpid())
    process = subprocess.Popen(a0_worker_command(h, "a0"), cwd=h.scratch, env=h.environment({"sleep_in_saved_forces": 300}),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    wait_for(h.fixture_dir / "saved-forces-reached.txt")
    code, output = terminate_and_collect(process)
    assert code == 3, output
    result = read(h.root_a / "A0_RESULT.json")
    first = admission.ALL_IDS[0]
    assert result["fatal"] == "deadline: terminated by the supervisor" and result["verdict"]["passed"] is False
    assert list(result["models"]) == [first] and result["models"][first]["replay"]["status"] == "verified"
    assert "evidence" not in result["models"][first] and result["audit"]["violations"] == 0


# ---- Second fix pass: plan section 10, the two tests that were still missing ----------------------------------------------------------------

def test_a_stage_0_worker_that_dies_between_the_stop_and_the_verdict_leaves_no_verdict_and_no_second_rerun(h):
    """Plan section 10: the single frozen re-run is in-process state. A worker killed after STAGE0_RERUN_ATTEMPTED.json and
    before the verdict leaves no verdict; the stage is spent and failed, the registry says so, and no later command can
    re-run (R1: the parent's registry outcome is the authority)."""
    assert h.run(*h.a0_arguments()).returncode == 0
    a0 = h.a0_result_sha()
    scenario = {"mismatch": True, "kill_in_rerun": True, "rerun_marker": str(h.root_b / "STAGE0_RERUN_ATTEMPTED.json")}
    done = h.run(*h.stage_arguments("0", a0=a0), scenario=scenario)
    assert done.returncode == 1 and "STAGE 0 INCOMPLETE, not a result" in done.stdout
    assert (h.root_b / "STAGE0_RERUN_ATTEMPTED.json").is_file()
    assert not (h.root_b / "STAGE0_RERUN.json").exists() and not (h.root_b / "worker-0.result.json").exists()
    record = exited(h, h.sha_b, "0", 1)
    assert record["exit"]["returncode"] == -9 and record["outcome"] == "failed" and record["spent"] is True
    assert (h.root_b / "rows" / "stage0" / "INCOMPLETE.json").is_file()
    again = h.run(*h.stage_arguments("0", a0=a0))
    assert again.returncode == 2 and "already spent" in again.stdout
    assert not (h.root_b / "STAGE0_RERUN.json").exists()
    assert h.run(*h.stage_arguments("0b", a0=a0, stage0="0" * 64)).returncode == 2


def test_the_a0_worker_runs_the_frozen_replay_with_the_expected_record_batches_and_five_single_calls(h):
    """The frozen `adapter_replay` is hash-pinned; this test shows the worker calls it as it must: once per model in the
    admission order, with that model's score directory, the data directory and the split hash, and that the arm then sees
    exactly the calls the frozen function makes: batches of four over the 974 validation records (243 of four and one of
    two), then the five single-molecule calls, before the evidence step makes its own."""
    from e3x_phase_common import SPLIT
    spy = h.fixture_dir / "calls.jsonl"
    done = h.run(*h.a0_arguments(), scenario={"spy_calls_file": str(spy)})
    assert done.returncode == 0, done.stdout + done.stderr
    lines = [json.loads(line) for line in spy.read_text().splitlines()]
    records = admission.REPLAY_RECORDS
    batches = [[min(4, records - start), 3, 3] for start in range(0, records, 4)]
    assert len(batches) == 244 and batches[:243] == [[4, 3, 3]] * 243 and batches[243] == [2, 3, 3]
    expected_calls = batches + [[3, 3]] * 5
    replays = [line["replay"] for line in lines if "replay" in line]
    assert [Path(r["score_directory"]).name for r in replays] == [f"score-{m}" for m in admission.ALL_IDS]
    assert all(r["data_directory"] == str(admission.data_directory(h.group9)) and r["split_hash"] == SPLIT for r in replays)
    during_replay, current, replaying = {}, None, False
    for line in lines:
        if "replay" in line:
            current, replaying = Path(line["replay"]["score_directory"]).name[len("score-"):], True
            during_replay[current] = []
        elif line.get("evidence_begins"):
            replaying = False
        elif replaying:
            assert line["model"] == current
            during_replay[current].append(line["shape"])
    assert list(during_replay) == list(admission.ALL_IDS)
    assert all(calls == expected_calls for calls in during_replay.values())
