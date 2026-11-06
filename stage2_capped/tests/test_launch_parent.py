"""Plan v3 sections 3 to 7 and 11 with the amendments: the real parent supervising the real workers (fixture
arms, no checkpoint) from a committed scratch tree, through the real entry points, as a subprocess.

The accepted study inputs and saved results are read for real; only the models are fixtures. Stage 0 compares
the fixture runs with 'saved' results built from the same fixture arms, so it can complete and Stage 1's control
wiring runs."""
import json
import subprocess
import sys

import pytest

from stage2_capped import launch_checks, launch_record, launch_state
from stage2_capped.tests.launch_fixtures import needs_real_files, scratch, scratch_base  # noqa: F401
from stage2_capped.tests.launch_harness import Harness

pytestmark = needs_real_files


@pytest.fixture
def harness(scratch, tmp_path):
    return Harness(scratch, tmp_path)


def read(path):
    return json.loads(path.read_text())


# ---- A complete launch ---------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def complete(scratch_base, tmp_path_factory):
    import shutil
    base = tmp_path_factory.mktemp("complete")
    tree = base / "tree"
    shutil.copytree(scratch_base, tree, symlinks=True)
    h = Harness(tree, base)
    runs = {"check": h.run(*h.a0_arguments("check")), "a0": h.run(*h.a0_arguments())}
    runs["stage0"] = h.run(*h.stage_arguments("0", a0=h.a0_result_sha()))
    a0 = h.a0_result_sha()
    runs["stage0b"] = h.run(*h.stage_arguments("0b", a0=a0, stage0=h.marker_sha("0")))
    runs["stage1"] = h.run(*h.stage_arguments("1", a0=a0, stage0=h.marker_sha("0")))
    runs["stage2"] = h.run(*h.stage_arguments("2", a0=a0, stage1=h.marker_sha("1"), stage0b=h.marker_sha("0b")))
    return h, runs


def test_a_complete_launch_runs_a0_and_the_four_stages_through_the_parent(complete):
    h, runs = complete
    for name, done in runs.items():
        assert done.returncode == 0, (name, done.stdout[-800:], done.stderr[-800:])
    assert "A0 verdict: PASS" in runs["a0"].stdout
    for stage, runs_expected in (("0", 80), ("0b", 32), ("1", 320), ("2", 128)):
        directory = h.root_b / "rows" / f"stage{stage}"
        marker = read(directory / "STAGE_COMPLETE.json")
        assert marker["complete"] is True and marker["runs"] == runs_expected and not (directory / "INCOMPLETE.json").exists()
        assert len(list(directory.glob("*/*/*.json"))) == runs_expected
        assert f"stage {stage}: complete; runs {runs_expected}" in runs[f"stage{stage}"].stdout


def test_the_check_command_wrote_nothing(complete):
    h, runs = complete
    assert "check passed: nothing was written" in runs["check"].stdout


def test_every_unit_has_a_started_and_an_exited_event_and_the_logs_chain(complete):
    h, _ = complete
    names = {name for name, _ in h.events()}
    for sha, units in ((h.sha_a, ["A0"]), (h.sha_b, ["0", "0b", "1", "2"])):
        for unit in units:
            assert f"{sha}-{unit}-1-started.json" in names and f"{sha}-{unit}-1-exited.json" in names
            assert read(h.registry / f"{sha}-{unit}-1-exited.json")["outcome"] == "complete"
    for root in (h.root_a, h.root_b):
        assert launch_state.EventLog(root).problems() == [] and len(launch_state.EventLog(root).files()) == 2 * (1 if root == h.root_a else 4)


def test_the_a0_result_holds_the_evidence_the_plan_lists(complete):
    h, _ = complete
    result = read(h.root_a / "A0_RESULT.json")
    assert result["record_sha256"] == h.sha_a and result["launch_id"] == "A-e2e" and result["commit"] == h.record_a["commit"]
    assert result["verdict"]["passed"] is True and launch_state.a0_verdict(result)[0]
    assert set(result["models"]) == set(__import__("stage2_capped.admission", fromlist=["x"]).ALL_IDS)
    assert result["runtime"]["jax_backend"] == "cpu" and result["audit"]["violations"] == 0
    assert len(result["modules_at_start"]) > 20 and set(result["modules_at_exit"]) >= set(result["modules_at_start"])
    assert result["frozen_file_hashes"] == launch_checks.FROZEN_FILES and set(result["diff_gate_hashes"]) == set(launch_checks.ALLOWLISTS)
    for entry in result["models"].values():
        assert entry["replay"]["status"] == "verified" and entry["evidence"]["batch1_latency"]["mean_seconds"] > 0


def test_the_stage_results_record_the_audit_the_modules_and_the_diff_gates(complete):
    h, _ = complete
    for stage in ("0", "0b", "1", "2"):
        result = read(h.root_b / f"worker-{stage}.result.json")
        assert result["ok"] is True and result["audit"]["violations"] == 0
        assert result["audit"]["opened_in_data_directory"]
        assert result["module_problems_at_exit"] == [] and result["frozen_file_hashes"] == launch_checks.FROZEN_FILES
        assert set(result["modules_at_exit"]) >= set(result["modules_at_start"])
    names = {p.name for p in h.root_b.iterdir()}
    assert {"LAUNCH.json", "LAUNCH_LOG", "logs", "rows", ".launch.lock"} <= names
    assert not list(h.root_b.rglob("*.partial"))


def test_no_worker_ran_a_relaxation_in_a0_and_exactly_the_expected_data_files_were_opened(complete):
    """N04: the audit lists are asserted equal to the expected sets, so a hook that never ran cannot pass. A0 builds no
    study inputs and opens the four dataset files; every stage also opens the Landscape17 archive (all five)."""
    h, _ = complete
    assert not (h.root_a / "rows").exists()
    tree, allowed = launch_checks.audit_scope(h.group9)
    five = launch_checks.allowed_relative(tree, allowed)
    four = launch_checks.required_relative(tree, allowed, "A0")
    assert len(four) == 4 and len(five) == 5 and set(four) < set(five)
    a0 = read(h.root_a / "A0_RESULT.json")["audit"]
    assert a0["opened_in_data_directory"] == four and a0["violations"] == 0
    for stage in ("0", "0b", "1", "2"):
        audit = read(h.root_b / f"worker-{stage}.result.json")["audit"]
        assert audit["opened_in_data_directory"] == five and audit["violations"] == 0, stage


def test_the_parents_and_the_workers_ledger_snapshots_are_equal_for_every_stage(complete):
    """Plan section 10 (e2-N10): the parent derives the ledger from evidence and writes it into the `started` event; the
    worker derives its own from the same evidence and writes it into its result. They must be the same ledger."""
    h, _ = complete
    for stage in ("0", "0b", "1", "2"):
        started = read(h.registry / f"{h.sha_b}-{stage}-1-started.json")["ledger_snapshot"]
        worker = read(h.root_b / f"worker-{stage}.result.json")["ledger_snapshot"]
        assert started == worker and started["a0_replay_passed"] is True, stage
        assert started["state"][stage] == "pending" and started["state"]["A0"] == "complete"
    # stage 2 was started with the hashes of stages 1 and 0b only: stage 0's marker is valid but untyped in that command
    # (A7), so its state is failed and it satisfies nothing, while stage 2 needs nothing from it
    assert read(h.registry / f"{h.sha_b}-2-1-started.json")["ledger_snapshot"]["state"] == {
        "A0": "complete", "0": "failed", "0b": "complete", "1": "complete", "2": "pending"}


def test_the_parent_prints_no_outcome_counts(complete):
    _, runs = complete
    for name in ("stage0", "stage0b", "stage1", "stage2"):
        out = runs[name].stdout.lower()
        assert "recover" not in out and "nonconverged" not in out and "stop_reason" not in out and "category" not in out


def test_the_tree_is_still_clean_after_a_complete_launch(complete):
    h, _ = complete
    assert launch_checks.tree_problems(h.scratch, h.record_a["commit"]) == []


def test_verify_launch_reads_a_complete_launch_and_finds_no_problem(complete):
    from stage2_capped import verify_launch
    h, _ = complete
    lines, problems = verify_launch.verify(h.folder)
    text = "\n".join(lines)
    assert problems == [] and lines[0] == "5 dispatches in the registry for this study"
    for unit in ("A0", "0", "0b", "1", "2"):
        assert f"unit {unit} completed in the first record pair" in text
    assert text.count("complete; spent True") == 5
    receipt = h.tmp / "receipt.json"
    receipt.write_text(json.dumps({"a0_result_sha256": h.a0_result_sha(),
                                   "stage_marker_sha256": {s: h.marker_sha(s) for s in ("0", "0b", "1", "2")}}))
    assert verify_launch.verify(h.folder, receipt)[1] == []
    done = subprocess.run([sys.executable, "-m", "stage2_capped.verify_launch", "--launch-folder", str(h.folder)],
                          cwd=h.scratch, env=h.environment(), capture_output=True, text=True)
    assert done.returncode == 0 and "problems: none" in done.stdout
