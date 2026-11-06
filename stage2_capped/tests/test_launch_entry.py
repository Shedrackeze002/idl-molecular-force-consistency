"""The production entry points: the command line, the worker `main` functions, `production_deps` and the argv the real
parent hands the real workers. The end-to-end tests run the real parent and workers with a test seam for their
dependencies; this file is what covers the seam's replacements (decision log C21, mutation check holes 9 and 10).

No model is loaded anywhere here: the real-file wiring runs the production dependencies with an arm that returns the
saved forces, under the launch's own audit guard (a read outside the five allowed data files ends the process)."""
import json
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest

from stage2_capped import admission, driver, launch, launch_a0, launch_arms, launch_checks, launch_record, launch_stage, launch_state
from stage2_capped.tests.launch_fixtures import REPO, needs_real_files, scratch, scratch_base  # noqa: F401
from stage2_capped.tests.launch_harness import Harness


@pytest.fixture
def sigterm_restored():
    previous = signal.getsignal(signal.SIGTERM)
    yield
    signal.signal(signal.SIGTERM, previous)


# ---- The argv the real parent hands the real workers --------------------------------------------------------------

def parent_args(**overrides):
    values = dict(record="/r/RECORD_B.json", approval="/r/approval.json", record_sha256="b" * 64, group9_root="/g9",
                  a0_result_sha256="a" * 64, stage0_sha256="0" * 64, stage0b_sha256="1" * 64, stage1_sha256="2" * 64)
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("unit", ["A0", "0", "0b", "1", "2"])
def test_the_production_worker_command_names_the_right_module_and_every_flag(unit):
    argv = launch.production_worker_command(unit, 3, parent_args(), {})
    assert argv[:2] == list(launch_record.CAFFEINATE_ARGV) and argv[2] == sys.executable and argv[3] == "-m"
    assert argv[4] == ("stage2_capped.launch_a0" if unit == "A0" else "stage2_capped.launch_stage")
    pairs = dict(zip(argv[5::2], argv[6::2]))
    expected = {"--record": "/r/RECORD_B.json", "--approval": "/r/approval.json", "--record-sha256": "b" * 64,
                "--attempt": "3", "--group9-root": "/g9"}
    if unit != "A0":
        expected.update({"--unit": unit, "--a0-result-sha256": "a" * 64, "--stage0-sha256": "0" * 64,
                         "--stage0b-sha256": "1" * 64, "--stage1-sha256": "2" * 64})
    assert pairs == expected and len(argv) == 5 + 2 * len(expected)


def test_the_production_worker_command_leaves_out_flags_that_were_not_typed():
    argv = launch.production_worker_command("1", 1, parent_args(stage0b_sha256=None, stage1_sha256=None, group9_root=None), {})
    assert "--stage0b-sha256" not in argv and "--stage1-sha256" not in argv and "--group9-root" not in argv
    assert "--stage0-sha256" in argv and "--a0-result-sha256" in argv


def test_every_flag_the_parent_passes_is_one_the_worker_parses():
    """The workers refuse an unknown flag, so a flag the parent adds but the worker lacks would refuse every launch."""
    args = parent_args()
    for unit in ("A0", "0", "0b", "1", "2"):
        argv = launch.production_worker_command(unit, 1, args, {})[5:]
        parser = launch_state.add_common_arguments(__import__("argparse").ArgumentParser()) if unit == "A0" \
            else launch_stage.build_parser()
        parsed = parser.parse_args(argv)
        assert parsed.attempt == 1 and parsed.record_sha256 == "b" * 64


# ---- production_deps ---------------------------------------------------------------------------------------------

def test_the_parent_stage_and_a0_production_dependencies_are_the_real_functions():
    parent = launch.production_deps()
    assert parent.admit_all is admission.admit_all and parent.worker_command is launch.production_worker_command
    from run_validation_recovery import GROUPS
    a0 = launch_a0.production_deps()
    assert a0.admit_all is admission.admit_all and a0.load_bindings is admission.load_bindings
    assert a0.data_directory is admission.data_directory and a0.provider_factory is launch_arms.real_arm_provider
    assert a0.groups == tuple(GROUPS) and callable(a0.replay) and callable(a0.load_partition) and callable(a0.saved_forces)
    stage = launch_stage.production_deps()
    assert stage.admit_all is admission.admit_all and stage.load_bindings is admission.load_bindings
    assert stage.data_directory is admission.data_directory and stage.provider_factory is launch_arms.real_arm_provider


def test_the_a0_replay_hands_the_frozen_function_its_four_arguments_in_order(monkeypatch):
    import run_validation_recovery
    seen = []
    monkeypatch.setattr(run_validation_recovery, "adapter_replay", lambda *a, **k: seen.append((a, k)) or "returned")
    assert launch_a0.production_deps().replay("ARM", "SCORE", "DATA", "SPLIT") == "returned"
    assert seen == [(("ARM", "SCORE", "DATA", "SPLIT"), {})]


def test_the_a0_partition_loader_pins_the_split_and_returns_the_validation_partition(monkeypatch):
    import force_consistency.data.prepare as prepare
    from e3x_phase_common import SPLIT
    seen = []
    monkeypatch.setattr(prepare, "load_training_bundle",
                        lambda *a, **k: seen.append((a, k)) or SimpleNamespace(validation="THE VALIDATION PARTITION"))
    assert launch_a0.production_deps().load_partition("DATA") == "THE VALIDATION PARTITION"
    assert seen == [(("DATA",), {"expected_split_hash": SPLIT})]


def test_the_a0_saved_forces_are_the_models_forces_not_the_reference_forces(tmp_path):
    np.savez(tmp_path / "best-validation.npz", record_ids=np.array(["a"]), forces=np.full((2, 3, 3), 7.0),
             reference_forces=np.full((2, 3, 3), -1.0))
    saved = launch_a0.production_deps().saved_forces(tmp_path)
    assert np.array_equal(saved, np.full((2, 3, 3), 7.0))


def test_the_stage_loaders_pass_the_record_hashes_and_the_group9_root_through(monkeypatch):
    from stage2_capped import stage0_compare
    seen = []
    monkeypatch.setattr(driver, "load_study_inputs", lambda *a, **k: seen.append(("inputs", a, k)) or "INPUTS")
    monkeypatch.setattr(stage0_compare, "load_saved", lambda *a, **k: seen.append(("saved", a, k)) or "SAVED")
    deps = launch_stage.production_deps()
    hashes = {"path": "f" * 64}
    assert deps.inputs_loader("LAUNCH-RECORDS-HASH", "G9") == "INPUTS" and deps.saved_loader(hashes, "G9") == "SAVED"
    assert seen == [("inputs", ("LAUNCH-RECORDS-HASH",), {"group9": "G9"}), ("saved", ("G9",), {"expected_hashes": hashes})]


def test_a_failure_after_the_spent_point_gets_the_label_of_its_kind():
    assert launch_a0.classify("restore", admission.PinMismatch("x")) == "pin"
    assert launch_a0.classify("replay", admission.AdmissionError("x")) == "admission"
    assert launch_a0.classify("replay", driver.DeadlineReached("x")) == "deadline"
    assert launch_a0.classify("restore", ValueError("x")) == "restore" and launch_a0.classify("replay", ValueError("x")) == "replay"
    assert launch_a0.classify("elsewhere", ValueError("x")) == "other"


# ---- The command line and the worker main functions --------------------------------------------------------------

def test_the_command_line_passes_each_sub_command_to_execute_with_the_production_dependencies(monkeypatch):
    seen = []
    monkeypatch.setattr(launch, "production_deps", lambda: "PRODUCTION DEPS")
    monkeypatch.setattr(launch, "execute", lambda command, args, deps: seen.append((command, args, deps)) or 0)
    base = ["--record", "r", "--approval", "a", "--record-sha256", "h"]
    assert launch.main(["a0", *base]) == 0 and launch.main(["check", "a0", *base]) == 0
    assert launch.main(["stage", "0b", *base, "--a0-result-sha256", "x", "--stage0-sha256", "y"]) == 0
    assert launch.main(["check", "stage", "2", *base, "--a0-result-sha256", "x"]) == 0
    assert [(c, d) for c, _, d in seen] == [("a0", "PRODUCTION DEPS"), ("check-a0", "PRODUCTION DEPS"),
                                            ("stage", "PRODUCTION DEPS"), ("check-stage", "PRODUCTION DEPS")]
    assert seen[2][1].unit == "0b" and seen[2][1].stage0_sha256 == "y" and seen[3][1].unit == "2"


@pytest.mark.parametrize("module, extra, check", [
    (launch_a0, [], lambda a: a.attempt == 3),
    (launch_stage, ["--unit", "0b", "--a0-result-sha256", "x", "--stage0-sha256", "y"],
     lambda a: a.attempt == 3 and a.unit == "0b" and a.stage0_sha256 == "y" and a.a0_result_sha256 == "x"),
])
def test_each_worker_main_installs_the_sigterm_handler_and_runs_with_its_production_dependencies(
        monkeypatch, sigterm_restored, module, extra, check):
    seen = {}
    monkeypatch.setattr(module, "production_deps", lambda: "PRODUCTION DEPS")

    def fake_run(args, deps):
        seen.update(args=args, deps=deps, handler=signal.getsignal(signal.SIGTERM))
        return 7
    monkeypatch.setattr(module, "run", fake_run)
    argv = ["--record", "r", "--approval", "a", "--record-sha256", "h", "--attempt", "3", *extra]
    assert module.main(argv) == 7
    assert seen["deps"] == "PRODUCTION DEPS" and check(seen["args"]) and seen["args"].record_sha256 == "h"
    with pytest.raises(driver.DeadlineReached):                  # the handler in place when `run` started turns SIGTERM into a labelled stop
        seen["handler"](signal.SIGTERM, None)


def test_the_sigterm_helpers_turn_a_stop_into_a_labelled_exception_and_then_ignore_a_late_one(sigterm_restored):
    def not_installed(signum, frame):                             # in place until install_sigterm replaces it
        raise AssertionError("install_sigterm did not install its handler")
    signal.signal(signal.SIGTERM, not_installed)
    launch_state.install_sigterm()
    with pytest.raises(driver.DeadlineReached, match="terminated by the supervisor"):
        os.kill(os.getpid(), signal.SIGTERM)
        time.sleep(1)
    launch_state.ignore_sigterm()
    assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN
    os.kill(os.getpid(), signal.SIGTERM)                          # a SIGTERM during finalization must not raise
    time.sleep(0.2)


# ---- A real parent, the real command line, the real model-free admission ---------------------------------------------

@needs_real_files
def test_the_real_command_line_checks_a_launch_without_writing_anything(scratch, tmp_path):
    h = Harness(scratch, tmp_path)
    done = h.run_production("check", "a0", "--record", str(h.path_a), "--approval", str(h.approval_a),
                            "--record-sha256", h.sha_a, "--group9-root", str(h.group9))
    assert done.returncode == 0 and "check passed: nothing was written" in done.stdout, done.stdout[-500:] + done.stderr[-500:]
    assert not h.root_a.exists() and not h.registry.exists() and not (h.folder / "violations").exists()
    refused = h.run_production("check", "stage", "0", "--record", str(h.path_b), "--approval", str(h.approval_b),
                               "--record-sha256", h.sha_b, "--group9-root", str(h.group9))
    assert refused.returncode == 2 and "--a0-result-sha256" in refused.stdout and not h.root_b.exists()
    wrong = h.run_production("a0", "--record", str(h.path_b), "--approval", str(h.approval_b), "--record-sha256", h.sha_b,
                             "--group9-root", str(h.group9))
    assert wrong.returncode == 2 and "needs record A" in wrong.stdout and not h.root_b.exists()


# ---- The production dependencies on the real files, with an arm that returns the saved forces ------------------------

def run_wiring(mode, tmp_path):
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(str(REPO / p) for p in (".", "src", "scripts")),
                       PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run([sys.executable, "-m", "stage2_capped.tests.launch_wiring", mode, str(tmp_path / "violations")],
                          cwd=REPO, env=environment, capture_output=True, text=True, timeout=1500)
    assert done.returncode == 0, f"exit {done.returncode}: {done.stdout[-400:]} {done.stderr[-1200:]}"
    assert not (tmp_path / "violations").exists()
    return json.loads(done.stdout.strip().splitlines()[-1])


@needs_real_files
def test_the_production_a0_dependencies_replay_the_real_partition_through_the_frozen_function(tmp_path):
    out = run_wiring("a0", tmp_path)
    assert out["models"] == list(admission.ALL_IDS) and out["records"] == admission.REPLAY_RECORDS == 974
    assert out["groups"] == ["13", "18", "18_2", "66", "78"]
    assert out["replay"] == {"status": "verified", "record_count": 974, "single_record_ids": out["first_five"],
                             "rtol": 1e-10, "atol": 1e-10}
    assert out["saved_forces_are_the_forces_array"] is True
    assert out["evidence_batch4_bit_identical"] == 974 == out["evidence_batch1_bit_identical"]
    tree, allowed = launch_checks.audit_scope(admission.group9_root())
    assert out["audit"]["violations"] == 0
    assert out["audit"]["opened_in_data_directory"] == launch_checks.required_relative(tree, allowed, "A0")


@needs_real_files
def test_the_production_stage_dependencies_load_the_real_inputs_and_check_the_110_hashes(tmp_path):
    out = run_wiring("stage", tmp_path)
    assert out["saved_tables"] == [50, 30] and out["changed_hash_refused_with"] == "SavedResultError"
    assert out["starts"] == 10 and out["reference_ids"] == 6                    # Stage 0: 5 models x (10 + 6) = 80 runs
    tree, allowed = launch_checks.audit_scope(admission.group9_root())
    assert out["audit"]["violations"] == 0
    assert out["audit"]["opened_in_data_directory"] == launch_checks.required_relative(tree, allowed, "0")
