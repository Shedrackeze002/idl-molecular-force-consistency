"""Plan v3 sections 2, 7 and 8: the tree, frozen-file, module, diff-gate, runtime and preflight checks, and the
audit hook. Launch-level checks run on a committed scratch copy of the tree; the audit hook, which can end the
process, runs in a subprocess."""
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from stage2_capped import admission, launch_checks
from stage2_capped.tests.launch_fixtures import REPO, git, scratch, scratch_base  # noqa: F401


def head(tree):
    return launch_checks.head_commit(tree)


# ---- The tree ------------------------------------------------------------------------------------

def test_a_clean_committed_tree_has_no_problems(scratch):
    assert launch_checks.tree_problems(scratch, head(scratch)) == []


def test_a_modified_tracked_file_is_a_problem(scratch):
    (scratch / "stage2_capped" / "rules.py").write_text("changed\n")
    problems = launch_checks.tree_problems(scratch, head(scratch))
    assert len(problems) == 1 and "not clean" in problems[0]


def test_an_untracked_file_is_a_problem(scratch):
    (scratch / "stage2_capped" / "stray.py").write_text("x = 1\n")
    assert any("not clean" in p for p in launch_checks.tree_problems(scratch, head(scratch)))


def test_a_head_that_is_not_the_recorded_commit_is_a_problem(scratch):
    problems = launch_checks.tree_problems(scratch, "0" * 40)
    assert any("HEAD is" in p and "record names" in p for p in problems)


@pytest.mark.parametrize("flag, marker", [("--assume-unchanged", "assume-unchanged"), ("--skip-worktree", "skip-worktree")])
def test_a_hidden_tracked_file_is_a_problem(scratch, flag, marker):
    git(scratch, "update-index", flag, "stage2_capped/rules.py")
    (scratch / "stage2_capped" / "rules.py").write_text("hidden change\n")
    problems = launch_checks.tree_problems(scratch, head(scratch))
    assert any("hide behind" in p for p in problems), problems


def test_a_failing_git_command_is_a_problem_not_a_pass(tmp_path):
    problems = launch_checks.tree_problems(tmp_path, "0" * 40)
    assert problems and "git" in problems[0]


# ---- Frozen files -------------------------------------------------------------------------------------

def test_the_twelve_frozen_files_match_in_a_scratch_copy(scratch):
    assert len(launch_checks.FROZEN_FILES) == 12 and launch_checks.frozen_problems(scratch) == []


@pytest.mark.parametrize("relative", sorted(launch_checks.FROZEN_FILES))
def test_one_changed_byte_in_any_frozen_file_is_a_problem(scratch, relative):
    path = scratch / relative
    path.write_bytes(path.read_bytes() + b" ")
    problems = launch_checks.frozen_problems(scratch)
    assert len(problems) == 1 and relative in problems[0]


def test_a_missing_frozen_file_is_a_problem(scratch):
    (scratch / "evaluation" / "relax.py").unlink()
    assert any("missing" in p for p in launch_checks.frozen_problems(scratch))


# ---- Imported modules against the committed bytes --------------------------------------------------------

def fake_module(path):
    return types.SimpleNamespace(__file__=str(path))


def test_modules_of_the_worktree_that_equal_their_committed_bytes_pass(scratch):
    modules = {"stage2_capped.rules": fake_module(scratch / "stage2_capped" / "rules.py"),
               "evaluation.relax": fake_module(scratch / "evaluation" / "relax.py"),
               "numpy": fake_module(Path(sys.prefix) / "lib" / "numpy.py")}
    inventory, problems = launch_checks.module_inventory(scratch, modules)
    assert problems == [] and set(inventory) == {"stage2_capped/rules.py", "evaluation/relax.py"}
    assert all(len(v) == 64 for v in inventory.values())


def test_a_module_whose_bytes_differ_from_head_is_a_problem(scratch):
    path = scratch / "stage2_capped" / "rules.py"
    path.write_text(path.read_text() + "\n# edited after the commit\n")
    _, problems = launch_checks.module_inventory(scratch, {"stage2_capped.rules": fake_module(path)})
    assert problems == ["stage2_capped/rules.py differs from the committed bytes"]


def test_a_module_that_is_not_committed_is_a_problem(scratch):
    path = scratch / "stage2_capped" / "extra_module.py"
    path.write_text("x = 1\n")
    _, problems = launch_checks.module_inventory(scratch, {"stage2_capped.extra_module": fake_module(path)})
    assert problems == ["stage2_capped/extra_module.py is not committed at HEAD"]


def test_a_project_module_imported_from_outside_the_worktree_is_a_problem(scratch, tmp_path):
    outside = tmp_path / "elsewhere" / "rules.py"
    outside.parent.mkdir()
    outside.write_text("x = 1\n")
    _, problems = launch_checks.module_inventory(scratch, {"stage2_capped.rules": fake_module(outside)})
    assert len(problems) == 1 and "outside the worktree" in problems[0]


def test_the_project_names_cover_the_source_scripts_and_two_packages(scratch):
    names = launch_checks.project_names(scratch)
    assert {"evaluation", "stage2_capped", "e3x_phase_common", "force_consistency"} <= names


# ---- The diff gates ---------------------------------------------------------------------------------------

def test_the_three_gates_pass_and_give_nine_hashes(scratch):
    hashes, problems = launch_checks.diff_gate_hashes(scratch)
    assert problems == [] and set(hashes) == {"relax_capped", "restore_fork", "pass_copy"}
    assert all(set(h) == {"derived", "frozen", "allowlist"} and all(len(x) == 64 for x in h.values())
               for h in hashes.values())
    assert sum(len(h) for h in hashes.values()) == 9


@pytest.mark.parametrize("target, derived", [("relax_capped", "stage2_capped/relax_capped.py"),
                                             ("restore_fork", "stage2_capped/restore_fork.py"),
                                             ("pass_copy", "stage2_capped/pass_copy.py")])
def test_one_changed_byte_in_a_gated_file_fails_its_gate(scratch, target, derived):
    path = scratch / derived
    path.write_bytes(path.read_bytes().replace(b"\n", b"\n# extra line\n", 1))
    _, problems = launch_checks.diff_gate_hashes(scratch)
    assert problems and all(p.startswith(f"{target}:") for p in problems)


def test_a_changed_allow_list_changes_its_reported_hash(scratch):
    before, _ = launch_checks.diff_gate_hashes(scratch)
    path = scratch / "stage2_capped" / "allowlist.json"
    path.write_bytes(path.read_bytes() + b"\n")
    after, _ = launch_checks.diff_gate_hashes(scratch)
    assert after["relax_capped"]["allowlist"] != before["relax_capped"]["allowlist"]
    assert after["restore_fork"] == before["restore_fork"]


# ---- Runtime and preflight --------------------------------------------------------------------------------

def test_thread_variables_that_are_set_are_problems():
    assert launch_checks.thread_variable_problems({}) == []
    problems = launch_checks.thread_variable_problems({"OMP_NUM_THREADS": "4", "XLA_FLAGS": "--x"})
    assert len(problems) == 2 and "OMP_NUM_THREADS" in problems[0]


def test_runtime_values_leave_the_backend_out_unless_asked():
    assert "jax_backend" not in launch_checks.runtime_values({})
    assert launch_checks.runtime_values({}, include_backend=True)["jax_backend"]


def test_runtime_problems_compare_every_field_and_require_cpu():
    recorded = launch_checks.runtime_values({"PYTHONPATH": "/a"})
    assert launch_checks.runtime_problems({**recorded, "jax_backend": "cpu"}, dict(recorded)) == []
    other = launch_checks.runtime_values({"PYTHONPATH": "/b"})
    problems = launch_checks.runtime_problems({**recorded, "jax_backend": "cpu"}, other)
    assert problems == ["runtime pythonpath differs from A0's"]
    assert any("not cpu" in p for p in launch_checks.runtime_problems({**recorded, "jax_backend": "gpu"}, recorded))
    assert any("not cpu" in p for p in launch_checks.runtime_problems({**recorded, "jax_backend": "cpu"},
                                                                      {**recorded, "jax_backend": "tpu"}))


def test_hardlink_problem_reports_a_missing_folder_and_a_failing_link(tmp_path, monkeypatch):
    assert launch_checks.hardlink_problem(tmp_path) is None
    assert "not an existing directory" in launch_checks.hardlink_problem(tmp_path / "absent")

    def refuse(*args, **kwargs):
        raise OSError("operation not supported")
    monkeypatch.setattr(os, "link", refuse)
    assert "cannot make a hard link" in launch_checks.hardlink_problem(tmp_path)
    assert not list(tmp_path.glob(".linktest-*"))


def test_mount_point_is_a_mount(tmp_path):
    assert os.path.ismount(launch_checks.mount_point(tmp_path))


def preflight(tmp_path, scratch, **overrides):
    volume = launch_checks.mount_point(tmp_path)
    environ = {"PATH": os.environ["PATH"], "PYTHONPATH": str(scratch)}
    arguments = dict(worktree=scratch, root=tmp_path / "root", launch_folder=tmp_path, output_volume=volume,
                     environ=environ)
    arguments.update(overrides)
    return launch_checks.preflight_problems(**arguments)


def test_a_good_preflight_has_no_problems_or_only_a_missing_caffeinate(tmp_path, scratch):
    problems = preflight(tmp_path, scratch)
    assert all("caffeinate" in p for p in problems), problems          # every launch module imports: that is checked too


def test_preflight_names_each_problem(tmp_path, scratch):
    assert any("does not exist" in p for p in preflight(tmp_path, scratch, root=tmp_path / "no" / "root"))
    assert any("inside the worktree" in p for p in preflight(tmp_path, scratch, root=scratch / "out"))
    assert any("lies on volume" in p or "volume" in p for p in preflight(tmp_path, scratch, output_volume="/not-a-volume"))
    assert any("OMP_NUM_THREADS" in p for p in preflight(tmp_path, scratch, environ={
        "PATH": os.environ["PATH"], "PYTHONPATH": str(scratch), "OMP_NUM_THREADS": "2"}))
    assert any("PYTHONPATH" in p for p in preflight(tmp_path, scratch, environ={
        "PATH": os.environ["PATH"], "PYTHONPATH": "relative/path"}))
    assert any("caffeinate" in p for p in preflight(tmp_path, scratch, environ={
        "PATH": str(tmp_path / "empty"), "PYTHONPATH": str(scratch)}))


def test_the_hard_link_test_runs_in_the_launch_folder_too(tmp_path, scratch, monkeypatch):
    folder = tmp_path / "launch-folder"
    folder.mkdir()
    real = launch_checks.hardlink_problem
    monkeypatch.setattr(launch_checks, "hardlink_problem",
                        lambda directory: "no links in the launch folder" if Path(directory) == folder else real(directory))
    assert "no links in the launch folder" in preflight(tmp_path, scratch, launch_folder=folder)


def test_an_output_root_that_is_the_worktree_itself_is_refused(tmp_path, scratch):
    assert any("inside the worktree" in p for p in preflight(tmp_path, scratch, root=scratch))


def test_preflight_fails_when_free_space_is_short(tmp_path, scratch):
    assert any("free space" in p for p in preflight(tmp_path, scratch, free_bytes=2 ** 62))


# ---- The audit hook (subprocess: the hook ends the process) ---------------------------------------------------

SCRIPT = r'''
import json, os, sys
from pathlib import Path
from stage2_capped.launch_checks import AuditGuard
tree, outside, violations = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
mode = sys.argv[4]
allowed = [tree / "processed" / "keep.bin"]
guard = AuditGuard(tree, allowed, violations, violation_name="AUDIT_VIOLATION-test-0-1").enable()
if mode == "allowed":
    open(allowed[0], "rb").read()
    open(outside / "other.npz", "rb").read()
    open(outside / "plain.txt", "rb").read()
    guard.disable()
    open(tree / "processed" / "extra.bin", "rb").read()          # no longer watched
    print(json.dumps(guard.report()))
elif mode == "violation":
    try:
        open(tree / "processed" / "extra.bin", "rb").read()
    except BaseException:
        pass                                                      # a broad except cannot swallow it
    print("survived")
elif mode == "symlink":
    try:
        open(tree / "processed" / "link.bin", "rb").read()
    except BaseException:
        pass
    print("survived")
elif mode == "fd":
    fd = os.open(allowed[0], os.O_RDONLY); os.close(fd)
    print(json.dumps(guard.report()))
'''


def run_guard(tmp_path, mode):
    tree = tmp_path / "data"
    (tree / "processed").mkdir(parents=True)
    (tree / "processed" / "keep.bin").write_bytes(b"allowed")
    (tree / "processed" / "extra.bin").write_bytes(b"not allowed")
    (tree / "processed" / "link.bin").symlink_to(tree / "processed" / "extra.bin")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "other.npz").write_bytes(b"x")
    (outside / "plain.txt").write_bytes(b"x")
    violations = tmp_path / "violations"
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(str(REPO / p) for p in (".", "src", "scripts")),
                       PYTHONDONTWRITEBYTECODE="1")
    done = subprocess.run([sys.executable, "-c", SCRIPT, str(tree), str(outside), str(violations), mode],
                          capture_output=True, env=environment, cwd=REPO)
    return done, violations


def test_the_hook_lists_allowed_opens_and_logs_data_files_elsewhere(tmp_path):
    import json
    done, violations = run_guard(tmp_path, "allowed")
    assert done.returncode == 0, done.stderr.decode()[-400:]
    report = json.loads(done.stdout)
    assert report["opened_in_data_directory"] == [os.path.join("processed", "keep.bin")]
    assert report["opened_data_files_elsewhere"] == [os.path.realpath(tmp_path / "outside" / "other.npz")]
    assert report["violations"] == 0 and not list(violations.glob("*"))


def test_any_other_open_in_the_tree_ends_the_process_and_leaves_a_record(tmp_path):
    import json
    done, violations = run_guard(tmp_path, "violation")
    assert done.returncode == launch_checks.AUDIT_EXIT_CODE and b"survived" not in done.stdout
    written = list(violations.glob("AUDIT_VIOLATION-test-0-1.json"))
    assert len(written) == 1 and not list(violations.glob("*.partial"))
    assert json.loads(written[0].read_text())["path"] == os.path.realpath(tmp_path / "data" / "processed" / "extra.bin")


def test_a_symlink_inside_the_tree_is_judged_by_its_target(tmp_path):
    done, violations = run_guard(tmp_path, "symlink")
    assert done.returncode == launch_checks.AUDIT_EXIT_CODE and list(violations.glob("*.json"))


def test_opening_by_descriptor_is_not_a_second_open(tmp_path):
    import json
    done, _ = run_guard(tmp_path, "fd")
    assert done.returncode == 0
    assert json.loads(done.stdout)["opened_in_data_directory"] == [os.path.join("processed", "keep.bin")]


# ---- More of the hook's cases (subprocess: a violation ends the process) -------------------------------------------

SCRIPT_MORE = r'''
import json, os, sys, threading
from pathlib import Path
from stage2_capped.launch_checks import AuditGuard
base, mode = Path(sys.argv[1]), sys.argv[2]
tree = base / "data"
(tree / "processed").mkdir(parents=True)
(tree / "processed" / "keep.bin").write_bytes(b"allowed")
(tree / "processed" / "extra.bin").write_bytes(b"not allowed")
(base / "data-other").mkdir()
(base / "data-other" / "x.bin").write_bytes(b"a neighbour whose name starts like the tree's")
(base / "outside").mkdir()
(base / "outside" / "points_in.bin").symlink_to(tree / "processed" / "extra.bin")
(base / "alias").symlink_to(tree / "processed")
violations = tree / "violations" if mode == "recursion" else base / "violations"
allowed = [base / "alias" / "keep.bin"] if mode == "alias" else [tree / "processed" / "keep.bin"]
guard = AuditGuard(tree, allowed, violations, violation_name="AUDIT_VIOLATION-more-0-1").enable()
if mode == "threads":
    stop = threading.Event()

    def hammer():
        while not stop.is_set():
            open(tree / "processed" / "keep.bin", "rb").close()
    for _ in range(4):
        threading.Thread(target=hammer, daemon=True).start()
    import time
    time.sleep(0.2)                                       # let the other threads settle into their loops
    for attempt in range(5000):
        try:
            open(tree / "processed" / "extra.bin", "rb").close()
        except BaseException:
            pass
        print("MISSED", attempt, flush=True)             # reached only if the forbidden open was not caught
elif mode == "link_from_outside":
    try:
        open(base / "outside" / "points_in.bin", "rb").close()          # outside by name, inside by target
    except BaseException:
        pass
    print("survived", flush=True)
elif mode == "neighbour":
    open(base / "data-other" / "x.bin", "rb").close()                    # not under the tree
    print(json.dumps(guard.report()), flush=True)
elif mode == "alias":
    open(tree / "processed" / "keep.bin", "rb").close()                   # allowed through a symlinked directory
    print(json.dumps(guard.report()), flush=True)
elif mode == "recursion":
    try:
        open(tree / "processed" / "extra.bin", "rb").close()
    except BaseException:
        pass
    print("survived", flush=True)
'''


def run_more(tmp_path, mode):
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(str(REPO / p) for p in (".", "src", "scripts")),
                       PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, "-c", SCRIPT_MORE, str(tmp_path), mode], capture_output=True, env=environment,
                          cwd=REPO, timeout=120)


def test_a_forbidden_open_is_caught_while_other_threads_open_allowed_files(tmp_path):
    """N05: the recursion flag is per thread. With a process-wide flag another thread's open slips past the hook while
    the first thread is inside it."""
    done = run_more(tmp_path, "threads")
    assert done.returncode == launch_checks.AUDIT_EXIT_CODE and b"MISSED" not in done.stdout, done.stdout[-300:]


def test_a_symlink_outside_the_tree_that_points_into_it_is_judged_by_its_target(tmp_path):
    done = run_more(tmp_path, "link_from_outside")
    assert done.returncode == launch_checks.AUDIT_EXIT_CODE and b"survived" not in done.stdout


def test_a_folder_whose_name_only_starts_like_the_tree_is_not_inside_it(tmp_path):
    done = run_more(tmp_path, "neighbour")
    assert done.returncode == 0, done.stderr.decode()[-300:]
    assert json.loads(done.stdout)["violations"] == 0


def test_an_allowed_path_given_through_a_symlinked_folder_is_accepted(tmp_path):
    done = run_more(tmp_path, "alias")
    assert done.returncode == 0, done.stderr.decode()[-300:]
    assert json.loads(done.stdout)["opened_in_data_directory"] == [os.path.join("processed", "keep.bin")]


def test_a_violation_folder_inside_the_tree_still_gets_its_record_written(tmp_path):
    """The violation writer opens files itself; when its folder is inside the watched tree those opens must not
    recurse (the per-thread flag), or the process ends before the record is renamed into place."""
    done = run_more(tmp_path, "recursion")
    assert done.returncode == launch_checks.AUDIT_EXIT_CODE and b"survived" not in done.stdout
    written = list((tmp_path / "data" / "violations").glob("AUDIT_VIOLATION-more-0-1.json"))
    assert len(written) == 1 and json.loads(written[0].read_text())["path"] == os.path.realpath(
        tmp_path / "data" / "processed" / "extra.bin")


def test_audit_problems_need_zero_violations_every_required_file_and_no_other_file():
    required, allowed = ["processed/a.npz"], ["processed/a.npz", "archive.zip"]
    ok = {"violations": 0, "opened_in_data_directory": ["processed/a.npz"]}
    assert launch_checks.audit_problems(ok, required, allowed) == []
    assert launch_checks.audit_problems({"violations": 0, "opened_in_data_directory": ["archive.zip", "processed/a.npz"]},
                                        required, allowed) == []
    assert launch_checks.audit_problems({"violations": 1, "opened_in_data_directory": ["processed/a.npz"]}, required, allowed)
    assert launch_checks.audit_problems(None, required, allowed)
    assert launch_checks.audit_problems({"violations": 0, "opened_in_data_directory": ["processed/b.npz", "processed/a.npz"]},
                                        required, allowed)                       # a file outside the allowed set
    # N04: a hook that never ran (an empty list) or a missing required file is a problem, whatever else is listed
    assert launch_checks.audit_problems({"violations": 0, "opened_in_data_directory": []}, required, allowed)
    assert launch_checks.audit_problems({"violations": 0}, required, allowed)
    assert launch_checks.audit_problems({"violations": 0, "opened_in_data_directory": ["archive.zip"]}, required, allowed)


def test_the_hook_does_nothing_until_enabled_and_disable_switches_it_off(tmp_path):
    guard = launch_checks.AuditGuard(tmp_path / "data", [], tmp_path / "v")
    assert launch_checks._ACTIVE is not guard
    guard.disable()                                                 # a guard that was never enabled
    assert guard.report()["violations"] == 0


def test_the_audit_scope_is_the_data_tree_with_the_four_files_and_the_archive():
    from stage2_capped.tests.launch_fixtures import real_files
    if not real_files():
        pytest.skip("saved Group9 files are absent")
    tree, allowed = launch_checks.audit_scope()
    assert tree.name == "data" and len(allowed) == 5
    assert {p.name for p in allowed} == set(admission.DATASET_FILES) | {"Landscape17.zip"}
    assert launch_checks.allowed_relative(tree, allowed) == sorted(
        [os.path.join("processed", "aspirin", n) for n in admission.DATASET_FILES] + ["Landscape17.zip"])
    four = sorted(os.path.join("processed", "aspirin", n) for n in admission.DATASET_FILES)
    assert launch_checks.required_relative(tree, allowed, "A0") == four                     # A0 builds no study inputs
    for stage in ("0", "0b", "1", "2"):
        assert launch_checks.required_relative(tree, allowed, stage) == sorted(four + ["Landscape17.zip"])
