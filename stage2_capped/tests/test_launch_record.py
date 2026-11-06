"""Plan v3 section 2: records are machine-readable inputs, approved by two bound reviews, and verified by one shared
function. Real accepted files are read (the saved results, the launch records); the tree is a committed scratch copy."""
import copy
import json
import shutil
from pathlib import Path

import pytest

from stage2_capped import admission, launch_checks, launch_record, stage0_compare
from stage2_capped.launch_record import LaunchRefused
from stage2_capped.tests.launch_fixtures import (build_record, git, needs_real_files, scratch, scratch_base,  # noqa: F401
                                                 write_approval, write_record)

pytestmark = needs_real_files


@pytest.fixture
def volume(tmp_path):
    return launch_checks.mount_point(tmp_path)


@pytest.fixture
def record_a(scratch, tmp_path, volume):
    return build_record(scratch, kind="A", launch_id="A-1", root=tmp_path / "outA", volume=volume)


@pytest.fixture
def record_b(scratch, tmp_path, volume):
    return build_record(scratch, kind="B", launch_id="B-1", root=tmp_path / "outB", volume=volume)


def problems_of(record, command, scratch, **kwargs):
    return launch_record.record_problems(record, command, worktree=scratch, **kwargs)


# ---- Structure ---------------------------------------------------------------------------------------

def test_a_good_record_a_and_record_b_have_no_problems(record_a, record_b, scratch):
    assert launch_record.structure_problems(record_a) == [] and launch_record.structure_problems(record_b) == []
    assert problems_of(record_a, "a0", scratch) == [] and problems_of(record_b, "stage", scratch) == []


@pytest.mark.parametrize("field", [f for f in launch_record.COMMON_FIELDS if f != "record"])
def test_a_missing_common_field_is_a_problem(record_a, field):
    broken = {k: v for k, v in record_a.items() if k != field}
    assert any(f"missing field {field}" in p for p in launch_record.structure_problems(broken))


def test_a_record_without_its_kind_is_refused(record_a):
    broken = {k: v for k, v in record_a.items() if k != "record"}
    assert launch_record.structure_problems(broken) == ['record must be "A" or "B"']
    assert launch_record.structure_problems([]) == ["a record is a JSON object"]


@pytest.mark.parametrize("field", launch_record.RECORD_B_FIELDS)
def test_a_missing_record_b_field_is_a_problem(record_b, field):
    broken = {k: v for k, v in record_b.items() if k != field}
    assert any(f"missing field {field}" in p for p in launch_record.structure_problems(broken))


def test_record_a_may_not_carry_record_b_fields_and_an_unknown_field_is_refused(record_a):
    assert any("unexpected" in p for p in launch_record.structure_problems({**record_a, "record_a_sha256": "a" * 64}))
    assert any("unexpected" in p for p in launch_record.structure_problems({**record_a, "tuning": 1}))


@pytest.mark.parametrize("change, expected", [
    ({"record": "C"}, "record must be"),
    ({"commit": "abc1234"}, "40 hex"),
    ({"commit": "A" * 40}, "40 hex"),
    ({"commit": "a" * 41}, "40 hex"),
    ({"launch_folder": "relative/launch"}, "absolute"),
    ({"launch_folder": ""}, "launch_folder"),
    ({"launch_id": ""}, "launch_id"),
    ({"output_root": "relative/root"}, "absolute"),
    ({"output_volume": "relative"}, "absolute"),
    ({"expected_launch_records_sha256": "x" * 64}, "expected_launch_records_sha256"),
    ({"rules_py_sha256": "short"}, "rules_py_sha256"),
    ({"rules_py_sha256": "a" * 65}, "rules_py_sha256"),
    ({"relaxer_sha256": None}, "relaxer_sha256"),
    ({"caffeinate": "caffeinate -i"}, "caffeinate"),
    ({"prior_dispatches": [{"file": "x", "sha256": "0" * 64}]}, "prior_dispatches"),
    ({"prior_dispatches": [["x", "nothex"]]}, "prior_dispatches"),
    ({"ceilings": {"a0_wall_seconds": 300.0}}, "ceilings"),
])
def test_a_malformed_value_is_a_problem(record_a, change, expected):
    assert any(expected in p for p in launch_record.structure_problems({**record_a, **change}))


def test_the_saved_hashes_must_be_exactly_110_pairs_of_hex(record_a):
    short = dict(list(record_a["saved_file_hashes"].items())[:109])
    assert any("110" in p for p in launch_record.structure_problems({**record_a, "saved_file_hashes": short}))
    long = {**record_a["saved_file_hashes"], "runs/extra/file.json": "1" * 64}
    assert len(long) == 111 and any("110" in p for p in launch_record.structure_problems({**record_a, "saved_file_hashes": long}))
    bad = {**record_a["saved_file_hashes"], next(iter(record_a["saved_file_hashes"])): "nothex"}
    assert any("110" in p for p in launch_record.structure_problems({**record_a, "saved_file_hashes": bad}))


def test_frozen_and_gate_hash_blocks_must_have_their_exact_keys(record_a):
    fewer = dict(list(record_a["frozen_file_hashes"].items())[:11])
    assert any("twelve" in p for p in launch_record.structure_problems({**record_a, "frozen_file_hashes": fewer}))
    gates = copy.deepcopy(record_a["diff_gate_hashes"])
    del gates["pass_copy"]["allowlist"]
    assert any("diff_gate_hashes" in p for p in launch_record.structure_problems({**record_a, "diff_gate_hashes": gates}))


# ---- Comparison with code and disk -------------------------------------------------------------------------

def test_a_record_for_the_wrong_command_is_refused(record_a, record_b, scratch):
    assert any("needs record B" in p for p in problems_of(record_a, "stage", scratch))
    assert any("needs record A" in p for p in problems_of(record_b, "a0", scratch))


def test_ceilings_must_lie_between_the_spec_floor_and_three_times_it(record_a, scratch):
    def with_stage(stage, **change):
        record = copy.deepcopy(record_a)
        record["ceilings"]["stages"][stage].update(change)
        return record
    assert problems_of(with_stage("1", wall_seconds=2100.0), "a0", scratch) == []
    assert problems_of(with_stage("1", wall_seconds=6300.0), "a0", scratch) == []       # exactly the cap
    assert any("outside" in p for p in problems_of(with_stage("1", wall_seconds=2099.0), "a0", scratch))
    assert any("outside" in p for p in problems_of(with_stage("1", wall_seconds=6300.1), "a0", scratch))
    assert any("runs" in p for p in problems_of(with_stage("0", runs=79), "a0", scratch))
    assert any("attempt_ceiling" in p for p in problems_of(with_stage("2", attempt_ceiling=999999), "a0", scratch))


def test_the_a0_ceiling_is_at_least_300_and_at_most_900(record_a, scratch):
    for value, ok in ((300.0, True), (900.0, True), (299.9, False), (900.1, False)):
        record = copy.deepcopy(record_a)
        record["ceilings"]["a0_wall_seconds"] = value
        assert (problems_of(record, "a0", scratch) == []) is ok, value


def test_the_cap_factor_and_the_caffeinate_argv_must_quote_the_code(record_a, scratch):
    assert any("ceiling_cap_factor" in p for p in problems_of({**record_a, "ceiling_cap_factor": 5.0}, "a0", scratch))
    assert any("caffeinate" in p for p in problems_of({**record_a, "caffeinate": ["caffeinate", "-s"]}, "a0", scratch))


def test_a_record_that_changes_one_saved_hash_is_refused(record_a, scratch):
    key = next(iter(record_a["saved_file_hashes"]))
    altered = {**record_a, "saved_file_hashes": {**record_a["saved_file_hashes"], key: "0" * 64}}
    assert any("saved_file_hashes differ" in p for p in problems_of(altered, "a0", scratch))


def test_a_saved_file_edited_on_disk_is_refused_by_the_record(record_a, scratch, tmp_path):
    real = admission.group9_root()
    mirror = tmp_path / "g9"
    (mirror / "reports" / "execution").mkdir(parents=True)
    inventories = [inventory for _, inventory in launch_record.SAVED_INVENTORIES]
    for relative in list(record_a["saved_file_hashes"]) + [admission.RECOVERY_RECORDS] + inventories:
        destination = mirror / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(real / relative, destination)
    assert problems_of(record_a, "a0", scratch, group9=mirror) == []
    victim = mirror / next(iter(record_a["saved_file_hashes"]))
    victim.write_bytes(victim.read_bytes() + b" ")
    assert any("saved_file_hashes differ" in p for p in problems_of(record_a, "a0", scratch, group9=mirror))


@pytest.mark.parametrize("field", ["expected_launch_records_sha256", "rules_py_sha256", "relaxer_sha256"])
def test_a_changed_single_hash_is_refused(record_a, scratch, field):
    altered = {**record_a, field: ("0" if record_a[field][0] != "0" else "1") + record_a[field][1:]}
    assert any(field.split("_sha256")[0].replace("_py", "") in p for p in problems_of(altered, "a0", scratch))


def test_changed_frozen_or_gate_hashes_are_refused(record_a, scratch):
    first = next(iter(record_a["frozen_file_hashes"]))
    frozen = {**record_a["frozen_file_hashes"], first: "0" * 64}
    assert any("frozen_file_hashes differ" in p for p in problems_of({**record_a, "frozen_file_hashes": frozen}, "a0", scratch))
    gates = copy.deepcopy(record_a["diff_gate_hashes"])
    gates["restore_fork"]["derived"] = "0" * 64
    assert any("diff_gate_hashes differ" in p for p in problems_of({**record_a, "diff_gate_hashes": gates}, "a0", scratch))


def mirror_of_the_saved_files(tmp_path, names):
    """A Group9 mirror holding copies of the given files (saved results, launch records, inventories)."""
    real = admission.group9_root()
    mirror = tmp_path / "g9"
    for relative in names:
        destination = mirror / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(real / relative, destination)
    return mirror


def test_a_record_whose_saved_hash_matches_the_disk_but_not_the_inventory_is_refused(scratch, tmp_path, volume):
    """N06: the record's 110 hashes are compared with the disk and with the two accepted closeout inventories, which
    were written independently. A saved file altered before the record was written makes record and disk agree; only
    the inventories notice."""
    real = admission.group9_root()
    inventories = [inventory for _, inventory in launch_record.SAVED_INVENTORIES]
    mirror = mirror_of_the_saved_files(tmp_path, list(stage0_compare.saved_file_hashes(real)) + [admission.RECOVERY_RECORDS]
                                       + inventories)
    victim = mirror / next(iter(stage0_compare.saved_file_hashes(real)))
    victim.write_bytes(victim.read_bytes() + b" ")                          # the disk was altered before the record
    record = build_record(scratch, kind="A", launch_id="A-1", root=tmp_path / "outA", volume=volume)
    record["saved_file_hashes"] = stage0_compare.saved_file_hashes(mirror)  # so record and disk agree
    problems = problems_of(record, "a0", scratch, group9=mirror)
    assert any("differs from the inventory" in p for p in problems), problems
    assert not any("saved_file_hashes differ" in p for p in problems)


# ---- Inventories and the Markdown text -----------------------------------------------------------------------

def test_the_records_saved_hashes_equal_the_two_accepted_inventories(record_a):
    assert launch_record.inventory_problems(record_a["saved_file_hashes"]) == []


def test_a_hash_that_differs_from_an_inventory_or_is_in_neither_is_refused(record_a):
    saved = dict(record_a["saved_file_hashes"])
    key = next(iter(saved))
    problems = launch_record.inventory_problems({**saved, key: "0" * 64})
    assert problems == [f"{key} differs from the inventory"]
    assert launch_record.inventory_problems({**saved, "runs/elsewhere/file.json": "0" * 64})


def test_the_markdown_text_must_carry_the_records_values(record_b):
    text = "Record B\n" + "\n".join(str(v) for v in (
        record_b["commit"], record_b["launch_id"], record_b["output_root"], record_b["output_volume"], record_b["launch_folder"],
        record_b["expected_launch_records_sha256"], record_b["rules_py_sha256"], record_b["relaxer_sha256"],
        record_b["record_a_sha256"], record_b["a0_launch_id"], record_b["a0_output_root"],
        *record_b["frozen_file_hashes"].values(), *(h for g in record_b["diff_gate_hashes"].values() for h in g.values())))
    assert launch_record.markdown_problems(text, record_b) == []
    assert launch_record.markdown_problems(text.replace(record_b["launch_id"], "x"), record_b) == [
        f"the Markdown text lacks {record_b['launch_id']!r}"]
    assert launch_record.markdown_problems(text.replace(record_b["launch_folder"], "x"), record_b) == [
        f"the Markdown text lacks {record_b['launch_folder']!r}"]
    for field in ("commit", "rules_py_sha256", "relaxer_sha256", "expected_launch_records_sha256"):
        assert f"the Markdown text lacks {record_b[field]!r}" in launch_record.markdown_problems(
            text.replace(record_b[field], "x"), record_b), field
    first_frozen = next(iter(record_b["frozen_file_hashes"].values()))
    assert f"the Markdown text lacks {first_frozen!r}" in launch_record.markdown_problems(text.replace(first_frozen, "x"), record_b)


# ---- The record file and the approval -------------------------------------------------------------------------------

def test_load_record_refuses_a_different_hash_and_a_non_json_file(record_a, tmp_path):
    path, digest = write_record(tmp_path / "records", record_a)
    assert launch_record.load_record(path, digest) == record_a
    with pytest.raises(LaunchRefused, match="SHA-256 differs"):
        launch_record.load_record(path, digest[:-1] + ("0" if digest[-1] != "0" else "1"))
    junk = tmp_path / "junk.json"
    junk.write_text("not json")
    with pytest.raises(LaunchRefused, match="not JSON"):
        launch_record.load_record(junk, launch_record.sha256_bytes(junk.read_bytes()))


def test_load_record_refuses_a_malformed_record_even_with_the_right_hash(record_a, tmp_path):
    broken = {k: v for k, v in record_a.items() if k != "commit"}
    path, digest = write_record(tmp_path / "records", {**broken, "record": "A"})
    with pytest.raises(LaunchRefused, match="missing field commit"):
        launch_record.load_record(path, digest)


def test_an_approval_with_two_distinct_bound_reviews_is_accepted(record_a, tmp_path):
    _, digest = write_record(tmp_path / "records", record_a)
    approval = write_approval(tmp_path / "records", digest)
    assert launch_record.load_approval(approval, digest)["status"] == launch_record.APPROVAL_STATUS


@pytest.mark.parametrize("edit, message", [
    (lambda a, d: a.update(status="draft"), "status"),
    (lambda a, d: a.update(record_sha256="0" * 64), "different record"),
    (lambda a, d: a.update(approved_by=" "), "approved_by"),
    (lambda a, d: a.update(approval_line=""), "approval_line"),
    (lambda a, d: a.update(review_sha256=dict(list(a["review_sha256"].items())[:1])), "two distinct"),
    (lambda a, d: a.update(review_sha256={k: list(a["review_sha256"].values())[0] for k in a["review_sha256"]}), "two distinct"),
])
def test_a_bad_approval_is_refused(record_a, tmp_path, edit, message):
    _, digest = write_record(tmp_path / "records", record_a)
    path = write_approval(tmp_path / "records", digest)
    approval = json.loads(path.read_text())
    edit(approval, digest)
    path.write_text(json.dumps(approval))
    with pytest.raises(LaunchRefused, match=message):
        launch_record.load_approval(path, digest)


def test_review_paths_must_be_absolute_and_values_must_be_strings(record_a, tmp_path):
    _, digest = write_record(tmp_path / "records", record_a)
    path = write_approval(tmp_path / "records", digest)
    approval = json.loads(path.read_text())
    (first, h1), (second, h2) = approval["review_sha256"].items()
    for broken in ({"relative/review-1.md": h1, second: h2}, {first: [h1], second: h2}, {first: h1, second: {"x": 1}}):
        path.write_text(json.dumps({**approval, "review_sha256": broken}))
        with pytest.raises(LaunchRefused, match="two distinct"):
            launch_record.load_approval(path, digest)


def test_a_review_that_changed_after_approval_is_refused(record_a, tmp_path):
    _, digest = write_record(tmp_path / "records", record_a)
    path = write_approval(tmp_path / "records", digest)
    review = next(iter(json.loads(path.read_text())["review_sha256"]))
    open(review, "a").write("edited\n")
    with pytest.raises(LaunchRefused, match="changed since"):
        launch_record.load_approval(path, digest)


def test_two_names_for_one_review_file_are_refused(record_a, tmp_path):
    _, digest = write_record(tmp_path / "records", record_a)
    path = write_approval(tmp_path / "records", digest)
    approval = json.loads(path.read_text())
    first = next(iter(approval["review_sha256"]))
    approval["review_sha256"] = {first: approval["review_sha256"][first], str(tmp_path / "records" / ".." / "records" / first.split("/")[-1]): approval["review_sha256"][first]}
    path.write_text(json.dumps(approval))
    with pytest.raises(LaunchRefused, match="two distinct"):
        launch_record.load_approval(path, digest)


def test_an_unreadable_approval_is_refused(tmp_path):
    with pytest.raises(LaunchRefused, match="cannot be read"):
        launch_record.load_approval(tmp_path / "absent.json", "0" * 64)


# ---- The shared context check --------------------------------------------------------------------------------

def stage_files(scratch, tmp_path, volume, kind="A", **kwargs):
    record = build_record(scratch, kind=kind, launch_id=f"{kind}-ctx", root=tmp_path / f"out{kind}", volume=volume, **kwargs)
    path, digest = write_record(tmp_path / "launch" / "records", record)
    return record, path, write_approval(tmp_path / "launch" / "records", digest), digest


def test_the_context_check_returns_the_record_for_a_clean_approved_tree(scratch, tmp_path, volume):
    record, path, approval, digest = stage_files(scratch, tmp_path, volume)
    assert launch_record.verify_launch_context(path, approval, digest, command="a0", worktree=scratch) == record


def test_the_context_check_refuses_a_dirty_tree_a_wrong_commit_and_a_hidden_change(scratch, tmp_path, volume):
    record, path, approval, digest = stage_files(scratch, tmp_path, volume)
    git(scratch, "update-index", "--assume-unchanged", "stage2_capped/rules.py")
    with pytest.raises(LaunchRefused, match="hide behind"):
        launch_record.verify_launch_context(path, approval, digest, command="a0", worktree=scratch)
    git(scratch, "update-index", "--no-assume-unchanged", "stage2_capped/rules.py")
    (scratch / "stage2_capped" / "stray.py").write_text("x = 1\n")
    with pytest.raises(LaunchRefused, match="not clean"):
        launch_record.verify_launch_context(path, approval, digest, command="a0", worktree=scratch)


@pytest.mark.parametrize("damage, message", [("missing", "cannot be read"), ("draft", "status"),
                                              ("another record", "different record")])
def test_the_context_check_refuses_a_launch_without_a_valid_approval(scratch, tmp_path, volume, damage, message):
    """Every launch-level test supplies a valid approval; this one takes it away. Parent and workers share the check."""
    record, path, approval, digest = stage_files(scratch, tmp_path, volume)
    if damage == "missing":
        approval.unlink()
    else:
        body = json.loads(approval.read_text())
        body.update({"draft": {"status": "draft"}, "another record": {"record_sha256": "0" * 64}}[damage])
        approval.write_text(json.dumps(body))
    with pytest.raises(LaunchRefused, match=message):
        launch_record.verify_launch_context(path, approval, digest, command="a0", worktree=scratch)


@pytest.mark.parametrize("relative", sorted(launch_checks.FROZEN_FILES))
def test_the_context_check_refuses_each_frozen_file_changed_in_a_new_commit(scratch, tmp_path, volume, relative):
    """The tree is clean (the change is committed), so neither the clean-tree check nor the dirty-file route finds it, and
    the diff gates look at three of the twelve files only: this line of the context check is the one that refuses."""
    record, path, approval, digest = stage_files(scratch, tmp_path, volume)
    victim = scratch / relative
    victim.write_bytes(victim.read_bytes() + b" ")
    git(scratch, "add", "-A")
    git(scratch, "commit", "-q", "-m", "a changed frozen file")
    with pytest.raises(LaunchRefused) as refused:
        launch_record.verify_launch_context(path, approval, digest, command="a0", worktree=scratch)
    assert f"frozen file {relative} differs from its recorded SHA-256" in refused.value.problems
    assert not any("not clean" in p for p in refused.value.problems)


def test_a_record_file_copied_to_another_folder_is_refused_because_its_registry_would_be_empty(scratch, tmp_path, volume):
    """N02: the registry of spent units lives where the approved record says; a copy of the record elsewhere must not
    get a registry of its own."""
    record, path, approval, digest = stage_files(scratch, tmp_path, volume)
    other = tmp_path / "elsewhere" / "records"
    other.mkdir(parents=True)
    shutil.copy2(path, other / path.name)
    shutil.copy2(approval, other / approval.name)
    with pytest.raises(LaunchRefused, match="names the launch folder"):
        launch_record.verify_launch_context(other / path.name, other / approval.name, digest, command="a0", worktree=scratch)
    stray = tmp_path / "launch" / "notes"
    stray.mkdir()
    shutil.copy2(path, stray / path.name)
    with pytest.raises(LaunchRefused, match="folder named records"):
        launch_record.verify_launch_context(stray / path.name, approval, digest, command="a0", worktree=scratch)


def test_the_launch_folder_is_the_same_through_a_symlinked_path(scratch, tmp_path, volume):
    record, path, approval, digest = stage_files(scratch, tmp_path, volume)
    (tmp_path / "launch-link").symlink_to(tmp_path / "launch")
    linked = tmp_path / "launch-link" / "records" / path.name
    assert launch_record.launch_folder_of(linked) == (tmp_path / "launch").resolve()
    assert launch_record.launch_folder_of(linked) == launch_record.launch_folder_of(path)
    assert launch_record.verify_launch_context(linked, approval, digest, command="a0", worktree=scratch) == record


def test_the_launch_folder_may_not_lie_in_the_worktree_or_contain_or_sit_in_the_output_root(scratch, tmp_path, volume):
    record = build_record(scratch, kind="A", launch_id="A-1", root=tmp_path / "outA", volume=volume)
    assert launch_record.placement_problems(record, scratch) == []
    inside_worktree = {**record, "launch_folder": str(scratch / "launch")}
    assert any("inside the worktree" in p for p in launch_record.placement_problems(inside_worktree, scratch))
    around_root = {**record, "launch_folder": str(tmp_path)}                    # the output root is inside it
    assert any("contain one another" in p for p in launch_record.placement_problems(around_root, scratch))
    inside_root = {**record, "launch_folder": str(tmp_path / "outA" / "launch")}
    assert any("contain one another" in p for p in launch_record.placement_problems(inside_root, scratch))
    assert any("inside the worktree" in p for p in problems_of(inside_worktree, "a0", scratch))        # record_problems uses it
    equal_root = {**record, "launch_folder": str(tmp_path / "outA")}                  # X20: equal is inside, both ways
    assert any("contain one another" in p for p in launch_record.placement_problems(equal_root, scratch))
    equal_worktree = {**record, "launch_folder": str(scratch)}
    assert any("inside the worktree" in p for p in launch_record.placement_problems(equal_worktree, scratch))


def test_a_launch_folder_the_record_names_through_a_symlink_is_accepted_and_is_the_same_registry(scratch, tmp_path, volume):
    """X16, decided by reading `folder_problems`: both sides are resolved before they are compared, so a record that names
    the launch folder through a symlink is accepted, and the registry it finds is the one a record file in the real folder
    finds. (A false refusal here would be fail-closed, but it would stop a launch for a spelling of the same folder.)"""
    (tmp_path / "launch-link").symlink_to(tmp_path / "launch")
    record, path, approval, digest = stage_files(scratch, tmp_path, volume, launch_folder=tmp_path / "launch-link")
    assert record["launch_folder"] == str(tmp_path / "launch-link")
    assert launch_record.folder_problems(record, path) == []
    assert launch_record.verify_launch_context(path, approval, digest, command="a0", worktree=scratch) == record
    assert Path(record["launch_folder"]).resolve() == launch_record.launch_folder_of(path) == (tmp_path / "launch").resolve()


def test_the_context_check_refuses_a_record_that_names_another_commit(scratch, tmp_path, volume):
    record, path, approval, digest = stage_files(scratch, tmp_path, volume, commit="1" * 40)
    with pytest.raises(LaunchRefused, match="record names"):
        launch_record.verify_launch_context(path, approval, digest, command="a0", worktree=scratch)


def test_the_context_check_refuses_the_wrong_command_and_a_wrong_typed_hash(scratch, tmp_path, volume):
    record, path, approval, digest = stage_files(scratch, tmp_path, volume)
    with pytest.raises(LaunchRefused, match="needs record B"):
        launch_record.verify_launch_context(path, approval, digest, command="stage", worktree=scratch)
    with pytest.raises(LaunchRefused, match="SHA-256 differs"):
        launch_record.verify_launch_context(path, approval, "0" * 64, command="a0", worktree=scratch)


def test_the_context_check_refuses_a_frozen_file_that_changed(scratch, tmp_path, volume):
    record, path, approval, digest = stage_files(scratch, tmp_path, volume)
    victim = scratch / "evaluation" / "relax.py"
    victim.write_bytes(victim.read_bytes() + b" ")
    with pytest.raises(LaunchRefused) as refused:
        launch_record.verify_launch_context(path, approval, digest, command="a0", worktree=scratch)
    assert any("frozen file evaluation/relax.py" in p for p in refused.value.problems)
