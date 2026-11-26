"""M3: pin policy, data-directory binding, the seven checkpoint identities, the saved-JSON
checks and the replay plan. Synthetic fixtures always run; checks against the saved Group9
files skip when those files are absent (STAGE2_REQUIRE_REAL_FILES=1 makes a skip a failure).
No model is loaded."""
import copy
import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from stage2_capped import admission as adm


def sha(data):
    return hashlib.sha256(data).hexdigest()


def real_files():
    try:
        root = adm.group9_root()
    except adm.AdmissionError:
        return False
    return all((root / adm.CONTROLLED_RUNS / m / "best.msgpack").is_file()
               for m in ("direct-0", "consistency-1-0")) and (root / adm.DATA_DIRECTORY).is_dir()


needs_real_files = pytest.mark.skipif(not real_files(), reason="saved Group9 files are absent")


def pin(**changes):
    base = {"code_fingerprint": "a046011a" + "0" * 56,
            "code_files": {"src/a.py": "1" * 64, "uv.lock": "2" * 64},
            "lock_sha256": "2" * 64,
            "python": "3.12.13",
            "dependencies": {"jax": "0.4.1", "numpy": "1.26"},
            "runtime_cpu": {"platform": "macOS-1", "machine": "arm64", "processor": "arm",
                            "logical_cpus": 18,
                            "thread_environment": {"OMP_NUM_THREADS": None, "XLA_FLAGS": None}},
            "helpers": {"e3x_phase_common.py": "3" * 64},
            "templates": {"aspirin-e3x-a.json": "4" * 64}}
    base.update(changes)
    return base


def cpu(**changes):
    return {**pin()["runtime_cpu"], **changes}


# ---- Pin policy ----------------------------------------------------------

def test_identical_pins_pass_with_nothing_recorded():
    report = adm.compare_pins(pin(), pin())
    assert report["problems"] == [] and report["recorded_differences"] == {}
    assert all(report["required_equal"].values())
    assert set(report["required_equal"]) == set(adm.REQUIRED_EQUAL) | {"runtime_cpu_except_platform"}


def test_python_platform_and_template_drift_is_recorded_not_fatal():
    drifted = pin(python="3.12.16", runtime_cpu=cpu(platform="macOS-2"),
                  templates={"aspirin-e3x-a.json": "5" * 64})
    report = adm.compare_pins(drifted, pin())
    assert report["problems"] == []
    assert set(report["recorded_differences"]) == {"python", "runtime_cpu.platform", "templates"}
    assert report["recorded_differences"]["python"] == {"current": "3.12.16", "accepted": "3.12.13"}
    assert report["recorded_differences"]["runtime_cpu.platform"] == {"current": "macOS-2", "accepted": "macOS-1"}


@pytest.mark.parametrize("changes", [
    {"machine": "x86_64"}, {"processor": "i386"}, {"logical_cpus": 2},
    {"thread_environment": {"OMP_NUM_THREADS": "1", "XLA_FLAGS": None}},
    {"thread_environment": {"OMP_NUM_THREADS": None, "XLA_FLAGS": "--xla_cpu_enable_fast_math=true"}},
], ids=["machine", "processor", "logical-cpus", "omp-threads", "xla-flags"])
def test_every_cpu_fact_except_the_platform_string_must_be_equal(changes):
    report = adm.compare_pins(pin(runtime_cpu=cpu(**changes)), pin())
    assert report["problems"] and any("runtime_cpu" in p for p in report["problems"])
    assert report["required_equal"]["runtime_cpu_except_platform"] is False
    with pytest.raises(adm.PinMismatch):
        adm.PinPolicy()(pin(runtime_cpu=cpu(**changes)), pin())


def test_a_runtime_cpu_block_with_other_fields_fails_closed():
    odd = cpu()
    odd["extra"] = 1
    assert adm.compare_pins(pin(runtime_cpu=odd), pin())["problems"]
    no_platform = {k: v for k, v in cpu().items() if k != "platform"}
    assert adm.compare_pins(pin(runtime_cpu=no_platform), pin(runtime_cpu=no_platform))["problems"]


@pytest.mark.parametrize("change", [
    {"code_files": {"src/a.py": "9" * 64, "uv.lock": "2" * 64}},          # a code file changed
    {"code_files": {"src/a.py": "1" * 64}},                               # a file removed
    {"code_files": {"src/a.py": "1" * 64, "uv.lock": "2" * 64, "src/b.py": "7" * 64}},  # added
    {"code_fingerprint": "b" * 64},
    {"lock_sha256": "8" * 64},
    {"dependencies": {"jax": "0.4.2", "numpy": "1.26"}},
    {"helpers": {"e3x_phase_common.py": "6" * 64}},
], ids=["file-changed", "file-removed", "file-added", "fingerprint", "lock", "dependencies", "helper"])
def test_any_required_difference_is_refused(change):
    report = adm.compare_pins(pin(**change), pin())
    assert report["problems"]
    with pytest.raises(adm.PinMismatch):
        adm.PinPolicy()(pin(**change), pin())


def test_a_code_file_difference_names_the_file():
    report = adm.compare_pins(pin(code_files={"src/a.py": "9" * 64, "uv.lock": "2" * 64}), pin())
    assert any("src/a.py" in problem for problem in report["problems"])


@pytest.mark.parametrize("bad", ["extra", "missing", "string", "none"])
def test_a_pin_with_other_keys_or_a_wrong_type_fails_closed(bad):
    odd = pin()
    if bad == "extra":
        odd["extra"] = 1
    elif bad == "missing":
        del odd["templates"]
    else:
        odd = "not a pin" if bad == "string" else None
    assert adm.compare_pins(odd, pin())["problems"]
    assert adm.compare_pins(pin(), odd)["problems"]


def test_the_policy_keeps_every_report_even_when_it_refuses():
    policy = adm.PinPolicy()
    policy(pin(python="3.12.16"), pin())
    with pytest.raises(adm.PinMismatch):
        policy(pin(lock_sha256="8" * 64), pin())
    assert len(policy.reports) == 2
    assert policy.reports[0]["recorded_differences"] and policy.reports[1]["problems"]


# ---- Paths ---------------------------------------------------------------

@pytest.fixture
def group9(tmp_path):
    (tmp_path / "reports" / "execution").mkdir(parents=True)
    return adm.group9_root(tmp_path)


def previous_root(group9):
    """A saved-record prefix at the root's depth: the same parent folder, another project folder name."""
    return str(group9.parent / "previous-root") + "/"


def test_relocate_maps_a_previous_root_keeps_new_paths_and_refuses_strangers(group9):
    assert adm.relocate(previous_root(group9) + "runs/x/best.msgpack", group9) == group9 / "runs/x/best.msgpack"
    assert adm.relocate(str(group9 / "runs/y"), group9) == group9 / "runs/y"
    other_parent = str(group9.parent.parent / "other-parent" / "previous-root") + "/"
    for stranger in ("/somewhere/else/runs/x", "relative/runs/x", "", other_parent + "runs/x",
                     str(group9.parent / "previous-root")):            # too short: nothing below the folder
        with pytest.raises(adm.AdmissionError):
            adm.relocate(stranger, group9)


def test_the_name_of_the_previous_folder_is_not_part_of_the_rule(group9):
    """The rule is the depth and the parent folder, not a folder name: any sibling project
    folder under the same parent maps onto the root. The bytes are still bound by hash."""
    for name in ("previous-root", "other-name", group9.name + "X"):
        assert adm.relocate(str(group9.parent / name / "runs/y"), group9) == group9 / "runs/y"


def test_relocate_refuses_dot_dot_paths_and_symlinks_that_leave_the_root(group9, tmp_path):
    for escape in (previous_root(group9) + "../../../../../../../etc/passwd",
                   previous_root(group9) + "runs/../../../../outside",
                   str(group9 / "runs/../../../outside")):
        with pytest.raises(adm.AdmissionError, match="outside the root"):
            adm.relocate(escape, group9)
    outside = tmp_path.parent / "outside-target"
    outside.mkdir(exist_ok=True)
    (group9 / "link").symlink_to(outside)
    with pytest.raises(adm.AdmissionError, match="outside the root"):
        adm.relocate(previous_root(group9) + "link/file", group9)
    # A `..` that stays inside the root is fine, and the result is the real path.
    assert adm.relocate(previous_root(group9) + "runs/../reports/x", group9) == group9 / "reports/x"


def test_a_relative_path_is_refused_even_when_the_current_directory_is_inside_the_root(group9, monkeypatch):
    (group9 / "runs").mkdir()
    monkeypatch.chdir(group9)
    with pytest.raises(adm.AdmissionError, match="not under a known root"):
        adm.relocate("runs/x", group9)


def test_group9_root_must_look_like_group9_and_says_where_it_came_from(tmp_path, monkeypatch):
    with pytest.raises(adm.AdmissionError):
        adm.group9_root(tmp_path)
    (tmp_path / "reports" / "execution").mkdir(parents=True)
    monkeypatch.setenv(adm.ROOT_ENVIRONMENT_VARIABLE, str(tmp_path))
    assert adm.group9_root() == tmp_path.resolve()
    assert adm.root_source() == "environment" and adm.root_source(tmp_path) == "argument"
    monkeypatch.delenv(adm.ROOT_ENVIRONMENT_VARIABLE)
    assert adm.root_source() == "default"


# ---- Identity checks on fabricated files ---------------------------------

def fabricate(tmp_path, *, name="direct-0", role="primary", folder="runs"):
    """A consistent model: files, launch record, acceptance, anchors and inventories."""
    base = tmp_path / folder
    run, score_directory = base / name, base / f"{name}-score"
    run.mkdir(parents=True), score_directory.mkdir()
    accepted = pin()
    launch = {"source_pin": accepted, "candidate": "C",
              "dataset_pin": {"manifest_identity": "4b9f2202" + "0" * 56,
                              "files": {n: "5" * 64 for n in adm.DATASET_FILES}}}
    checkpoint, predictions = b"checkpoint bytes", b"prediction bytes"
    (run / "best.msgpack").write_bytes(checkpoint)
    for file in adm.RUN_FILES:
        (run / file).write_text(json.dumps({"file": file}))
    (score_directory / "best-validation.npz").write_bytes(predictions)
    score = {"best_update": 1500, "best_checkpoint_sha256": sha(checkpoint), "status": "verified",
             "source_pin": accepted, "candidate": "C", "source_matches": True,
             "best_reload_verified": True, "prediction_sha256": sha(predictions),
             "equal_connection_force_mae_eV_per_A": 0.038}
    (score_directory / "score.json").write_text(json.dumps(score))
    launch_file = base / f"{name}-launch.json"
    launch_file.write_text(json.dumps(launch))
    score_sha = sha((score_directory / "score.json").read_bytes())
    item = {"id": name, "role": role, "run_directory": run, "score_directory": score_directory,
            "launch_record": launch, "source": "fabricated",
            "acceptance": {"checkpoint_sha256": sha(checkpoint), "best_update": 1500, "score_sha256": score_sha}}
    anchors = {name: {"best_update": 1500, "checkpoint_sha256": sha(checkpoint), "score_sha256": score_sha}}
    which, entries = adm.inventory_entries(item)
    inventories = {which: {n: {"sha256": sha(Path(p).read_bytes())} for n, p in entries.items()}}
    return item, accepted, inventories, anchors


def problems_of(item, accepted, inventories, anchors, root=None):
    return adm.identity_problems(item, accepted, inventories, root, anchors)


def test_a_consistent_fabricated_model_passes(tmp_path):
    item, accepted, inventories, anchors = fabricate(tmp_path)
    assert problems_of(item, accepted, inventories, anchors) == []
    record = adm.identity_record(item)
    assert record["best_msgpack_sha256"] == item["acceptance"]["checkpoint_sha256"]
    assert set(record["run_files_sha256"]) == set(adm.RUN_FILES) and record["launch_file_sha256"]


def rewrite_score(item, **changes):
    path = Path(item["score_directory"]) / "score.json"
    score = json.loads(path.read_text())
    score.update(changes)
    path.write_text(json.dumps(score))
    return sha(path.read_bytes())


@pytest.mark.parametrize("what", [
    "checkpoint-bytes", "score-bytes", "best-update", "status", "predictions", "candidate",
    "source-matches", "reload-not-verified", "score-names-other-checkpoint", "launch-pin",
    "dataset-identity", "missing-file", "best-update-not-int", "score-pin-only",
    "launch-and-score-pin-vs-accepted"])
def test_every_identity_check_can_fail(tmp_path, what):
    item, accepted, inventories, anchors = fabricate(tmp_path)
    run, sd = Path(item["run_directory"]), Path(item["score_directory"])
    if what == "checkpoint-bytes":
        (run / "best.msgpack").write_bytes(b"other")
    elif what == "score-bytes":
        item["acceptance"]["score_sha256"] = "0" * 64
    elif what == "best-update":
        item["acceptance"]["best_update"] = 1501
    elif what == "best-update-not-int":
        item["acceptance"]["best_update"] = 1500.0
    elif what == "status":
        item["acceptance"]["score_sha256"] = rewrite_score(item, status="pending")
    elif what == "predictions":
        (sd / "best-validation.npz").write_bytes(b"changed")
    elif what == "candidate":
        item["acceptance"]["score_sha256"] = rewrite_score(item, candidate="D")
    elif what == "source-matches":
        item["acceptance"]["score_sha256"] = rewrite_score(item, source_matches=False)
    elif what == "reload-not-verified":
        item["acceptance"]["score_sha256"] = rewrite_score(item, best_reload_verified=None)
    elif what == "score-names-other-checkpoint":
        item["acceptance"]["score_sha256"] = rewrite_score(item, best_checkpoint_sha256="0" * 64)
    elif what == "launch-pin":
        item["launch_record"] = {**item["launch_record"], "source_pin": pin(python="3.12.99")}
    elif what == "dataset-identity":
        item["launch_record"] = {**item["launch_record"], "dataset_pin": {
            **item["launch_record"]["dataset_pin"], "manifest_identity": "ffffffff" + "0" * 56}}
    elif what == "missing-file":
        (sd / "best-validation.npz").unlink()
    elif what == "score-pin-only":  # launch pin still equals the accepted pin
        item["acceptance"]["score_sha256"] = rewrite_score(item, source_pin=pin(python="3.12.99"))
    elif what == "launch-and-score-pin-vs-accepted":  # score and launch agree with each other only
        other = pin(python="3.12.99")
        item["launch_record"] = {**item["launch_record"], "source_pin": other}
        item["acceptance"]["score_sha256"] = rewrite_score(item, source_pin=other)
    problems = problems_of(item, accepted, inventories, anchors)
    expected = {
        "checkpoint-bytes": ["best.msgpack differs from the accepted checkpoint hash"],
        "score-bytes": ["score.json differs from the accepted score hash"],
        "best-update": ["best_update differs from the accepted binding"],
        "best-update-not-int": ["best_update differs from the accepted binding"],
        "status": ["score status is not verified"],
        "predictions": ["saved predictions differ from the score's prediction hash"],
        "candidate": ["score candidate differs from the launch record"],
        "source-matches": ["score does not confirm source match and best reload"],
        "reload-not-verified": ["score does not confirm source match and best reload"],
        "score-names-other-checkpoint": ["score.json names a different best checkpoint"],
        "launch-pin": ["score source pin differs from the launch record",
                       "launch record source pin differs from the accepted pin"],
        "dataset-identity": ["launch dataset identity is not the one the spec records"],
        "missing-file": ["missing file"],
        "score-pin-only": ["score source pin differs from the launch record"],
        "launch-and-score-pin-vs-accepted": ["launch record source pin differs from the accepted pin"],
    }[what]
    for text in expected:  # each check is the one that fires, not just any check
        assert any(text in p for p in problems), (text, problems)


def test_an_accepted_pin_other_than_the_one_the_spec_records_is_refused(tmp_path):
    item, accepted, inventories, anchors = fabricate(tmp_path)
    other = pin(code_fingerprint="c" * 64)
    item["launch_record"] = {**item["launch_record"], "source_pin": other}
    rewrite_score(item, source_pin=other)
    item["acceptance"]["score_sha256"] = sha((Path(item["score_directory"]) / "score.json").read_bytes())
    assert any("spec records" in p for p in problems_of(item, other, inventories, anchors))


def test_an_unknown_model_id_is_refused(tmp_path):
    item, accepted, inventories, anchors = fabricate(tmp_path)
    item["id"] = "made-up"
    assert any("unknown model id" in p for p in problems_of(item, accepted, inventories, anchors))


# ---- Independent anchors --------------------------------------------------

@pytest.mark.parametrize("field", ["checkpoint_sha256", "score_sha256", "best_update"])
def test_an_accepted_model_must_match_its_independent_anchor_not_just_its_own_records(tmp_path, field):
    """A forged copy whose checkpoint, score and binding were rewritten together is refused."""
    item, accepted, inventories, anchors = fabricate(tmp_path)
    anchors[item["id"]][field] = 1501 if field == "best_update" else "0" * 64
    assert any("independent anchor" in p for p in problems_of(item, accepted, inventories, anchors))


def test_a_primary_model_without_an_anchor_is_refused(tmp_path):
    item, accepted, inventories, anchors = fabricate(tmp_path)
    assert any("no independent anchor" in p for p in problems_of(item, accepted, inventories, {}))


def test_a_forged_copy_in_a_differently_named_folder_is_not_waved_through(tmp_path):
    """The inventory check is keyed on the model id, so renaming the folder cannot skip it."""
    item, accepted, inventories, anchors = fabricate(tmp_path, folder="elsewhere")
    (Path(item["run_directory"]) / "best.msgpack").write_bytes(b"forged")
    rewritten = rewrite_score(item, best_checkpoint_sha256=sha(b"forged"))
    item["acceptance"] = {**item["acceptance"], "checkpoint_sha256": sha(b"forged"), "score_sha256": rewritten}
    forged_anchor = {item["id"]: {**anchors[item["id"]], "checkpoint_sha256": sha(b"forged"),
                                  "score_sha256": rewritten}}
    problems = problems_of(item, accepted, inventories, forged_anchor)
    assert any("differs from the controlled inventory" in p for p in problems)


# ---- Inventories ----------------------------------------------------------

def test_the_inventory_is_required_and_must_list_every_file_with_the_same_hash(tmp_path):
    item, accepted, inventories, anchors = fabricate(tmp_path)
    assert problems_of(item, accepted, inventories, anchors) == []
    assert any("inventory is missing" in p for p in problems_of(item, accepted, None, anchors))
    assert any("inventory is missing" in p for p in problems_of(item, accepted, {}, anchors))
    assert any("inventory is missing" in p for p in problems_of(item, accepted, {"energy": {"x": 1}}, anchors))
    table = inventories["controlled"]
    for name in table:
        wrong = {"controlled": {**table, name: {"sha256": "0" * 64}}}
        missing = {"controlled": {k: v for k, v in table.items() if k != name}}
        assert any(name in p for p in problems_of(item, accepted, wrong, anchors)), name
        assert any(name in p for p in problems_of(item, accepted, missing, anchors)), name


def test_a_file_the_inventory_lists_but_that_is_missing_on_disk_is_a_problem(tmp_path):
    item, accepted, inventories, anchors = fabricate(tmp_path)
    (Path(item["run_directory"]) / "config.json").unlink()
    assert any("config.json is missing on disk" in p for p in problems_of(item, accepted, inventories, anchors))


def test_the_inventory_names_the_run_files_too(tmp_path):
    item, accepted, inventories, anchors = fabricate(tmp_path)
    _, entries = adm.inventory_entries(item)
    for name in adm.RUN_FILES:
        assert f"{item['id']}/{name}" in entries
    (Path(item["run_directory"]) / "config.json").write_text("{}")
    assert any("config.json" in p for p in problems_of(item, accepted, inventories, anchors))


def test_energy_zero_is_checked_against_the_original_run_inventory(tmp_path):
    item, accepted, inventories, anchors = fabricate(tmp_path, name="energy-0")
    assert list(inventories) == ["energy"] and all(n.startswith("C/") for n in inventories["energy"])
    assert problems_of(item, accepted, inventories, anchors) == []
    assert any("energy completed-output inventory is missing" in p
               for p in problems_of(item, accepted, {"controlled": {"x": 1}}, anchors))


def test_lambda_one_must_match_the_identities_the_spec_records(tmp_path):
    item, accepted, inventories, anchors = fabricate(tmp_path, name="consistency-1-0", role="exploratory")
    problems = problems_of(item, accepted, inventories, anchors)
    assert any("best update differs from the spec" in p for p in problems)
    assert any("checkpoint hash differs from the spec" in p for p in problems)
    assert any("score hash differs from the spec" in p for p in problems)


@pytest.mark.parametrize("value, prefix_suffix, ok", [
    ("9f092d78" + "0" * 52 + "2aa4", ("9f092d78", "2aa4"), True),
    ("9f092d79" + "0" * 52 + "2aa4", ("9f092d78", "2aa4"), False),    # prefix only differs
    ("9f092d78" + "0" * 52 + "2aa5", ("9f092d78", "2aa4"), False),    # suffix only differs
])
def test_the_affix_check_needs_both_the_prefix_and_the_suffix(value, prefix_suffix, ok):
    assert adm._affixes_ok(value, prefix_suffix) is ok


def test_directories_outside_the_root_are_flagged(tmp_path):
    item, accepted, inventories, anchors = fabricate(tmp_path)
    assert problems_of(item, accepted, inventories, anchors, root=tmp_path) == []
    assert any("outside the root" in p for p in
               problems_of(item, accepted, inventories, anchors, root=tmp_path / "runs" / "elsewhere"))


# ---- Dataset binding (monkeypatched pin) ---------------------------------

def test_the_dataset_pin_must_match_and_must_not_name_the_test_partition(tmp_path, monkeypatch):
    import e3x_phase_common
    pin_value = {"manifest_identity": "4b9f2202" + "0" * 56,
                 "files": {n: "5" * 64 for n in adm.DATASET_FILES}}
    monkeypatch.setattr(e3x_phase_common, "dataset_pin", lambda directory: copy.deepcopy(pin_value))
    launch = {"dataset_pin": copy.deepcopy(pin_value)}
    assert adm.verify_dataset(tmp_path, launch) == pin_value
    for name in adm.DATASET_FILES:
        altered = copy.deepcopy(launch)
        altered["dataset_pin"]["files"][name] = "6" * 64
        with pytest.raises(adm.AdmissionError, match=name):
            adm.verify_dataset(tmp_path, altered)
    altered = copy.deepcopy(launch)
    altered["dataset_pin"]["manifest_identity"] = "e" * 64
    with pytest.raises(adm.AdmissionError, match="identity differs"):
        adm.verify_dataset(tmp_path, altered)
    extra = copy.deepcopy(launch)
    extra["dataset_pin"]["files"]["t" + "est.npz"] = "7" * 64
    with pytest.raises(adm.AdmissionError, match="unexpected files"):
        adm.verify_dataset(tmp_path, extra)


def test_the_dataset_files_never_include_the_test_partition():
    assert not any("test" in name for name in adm.DATASET_FILES)


# ---- Constants and static guarantees -------------------------------------

def test_the_fixed_settings_are_the_spec_values():
    assert adm.REPLAY_RECORDS == 974 and adm.REPLAY_BATCH == 4 and adm.REPLAY_TOLERANCE == 1e-10
    assert adm.REPLAY_CONNECTIONS == 5
    assert adm.ALL_IDS == adm.PRIMARY_IDS + adm.LAMBDA1_IDS and len(adm.ALL_IDS) == 7
    assert set(adm.EXPECTED_PRIMARY) == set(adm.PRIMARY_IDS)
    assert all(len(v["checkpoint_sha256"]) == len(v["score_sha256"]) == 64 for v in adm.EXPECTED_PRIMARY.values())


def test_admission_loads_no_model_and_never_names_the_test_partition():
    source = Path(adm.__file__).read_text()
    for forbidden in ("load_checkpoint", "model_and_predictor", "initialize_state",
                      "jax.jit", "msgpack_restore", "from_bytes", "import restore_fork",
                      "restore_fork import", "e3x_adapter import"):
        assert forbidden not in source, forbidden
    for module in Path(adm.__file__).parent.glob("*.py"):
        assert "test.npz" not in module.read_text(), module.name


def test_the_production_entry_point_cannot_be_given_a_pin_or_a_policy():
    assert list(inspect.signature(adm.admit_all).parameters) == ["root"]
    assert set(inspect.signature(adm._admit_all).parameters) == {"root", "policy", "actual_pin"}


def test_the_module_report_names_each_module_and_its_hash():
    report = adm.module_report()
    assert set(report) == set(adm.MODULES_IN_USE)
    for name, entry in report.items():
        assert entry["file"] and Path(entry["file"]).is_file() and entry["sha256"] == adm.file_sha256(entry["file"]), name


# ---- STAGE2_REQUIRE_REAL_FILES --------------------------------------------

def test_a_skip_becomes_a_failure_when_real_files_are_required(tmp_path):
    test = Path(__file__).with_name("test_admission.py")
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(test),
               "-k", "test_all_seven_identities_admit_from_the_saved_files"]
    base = {**os.environ, "STAGE2_GROUP9_ROOT": str(tmp_path / "nonexistent"), "PYTHONDONTWRITEBYTECODE": "1"}
    cwd = Path(__file__).resolve().parents[2]
    relaxed = subprocess.run(command, env={**base, "STAGE2_REQUIRE_REAL_FILES": "0"}, capture_output=True,
                             text=True, cwd=cwd)
    assert relaxed.returncode == 0 and "1 skipped" in relaxed.stdout
    strict = subprocess.run(command, env={**base, "STAGE2_REQUIRE_REAL_FILES": "1"}, capture_output=True,
                            text=True, cwd=cwd)
    summary = [line for line in strict.stdout.splitlines() if line.strip()][-1]
    assert strict.returncode != 0 and ("1 error" in summary or "1 failed" in summary) and "skipped" not in summary
    assert "requires the saved Group9 files" in strict.stdout


# ---- The real saved files ------------------------------------------------

@needs_real_files
def test_all_seven_identities_admit_from_the_saved_files():
    report = adm.admit_all()
    assert list(report["identities"]) == list(adm.ALL_IDS)
    assert [report["identities"][i]["role"] for i in adm.ALL_IDS] == ["primary"] * 5 + ["exploratory"] * 2
    assert all(report["pin_policy"]["required_equal"].values())
    assert set(report["pin_policy"]["recorded_differences"]) <= {"python", "runtime_cpu.platform", "templates"}
    plan = report["replay_plan"]
    assert plan["records"] == 974 and plan["rtol"] == 1e-10 and plan["models"] == list(adm.ALL_IDS)
    assert plan["through"] == "arm['forces']" and plan["batch"] == 4 and plan["single_molecule_calls_per_connection"] == 1


@needs_real_files
def test_the_report_names_the_root_the_inputs_the_pin_and_the_modules():
    report = adm.admit_all()
    root = adm.group9_root()
    assert report["root"] == str(root) and report["root_source"] in ("default", "environment")
    assert report["launch_records_sha256"] == adm.file_sha256(root / adm.RECOVERY_RECORDS)
    assert report["inventory_sha256"] == {"controlled": adm.file_sha256(root / adm.CONTROLLED_INVENTORY),
                                          "energy": adm.file_sha256(root / adm.ENERGY_INVENTORY)}
    from e3x_phase_common import source_pin
    assert report["current_pin"]["code_fingerprint"] == source_pin()["code_fingerprint"]
    assert report["accepted_pin"] == adm.accepted_source_pin()
    assert set(report["modules"]) == set(adm.MODULES_IN_USE) and all(m["sha256"] for m in report["modules"].values())
    for cid, record in report["identities"].items():
        assert Path(record["run_directory"]).is_dir() and set(record["run_files_sha256"]) == set(adm.RUN_FILES), cid
    saved = Path(report["identities"]["energy-0"]["saved_run_directory"]).parts   # another folder, same parent
    assert saved[:len(root.parts) - 1] == root.parts[:-1] and saved[:len(root.parts)] != root.parts


@needs_real_files
def test_the_pin_report_is_the_policys_own_and_shows_python_drift(monkeypatch):
    import e3x_phase_common
    real = e3x_phase_common.source_pin()
    monkeypatch.setattr(e3x_phase_common, "source_pin", lambda: {**real, "python": "3.99.0"})
    report = adm.admit_all()
    assert report["pin_policy"]["recorded_differences"]["python"]["current"] == "3.99.0"
    assert report["current_pin"]["python"] == "3.99.0"


@needs_real_files
def test_the_report_says_how_the_root_was_chosen(monkeypatch):
    root = adm.group9_root()
    monkeypatch.setenv(adm.ROOT_ENVIRONMENT_VARIABLE, str(root))
    assert adm.admit_all()["root_source"] == "environment"
    assert adm.admit_all(root)["root_source"] == "argument"


@needs_real_files
def test_a_substituted_pin_is_only_reachable_through_the_private_entry_point():
    actual = copy.deepcopy(adm.accepted_source_pin())
    first = sorted(actual["code_files"])[0]
    actual["code_files"][first] = "0" * 64
    with pytest.raises(adm.PinMismatch, match=first):
        adm._admit_all(None, adm.PinPolicy(), actual)


@needs_real_files
def test_the_five_anchors_equal_the_records_and_appear_in_independent_files():
    root = adm.group9_root()
    records = json.loads((root / adm.RECOVERY_RECORDS).read_text())["models"]
    for cid in adm.PRIMARY_IDS:
        assert records[cid]["binding"] == adm.EXPECTED_PRIMARY[cid]
    paired = (root / "reports/execution/e3x-paired-trace-2026-10-18/LAUNCH_RECORDS.json").read_text()
    controlled = (root / "reports/execution/e3x-controlled-2026-10-16/RESULT_ACCEPTANCE.json").read_text()
    repair = (root / "reports/execution/e3x-score-repair-2026-10-16/RESULT_ACCEPTANCE.json").read_text()
    for cid, anchor in adm.EXPECTED_PRIMARY.items():
        assert anchor["checkpoint_sha256"] in paired and anchor["score_sha256"] in paired, cid
        accepted_text = repair if cid == "energy-0" else controlled
        assert anchor["checkpoint_sha256"] in accepted_text, cid
        if cid != "energy-0":
            assert anchor["score_sha256"] in controlled, cid


def linked_copy(tmp_path, item):
    """A copy of one model whose files are symlinks, so single files can be replaced."""
    item = copy.deepcopy(item)
    run, sd = tmp_path / "run", tmp_path / "score"
    run.mkdir(), sd.mkdir()
    for src, dst in ((Path(item["run_directory"]), run), (Path(item["score_directory"]), sd)):
        for file in src.iterdir():
            if file.is_file():
                (dst / file.name).symlink_to(file)
    item["run_directory"], item["score_directory"] = run, sd
    return item


def replace_json(directory, name, **changes):
    path = Path(directory) / name
    data = json.loads(path.read_text())
    path.unlink()
    data.update(changes)
    path.write_text(json.dumps(data))


@needs_real_files
def test_a_forged_accepted_model_is_refused_on_real_files(tmp_path):
    """Change a real model's checkpoint and rewrite its score.json and binding to agree."""
    item = linked_copy(tmp_path, {i["id"]: i for i in adm.load_bindings()}["direct-0"])
    forged = b"forged checkpoint"
    (item["run_directory"] / "best.msgpack").unlink()
    (item["run_directory"] / "best.msgpack").write_bytes(forged)
    replace_json(item["score_directory"], "score.json", best_checkpoint_sha256=sha(forged))
    item["acceptance"] = {**item["acceptance"], "checkpoint_sha256": sha(forged),
                          "score_sha256": sha((item["score_directory"] / "score.json").read_bytes())}
    problems = adm.identity_problems(item, adm.accepted_source_pin(), adm.load_inventories())
    assert any("independent anchor" in p for p in problems) and any("inventory" in p for p in problems)


@needs_real_files
def test_the_five_primary_bindings_equal_the_accepted_records_and_lambda_one_the_spec():
    items = {i["id"]: i for i in adm.load_bindings()}
    records = json.loads((adm.group9_root() / adm.RECOVERY_RECORDS).read_text())["models"]
    for cid in adm.PRIMARY_IDS:
        assert items[cid]["acceptance"] == records[cid]["binding"]
        assert items[cid]["launch_record"] == records[cid]["launch_record"]
    for cid, expected in adm.EXPECTED_LAMBDA1.items():
        assert items[cid]["acceptance"]["best_update"] == expected["best_update"]
        assert items[cid]["acceptance"]["checkpoint_sha256"].startswith(expected["checkpoint"][0])
        assert items[cid]["acceptance"]["score_sha256"].endswith(expected["score"][1])


@needs_real_files
def test_the_energy_baseline_is_seed_zero_and_the_controlled_energy_one_is_not_used():
    items = {i["id"]: i for i in adm.load_bindings()}
    config = items["energy-0"]["launch_record"]["config"]
    assert config["arm"] == "energy" and config["seed"] == 0
    assert "energy-1" not in items
    assert "aspirin-e3x-controlled" not in str(items["energy-0"]["run_directory"])


@needs_real_files
def test_all_seven_launch_records_share_the_accepted_pin_and_dataset_pin():
    items = adm.load_bindings()
    accepted = adm.accepted_source_pin()
    assert all(i["launch_record"]["source_pin"] == accepted for i in items)
    assert all(i["launch_record"]["dataset_pin"] == items[0]["launch_record"]["dataset_pin"] for i in items)


@needs_real_files
def test_the_inventories_cover_every_model_file():
    inventories = adm.load_inventories()
    for item in adm.load_bindings():
        which, entries = adm.inventory_entries(item)
        assert set(entries) <= set(inventories[which]), item["id"]


@needs_real_files
def test_a_tampered_binding_is_caught_on_real_files():
    accepted = adm.accepted_source_pin()
    for field, value in (("checkpoint_sha256", "0" * 64), ("score_sha256", "0" * 64), ("best_update", 1)):
        item = copy.deepcopy(adm.load_bindings()[1])
        item["run_directory"], item["score_directory"] = Path(item["run_directory"]), Path(item["score_directory"])
        item["acceptance"][field] = value
        assert adm.identity_problems(item, accepted, adm.load_inventories())


@needs_real_files
def test_a_changed_dataset_file_is_refused_on_real_files():
    item = copy.deepcopy(adm.load_bindings()[0])
    item["launch_record"]["dataset_pin"]["files"]["train.npz"] = "0" * 64
    with pytest.raises(adm.AdmissionError, match="train.npz"):
        adm.verify_dataset(adm.data_directory(), item["launch_record"])


# ---- The saved-JSON checks (real files) ----------------------------------

@needs_real_files
def test_the_saved_json_checks_pass_for_all_seven_models():
    for item in adm.load_bindings():
        assert adm.saved_json_problems(item) == [], item["id"]


@needs_real_files
@pytest.mark.parametrize("what, expected", [
    ("config", "run configuration differs from the launch record"),
    ("provenance", "run provenance differs from the summary"),
    ("summary-status", "not a complete accepted fit"),
    ("summary-stop-reason", "not a complete accepted fit"),
    ("summary-updates", "not a complete accepted fit"),
    ("summary-best-update", "not a complete accepted fit"),
    ("computation-hash", "objective or configuration differs"),
    ("provenance-python", "provenance python differs from the launch pin"),
    ("proof-update", "best-checkpoint proof"),
    ("proof-status", "best-checkpoint proof"),
])
def test_each_saved_json_check_can_fail(tmp_path, what, expected):
    item = linked_copy(tmp_path, adm.load_bindings()[1])
    run = item["run_directory"]
    if what == "config":
        replace_json(run, "config.json", seed=99)
    elif what == "provenance":
        replace_json(run, "provenance.json", arm="direct-changed")
    elif what == "summary-status":
        replace_json(run, "summary.json", status="failed")
    elif what == "summary-stop-reason":
        replace_json(run, "summary.json", stop_reason="wall_seconds")
    elif what == "summary-updates":
        replace_json(run, "summary.json", updates=1)
    elif what == "summary-best-update":
        replace_json(run, "summary.json", best_update=1)
    elif what == "computation-hash":
        summary = json.loads((run / "summary.json").read_text())
        provenance = {**summary["provenance"], "config_computation_hash": "0" * 64}
        replace_json(run, "provenance.json", **provenance)
        replace_json(run, "summary.json", provenance=provenance)
    elif what == "provenance-python":
        summary = json.loads((run / "summary.json").read_text())
        provenance = {**summary["provenance"], "python": "3.12.99"}
        replace_json(run, "provenance.json", **provenance)
        replace_json(run, "summary.json", provenance=provenance)
    elif what == "proof-update":
        replace_json(run, "best-checkpoint-verification.json", update=1)
    elif what == "proof-status":
        replace_json(run, "best-checkpoint-verification.json", status="failed")
    problems = adm.saved_json_problems(item)
    assert any(expected in p for p in problems), problems


@needs_real_files
def test_unreadable_run_files_are_a_problem_not_a_crash(tmp_path):
    item = linked_copy(tmp_path, adm.load_bindings()[1])
    (item["run_directory"] / "summary.json").unlink()
    assert any("unreadable" in p for p in adm.saved_json_problems(item))


# ---- admit_all wiring (real files) ----------------------------------------

@needs_real_files
def test_admit_all_refuses_when_any_identity_fails(monkeypatch):
    real = adm.identity_problems

    def one_failure(item, accepted, inventories, root=None, anchors=None):
        return ["forced failure"] if item["id"] == "direct-1" else real(item, accepted, inventories, root, anchors)

    monkeypatch.setattr(adm, "identity_problems", one_failure)
    with pytest.raises(adm.AdmissionError, match="direct-1"):
        adm.admit_all()


@needs_real_files
def test_admit_all_refuses_when_a_saved_json_check_fails(monkeypatch):
    monkeypatch.setattr(adm, "saved_json_problems", lambda item: ["forced"] if item["id"] == "energy-0" else [])
    with pytest.raises(adm.AdmissionError, match="energy-0"):
        adm.admit_all()


@needs_real_files
def test_admit_all_verifies_the_dataset_for_every_model_and_passes_the_inventories(monkeypatch):
    verified, inventories = [], []
    real_verify, real_problems = adm.verify_dataset, adm.identity_problems

    def spy_verify(directory, launch_record):
        verified.append(launch_record["config"]["seed"])
        return real_verify(directory, launch_record)

    def spy_problems(item, accepted, inventory, root=None, anchors=None):
        inventories.append(inventory)
        return real_problems(item, accepted, inventory, root, anchors)

    monkeypatch.setattr(adm, "verify_dataset", spy_verify)
    monkeypatch.setattr(adm, "identity_problems", spy_problems)
    adm.admit_all()
    assert len(verified) == 7 and len(inventories) == 7
    assert all(set(i) == {"controlled", "energy"} and i["controlled"] and i["energy"] for i in inventories)


@needs_real_files
def test_admit_all_refuses_a_dataset_that_does_not_match(monkeypatch):
    def refuse(directory, launch_record):
        raise adm.AdmissionError("dataset pin mismatch (forced)")

    monkeypatch.setattr(adm, "verify_dataset", refuse)
    with pytest.raises(adm.AdmissionError, match="dataset pin mismatch"):
        adm.admit_all()


@needs_real_files
def test_admit_all_runs_the_replay_plan_once_and_fails_if_it_fails(monkeypatch):
    calls, real = [], adm.replay_plan

    def spy(items, directory, split_hash=None):
        calls.append(len(items))
        return real(items, directory, split_hash)

    monkeypatch.setattr(adm, "replay_plan", spy)
    adm.admit_all()
    assert calls == [7]

    def refuse(items, directory, split_hash=None):
        raise adm.AdmissionError("replay inputs differ (forced)")

    monkeypatch.setattr(adm, "replay_plan", refuse)
    with pytest.raises(adm.AdmissionError, match="replay inputs differ"):
        adm.admit_all()


def altered_predictions(tmp_path, item, **changes):
    with np.load(Path(item["score_directory"]) / "best-validation.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    for key, value in changes.items():
        arrays[key] = value(arrays[key])
    directory = tmp_path / "score"
    directory.mkdir()
    np.savez(directory / "best-validation.npz", **arrays)
    return {**item, "score_directory": directory}


def with_nan(a):
    b = a.copy()
    b.reshape(-1)[5] = float("nan")
    return b


@needs_real_files
@pytest.mark.parametrize("what, change, expected", [
    ("record-ids", {"record_ids": lambda a: a[::-1].copy()}, "different records"),
    ("reference-forces", {"reference_forces": lambda a: a + 1e-3}, "reference forces differ"),
    ("shape", {"forces": lambda a: a[:-1].copy()}, "wrong shape"),
    ("not-finite", {"forces": with_nan}, "not finite float64"),
    ("float32", {"forces": lambda a: a.astype("float32")}, "not finite float64"),
    ("energies", {"energy": lambda a: a[:-1].copy()}, "energies"),
])
def test_the_replay_plan_refuses_predictions_that_do_not_fit_the_data(tmp_path, what, change, expected):
    item = copy.deepcopy(adm.load_bindings()[1])
    altered = altered_predictions(tmp_path, item, **change)
    with pytest.raises(adm.AdmissionError, match=expected):
        adm.replay_plan([altered], adm.data_directory())


@needs_real_files
def test_the_replay_plan_refuses_a_wrong_record_count(monkeypatch):
    item = copy.deepcopy(adm.load_bindings()[1])
    monkeypatch.setattr(adm, "REPLAY_RECORDS", 973)
    with pytest.raises(adm.AdmissionError, match="validation partition has 974"):
        adm.replay_plan([item], adm.data_directory())
