"""M3: pin policy, data-directory binding, the seven checkpoint identities and the
replay plan. Synthetic fixtures always run; checks against the saved Group9 files
skip when those files are absent. No model is loaded."""
import copy
import hashlib
import json
from pathlib import Path

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
            "runtime_cpu": {"platform": "macOS-1", "logical_cpus": 8},
            "helpers": {"e3x_phase_common.py": "3" * 64},
            "templates": {"aspirin-e3x-a.json": "4" * 64}}
    base.update(changes)
    return base


# ---- Pin policy ----------------------------------------------------------

def test_identical_pins_pass_with_nothing_recorded():
    report = adm.compare_pins(pin(), pin())
    assert report["problems"] == [] and report["recorded_differences"] == {}
    assert all(report["required_equal"].values()) and set(report["required_equal"]) == set(adm.REQUIRED_EQUAL)


def test_python_platform_and_template_drift_is_recorded_not_fatal():
    drifted = pin(python="3.12.16", runtime_cpu={"platform": "macOS-2", "logical_cpus": 8},
                  templates={"aspirin-e3x-a.json": "5" * 64})
    report = adm.compare_pins(drifted, pin())
    assert report["problems"] == []
    assert set(report["recorded_differences"]) == {"python", "runtime_cpu", "templates"}
    assert report["recorded_differences"]["python"] == {"current": "3.12.16", "accepted": "3.12.13"}


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


@pytest.mark.parametrize("bad", [{"extra": 1}, {"missing": True}, "not a pin", None])
def test_a_pin_with_other_keys_or_a_wrong_type_fails_closed(bad):
    odd = pin()
    if bad == {"extra": 1}:
        odd["extra"] = 1
    elif bad == {"missing": True}:
        del odd["templates"]
    else:
        odd = bad
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

def test_relocate_maps_a_previous_root_keeps_new_paths_and_refuses_strangers(tmp_path):
    (tmp_path / "reports" / "execution").mkdir(parents=True)
    root = adm.group9_root(tmp_path)
    previous = str(root.parent / "previous-root") + "/"          # same parent folder, another project folder name
    assert adm.relocate(previous + "runs/x/best.msgpack", root) == root / "runs/x/best.msgpack"
    assert adm.relocate(str(root / "runs/y"), root) == root / "runs/y"
    with pytest.raises(adm.AdmissionError):
        adm.relocate("/somewhere/else/runs/x", root)


def test_group9_root_must_look_like_group9(tmp_path, monkeypatch):
    with pytest.raises(adm.AdmissionError):
        adm.group9_root(tmp_path)
    (tmp_path / "reports" / "execution").mkdir(parents=True)
    monkeypatch.setenv(adm.ROOT_ENVIRONMENT_VARIABLE, str(tmp_path))
    assert adm.group9_root() == tmp_path.resolve()


# ---- Identity checks on fabricated files ---------------------------------

def fabricate(tmp_path, *, name="m", controlled=False, role="primary"):
    base = tmp_path / (Path(adm.CONTROLLED_RUNS).name if controlled else "runs")
    run, score_directory = base / name, base / f"{name}-score"
    run.mkdir(parents=True), score_directory.mkdir()
    accepted = pin()
    launch = {"source_pin": accepted, "candidate": "C",
              "dataset_pin": {"manifest_identity": "4b9f2202" + "0" * 56,
                              "files": {n: "5" * 64 for n in adm.DATASET_FILES}}}
    checkpoint = b"checkpoint bytes"
    predictions = b"prediction bytes"
    (run / "best.msgpack").write_bytes(checkpoint)
    (score_directory / "best-validation.npz").write_bytes(predictions)
    score = {"best_update": 1500, "best_checkpoint_sha256": sha(checkpoint), "status": "verified",
             "source_pin": accepted, "candidate": "C", "source_matches": True,
             "best_reload_verified": True, "prediction_sha256": sha(predictions),
             "equal_connection_force_mae_eV_per_A": 0.038}
    (score_directory / "score.json").write_text(json.dumps(score))
    launch_file = base / f"{name}-launch.json"
    launch_file.write_text(json.dumps(launch))
    item = {"id": name, "role": role, "run_directory": run, "score_directory": score_directory,
            "launch_record": launch, "source": "fabricated",
            "acceptance": {"checkpoint_sha256": sha(checkpoint), "best_update": 1500,
                           "score_sha256": sha((score_directory / "score.json").read_bytes())}}
    inventory = {f"{name}/best.msgpack": sha(checkpoint),
                 f"{name}-score/score.json": item["acceptance"]["score_sha256"],
                 f"{name}-score/best-validation.npz": sha(predictions),
                 f"{name}-launch.json": sha(launch_file.read_bytes())}
    return item, accepted, inventory


def test_a_consistent_fabricated_model_passes(tmp_path):
    item, accepted, inventory = fabricate(tmp_path, controlled=True)
    assert adm.identity_problems(item, accepted, inventory) == []
    assert adm.identity_record(item)["best_msgpack_sha256"] == item["acceptance"]["checkpoint_sha256"]


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
    item, accepted, _ = fabricate(tmp_path)
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
    problems = adm.identity_problems(item, accepted)
    assert problems
    expected_text = {"score-pin-only": "score source pin", "launch-and-score-pin-vs-accepted":
                     "launch record source pin differs from the accepted pin"}
    if what in expected_text:
        assert any(expected_text[what] in p for p in problems), problems


def test_an_accepted_pin_other_than_the_one_the_spec_records_is_refused(tmp_path):
    item, accepted, _ = fabricate(tmp_path)
    other = pin(code_fingerprint="c" * 64)
    item["launch_record"] = {**item["launch_record"], "source_pin": other}
    rewrite_score(item, source_pin=other)
    item["acceptance"]["score_sha256"] = sha((Path(item["score_directory"]) / "score.json").read_bytes())
    assert any("spec records" in p for p in adm.identity_problems(item, other))


def test_the_inventory_must_list_each_controlled_file_with_the_same_hash(tmp_path):
    item, accepted, inventory = fabricate(tmp_path, controlled=True)
    for name in inventory:
        wrong = {**inventory, name: "0" * 64}
        missing = {k: v for k, v in inventory.items() if k != name}
        assert adm.identity_problems(item, accepted, wrong), name
        assert adm.identity_problems(item, accepted, missing), name


def test_a_missing_launch_file_is_a_problem_for_a_controlled_model(tmp_path):
    item, accepted, inventory = fabricate(tmp_path, controlled=True)
    (Path(item["run_directory"]).parent / "m-launch.json").unlink()
    assert any("launch file for m is missing" in p
               for p in adm.identity_problems(item, accepted, inventory))


def test_the_inventory_is_not_required_for_models_outside_the_controlled_run(tmp_path):
    item, accepted, _ = fabricate(tmp_path, controlled=False)
    assert adm.identity_problems(item, accepted, {}) == []


def test_lambda_one_must_match_the_identities_the_spec_records(tmp_path):
    item, accepted, inventory = fabricate(tmp_path, name="consistency-1-0", controlled=True,
                                          role="exploratory")
    problems = adm.identity_problems(item, accepted, inventory)
    assert any("best update differs from the spec" in p for p in problems)
    assert any("checkpoint hash differs from the spec" in p for p in problems)
    assert any("score hash differs from the spec" in p for p in problems)


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


# ---- Static guarantees ---------------------------------------------------

def test_admission_loads_no_model_and_never_names_the_test_partition():
    source = Path(adm.__file__).read_text()
    for forbidden in ("load_checkpoint", "model_and_predictor", "initialize_state",
                      "jax.jit", "msgpack_restore", "from_bytes", "import restore_fork",
                      "restore_fork import", "e3x_adapter import"):
        assert forbidden not in source, forbidden
    for module in Path(adm.__file__).parent.glob("*.py"):
        assert "test.npz" not in module.read_text(), module.name


# ---- The real saved files ------------------------------------------------

@needs_real_files
def test_all_seven_identities_admit_from_the_saved_files():
    report = adm.admit_all()
    assert list(report["identities"]) == list(adm.ALL_IDS)
    assert [report["identities"][i]["role"] for i in adm.ALL_IDS] == ["primary"] * 5 + ["exploratory"] * 2
    assert all(report["pin_policy"]["required_equal"].values())
    assert set(report["pin_policy"]["recorded_differences"]) <= set(adm.RECORDED_ONLY)
    plan = report["replay_plan"]
    assert plan["records"] == 974 and plan["rtol"] == 1e-10 and plan["models"] == list(adm.ALL_IDS)


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
    assert all(i["launch_record"]["dataset_pin"] == items[0]["launch_record"]["dataset_pin"]
               for i in items)


@needs_real_files
def test_a_tampered_binding_is_caught_on_real_files():
    accepted = adm.accepted_source_pin()
    for field, value in (("checkpoint_sha256", "0" * 64), ("score_sha256", "0" * 64),
                         ("best_update", 1)):
        item = copy.deepcopy(adm.load_bindings()[1])
        item["run_directory"] = Path(item["run_directory"])
        item["score_directory"] = Path(item["score_directory"])
        item["acceptance"][field] = value
        assert adm.identity_problems(item, accepted, adm.load_inventory())


@needs_real_files
def test_a_changed_code_file_is_refused_at_admission():
    actual = copy.deepcopy(adm.accepted_source_pin())
    first = sorted(actual["code_files"])[0]
    actual["code_files"][first] = "0" * 64
    with pytest.raises(adm.PinMismatch, match=first):
        adm.admit_all(actual_pin=actual)


@needs_real_files
def test_a_changed_dataset_file_is_refused_on_real_files():
    item = copy.deepcopy(adm.load_bindings()[0])
    item["launch_record"]["dataset_pin"]["files"]["train.npz"] = "0" * 64
    with pytest.raises(adm.AdmissionError, match="train.npz"):
        adm.verify_dataset(adm.data_directory(), item["launch_record"])


@needs_real_files
def test_the_replay_plan_refuses_predictions_that_do_not_match_the_data(tmp_path):
    import numpy as np
    item = copy.deepcopy(adm.load_bindings()[1])
    source = Path(item["score_directory"]) / "best-validation.npz"
    with np.load(source, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["reference_forces"] = arrays["reference_forces"] + 1e-3
    fake_score = tmp_path / "score"
    fake_score.mkdir()
    np.savez(fake_score / "best-validation.npz", **arrays)
    item["score_directory"] = fake_score
    with pytest.raises(adm.AdmissionError, match="reference forces differ"):
        adm.replay_plan([item], adm.data_directory())


def _predictions_copy(tmp_path, item, **changes):
    import numpy as np
    with np.load(Path(item["score_directory"]) / "best-validation.npz", allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    for key, value in changes.items():
        arrays[key] = value(arrays[key])
    directory = tmp_path / "score"
    directory.mkdir()
    np.savez(directory / "best-validation.npz", **arrays)
    return {**item, "score_directory": directory}


@needs_real_files
def test_the_replay_plan_refuses_other_record_ids_a_wrong_shape_and_a_wrong_count(tmp_path, monkeypatch):
    item = copy.deepcopy(adm.load_bindings()[1])
    directory = adm.data_directory()
    (tmp_path / "ids").mkdir()
    ids = _predictions_copy(tmp_path / "ids", item, record_ids=lambda a: a[::-1].copy())
    with pytest.raises(adm.AdmissionError, match="different records"):
        adm.replay_plan([ids], directory)
    (tmp_path / "shape").mkdir()
    short = _predictions_copy(tmp_path / "shape", item, forces=lambda a: a[:-1].copy())
    with pytest.raises(adm.AdmissionError, match="wrong shape"):
        adm.replay_plan([short], directory)
    monkeypatch.setattr(adm, "REPLAY_RECORDS", 973)
    with pytest.raises(adm.AdmissionError, match="validation partition has 974"):
        adm.replay_plan([item], directory)


@needs_real_files
def test_admit_all_refuses_when_any_identity_fails(monkeypatch):
    real = adm.identity_problems

    def one_failure(item, accepted, inventory=None, root=None):
        return ["forced failure"] if item["id"] == "direct-1" else real(item, accepted, inventory, root)

    monkeypatch.setattr(adm, "identity_problems", one_failure)
    with pytest.raises(adm.AdmissionError, match="direct-1"):
        adm.admit_all()


@needs_real_files
def test_admit_all_verifies_the_dataset_for_every_model_and_passes_the_inventory(monkeypatch):
    verified, inventories = [], []
    real_verify, real_problems = adm.verify_dataset, adm.identity_problems

    def spy_verify(directory, launch_record):
        verified.append(launch_record["config"]["seed"])
        return real_verify(directory, launch_record)

    def spy_problems(item, accepted, inventory=None, root=None):
        inventories.append(inventory)
        return real_problems(item, accepted, inventory, root)

    monkeypatch.setattr(adm, "verify_dataset", spy_verify)
    monkeypatch.setattr(adm, "identity_problems", spy_problems)
    adm.admit_all()
    assert len(verified) == 7 and len(inventories) == 7
    assert all(isinstance(i, dict) and i for i in inventories)


@needs_real_files
def test_admit_all_refuses_a_dataset_that_does_not_match(monkeypatch):
    def refuse(directory, launch_record):
        raise adm.AdmissionError("dataset pin mismatch (forced)")

    monkeypatch.setattr(adm, "verify_dataset", refuse)
    with pytest.raises(adm.AdmissionError, match="dataset pin mismatch"):
        adm.admit_all()
