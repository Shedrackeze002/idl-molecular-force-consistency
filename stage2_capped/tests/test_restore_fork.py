"""M3: the named fork of the frozen E3x restore code. No model is loaded: every test
stops before a checkpoint is deserialized."""
import inspect
import json
from pathlib import Path

import pytest

from stage2_capped import admission, diff_gate, restore_fork

ALLOWLIST = diff_gate.ROOT / "stage2_capped/allowlist_restore.json"
FROZEN = "evaluation/e3x_adapter.py"
FORK = "stage2_capped/restore_fork.py"


def real_files():
    try:
        root = admission.group9_root()
        return all((root / "runs/aspirin-e3x-controlled-2026-10-16" / m / "best.msgpack").is_file()
                   for m in ("direct-0", "consistency-0.1-0"))
    except admission.AdmissionError:
        return False


needs_real_files = pytest.mark.skipif(not real_files(), reason="saved Group9 run files are absent")


class Stop(Exception):
    """Raised by a patched function to end a restore at a known point."""


# ---- Diff gate -----------------------------------------------------------

def test_diff_gate_passes_on_the_committed_files():
    diff_gate.assert_gate("restore_fork", allowlist_path=ALLOWLIST)


def test_the_allowlist_pins_the_frozen_file_and_lists_only_three_categories():
    entry = json.loads(ALLOWLIST.read_text())["targets"]["restore_fork"]
    assert entry["frozen_file"] == FROZEN
    assert entry["frozen_sha256"] == diff_gate.sha256_file(diff_gate.ROOT / FROZEN)
    assert set(entry["allowed_categories"]) == {"module_header", "pin_policy", "data_directory"}
    assert len(entry["hunks"]) == 6
    for hunk in entry["hunks"]:
        assert hunk["purpose"] and set(hunk["categories"]) <= set(entry["allowed_categories"])
    assert entry["frozen_sha256"] in (diff_gate.ROOT / FORK).read_text()  # the fork names its source


def _root_with(tmp_path, frozen_bytes, fork_bytes):
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "stage2_capped").mkdir()
    (tmp_path / FROZEN).write_bytes(frozen_bytes)
    (tmp_path / FORK).write_bytes(fork_bytes)
    return tmp_path


def test_diff_gate_catches_an_unlisted_change(tmp_path):
    frozen = (diff_gate.ROOT / FROZEN).read_bytes()
    text = (diff_gate.ROOT / FORK).read_text()
    # Restore the old (broken) behavior of one allow-listed line AND change another.
    tampered = text.replace("or score['status']!='verified'", "or score['status']!='anything'")
    assert tampered != text
    root = _root_with(tmp_path, frozen, tampered.encode())
    problems = diff_gate.check_gate("restore_fork", allowlist_path=ALLOWLIST, root=root)
    assert problems and any("hunk" in p for p in problems)


@pytest.mark.parametrize("old, new", [
    # a pin policy that is never called
    ("    pin_policy(source_pin(),launch_record['source_pin'])\n", "    pass\n"),
    # a data directory read from the saved configuration after all
    ("require_equal_pin(dataset_pin(data_directory),launch['dataset_pin'])",
     "require_equal_pin(dataset_pin(cfg.data_directory),launch['dataset_pin'])"),
    # a loosened frozen comparison of saved files
    ("if asdict(cfg)!=launch['config'] or prov!=summary['provenance']:",
     "if prov!=summary['provenance']:"),
])
def test_diff_gate_catches_each_kind_of_edit(tmp_path, old, new):
    frozen = (diff_gate.ROOT / FROZEN).read_bytes()
    text = (diff_gate.ROOT / FORK).read_text()
    assert text.count(old) == 1
    root = _root_with(tmp_path, frozen, text.replace(old, new).encode())
    assert diff_gate.check_gate("restore_fork", allowlist_path=ALLOWLIST, root=root)


def test_diff_gate_catches_a_changed_frozen_file(tmp_path):
    root = _root_with(tmp_path, (diff_gate.ROOT / FROZEN).read_bytes() + b"# edited\n",
                      (diff_gate.ROOT / FORK).read_bytes())
    problems = diff_gate.check_gate("restore_fork", allowlist_path=ALLOWLIST, root=root)
    assert any("pinned SHA-256" in p for p in problems)


def test_diff_gate_catches_a_missing_hunk_and_a_bad_category(tmp_path):
    allowlist = json.loads(ALLOWLIST.read_text())
    short = json.loads(json.dumps(allowlist))
    short["targets"]["restore_fork"]["hunks"] = short["targets"]["restore_fork"]["hunks"][:-1]
    path = tmp_path / "short.json"
    path.write_text(json.dumps(short))
    assert diff_gate.check_gate("restore_fork", allowlist_path=path)
    bad = json.loads(json.dumps(allowlist))
    bad["targets"]["restore_fork"]["hunks"][2]["categories"] = ["data_directory", "other"]
    path2 = tmp_path / "bad.json"
    path2.write_text(json.dumps(bad))
    assert any("category" in p for p in diff_gate.check_gate("restore_fork", allowlist_path=path2))


# ---- Interface -----------------------------------------------------------

def test_the_data_directory_and_the_pin_policy_are_required_keywords():
    parameters = inspect.signature(restore_fork.e3x_checkpoint_arm).parameters
    for name in ("acceptance", "data_directory", "pin_policy"):
        assert parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
        assert parameters[name].default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        restore_fork.e3x_checkpoint_arm("run", {}, "score", acceptance={})


def test_the_pin_policy_runs_first_and_its_refusal_stops_everything(tmp_path):
    calls = []

    def policy(actual, expected):
        calls.append((set(actual), expected))
        raise admission.PinMismatch("refused")

    launch = {"source_pin": {"marker": 1}}
    with pytest.raises(admission.PinMismatch):  # not FileNotFoundError: no file was read
        restore_fork.e3x_checkpoint_arm(tmp_path / "no-run", launch, tmp_path / "no-score",
                                        acceptance={}, data_directory=tmp_path, pin_policy=policy)
    assert len(calls) == 1 and calls[0][1] == {"marker": 1}
    assert admission.PIN_KEYS <= calls[0][0]  # it was given the real current source pin


def test_after_the_policy_the_frozen_binding_checks_still_run(tmp_path):
    run, score_directory = tmp_path / "run", tmp_path / "score"
    run.mkdir(), score_directory.mkdir()
    (run / "best.msgpack").write_bytes(b"checkpoint")
    (score_directory / "score.json").write_text(json.dumps({"best_update": 1}))
    acceptance = {"checkpoint_sha256": "0" * 64, "score_sha256": "1" * 64, "best_update": 1}
    with pytest.raises(ValueError, match="Accepted BEST/score binding mismatch"):
        restore_fork.e3x_checkpoint_arm(run, {"source_pin": {}}, score_directory,
                                        acceptance=acceptance, data_directory=tmp_path,
                                        pin_policy=lambda actual, expected: None)


# ---- The data directory is the override, not the saved path (real files) -

@needs_real_files
def test_the_override_reaches_the_dataset_pin_and_the_saved_path_is_never_used(monkeypatch):
    items = {item["id"]: item for item in admission.load_bindings()}
    item = items["direct-0"]
    override = admission.data_directory()
    seen = {}

    def stop_at_dataset_pin(directory):
        seen["directory"] = Path(directory)
        raise Stop

    def must_not_be_reached(*args, **kwargs):
        raise AssertionError("the bundle would be loaded: the run went past the dataset pin")

    monkeypatch.setattr(restore_fork, "dataset_pin", stop_at_dataset_pin)
    import force_consistency.data.prepare as prepare
    monkeypatch.setattr(prepare, "load_training_bundle", must_not_be_reached)
    with pytest.raises(Stop):
        restore_fork.e3x_checkpoint_arm(
            item["run_directory"], item["launch_record"], item["score_directory"],
            acceptance=item["acceptance"], data_directory=override,
            pin_policy=admission.PinPolicy())
    saved = json.loads((Path(item["run_directory"]) / "config.json").read_text())["data_directory"]
    root_parts = admission.group9_root().parts
    saved_parts = Path(saved).parts          # written under another project folder in the same parent
    assert saved_parts[:len(root_parts) - 1] == root_parts[:-1] and saved_parts[:len(root_parts)] != root_parts
    assert not Path(saved).exists()
    assert seen["directory"] == override and str(seen["directory"]) != saved


# ---- No model loading, no test partition ---------------------------------

def test_the_fork_adds_no_model_loading_and_no_test_partition_access():
    frozen = (diff_gate.ROOT / FROZEN).read_text().split("\n")
    fork = (diff_gate.ROOT / FORK).read_text().split("\n")
    added = [line for line in fork if line not in frozen]
    text = "\n".join(added)
    for forbidden in ("load_checkpoint", "model_and_predictor", "initialize_state", "test.npz",
                      "bundle.test", "jax"):
        assert forbidden not in text
