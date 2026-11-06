"""Shared fixtures for the launch tests: a committed scratch copy of the tree, records built from the real
files, an approval, a valid A0 result and a valid stage directory.

A launch-level test needs a clean git tree whose HEAD is the commit the record names, so a test never
runs against the live worktree: `scratch_base` copies the directories the mutation method lists
(decision log C20) into a temporary folder, runs `git init` and commits once; `scratch` is a private
copy of that committed tree for one test. There is no switch that skips the tree checks.
"""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from stage2_capped import admission, driver, launch_checks, launch_record, launch_state, stage0_compare

REPO = Path(__file__).resolve().parents[2]
TREE_ITEMS = ("evaluation", "stage2_capped", "tests", "manifests", "scripts", "configs", "src",
              "pyproject.toml", "uv.lock")
IDENTITY = ("Launch Test", "@".join(("tester", "scratch.invalid")))


def git(cwd, *args):
    ident = ("-c", f"user.name={IDENTITY[0]}", "-c", f"user.email={IDENTITY[1]}")
    return subprocess.run(["git", *ident, *args], cwd=cwd, check=True, capture_output=True).stdout


def real_files():
    try:
        root = admission.group9_root()
        return (root / admission.DATA_DIRECTORY).is_dir() and (root / admission.RECOVERY_RECORDS).is_file()
    except admission.AdmissionError:
        return False


needs_real_files = pytest.mark.skipif(not real_files(), reason="saved Group9 files are absent")


def build_scratch_base(destination):
    destination = Path(destination)
    destination.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc", "*.partial")
    for item in TREE_ITEMS:
        source = REPO / item
        if source.is_dir():
            shutil.copytree(source, destination / item, ignore=ignore)
        else:
            shutil.copy2(source, destination / item)
    git(destination, "init", "-q")
    git(destination, "add", "-A")
    git(destination, "commit", "-q", "-m", "scratch tree")
    return destination


@pytest.fixture(scope="session")
def scratch_base(tmp_path_factory):
    return build_scratch_base(tmp_path_factory.mktemp("scratch-base") / "tree")


@pytest.fixture
def scratch(scratch_base, tmp_path):
    destination = tmp_path / "tree"
    shutil.copytree(scratch_base, destination, symlinks=True)
    return destination


# ---- Records -----------------------------------------------------------------------------------

def sha256_file(path):
    return launch_checks.sha256_file(path)


def build_record(tree, *, kind, launch_id, root, volume, prior=(), a0=None, commit=None, launch_folder=None,
                 ceilings=None):
    """A record whose every hash is computed from the real files and the given tree. The launch folder defaults to
    `launch` beside the output root; `ceilings` overrides the spec's values (a wall time above the floor)."""
    group9 = admission.group9_root()
    gates, problems = launch_checks.diff_gate_hashes(tree)
    assert not problems, problems
    record = {"schema": 1, "record": kind, "commit": commit or launch_checks.head_commit(tree),
              "launch_id": launch_id, "output_root": str(root), "output_volume": volume,
              "root_source": admission.root_source(),
              "launch_folder": str(launch_folder or Path(root).parent / "launch"),
              "expected_launch_records_sha256": sha256_file(group9 / admission.RECOVERY_RECORDS),
              "saved_file_hashes": stage0_compare.saved_file_hashes(group9),
              "frozen_file_hashes": dict(launch_checks.FROZEN_FILES), "diff_gate_hashes": gates,
              "rules_py_sha256": sha256_file(Path(tree) / "stage2_capped" / "rules.py"),
              "relaxer_sha256": driver.relaxer_sha256(), "ceilings": ceilings or launch_record.expected_ceilings(),
              "ceiling_cap_factor": launch_record.CEILING_CAP_FACTOR,
              "caffeinate": list(launch_record.CAFFEINATE_ARGV), "prior_dispatches": [list(p) for p in prior]}
    if kind == "B":
        a0 = a0 or {"record_a_sha256": "a" * 64, "a0_launch_id": "A0-launch", "a0_output_root": "/tmp/a0-root"}
        record.update(a0)
    return record


def write_record(directory, record):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"RECORD_{record['record']}.json"
    data = launch_state.dumps(record)
    path.write_bytes(data)
    return path, launch_record.sha256_bytes(data)


def write_approval(directory, record_sha256, name="approval"):
    directory = Path(directory)
    reviews = {}
    for number in (1, 2):
        review = directory / f"review-{name}-{number}.md"
        review.write_text(f"independent review {number} of {record_sha256}\n")
        reviews[str(review)] = launch_record.sha256_bytes(review.read_bytes())
    path = directory / f"{name}.json"
    path.write_text(json.dumps({"status": launch_record.APPROVAL_STATUS, "record_sha256": record_sha256,
                                "review_sha256": reviews, "approved_by": "test approver",
                                "approval_line": "approved for this launch"}))
    return path


# ---- A0 result and stage directories -------------------------------------------------------------

FIVE = ["rec-0", "rec-1", "rec-2", "rec-3", "rec-4"]
GROUPS = ["13", "18", "18_2", "66", "78"]


def make_admission_report(record_sha256=None):
    identities = {}
    for model_id in admission.ALL_IDS:
        identities[model_id] = {"id": model_id, "best_update": 100, "best_msgpack_sha256": model_id[:1] * 64,
                                "score_json_sha256": "5" * 64}
    return {"launch_records_sha256": record_sha256, "identities": identities,
            "dataset_pin": {"manifest_identity": "dataset-identity"},
            "accepted_pin": {"code_fingerprint": "fingerprint"}}


def make_a0_result(record_a_sha256="a" * 64, launch_id="A0-launch", output_root="/tmp/a0-root", commit="c" * 40,
                   launch_records_sha256="f" * 64, runtime=None, latency=0.002):
    models = {}
    report = make_admission_report(launch_records_sha256)
    for model_id in admission.ALL_IDS:
        arm, seed, lam = driver.MODEL_IDENTITY[model_id]
        accepted = report["identities"][model_id]
        models[model_id] = {
            "replay": {"status": "verified", "record_count": admission.REPLAY_RECORDS, "rtol": admission.REPLAY_TOLERANCE,
                       "atol": admission.REPLAY_TOLERANCE, "single_record_ids": list(FIVE),
                       "maximum_absolute_force_difference": 1e-13,
                       "checkpoint_identity": {"checkpoint_sha256": accepted["best_msgpack_sha256"],
                                               "score_sha256": accepted["score_json_sha256"],
                                               "best_update": accepted["best_update"], "arm": arm, "seed": seed,
                                               "lambda": lam, "dataset_identity": "dataset-identity",
                                               "source_fingerprint": "fingerprint"}},
            "saved_force_abs_max": 40.0,
            "evidence": {"batch4": {"max_scaled_residual": 0.001},
                         "batch1_latency": {"mean_seconds": latency, "p99_seconds": latency * 3}}}
    return {"record_sha256": record_a_sha256, "launch_id": launch_id, "output_root": str(output_root),
            "commit": commit, "admission": report, "models": models,
            "replay_scope": {"first_five_record_ids": list(FIVE), "groups": list(GROUPS)},
            "runtime": runtime if runtime is not None else {"jax_backend": "cpu"}}


def make_stage_directory(root, stage, record, launch_id, relaxer_sha256, *, runs=None, force_evaluations=1000,
                         custom=False, complete=True, manifest_overrides=None, marker_overrides=None):
    """A stage directory that `marker_problems` accepts; keyword arguments break one thing at a time."""
    spec = driver.STAGE_SPECS[stage]
    directory = Path(root) / "rows" / f"stage{stage}"
    runs = spec["runs"] if runs is None else runs
    for index in range(runs):
        path = directory / f"model{index % 5}" / "cinf" / f"row-{index}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n")
    manifest = {"stage": stage, "launch_id": launch_id, "custom_tasks": custom, "runs": runs,
                "relaxer_sha256": relaxer_sha256}
    if stage == "0":
        manifest["stage0"] = {"verdict": {"proceed": True}, "compared": spec["runs"],
                              "saved_file_hashes": record["saved_file_hashes"]}
    manifest.update(manifest_overrides or {})
    manifest_bytes = launch_state.dumps(manifest)
    (directory / "STAGE_MANIFEST.json").write_bytes(manifest_bytes)
    marker = {"stage": stage, "runs": spec["runs"], "force_evaluations": force_evaluations,
              "attempt_ceiling": spec["attempt_ceiling"], "complete": complete, "custom_tasks": custom,
              "manifest_sha256": launch_record.sha256_bytes(manifest_bytes)}
    marker.update(marker_overrides or {})
    marker_bytes = launch_state.dumps(marker)
    (directory / "STAGE_COMPLETE.json").write_bytes(marker_bytes)
    return directory, launch_record.sha256_bytes(marker_bytes)
