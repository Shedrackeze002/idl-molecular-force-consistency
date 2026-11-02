"""Admission for the Stage 2 movement-cap study (spec v4 section 5). No model is loaded.

Four pieces, all checkable from saved files:

- Pin policy. `code_files`, helpers, lockfile, dependencies and the code
  fingerprint must equal the accepted launch pin; Python, platform (runtime_cpu)
  and templates may differ and are recorded. Anything else fails closed. The
  policy is injected into the restore fork (restore_fork.py).
- Data directory. The run configurations still name a directory that no longer
  exists. The override is `<Group9>/data/processed/aspirin`, bound to the launch
  record's dataset pin (manifest identity plus the SHA-256 of four files). The
  closed test partition is never hashed or opened here.
- Seven checkpoint identities: the accepted five (from the accepted recovery
  launch records) and the two lambda = 1 fits (from their score.json, launch
  record and the controlled output inventory, tied to the identities the spec
  records). Run and score paths saved in the records point at a tree that has
  moved, so they are relocated, never trusted.
- The replay plan for launch A0: which models, 974 validation records, tolerance
  1e-10, and a data-only check that each saved prediction file lines up with the
  validation partition.

The restore itself (a model load) happens only in launch A0.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath

DEFAULT_GROUP9_ROOT = Path(__file__).resolve().parents[4]
ROOT_ENVIRONMENT_VARIABLE = "STAGE2_GROUP9_ROOT"

RECOVERY_RECORDS = "reports/execution/e3x-validation-recovery-2026-10-17/LAUNCH_RECORDS.json"
CONTROLLED_RUNS = "runs/aspirin-e3x-controlled-2026-10-16"
CONTROLLED_INVENTORY = "reports/execution/e3x-controlled-2026-10-16/COMPLETED_OUTPUT_INVENTORY.json"
DATA_DIRECTORY = "data/processed/aspirin"
DATASET_FILES = ("manifest.json", "train.npz", "validation.npz", "validation_minima_diagnostics.npz")

PRIMARY_IDS = ("energy-0", "direct-0", "consistency-0.1-0", "direct-1", "consistency-0.1-1")
LAMBDA1_IDS = ("consistency-1-0", "consistency-1-1")
ALL_IDS = PRIMARY_IDS + LAMBDA1_IDS

# What the spec (section 5) records for the lambda = 1 fits, as the coordinator and
# the engineering reviewer verified them. Full hashes are bound from the files; the
# check is that the files still carry these prefixes and suffixes.
EXPECTED_LAMBDA1 = {
    "consistency-1-0": {"best_update": 183000, "checkpoint": ("9f092d78", "2aa4"),
                        "score": ("f6aaee9c", "5c6")},
    "consistency-1-1": {"best_update": 196500, "checkpoint": ("5cb0d134", "b840"),
                        "score": ("848dff01", "35f")},
}
EXPECTED_CODE_FINGERPRINT_PREFIX = "a046011a"
EXPECTED_DATASET_IDENTITY_PREFIX = "4b9f2202"

REPLAY_RECORDS = 974
REPLAY_BATCH = 4
REPLAY_TOLERANCE = 1e-10

# Source-pin policy (spec section 5).
REQUIRED_EQUAL = ("code_fingerprint", "code_files", "lock_sha256", "dependencies", "helpers")
RECORDED_ONLY = ("python", "runtime_cpu", "templates")
PIN_KEYS = frozenset(REQUIRED_EQUAL + RECORDED_ONLY)


class AdmissionError(ValueError):
    """Admission refused: a pin, path, identity or data check failed."""


class PinMismatch(AdmissionError):
    """The current source differs from the accepted pin where equality is required."""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


# ---- Paths ---------------------------------------------------------------

def group9_root(root=None):
    root = Path(root or os.environ.get(ROOT_ENVIRONMENT_VARIABLE) or DEFAULT_GROUP9_ROOT)
    if not (root / "reports" / "execution").is_dir():
        raise AdmissionError(f"{root} is not the Group9 root (no reports/execution)")
    return root.resolve()


def relocate(path, root):
    """Map a path saved in the accepted records to the current root. The records were written
    when the project folder had another name but sat in the same parent folder, so everything
    up to and including that folder is replaced by `root` and the remainder is kept. The saved
    path must be absolute and lie under the root's parent folder; any other path is refused.
    A path already under the root maps to itself."""
    root = Path(root)
    saved = PurePosixPath(str(path))
    depth = len(root.parts)
    if not saved.is_absolute() or len(saved.parts) <= depth or saved.parts[:depth - 1] != root.parts[:depth - 1]:
        raise AdmissionError(f"saved path is outside the known roots: {path}")
    return root.joinpath(*saved.parts[depth:])


# ---- Pin policy ----------------------------------------------------------

def _dict_difference(actual, expected):
    changed = sorted(k for k in set(actual) & set(expected) if actual[k] != expected[k])
    added = sorted(set(actual) - set(expected))
    removed = sorted(set(expected) - set(actual))
    return {"changed": changed, "added": added, "removed": removed}


def compare_pins(actual, expected):
    """Apply the pin policy. Returns {"problems", "required_equal", "recorded_differences"}.

    Required keys must be equal. Recorded-only keys may differ and are listed with
    both values. A pin with any other or missing key fails closed."""
    problems, required, recorded = [], {}, {}
    for name, pin in (("current", actual), ("accepted", expected)):
        if not isinstance(pin, dict) or set(pin) != PIN_KEYS:
            extra = sorted(set(pin) - PIN_KEYS) if isinstance(pin, dict) else "not an object"
            missing = sorted(PIN_KEYS - set(pin)) if isinstance(pin, dict) else "not an object"
            problems.append(f"{name} pin has unexpected keys (extra {extra}, missing {missing})")
    if problems:
        return {"problems": problems, "required_equal": required, "recorded_differences": recorded}
    for key in REQUIRED_EQUAL:
        same = actual[key] == expected[key]
        required[key] = same
        if not same:
            detail = ""
            if isinstance(actual[key], dict) and isinstance(expected[key], dict):
                detail = " " + json.dumps(_dict_difference(actual[key], expected[key]))
            problems.append(f"{key} differs from the accepted pin{detail}")
    for key in RECORDED_ONLY:
        if actual[key] != expected[key]:
            recorded[key] = {"current": actual[key], "accepted": expected[key]}
    return {"problems": problems, "required_equal": required, "recorded_differences": recorded}


class PinPolicy:
    """Callable for restore_fork.e3x_checkpoint_arm(pin_policy=...). Raises PinMismatch
    if a required key differs; keeps every report so the launch record can state the
    recorded differences."""

    def __init__(self):
        self.reports = []

    def __call__(self, actual, expected):
        report = compare_pins(actual, expected)
        self.reports.append(report)
        if report["problems"]:
            raise PinMismatch("; ".join(report["problems"]))
        return report


def accepted_source_pin(root=None):
    """The source pin of the accepted recovery launch records."""
    return read_json(group9_root(root) / RECOVERY_RECORDS)["source_pin"]


def admit_source(policy, root=None, actual=None):
    """Apply the pin policy to the current source. `actual` defaults to source_pin()."""
    if actual is None:
        from e3x_phase_common import source_pin
        actual = source_pin()
    return policy(actual, accepted_source_pin(root))


# ---- Dataset -------------------------------------------------------------

def data_directory(root=None):
    directory = group9_root(root) / DATA_DIRECTORY
    if not directory.is_dir():
        raise AdmissionError(f"data directory {directory} does not exist")
    return directory


def verify_dataset(directory, launch_record):
    """Bind the data directory to the launch record's dataset pin: the manifest
    identity and the SHA-256 of the four files. The test partition is not read."""
    from e3x_phase_common import dataset_pin
    expected = launch_record["dataset_pin"]
    if set(expected["files"]) != set(DATASET_FILES):
        raise AdmissionError(f"dataset pin names unexpected files: {sorted(expected['files'])}")
    actual = dataset_pin(directory)
    if actual != expected:
        different = sorted(n for n in DATASET_FILES if actual["files"].get(n) != expected["files"].get(n))
        raise AdmissionError(f"dataset pin mismatch (files {different}, identity "
                             f"{'ok' if actual['manifest_identity'] == expected['manifest_identity'] else 'differs'})")
    return actual


# ---- The seven identities ------------------------------------------------

def load_bindings(root=None):
    """The seven models with relocated paths, launch records and acceptance bindings."""
    root = group9_root(root)
    records = read_json(root / RECOVERY_RECORDS)
    items = []
    for cid in PRIMARY_IDS:
        try:
            model = records["models"][cid]
        except KeyError:
            raise AdmissionError(f"accepted recovery records lack model {cid}") from None
        items.append({"id": cid, "role": "primary",
                      "run_directory": relocate(model["run_directory"], root),
                      "score_directory": relocate(model["score_directory"], root),
                      "launch_record": model["launch_record"],
                      "acceptance": dict(model["binding"]),
                      "source": RECOVERY_RECORDS})
    for cid in LAMBDA1_IDS:
        run = root / CONTROLLED_RUNS / cid
        score_directory = root / CONTROLLED_RUNS / f"{cid}-score"
        score = read_json(score_directory / "score.json")
        items.append({"id": cid, "role": "exploratory", "run_directory": run,
                      "score_directory": score_directory,
                      "launch_record": read_json(root / CONTROLLED_RUNS / f"{cid}-launch.json"),
                      "acceptance": {"checkpoint_sha256": score["best_checkpoint_sha256"],
                                     "best_update": score["best_update"],
                                     "score_sha256": digest((score_directory / "score.json").read_bytes())},
                      "source": f"{CONTROLLED_RUNS}/{cid}-score/score.json"})
    return items


def _affixes_ok(value, prefix_suffix):
    prefix, suffix = prefix_suffix
    return value.startswith(prefix) and value.endswith(suffix)


def identity_problems(item, accepted_pin, inventory=None, root=None):
    """Bytes-only checks of one model's identity (no model load). The first block
    mirrors the pre-restore checks of the frozen e3x_checkpoint_arm, minus the
    current-runtime pin; launch A0 runs the frozen ones for real."""
    problems = []
    run, score_directory = Path(item["run_directory"]), Path(item["score_directory"])
    launch, acceptance = item["launch_record"], item["acceptance"]
    checkpoint, score_path = run / "best.msgpack", score_directory / "score.json"
    predictions = score_directory / "best-validation.npz"
    for path in (checkpoint, score_path, predictions):
        if not path.is_file():
            return [f"missing file {path}"]
    score = read_json(score_path)
    checkpoint_sha = digest(checkpoint.read_bytes())
    score_sha = digest(score_path.read_bytes())
    if checkpoint_sha != acceptance["checkpoint_sha256"]:
        problems.append("best.msgpack differs from the accepted checkpoint hash")
    if score_sha != acceptance["score_sha256"]:
        problems.append("score.json differs from the accepted score hash")
    if type(acceptance["best_update"]) is not int or score["best_update"] != acceptance["best_update"]:
        problems.append("best_update differs from the accepted binding")
    if score["best_checkpoint_sha256"] != acceptance["checkpoint_sha256"]:
        problems.append("score.json names a different best checkpoint")
    if score["status"] != "verified":
        problems.append("score status is not verified")
    if score["source_pin"] != launch["source_pin"]:
        problems.append("score source pin differs from the launch record")
    if score["candidate"] != launch["candidate"]:
        problems.append("score candidate differs from the launch record")
    if any(score[k] is not True for k in ("source_matches", "best_reload_verified")):
        problems.append("score does not confirm source match and best reload")
    if digest(predictions.read_bytes()) != score["prediction_sha256"]:
        problems.append("saved predictions differ from the score's prediction hash")

    # The launch record belongs to the accepted source and dataset.
    if launch["source_pin"] != accepted_pin:
        problems.append("launch record source pin differs from the accepted pin")
    if not accepted_pin["code_fingerprint"].startswith(EXPECTED_CODE_FINGERPRINT_PREFIX):
        problems.append("accepted code fingerprint is not the one the spec records")
    if not launch["dataset_pin"]["manifest_identity"].startswith(EXPECTED_DATASET_IDENTITY_PREFIX):
        problems.append("launch dataset identity is not the one the spec records")

    if item["role"] == "exploratory":
        expected = EXPECTED_LAMBDA1[item["id"]]
        if acceptance["best_update"] != expected["best_update"]:
            problems.append("lambda = 1 best update differs from the spec's record")
        if not _affixes_ok(checkpoint_sha, expected["checkpoint"]):
            problems.append("lambda = 1 checkpoint hash differs from the spec's record")
        if not _affixes_ok(score_sha, expected["score"]):
            problems.append("lambda = 1 score hash differs from the spec's record")
    if inventory is not None and run.parent.name == Path(CONTROLLED_RUNS).name:
        # The six controlled fits are all in the completed-output inventory.
        launch_file = run.parent / f"{run.name}-launch.json"
        listed = {f"{run.name}/best.msgpack": checkpoint_sha,
                  f"{score_directory.name}/score.json": score_sha,
                  f"{score_directory.name}/best-validation.npz": digest(predictions.read_bytes()),
                  f"{run.name}-launch.json": digest(launch_file.read_bytes())
                  if launch_file.is_file() else None}
        for name, value in listed.items():
            if value is None:
                problems.append(f"launch file for {run.name} is missing")
            elif inventory.get(name) != value:
                problems.append(f"{name} is missing from or differs from the completed-output inventory")
    return problems


def identity_record(item):
    """The identity of one model for the launch record (hashes recomputed from files)."""
    run, score_directory = Path(item["run_directory"]), Path(item["score_directory"])
    score = read_json(score_directory / "score.json")
    return {"id": item["id"], "role": item["role"], "best_update": score["best_update"],
            "best_msgpack_sha256": digest((run / "best.msgpack").read_bytes()),
            "score_json_sha256": digest((score_directory / "score.json").read_bytes()),
            "predictions_sha256": digest((score_directory / "best-validation.npz").read_bytes()),
            "equal_connection_force_mae_eV_per_A": score.get("equal_connection_force_mae_eV_per_A"),
            "source": item["source"]}


def load_inventory(root=None):
    return read_json(group9_root(root) / CONTROLLED_INVENTORY)


# ---- Replay plan (launch A0) --------------------------------------------

def replay_plan(items, directory, split_hash=None):
    """Data-only check that each model's saved predictions line up with the
    validation partition, and the fixed replay settings for A0."""
    import numpy as np
    from e3x_phase_common import SPLIT
    from force_consistency.data.prepare import load_training_bundle
    bundle = load_training_bundle(directory, expected_split_hash=split_hash or SPLIT)
    part = bundle.validation
    if len(part.record_ids) != REPLAY_RECORDS:
        raise AdmissionError(f"validation partition has {len(part.record_ids)} records, "
                             f"not {REPLAY_RECORDS}")
    for item in items:
        with np.load(Path(item["score_directory"]) / "best-validation.npz", allow_pickle=False) as z:
            if z["record_ids"].tolist() != list(part.record_ids):
                raise AdmissionError(f"{item['id']}: saved predictions list different records")
            if not np.array_equal(z["reference_forces"], part.forces):
                raise AdmissionError(f"{item['id']}: saved reference forces differ from the data")
            if z["forces"].shape != part.forces.shape:
                raise AdmissionError(f"{item['id']}: saved forces have the wrong shape")
    return {"models": [item["id"] for item in items], "records": REPLAY_RECORDS,
            "batch": REPLAY_BATCH, "rtol": REPLAY_TOLERANCE, "atol": REPLAY_TOLERANCE,
            "data_directory": str(directory)}


# ---- Everything launch A0 needs before it loads a model ------------------

def admit_all(root=None, policy=None, actual_pin=None):
    """Run every no-model admission check. Raises on the first failure; returns the
    report that goes into launch record A."""
    root = group9_root(root)
    policy = policy or PinPolicy()
    admit_source(policy, root, actual_pin)
    accepted = accepted_source_pin(root)
    inventory = load_inventory(root)
    items = load_bindings(root)
    directory = data_directory(root)
    identities, failures = {}, {}
    for item in items:
        problems = identity_problems(item, accepted, inventory, root)
        if problems:
            failures[item["id"]] = problems
        else:
            identities[item["id"]] = identity_record(item)
            verify_dataset(directory, item["launch_record"])
    if failures:
        raise AdmissionError("identity checks failed: " + json.dumps(failures))
    return {"pin_policy": policy.reports[-1], "data_directory": str(directory),
            "dataset_pin": items[0]["launch_record"]["dataset_pin"],
            "identities": identities, "replay_plan": replay_plan(items, directory)}
