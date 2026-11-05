"""Admission for the Stage 2 movement-cap study (spec v4 section 5). No model is loaded.

Five pieces, all checkable from saved files:

- Pin policy. `code_files`, helpers, lockfile, dependencies and the code fingerprint
  must equal the accepted launch pin; Python, the platform string and the templates may
  differ and are recorded. Every other CPU fact (machine, processor, logical CPUs, thread
  and XLA environment) must also be equal. Anything else fails closed. The policy is a
  `PinPolicy` object, and the restore fork refuses any other kind of callable.
- Data directory. The run configurations still name a directory that no longer
  exists. The override is `<Group9>/data/processed/aspirin`, bound to the launch
  record's dataset pin (manifest identity plus the SHA-256 of four files). The
  closed test partition is never hashed or opened here.
- Seven checkpoint identities: the accepted five (from the accepted recovery
  launch records) and the two lambda = 1 fits (from their score.json). Each is
  anchored independently of the records it comes from: the five by full SHA-256
  constants compared with the controlled-run and paired-trace records, the lambda = 1
  fits by the spec's recorded identities, and all but one by the completed-output
  inventories, keyed on the model id. The saved-against-saved JSON checks that the
  frozen restore makes before it loads a model (configuration, provenance, summary,
  best-checkpoint proof) run here too. Paths saved in the records point at a tree that
  has moved, so they are relocated, never trusted.
- The replay plan for launch A0: which models, 974 validation records, tolerance
  1e-10, and a data-only check that each saved prediction file lines up with the
  validation partition and is finite.
- A report that names the resolved root, the inputs' SHA-256, the current pin and the
  modules in use.

The restore itself (a model load) happens only in launch A0.
"""
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath

DEFAULT_GROUP9_ROOT = Path(__file__).resolve().parents[4]
ROOT_ENVIRONMENT_VARIABLE = "STAGE2_GROUP9_ROOT"

RECOVERY_RECORDS = "reports/execution/e3x-validation-recovery-2026-10-17/LAUNCH_RECORDS.json"
CONTROLLED_RUNS = "runs/aspirin-e3x-controlled-2026-10-16"
CONTROLLED_INVENTORY = "reports/execution/e3x-controlled-2026-10-16/COMPLETED_OUTPUT_INVENTORY.json"
ENERGY_INVENTORY = "reports/execution/e3x-score-repair-2026-10-16/ORIGINAL_RUN_INVENTORY.json"
DATA_DIRECTORY = "data/processed/aspirin"
DATASET_FILES = ("manifest.json", "train.npz", "validation.npz", "validation_minima_diagnostics.npz")
RUN_FILES = ("config.json", "provenance.json", "summary.json", "best-checkpoint-verification.json")

PRIMARY_IDS = ("energy-0", "direct-0", "consistency-0.1-0", "direct-1", "consistency-0.1-1")
LAMBDA1_IDS = ("consistency-1-0", "consistency-1-1")
ALL_IDS = PRIMARY_IDS + LAMBDA1_IDS
CONTROLLED_IDS = tuple(i for i in ALL_IDS if i != "energy-0")  # energy-0 is the original C run

# Full identities of the accepted five. The checkpoint hashes were compared with the
# controlled-run RESULT_ACCEPTANCE.json and the paired-trace LAUNCH_RECORDS.json (energy-0:
# the score-repair RESULT_ACCEPTANCE.json and the paired-trace records); the score hashes
# with the same controlled-run and paired-trace records (energy-0: paired-trace only).
EXPECTED_PRIMARY = {
    "energy-0": {"best_update": 184500,
                 "checkpoint_sha256": "905cb4fc4a95e89476680bf26f3a2c074407bddb1fdc49006c2d5f47590aa389",
                 "score_sha256": "94256ed5525468461e49b19ea36ec8a12e3527f4d9b09f185133c4cfa0fdca6b"},
    "direct-0": {"best_update": 174000,
                 "checkpoint_sha256": "e18821f7dc79b8059d0c12ff7358e587b15d62e0d74666d321b040645651f654",
                 "score_sha256": "d8dc1d90d5ea0588d5ff9972ff353c12c9db4e2caa69f02eac911f0b326ce589"},
    "consistency-0.1-0": {"best_update": 125500,
                          "checkpoint_sha256": "32202e1573d93f6ab874da671e679a0cfffd315549ef3476bcab1dc3df540fac",
                          "score_sha256": "de9605d89fb6e8d36bce0c7f9f1d25c02e40eb4ecc2b0a3cd3c7882d506e9076"},
    "direct-1": {"best_update": 189500,
                 "checkpoint_sha256": "65882da97b1c0ae5af3018b4ace7af539a98343355bcaeac68212a20fd618d85",
                 "score_sha256": "acb326be3ecd118f2016219e4e8bc7f90c0c1308491f9cc6fe78bcf0886e5cc1"},
    "consistency-0.1-1": {"best_update": 187000,
                          "checkpoint_sha256": "a89a5d879edd2d02a5e516eacb740fb1fc91bee05d9d0c8f4b6142ace7f4e1a1",
                          "score_sha256": "8f510db945c779eb749fea2a16361ba97bcba6cbd35fa8ddbaef8e79356f25ba"},
}

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
REPLAY_CONNECTIONS = 5

# Source-pin policy (spec section 5).
REQUIRED_EQUAL = ("code_fingerprint", "code_files", "lock_sha256", "dependencies", "helpers")
RECORDED_ONLY = ("python", "runtime_cpu", "templates")
PIN_KEYS = frozenset(REQUIRED_EQUAL + RECORDED_ONLY)
RUNTIME_RECORDED_FIELDS = ("platform",)  # inside runtime_cpu; the rest must be equal


class AdmissionError(ValueError):
    """Admission refused: a pin, path, identity or data check failed."""


class PinMismatch(AdmissionError):
    """The current source differs from the accepted pin where equality is required."""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_sha256(path):
    return digest(Path(path).read_bytes())


def read_json(path):
    return json.loads(Path(path).read_text())


# ---- Paths ---------------------------------------------------------------

def root_source(root=None):
    if root is not None:
        return "argument"
    return "environment" if os.environ.get(ROOT_ENVIRONMENT_VARIABLE) else "default"


def group9_root(root=None):
    root = Path(root or os.environ.get(ROOT_ENVIRONMENT_VARIABLE) or DEFAULT_GROUP9_ROOT)
    if not (root / "reports" / "execution").is_dir():
        raise AdmissionError(f"{root} is not the Group9 root (no reports/execution)")
    return root.resolve()


def relocate(path, root):
    """Map a path saved in the accepted records to the current root. The records were written
    when the project folder had another name but sat in the same parent folder, so everything
    up to and including that folder is replaced by `root` and the remainder is kept. The saved
    path must be absolute and lie under the root's parent folder. The result is the real path
    (symlinks and `..` resolved) and must lie under the root; any other path is refused."""
    root = Path(root).resolve()
    saved = PurePosixPath(str(path))
    depth = len(root.parts)
    if not saved.is_absolute() or len(saved.parts) <= depth or saved.parts[:depth - 1] != root.parts[:depth - 1]:
        raise AdmissionError(f"saved path is not under a known root: {str(path)!r}")
    resolved = Path(os.path.realpath(root.joinpath(*saved.parts[depth:])))
    if not resolved.is_relative_to(root):
        raise AdmissionError(f"saved path resolves outside the root: {str(path)!r}")
    return resolved


# ---- Pin policy ----------------------------------------------------------

def _dict_difference(actual, expected):
    changed = sorted(k for k in set(actual) & set(expected) if actual[k] != expected[k])
    added = sorted(set(actual) - set(expected))
    removed = sorted(set(expected) - set(actual))
    return {"changed": changed, "added": added, "removed": removed}


def compare_pins(actual, expected):
    """Apply the pin policy. Returns {"problems", "required_equal", "recorded_differences"}.

    Required keys must be equal. Python and the templates may differ, and so may the
    platform string inside runtime_cpu; each difference is listed with both values. Every
    other runtime_cpu field must be equal. A pin with any other or missing key fails closed."""
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
    for key in ("python", "templates"):
        if actual[key] != expected[key]:
            recorded[key] = {"current": actual[key], "accepted": expected[key]}
    cpu_now, cpu_then = actual["runtime_cpu"], expected["runtime_cpu"]
    if (not isinstance(cpu_now, dict) or not isinstance(cpu_then, dict)
            or set(cpu_now) != set(cpu_then) or any(f not in cpu_then for f in RUNTIME_RECORDED_FIELDS)):
        required["runtime_cpu_except_platform"] = False
        problems.append("runtime_cpu has an unexpected structure")
    else:
        other = sorted(k for k in cpu_then if k not in RUNTIME_RECORDED_FIELDS and cpu_now[k] != cpu_then[k])
        required["runtime_cpu_except_platform"] = not other
        if other:
            problems.append(f"runtime_cpu differs in {other}; only the platform string may differ")
        for field in RUNTIME_RECORDED_FIELDS:
            if cpu_now[field] != cpu_then[field]:
                recorded[f"runtime_cpu.{field}"] = {"current": cpu_now[field], "accepted": cpu_then[field]}
    return {"problems": problems, "required_equal": required, "recorded_differences": recorded}


class PinPolicy:
    """The only policy the restore fork accepts. Raises PinMismatch if a required key
    differs; keeps every report so the launch record can state the recorded differences."""

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
                      "saved_run_directory": model["run_directory"],
                      "saved_score_directory": model["score_directory"],
                      "launch_record": model["launch_record"],
                      "acceptance": dict(model["binding"]),
                      "source": RECOVERY_RECORDS})
    for cid in LAMBDA1_IDS:
        run = root / CONTROLLED_RUNS / cid
        score_directory = root / CONTROLLED_RUNS / f"{cid}-score"
        score = read_json(score_directory / "score.json")
        items.append({"id": cid, "role": "exploratory", "run_directory": run,
                      "score_directory": score_directory,
                      "saved_run_directory": str(run), "saved_score_directory": str(score_directory),
                      "launch_record": read_json(root / CONTROLLED_RUNS / f"{cid}-launch.json"),
                      "acceptance": {"checkpoint_sha256": score["best_checkpoint_sha256"],
                                     "best_update": score["best_update"],
                                     "score_sha256": file_sha256(score_directory / "score.json")},
                      "source": f"{CONTROLLED_RUNS}/{cid}-score/score.json"})
    return items


def load_inventories(root=None):
    """The two completed-output inventories: the controlled run (six models) and the
    original energy C run."""
    root = group9_root(root)
    return {"controlled": read_json(root / CONTROLLED_INVENTORY),
            "energy": read_json(root / ENERGY_INVENTORY)}


def _affixes_ok(value, prefix_suffix):
    prefix, suffix = prefix_suffix
    return value.startswith(prefix) and value.endswith(suffix)


def inventory_entries(item):
    """(inventory name, {file name in the inventory: path}) for one model. The six
    controlled models are keyed by model id; energy-0 is the original run's `C`."""
    run, score_directory = Path(item["run_directory"]), Path(item["score_directory"])
    if item["id"] == "energy-0":
        return "energy", {f"C/{name}": run / name for name in ("best.msgpack",) + RUN_FILES}
    cid = item["id"]
    entries = {f"{cid}/{name}": run / name for name in ("best.msgpack",) + RUN_FILES}
    entries[f"{cid}-score/score.json"] = score_directory / "score.json"
    entries[f"{cid}-score/best-validation.npz"] = score_directory / "best-validation.npz"
    entries[f"{cid}-launch.json"] = run.parent / f"{cid}-launch.json"
    return "controlled", entries


def identity_problems(item, accepted_pin, inventories, root=None, anchors=None):
    """Bytes-only checks of one model's identity (no model load): the pre-restore binding
    checks of the frozen e3x_checkpoint_arm (minus the current-runtime pin), the model's
    independent anchor, and the completed-output inventory, which must be present and
    list every file. launch A0 runs the frozen checks for real."""
    problems = []
    anchors = EXPECTED_PRIMARY if anchors is None else anchors
    run, score_directory = Path(item["run_directory"]), Path(item["score_directory"])
    launch, acceptance = item["launch_record"], item["acceptance"]
    if item["id"] not in ALL_IDS:
        return [f"unknown model id {item['id']!r}"]
    if root is not None:
        resolved = Path(root).resolve()
        for label, path in (("run", run), ("score", score_directory)):
            if not Path(os.path.realpath(path)).is_relative_to(resolved):
                problems.append(f"{label} directory resolves outside the root")
    checkpoint, score_path = run / "best.msgpack", score_directory / "score.json"
    predictions = score_directory / "best-validation.npz"
    for path in (checkpoint, score_path, predictions):
        if not path.is_file():
            return problems + [f"missing file {path}"]
    score = read_json(score_path)
    checkpoint_sha = file_sha256(checkpoint)
    score_sha = file_sha256(score_path)
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
    if file_sha256(predictions) != score["prediction_sha256"]:
        problems.append("saved predictions differ from the score's prediction hash")

    # The launch record belongs to the accepted source and dataset.
    if launch["source_pin"] != accepted_pin:
        problems.append("launch record source pin differs from the accepted pin")
    if not accepted_pin["code_fingerprint"].startswith(EXPECTED_CODE_FINGERPRINT_PREFIX):
        problems.append("accepted code fingerprint is not the one the spec records")
    if not launch["dataset_pin"]["manifest_identity"].startswith(EXPECTED_DATASET_IDENTITY_PREFIX):
        problems.append("launch dataset identity is not the one the spec records")

    # Independent anchors: the five accepted models by full hashes, lambda = 1 by the spec.
    if item["role"] == "primary":
        anchor = anchors.get(item["id"])
        if anchor is None:
            problems.append("no independent anchor for this model")
        else:
            if checkpoint_sha != anchor["checkpoint_sha256"]:
                problems.append("best.msgpack differs from the independent anchor")
            if score_sha != anchor["score_sha256"]:
                problems.append("score.json differs from the independent anchor")
            if acceptance["best_update"] != anchor["best_update"]:
                problems.append("best update differs from the independent anchor")
    else:
        expected = EXPECTED_LAMBDA1.get(item["id"])
        if expected is None:
            problems.append("no recorded identity for this lambda = 1 model")
        else:
            if acceptance["best_update"] != expected["best_update"]:
                problems.append("lambda = 1 best update differs from the spec's record")
            if not _affixes_ok(checkpoint_sha, expected["checkpoint"]):
                problems.append("lambda = 1 checkpoint hash differs from the spec's record")
            if not _affixes_ok(score_sha, expected["score"]):
                problems.append("lambda = 1 score hash differs from the spec's record")

    # The completed-output inventory: required, keyed on the model id, every file listed.
    which, entries = inventory_entries(item)
    inventory = (inventories or {}).get(which)
    if not inventory:
        problems.append(f"the {which} completed-output inventory is missing")
    else:
        for name, path in entries.items():
            if not Path(path).is_file():
                problems.append(f"{name} is missing on disk")
            elif inventory.get(name) is None:
                problems.append(f"{name} is not listed in the {which} inventory")
            else:
                listed = inventory[name]["sha256"] if isinstance(inventory[name], dict) else inventory[name]
                if listed != file_sha256(path):
                    problems.append(f"{name} differs from the {which} inventory")
    return problems


def saved_json_problems(item):
    """The JSON-only saved-against-saved checks of the frozen restore (before it loads a
    model): run configuration against the launch record, provenance against the summary,
    a complete accepted fit, the computation hash, the provenance code keys against the
    launch pin, and the best-checkpoint proof."""
    from dataclasses import asdict
    from force_consistency.data.splits import canonical_hash
    from force_consistency.train import RunConfig
    problems = []
    run, launch, acceptance = Path(item["run_directory"]), item["launch_record"], item["acceptance"]
    try:
        cfg = RunConfig(**read_json(run / "config.json"))
        prov = read_json(run / "provenance.json")
        summary = read_json(run / "summary.json")
        proof = read_json(run / "best-checkpoint-verification.json")
        score = read_json(Path(item["score_directory"]) / "score.json")
    except (OSError, ValueError, TypeError, KeyError) as error:
        return [f"run files unreadable: {type(error).__name__}: {error}"]
    try:
        if asdict(cfg) != launch["config"]:
            problems.append("run configuration differs from the launch record")
        if prov != summary["provenance"]:
            problems.append("run provenance differs from the summary's")
        if (summary["status"] != "passed" or summary["updates"] != cfg.max_updates
                or summary["stop_reason"] != "max_updates" or score["updates"] != cfg.max_updates
                or summary["best_update"] != score["best_update"]):
            problems.append("the fit is not a complete accepted fit")
        computation = canonical_hash({k: v for k, v in asdict(cfg).items()
                                      if k not in ("data_directory", "run_directory", "resume_checkpoint",
                                                   "max_updates", "wall_seconds")})
        if (prov["config_computation_hash"] != computation or prov["arm"] != cfg.arm
                or prov["loss_weights"] != cfg.loss_weights
                or prov["consistency_strength"] != cfg.consistency_strength):
            problems.append("objective or configuration differs from the provenance")
        for key in ("code_fingerprint", "code_files", "lock_sha256", "python", "dependencies"):
            if prov[key] != launch["source_pin"][key]:
                problems.append(f"provenance {key} differs from the launch pin")
        if (proof["status"] != "passed" or proof["role"] != "best"
                or proof["update"] != acceptance["best_update"]):
            problems.append("the best-checkpoint proof is missing or for another update")
    except (KeyError, TypeError) as error:
        problems.append(f"saved record is malformed: {type(error).__name__}: {error}")
    return problems


def identity_record(item):
    """The identity of one model for the launch record (hashes recomputed from files)."""
    run, score_directory = Path(item["run_directory"]), Path(item["score_directory"])
    score = read_json(score_directory / "score.json")
    record = {"id": item["id"], "role": item["role"], "best_update": score["best_update"],
              "run_directory": str(run), "score_directory": str(score_directory),
              "saved_run_directory": item.get("saved_run_directory"),
              "saved_score_directory": item.get("saved_score_directory"),
              "best_msgpack_sha256": file_sha256(run / "best.msgpack"),
              "score_json_sha256": file_sha256(score_directory / "score.json"),
              "predictions_sha256": file_sha256(score_directory / "best-validation.npz"),
              "run_files_sha256": {name: file_sha256(run / name) for name in RUN_FILES},
              "equal_connection_force_mae_eV_per_A": score.get("equal_connection_force_mae_eV_per_A"),
              "source": item["source"]}
    _, entries = inventory_entries(item)
    launch_names = [n for n in entries if n.endswith("-launch.json")]
    record["launch_file_sha256"] = file_sha256(entries[launch_names[0]]) if launch_names else None
    return record


# ---- Replay plan (launch A0) --------------------------------------------

def replay_plan(items, directory, split_hash=None):
    """Data-only check that each model's saved predictions line up with the validation
    partition and are finite float64, and the fixed replay settings for A0. The replay
    itself (a model call through arm['forces'], in batches of REPLAY_BATCH plus one
    single-molecule call per connection) happens in A0."""
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
            if z["forces"].dtype != np.float64 or not np.isfinite(z["forces"]).all():
                raise AdmissionError(f"{item['id']}: saved forces are not finite float64")
            if z["energy"].shape != (REPLAY_RECORDS,) or not np.isfinite(z["energy"]).all():
                raise AdmissionError(f"{item['id']}: saved energies are not finite, one per record")
    return {"models": [item["id"] for item in items], "records": REPLAY_RECORDS,
            "batch": REPLAY_BATCH, "single_molecule_calls_per_connection": 1,
            "connections": REPLAY_CONNECTIONS, "rtol": REPLAY_TOLERANCE, "atol": REPLAY_TOLERANCE,
            "through": "arm['forces']", "data_directory": str(directory)}


# ---- Report ---------------------------------------------------------------

MODULES_IN_USE = ("stage2_capped.admission", "stage2_capped.restore_fork", "stage2_capped.diff_gate",
                  "stage2_capped.relax_capped", "evaluation.adapter", "e3x_phase_common")


def module_report(names=MODULES_IN_USE):
    """File and SHA-256 of each module the admission and the restore depend on, which no
    source pin covers. Found without importing them."""
    report = {}
    for name in names:
        spec = importlib.util.find_spec(name)
        origin = getattr(spec, "origin", None)
        report[name] = {"file": origin, "sha256": file_sha256(origin) if origin else None}
    return report


# ---- Everything launch A0 needs before it loads a model ------------------

def admit_all(root=None):
    """Run every no-model admission check on the real source and the real files. Raises on
    the first failure; returns the report that goes into launch record A."""
    return _admit_all(root, PinPolicy(), None)


def _admit_all(root, policy, actual_pin):
    """admit_all with a substitutable policy and current pin. Tests only: a launch calls
    admit_all, which always reads the real source pin."""
    source = root_source(root)
    root = group9_root(root)
    if actual_pin is None:
        from e3x_phase_common import source_pin
        actual_pin = source_pin()
    admit_source(policy, root, actual_pin)
    accepted = accepted_source_pin(root)
    inventories = load_inventories(root)
    items = load_bindings(root)
    directory = data_directory(root)
    identities, failures = {}, {}
    for item in items:
        problems = identity_problems(item, accepted, inventories, root) + saved_json_problems(item)
        if problems:
            failures[item["id"]] = problems
        else:
            identities[item["id"]] = identity_record(item)
            verify_dataset(directory, item["launch_record"])
    if failures:
        raise AdmissionError("identity checks failed: " + json.dumps(failures))
    return {"root": str(root), "root_source": source,
            "launch_records_sha256": file_sha256(root / RECOVERY_RECORDS),
            "inventory_sha256": {"controlled": file_sha256(root / CONTROLLED_INVENTORY),
                                 "energy": file_sha256(root / ENERGY_INVENTORY)},
            "pin_policy": policy.reports[-1], "current_pin": actual_pin, "accepted_pin": accepted,
            "modules": module_report(), "data_directory": str(directory),
            "dataset_pin": items[0]["launch_record"]["dataset_pin"], "identities": identities,
            "replay_plan": replay_plan(items, directory)}
