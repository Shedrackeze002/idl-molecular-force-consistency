"""State of a launch, derived from evidence (plan v3 sections 3, 4 and 6; amendments A1, A2, A3, A7).

There is no mutable ledger file. Each command rebuilds the `StageLedger` from artifacts that are written
once and never overwritten: the A0 result, each stage's marker, the stage directories. Around them:

- `EventLog`: numbered, hash-chained event files in the output root. A record for the reviewers.
- `Registry`: event files outside every output root, named `<record sha>-<unit>-<attempt>-<event>.json`.
  A *unit* is A0 or a stage; an *attempt* is `<unit>-<n>`. A `started` event without an `exited` event
  counts as spent whatever the root holds; a later command may only confirm that (A2).
- `Lock`: an exclusive `flock` on a lock file in the output root.
- The completion decision: the parent calls a stage COMPLETE only when every condition in
  `completion_problems` holds, never from the exit code alone.
- The A0 verdict rule and the post-A0 resource guard. The batch-1 latency that A0 measures is used by
  the guard only; every other evidence value triggers no action.
"""
import fcntl
import json
import math
import os
import re
import signal
import time
from pathlib import Path

from stage2_capped import admission, driver, launch_checks, launch_io
from stage2_capped.launch_record import LaunchRefused, UNITS, launch_folder_of, sha256_bytes  # noqa: F401 - re-exported
from stage2_capped.stage_gate import StageLedger

LOG_DIRECTORY = "LAUNCH_LOG"
LOCK_NAME = ".launch.lock"
SETUP_SECONDS_PER_MODEL = 6.0
GUARD_LATENCY_FACTOR = 2.0
_REGISTRY_NAME = re.compile(r"^(?P<sha>[0-9a-f]{64})-(?P<unit>A0|0b|0|1|2)-(?P<attempt>\d+)-(?P<event>started|exited)\.json$")


def dumps(payload):
    return (json.dumps(payload, indent=1, allow_nan=False, sort_keys=True) + "\n").encode("utf-8")


# ---- Event log ---------------------------------------------------------------

class EventLog:
    """`LAUNCH_LOG/NNNN-<event>.json` in the output root, each holding the SHA-256 of the file before it."""

    def __init__(self, root):
        self.directory = Path(root) / LOG_DIRECTORY

    def files(self):
        return sorted(self.directory.glob("[0-9][0-9][0-9][0-9]-*.json")) if self.directory.is_dir() else []

    def head_sha256(self):
        files = self.files()
        return sha256_bytes(files[-1].read_bytes()) if files else None

    def append(self, event, **fields):
        number = len(self.files()) + 1
        payload = {"n": number, "event": event, "previous_sha256": self.head_sha256(), "time": time.time(), **fields}
        data = dumps(payload)
        path = launch_io.atomic_write_bytes(self.directory / f"{number:04d}-{event}.json", data)
        return path, sha256_bytes(data)

    def problems(self):
        previous, problems = None, []
        for number, path in enumerate(self.files(), start=1):
            payload = json.loads(path.read_text())
            if payload.get("n") != number or payload.get("previous_sha256") != previous:
                problems.append(f"{path.name}: the chain is broken")
            previous = sha256_bytes(path.read_bytes())
        return problems


# ---- Registry ------------------------------------------------------------------

class Registry:
    """Launch events outside every output root. Entries are written once; none is ever removed here."""

    def __init__(self, directory):
        self.directory = Path(directory)

    def entries(self):
        found = []
        if self.directory.is_dir():
            for path in sorted(self.directory.iterdir()):
                match = _REGISTRY_NAME.match(path.name)
                if match:
                    data = path.read_bytes()
                    found.append({"file": path.name, "sha256": sha256_bytes(data), "record_sha256": match["sha"],
                                  "unit": match["unit"], "attempt": int(match["attempt"]), "event": match["event"],
                                  "payload": json.loads(data)})
        return found

    def attempts(self, record_sha256, unit):
        attempts = {}
        for entry in self.entries():
            if entry["record_sha256"] == record_sha256 and entry["unit"] == unit:
                attempts.setdefault(entry["attempt"], {"started": None, "exited": None})[entry["event"]] = entry
        return attempts

    def next_attempt(self, record_sha256, unit):
        return 1 + max(self.attempts(record_sha256, unit), default=0)

    def write(self, record_sha256, unit, attempt, event, **fields):
        if unit not in UNITS or event not in ("started", "exited"):
            raise ValueError("unknown unit or event")
        payload = {"record_sha256": record_sha256, "unit": unit, "attempt": attempt, "event": event,
                   "time": time.time(), **fields}
        data = dumps(payload)
        name = self.event_name(record_sha256, unit, attempt, event)
        launch_io.atomic_write_bytes(self.directory / name, data)
        return {"file": name, "sha256": sha256_bytes(data)}

    @staticmethod
    def event_name(record_sha256, unit, attempt, event):
        return f"{record_sha256}-{unit}-{attempt}-{event}.json"

    def holds(self, record_sha256, unit, attempt, event):
        """Whether this event's file is in the registry. A write that linked its file and then raised still counts: the
        event is there, and a `started` event with no `exited` event is spent."""
        return (self.directory / self.event_name(record_sha256, unit, attempt, event)).exists()

    def spent_attempts(self, record_sha256, unit):
        """Attempts that passed the spent point: an `exited` event with spent true, or a `started`
        event with no `exited` event (it counts as spent whatever the root holds)."""
        spent = []
        for number, pair in sorted(self.attempts(record_sha256, unit).items()):
            if pair["started"] is not None and pair["exited"] is None:
                spent.append(number)
            elif pair["exited"] is not None and pair["exited"]["payload"].get("spent") is True:
                spent.append(number)
        return spent

    def is_spent(self, record_sha256, unit):
        return bool(self.spent_attempts(record_sha256, unit))

    def confirm_spent(self, record_sha256, unit, evidence):
        """A2: for every `started` event with no `exited` event, write an `exited` event that says spent
        and carries the evidence found. Never writes `refused_before_start` for an orphaned start."""
        written = []
        for number, pair in sorted(self.attempts(record_sha256, unit).items()):
            if pair["started"] is not None and pair["exited"] is None:
                written.append(self.write(record_sha256, unit, number, "exited", outcome="confirmed_spent_after_missing_exit",
                                          spent=True, evidence=evidence))
        return written

    def violation_events(self, pair_sha256s):
        """Exited events whose outcome is `audit_violation` under either record of the pair (B2). The fixed exit code
        voids the pair even when the violation file itself could not be written."""
        return sorted(e["file"] for e in self.entries() if e["event"] == "exited" and e["record_sha256"] in pair_sha256s
                      and e["payload"].get("outcome") == "audit_violation")

    def prior_problems(self, pair_sha256s, prior_dispatches):
        """Every event under a record SHA-256 outside this pair must be listed in `prior_dispatches`, and
        every listed entry must exist; events under the pair's own SHA-256s are the current attempts."""
        actual = {(e["file"], e["sha256"]) for e in self.entries() if e["record_sha256"] not in pair_sha256s}
        listed = {(name, digest) for name, digest in prior_dispatches}
        problems = [f"registry event {name} is not listed in prior_dispatches" for name, _ in sorted(actual - listed)]
        problems += [f"prior_dispatches lists {name}, which the registry does not hold" for name, _ in sorted(listed - actual)]
        return problems

    def unit_problems(self, record_sha256, unit):
        """Why the registry does not show `unit` as completed under this record (empty list: it does). At least one
        attempt must have passed the spent point, and every such attempt must have ended with outcome `complete`:
        a unit the parent called failed, or one whose exit it never saw (R1), is failed whatever its files say."""
        spent = self.spent_attempts(record_sha256, unit)
        if not spent:
            return [f"the registry shows no dispatch of unit {unit} under this record"]
        problems = []
        for number in spent:
            exited = self.attempts(record_sha256, unit)[number]["exited"]
            outcome = None if exited is None else exited["payload"].get("outcome")
            if outcome != "complete":
                problems.append(f"the registry records attempt {number} of unit {unit} as "
                                f"{outcome or 'started, with no exit seen'}, not complete")
        return problems

    def a0_problems(self, record_a_sha256):
        """Exactly one attempt of the A0 unit passed the spent point, and it ended with a result."""
        spent = self.spent_attempts(record_a_sha256, "A0")
        if len(spent) != 1:
            return [f"the registry shows {len(spent)} spent A0 attempts for record A; exactly one is required"]
        exited = self.attempts(record_a_sha256, "A0")[spent[0]]["exited"]
        if exited is None or exited["payload"].get("outcome") != "complete":
            return ["the one spent A0 attempt did not end complete"]
        return []


# ---- Lock ----------------------------------------------------------------------

class LockHeld(RuntimeError):
    """Another launch command holds the output root's lock."""


class Lock:
    """Exclusive flock on a file in the output root, held for the whole command. The lock file is the one
    file not written by `atomic_write_bytes`: it carries no evidence. Python file descriptors are not
    inherited by the worker, so a dead parent releases it."""

    def __init__(self, root):
        self.path = Path(root) / LOCK_NAME
        self.descriptor = None

    def __enter__(self):
        self.descriptor = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            fcntl.flock(self.descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            os.close(self.descriptor)
            self.descriptor = None
            raise LockHeld(f"another launch command holds {self.path}") from None
        return self

    def __exit__(self, *exc):
        if self.descriptor is not None:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
            os.close(self.descriptor)
            self.descriptor = None


# ---- The root's contents (A1) -----------------------------------------------------

_ATTEMPT_FILES = (re.compile(r"^A0_REFUSED-\d+\.json$"), re.compile(r"^worker-(0b|0|1|2)\.attempt-\d+\.result\.json$"),
                  re.compile(r"^WORKER_PID-(A0|0b|0|1|2)-\d+\.json$"),
                  re.compile(r"^logs/[^/]+$"), re.compile(r"^LAUNCH_LOG/\d{4}-[^/]+\.json$"),
                  re.compile(r"^LAUNCH\.json$"), re.compile(r"^" + re.escape(LOCK_NAME) + r"$"))
_FIXED_STAGE_FILES = {stage: (f"worker-{stage}.result.json", "STAGE0_RERUN_ATTEMPTED.json", "STAGE0_RERUN.json")
                      for stage in driver.STAGE_SPECS}


def root_contents_problems(root, unit):
    """A1 and B1: root reuse is per unit. In the first command of a root (`a0`, `stage 0`) the root may hold
    only LAUNCH.json, the launch log, the lock file, worker pid files, worker logs and attempt files. In a
    later `stage` command it may also hold the trees and fixed-name files of the *other* stages (their markers
    are validated against the typed hashes by `derive_ledger`), but the unit's own artifacts must be absent: no
    `rows/stage<unit>/` and none of the unit's fixed-name files."""
    root = Path(root)
    problems = []
    first_command = unit in ("A0", "0")
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        relative = path.relative_to(root).as_posix()
        if any(pattern.match(relative) for pattern in _ATTEMPT_FILES):
            continue
        if not first_command:
            parts = relative.split("/")
            if parts[0] == "rows" and len(parts) > 1 and parts[1].startswith("stage") and parts[1][5:] in driver.STAGE_SPECS \
                    and parts[1][5:] != unit:
                continue
            if len(parts) == 1 and any(parts[0] in names for other, names in _FIXED_STAGE_FILES.items() if other != unit):
                continue
        problems.append(f"the root holds {relative}, which a start of unit {unit} under the same record may not find")
    return problems


def spent_signal(root, unit):
    """The spent point, as the parent and the worker both test it (B4): A0 is spent once
    `A0_RESTORE_BEGUN.json` or `A0_RESULT.json` exists; a stage once its stage directory exists."""
    root = Path(root)
    if unit == "A0":
        return (root / "A0_RESTORE_BEGUN.json").exists() or (root / "A0_RESULT.json").exists()
    return stage_directory(root, unit).exists()


# ---- Stage markers and the completion decision ------------------------------------------

def stage_directory(root, stage):
    return Path(root) / "rows" / f"stage{stage}"


def marker_problems(root, stage, record, launch_id, relaxer_sha256):
    """(problems, marker SHA-256): the content checks of a stage's marker and manifest (plan section 6)."""
    directory = stage_directory(root, stage)
    spec = driver.STAGE_SPECS[stage]
    problems = []
    if (directory / "INCOMPLETE.json").exists():
        problems.append("INCOMPLETE.json exists")
    marker_path, manifest_path = directory / "STAGE_COMPLETE.json", directory / "STAGE_MANIFEST.json"
    if not marker_path.is_file() or not manifest_path.is_file():
        return problems + ["STAGE_COMPLETE.json or STAGE_MANIFEST.json is missing"], None
    marker_bytes, manifest_bytes = marker_path.read_bytes(), manifest_path.read_bytes()
    try:
        marker, manifest = json.loads(marker_bytes), json.loads(manifest_bytes)
    except ValueError:
        return problems + ["a marker file is not JSON"], sha256_bytes(marker_bytes)
    if marker.get("complete") is not True:
        problems.append("the marker does not say complete")
    if marker.get("stage") != stage:
        problems.append("the marker names a different stage")
    if marker.get("runs") != spec["runs"]:
        problems.append(f"the marker records {marker.get('runs')} runs, the stage has {spec['runs']}")
    if marker.get("custom_tasks") is not False:
        problems.append("the marker is from a custom task list")
    forces = marker.get("force_evaluations")
    if not isinstance(forces, int) or forces > spec["attempt_ceiling"]:
        problems.append("force_evaluations is missing or above the attempt ceiling")
    if marker.get("manifest_sha256") != sha256_bytes(manifest_bytes):
        problems.append("manifest_sha256 does not match STAGE_MANIFEST.json")
    if manifest.get("launch_id") != launch_id or manifest.get("stage") != stage:
        problems.append("the manifest names a different launch or stage")
    if manifest.get("relaxer_sha256") != relaxer_sha256:
        problems.append("the manifest names a different relaxer")
    if stage == "0":
        stage0 = manifest.get("stage0") or {}
        if not (stage0.get("verdict") or {}).get("proceed") is True or stage0.get("compared") != spec["runs"]:
            problems.append("the Stage 0 verdict does not proceed over all compared runs")
        if stage0.get("saved_file_hashes") != record["saved_file_hashes"]:
            problems.append("the Stage 0 manifest's saved-file hashes differ from the record's")
    rows = len(list(directory.glob("*/*/*.json")))
    if rows != spec["runs"]:
        problems.append(f"{rows} row files on disk, the stage has {spec['runs']}")
    return problems, sha256_bytes(marker_bytes)


def completion_problems(exit_record, root, stage, record, launch_id, relaxer_sha256, audit_report, required_relative_paths,
                        allowed_relative_paths):
    """The parent's completion decision. COMPLETE only if the list is empty: exit code 0, not timed out, the
    marker and manifest validate, no INCOMPLETE marker, and the audit log shows zero violations and exactly the
    files a stage must open (all of them, and no others). A stage that finishes after its wall ceiling (timed
    out) is FAILED."""
    problems = []
    if exit_record.get("returncode") != 0:
        problems.append(f"the worker exited with code {exit_record.get('returncode')}")
    if exit_record.get("timed_out"):
        problems.append("the worker was stopped at the wall ceiling")
    found, _ = marker_problems(root, stage, record, launch_id, relaxer_sha256)
    problems += found
    problems += launch_checks.audit_problems(audit_report, required_relative_paths, allowed_relative_paths)
    return problems


def derive_ledger(root, record, typed, *, a0_valid, relaxer_sha256, registry, record_sha256):
    """The StageLedger rebuilt from evidence. `typed` maps a stage to the marker hash typed on the command
    line. A stage directory with no valid marker, or with INCOMPLETE.json, is failed; a valid marker whose
    hash was not typed in this command is untyped (A7): failed for the ledger, so it blocks a start of that
    stage and satisfies no prerequisite. A typed hash that does not match a valid marker is refused. A stage
    whose marker is valid and typed is complete only if the registry also records it as complete under this
    record (N03): the parent's FAILED decision is written there, and a stage can finish its marker and still be
    called failed (exit code, exit-time module check, wall ceiling)."""
    state, notes = {"A0": "complete" if a0_valid else "pending"}, {}
    for stage in driver.STAGE_SPECS:
        directory = stage_directory(root, stage)
        if not directory.exists():
            if stage in typed:
                raise LaunchRefused(f"a hash was typed for stage {stage}, which has no directory in the root")
            state[stage] = "pending"
            continue
        problems, marker_sha = marker_problems(root, stage, record, record["launch_id"], relaxer_sha256)
        if problems:
            state[stage], notes[stage] = "failed", problems
        elif stage not in typed:
            state[stage], notes[stage] = "failed", ["complete-untyped: no marker hash was typed for this stage"]
        elif typed[stage] != marker_sha:
            raise LaunchRefused(f"the typed hash for stage {stage} differs from its STAGE_COMPLETE.json")
        elif registry.unit_problems(record_sha256, stage):
            state[stage], notes[stage] = "failed", registry.unit_problems(record_sha256, stage)
        else:
            state[stage] = "complete"
    snapshot = {"state": state, "a0_replay_passed": bool(a0_valid), "events": []}
    return StageLedger.from_snapshot(snapshot), notes


# ---- A0 evidence ----------------------------------------------------------------------

def _differs(value, accepted):
    """True unless both are present and equal: a field missing on both sides is not a match (N10)."""
    return value is None or accepted is None or value != accepted


def identity_problems(model_id, identity, accepted, admission_report):
    """B5: the arm's checkpoint identity against the admission identity, field by field, not dict equality.
    checkpoint_sha256 is the best.msgpack hash, score_sha256 the score.json hash; arm and seed equal the
    study's table; lambda is compared for the consistency arms only (the recorded launch configs fix
    consistency_strength for those; a direct or energy arm's stored value is not pinned); the dataset identity
    and the source fingerprint equal the accepted records'."""
    problems = []
    if _differs(identity.get("checkpoint_sha256"), accepted.get("best_msgpack_sha256")):
        problems.append("checkpoint_sha256 differs from the admitted best.msgpack hash")
    if _differs(identity.get("score_sha256"), accepted.get("score_json_sha256")):
        problems.append("score_sha256 differs from the admitted score.json hash")
    if _differs(identity.get("best_update"), accepted.get("best_update")):
        problems.append("best_update differs from the admitted one")
    arm, seed, lam = driver.MODEL_IDENTITY[model_id]
    if identity.get("arm") != arm or identity.get("seed") != seed:
        problems.append("arm or seed differs from the study's model table")
    if arm == "consistency" and identity.get("lambda") != lam:
        problems.append("lambda differs from the study's model table")
    if _differs(identity.get("dataset_identity"), (admission_report.get("dataset_pin") or {}).get("manifest_identity")):
        problems.append("dataset_identity differs from the launch record's dataset pin")
    if _differs(identity.get("source_fingerprint"), (admission_report.get("accepted_pin") or {}).get("code_fingerprint")):
        problems.append("source_fingerprint differs from the accepted code fingerprint")
    return problems


def a0_verdict(result):
    """(passed, reasons). PASS only if all seven entries carry the frozen replay return with status verified,
    974 records, tolerances 1e-10, the five single-record ids equal to the partition's first five records in the
    frozen group order, the arm's identity equal to the admission identity, and a finite maximum difference.
    Two consistency checks (A6, not gates): the frozen return's maximum must not exceed 1e-10 + 1e-10 x the
    largest saved force, and the evidence's batch-4 scaled residual must be at most 1 whenever the frozen return
    says verified; a contradiction fails that entry with the reason."""
    reasons = []
    models, identities = result.get("models", {}), (result.get("admission") or {}).get("identities", {})
    scope = result.get("replay_scope") or {}
    for model_id in admission.ALL_IDS:
        entry = models.get(model_id)
        if not isinstance(entry, dict) or not isinstance(entry.get("replay"), dict):
            reasons.append(f"{model_id}: no replay return ({(entry or {}).get('failure')})")
            continue
        replay, tolerance = entry["replay"], admission.REPLAY_TOLERANCE
        if replay.get("status") != "verified":
            reasons.append(f"{model_id}: status {replay.get('status')!r}")
        if replay.get("record_count") != admission.REPLAY_RECORDS:
            reasons.append(f"{model_id}: record_count {replay.get('record_count')!r}")
        if replay.get("rtol") != tolerance or replay.get("atol") != tolerance:
            reasons.append(f"{model_id}: tolerances differ from {tolerance}")
        if not scope.get("first_five_record_ids") or replay.get("single_record_ids") != scope.get("first_five_record_ids"):
            reasons.append(f"{model_id}: single_record_ids are not the partition's first five records")
        maximum = replay.get("maximum_absolute_force_difference")
        if not isinstance(maximum, (int, float)) or isinstance(maximum, bool) or not math.isfinite(maximum):
            reasons.append(f"{model_id}: the maximum difference is not finite")
        reasons += [f"{model_id}: {problem}" for problem in identity_problems(
            model_id, replay.get("checkpoint_identity") or {}, identities.get(model_id) or {}, result.get("admission") or {})]
        saved_max = entry.get("saved_force_abs_max")
        if isinstance(maximum, (int, float)) and isinstance(saved_max, (int, float)) and math.isfinite(maximum) \
                and replay.get("status") == "verified" and maximum > tolerance + tolerance * saved_max:
            reasons.append(f"{model_id}: the frozen return's maximum exceeds atol + rtol x the largest saved force")
        residual = ((entry.get("evidence") or {}).get("batch4") or {}).get("max_scaled_residual")
        if replay.get("status") == "verified" and isinstance(residual, (int, float)) and not residual <= 1.0:
            reasons.append(f"{model_id}: the evidence's batch-4 scaled residual {residual} contradicts a verified replay")
    if set(scope.get("groups", [])) != {"13", "18", "18_2", "66", "78"} or len(scope.get("groups", [])) != 5:
        reasons.append("the replay scope does not list the five frozen connection groups")
    return not reasons, reasons


def a0_evidence(record_b, typed_sha256, registry, *, current_runtime, group9=None):
    """The validated A0 result record B points at. Returns (result, problems)."""
    path = Path(record_b["a0_output_root"]) / "A0_RESULT.json"
    if not path.is_file():
        return None, [f"{path} does not exist"]
    data = path.read_bytes()
    problems = []
    if sha256_bytes(data) != typed_sha256:
        problems.append("A0_RESULT.json has a different SHA-256 from the typed value")
    result = json.loads(data)
    for field, wanted in (("record_sha256", record_b["record_a_sha256"]), ("launch_id", record_b["a0_launch_id"]),
                          ("output_root", record_b["a0_output_root"]), ("commit", record_b["commit"])):
        if result.get(field) != wanted:
            problems.append(f"A0_RESULT.json {field} is not the one record B names")
    if (result.get("admission") or {}).get("launch_records_sha256") != record_b["expected_launch_records_sha256"]:
        problems.append("A0_RESULT.json names different accepted launch records")
    passed, reasons = a0_verdict(result)
    if not passed:
        problems += [f"A0 verdict: {reason}" for reason in reasons]
    problems += registry.a0_problems(record_b["record_a_sha256"])
    problems += launch_checks.runtime_problems(result.get("runtime", {}), current_runtime)
    return result, problems


def _latency(a0_result, model_id):
    return (((a0_result.get("models") or {}).get(model_id) or {}).get("evidence") or {}).get("batch1_latency") or {}


def _usable(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def guard_problems(a0_result, stage, ceiling_seconds):
    """The post-A0 resource guard (A3, R2): refuse before anything is spent if worst-case attempts x the largest
    per-model MEAN batch-1 latency over the models this stage uses x 2 + setup exceeds the record's ceiling. A
    missing, non-finite or non-positive mean for any model the stage uses is a refusal. It can stop a launch and
    can never raise a ceiling. The p99 is a flag only (`guard_flags`): over 974 calls it is the tenth-slowest
    call, set by system jitter, and a false refusal would cost a new record B."""
    spec = driver.STAGE_SPECS[stage]
    means = []
    for model_id in spec["models"]:
        value = _latency(a0_result, model_id).get("mean_seconds")
        if not _usable(value):
            return [f"no usable batch-1 mean latency for {model_id} in the A0 result"]
        means.append(value)
    needed = spec["runs"] * driver.MAX_FORCE_EVALUATIONS * max(means) * GUARD_LATENCY_FACTOR \
        + SETUP_SECONDS_PER_MODEL * len(spec["models"])
    if needed > ceiling_seconds:
        return [f"stage {stage}: worst case {needed:.0f} s at the A0 mean latency exceeds the {ceiling_seconds} s ceiling"]
    return []


def guard_flags(a0_result, stage, ceiling_seconds):
    """Flags printed with the guard's verdict, never a refusal: the same bound computed with the largest p99."""
    spec = driver.STAGE_SPECS[stage]
    p99s = [_latency(a0_result, m).get("p99_seconds") for m in spec["models"]]
    if not all(_usable(v) for v in p99s):
        return ["the A0 result has no usable batch-1 p99 for every model of this stage"]
    needed = spec["runs"] * driver.MAX_FORCE_EVALUATIONS * max(p99s) * GUARD_LATENCY_FACTOR \
        + SETUP_SECONDS_PER_MODEL * len(spec["models"])
    return [f"flag: with the largest p99 the worst case would be {needed:.0f} s against {ceiling_seconds} s"] \
        if needed > ceiling_seconds else []


def guard_summary(a0_result, stage, ceiling_seconds):
    """The guard's answer for one stage as one printable line: the stage, the models it uses, the largest per-model
    mean batch-1 latency the guard uses (and whose it is), the ceiling, and OK or REFUSED with the reason. The verdict
    is `guard_problems`' own; nothing is computed twice."""
    spec = driver.STAGE_SPECS[stage]
    means = {model_id: _latency(a0_result, model_id).get("mean_seconds") for model_id in spec["models"]}
    unusable = [model_id for model_id, value in means.items() if not _usable(value)]
    if unusable:
        used = "none (no usable mean for " + ", ".join(unusable) + ")"
    else:
        slowest = max(means, key=means.get)
        used = f"{means[slowest]:.6g} s ({slowest})"
    problems = guard_problems(a0_result, stage, ceiling_seconds)
    verdict = "OK" if not problems else "REFUSED: " + "; ".join(problems)
    return (f"post-A0 guard, stage {stage}: models {', '.join(spec['models'])}; mean batch-1 latency used {used}; "
            f"ceiling {ceiling_seconds:g} s; {verdict}")


# ---- The supervisor's stop signal ---------------------------------------------------------

def install_sigterm():
    """A worker turns the supervisor's SIGTERM (sent at the wall ceiling) into `DeadlineReached`, so the stop is
    labelled and the result is written before the supervisor's SIGKILL, which follows its finalization window."""
    def terminate(signum, frame):
        raise driver.DeadlineReached("terminated by the supervisor")
    signal.signal(signal.SIGTERM, terminate)


def ignore_sigterm():
    """Once the work is done or stopped, the result must be written: a late SIGTERM would raise `DeadlineReached` in
    the middle of the write and lose it. The supervisor still ends the process with SIGKILL after its window. Only the
    main thread can change a handler."""
    try:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    except ValueError:
        pass


# ---- Violations and orphaned workers (B2, B4) ---------------------------------------------

def violation_files(launch_folder, pair_sha256s):
    """Audit violation files for either record of the pair. They live in `<launch folder>/violations/`, outside
    every output root, named `AUDIT_VIOLATION-<record sha>-<unit>-<attempt>.json`. Any one voids the unit and
    makes the record pair unusable, in any phase: it is not a correctable refusal."""
    directory = Path(launch_folder) / "violations"
    if not directory.is_dir():
        return []
    return sorted(p.name for p in directory.glob("AUDIT_VIOLATION-*.json")
                  if any(sha in p.name for sha in pair_sha256s))


def process_start(pid):
    """The start time of a process as `ps` prints it (None if there is no such process). A bare pid can be
    reused, so a worker's pid file records the start time too."""
    import subprocess
    result = subprocess.run(["ps", "-o", "lstart=", "-p", str(int(pid))], capture_output=True, check=False)
    text = result.stdout.decode(errors="replace").strip()
    return text or None


def worker_alive(pid_record):
    """True if the process the worker's pid file names is still running (same pid and start time), or if
    any process of its group lives (`killpg(pid, 0)`; under `caffeinate -i` the worker's pid is the group's)."""
    pid = pid_record["pid"]
    if process_start(pid) == pid_record["start_time"]:
        return True
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:                  # the group exists and cannot be signalled: it is alive
        return True
    return True


def write_worker_pid(root, unit, attempt):
    """The worker's first act: its pid and start time, attempt-numbered, written once."""
    pid = os.getpid()
    payload = {"unit": unit, "attempt": attempt, "pid": pid, "ppid": os.getppid(), "start_time": process_start(pid)}
    return launch_io.atomic_write_bytes(Path(root) / f"WORKER_PID-{unit}-{attempt}.json", dumps(payload))


# ---- What a worker shares with the parent (B4) ----------------------------------------------------------------

def started_event(root, unit, attempt):
    """The parent's latest `started` event for exactly this unit and attempt, or None. A worker started by hand has none.
    The latest, because a parent whose registry write failed after its log write left an older event with the same
    attempt number (nothing was registered, so the next command reuses the number)."""
    for path in reversed(EventLog(root).files()):
        payload = json.loads(path.read_text())
        if payload.get("event") == "started" and payload.get("unit") == unit and payload.get("attempt") == attempt:
            return payload
    return None


def result_path(root, unit, attempt):
    """Where the worker writes its result: the fixed name once the unit has passed the spent point, an
    attempt-numbered name before (so a refusal never occupies a fixed name and never blocks a restart). The
    parent and the worker use the same spent test."""
    root = Path(root)
    spent = spent_signal(root, unit)
    if unit == "A0":
        return root / ("A0_RESULT.json" if spent else f"A0_REFUSED-{attempt}.json")
    return root / (f"worker-{unit}.result.json" if spent else f"worker-{unit}.attempt-{attempt}.result.json")


def add_common_arguments(parser):
    parser.add_argument("--record", required=True)
    parser.add_argument("--approval", required=True)
    parser.add_argument("--record-sha256", required=True)
    parser.add_argument("--attempt", required=True, type=int)
    parser.add_argument("--group9-root", default=None)
    return parser


def a0_completion_problems(exit_record, root, audit_required_relative, audit_allowed_relative):
    """The parent's decision that A0 is complete: exit 0, not timed out, A0_RESULT.json exists and the verdict
    recomputed from its stored values passes, the audit log is clean and lists the files A0 must open, and no
    module differed at exit."""
    problems = []
    if exit_record.get("returncode") != 0:
        problems.append(f"the worker exited with code {exit_record.get('returncode')}")
    if exit_record.get("timed_out"):
        problems.append("the worker was stopped at the wall ceiling")
    path = Path(root) / "A0_RESULT.json"
    if not path.is_file():
        return problems + ["A0_RESULT.json does not exist"]
    result = json.loads(path.read_text())
    passed, reasons = a0_verdict(result)
    if not passed:
        problems += [f"A0 verdict: {reason}" for reason in reasons]
    if result.get("fatal"):
        problems.append(f"the worker recorded a fatal stop: {result['fatal']}")
    if result.get("module_problems_at_exit"):
        problems.append("a module differed from the committed bytes at exit")
    problems += launch_checks.audit_problems(result.get("audit"), audit_required_relative, audit_allowed_relative)
    return problems
