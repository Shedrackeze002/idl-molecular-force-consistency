"""Launch records A and B as machine-readable inputs, and their approval (plan v3 section 2).

A record is a JSON file whose SHA-256 is the value a launch command is given. It names the commit, the
launch id and output root, every hash that exists before the launch (the 110 saved-result hashes, the
frozen files, the nine diff-gate hashes, the rules file, the relaxer), the ceilings, and, in record B,
the A0 record and root it follows. Values that exist only after a run (the A0 result hash, a stage's
marker hash) are never record fields; they are typed on the command line and recomputed.

An approval file holds the record's SHA-256 and two distinct review files by hash, as the frozen
`verify_launch_approval` precedent does. `verify_launch_context` runs the record, approval, tree and
frozen-file checks; the parent and every worker call it, so a worker started by hand meets the same
checks as one the parent starts.
"""
import hashlib
import json
import os
import re
from pathlib import Path

from stage2_capped import admission, driver, launch_checks, stage0_compare

CEILING_CAP_FACTOR = 3.0
CAFFEINATE_ARGV = ("caffeinate", "-i")
UNITS = ("A0", "0", "0b", "1", "2")
APPROVAL_STATUS = "approved_for_launch"
SAVED_INVENTORIES = (
    (stage0_compare.RECOVERY_RUN, "reports/execution/e3x-validation-recovery-2026-10-17/COMPLETED_OUTPUT_INVENTORY.json"),
    (stage0_compare.PAIRED_TRACE_RUN, "reports/execution/e3x-paired-trace-2026-10-18/COMPLETED_OUTPUT_INVENTORY.json"))

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
COMMON_FIELDS = ("schema", "record", "commit", "launch_id", "output_root", "output_volume", "root_source", "launch_folder",
                 "expected_launch_records_sha256", "saved_file_hashes", "frozen_file_hashes", "diff_gate_hashes",
                 "rules_py_sha256", "relaxer_sha256", "ceilings", "ceiling_cap_factor", "caffeinate",
                 "prior_dispatches")
RECORD_B_FIELDS = ("record_a_sha256", "a0_launch_id", "a0_output_root")


class LaunchRefused(RuntimeError):
    """A launch precondition failed. `problems` lists every reason found."""

    def __init__(self, problems):
        self.problems = [problems] if isinstance(problems, str) else list(problems)
        super().__init__("; ".join(self.problems))


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def is_hex64(value):
    return isinstance(value, str) and bool(_HEX64.match(value))


def launch_folder_of(record_path):
    """`<launch folder>/records/RECORD_x.json` lives in `<launch folder>`, beside `registry/` and `violations/`.
    The folder is resolved, so a symlinked path to the same records folder names the same registry."""
    return Path(record_path).resolve().parent.parent


def expected_ceilings():
    """The ceilings the spec fixes: the floor each wall time must reach and the attempt ceilings."""
    return {"a0_wall_seconds": driver.A0_WALL_SECONDS,
            "stages": {stage: {"runs": spec["runs"], "attempt_ceiling": spec["attempt_ceiling"],
                               "wall_seconds": spec["wall_seconds"]} for stage, spec in driver.STAGE_SPECS.items()}}


# ---- The record ------------------------------------------------------------

def structure_problems(record):
    """Shape and format of a record (no comparison with the disk)."""
    if not isinstance(record, dict):
        return ["a record is a JSON object"]
    kind = record.get("record")
    if kind not in ("A", "B"):
        return ["record must be \"A\" or \"B\""]
    required = COMMON_FIELDS + (RECORD_B_FIELDS if kind == "B" else ())
    problems = [f"missing field {name}" for name in required if name not in record]
    extra = sorted(set(record) - set(required))
    if extra:
        problems.append(f"unexpected fields {extra}")
    if problems:
        return problems
    if not (isinstance(record["commit"], str) and _HEX40.match(record["commit"])):
        problems.append("commit must be the full 40 hex characters")
    for name in ("launch_id", "output_root", "output_volume", "root_source", "launch_folder"):
        if not isinstance(record[name], str) or not record[name]:
            problems.append(f"{name} must be a non-empty string")
    for name in ("output_root", "output_volume", "launch_folder"):
        if isinstance(record[name], str) and not record[name].startswith("/"):
            problems.append(f"{name} must be an absolute path")
    for name in ("expected_launch_records_sha256", "rules_py_sha256", "relaxer_sha256"):
        if not is_hex64(record[name]):
            problems.append(f"{name} must be 64 hex characters")
    saved = record["saved_file_hashes"]
    if not isinstance(saved, dict) or len(saved) != driver.SAVED_FILE_COUNT or not all(
            isinstance(k, str) and is_hex64(v) for k, v in saved.items()):
        problems.append(f"saved_file_hashes must hold {driver.SAVED_FILE_COUNT} path to SHA-256 pairs")
    frozen = record["frozen_file_hashes"]
    if not isinstance(frozen, dict) or set(frozen) != set(launch_checks.FROZEN_FILES) or not all(
            is_hex64(v) for v in frozen.values()):
        problems.append("frozen_file_hashes must name the twelve frozen files")
    gates = record["diff_gate_hashes"]
    if not isinstance(gates, dict) or set(gates) != set(launch_checks.ALLOWLISTS) or not all(
            isinstance(v, dict) and set(v) == {"derived", "frozen", "allowlist"} and all(is_hex64(h) for h in v.values())
            for v in gates.values()):
        problems.append("diff_gate_hashes must hold derived, frozen and allowlist hashes for each of the three targets")
    ceilings = record["ceilings"]
    if not isinstance(ceilings, dict) or set(ceilings) != {"a0_wall_seconds", "stages"} or not isinstance(
            ceilings.get("stages"), dict) or set(ceilings["stages"]) != set(driver.STAGE_SPECS):
        problems.append("ceilings must hold a0_wall_seconds and the four stages")
    else:
        for stage, entry in ceilings["stages"].items():
            if not isinstance(entry, dict) or set(entry) != {"runs", "attempt_ceiling", "wall_seconds"}:
                problems.append(f"ceilings for stage {stage} must hold runs, attempt_ceiling and wall_seconds")
    if not isinstance(record["caffeinate"], list):
        problems.append("caffeinate must be the argv list")
    prior = record["prior_dispatches"]
    if not isinstance(prior, list) or not all(
            isinstance(item, list) and len(item) == 2 and isinstance(item[0], str) and is_hex64(item[1])
            for item in prior):
        problems.append("prior_dispatches must be a list of [event file name, SHA-256] pairs")
    if kind == "B":
        if not is_hex64(record["record_a_sha256"]):
            problems.append("record_a_sha256 must be 64 hex characters")
        for name in ("a0_launch_id", "a0_output_root"):
            if not isinstance(record[name], str) or not record[name]:
                problems.append(f"{name} must be a non-empty string")
    return problems


def ceiling_problems(ceilings, cap_factor=CEILING_CAP_FACTOR):
    """Runs and attempt ceilings equal the spec's; each wall ceiling lies between the spec's value (the
    floor) and the code constant cap (3 times the spec's value); the A0 ceiling likewise."""
    problems = []
    for stage, spec in driver.STAGE_SPECS.items():
        entry = ceilings["stages"][stage]
        for name in ("runs", "attempt_ceiling"):
            if entry[name] != spec[name]:
                problems.append(f"stage {stage} {name} is {entry[name]}, the spec fixes {spec[name]}")
        wall = entry["wall_seconds"]
        if not isinstance(wall, (int, float)) or isinstance(wall, bool) or not (spec["wall_seconds"] <= wall <= cap_factor * spec["wall_seconds"]):
            problems.append(f"stage {stage} wall ceiling {wall!r} lies outside [{spec['wall_seconds']}, "
                            f"{cap_factor * spec['wall_seconds']}]")
    a0 = ceilings["a0_wall_seconds"]
    if not isinstance(a0, (int, float)) or isinstance(a0, bool) or not (driver.A0_WALL_SECONDS <= a0 <= cap_factor * driver.A0_WALL_SECONDS):
        problems.append(f"the A0 wall ceiling {a0!r} lies outside [{driver.A0_WALL_SECONDS}, "
                        f"{cap_factor * driver.A0_WALL_SECONDS}]")
    return problems


def record_problems(record, command, *, worktree, group9=None):
    """Every way a record disagrees with the command, the code or the disk (empty list: it agrees)."""
    problems = structure_problems(record)
    if problems:
        return problems
    wanted = "A" if command == "a0" else "B"
    if record["record"] != wanted:
        problems.append(f"command {command!r} needs record {wanted}, not record {record['record']}")
    problems += ceiling_problems(record["ceilings"])
    if record["ceiling_cap_factor"] != CEILING_CAP_FACTOR:
        problems.append(f"ceiling_cap_factor must quote the code constant {CEILING_CAP_FACTOR}")
    if record["caffeinate"] != list(CAFFEINATE_ARGV):
        problems.append(f"caffeinate must be {list(CAFFEINATE_ARGV)}")
    if record["frozen_file_hashes"] != launch_checks.FROZEN_FILES:
        problems.append("frozen_file_hashes differ from the twelve recorded hashes in the code")
    problems += launch_checks.frozen_problems(worktree)
    root = admission.group9_root(group9)
    if launch_checks.sha256_file(root / admission.RECOVERY_RECORDS) != record["expected_launch_records_sha256"]:
        problems.append("expected_launch_records_sha256 differs from the accepted recovery launch records")
    if stage0_compare.saved_file_hashes(root) != record["saved_file_hashes"]:
        problems.append("saved_file_hashes differ from the saved result files on disk")
    problems += inventory_problems(record["saved_file_hashes"], group9)
    problems += placement_problems(record, worktree)
    gates, gate_problems = launch_checks.diff_gate_hashes(worktree)
    problems += gate_problems
    if gates != record["diff_gate_hashes"]:
        problems.append("diff_gate_hashes differ from the hashes of the gated files and allow-lists")
    if launch_checks.sha256_file(Path(worktree) / "stage2_capped" / "rules.py") != record["rules_py_sha256"]:
        problems.append("rules_py_sha256 differs from rules.py")
    if driver.relaxer_sha256() != record["relaxer_sha256"]:
        problems.append("relaxer_sha256 differs from the relaxer in use")
    return problems


def _inside(inner, outer):
    inner, outer = Path(inner).resolve(), Path(outer).resolve()
    return inner == outer or outer in inner.parents


def placement_problems(record, worktree):
    """The launch folder (records, registry, violations) lies outside the worktree and outside the output root, and
    the output root lies outside the launch folder: a spent unit's registry must survive a deleted root, and the
    registry's own writes must never dirty the tree the checks watch."""
    folder, root = record["launch_folder"], record["output_root"]
    problems = []
    if _inside(folder, worktree):
        problems.append(f"the launch folder {folder} lies inside the worktree")
    if _inside(folder, root) or _inside(root, folder):
        problems.append(f"the launch folder {folder} and the output root {root} contain one another")
    return problems


def inventory_problems(saved_hashes, group9=None):
    """The record's 110 saved-file hashes against the two accepted-run closeout inventories, which
    were written independently of this code. Offline check on the machine that has the saved files."""
    root = admission.group9_root(group9)
    expected = {}
    for run_directory, inventory in SAVED_INVENTORIES:
        listed = json.loads((root / inventory).read_text())
        for name, entry in listed.items():
            expected[f"{run_directory}/{name}"] = entry["sha256"] if isinstance(entry, dict) else entry
    problems = []
    for path, digest in sorted(saved_hashes.items()):
        if path not in expected:
            problems.append(f"{path} is in the record but in neither inventory")
        elif expected[path] != digest:
            problems.append(f"{path} differs from the inventory")
    return problems


def markdown_problems(markdown, record):
    """The Markdown text of a record must carry the same scalar values as its JSON body."""
    values = [record["record"] and f'Record {record["record"]}', record["commit"], record["launch_id"],
              record["output_root"], record["output_volume"], record["launch_folder"],
              record["expected_launch_records_sha256"], record["rules_py_sha256"], record["relaxer_sha256"]]
    values += list(record["frozen_file_hashes"].values())
    for gate in record["diff_gate_hashes"].values():
        values += list(gate.values())
    if record["record"] == "B":
        values += [record["record_a_sha256"], record["a0_launch_id"], record["a0_output_root"]]
    return [f"the Markdown text lacks {value!r}" for value in values if value not in markdown]


def load_record(path, record_sha256):
    """The record file, refused unless its SHA-256 is the one the command line names."""
    data = Path(path).read_bytes()
    if sha256_bytes(data) != record_sha256:
        raise LaunchRefused("the record file's SHA-256 differs from the one given on the command line")
    try:
        record = json.loads(data)
    except ValueError as error:
        raise LaunchRefused(f"the record is not JSON: {error}") from error
    problems = structure_problems(record)
    if problems:
        raise LaunchRefused(problems)
    return record


# ---- The approval ----------------------------------------------------------

def load_approval(path, record_sha256):
    """Two exact independent reviews bound to this record's SHA-256, and an approval line."""
    try:
        approval = json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise LaunchRefused(f"the approval file cannot be read: {error}") from error
    reviews = approval.get("review_sha256") if isinstance(approval, dict) else None
    problems = []
    if not isinstance(approval, dict) or approval.get("status") != APPROVAL_STATUS:
        problems.append(f"the approval status must be {APPROVAL_STATUS!r}")
        raise LaunchRefused(problems)
    if approval.get("record_sha256") != record_sha256:
        problems.append("the approval names a different record SHA-256")
    for field in ("approved_by", "approval_line"):
        if not isinstance(approval.get(field), str) or not approval[field].strip():
            problems.append(f"the approval lacks {field}")
    if not isinstance(reviews, dict) or len(reviews) != 2 \
            or not all(isinstance(p, str) and os.path.isabs(p) for p in reviews) \
            or not all(isinstance(v, str) for v in reviews.values()) \
            or len(set(reviews.values())) != 2 or len({str(Path(p).resolve()) for p in reviews}) != 2:
        problems.append("two distinct review files (absolute paths) with distinct hashes are required")
    else:
        for review, digest in reviews.items():
            try:
                if sha256_bytes(Path(review).read_bytes()) != digest:
                    problems.append(f"review {review} changed since it was approved")
            except OSError:
                problems.append(f"review {review} cannot be read")
    if problems:
        raise LaunchRefused(problems)
    return approval


# ---- The shared context check ---------------------------------------------

def folder_problems(record, record_path):
    """The registry of spent units is where the approved record says (the launch folder), never where a copy of the
    record happens to sit: a record file copied or moved elsewhere has to be refused, or it would see an empty
    registry and could start a spent unit again."""
    path = Path(record_path).resolve()
    if path.parent.name != "records":
        return [f"the record file must lie in a folder named records, not {path.parent}"]
    named = Path(record["launch_folder"]).resolve()
    if launch_folder_of(path) != named:
        return [f"the record names the launch folder {named}, but this record file lies in {launch_folder_of(path)}"]
    return []


def verify_launch_context(record_path, approval_path, record_sha256, *, command, worktree, group9=None):
    """The record, its approval, the tree and the frozen files, in one call. Returns the record.
    Raises LaunchRefused with every problem found. Called by the parent and by every worker."""
    record = load_record(record_path, record_sha256)
    load_approval(approval_path, record_sha256)
    problems = record_problems(record, command, worktree=worktree, group9=group9)
    problems += folder_problems(record, record_path)
    problems += launch_checks.tree_problems(worktree, record["commit"])
    if problems:
        raise LaunchRefused(problems)
    return record
