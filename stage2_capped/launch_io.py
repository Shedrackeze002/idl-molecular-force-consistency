"""Output roots, fixed names, atomic writes, the storage guard and process supervision
for the Stage 2 driver (spec v4 sections 5 and 10).

- Each launch gets a new output root, marked with its launch id (LAUNCH.json); nothing is
  overwritten, and a partial write is never mistaken for a finished file (`.partial`
  fsynced, then a hard link that fails if the target exists, then the `.partial` removed).
- Result rows and traces have fixed names built from the stage, the checkpoint, the
  fixed cap label (cinf, c0p05, c0p10, c0p25, c1p00) and the start.
- The storage guard measures the total size of all files under the root, including
  partial files, against 1 GiB, and checks free space with the frozen
  launch_e3x_candidates.require_storage.
- A stage that stops leaves its rows in place and an INCOMPLETE.json (never
  overwritten); there is no automatic resume or retry. A worker killed hard cannot write
  the marker, so the supervisor writes it afterwards (`mark_incomplete_if_absent`).
- Wall-time ceilings are enforced by the frozen bounded_process (SIGTERM, 60 s for
  finalization, SIGKILL), not by `invoke`.
"""
import json
import os
import re
from pathlib import Path

from stage2_capped import row_schema

GIB = 2 ** 30
STAGES = row_schema.STAGES
FINALIZATION_SECONDS = 60.0
LAUNCH_MARKER = "LAUNCH.json"


class StorageExceeded(RuntimeError):
    """The files under the output root reached the storage ceiling."""


def fresh_root(path, launch_id):
    """A new output root, marked with its launch id. An existing path is refused."""
    if not isinstance(launch_id, str) or not launch_id:
        raise ValueError("a launch id is required")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    atomic_write_bytes(path / LAUNCH_MARKER,
                       (json.dumps({"launch_id": launch_id, "schema": 1}) + "\n").encode("utf-8"))
    return path


def require_launch_root(root, launch_id):
    """The root was made by fresh_root for this launch id; anything else is refused."""
    marker = Path(root) / LAUNCH_MARKER
    if not marker.is_file():
        raise ValueError(f"{root} was not made by fresh_root: no {LAUNCH_MARKER}")
    recorded = json.loads(marker.read_text()).get("launch_id")
    if recorded != launch_id:
        raise ValueError(f"{root} belongs to launch {recorded!r}, not {launch_id!r}")


def slug(text):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", text)


def output_paths(root, stage, checkpoint_id, cap_label, start_kind, start_id):
    """The fixed row and trace paths of one run."""
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    if cap_label not in row_schema.CAP_VALUES:
        raise ValueError(f"unknown cap label {cap_label!r}")
    if start_kind not in row_schema.START_KINDS:
        raise ValueError(f"unknown start kind {start_kind!r}")
    name = f"{start_kind}-{slug(start_id)}"
    base = Path(root) / "rows" / f"stage{stage}" / slug(checkpoint_id) / cap_label
    return {"row": base / f"{name}.json", "trace": base / f"{name}.npy",
            "relative_trace": str((base / f"{name}.npy").relative_to(root))}


def _fsync_directory(directory):
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_bytes(path, data):
    """Write `data` to a path that must not exist: `.partial`, flush, fsync, then a hard link
    to the final name (which fails if another file appeared meanwhile), then drop the
    `.partial` and fsync the folder. The final name never holds a partial file."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with open(temporary, "xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        os.unlink(temporary)
        raise FileExistsError(f"refusing to overwrite {path}") from None
    os.unlink(temporary)
    _fsync_directory(path.parent)
    return path


def write_row(path, row):
    """Validate and write one result row (no NaN or infinity, no overwrite)."""
    return atomic_write_bytes(path, (row_schema.dumps_row(row) + "\n").encode("utf-8"))


def mark_incomplete(stage_directory, reason, completed, failed_task):
    """Mark a stopped stage. Existing rows stay; the marker is never overwritten."""
    payload = {"complete": False, "reason": str(reason), "completed_runs": list(completed),
               "failed_task": failed_task, "automatic_retry": False,
               "partial_rows_are_operational_evidence_not_a_complete_stage": True}
    return atomic_write_bytes(Path(stage_directory) / "INCOMPLETE.json",
                              (json.dumps(payload, indent=1, allow_nan=False) + "\n").encode("utf-8"))


def mark_incomplete_if_absent(stage_directory, reason):
    """The supervisor's marker for a worker that was killed before it could mark its own stage.
    Does nothing (returns None) if the stage directory is missing, or already holds
    INCOMPLETE.json or STAGE_COMPLETE.json. The finished rows on disk are listed by their paths
    relative to the stage folder; a row still being written is a `.partial` file and is not listed."""
    directory = Path(stage_directory)
    if not directory.is_dir() or (directory / "INCOMPLETE.json").exists() \
            or (directory / "STAGE_COMPLETE.json").exists():
        return None
    rows = sorted(str(p.relative_to(directory)) for p in directory.glob("*/*/*.json"))
    return mark_incomplete(directory, f"supervisor: {reason}", rows, None)


# ---- Storage -------------------------------------------------------------

def total_size(root):
    """Total size in bytes of every file under `root`, partial files included."""
    return sum(p.stat().st_size for p in Path(root).rglob("*") if p.is_file())


def require_free(path, required_bytes):
    """Free-space check, using the frozen require_storage."""
    from launch_e3x_candidates import require_storage
    return require_storage(path, required_bytes)


class StorageGuard:
    def __init__(self, root, limit=GIB):
        self.root = Path(root)
        self.limit = limit

    def used(self):
        return total_size(self.root)

    def check(self, next_bytes=0):
        used = self.used()
        if used + next_bytes > self.limit:
            raise StorageExceeded(f"{used} bytes under {self.root} plus {next_bytes} reserved "
                                  f"exceed the {self.limit}-byte ceiling")
        return used


# ---- Supervision ---------------------------------------------------------

def supervise(command, logfile, numerical_seconds, finalization_seconds=FINALIZATION_SECONDS):
    """Run a worker with a numerical wall ceiling: SIGTERM at the ceiling, SIGKILL after
    the finalization window. Returns the frozen bounded_process record."""
    from trace_validation_recovery import bounded_process
    return bounded_process(command, logfile, numerical_seconds, finalization_seconds)
