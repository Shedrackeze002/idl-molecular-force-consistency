"""Output roots, fixed names, atomic writes, the storage guard and process supervision
for the Stage 2 driver (spec v4 sections 5 and 10).

- Each launch gets a new output root; nothing is overwritten, and a partial write is
  never mistaken for a finished file (`.partial` then rename).
- Result rows and traces have fixed names built from the stage, the checkpoint, the
  fixed cap label (cinf, c0p05, c0p10, c0p25, c1p00) and the start.
- The storage guard measures the total size of all files under the root, including
  partial files, against 1 GiB, and checks free space with the frozen
  launch_e3x_candidates.require_storage.
- A stage that stops leaves its rows in place and an INCOMPLETE.json (never
  overwritten); there is no automatic resume or retry.
- Wall-time ceilings are enforced by the frozen bounded_process (SIGTERM, 60 s for
  finalization, SIGKILL), not by `invoke`.
"""
import os
import re
from pathlib import Path

from stage2_capped import row_schema

GIB = 2 ** 30
STAGES = row_schema.STAGES
FINALIZATION_SECONDS = 60.0


class StorageExceeded(RuntimeError):
    """The files under the output root reached the storage ceiling."""


def fresh_root(path):
    """A new, empty output root. An existing path is refused."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    return path


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


def atomic_write_bytes(path, data):
    """Write `data` to a path that must not exist: `.partial`, flush, fsync, rename."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with open(temporary, "xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return path


def write_row(path, row):
    """Validate and write one result row (no NaN or infinity, no overwrite)."""
    return atomic_write_bytes(path, (row_schema.dumps_row(row) + "\n").encode("utf-8"))


def mark_incomplete(stage_directory, reason, completed, failed_task):
    """Mark a stopped stage. Existing rows stay; the marker is never overwritten."""
    import json
    payload = {"complete": False, "reason": str(reason), "completed_runs": list(completed),
               "failed_task": failed_task, "automatic_retry": False,
               "partial_rows_are_operational_evidence_not_a_complete_stage": True}
    return atomic_write_bytes(Path(stage_directory) / "INCOMPLETE.json",
                              (json.dumps(payload, indent=1, allow_nan=False) + "\n").encode("utf-8"))


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
