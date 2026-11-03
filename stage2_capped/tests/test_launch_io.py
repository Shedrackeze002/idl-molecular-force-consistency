"""M4: output roots, fixed names, atomic writes, storage guard and supervision."""
import json
import os
import sys
import time

import pytest

from stage2_capped import launch_io as io
from stage2_capped import row_schema
from stage2_capped.tests.table_builder import STARTS, build_rows


def test_a_launch_root_is_new_and_never_reused(tmp_path):
    root = io.fresh_root(tmp_path / "launch", "L-1")
    assert root.is_dir()
    with pytest.raises(FileExistsError):
        io.fresh_root(tmp_path / "launch", "L-1")


def test_an_existing_folder_without_a_marker_is_never_taken_as_a_launch_root(tmp_path):
    existing = tmp_path / "already-here"
    existing.mkdir()
    (existing / "stray.txt").write_text("x")
    with pytest.raises(FileExistsError):
        io.fresh_root(existing, "L-1")
    assert [p.name for p in existing.iterdir()] == ["stray.txt"]              # nothing was added


def test_a_launch_root_carries_its_launch_id_and_other_roots_are_refused(tmp_path):
    root = io.fresh_root(tmp_path / "launch", "L-1")
    assert json.loads((root / io.LAUNCH_MARKER).read_text())["launch_id"] == "L-1"
    io.require_launch_root(root, "L-1")
    with pytest.raises(ValueError, match="belongs to launch 'L-1'"):
        io.require_launch_root(root, "L-2")
    bare = tmp_path / "bare"
    bare.mkdir()
    with pytest.raises(ValueError, match="not made by fresh_root"):
        io.require_launch_root(bare, "L-1")
    for bad in ("", None, 7):
        with pytest.raises(ValueError, match="launch id"):
            io.fresh_root(tmp_path / f"x{bad}", bad)


def test_output_names_are_fixed_per_cap_and_run(tmp_path):
    names = {}
    for cap_label in row_schema.CAP_VALUES:
        paths = io.output_paths(tmp_path, "1", "direct-0", cap_label, "original", "aspirin/13/plus")
        names[cap_label] = paths
        assert paths["row"].name == "original-aspirin_13_plus.json"
        assert paths["trace"].suffix == ".npy" and paths["relative_trace"].endswith(".npy")
        assert cap_label in paths["row"].parts and "stage1" in paths["row"].parts
    assert len({p["row"] for p in names.values()}) == 5          # one directory per cap
    assert io.output_paths(tmp_path, "0", "energy-0", "cinf", "reference", "min3")["row"].name == "reference-min3.json"


@pytest.mark.parametrize("bad", [("9", "x", "cinf", "original", "s"), ("0", "x", "c0p07", "original", "s"),
                                 ("0", "x", "cinf", "other", "s")])
def test_unknown_stages_caps_and_kinds_are_refused(tmp_path, bad):
    with pytest.raises(ValueError):
        io.output_paths(tmp_path, *bad)


def test_atomic_write_leaves_no_partial_file_and_never_overwrites(tmp_path):
    path = io.atomic_write_bytes(tmp_path / "a" / "b.json", b"{}")
    assert path.read_bytes() == b"{}" and not list(tmp_path.rglob("*.partial"))
    with pytest.raises(FileExistsError):
        io.atomic_write_bytes(path, b"other")
    assert path.read_bytes() == b"{}"


def test_atomic_write_fsyncs_the_file_before_the_final_name_exists_and_then_the_folder(tmp_path, monkeypatch):
    events = []
    real_fsync, real_link = os.fsync, os.link
    monkeypatch.setattr(os, "fsync", lambda fd: (events.append(("fsync", os.fstat(fd).st_mode & 0o170000)),
                                                  real_fsync(fd))[1])
    monkeypatch.setattr(os, "link", lambda a, b: (events.append(("link", (tmp_path / "f.json").exists())),
                                                   real_link(a, b))[1])
    io.atomic_write_bytes(tmp_path / "f.json", b"data")
    kinds = [e[0] for e in events]
    assert kinds == ["fsync", "link", "fsync"]                      # file, then link, then the folder
    assert events[0][1] == 0o100000 and events[2][1] == 0o040000    # a regular file, then a directory
    assert events[1][1] is False                                    # the final name did not exist at link time


def test_a_crash_before_the_final_name_leaves_only_a_partial_file(tmp_path, monkeypatch):
    def crash(source, target):
        raise KeyboardInterrupt("simulated kill")
    monkeypatch.setattr(os, "link", crash)
    with pytest.raises(KeyboardInterrupt):
        io.atomic_write_bytes(tmp_path / "row.json", b"{}")
    assert not (tmp_path / "row.json").exists()                     # never a half-written final file
    assert [p.name for p in tmp_path.iterdir()] == ["row.json.partial"]


def test_a_file_that_appears_while_writing_is_never_overwritten(tmp_path, monkeypatch):
    target = tmp_path / "row.json"
    real_link = os.link

    def racing(source, destination):
        target.write_bytes(b"someone else")                         # created after the exists() check
        return real_link(source, destination)

    monkeypatch.setattr(os, "link", racing)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        io.atomic_write_bytes(target, b"mine")
    assert target.read_bytes() == b"someone else" and not list(tmp_path.glob("*.partial"))


def test_the_supervisor_marks_a_killed_stage_once_and_lists_only_finished_rows(tmp_path):
    stage = tmp_path / "rows" / "stage1"
    io.write_row(stage / "m" / "c0p05" / "original-a.json", build_rows()[0])
    (stage / "m" / "c0p05" / "original-b.json.partial").write_bytes(b"{")      # a row cut off by SIGKILL
    marker = io.mark_incomplete_if_absent(stage, "worker killed after the finalization window")
    payload = json.loads(marker.read_text())
    assert payload["complete"] is False and payload["failed_task"] is None
    assert payload["reason"].startswith("supervisor: ") and payload["automatic_retry"] is False
    assert payload["completed_runs"] == ["m/c0p05/original-a.json"]
    assert io.mark_incomplete_if_absent(stage, "again") is None                # never overwritten
    assert json.loads(marker.read_text()) == payload


def test_the_supervisor_leaves_finished_and_missing_stages_alone(tmp_path):
    assert io.mark_incomplete_if_absent(tmp_path / "rows" / "stage0", "x") is None   # no folder, no marker
    done = tmp_path / "rows" / "stage1"
    done.mkdir(parents=True)
    (done / "STAGE_COMPLETE.json").write_text("{}")
    assert io.mark_incomplete_if_absent(done, "x") is None and not (done / "INCOMPLETE.json").exists()


def test_rows_are_validated_and_written_without_nan_or_infinity(tmp_path):
    row = build_rows()[0]
    path = io.write_row(tmp_path / "row.json", row)
    assert json.loads(path.read_text()) == row
    bad = dict(row, elapsed_s=float("nan"))
    with pytest.raises(ValueError):
        io.write_row(tmp_path / "bad.json", bad)
    assert not (tmp_path / "bad.json").exists()


def test_a_stopped_stage_keeps_its_rows_and_gets_a_marker_that_is_never_overwritten(tmp_path):
    stage = tmp_path / "rows" / "stage1"
    io.write_row(stage / "a.json", build_rows()[0])
    marker = io.mark_incomplete(stage, "boom", ["t1", "t2"], "t3")
    payload = json.loads(marker.read_text())
    assert payload["complete"] is False and payload["completed_runs"] == ["t1", "t2"]
    assert payload["failed_task"] == "t3" and payload["automatic_retry"] is False
    assert (stage / "a.json").exists()
    with pytest.raises(FileExistsError):
        io.mark_incomplete(stage, "again", [], None)


def test_the_storage_total_counts_every_file_including_partial_ones(tmp_path):
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "a.bin").write_bytes(b"x" * 100)
    (tmp_path / "b.json.partial").write_bytes(b"y" * 50)
    (tmp_path / "d" / "e").mkdir()
    (tmp_path / "d" / "e" / "c.npy").write_bytes(b"z" * 25)
    assert io.total_size(tmp_path) == 175


def test_the_storage_guard_trips_at_the_ceiling_and_counts_the_next_write(tmp_path):
    guard = io.StorageGuard(tmp_path, limit=200)
    (tmp_path / "a").write_bytes(b"x" * 150)
    assert guard.check() == 150
    assert guard.check(next_bytes=50) == 150
    with pytest.raises(io.StorageExceeded):
        guard.check(next_bytes=51)
    (tmp_path / "b").write_bytes(b"x" * 51)
    with pytest.raises(io.StorageExceeded):
        guard.check()


def test_the_default_storage_ceiling_is_one_gibibyte(tmp_path):
    assert io.StorageGuard(tmp_path).limit == 2 ** 30 == io.GIB


def test_the_free_space_check_uses_the_frozen_require_storage(tmp_path, monkeypatch):
    import shutil
    from collections import namedtuple
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(shutil, "disk_usage", lambda p: usage(10, 5, 5))
    assert io.require_free(tmp_path / "not-yet", 4)["free_bytes"] == 5
    with pytest.raises(OSError, match="Insufficient"):
        io.require_free(tmp_path, 6)
    with pytest.raises(OSError):
        io.require_free(tmp_path, 0)


def test_supervise_returns_the_frozen_record_for_a_normal_exit(tmp_path):
    record = io.supervise([sys.executable, "-c", "print('ok')"], tmp_path / "log.txt", 30)
    assert record["returncode"] == 0 and not record["timed_out"] and not record["hard_kill"]
    assert "ok" in (tmp_path / "log.txt").read_text()


def test_supervise_terminates_a_worker_that_overruns_and_keeps_its_partial_output(tmp_path):
    out = tmp_path / "progress.txt"
    code = (f"import time,signal,sys\nsignal.signal(signal.SIGTERM, lambda *a: (open({str(out)!r},'a').write('finalized\\n'), sys.exit(3)))\n"
            f"open({str(out)!r},'w').write('row 1\\n')\ntime.sleep(60)\n")
    started = time.perf_counter()
    record = io.supervise([sys.executable, "-c", code], tmp_path / "log.txt", 2, 10)
    assert record["timed_out"] and not record["hard_kill"] and record["returncode"] == 3
    assert time.perf_counter() - started < 30
    assert out.read_text() == "row 1\nfinalized\n"


def test_supervise_kills_a_worker_that_ignores_sigterm_after_the_finalization_window(tmp_path):
    code = "import signal,time\nsignal.signal(signal.SIGTERM, signal.SIG_IGN)\ntime.sleep(60)\n"
    started = time.perf_counter()
    record = io.supervise([sys.executable, "-c", code], tmp_path / "log.txt", 1, 1)
    assert record["timed_out"] and record["hard_kill"] and record["returncode"] != 0
    assert time.perf_counter() - started < 30


def test_the_default_finalization_window_is_sixty_seconds():
    import inspect
    assert io.FINALIZATION_SECONDS == 60.0
    assert inspect.signature(io.supervise).parameters["finalization_seconds"].default == 60.0
