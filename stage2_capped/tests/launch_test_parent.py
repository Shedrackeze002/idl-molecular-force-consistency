"""Test-only driver: the real parent (`launch.execute`) with a worker command that runs the fixture worker
(plan v3 section 10, B7). The production command line does not expose `worker_command`; this module passes it
through the library function. Run as a subprocess from a committed scratch tree with PYTHONPATH set to that tree."""
import json
import os
import sys
from pathlib import Path

from stage2_capped import launch, launch_io, launch_record, launch_state
from stage2_capped.tests import launch_test_worker

SCENARIO = json.loads(os.environ.get("LAUNCH_TEST_SCENARIO", "{}"))


def worker_command(unit, attempt, args, record):
    argv = [*launch_record.CAFFEINATE_ARGV, sys.executable, "-m", "stage2_capped.tests.launch_test_worker",
            "--kind", "a0" if unit == "A0" else "stage", "--record", args.record, "--approval", args.approval,
            "--record-sha256", args.record_sha256, "--attempt", str(attempt)]
    if args.group9_root:
        argv += ["--group9-root", args.group9_root]
    if unit != "A0":
        argv += ["--unit", unit]
        for flag in ("a0_result_sha256", "stage0_sha256", "stage0b_sha256", "stage1_sha256"):
            if getattr(args, flag, None):
                argv += ["--" + flag.replace("_", "-"), getattr(args, flag)]
    return argv


def install_spies():
    """Scenario `spy_file`: one JSON line per supervise call (the wall seconds it was given) and per guard call (the
    ceiling it was given). Scenario `wall_override`: the supervisor gets that many seconds instead."""
    spy, override = SCENARIO.get("spy_file"), SCENARIO.get("wall_override")
    if not (spy or override):
        return
    from stage2_capped import launch_io, launch_state
    real_supervise, real_guard = launch_io.supervise, launch_state.guard_problems

    def note(**fields):
        if spy:
            with open(spy, "a") as handle:
                handle.write(json.dumps(fields) + "\n")

    def supervise(command, logfile, numerical_seconds, finalization_seconds):
        note(what="supervise", wall=numerical_seconds)
        return real_supervise(command, logfile, override or numerical_seconds, finalization_seconds)

    def guard(a0_result, stage, ceiling_seconds):
        note(what="guard", stage=stage, ceiling=ceiling_seconds)
        return real_guard(a0_result, stage, ceiling_seconds)
    launch_io.supervise, launch_state.guard_problems = supervise, guard


def install_faults():
    """Scenario `supervise_raises`: the supervisor cannot run the worker at all. Scenario `print_a0_raises`: the parent's
    own printing fails after the unit's exit has been recorded. Scenario `registry_raises_after_started_write`: the
    registry write of the `started` event links its file and then raises, as a failing unlink or directory fsync after
    the link would."""
    if SCENARIO.get("registry_raises_after_started_write"):
        original = launch_state.Registry.write

        def write(self, record_sha256, unit, attempt, event, **fields):
            result = original(self, record_sha256, unit, attempt, event, **fields)
            if event == "started":
                raise OSError("scenario: the registry write linked its file and then failed")
            return result
        launch_state.Registry.write = write
    if SCENARIO.get("supervise_raises"):
        def supervise(*arguments, **keywords):
            raise OSError("scenario: the worker could not be started")
        launch_io.supervise = supervise
    if SCENARIO.get("print_a0_raises"):
        def broken(*arguments, **keywords):
            raise RuntimeError("scenario: printing failed after the exit was recorded")
        launch.print_a0 = broken


def before_lock_hook(args):
    """Scenario `before_lock`: what another parent could have done in the window between this parent's checks and its
    lock (R2-N02): `spent_unit` writes a spent attempt of that unit into the registry, `root_file` puts a file into the
    output root, `create_root` makes the output root. The production command line has no such argument."""
    spec = SCENARIO.get("before_lock")
    if not spec:
        return None

    def hook():
        record = json.loads(Path(args.record).read_text())
        root = Path(record["output_root"])
        if spec.get("create_root"):
            launch_io.fresh_root(root, record["launch_id"])
        if spec.get("spent_unit"):
            registry = launch_state.Registry(launch_state.launch_folder_of(args.record) / "registry")
            registry.write(args.record_sha256, spec["spent_unit"], 1, "started", launch_id=record["launch_id"])
            registry.write(args.record_sha256, spec["spent_unit"], 1, "exited", outcome="failed", spent=True)
        if spec.get("root_file"):
            (root / spec["root_file"]).write_text("{}\n")
    return hook


def main(argv=None):
    install_spies()
    install_faults()
    args = launch.build_parser().parse_args(argv)
    command = f"check-{args.what}" if args.command == "check" else args.command
    deps = launch.ParentDeps(admit_all=launch_test_worker.parent_admit_all, worker_command=worker_command)
    return launch.execute(command, args, deps, finalization_seconds=float(SCENARIO.get("finalization", 5.0)),
                          before_lock=before_lock_hook(args))


if __name__ == "__main__":
    sys.exit(main())
