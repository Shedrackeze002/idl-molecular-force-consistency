"""The launch parent and the only command line (plan v3 sections 3, 4, 6, 7, 11; amendments A1 to A7, B1 to B9).

    python -m stage2_capped.launch check a0|stage ...   everything that needs no model; writes nothing
    python -m stage2_capped.launch a0 --record R --approval P --record-sha256 H
    python -m stage2_capped.launch stage U --record R --approval P --record-sha256 H --a0-result-sha256 H [...]

There is no tuning flag, no root or launch-id flag (both come from the approved record) and no ledger: state is
derived from evidence. The parent takes the output root's lock, writes a `started` event to the root's log and to
the registry before it spawns the worker, supervises the worker under the record's wall ceiling inside
`caffeinate -i`, then decides from the files the worker left, never from the exit code alone, and writes an `exited`
event. A refusal before the spent point spends nothing and may be corrected under the same record; anything after
it, and any audit violation, spends the unit (a violation voids the record pair).

Run it from the worktree with an absolute PYTHONPATH: the supervisor passes no cwd or environment.
Exit codes: 0 complete, 1 spent and not complete, 2 refused (nothing spent) or a failed `check`.
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from stage2_capped import admission, driver, launch_checks, launch_io, launch_record, launch_state

EXIT_COMPLETE, EXIT_SPENT, EXIT_REFUSED = 0, 1, 2
REQUIRED_TYPED = {"0": (), "0b": ("0",), "1": ("0",), "2": ("1", "0b")}
FLAG_FOR_STAGE = {"0": "stage0_sha256", "0b": "stage0b_sha256", "1": "stage1_sha256"}


@dataclass
class ParentDeps:
    """`admit_all` is the model-free admission check; `worker_command` builds the worker's argv. The command line
    does not expose `worker_command`; tests pass their own through the library function."""
    admit_all: Callable
    worker_command: Optional[Callable] = None


def production_worker_command(unit, attempt, args, record):
    module = "stage2_capped.launch_a0" if unit == "A0" else "stage2_capped.launch_stage"
    argv = [*launch_record.CAFFEINATE_ARGV, sys.executable, "-m", module, "--record", args.record, "--approval",
            args.approval, "--record-sha256", args.record_sha256, "--attempt", str(attempt)]
    if args.group9_root:
        argv += ["--group9-root", args.group9_root]
    if unit != "A0":
        argv += ["--unit", unit]
        for flag in ("a0_result_sha256", "stage0_sha256", "stage0b_sha256", "stage1_sha256"):
            if getattr(args, flag, None):
                argv += ["--" + flag.replace("_", "-"), getattr(args, flag)]
    return argv


def production_deps():
    return ParentDeps(admit_all=admission.admit_all, worker_command=production_worker_command)


def unit_of(command, args):
    return "A0" if command == "a0" else args.unit


def typed_of(args):
    return {stage: getattr(args, flag) for stage, flag in FLAG_FOR_STAGE.items() if getattr(args, flag, None)}


# ---- Reconciliation ----------------------------------------------------------------------------------------

def reconcile(registry, root, record_sha256, record):
    """A2 and B4: for every `started` event without an `exited` event, confirm it spent (never write a refusal).
    For a stage whose directory exists without a marker, a worker that is still alive blocks the command; a dead
    one is marked INCOMPLETE. Returns a list of problems."""
    problems = []
    for unit in launch_record.UNITS:
        pending = [n for n, pair in registry.attempts(record_sha256, unit).items()
                   if pair["started"] is not None and pair["exited"] is None]
        for attempt in pending:
            pid_file = Path(root) / f"WORKER_PID-{unit}-{attempt}.json"
            if pid_file.is_file():
                if launch_state.worker_alive(json.loads(pid_file.read_text())):
                    problems.append(f"the worker of {unit} attempt {attempt} may still be running")
                    continue
            if unit != "A0":
                launch_io.mark_incomplete_if_absent(launch_state.stage_directory(root, unit), "supervisor: the worker is gone")
        if pending and not problems:
            registry.confirm_spent(record_sha256, unit, {"spent_signal": launch_state.spent_signal(root, unit)})
    return problems


# ---- The checks that need no model -----------------------------------------------------------------------------------

def unit_state_problems(registry, record_sha256, unit, root, launch_id):
    """The unit is unspent under this record and the root holds nothing a start of it may not find (A1, B1). The parent
    asks it twice: while it gathers its checks, and again under the root's lock (R2-N02), because the first answer can be
    stale by the time the lock is free."""
    problems = []
    if registry.is_spent(record_sha256, unit):
        problems.append(f"unit {unit} is already spent under this record")
    if Path(root).exists():
        try:
            launch_io.require_launch_root(root, launch_id)
        except ValueError as error:
            problems.append(str(error))
        else:
            problems += launch_state.root_contents_problems(root, unit)
    return problems


def gather_problems(command, args, deps, *, worktree, environ=None):
    """(record, unit, problems, hook report, ledger snapshot): every reason this command cannot start, in order of
    cost. Writes nothing. The record and approval checks come first; the model-free admission check runs under the
    audit hook. The ledger snapshot is the one the parent derived for a stage command (None otherwise): it goes
    into the `started` event, and the worker writes the one it derived, so the two can be compared."""
    unit, snapshot = unit_of(command, args), None
    try:
        record = launch_record.verify_launch_context(args.record, args.approval, args.record_sha256, command=command,
                                                     worktree=worktree, group9=args.group9_root)
    except launch_record.LaunchRefused as error:
        return None, unit, error.problems, None, None
    except admission.AdmissionError as error:               # an unusable Group9 root is a refusal (2), not a traceback (1)
        return None, unit, [f"{type(error).__name__}: {error}"], None, None
    problems = []
    folder = launch_state.launch_folder_of(args.record)
    registry = launch_state.Registry(folder / "registry")
    root = Path(record["output_root"])
    pair = {args.record_sha256} | ({record["record_a_sha256"]} if record["record"] == "B" else set())
    voided = launch_state.violation_files(folder, pair) + registry.violation_events(pair)
    if voided:
        problems.append(f"an audit violation voids this record pair: {voided}")
    problems += registry.prior_problems(pair, record["prior_dispatches"])
    problems += unit_state_problems(registry, args.record_sha256, unit, root, record["launch_id"])
    problems += launch_checks.preflight_problems(worktree=worktree, root=root, launch_folder=folder,
                                                 output_volume=record["output_volume"], environ=environ)
    if command == "stage":
        typed = typed_of(args)
        missing = [f"stage {unit} needs the typed marker hash of stage {s}" for s in REQUIRED_TYPED[unit] if s not in typed]
        problems += missing
        a0_hash = getattr(args, "a0_result_sha256", None)
        if not a0_hash:
            problems.append("every stage command needs --a0-result-sha256")
        else:
            a0, found = launch_state.a0_evidence(record, a0_hash, registry, current_runtime=launch_checks.runtime_values(environ),
                                                 group9=args.group9_root)
            problems += found
            if not found:
                problems += launch_state.guard_problems(a0, unit, record["ceilings"]["stages"][unit]["wall_seconds"])
                for flag in launch_state.guard_flags(a0, unit, record["ceilings"]["stages"][unit]["wall_seconds"]):
                    print(flag)
                if not missing:
                    try:
                        ledger, _ = launch_state.derive_ledger(root, record, typed, a0_valid=True,
                                                               relaxer_sha256=driver.relaxer_sha256(), registry=registry,
                                                               record_sha256=args.record_sha256)
                        snapshot = ledger.snapshot()
                        problems += ledger.blockers(unit)
                    except launch_record.LaunchRefused as error:
                        problems += error.problems
    guard, report = None, None
    if not problems:
        group9 = admission.group9_root(args.group9_root)
        tree, allowed = launch_checks.audit_scope(group9)
        guard = launch_checks.AuditGuard(tree, allowed, folder / "violations",
                                         violation_name=f"AUDIT_VIOLATION-{args.record_sha256}-{unit}-parent").enable()
        try:
            deps.admit_all(group9)
        except Exception as error:  # noqa: BLE001 - a failing admission check is a refusal
            problems.append(f"admission: {type(error).__name__}: {error}")
        finally:
            guard.disable()
        report = guard.report()
    return record, unit, problems, report, snapshot


# ---- Printing ----------------------------------------------------------------------------------------------------

def print_guard(result, record, complete):
    """When A0 ends: the post-A0 guard for every stage, so a missing latency or a ceiling a stage cannot meet shows now,
    not at the first stage command. Record A's ceilings are the ones on hand here; each stage command applies record B's."""
    if not complete:
        print("post-A0 guard: not applicable, A0 did not complete")
        return
    print("post-A0 guard (record A's ceilings; a stage command applies record B's):")
    for stage in driver.STAGE_SPECS:
        print("  " + launch_state.guard_summary(result, stage, record["ceilings"]["stages"][stage]["wall_seconds"]))


def print_a0(root, record, complete):
    result = json.loads((Path(root) / "A0_RESULT.json").read_text())
    passed, _ = launch_state.a0_verdict(result)
    print(f"A0 verdict: {'PASS' if passed else 'FAIL'}")
    for model_id, entry in result["models"].items():
        replay = entry.get("replay") or {}
        evidence = entry.get("evidence") or {}
        print(f"  {model_id}: {replay.get('status') or entry.get('failure')}; "
              f"batch4 max_abs {(evidence.get('batch4') or {}).get('max_abs_difference')}, "
              f"batch1 max_abs {(evidence.get('batch1') or {}).get('max_abs_difference')}, "
              f"bit-identical {(evidence.get('batch4') or {}).get('bit_identical_records')}/"
              f"{(evidence.get('batch1') or {}).get('bit_identical_records')}")
    print_guard(result, record, complete)


def print_stage(unit, complete, root, exit_record):
    directory = launch_state.stage_directory(root, unit)
    marker_path = directory / ("STAGE_COMPLETE.json" if complete else "INCOMPLETE.json")
    runs = len(list(directory.glob("*/*/*.json")))
    digest = launch_record.sha256_bytes(marker_path.read_bytes()) if marker_path.is_file() else "none"
    if complete:
        marker = json.loads(marker_path.read_text())
        print(f"stage {unit}: complete; runs {marker['runs']}; wall {exit_record['elapsed_seconds']:.1f} s; "
              f"marker {digest}; attempts (operational total) {marker['force_evaluations']}")
    else:
        print(f"STAGE {unit} INCOMPLETE, not a result; completed-run count {runs}; INCOMPLETE.json {digest}")


# ---- The command ---------------------------------------------------------------------------------------------------------

def refuse(problems):
    for problem in problems:
        print(f"refused: {problem}")
    return EXIT_REFUSED


def is_launch_root(record):
    """The output root exists and was made by `fresh_root` for this record's launch id."""
    try:
        launch_io.require_launch_root(Path(record["output_root"]), record["launch_id"])
    except ValueError:
        return False
    return True


def execute(command, args, deps, *, worktree=None, environ=None, finalization_seconds=launch_io.FINALIZATION_SECONDS,
            before_lock=None):
    """Run one launch command. Returns the exit code.

    Exit 1 means a unit was spent, so it is reachable only after the `started` event is in the registry: any exception
    raised before that (a missing record file, an unreadable saved file, an A0 result that is not JSON, a root created
    by another parent, a lock file that cannot be opened) is a refusal that spent nothing, printed and exit 2 (R2-N01).

    `before_lock` is a test-only injection point; the command line never passes it. It is called once, after the
    model-free checks and before the root is created or locked, so a test can change the world inside the window in
    which a stalled parent's checks would go stale."""
    started = {"registered": False}
    try:
        return run_command(command, args, deps, worktree=Path(worktree or driver.WORKTREE), environ=environ,
                           finalization_seconds=finalization_seconds, before_lock=before_lock, started=started)
    except Exception as error:  # noqa: BLE001 - before the `started` event nothing is spent: a refusal, not a traceback
        if started["registered"]:
            raise
        return refuse([f"{type(error).__name__}: {error}"])


def run_command(command, args, deps, *, worktree, environ, finalization_seconds, before_lock, started):
    """A command that is going to write anything (a reconciliation, an event, a root) holds the output root's lock for
    the whole of it (plan section 4): a parent that cannot take the lock writes nothing, whatever it was about to do."""
    checking = command.startswith("check-")
    record, unit, problems, hook_report, snapshot = gather_problems(command.replace("check-", ""), args, deps,
                                                                    worktree=worktree, environ=environ)
    if checking or record is None:
        if problems:
            return refuse(problems)
        print("check passed: nothing was written")
        return EXIT_COMPLETE
    root = Path(record["output_root"])
    existed = root.exists()
    if before_lock is not None:
        before_lock()
    if existed and not is_launch_root(record):
        return refuse(problems)                       # a directory this launch did not make: no lock file goes into it
    if not existed:
        if problems:
            return refuse(problems)
        launch_io.fresh_root(root, record["launch_id"])
    try:
        with launch_state.Lock(root):
            registry = launch_state.Registry(launch_state.launch_folder_of(args.record) / "registry")
            if existed:
                problems += reconcile(registry, root, args.record_sha256, record)
            if not problems:                          # the answers gathered before the lock may be stale (R2-N02)
                problems += unit_state_problems(registry, args.record_sha256, unit, root, record["launch_id"])
            if problems:
                return refuse(problems)
            return start_unit(command, unit, args, deps, record, hook_report, snapshot, worktree=worktree,
                              finalization_seconds=finalization_seconds, started=started)
    except launch_state.LockHeld as error:
        return refuse(problems + [str(error)])


def start_unit(command, unit, args, deps, record, hook_report, snapshot, *, worktree, finalization_seconds, started):
    """Everything from the `started` event to the `exited` event. The caller holds the root's lock."""
    root, folder = Path(record["output_root"]), launch_state.launch_folder_of(args.record)
    registry = launch_state.Registry(folder / "registry")
    attempt = registry.next_attempt(args.record_sha256, unit)
    log = launch_state.EventLog(root)
    fields = {"unit": unit, "attempt": attempt, "parent_pid": os.getpid(), "typed": typed_of(args),
              "a0_result_sha256": getattr(args, "a0_result_sha256", None), "model_free_check_opened": hook_report,
              "ledger_snapshot": snapshot}
    log.append("started", **fields)
    try:
        registry.write(args.record_sha256, unit, attempt, "started", launch_id=record["launch_id"],
                       output_root=record["output_root"], parent_pid=os.getpid(), model_free_check_opened=hook_report,
                       ledger_snapshot=snapshot)
    finally:
        # From the moment the registry holds the event a unit is spent unless its exit says otherwise, even if the write
        # raised after linking its file (R3-N01): that failure must exit 1, not report a refusal that spent nothing.
        started["registered"] = registry.holds(args.record_sha256, unit, attempt, "started")
    (root / "logs").mkdir(exist_ok=True)
    wall = record["ceilings"]["a0_wall_seconds"] if unit == "A0" else record["ceilings"]["stages"][unit]["wall_seconds"]
    build = deps.worker_command or production_worker_command
    exit_record, complete, outcome, problems = {"returncode": None, "timed_out": False, "elapsed_seconds": 0.0}, False, "failed", []
    try:
        exit_record = launch_io.supervise(build(unit, attempt, args, record), root / "logs" / f"worker-{unit}-{attempt}.log",
                                          wall, finalization_seconds)
    except Exception as error:  # noqa: BLE001 - a worker that could not be run left no files: decided like any other exit
        exit_record = {**exit_record, "supervisor_error": f"{type(error).__name__}: {error}"}
    finally:
        try:
            complete, outcome, problems = decide(unit, args, record, root, folder, exit_record, worktree)
        except Exception as error:  # noqa: BLE001 - a decision that cannot be made is a failed unit
            complete, outcome, problems = False, "failed", [f"the completion decision failed: {type(error).__name__}: {error}"]
        if "supervisor_error" in exit_record:
            problems = problems + [f"the worker could not be run: {exit_record['supervisor_error']}"]
        if unit != "A0" and not complete:
            launch_io.mark_incomplete_if_absent(launch_state.stage_directory(root, unit),
                                                f"worker did not finish ({exit_record.get('returncode')})")
        evidence = {"exit": exit_record, "problems": problems, "result_files": result_hashes(root, unit)}
        log.append("exited", unit=unit, attempt=attempt, outcome=outcome, spent=outcome != "refused_before_start",
                   **evidence)
        registry.write(args.record_sha256, unit, attempt, "exited", outcome=outcome,
                       spent=outcome != "refused_before_start", exit=exit_record, evidence=evidence["result_files"])
    for problem in problems:
        print(f"{outcome}: {problem}")
    if outcome == "refused_before_start":
        return EXIT_REFUSED
    if unit == "A0":
        if (root / "A0_RESULT.json").is_file():
            print_a0(root, record, complete)
    else:
        print_stage(unit, complete, root, exit_record)
    return EXIT_COMPLETE if complete else EXIT_SPENT


def result_hashes(root, unit):
    hashes = {}
    candidates = ["A0_RESULT.json", "A0_RESTORE_BEGUN.json"] if unit == "A0" else [f"worker-{unit}.result.json"]
    for name in candidates:
        path = Path(root) / name
        if path.is_file():
            hashes[name] = launch_record.sha256_bytes(path.read_bytes())
    if unit != "A0":
        marker = launch_state.stage_directory(root, unit) / "STAGE_COMPLETE.json"
        if marker.is_file():
            hashes["STAGE_COMPLETE.json"] = launch_record.sha256_bytes(marker.read_bytes())
    return hashes


def decide(unit, args, record, root, folder, exit_record, worktree):
    """(complete, outcome, problems). An audit violation (the fixed exit code or a violation file) voids the unit
    and the pair; no spent signal means a refusal; otherwise the completion decision."""
    pair = {args.record_sha256} | ({record["record_a_sha256"]} if record["record"] == "B" else set())
    if exit_record.get("returncode") == launch_checks.AUDIT_EXIT_CODE or launch_state.violation_files(folder, pair):
        return False, "audit_violation", ["an audit violation voids this unit and the record pair"]
    if not launch_state.spent_signal(root, unit):
        return False, "refused_before_start", [f"the worker stopped before the spent point (exit {exit_record.get('returncode')})"]
    group9 = admission.group9_root(args.group9_root)
    tree, allowed = launch_checks.audit_scope(group9)
    allowed_relative = launch_checks.allowed_relative(tree, allowed)
    required = launch_checks.required_relative(tree, allowed, unit)
    problems = launch_checks.tree_problems(worktree, record["commit"])
    if unit == "A0":
        problems += launch_state.a0_completion_problems(exit_record, root, required, allowed_relative)
    else:
        result_path = Path(root) / f"worker-{unit}.result.json"
        audit = json.loads(result_path.read_text()).get("audit") if result_path.is_file() else None
        problems += launch_state.completion_problems(exit_record, root, unit, record, record["launch_id"],
                                                     driver.relaxer_sha256(), audit, required, allowed_relative)
    return (not problems), ("complete" if not problems else "failed"), problems


# ---- The command line ------------------------------------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(prog="python -m stage2_capped.launch")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, stage):
        p.add_argument("--record", required=True)
        p.add_argument("--approval", required=True)
        p.add_argument("--record-sha256", required=True)
        p.add_argument("--group9-root", default=None)
        if stage:
            for name in ("a0-result-sha256", "stage0-sha256", "stage0b-sha256", "stage1-sha256"):
                p.add_argument(f"--{name}", default=None)
        return p
    check = sub.add_parser("check")
    check_sub = check.add_subparsers(dest="what", required=True)
    common(check_sub.add_parser("a0"), False)
    stage_check = common(check_sub.add_parser("stage"), True)
    stage_check.add_argument("unit", choices=sorted(driver.STAGE_SPECS))
    common(sub.add_parser("a0"), False)
    stage = common(sub.add_parser("stage"), True)
    stage.add_argument("unit", choices=sorted(driver.STAGE_SPECS))
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    command = f"check-{args.what}" if args.command == "check" else args.command
    return execute(command, args, production_deps())


if __name__ == "__main__":
    sys.exit(main())
