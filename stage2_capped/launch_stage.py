"""The stage worker (plan v3 section 6): one process runs one stage of the fixed schedule.

`run(args, deps)` does the work and `main()` builds the production dependencies; nothing on the command line or in
the environment selects another set. The worker never takes a ledger from its parent: it derives the same state from
the same evidence (`launch_state.derive_ledger`) and requires the A0 result the parent validated.

Before the spent point (the stage directory, created by `run_stage` after every arm is built and before the first
force call) every failure is a refusal: written under an attempt-numbered name, exit 2, spending nothing. After it, a
result is written under the fixed name. Exit 0: the stage ran to its marker. Exit 3: a labelled stop (a driver stop, a
storage trip, a Stage 0 stop). Exit 4: another exception.

Never passed to the driver: `allow_unledgered`, `allow_custom_tasks`, `allow_custom` or `tasks`.
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from stage2_capped import admission, driver, launch_checks, launch_io, launch_record, launch_state
from stage2_capped.admission import PinPolicy
from stage2_capped.stage_gate import Stage0Controller

EXIT_OK, EXIT_REFUSED, EXIT_STOPPED, EXIT_OTHER = 0, 2, 3, 4
CONTROL_STAGE = {"1": "0", "2": "0b"}


@dataclass
class StageDeps:
    """Data sources, never enforcement switches."""
    admit_all: Callable
    load_bindings: Callable
    data_directory: Callable
    provider_factory: Callable
    inputs_loader: Callable
    saved_loader: Callable


class Refused(RuntimeError):
    """A precondition failed before the spent point."""


def production_deps():
    from stage2_capped import launch_arms, stage0_compare

    def saved_loader(expected_hashes, group9):
        return stage0_compare.load_saved(group9, expected_hashes=expected_hashes)
    return StageDeps(admit_all=admission.admit_all, load_bindings=admission.load_bindings,
                     data_directory=admission.data_directory, provider_factory=launch_arms.real_arm_provider,
                     inputs_loader=lambda expected, group9: driver.load_study_inputs(expected, group9=group9),
                     saved_loader=saved_loader)


def typed_hashes(args):
    typed = {"0": args.stage0_sha256, "0b": args.stage0b_sha256, "1": args.stage1_sha256}
    return {stage: value for stage, value in typed.items() if value}


def write_json(path, payload):
    return launch_io.atomic_write_bytes(path, launch_state.dumps(payload))


def control_problems(index, tasks, root, control_stage, typed, record, context, relaxer_sha256):
    """S11: the index covers every control key of this stage's schedule, the control marker is the typed one, and
    its manifest names this launch, this relaxer and these checkpoints."""
    problems = []
    missing = [driver.task_id(t) for t in tasks if (t["model_id"], t["kind"], driver.start_id_of(t)) not in index.records]
    if missing:
        problems.append(f"the control index lacks {len(missing)} of this stage's control runs, first {missing[0]}")
    directory = launch_state.stage_directory(root, control_stage)
    marker_sha = launch_record.sha256_bytes((directory / "STAGE_COMPLETE.json").read_bytes())
    if typed.get(control_stage) != marker_sha:
        problems.append(f"the control root's stage {control_stage} marker is not the typed one")
    manifest = json.loads((directory / "STAGE_MANIFEST.json").read_text())
    if manifest.get("launch_id") != record["launch_id"] or manifest.get("relaxer_sha256") != relaxer_sha256:
        problems.append("the control manifest names a different launch or relaxer")
    for model_id in dict.fromkeys(t["model_id"] for t in tasks):
        recorded = (manifest.get("arms", {}).get(model_id) or {}).get("checkpoint_identity")
        current = json.loads(json.dumps(context.arm(model_id).get("checkpoint_identity")))
        if recorded != current:
            problems.append(f"{model_id}: the control's checkpoint identity differs from this launch's")
    return problems


def run(args, deps, *, environ=None, worktree=None):
    """Run one stage once. Returns the exit code."""
    worktree = Path(worktree or driver.WORKTREE)
    unit, attempt = args.unit, args.attempt
    record_path = Path(args.record).resolve()
    record = launch_record.load_record(args.record, args.record_sha256)
    root, launch_folder = Path(record["output_root"]), launch_state.launch_folder_of(record_path)
    guard, payload, code = None, {"unit": unit, "attempt": attempt}, EXIT_OK
    try:
        try:
            launch_state.write_worker_pid(root, unit, attempt)                 # the first act (B4)
            started = launch_state.started_event(root, unit, attempt)
            if started is None or started.get("parent_pid") != os.getppid():
                raise Refused("no `started` event for this attempt names this process's parent")
            try:
                launch_record.verify_launch_context(args.record, args.approval, args.record_sha256, command="stage",
                                                    worktree=worktree, group9=args.group9_root)
            except launch_record.LaunchRefused as error:
                raise Refused(str(error)) from error
            problems = launch_checks.thread_variable_problems(environ)
            modules_at_start, module_problems = launch_checks.module_inventory(worktree)
            if problems or module_problems:
                raise Refused("; ".join(problems + module_problems))
            group9 = admission.group9_root(args.group9_root)
            tree, allowed = launch_checks.audit_scope(group9)
            guard = launch_checks.AuditGuard(
                tree, allowed, launch_folder / "violations",
                violation_name=f"AUDIT_VIOLATION-{args.record_sha256}-{unit}-{attempt}").enable()
            relaxer = driver.relaxer_sha256()
            registry = launch_state.Registry(launch_folder / "registry")
            current_runtime = launch_checks.runtime_values(environ, include_backend=True)
            a0, a0_problems = launch_state.a0_evidence(record, args.a0_result_sha256, registry,
                                                       current_runtime=current_runtime, group9=group9)
            if a0_problems:
                raise Refused("; ".join(a0_problems))
            typed = typed_hashes(args)
            ledger, _ = launch_state.derive_ledger(root, record, typed, a0_valid=True, relaxer_sha256=relaxer,
                                                   registry=registry, record_sha256=args.record_sha256)
            payload["ledger_snapshot"] = ledger.snapshot()
            if ledger.blockers(unit):
                raise Refused("; ".join(ledger.blockers(unit)))
            contents = launch_state.root_contents_problems(root, unit)
            if contents:
                raise Refused("; ".join(contents))
            report = deps.admit_all(group9)
            if report["identities"] != a0["admission"]["identities"] \
                    or report["launch_records_sha256"] != record["expected_launch_records_sha256"]:
                raise Refused("admission now differs from the identities or launch records the A0 result holds")
            items = deps.load_bindings(group9)
            data_directory = deps.data_directory(group9)
            policy = PinPolicy()
            provider = deps.provider_factory(items, data_directory, policy)
            inputs = deps.inputs_loader(record["expected_launch_records_sha256"], group9)
            context = driver.Context(root, record["launch_id"], inputs, provider)
            tasks = driver.build_schedule(unit, sorted(inputs.starts), inputs.reference_ids)
            saved = controller = control_index = None
            if unit == "0":
                saved = deps.saved_loader(record["saved_file_hashes"], group9)
                controller = Stage0Controller(expected_runs=driver.STAGE_SPECS["0"]["runs"])
            if unit in CONTROL_STAGE:
                control_index = driver.ControlIndex.from_roots([root], stages=(CONTROL_STAGE[unit],))
                found = control_problems(control_index, tasks, root, CONTROL_STAGE[unit], typed, record, context, relaxer)
                if found:
                    raise Refused("; ".join(found))
        except Refused as error:
            launch_io.atomic_write_bytes(launch_state.result_path(root, unit, attempt), launch_state.dumps(
                {**payload, "label": "refused_before_start", "reason": str(error)}))
            return EXIT_REFUSED
        except Exception as error:  # noqa: BLE001 - any precondition failure is a refusal that spends nothing
            if launch_state.spent_signal(root, unit):
                raise
            launch_io.atomic_write_bytes(launch_state.result_path(root, unit, attempt), launch_state.dumps(
                {**payload, "label": "refused_before_start", "reason": f"{type(error).__name__}: {error}"}))
            return EXIT_REFUSED

        payload.update(ok=False, record_sha256=args.record_sha256, launch_id=record["launch_id"])
        try:
            saved_hashes = record["saved_file_hashes"] if unit == "0" else None
            try:
                payload["summary"] = driver.run_stage(unit, context, ledger=ledger, saved=saved, controller=controller,
                                                      control_index=control_index, saved_hashes=saved_hashes)
                payload["ok"] = True
            except driver.Stage0Stop as stop:
                payload["error"] = f"Stage0Stop: {stop}"
                if stop.verdict["verdict"] == "frozen re-run required":
                    task = next(t for t in tasks if driver.task_id(t) == stop.task_id)
                    write_json(root / "STAGE0_RERUN_ATTEMPTED.json", {"task": stop.task_id})
                    verdict = driver.run_frozen_rerun(context, task, saved, controller)   # authorizes itself, once
                    write_json(root / "STAGE0_RERUN.json", verdict)
                    payload["frozen_rerun"] = verdict
                code = EXIT_STOPPED
        except (driver.DriverError, launch_io.StorageExceeded) as error:       # a labelled stop of the stage
            payload.setdefault("error", f"{type(error).__name__}: {error}")
            code = EXIT_STOPPED
        except Exception as error:  # noqa: BLE001 - reported, then nonzero
            payload["error"] = f"{type(error).__name__}: {error}"
            code = EXIT_OTHER
        launch_state.ignore_sigterm()                       # from here only the result matters
        try:
            modules_at_exit, exit_problems = launch_checks.module_inventory(worktree)
            payload.update(modules_at_start=modules_at_start, modules_at_exit=modules_at_exit,
                           module_problems_at_exit=exit_problems, runtime=current_runtime,
                           diff_gate_hashes=launch_checks.diff_gate_hashes(worktree)[0],
                           frozen_file_hashes=dict(launch_checks.FROZEN_FILES), audit=guard.report(),
                           pin_policy_reports=list(policy.reports))
            if exit_problems and code == EXIT_OK:
                code = EXIT_STOPPED
                payload["ok"] = False
        except Exception as error:  # noqa: BLE001 - the marker exists; the result file must too (N07 c)
            launch_state.ignore_sigterm()
            payload["post_run_error"] = f"{type(error).__name__}: {error}"
            payload["ok"] = False
            code = code or EXIT_OTHER
        write_json(launch_state.result_path(root, unit, attempt), payload)
        return code
    finally:
        if guard is not None:
            guard.disable()


def build_parser():
    parser = launch_state.add_common_arguments(argparse.ArgumentParser(description="Stage worker; started by launch.py only"))
    parser.add_argument("--unit", required=True, choices=sorted(driver.STAGE_SPECS))
    for name in ("a0-result-sha256", "stage0-sha256", "stage0b-sha256", "stage1-sha256"):
        parser.add_argument(f"--{name}", default=None)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    launch_state.install_sigterm()
    return run(args, production_deps())


if __name__ == "__main__":
    sys.exit(main())
