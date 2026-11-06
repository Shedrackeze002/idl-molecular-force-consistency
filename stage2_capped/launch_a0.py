"""The A0 worker (plan v3 section 5): seven admissions and the replay, no relaxation.

A0 is the first model step of the study. One process runs `run(args, deps)`; `main()` builds the production
dependencies and nothing on the command line or in the environment selects another set.

Order: the model-free preconditions (they spend nothing: a failure is a refusal, written under an attempt-numbered
name, exit 2); then the spent point, `A0_RESTORE_BEGUN.json`, written once with `atomic_write_bytes` immediately
before the first provider call; then for each of the seven models, in `admission.ALL_IDS` order, a restore, the
frozen `adapter_replay` (unchanged: it is the gate) and the measurement-only evidence. A failure of one model is
labelled (`admission`, `pin`, `restore`, `replay`, `deadline`, `other`) and the rest are still replayed. The verdict
is recomputed from the stored values by `launch_state.a0_verdict`; any failed model fails A0. Exit 0: written and
pass. Exit 2: refused before the spent point. Exit 3: written, verdict fail (or the deadline). Exit 4: another
exception after the spent point (a result is still written).
"""
import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from stage2_capped import admission, driver, launch_checks, launch_io, launch_record, launch_state, replay_evidence
from stage2_capped.admission import PinPolicy

EXIT_PASS, EXIT_REFUSED, EXIT_FAILED, EXIT_OTHER = 0, 2, 3, 4


@dataclass
class A0Deps:
    """Data sources and the frozen replay, never enforcement switches."""
    admit_all: Callable
    load_bindings: Callable
    data_directory: Callable
    provider_factory: Callable
    replay: Callable
    load_partition: Callable
    saved_forces: Callable
    groups: tuple


class Refused(RuntimeError):
    """A precondition failed before the spent point."""


def production_deps():
    from stage2_capped import launch_arms

    def frozen_replay(arm, score_directory, data_directory, split_hash):
        from run_validation_recovery import adapter_replay
        return adapter_replay(arm, score_directory, data_directory, split_hash)

    def partition(data_directory):
        from e3x_phase_common import SPLIT
        from force_consistency.data.prepare import load_training_bundle
        return load_training_bundle(data_directory, expected_split_hash=SPLIT).validation

    def saved_forces(score_directory):
        with np.load(Path(score_directory) / "best-validation.npz", allow_pickle=False) as archive:
            return archive["forces"].copy()

    from run_validation_recovery import GROUPS
    return A0Deps(admit_all=admission.admit_all, load_bindings=admission.load_bindings,
                  data_directory=admission.data_directory, provider_factory=launch_arms.real_arm_provider,
                  replay=frozen_replay, load_partition=partition, saved_forces=saved_forces, groups=tuple(GROUPS))


def classify(phase, error):
    if isinstance(error, driver.DeadlineReached):
        return "deadline"
    if isinstance(error, admission.PinMismatch):
        return "pin"
    if isinstance(error, admission.AdmissionError):
        return "admission"
    return phase if phase in ("restore", "replay") else "other"


def write_result(path, payload):
    return launch_io.atomic_write_bytes(path, launch_state.dumps(payload))


def run(args, deps, *, environ=None, worktree=None):
    """Run A0 once. Returns the exit code. Every outcome that matters is a file; the code is a summary."""
    worktree = Path(worktree or driver.WORKTREE)
    record_path = Path(args.record).resolve()
    record = launch_record.load_record(args.record, args.record_sha256)
    root, attempt, launch_folder = Path(record["output_root"]), args.attempt, launch_state.launch_folder_of(record_path)
    guard = None
    try:
        try:
            launch_state.write_worker_pid(root, "A0", attempt)                 # the first act (B4)
            started = launch_state.started_event(root, "A0", attempt)
            if started is None or started.get("parent_pid") != os.getppid():
                raise Refused("no `started` event for this attempt names this process's parent")
            try:
                launch_record.verify_launch_context(args.record, args.approval, args.record_sha256, command="a0",
                                                    worktree=worktree, group9=args.group9_root)
            except launch_record.LaunchRefused as error:
                raise Refused(str(error)) from error
            problems = launch_checks.thread_variable_problems(environ)
            if problems:
                raise Refused("; ".join(problems))
            modules_at_start, module_problems = launch_checks.module_inventory(worktree)
            if module_problems:
                raise Refused("; ".join(module_problems))
            gates, gate_problems = launch_checks.diff_gate_hashes(worktree)       # before the spent point (N07 a)
            if gate_problems:
                raise Refused("; ".join(gate_problems))
            group9 = admission.group9_root(args.group9_root)
            tree, allowed = launch_checks.audit_scope(group9)
            guard = launch_checks.AuditGuard(
                tree, allowed, launch_folder / "violations",
                violation_name=f"AUDIT_VIOLATION-{args.record_sha256}-A0-{attempt}").enable()
            report = deps.admit_all(group9)
            launch_state.dumps(report)                                         # a report that cannot be written is a refusal now
            items = deps.load_bindings(group9)
            data_directory = deps.data_directory(group9)
            policy = PinPolicy()
            provider = deps.provider_factory(items, data_directory, policy)
            partition = deps.load_partition(data_directory)
            scope = {"first_five_record_ids": [str(r) for r in list(partition.record_ids)[:5]],
                     "groups": list(deps.groups)}
        except Refused as error:
            write_result(launch_state.result_path(root, "A0", attempt),
                         {"unit": "A0", "attempt": attempt, "label": "refused_before_start", "reason": str(error)})
            return EXIT_REFUSED
        except Exception as error:  # noqa: BLE001 - any precondition failure is a refusal that spends nothing
            write_result(launch_state.result_path(root, "A0", attempt),
                         {"unit": "A0", "attempt": attempt, "label": "refused_before_start",
                          "reason": f"{type(error).__name__}: {error}"})
            return EXIT_REFUSED

        # The spent point: written once, immediately before the first provider call.
        launch_io.atomic_write_bytes(root / "A0_RESTORE_BEGUN.json", launch_state.dumps(
            {"unit": "A0", "attempt": attempt, "record_sha256": args.record_sha256, "pid": os.getpid()}))
        models, code, fatal = {}, EXIT_PASS, None
        try:
            for item in items:
                model_id, entry, phase, arm = item["id"], {}, "restore", None
                models[model_id] = entry
                try:
                    arm = provider(model_id)
                    json.dumps(arm["checkpoint_identity"], sort_keys=True, allow_nan=False)
                    phase = "replay"
                    entry["replay"] = deps.replay(arm, item["score_directory"], data_directory, _split())
                except driver.DeadlineReached:
                    entry["failure"] = "deadline: terminated by the supervisor"
                    raise
                except Exception as error:  # noqa: BLE001 - one model's failure never stops the others
                    entry["failure"] = f"{classify(phase, error)}: {type(error).__name__}: {error}"
                if arm is not None:
                    try:                  # evidence is measurement only: its failure never stops the other models (N07 b)
                        saved = deps.saved_forces(item["score_directory"])
                        entry["evidence"] = replay_evidence.collect(arm, partition.positions, saved)
                        entry["saved_force_abs_max"] = float(np.max(np.abs(saved)))
                    except driver.DeadlineReached:
                        raise
                    except Exception as error:  # noqa: BLE001 - recorded, never raised
                        entry.setdefault("evidence", {"error": f"{type(error).__name__}: {error}"})
                arm = None
        except driver.DeadlineReached as error:
            fatal = f"deadline: {error}"
        except Exception as error:  # noqa: BLE001 - a result is still written
            fatal, code = f"other: {type(error).__name__}: {error}", EXIT_OTHER
        launch_state.ignore_sigterm()                       # from here only the result matters
        try:
            modules_at_exit, exit_problems = launch_checks.module_inventory(worktree)
            result = {"unit": "A0", "attempt": attempt, "record_sha256": args.record_sha256,
                      "launch_id": record["launch_id"], "output_root": record["output_root"], "commit": record["commit"],
                      "admission": report, "models": models, "replay_scope": scope,
                      "runtime": launch_checks.runtime_values(environ, include_backend=True),
                      "diff_gate_hashes": gates, "frozen_file_hashes": dict(launch_checks.FROZEN_FILES),
                      "modules_at_start": modules_at_start, "modules_at_exit": modules_at_exit,
                      "module_problems_at_exit": exit_problems, "pin_policy_reports": list(policy.reports),
                      "audit": guard.report(), "fatal": fatal}
            passed, reasons = launch_state.a0_verdict(result)
            passed = passed and fatal is None and not exit_problems
            result["verdict"] = {"passed": passed, "reasons": reasons + ([fatal] if fatal else []) + exit_problems}
        except Exception as error:  # noqa: BLE001 - a failing A0 still leaves its result file (N07)
            launch_state.ignore_sigterm()
            fatal = fatal or f"finalization: {type(error).__name__}: {error}"
            result = {"unit": "A0", "attempt": attempt, "record_sha256": args.record_sha256, "models": models,
                      "fatal": fatal, "verdict": {"passed": False, "reasons": [fatal]}}
            passed, code = False, code or EXIT_OTHER
        write_result(launch_state.result_path(root, "A0", attempt), result)
        return code if code else (EXIT_PASS if passed else EXIT_FAILED)
    finally:
        if guard is not None:
            guard.disable()


def _split():
    from e3x_phase_common import SPLIT
    return SPLIT


def main(argv=None):
    parser = launch_state.add_common_arguments(argparse.ArgumentParser(description="A0 worker; started by launch.py only"))
    args = parser.parse_args(argv)
    launch_state.install_sigterm()
    return run(args, production_deps())


if __name__ == "__main__":
    sys.exit(main())
