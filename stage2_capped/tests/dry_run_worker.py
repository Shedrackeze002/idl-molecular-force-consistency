"""Synthetic worker for the driver dry run (tests only; not collected by pytest).

The parent passes its ledger snapshot, taken before it starts the stage; the worker rebuilds the
ledger from it, so the start rules hold inside the worker too. A worker killed hard cannot mark its
stage INCOMPLETE; the parent does (launch_io.mark_incomplete_if_absent).

One process runs one stage on the repository's synthetic landscape with fixture arms. The
parent (test_dry_run.py) holds the stage ledger and supervises the process with
launch_io.supervise, exactly as a launch script will. No model is loaded.

What a real worker has to do, and this one does: install a SIGTERM handler that raises, so
run_stage marks the stage INCOMPLETE and keeps its rows; map a driver stop to a nonzero
exit code and a result file; and on a Stage 0 mismatch run the single pre-authorized frozen
re-run in the same process (the controller's state lives in memory) before exiting.
"""
import argparse
import copy
import importlib.util
import json
import os
import signal
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

STAGE_MODELS = {"0": ["direct-0", "energy-0"], "0b": ["consistency-1-0"],
                "1": ["direct-0", "energy-0"], "2": ["consistency-1-0"]}
STAGE_CAPS = {"0": [None], "0b": [None], "1": [0.05, 0.25], "2": [0.05, 0.25]}


def unwrap(fixture):
    return getattr(fixture, "__wrapped__", None) or fixture._get_wrapped_function()


def load_landscape():
    spec = importlib.util.spec_from_file_location("repo_tests_conftest", REPO / "tests" / "conftest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    land = unwrap(module.synthetic_landscape)()
    return land, copy.deepcopy(land["manifest"]), unwrap(module.synthetic_endpoint_classification)(land)


def rows_written(root, stage):
    """Result rows already written by this stage (earlier stages share the root)."""
    return len(list((Path(root) / "rows" / f"stage{stage}").glob("*/*/*.json")))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--launch-id", required=True)
    parser.add_argument("--stage", required=True, choices=sorted(STAGE_MODELS))
    parser.add_argument("--ledger", required=True, help="the parent's ledger snapshot, as JSON")
    parser.add_argument("--stall-after-rows", type=int, default=None)
    parser.add_argument("--ignore-sigterm", action="store_true")
    parser.add_argument("--storage-limit", type=int, default=None)
    parser.add_argument("--perturb", action="store_true")
    args = parser.parse_args(argv)

    from evaluation.adapter import callable_arm
    import stage2_capped.pass_copy as pass_copy
    from stage2_capped import driver, launch_io
    from stage2_capped.stage_gate import Stage0Controller, StageLedger
    from stage2_capped.tests import fixtures as fx
    from stage2_capped.tests.test_driver import TARGET, frozen_saved, make_inputs, make_tasks

    def terminate(signum, frame):
        raise driver.DeadlineReached("terminated by the supervisor")

    signal.signal(signal.SIGTERM, signal.SIG_IGN if args.ignore_sigterm else terminate)

    ledger = StageLedger.from_snapshot(json.loads(args.ledger))
    land, manifest, classification = load_landscape()
    inputs = make_inputs(land, manifest, classification)
    species = next(iter(inputs.starts.values()))["species"]
    root = Path(args.root)

    def provider(model_id):
        force = fx.harmonic(TARGET, 4.0)

        def stalling(positions):
            if args.stall_after_rows is not None and rows_written(root, args.stage) >= args.stall_after_rows:
                time.sleep(120)
            return force(positions)
        return callable_arm(stalling, species, model_id)

    limit = args.storage_limit or driver.STORAGE_LIMIT_BYTES
    context = driver.Context(root, args.launch_id, inputs, provider, storage_limit=limit)
    tasks = make_tasks(context, args.stage, STAGE_MODELS[args.stage], STAGE_CAPS[args.stage])

    if args.perturb:  # the capped code now differs from the frozen relax by one part in 1e12
        for module in (driver, pass_copy):
            real = module.relax_capped

            def broken(*a, _real=real, **k):
                result = _real(*a, **k)
                result["final_positions"] = result["final_positions"] + 1e-12
                return result
            module.relax_capped = broken

    result = {"stage": args.stage, "ok": False}
    code = 0
    try:
        if args.stage == "0":
            saved = frozen_saved(context, tasks)
            controller = Stage0Controller(expected_runs=len(tasks))
            try:
                summary = driver.run_stage("0", context, ledger=ledger, saved=saved, controller=controller,
                                           tasks=tasks, allow_custom_tasks=True)
            except driver.Stage0Stop as stop:
                result["error"] = str(stop)
                if stop.verdict["verdict"] == "frozen re-run required":
                    task = next(t for t in tasks if driver.task_id(t) == stop.task_id)
                    verdict = driver.run_frozen_rerun(context, task, saved, controller)
                    result["frozen_rerun"] = verdict
                    (root / "rows" / "stage0" / "STAGE0_RERUN.json").write_text(json.dumps(verdict, indent=1))
                raise
        else:
            index = driver.ControlIndex.from_roots([root], allow_custom=True)
            summary = driver.run_stage(args.stage, context, ledger=ledger, tasks=tasks, allow_custom_tasks=True,
                                       control_index=index if args.stage in ("1", "2") else None)
        result.update(ok=True, summary=summary)
    except (driver.DriverError, launch_io.StorageExceeded) as error:   # a labelled stop of the stage
        result.setdefault("error", f"{type(error).__name__}: {error}")
        code = 3
    except Exception as error:  # noqa: BLE001 - a worker reports it and exits nonzero
        result["error"] = f"{type(error).__name__}: {error}"
        code = 4
    (root / f"worker-{args.stage}.result.json").write_text(json.dumps(result, indent=1, default=str))
    print(json.dumps({"stage": args.stage, "ok": result["ok"]}))
    return code


if __name__ == "__main__":
    sys.exit(main())
