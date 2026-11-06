"""Test-only entry point: the real worker `run` functions with fixture dependencies (plan v3 section 10).

The production workers take their dependencies from `production_deps()`; nothing on their command lines or in the
environment selects another set. This module is the test seam: it runs `launch_a0.run` or `launch_stage.run` with
dependencies that restore no checkpoint, so the end-to-end tests exercise the real worker code, the real parent and
the real files. Scenarios arrive in the environment variable LAUNCH_TEST_SCENARIO (a JSON object); the fixture
folder in LAUNCH_TEST_DIR. Neither is read by any module outside `tests/`.
"""
import argparse
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from evaluation.adapter import callable_arm
from stage2_capped import (admission, driver, launch_a0, launch_checks, launch_io, launch_stage, launch_state,
                           replay_evidence)

SCENARIO = json.loads(os.environ.get("LAUNCH_TEST_SCENARIO", "{}"))
SPECIES3 = ("H", "H", "H")
GROUPS = ("13", "18", "18_2", "66", "78")
PARTITION_SIZE = admission.REPLAY_RECORDS


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def fixture_report(group9, forbidden=False):
    """What the real admit_all would return for fixture identities. It reads the four pinned data files, as the
    real one does, so the audit log shows what a real launch opens."""
    directory = admission.data_directory(group9)
    for name in admission.DATASET_FILES:
        (directory / name).read_bytes()
    if forbidden:               # a file under the data tree that is not one of the allowed five (a README)
        (Path(group9) / "data" / "landscape17_sample" / "README.md").read_bytes()
    identities = {m: {"id": m, "best_update": 100, "best_msgpack_sha256": sha(f"ckpt-{m}"),
                      "score_json_sha256": sha(f"score-{m}")} for m in admission.ALL_IDS}
    return {"launch_records_sha256": hashlib.sha256((Path(group9) / admission.RECOVERY_RECORDS).read_bytes()).hexdigest(),
            "identities": identities, "dataset_pin": {"manifest_identity": "fixture-dataset"},
            "accepted_pin": {"code_fingerprint": "fixture-fingerprint"}}


def parent_admit_all(group9):
    return fixture_report(group9, forbidden=SCENARIO.get("open_forbidden_parent", False))


def identity_of(model_id):
    arm, seed, lam = driver.MODEL_IDENTITY[model_id]
    return {"checkpoint_sha256": sha(f"ckpt-{model_id}"), "score_sha256": sha(f"score-{model_id}"), "best_update": 100,
            "arm": arm, "seed": seed, "lambda": lam, "dataset_identity": "fixture-dataset",
            "source_fingerprint": "fixture-fingerprint"}


def precondition_gate(attempt):
    """Scenario: fail a model-free precondition in the worker on the listed attempts."""
    if attempt in SCENARIO.get("fail_precondition_attempts", []):
        raise RuntimeError("scenario: a precondition failed")


def worker_admit_all(group9):
    precondition_gate(int(os.environ.get("LAUNCH_TEST_ATTEMPT", "1")))
    report = fixture_report(group9, forbidden=SCENARIO.get("open_forbidden", False))
    if SCENARIO.get("drift_stage_identity") and os.environ.get("LAUNCH_TEST_KIND") == "stage":
        report["identities"]["energy-0"]["best_msgpack_sha256"] = sha("a checkpoint that changed after A0")
    return report


_ONCE = {}


def after_the_spent_point():
    """Scenarios that act once, from inside the first force call (after the spent point in A0 and in a stage)."""
    if _ONCE.get("done"):
        return
    _ONCE["done"] = True
    if SCENARIO.get("dirty_worktree"):                  # a file appears in the worktree during the run
        (driver.WORKTREE / "stray_after_the_spent_point.txt").write_text("x\n")
    if SCENARIO.get("module_outside_at_exit"):          # a project module now claims to come from outside the worktree
        elsewhere = Path(os.environ["LAUNCH_TEST_DIR"]) / "row_schema_elsewhere.py"
        elsewhere.write_text("x = 1\n")
        sys.modules["stage2_capped.row_schema"].__file__ = str(elsewhere)


# ---- A0 fixtures ---------------------------------------------------------------------------------------------

def make_partition():
    rng = np.random.default_rng(11)
    positions = rng.normal(size=(PARTITION_SIZE, 3, 3))
    ids = [f"rec-{i}" for i in range(PARTITION_SIZE)]
    memberships = {rid: {"groups": [GROUPS[i] if i < 5 else "other"]} for i, rid in enumerate(ids)}
    return SimpleNamespace(record_ids=ids, positions=positions, forces=-3.0 * positions, memberships=memberships,
                           atomic_numbers=np.ones((PARTITION_SIZE, 3), dtype=int))


def stiffness(model_id):
    return 2.0 + admission.ALL_IDS.index(model_id)


def arm_for(model_id, species, force_scale=1.0):
    def force(array):
        if SCENARIO.get("spy_calls_file"):          # one line per force call: which model, what shape
            with open(SCENARIO["spy_calls_file"], "a") as handle:
                handle.write(json.dumps({"model": model_id, "shape": list(np.shape(array))}) + "\n")
        after_the_spent_point()
        if SCENARIO.get("slow_force_seconds"):
            time.sleep(SCENARIO["slow_force_seconds"])
        out = -stiffness(model_id) * np.asarray(array, dtype=np.float64) * force_scale
        offset = SCENARIO.get("perturb", {}).get(model_id)
        if offset:
            out = out + offset
        if model_id in SCENARIO.get("raise", []):
            raise RuntimeError("scenario: the model refused")
        return out
    arm = callable_arm(force, species, model_id)
    arm.update(checkpoint_identity=identity_of(model_id))
    return arm


def a0_deps(fixture_dir):
    import force_consistency.data.prepare as prepare
    partition = make_partition()
    prepare.load_training_bundle = lambda directory, expected_split_hash=None: SimpleNamespace(validation=partition)
    items = []
    for model_id in admission.ALL_IDS:
        score = Path(fixture_dir) / f"score-{model_id}"
        score.mkdir(parents=True, exist_ok=True)
        if not (score / "best-validation.npz").exists():
            np.savez(score / "best-validation.npz", record_ids=np.array(partition.record_ids),
                     reference_forces=partition.forces, forces=-stiffness(model_id) * partition.positions)
        items.append({"id": model_id, "score_directory": str(score), "run_directory": "unused", "launch_record": {},
                      "acceptance": {}})
    from run_validation_recovery import GROUPS as FROZEN_GROUPS, adapter_replay

    def provider_factory(bindings, data_directory, policy):
        def provider(model_id):
            with open(Path(os.environ["LAUNCH_TEST_DIR"]) / "provider-order.txt", "a") as handle:
                handle.write(model_id + "\n")
            if SCENARIO.get("restore_fail") == model_id:
                raise ValueError("scenario: restore refused")
            if SCENARIO.get("sleep_in_provider"):
                time.sleep(SCENARIO["sleep_in_provider"])
            return arm_for(model_id, SPECIES3)
        return provider

    def saved(score_directory):
        if SCENARIO.get("sleep_in_saved_forces"):       # the evidence step: signal that it was reached, then wait
            (Path(os.environ["LAUNCH_TEST_DIR"]) / "saved-forces-reached.txt").write_text(Path(score_directory).name)
            time.sleep(SCENARIO["sleep_in_saved_forces"])
        if SCENARIO.get("saved_forces_raise") or Path(score_directory).name in [
                f"score-{m}" for m in SCENARIO.get("saved_forces_raise_for", [])]:
            raise KeyError("scenario: the saved forces cannot be read")
        with np.load(Path(score_directory) / "best-validation.npz") as archive:
            return archive["forces"].copy()
    def bindings(group9):
        if not SCENARIO.get("items_raise"):
            return items

        def broken():                                   # the loop's own iterator fails after the first model
            yield items[0]
            raise RuntimeError("scenario: the bindings iterator failed")
        return broken()
    replay = adapter_replay
    if SCENARIO.get("spy_calls_file"):
        def note(**fields):
            with open(SCENARIO["spy_calls_file"], "a") as handle:
                handle.write(json.dumps(fields) + "\n")

        def replay(arm, score_directory, data_directory, split_hash):           # the frozen function, called as given
            note(replay={"score_directory": str(score_directory), "data_directory": str(data_directory), "split_hash": split_hash})
            return adapter_replay(arm, score_directory, data_directory, split_hash)
        real_collect = replay_evidence.collect

        def collect(*arguments, **keywords):
            note(evidence_begins=True)
            return real_collect(*arguments, **keywords)
        replay_evidence.collect = collect
    return launch_a0.A0Deps(
        admit_all=worker_admit_all, load_bindings=bindings, data_directory=admission.data_directory, provider_factory=provider_factory,
        replay=replay, load_partition=lambda directory: partition, saved_forces=saved, groups=tuple(FROZEN_GROUPS))


# ---- Stage fixtures --------------------------------------------------------------------------------------------

def stage_deps():
    state = {}                      # the study inputs, loaded with the worker's own group9 root when asked for

    def zero_force(p):
        after_the_spent_point()
        if SCENARIO.get("kill_in_force"):
            os.kill(os.getpid(), signal.SIGKILL)
        if SCENARIO.get("sleep_in_stage_force") and Path(SCENARIO["stage_directory"]).exists():
            time.sleep(SCENARIO["sleep_in_stage_force"])             # only in the run itself, not while the saved results are built
        return np.zeros_like(p)

    def zero_arm(model_id):
        arm = callable_arm(zero_force, state["species"], model_id)
        arm.update(checkpoint_identity=identity_of(model_id))
        return arm

    def mismatched_force(p):
        if SCENARIO.get("kill_in_rerun") and Path(SCENARIO["rerun_marker"]).exists():
            os.kill(os.getpid(), signal.SIGKILL)          # after STAGE0_RERUN_ATTEMPTED.json, before the verdict is written
        return -0.5 * np.asarray(p, dtype=np.float64)

    def run_arm(model_id):
        if SCENARIO.get("mismatch"):
            arm = callable_arm(mismatched_force, state["species"], model_id)
            arm.update(checkpoint_identity=identity_of(model_id))
            return arm
        return zero_arm(model_id)

    def saved_loader(expected_hashes, group9):
        import tempfile
        from stage2_capped.tests.test_driver import frozen_saved
        folder = Path(tempfile.mkdtemp(prefix="saved-root-"))
        inputs = state["inputs"]
        context = driver.Context(launch_io.fresh_root(folder / "root", "saved"), "saved", inputs, zero_arm)
        tasks = driver.build_schedule("0", sorted(inputs.starts), inputs.reference_ids)
        return frozen_saved(context, tasks)
    def inputs_loader(expected, group9):
        state["inputs"] = driver.load_study_inputs(expected, group9=group9)
        state["species"] = next(iter(state["inputs"].starts.values()))["species"]
        return state["inputs"]
    if SCENARIO.get("unexpected_in_run"):
        def explode(*args, **kwargs):
            raise KeyError("scenario: an unexpected failure inside a run")
        driver.execute_task = explode
    return launch_stage.StageDeps(
        admit_all=worker_admit_all, load_bindings=lambda group9: [],
        data_directory=admission.data_directory, provider_factory=lambda items, directory, policy: run_arm,
        inputs_loader=inputs_loader, saved_loader=saved_loader)


def main(argv=None):
    parser = launch_state.add_common_arguments(argparse.ArgumentParser())
    parser.add_argument("--kind", required=True, choices=["a0", "stage"])
    parser.add_argument("--unit", default="A0")
    for name in ("a0-result-sha256", "stage0-sha256", "stage0b-sha256", "stage1-sha256"):
        parser.add_argument(f"--{name}", default=None)
    args = parser.parse_args(argv)
    os.environ["LAUNCH_TEST_ATTEMPT"] = str(args.attempt)
    os.environ["LAUNCH_TEST_KIND"] = args.kind
    if SCENARIO.get("gates_raise_on_call"):             # the diff gates fail on their Nth call in this process
        real_gates, calls = launch_checks.diff_gate_hashes, []

        def gates(worktree):
            calls.append(1)
            if len(calls) == SCENARIO["gates_raise_on_call"]:
                raise OSError("scenario: the diff gates could not be computed")
            return real_gates(worktree)
        launch_checks.diff_gate_hashes = gates
    if SCENARIO.get("raise_at_exit"):                   # the exit-time module inventory fails (the second call)
        real_inventory, inventory_calls = launch_checks.module_inventory, []

        def inventory(worktree, modules=None):
            inventory_calls.append(1)
            if len(inventory_calls) == 2:
                raise OSError("scenario: the module inventory could not be computed at exit")
            return real_inventory(worktree, modules)
        launch_checks.module_inventory = inventory
    if SCENARIO.get("module_outside_at_start"):         # a project module claims to come from outside the worktree from the first moment
        import stage2_capped.row_schema                  # noqa: F401 - make sure it is imported
        elsewhere = Path(os.environ["LAUNCH_TEST_DIR"]) / "row_schema_elsewhere_at_start.py"
        elsewhere.write_text("x = 1\n")
        sys.modules["stage2_capped.row_schema"].__file__ = str(elsewhere)
    if SCENARIO.get("sigterm_at_exit"):                 # a late SIGTERM arrives while the exit-time inventory runs
        late_inventory, late_calls = launch_checks.module_inventory, []

        def inventory_with_a_late_sigterm(worktree, modules=None):
            late_calls.append(1)
            if len(late_calls) == 2:
                os.kill(os.getpid(), signal.SIGTERM)
                time.sleep(0.3)
            return late_inventory(worktree, modules)
        launch_checks.module_inventory = inventory_with_a_late_sigterm
    if SCENARIO.get("ignore_sigterm"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    else:
        launch_state.install_sigterm()
    if args.kind == "a0":
        return launch_a0.run(args, a0_deps(os.environ["LAUNCH_TEST_DIR"]))
    return launch_stage.run(args, stage_deps())


if __name__ == "__main__":
    sys.exit(main())
