"""Run the historical uncapped recovery controls with the admitted MACE model."""

import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import os
import re
import shutil
import signal
import subprocess
import sys
import time

from mace_loader import MODEL_BYTES, MODEL_SHA256, _sha256


ROOT = Path(__file__).resolve().parents[1]
CONNECTIONS = ("13", "18", "18_2", "66", "78")
BRANCHES = {f"aspirin/{group}/{side}" for group in CONNECTIONS for side in ("minus", "plus")}
REFERENCE_IDS = tuple(f"Landscape17/aspirin/Conformations/coordsmin_{i}.xyz"
                      for i in (1, 3, 6, 7, 8, 9))
RESOURCES = {"wall_seconds": 3600, "output_bytes": 256 * 1024**2,
             "log_bytes": 16 * 1024**2, "force_requests": 24000}


def read_json(path):
    return json.loads(Path(path).read_bytes())


def persist(path, value):
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_suffix(".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def summarize(branches, references, geometry):
    if set(branches) != BRANCHES or set(references) != set(REFERENCE_IDS):
        raise ValueError("Results do not match the fixed control schedule")
    counts = [row["force_evaluation_count"]
              for row in [*branches.values(), *references.values()]]
    if any(type(count) is not int or not 1 <= count <= 1500 for count in counts):
        raise ValueError("Control force count exceeds the historical limit")
    categories = Counter(row["category"] for row in branches.values())
    recovered = categories["matched_reference"]
    unchanged = (geometry["own_geometry_count"]
                 if geometry["own_geometry_status"] == "verified" else None)
    return {
        "original_recovered": recovered,
        "original_start_count": len(branches),
        "original_outcome_counts": dict(categories),
        "recovered_by_connection": {
            g: sum(branches[f"aspirin/{g}/{side}"]["category"] == "matched_reference"
                   for side in ("minus", "plus")) for g in CONNECTIONS},
        "reference_own_matches": sum(r["operational_force_converged_own_match"]
                                     for r in references.values()),
        "reference_start_count": len(references),
        "no_relaxation_matches": unchanged,
        "exceeds_no_relaxation": None if unchanged is None else recovered > unchanged,
        "force_requests": sum(counts),
    }


def verify_record(path, digest):
    if _sha256(path) != digest:
        raise ValueError("Recovery record digest differs")
    record = read_json(path)
    if record["resources"] != RESOURCES:
        raise ValueError("Recovery bounds differ")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if commit != record["commit"]:
        raise ValueError("Recovery source revision differs")
    if (record["python"] != sys.executable
            or record["python_sha256"] != _sha256(Path(sys.executable))):
        raise ValueError("Recovery interpreter differs")
    for name, sha in record["source_sha256"].items():
        if _sha256(ROOT / name) != sha:
            raise ValueError(f"Recovery source differs: {name}")
    for name, item in record["inputs"].items():
        if _sha256(Path(item["path"])) != item["sha256"]:
            raise ValueError(f"Recovery input differs: {name}")
    return record


def preflight(record):
    from audit_mace_parity import audit_saved_parity
    from mace_loader import _authenticated_bridge, _live_admission_check

    packet = Path(record["packet"])
    parity = Path(record["parity_attempt"])
    admission = read_json(parity / "admission-receipt.json")
    runtime = admission["wsl_admission"]
    live = _live_admission_check(runtime)
    distributions = list(importlib.metadata.distributions())
    versions = {re.sub(r"[-_.]+", "-", d.metadata["Name"].lower()): d.version
                for d in distributions}
    if len(versions) != len(distributions) or versions != live["packages"]:
        raise ValueError("Complete live distribution inventory differs from parity")
    launch = admission["launch_receipt"]
    if (launch["python_executable"] != record["python"]
            or launch["python_sha256"] != record["python_sha256"]
            or live.get("framework_argv0") != record["framework_argv0"]):
        raise ValueError("Recovery runtime differs from the admitted parity runtime")
    saved = audit_saved_parity(packet, parity, persist=False)
    if (saved != read_json(parity / "independent-audit.json")
            or saved["parity"]["status"] != "PARITY_PASS"
            or saved["score_status"] != "SCORE_REPRODUCED"):
        raise ValueError("Retained parity audit could not be reproduced")
    bridge, _ = _authenticated_bridge()

    from evaluation.validation_controls import validation_inputs

    inputs = {name: Path(item["path"]) for name, item in record["inputs"].items()}
    manifest = read_json(inputs["manifest"])
    classification = read_json(inputs["classification"])
    config = read_json(inputs["config"])
    historical = read_json(inputs["historical"])
    if config["relaxation"] != historical["evaluation"]["relaxation"]:
        raise ValueError("Historical solver settings differ")
    starts, references, policy, binding = validation_inputs(inputs["archive"], manifest, classification)
    own_ids = {classification[key]["reference_class"] for key in starts}
    if (set(starts) != BRANCHES or own_ids != set(REFERENCE_IDS)
            or binding != historical["start_binding"]
            or any(entry["frame_index"] != 0 for entry in starts.values())
            or any(list(entry["species"]) != launch["species"] for entry in starts.values())):
        raise ValueError("Historical validation start or reference identity differs")
    prepared = read_json(inputs["prepared"])
    train_ids = set(prepared["partitions"]["train"]["record_ids"])
    seen = {key: classification[key]["reference_class"] in train_ids for key in starts}
    return bridge, starts, references, policy, binding, manifest, classification, config["relaxation"], seen


def worker(record, path, digest):
    command = [*record["command"], "--sha256", digest]
    output = Path(record["output"])
    started = read_json(output / "started.json")
    if (sys.argv != command[1:]
            or sys.orig_argv != [record["framework_argv0"]["path"], *command[1:]]
            or started != {"record_sha256": digest, "resources": RESOURCES,
                           "supervisor_pid": os.getppid()}):
        raise ValueError("Unexpected worker invocation or interpreter options")
    bridge, starts, refs, policy, binding, manifest, classification, procedure, seen = preflight(record)

    from evaluation.relax import relax
    from evaluation.runner import _arm_force_supplier, evaluate_starts
    from evaluation.validation_controls import geometry_controls
    from mace_loader import _construct_calculator
    from run_validation_recovery import safe_json
    from trace_validation_recovery import score_reference_control

    geometry = geometry_controls(starts, refs, classification, policy)
    persist(output / "no-relaxation.json", safe_json(geometry))
    persist(output / "input-binding.json", {"validation": binding, "seen_minimum": seen})
    source = Path(record["parity_attempt"]) / "private" / "final_epoch.model"
    snapshot = output / "final_epoch.model"
    with source.open("rb") as incoming, snapshot.open("xb") as outgoing:
        shutil.copyfileobj(incoming, outgoing)
    if snapshot.stat().st_size != MODEL_BYTES or _sha256(snapshot) != MODEL_SHA256:
        raise ValueError("Recovery model copy differs from the admitted checkpoint")
    species = next(iter(starts.values()))["species"]
    identity = {"model_sha256": MODEL_SHA256, "model_bytes": MODEL_BYTES, "epoch_index": 2047}
    arm = bridge(_construct_calculator(snapshot), species, arm_id="mace-runA-final-ema",
                 checkpoint_identity=identity)
    original_force = arm["forces"]
    requests = 0

    def force(*args, **kwargs):
        nonlocal requests
        if requests >= RESOURCES["force_requests"]:
            raise RuntimeError("Recovery force request limit reached")
        requests += 1
        return original_force(*args, **kwargs)

    arm["forces"] = force
    branches, references = {}, {}
    for index, (key, start) in enumerate(sorted(starts.items())):
        started = time.monotonic()
        result = evaluate_starts({key: start}, arm, procedure, classification, refs, policy,
                                 split_manifest=manifest, seen_minimum=seen,
                                 authorise_test_partition=False)["branches"][key]
        result["elapsed_seconds"] = time.monotonic() - started
        result["_audit_record"]["arm"]["checkpoint_identity"] = identity
        branches[key] = result
        persist(output / f"original-{index:02d}.json", safe_json(result))
    for index, key in enumerate(REFERENCE_IDS):
        started = time.monotonic()
        ref = refs[key]
        supplier = _arm_force_supplier(arm, ref.species, key + "/reference-control")
        result = relax(ref.positions, supplier, procedure, arm=arm)
        score = score_reference_control(result, ref, policy)
        score.update(reference_id=key, force_evaluation_count=result["force_evaluation_count"],
                     step_count=result["step_count"], elapsed_seconds=time.monotonic() - started)
        references[key] = score
        persist(output / f"reference-{index:02d}.json",
                safe_json({"score": score, "relaxation": result}))
    summary = summarize(branches, references, geometry)
    if summary["force_requests"] != requests or _sha256(snapshot) != MODEL_SHA256:
        raise ValueError("Recovery request count or checkpoint identity differs")
    verify_record(path, digest)
    persist(output / "result.json", {**summary, "model": identity, "record_sha256": digest})


def output_size(directory):
    total = 0
    for path in directory.iterdir():
        try:
            total += path.stat().st_size
        except FileNotFoundError:
            # Atomic per-row persistence may replace a temporary file here.
            continue
    return total


def supervise(record, digest, approval):
    signals = []

    def interrupt(signum, _frame):
        if not signals:
            signals.append(signum)

    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for sig in handlers:
            signal.signal(sig, interrupt)
        return _supervise(record, digest, approval, signals)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


def _supervise(record, digest, approval, signals):
    from supervise_mace_parity import _group_alive, _terminate_owned_group

    if approval != {"status": "APPROVED", "record_sha256": digest,
                    "commit": record["commit"], "reviews": ["engineering", "scientific"]}:
        raise ValueError("Reviewed recovery launch approval differs")
    output = Path(record["output"])
    started = time.monotonic()
    output.mkdir(exist_ok=False)
    persist(output / "started.json", {"record_sha256": digest, "resources": RESOURCES,
                                      "supervisor_pid": os.getpid()})
    reason = "worker setup failure"
    code = None
    try:
        if signals:
            raise InterruptedError("Recovery interrupted before spawn")
        with (output / "worker.log").open("xb") as log:
            process = subprocess.Popen([*record["command"], "--sha256", digest], cwd=ROOT, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while process.poll() is None:
                    if signals:
                        raise InterruptedError("Recovery interrupted")
                    if time.monotonic() - started >= RESOURCES["wall_seconds"]:
                        raise TimeoutError("Recovery wall limit reached")
                    if ((output / "worker.log").stat().st_size > RESOURCES["log_bytes"]
                            or output_size(output) > RESOURCES["output_bytes"]):
                        raise RuntimeError("Recovery output limit reached")
                    time.sleep(0.25)
                code = process.returncode
                if _group_alive(process):
                    raise RuntimeError("Recovery worker left a running process group")
            finally:
                if _group_alive(process):
                    _terminate_owned_group(process)
        if (signals or code != 0 or not (output / "result.json").is_file()
                or time.monotonic() - started >= RESOURCES["wall_seconds"]
                or (output / "worker.log").stat().st_size > RESOURCES["log_bytes"]
                or output_size(output) > RESOURCES["output_bytes"]):
            raise RuntimeError("Recovery worker or final bounds check failed")
        reason = None
    except BaseException as exc:
        reason = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        persist(output / "terminal.json", {"complete": reason is None, "worker_exit_code": code,
                                          "elapsed_seconds": time.monotonic() - started,
                                          "failure": reason,
                                          "interrupt_signal": signals[0] if signals else None})
    terminal_sha = _sha256(output / "terminal.json")
    size = output_size(output)
    elapsed = time.monotonic() - started
    if signals or elapsed > RESOURCES["wall_seconds"] or size > RESOURCES["output_bytes"]:
        refusal = {"elapsed_seconds": elapsed, "output_bytes": size,
                   "interrupt_signal": signals[0] if signals else None}
        persist(output / "publication-refused.json", refusal)
        raise RuntimeError("Recovery terminal publication exceeded bounds or was interrupted")
    return {"terminal_sha256": terminal_sha, "publication_elapsed_seconds": elapsed,
            "output_bytes": size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preflight", "supervise", "worker"])
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--approval", type=Path)
    args = parser.parse_args()
    record = verify_record(args.record, args.sha256)
    if args.mode == "preflight":
        preflight(record)
        print("Historical input and saved MACE parity checks passed.")
    elif args.mode == "worker":
        worker(record, args.record, args.sha256)
    else:
        observation = supervise(record, args.sha256, read_json(args.approval))
        print(json.dumps(observation, sort_keys=True))


if __name__ == "__main__":
    main()
