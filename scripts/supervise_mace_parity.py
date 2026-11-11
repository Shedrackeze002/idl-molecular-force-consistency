"""Bound a single MACE parity worker process and preserve every outcome."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

WALL_SECONDS = 1800
CLEANUP_SECONDS = 10
OUTPUT_BYTES = 256 * 1024 * 1024
LOG_BYTES = 16 * 1024 * 1024


def _tree_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def _group_alive(process: subprocess.Popen) -> bool:
    if os.name == "nt":
        return process.poll() is None
    process.poll()  # reap an exited leader; descendants may still keep the group alive
    try:
        os.killpg(process.pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def _terminate_owned_group(process: subprocess.Popen) -> None:
    """Terminate the entire owned group, even if its leader already exited."""
    if os.name == "nt":
        if process.poll() is None:
            process.send_signal(signal.CTRL_BREAK_EVENT)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        return
    if _group_alive(process):
        os.killpg(process.pid, signal.SIGTERM)
    grace = time.monotonic() + 5
    while _group_alive(process) and time.monotonic() < grace:
        time.sleep(0.05)
    if _group_alive(process):
        os.killpg(process.pid, signal.SIGKILL)
    deadline = time.monotonic() + max(1, CLEANUP_SECONDS - 5)
    while _group_alive(process) and time.monotonic() < deadline:
        time.sleep(0.05)
    if process.poll() is None:
        process.wait(timeout=max(0.1, deadline - time.monotonic()))


def supervise(packet_root: Path, output: Path, *, timeout: float = WALL_SECONDS,
              command: list[str] | None = None,
              launch_receipt: Path | None = None,
              wsl_admission_receipt: Path | None = None,
              interrupted_signal: Callable[[], int | None] | None = None) -> dict:
    if interrupted_signal is None:
        interrupted_signal = lambda: None
    started = time.monotonic()
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    process = None
    stage = "setup"
    worker = command
    receipt = {"status": "RUNNING", "command": command, "timeout_seconds": timeout,
               "output_limit_bytes": OUTPUT_BYTES, "log_limit_bytes": LOG_BYTES,
               "cleanup_seconds": CLEANUP_SECONDS, "started_unix": time.time(),
               "packet_root": str(Path(packet_root).resolve()), "attempt_root": str(output.resolve()),
               "fresh_output": True, "synthetic_command": command is not None}
    status_path = output / "supervisor-status.json"
    status_path.write_text(json.dumps(receipt, indent=2) + "\n")
    if command is None:
        if launch_receipt is None or wsl_admission_receipt is None:
            receipt.update(status="SETUP_FAILED", failure_stage=stage, error_type="ValueError",
                           error="production launch requires reviewed launch and WSL admission receipts")
            status_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            return receipt
    stdout_path, stderr_path = output / "stdout.log", output / "stderr.log"
    reason = None
    try:
        if command is None:
            launch_source = Path(launch_receipt)
            admission_source = Path(wsl_admission_receipt)
            launch = json.loads(launch_source.read_text(encoding="utf-8"))
            admission_name = launch.get("wsl_admission_receipt")
            if not isinstance(admission_name, str) or Path(admission_name).name != admission_name:
                raise ValueError("launch receipt must use a local WSL admission receipt filename")
            from mace_loader import (_launch_approval_refs, _regular_file, _sha256,
                                     _validate_launch_binding, _validate_live_invocation)
            refs = _launch_approval_refs(launch)
            shutil.copyfile(_regular_file(launch_source.parent, Path(launch_source.name)),
                            output / "launch-receipt.json")
            for ref in refs:
                source = admission_source if ref["path"] == admission_name else launch_source.parent / ref["path"]
                shutil.copyfile(_regular_file(source.parent, Path(source.name)), output / ref["path"])
            execution = _validate_launch_binding(launch, output)
            _validate_live_invocation(execution, Path(packet_root), output, worker=False)
            actual_resources = {"timeout_seconds": timeout, "cleanup_seconds": CLEANUP_SECONDS,
                                "output_limit_bytes": OUTPUT_BYTES, "log_limit_bytes": LOG_BYTES}
            if (actual_resources != execution["resources"]
                    or any(type(value) is not int for value in actual_resources.values())):
                raise ValueError("launch resources differ from actual supervisor settings")
            receipt["launch_receipt_sha256"] = _sha256(output / "launch-receipt.json")
        worker = command or [sys.executable, str(Path(__file__).resolve().with_name("check_mace_parity.py")),
                             "--packet-root", str(Path(packet_root).resolve()),
                             "--attempt-root", str(output.resolve())]
        receipt["command"] = worker
        status_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        if interrupted_signal() is not None:
            reason = "INTERRUPTED"
        elif time.monotonic() - started > timeout:
            reason = "TIMEOUT"
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        start_new_session = os.name != "nt"
        stage = "spawn"
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
          if reason is None:
            process = subprocess.Popen(worker, stdout=stdout, stderr=stderr,
                                       creationflags=creationflags, start_new_session=start_new_session)
            stage = "monitor"
            while process.poll() is None:
                elapsed = time.monotonic() - started
                if interrupted_signal() is not None:
                    reason = "INTERRUPTED"
                elif elapsed > timeout:
                    reason = "TIMEOUT"
                elif stdout_path.stat().st_size > LOG_BYTES or stderr_path.stat().st_size > LOG_BYTES:
                    reason = "LOG_OVERFLOW"
                elif _tree_bytes(output) > OUTPUT_BYTES:
                    reason = "OUTPUT_OVERFLOW"
                if reason:
                    break
                time.sleep(0.05)
            if reason is None:
                if interrupted_signal() is not None:
                    reason = "INTERRUPTED"
                elif _group_alive(process):
                    reason = "CHILD_PROCESS_LEAK"
            if reason is not None:
                stage = "cleanup"
                _terminate_owned_group(process)
        stage = "finalize"
        elapsed = time.monotonic() - started
        exit_code = None if process is None else process.returncode
        if reason:
            status = reason
        elif elapsed > timeout:
            status = "TIMEOUT"
        elif stdout_path.stat().st_size > LOG_BYTES or stderr_path.stat().st_size > LOG_BYTES:
            status = "LOG_OVERFLOW"
        elif exit_code != 0:
            status = "WORKER_FAILED"
        elif _tree_bytes(output) > OUTPUT_BYTES:
            status = "OUTPUT_OVERFLOW"
        else:
            status = "WORKER_COMPLETED"
        receipt.update(status=status, exit_code=exit_code, elapsed_seconds=elapsed,
                       stdout_bytes=stdout_path.stat().st_size if stdout_path.exists() else 0,
                       stderr_bytes=stderr_path.stat().st_size if stderr_path.exists() else 0)
        status_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        final_elapsed = time.monotonic() - started
        final_bytes = _tree_bytes(output)
        if receipt["status"] != "INTERRUPTED":
            if final_elapsed > timeout:
                receipt["status"] = "TIMEOUT"
            elif stdout_path.stat().st_size > LOG_BYTES or stderr_path.stat().st_size > LOG_BYTES:
                receipt["status"] = "LOG_OVERFLOW"
            elif final_bytes > OUTPUT_BYTES:
                receipt["status"] = "OUTPUT_OVERFLOW"
        receipt["elapsed_seconds"] = final_elapsed
        receipt["output_bytes"] = final_bytes
        status_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        receipt["output_bytes"] = _tree_bytes(output)
        if receipt["output_bytes"] > OUTPUT_BYTES and receipt["status"] != "INTERRUPTED":
            receipt["status"] = "OUTPUT_OVERFLOW"
            status_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        # The last successful persistence is part of the wall-clock contract.
        persisted_elapsed = time.monotonic() - started
        if persisted_elapsed > timeout and receipt["status"] not in {"TIMEOUT", "INTERRUPTED"}:
            receipt["status"] = "TIMEOUT"
            receipt["elapsed_seconds"] = persisted_elapsed
            status_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        signal_number = interrupted_signal()
        if signal_number is not None:
            receipt["interrupt_signal"] = signal_number
            if receipt["status"] == "WORKER_COMPLETED":
                receipt["status"] = "INTERRUPTED"
            status_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return receipt

    except (Exception, KeyboardInterrupt) as exc:
        cleanup_failed = stage == "cleanup"
        if process is not None and stage not in {"cleanup", "finalize"}:
            try:
                _terminate_owned_group(process)
            except (Exception, KeyboardInterrupt) as cleanup_error:
                cleanup_failed = True
                receipt["cleanup_error"] = str(cleanup_error)
        signal_number = interrupted_signal()
        if isinstance(exc, KeyboardInterrupt) and signal_number is None:
            signal_number = int(signal.SIGINT)
        if signal_number is not None:
            receipt["interrupt_signal"] = signal_number
        if cleanup_failed:
            failure_status = "CLEANUP_FAILED"
        elif signal_number is not None:
            failure_status = "INTERRUPTED"
        else:
            failure_status = {"spawn": "SPAWN_FAILED", "monitor": "CLEANUP_FAILED",
                              "finalize": "FINALIZE_FAILED"}.get(stage, "SETUP_FAILED")
        receipt.update(status=failure_status,
                       failure_stage=stage, error_type=type(exc).__name__, error=str(exc),
                       elapsed_seconds=time.monotonic() - started,
                       exit_code=None if process is None else process.poll())
        status_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--wsl-admission-receipt", type=Path, required=True)
    args = parser.parse_args()
    signal_number = None

    def latch_interrupt(signum, _frame):
        nonlocal signal_number
        if signal_number is None:
            signal_number = signum

    handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        for signum in handlers:
            signal.signal(signum, latch_interrupt)
        result = supervise(args.packet_root, args.output,
                           launch_receipt=args.launch_receipt,
                           wsl_admission_receipt=args.wsl_admission_receipt,
                           interrupted_signal=lambda: signal_number)
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "WORKER_COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
