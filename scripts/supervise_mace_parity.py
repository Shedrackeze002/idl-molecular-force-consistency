"""Bound a single MACE parity worker process and preserve every outcome."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

WALL_SECONDS = 1800
CLEANUP_SECONDS = 10
OUTPUT_BYTES = 256 * 1024 * 1024
LOG_BYTES = 16 * 1024 * 1024


def _tree_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        try:
            entry = path.lstat()
        except FileNotFoundError:
            # Worker temporary files can disappear during atomic replacement.
            continue
        if stat.S_ISREG(entry.st_mode):
            total += entry.st_size
    return total


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


def _receipt_text(receipt: dict) -> str:
    return json.dumps(receipt, indent=2, sort_keys=True) + "\n"


def _write_terminal(path: Path, receipt: dict) -> None:
    staged = path.with_suffix(".json.tmp")
    staged.write_text(_receipt_text(receipt), encoding="utf-8", newline="\n")
    staged.replace(path)


def _finalize(output: Path, status_path: Path, receipt: dict, started: float,
              reason: str | None, exit_code: int | None,
              observe_interrupt: Callable[[bool], int | None]) -> dict:
    elapsed = time.monotonic() - started
    stdout_bytes = (output / "stdout.log").stat().st_size
    stderr_bytes = (output / "stderr.log").stat().st_size
    tree_bytes = _tree_bytes(output)
    receipt.update(exit_code=exit_code, elapsed_seconds=elapsed, output_bytes=tree_bytes,
                   stdout_bytes=stdout_bytes, stderr_bytes=stderr_bytes)
    if reason is not None:
        receipt["status"] = reason
    elif elapsed > receipt["timeout_seconds"]:
        receipt["status"] = "TIMEOUT"
    elif max(stdout_bytes, stderr_bytes) > LOG_BYTES:
        receipt["status"] = "LOG_OVERFLOW"
    elif exit_code != 0:
        receipt["status"] = "WORKER_FAILED"
    elif tree_bytes > OUTPUT_BYTES:
        receipt["status"] = "OUTPUT_OVERFLOW"
    else:
        receipt["status"] = "WORKER_COMPLETED"

    signum = observe_interrupt(True)
    if signum is not None:
        receipt["interrupt_signal"] = signum
        if receipt["status"] == "WORKER_COMPLETED":
            receipt["status"] = "INTERRUPTED"

    if receipt["status"] == "WORKER_COMPLETED":
        # Both receipt files occupy space until the atomic replacement.
        prior_size = status_path.stat().st_size
        other_bytes = tree_bytes - prior_size
        text = _receipt_text(receipt)
        while receipt["output_bytes"] != other_bytes + len(text.encode("utf-8")):
            receipt["output_bytes"] = other_bytes + len(text.encode("utf-8"))
            text = _receipt_text(receipt)

        staged = status_path.with_suffix(".json.tmp")
        staging_bytes = prior_size + receipt["output_bytes"]
        if staging_bytes > OUTPUT_BYTES:
            receipt.update(status="OUTPUT_OVERFLOW", projected_staging_bytes=staging_bytes)
        else:
            staged.write_text(text, encoding="utf-8", newline="\n")
            staged_tree_bytes = _tree_bytes(output)
            final_log_bytes = max((output / name).stat().st_size for name in ("stdout.log", "stderr.log"))
            observe_interrupt(True)
            final_elapsed = time.monotonic() - started
            if final_elapsed > receipt["timeout_seconds"]:
                receipt["status"] = "TIMEOUT"
            elif final_log_bytes > LOG_BYTES:
                receipt["status"] = "LOG_OVERFLOW"
            elif staged_tree_bytes > OUTPUT_BYTES:
                receipt.update(status="OUTPUT_OVERFLOW", output_bytes=staged_tree_bytes)
            elif staged_tree_bytes - prior_size != receipt["output_bytes"]:
                raise ValueError("output changed during completion staging")
            signum = observe_interrupt(False)
            if signum is not None:
                receipt["interrupt_signal"] = signum
                if receipt["status"] == "WORKER_COMPLETED":
                    receipt["status"] = "INTERRUPTED"
            if receipt["status"] == "WORKER_COMPLETED":
                staged.replace(status_path)
                observe_interrupt(True)
                final_elapsed = time.monotonic() - started
                signum = observe_interrupt(False)
                if signum is not None:
                    receipt.update(status="INTERRUPTED", interrupt_signal=signum)
                elif final_elapsed > receipt["timeout_seconds"]:
                    receipt["status"] = "TIMEOUT"
                else:
                    return {**receipt, "publication": {
                        "receipt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                        "elapsed_seconds": final_elapsed,
                    }}
            receipt["elapsed_seconds"] = final_elapsed

    _write_terminal(status_path, receipt)
    return receipt


def supervise(packet_root: Path, output: Path, *, timeout: float = WALL_SECONDS,
              command: list[str] | None = None,
              launch_receipt: Path | None = None,
              wsl_admission_receipt: Path | None = None,
              interrupted_signal: Callable[[], int | None] | None = None) -> dict:
    signals: list[int] = []

    def latch_interrupt(signum, _frame):
        if not signals:
            signals.append(signum)

    def observe_interrupt(poll_callback: bool):
        if poll_callback and not signals and interrupted_signal is not None:
            observed = interrupted_signal()
            if observed is not None:
                latch_interrupt(observed, None)
        return signals[0] if signals else None

    handlers = {}
    if threading.current_thread() is threading.main_thread():
        handlers = {signum: signal.getsignal(signum) for signum in (signal.SIGINT, signal.SIGTERM)}
    try:
        for signum in handlers:
            signal.signal(signum, latch_interrupt)
        return _supervise(packet_root, output, timeout, command, launch_receipt,
                          wsl_admission_receipt, observe_interrupt)
    finally:
        for signum, handler in handlers.items():
            signal.signal(signum, handler)


def _supervise(packet_root: Path, output: Path, timeout: float, command: list[str] | None,
               launch_receipt: Path | None, wsl_admission_receipt: Path | None,
               observe_interrupt: Callable[[bool], int | None]) -> dict:
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
            _write_terminal(status_path, receipt)
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
        if observe_interrupt(True) is not None:
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
                if observe_interrupt(True) is not None:
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
                if observe_interrupt(True) is not None:
                    reason = "INTERRUPTED"
                elif _group_alive(process):
                    reason = "CHILD_PROCESS_LEAK"
            if reason is not None:
                stage = "cleanup"
                _terminate_owned_group(process)
        stage = "finalize"
        exit_code = None if process is None else process.returncode
        return _finalize(output, status_path, receipt, started, reason, exit_code, observe_interrupt)

    except (Exception, KeyboardInterrupt) as exc:
        cleanup_failed = stage == "cleanup"
        if process is not None and stage not in {"cleanup", "finalize"}:
            try:
                _terminate_owned_group(process)
            except (Exception, KeyboardInterrupt) as cleanup_error:
                cleanup_failed = True
                receipt["cleanup_error"] = str(cleanup_error)
        signal_number = observe_interrupt(True)
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
        _write_terminal(status_path, receipt)
        return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--wsl-admission-receipt", type=Path, required=True)
    args = parser.parse_args()
    result = supervise(args.packet_root, args.output,
                       launch_receipt=args.launch_receipt,
                       wsl_admission_receipt=args.wsl_admission_receipt)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "WORKER_COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
