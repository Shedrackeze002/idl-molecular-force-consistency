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
              wsl_admission_receipt: Path | None = None) -> dict:
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
               "started_unix": time.time()}
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
            shutil.copyfile(launch_source, output / "launch-receipt.json")
            shutil.copyfile(admission_source, output / admission_name)
        worker = command or [sys.executable, str(Path(__file__).with_name("check_mace_parity.py")),
                             "--packet-root", str(Path(packet_root).resolve()),
                             "--attempt-root", str(output.resolve())]
        receipt["command"] = worker
        if time.monotonic() - started > timeout:
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
                if elapsed > timeout:
                    reason = "TIMEOUT"
                elif stdout_path.stat().st_size > LOG_BYTES or stderr_path.stat().st_size > LOG_BYTES:
                    reason = "LOG_OVERFLOW"
                elif _tree_bytes(output) > OUTPUT_BYTES:
                    reason = "OUTPUT_OVERFLOW"
                if reason:
                    _terminate_owned_group(process)
                    break
                time.sleep(0.05)
            if reason is None and _group_alive(process):
                reason = "CHILD_PROCESS_LEAK"
                _terminate_owned_group(process)
    except Exception as exc:
        if process is not None:
            try:
                _terminate_owned_group(process)
            except Exception:
                pass
        failure_status = {"spawn": "SPAWN_FAILED", "monitor": "CLEANUP_FAILED"}.get(stage, "SETUP_FAILED")
        receipt.update(status=failure_status,
                       failure_stage=stage, error_type=type(exc).__name__, error=str(exc),
                       elapsed_seconds=time.monotonic() - started,
                       exit_code=None if process is None else process.poll())
        status_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return receipt
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
    if receipt["output_bytes"] > OUTPUT_BYTES:
        receipt["status"] = "OUTPUT_OVERFLOW"
        status_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    # The last successful persistence is part of the wall-clock contract.
    persisted_elapsed = time.monotonic() - started
    if persisted_elapsed > timeout and receipt["status"] != "TIMEOUT":
        receipt["status"] = "TIMEOUT"
        receipt["elapsed_seconds"] = persisted_elapsed
        status_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
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
