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


def supervise(packet_root: Path, output: Path, *, timeout: float = WALL_SECONDS,
              command: list[str] | None = None,
              launch_receipt: Path | None = None,
              wsl_admission_receipt: Path | None = None) -> dict:
    output = Path(output)
    if command is None:
        if launch_receipt is None or wsl_admission_receipt is None:
            raise ValueError("production launch requires reviewed launch and WSL admission receipts")
        launch_source = Path(launch_receipt)
        admission_source = Path(wsl_admission_receipt)
        launch = json.loads(launch_source.read_text(encoding="utf-8"))
        admission_name = launch.get("wsl_admission_receipt")
        if not isinstance(admission_name, str) or Path(admission_name).name != admission_name:
            raise ValueError("launch receipt must use a local WSL admission receipt filename")
    output.mkdir(parents=True, exist_ok=False)
    if command is None:
        shutil.copyfile(launch_source, output / "launch-receipt.json")
        shutil.copyfile(admission_source, output / admission_name)
    stdout_path, stderr_path = output / "stdout.log", output / "stderr.log"
    worker = command or [sys.executable, str(Path(__file__).with_name("check_mace_parity.py")),
                         "--packet-root", str(Path(packet_root).resolve()),
                         "--attempt-root", str(output.resolve())]
    receipt = {"status": "RUNNING", "command": worker, "timeout_seconds": timeout,
               "output_limit_bytes": OUTPUT_BYTES, "log_limit_bytes": LOG_BYTES,
               "started_unix": time.time()}
    (output / "supervisor-status.json").write_text(json.dumps(receipt, indent=2) + "\n")
    started = time.monotonic()
    creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    start_new_session = os.name != "nt"
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(worker, stdout=stdout, stderr=stderr,
                                   creationflags=creationflags, start_new_session=start_new_session)
        reason = None
        while process.poll() is None:
            elapsed = time.monotonic() - started
            if elapsed > timeout:
                reason = "TIMEOUT"
            elif stdout_path.stat().st_size > LOG_BYTES or stderr_path.stat().st_size > LOG_BYTES:
                reason = "LOG_OVERFLOW"
            elif _tree_bytes(output) > OUTPUT_BYTES:
                reason = "OUTPUT_OVERFLOW"
            if reason:
                if os.name == "nt":
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=min(5, CLEANUP_SECONDS))
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        process.kill()
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=max(1, CLEANUP_SECONDS - 5))
                break
            time.sleep(0.05)
    elapsed = time.monotonic() - started
    exit_code = process.returncode
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
                   output_bytes=_tree_bytes(output), stdout_bytes=stdout_path.stat().st_size,
                   stderr_bytes=stderr_path.stat().st_size)
    (output / "supervisor-status.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
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
