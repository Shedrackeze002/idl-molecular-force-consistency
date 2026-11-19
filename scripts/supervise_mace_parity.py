"""Bound a single MACE parity worker process and preserve every outcome."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
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


class SupervisorInterrupted(BaseException):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(signal.Signals(signum).name)


@contextmanager
def _interruption_handlers():
    """Keep handlers installed through cleanup; repeated signals cannot orphan it."""
    previous = {}
    interrupted = False

    def interrupt(signum, frame):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            raise SupervisorInterrupted(signum)

    def begin_cleanup():
        nonlocal interrupted
        interrupted = True

    signals = (signal.SIGINT, signal.SIGTERM)
    if os.name == "posix":
        signals += (signal.SIGHUP,)
    try:
        for signum in signals:
            previous[signum] = signal.signal(signum, interrupt)
        yield begin_cleanup
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


@contextmanager
def _defer_spawn_signals():
    """Assign the child to cleanup state before delivering a pending signal."""
    if os.name == "posix":
        previous = signal.pthread_sigmask(signal.SIG_BLOCK,
                                          {signal.SIGINT, signal.SIGTERM, signal.SIGHUP})
        try:
            yield
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)
    else:
        yield


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
    if _group_alive(process):
        raise RuntimeError("owned worker group remains alive after cleanup deadline")


def supervise(packet_root: Path, output: Path, *, timeout: float = WALL_SECONDS,
              command: list[str] | None = None,
              launch_receipt: Path | None = None,
              wsl_admission_receipt: Path | None = None,
              worker_spawn=None) -> dict:
    with _interruption_handlers() as begin_cleanup:
        return _supervise(packet_root, output, timeout=timeout, command=command,
                          launch_receipt=launch_receipt,
                          wsl_admission_receipt=wsl_admission_receipt, worker_spawn=worker_spawn,
                          begin_cleanup=begin_cleanup)


def _supervise(packet_root: Path, output: Path, *, timeout: float,
               command: list[str] | None, launch_receipt: Path | None,
               wsl_admission_receipt: Path | None, worker_spawn, begin_cleanup) -> dict:
    started = (float(os.environ.get("MACE_TRUST_STARTED", time.monotonic()))
               if command is None else time.monotonic())
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
            admission = json.loads(admission_source.read_text(encoding="utf-8"))
            admission_name = launch.get("wsl_admission_receipt")
            if not isinstance(admission_name, str) or Path(admission_name).name != admission_name:
                raise ValueError("launch receipt must use a local WSL admission receipt filename")
            from mace_loader import _validated_launch_identity
            implementation_commit = admission.get("identities", {}).get("implementation_commit")
            if not isinstance(implementation_commit, str):
                raise ValueError("WSL admission lacks an implementation commit")
            _validated_launch_identity(launch, output, packet_root, implementation_commit,
                                       require_live_interpreter=True)
            if not callable(worker_spawn):
                raise ValueError("production worker requires the external launcher's supervisor handoff")
            shutil.copyfile(launch_source, output / "launch-receipt.json")
            shutil.copyfile(admission_source, output / admission_name)
        worker = command or launch["command"]
        receipt["command"] = worker
        if time.monotonic() - started > timeout:
            reason = "TIMEOUT"
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        start_new_session = os.name != "nt"
        stage = "spawn"
        with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
          if reason is None:
            if command is None:
                with _defer_spawn_signals():
                    process = worker_spawn(stdout, stderr)
            else:
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
                    begin_cleanup()
                    _terminate_owned_group(process)
                    break
                time.sleep(0.05)
            if reason is None and _group_alive(process):
                reason = "CHILD_PROCESS_LEAK"
                begin_cleanup()
                _terminate_owned_group(process)
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
    except BaseException as exc:
        interrupted = isinstance(exc, (SupervisorInterrupted, KeyboardInterrupt, SystemExit))
        cleanup_error = None
        if process is not None:
            try:
                begin_cleanup()
                _terminate_owned_group(process)
            except BaseException as cleanup_exc:
                cleanup_error = str(cleanup_exc)
        failure_status = ("INTERRUPTED" if interrupted else
                          {"spawn": "SPAWN_FAILED", "monitor": "CLEANUP_FAILED"}.get(stage, "SETUP_FAILED"))
        if cleanup_error is not None:
            failure_status = "CLEANUP_FAILED"
        receipt.update(status=failure_status,
                       failure_stage=stage, error_type=type(exc).__name__, error=str(exc),
                       elapsed_seconds=time.monotonic() - started,
                       exit_code=None if process is None else process.poll())
        if interrupted:
            receipt["interrupted"] = True
            receipt["signal"] = getattr(exc, "signum", None)
        if cleanup_error is not None:
            receipt["cleanup_error"] = cleanup_error
        status_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        return receipt


def main() -> int:
    if not sys.flags.isolated or not sys.flags.no_site:
        raise SystemExit("use python -I -S <external-trusted-launcher> --entry supervisor "
                         "--policy <approved-policy> --policy-sha256 <independently-pinned-digest>; "
                         "see docs/run-mace-parity.md")
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launch-receipt", type=Path, required=True)
    parser.add_argument("--wsl-admission-receipt", type=Path, required=True)
    args = parser.parse_args()
    result = supervise(args.packet_root, args.output,
                       launch_receipt=args.launch_receipt,
                       wsl_admission_receipt=args.wsl_admission_receipt,
                       worker_spawn=globals().get("__trusted_worker_spawn__"))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "WORKER_COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
