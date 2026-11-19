"""Provision a reviewed copy OUTSIDE the checkout; run with Python -I -S.

The coordinator separately pins this file and the external policy SHA-256.
No project imports occur here. Verified commit bytes are staged into a private
execution checkout before any project entry point is started.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import signal
import subprocess
import sys
import time
import traceback
import zlib

IMPLEMENTATION_FILES = (
    "scripts/mace_bootstrap.py", "scripts/mace_loader.py", "scripts/check_mace_parity.py",
    "scripts/supervise_mace_parity.py", "scripts/audit_mace_parity.py",
    "trust/mace_trusted_launcher.py", "tests/test_mace_trusted_launcher.py",
    "docs/run-mace-parity.md", "tests/test_mace_loader.py", "tests/test_mace_parity.py",
    "tests/test_mace_parity_supervisor.py", "tests/test_mace_parity_audit.py",
    "tests/mace_test_runtime.py",
)
FROZEN_FILES = ("scripts/mace_reference_adapter.py", "evaluation/adapter.py")
APPROVED = ("APPROVED", "ACCEPTED")


def digest(data):
    return hashlib.sha256(data).hexdigest()


def absolute(value):
    if not isinstance(value, str) or not Path(value).is_absolute():
        raise ValueError("policy paths must be absolute")
    return Path(value).resolve()


def outside(path, checkout):
    if path == checkout or checkout in path.parents:
        raise ValueError("trusted launcher, policy, receipts and execution root must be outside the mutable checkout")


def git(checkout, *args):
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1")
    return subprocess.check_output(
        ["git", "--no-replace-objects", "-c", "core.hooksPath=" + os.devnull,
         "-C", str(checkout), *args], env=env, stderr=subprocess.PIPE, timeout=60)


def pinned_bytes(path, sha256):
    if path.is_symlink() or not path.is_file():
        raise ValueError("trusted input must be a regular non-symlink file")
    raw = path.read_bytes()
    if digest(raw) != sha256:
        raise ValueError("external policy/receipt SHA-256 mismatch")
    return raw


def pinned_json(path, sha256):
    return json.loads(pinned_bytes(path, sha256))


def verify_sources(checkout, commit):
    if git(checkout, "rev-parse", "HEAD").decode().strip() != commit:
        raise ValueError("checkout HEAD differs from externally approved commit")
    # Resolve a commit object, refusing tags, replacement objects and path labels.
    if git(checkout, "cat-file", "-t", commit).strip() != b"commit":
        raise ValueError("approval does not pin an immutable commit")
    verified = {}
    for name in (*IMPLEMENTATION_FILES, *FROZEN_FILES):
        path = checkout / name
        if path.is_symlink() or not path.is_file() or checkout not in path.resolve().parents:
            raise ValueError("missing, symlinked or escaping source: " + name)
        expected = git(checkout, "cat-file", "blob", commit + ":" + name)
        if path.read_bytes() != expected:
            raise ValueError("raw source differs before project startup: " + name)
        verified[name] = expected
    return verified


def stage_git_objects(source, execution, commit, baseline):
    """Copy only required raw Git objects; never run upload-pack or source hooks."""
    git(execution, "-c", "init.templateDir=", "init", "--quiet")
    objects = {}
    def capture(kind, oid):
        raw = git(source, "cat-file", kind, oid)
        encoded = kind.encode() + b" " + str(len(raw)).encode() + b"\0" + raw
        if hashlib.sha1(encoded).hexdigest() != oid:
            raise ValueError("Git object content differs from immutable object ID")
        objects[oid] = encoded
    for revision, names in ((commit, (*IMPLEMENTATION_FILES, *FROZEN_FILES)),
                            (baseline, FROZEN_FILES)):
        capture("commit", revision)
        tree = git(source, "rev-parse", revision + "^{tree}").decode().strip()
        capture("tree", tree)
        for name in names:
            parts = Path(name).parts
            for index in range(1, len(parts)):
                relative = "/".join(parts[:index])
                oid = git(source, "rev-parse", revision + ":" + relative).decode().strip()
                capture("tree", oid)
            oid = git(source, "rev-parse", revision + ":" + name).decode().strip()
            capture("blob", oid)
    for oid, raw in objects.items():
        path = execution / ".git/objects" / oid[:2] / oid[2:]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(zlib.compress(raw))
    (execution / ".git/HEAD").write_text(commit + "\n")


def prepare(policy_path, policy_sha256, entry):
    if entry not in ("supervisor", "audit"):
        raise ValueError("worker preparation is private to the bounded supervisor")
    policy = pinned_json(policy_path, policy_sha256)
    if policy.get("schema") != 1 or policy.get("status") not in APPROVED:
        raise ValueError("external launch policy is not approved")
    checkout = absolute(policy.get("checkout_root"))
    execution = absolute(policy.get("execution_root"))
    for path in (Path(__file__).resolve(), policy_path, execution):
        outside(path, checkout)
    if checkout == execution or execution in checkout.parents:
        raise ValueError("execution root must not contain the checkout")
    if digest(Path(__file__).read_bytes()) != policy.get("trusted_launcher_sha256"):
        raise ValueError("external launcher differs from the coordinator-pinned bytes")
    commit = policy.get("implementation_commit")
    if (not isinstance(commit, str) or len(commit) != 40
            or any(c not in "0123456789abcdef" for c in commit)):
        raise ValueError("external policy requires a full commit SHA")
    baseline = policy.get("baseline_commit")
    if (not isinstance(baseline, str) or len(baseline) != 40
            or any(c not in "0123456789abcdef" for c in baseline)):
        raise ValueError("external policy requires a full baseline SHA")
    if policy.get("python_executable") != sys.executable:
        raise ValueError("interpreter differs from the external approval")
    launch_path = absolute(policy.get("launch_receipt"))
    admission_path = absolute(policy.get("wsl_admission_receipt"))
    for path in (launch_path, admission_path):
        outside(path, checkout)
        if execution == path or execution in path.parents:
            raise ValueError("reviewed receipts must be retained outside the execution root")
    launch_raw = pinned_bytes(launch_path, policy.get("launch_sha256"))
    admission_raw = pinned_bytes(admission_path, policy.get("wsl_admission_sha256"))
    launch = json.loads(launch_raw)
    admission = json.loads(admission_raw)
    if (launch.get("status") not in APPROVED or admission.get("status") not in APPROVED
            or launch.get("implementation_commit") != commit
            or admission.get("identities", {}).get("implementation_commit") != commit
            or launch.get("wsl_admission_sha256") != policy.get("wsl_admission_sha256")):
        raise ValueError("external policy, launch and WSL approval disagree")
    packet = absolute(launch.get("packet_root"))
    attempt = absolute(launch.get("attempt_root"))
    supervisor = [sys.executable, "-I", "-S", str(Path(__file__).resolve()),
                  "--entry", "supervisor", "--policy", str(policy_path)]
    if (launch.get("trusted_launcher") != str(Path(__file__).resolve())
            or launch.get("trusted_policy") != str(policy_path)):
        raise ValueError("launch does not bind the external launcher and policy")
    if launch.get("command") != supervisor:
        raise ValueError("launch command is not bound to the trusted execution snapshot")
    if entry == "supervisor":
        if execution.exists() or attempt.exists():
            raise ValueError("execution and attempt roots must both be fresh")
        sources = verify_sources(checkout, commit)
        # These files are loaded from captured verified bytes, not copied again
        # from the mutable checkout. Rechecks cannot introduce a check/use race.
        execution.mkdir(mode=0o700, parents=True, exist_ok=False)
        stage_git_objects(checkout, execution, commit, baseline)
        for name, raw in sources.items():
            target = execution / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(raw)
            target.chmod(0o444)
        verified_execution = verify_sources(execution, commit)
        receipts = execution / ".trusted-receipts"
        receipts.mkdir(mode=0o700)
        for name, raw in (("launch.json", launch_raw), ("admission.json", admission_raw)):
            path = receipts / name
            path.write_bytes(raw)
            path.chmod(0o444)
    else:
        verified_execution = verify_sources(execution, commit)
    launch_path = execution / ".trusted-receipts/launch.json"
    admission_path = execution / ".trusted-receipts/admission.json"
    pinned_json(launch_path, policy.get("launch_sha256"))
    pinned_json(admission_path, policy.get("wsl_admission_sha256"))
    # The separately provisioned launcher itself must come from this commit.
    if git(execution, "cat-file", "blob", commit + ":trust/mace_trusted_launcher.py") != Path(__file__).read_bytes():
        raise ValueError("external launcher is not the reviewed commit version")
    entries = {"supervisor": "scripts/supervise_mace_parity.py",
               "audit": "scripts/audit_mace_parity.py"}
    selected = entries[entry]
    command = [str(execution / selected), "--packet-root", str(packet)]
    if entry == "supervisor":
        command += ["--output", str(attempt), "--launch-receipt", str(launch_path),
                    "--wsl-admission-receipt", str(admission_path)]
    else:
        command += ["--attempt-root", str(attempt)]
    return command, execution, verified_execution, selected


class OwnedWorker:
    """Popen-compatible wait interface for a worker forked by its supervisor."""

    def __init__(self, pid):
        self.pid = pid
        self.returncode = None

    def poll(self):
        if self.returncode is None:
            pid, status = os.waitpid(self.pid, os.WNOHANG)
            if pid:
                self.returncode = os.waitstatus_to_exitcode(status)
        return self.returncode

    def wait(self, timeout=None):
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.poll() is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise subprocess.TimeoutExpired("supervisor-owned worker", timeout)
            time.sleep(0.01)
        return self.returncode


def main():
    WALL_SECONDS = 1800
    CLEANUP_SECONDS = 10
    OUTPUT_BYTES = 256 * 1024 * 1024
    LOG_BYTES = 16 * 1024 * 1024
    TERMINATION_SIGNALS = (signal.SIGINT, signal.SIGTERM)
    if os.name == "posix":
        TERMINATION_SIGNALS += (signal.SIGHUP, signal.SIGQUIT)


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

        try:
            for signum in TERMINATION_SIGNALS:
                previous[signum] = signal.signal(signum, interrupt)
            yield begin_cleanup
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)


    @contextmanager
    def _defer_spawn_signals():
        """Assign the child to cleanup state before delivering a pending signal."""
        if os.name == "posix":
            previous = signal.pthread_sigmask(signal.SIG_BLOCK, TERMINATION_SIGNALS)
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
                  wsl_admission_receipt: Path | None = None) -> dict:
        with _interruption_handlers() as begin_cleanup:
            return _supervise(packet_root, output, timeout=timeout, command=command,
                              launch_receipt=launch_receipt,
                              wsl_admission_receipt=wsl_admission_receipt,
                              begin_cleanup=begin_cleanup)


    def _supervise(packet_root: Path, output: Path, *, timeout: float,
                   command: list[str] | None, launch_receipt: Path | None,
                   wsl_admission_receipt: Path | None, begin_cleanup) -> dict:
        started = (float(os.environ.get("MACE_TRUST_STARTED", time.monotonic()))
                   if command is None else time.monotonic())
        output = Path(output)
        if output.exists():
            raise FileExistsError(output)
        created = False
        process = None
        stage = "setup"
        worker = command
        receipt = {"status": "RUNNING", "command": command, "timeout_seconds": timeout,
                   "output_limit_bytes": OUTPUT_BYTES, "log_limit_bytes": LOG_BYTES,
                   "started_unix": time.time()}
        status_path = output / "supervisor-status.json"
        stdout_path, stderr_path = output / "stdout.log", output / "stderr.log"
        reason = None
        try:
            # Defer delivery just until ownership of the fresh directory is known.
            # Never overwrite a pre-existing/racing attempt in an error handler.
            with _defer_spawn_signals():
                output.mkdir(parents=True, exist_ok=False)
                created = True
            stdout_path.touch(exist_ok=False)
            stderr_path.touch(exist_ok=False)
            status_path.write_text(json.dumps(receipt, indent=2) + "\n")
            if command is None and (launch_receipt is None or wsl_admission_receipt is None):
                raise ValueError("production launch requires reviewed launch and WSL admission receipts")
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
            if not created:
                raise
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
            # An interruption may precede one or both log creations. Preserve any
            # existing worker output; create only missing empty files.
            stdout_path.touch(exist_ok=True)
            stderr_path.touch(exist_ok=True)
            status_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            return receipt

    def dispatch(command, verified, selected, *, module_name="__main__"):
        """Execute captured authenticated bytes; never read a project entry again."""
        sys.argv = command
        namespace = {"__name__": module_name, "__file__": command[0]}
        exec(compile(verified[selected], command[0], "exec"), namespace)
        return namespace

    def supervisor_worker_spawn(policy_path, policy_sha256):
        """Return a single-use in-memory handoff, never a worker CLI or env token.

        Production is WSL/POSIX. The child inherits the already isolated interpreter
        and verified import map. Only the verified supervisor receives this closure.
        Its logs are open and status is RUNNING before it calls the closure.
        """
        owner = os.getpid()
        used = False

        def spawn(stdout, stderr):
            nonlocal used
            if used or os.getpid() != owner:
                raise ValueError("worker handoff is single-use and supervisor-owned")
            used = True
            if os.name != "posix" or not hasattr(os, "fork"):
                raise ValueError("production worker requires the approved WSL/POSIX runtime")
            ready_read, ready_write = os.pipe()
            try:
                pid = os.fork()
            except BaseException:
                os.close(ready_read)
                os.close(ready_write)
                raise
            if pid == 0:
                os.close(ready_read)
                code = 2
                try:
                    os.setsid()
                    # The supervisor defers terminal signals across assignment
                    # of this child to its cleanup state. Do not inherit its
                    # blocked mask or Python interruption handlers.
                    terminal_signals = {signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT}
                    for terminal_signal in terminal_signals:
                        signal.signal(terminal_signal, signal.SIG_DFL)
                    signal.pthread_sigmask(signal.SIG_UNBLOCK, terminal_signals)
                    os.dup2(stdout.fileno(), 1)
                    os.dup2(stderr.fileno(), 2)
                    os.write(ready_write, b"R")
                    os.close(ready_write)
                    # Audit preparation authenticates the entire retained tree
                    # without staging or exposing a public worker selection.
                    command, execution, verified, _ = prepare(policy_path, policy_sha256, "audit")
                    selected = "scripts/check_mace_parity.py"
                    command[0] = str(execution / selected)
                    launch = json.loads((execution / ".trusted-receipts/launch.json").read_bytes())
                    worker_pid = os.getpid()
                    deadline = float(os.environ["MACE_TRUST_STARTED"]) + 1800

                    def check_liveness(packet_root=None, attempt_root=None):
                        if (os.getpid() != worker_pid or os.getppid() != owner
                                or os.getsid(0) != worker_pid or time.monotonic() >= deadline):
                            raise ValueError("supervised worker is no longer live")
                        for supplied, approved in ((packet_root, launch["packet_root"]),
                                                   (attempt_root, launch["attempt_root"])):
                            if supplied is not None and str(Path(supplied).resolve()) != approved:
                                raise ValueError("worker does not authorize these paths")

                    import mace_loader

                    def construct(snapshot):
                        # The real constructor is private to this authenticated
                        # child. No project module/global exports this provider.
                        check_liveness()
                        mace_loader._require_standard_finders()
                        validate = mace_loader.validate_dependency_imports
                        validate(("mace.calculators.mace", "torch", "numpy", "ase.io"))
                        import torch
                        import mace
                        validate(("mace.calculators.mace",))
                        import mace.calculators
                        validate(("mace.calculators.mace",))
                        from mace.calculators.mace import MACECalculator
                        validate(("mace.calculators.mace",))
                        if MACECalculator.__module__ != "mace.calculators.mace":
                            raise ValueError("constructor is not defined in the approved MACE module")
                        torch.set_num_threads(1)
                        return MACECalculator(model_paths=[str(snapshot)], device="cpu",
                            default_dtype="float64", enable_cueq=False, enable_oeq=False, compile_mode=None)

                    def load_arm(packet_root, attempt_root, *, expected_species):
                        check_liveness(packet_root, attempt_root)
                        verify_packet = mace_loader.verify_packet
                        _approved_launch = mace_loader._approved_launch
                        _validate_import_origins = mace_loader._validate_import_origins
                        _regular_file = mace_loader._regular_file
                        MODEL_RELATIVE = mace_loader.MODEL_RELATIVE
                        _sha256 = mace_loader._sha256
                        MODEL_SHA256 = mace_loader.MODEL_SHA256
                        MODEL_BYTES = mace_loader.MODEL_BYTES
                        AdmissionError = mace_loader.AdmissionError
                        packet = verify_packet(packet_root)
                        launch, admission, live = _approved_launch(Path(attempt_root), Path(packet_root))
                        _validate_import_origins(live)
                        species = launch.get("species")
                        from ase.data import atomic_numbers
                        if (not isinstance(species, list) or len(species) != 21
                                or any(not isinstance(x, str) or atomic_numbers.get(x, 0) <= 0 for x in species)):
                            raise AdmissionError("launch receipt must contain exactly 21 real element symbols")
                        if expected_species is None or tuple(species) != tuple(expected_species):
                            raise AdmissionError("launch species order differs from validated input frames")
                        snapshot = Path(attempt_root) / "private" / "final_epoch.model"
                        snapshot.parent.mkdir(parents=True, exist_ok=False)
                        source = _regular_file(Path(packet_root), MODEL_RELATIVE)
                        with source.open("rb") as incoming, snapshot.open("xb") as outgoing:
                            shutil.copyfileobj(incoming, outgoing)
                            outgoing.flush()
                            os.fsync(outgoing.fileno())
                        before = _sha256(snapshot)
                        if before != MODEL_SHA256 or snapshot.stat().st_size != MODEL_BYTES:
                            raise AdmissionError("copied model snapshot failed identity check")
                        _validate_import_origins(live)
                        check_liveness(packet_root, attempt_root)
                        calculator = construct(snapshot.resolve())
                        after = _sha256(snapshot)
                        if after != before:
                            raise AdmissionError("model snapshot changed during construction")
                        from mace_reference_adapter import calculator_arm
                        identity = {"model_sha256": MODEL_SHA256, "model_bytes": MODEL_BYTES,
                                    "snapshot_path": str(snapshot.resolve()), "epoch_index": 2047}
                        arm = calculator_arm(calculator, species, arm_id="mace-runA-final-ema",
                                             checkpoint_identity=identity)
                        receipt = {"status": "ADMITTED", "packet": packet,
                                   "model_snapshot": identity, "post_construction_sha256": after,
                                   "constructor": {"device": "cpu", "torch_threads": 1,
                                                   "default_dtype": "float64", "enable_cueq": False,
                                                   "enable_oeq": False, "compile_mode": None},
                                   "launch_receipt": launch, "wsl_admission": admission,
                                   "live_identity_comparison": live}
                        (Path(attempt_root) / "admission-receipt.json").write_text(
                            json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                        return arm, receipt

                    check_liveness()
                    worker = dispatch(command, verified, selected, module_name="verified_worker_library")
                    result = worker["_run_parity"](Path(launch["packet_root"]), Path(launch["attempt_root"]),
                                                  load_arm=load_arm, check_liveness=check_liveness)
                    print(json.dumps(result, indent=2, sort_keys=True))
                    code = 0 if result.get("status") in {"PARITY_PASS", "PARITY_FAIL"} else 2
                except SystemExit as exc:
                    code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 2)
                except BaseException:
                    traceback.print_exc()
                finally:
                    try:
                        sys.stdout.flush()
                        sys.stderr.flush()
                    finally:
                        os._exit(code)
            os.close(ready_write)
            process = OwnedWorker(pid)
            try:
                # Do not return until the owned process group exists. Startup and
                # the repeated trust checks remain inside the shared wall clock.
                readable, _, _ = select.select([ready_read], [], [], 5)
                if not readable or os.read(ready_read, 1) != b"R":
                    raise ValueError("worker did not establish its owned process group")
            except BaseException:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
                raise
            finally:
                os.close(ready_read)
            return process

        return spawn

    if not sys.flags.isolated or not sys.flags.no_site:
        raise SystemExit("trusted launcher requires Python -I -S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--policy-sha256")
    parser.add_argument("--entry", choices=("supervisor", "audit"), required=True)
    args = parser.parse_args()
    started = time.monotonic()
    try:
        # This independent digest cannot be embedded in the receipt that this
        # policy itself hashes, which would create a circular digest binding.
        policy_sha256 = args.policy_sha256 or os.environ.get("MACE_TRUST_POLICY_SHA256")
        if not policy_sha256:
            raise ValueError("an independently pinned external policy digest is required")
        command, execution, verified, selected = prepare(args.policy.resolve(), policy_sha256, args.entry)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print("TRUST_REFUSED: " + str(exc), file=sys.stderr)
        return 2
    # Do not inherit Python/Git environment knobs into verified project code.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PYTHON", "GIT_"))}
    env["MACE_TRUST_STARTED"] = str(started)
    env["MACE_TRUST_POLICY_SHA256"] = policy_sha256
    os.environ.clear()
    os.environ.update(env)
    os.chdir(execution)
    sys.dont_write_bytecode = True
    # Dispatch exists only in this separately provisioned trust root. Execute
    # the captured authenticated bytes, never re-read an entry or a pyc file.
    bootstrap_path = str(execution / "scripts/mace_bootstrap.py")
    namespace = {"__name__": "verified_mace_bootstrap", "__file__": bootstrap_path}
    exec(compile(verified["scripts/mace_bootstrap.py"], bootstrap_path, "exec"), namespace)
    namespace["configure_imports"]()
    if args.entry == "supervisor":
        worker_spawn = supervisor_worker_spawn(args.policy.resolve(), policy_sha256)
        fields = dict(zip(command[1::2], command[2::2]))
        result = supervise(Path(fields["--packet-root"]), Path(fields["--output"]),
                           launch_receipt=Path(fields["--launch-receipt"]),
                           wsl_admission_receipt=Path(fields["--wsl-admission-receipt"]))
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if result["status"] == "WORKER_COMPLETED" else 2
    dispatch(command, verified, selected)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
