"""Launch-time checks for the Stage 2 launch worker (plan v3 sections 2, 7 and 8).

Everything here reads bytes, git state or the environment; nothing loads a model.

- The tree checks: HEAD equals the commit a record names, the working tree is clean (untracked files
  included), no tracked file hides behind an assume-unchanged or skip-worktree flag, the twelve frozen
  files keep their hashes, and every project module that is imported is a file of this worktree whose
  bytes equal the committed bytes.
- The three diff gates, with the nine hashes a record names (derived file, frozen file and allow-list
  file, per target).
- The preflight: environment, volume, hard links, free space, and the tools a launch needs.
- The audit hook that watches file opens under the whole Group9 data tree (see `AuditGuard`).

The twelve frozen-file hashes live here, not in a test module, so a launch can check them at run time.
"""
import hashlib
import importlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from stage2_capped import admission, diff_gate, launch_io

# The twelve files the accepted evidence depends on, with the SHA-256 they had at the base commit.
FROZEN_FILES = {
    "evaluation/adapter.py":
        "640a4afc31e971fa33876cb0b6a022416accd0a3c348a597bbebe62483ba65c5",
    "evaluation/e3x_adapter.py":
        "853db9a7792b235e8f1f37de14e779aae4da7c088e08b1e8d8e80702850d9980",
    "evaluation/outcomes.py":
        "8f5f7caa15b5b3125dd394a86424925b1a65d7d35c25622200bbe4fafc167871",
    "evaluation/relax.py":
        "b5321a21dd17dd0f23e7d89575af96263f115812826faf7e73ebd2cd372b4c4e",
    "evaluation/runner.py":
        "28591f542e803bb5ac3a5935500a1a0ac21774b04040caf992d82e1b13100a90",
    "evaluation/validation_controls.py":
        "8df71e4c15acc04b9d6c10642aa3dbb14b52a600d3e966de9d43c76697d39af4",
    "scripts/diagnose_validation_recovery.py":
        "00f90d33279f21578d0bd7b5c58f31fbb953c0d4e4c311f59349ed3e3e026053",
    "scripts/e3x_phase_common.py":
        "a1d514b581f4375cbc6449523cddf1cd5f732e5a3f0644b2c4083bc86d3e23e7",
    "scripts/recovery_trace.py":
        "3fe0761f495e8be746e0afffc7d10fbb8f9b0fdd93c1ef4518cddf1cac3fb7b7",
    "scripts/report_shape_recovery.py":
        "96fc3ddb1f5ac44c300779c56b8e49b42b6e3401b78a22a54176500a5512272d",
    "scripts/run_validation_recovery.py":
        "9a16a6748752116c4a311ea93ef391557cbdcf0f34344e4e64b9a05aca159b76",
    "scripts/trace_validation_recovery.py":
        "7836c4fdf513b4456a3a5a83a123bb1b4d7688e4e4d5ab8f3fc1494914af72da",
}

THREAD_VARIABLES = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "XLA_FLAGS")
ALLOWLISTS = {"relax_capped": "allowlist.json", "restore_fork": "allowlist_restore.json",
              "pass_copy": "allowlist_pass.json"}
LAUNCH_MODULES = ("stage2_capped.launch_a0", "stage2_capped.launch_stage", "stage2_capped.launch_arms",
                  "stage2_capped.replay_evidence", "stage2_capped.launch_state", "stage2_capped.launch_record")
AUDIT_EXIT_CODE = 97
DATA_SUFFIXES = (".npz", ".zip")


class CheckError(RuntimeError):
    """A git or file operation a check depends on failed."""


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---- Git and the tree -----------------------------------------------------

def _git(worktree, *args):
    result = subprocess.run(["git", "-C", str(worktree), *args], capture_output=True, check=False)
    if result.returncode != 0:
        raise CheckError(f"git {' '.join(args)} failed: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout


def head_commit(worktree):
    return _git(worktree, "rev-parse", "HEAD").decode().strip()


def committed_bytes(worktree, relative_path):
    """The bytes git holds at HEAD for a path (git's own blob ids are SHA-1, so callers hash these)."""
    return _git(worktree, "show", f"HEAD:{relative_path}")


def tree_problems(worktree, commit):
    """Why this tree is not the committed tree a record names (empty list: it is)."""
    try:
        head = head_commit(worktree)
        status = _git(worktree, "status", "--porcelain", "--untracked-files=all").decode(errors="replace")
        flags = _git(worktree, "ls-files", "-v").decode(errors="replace").splitlines()
    except CheckError as error:
        return [str(error)]
    problems = []
    if head != commit:
        problems.append(f"HEAD is {head}; the record names {commit}")
    if status.strip():
        lines = status.splitlines()
        problems.append(f"the working tree is not clean ({len(lines)} entries, first {lines[0]!r})")
    hidden = [line for line in flags if line[:1].islower() or line[:1] == "S"]
    if hidden:
        problems.append(f"{len(hidden)} tracked files hide behind an assume-unchanged or skip-worktree flag, "
                        f"first {hidden[0]!r}")
    return problems


def frozen_problems(worktree, expected=None):
    """The twelve frozen files against the recorded hashes."""
    expected = FROZEN_FILES if expected is None else expected
    problems = []
    for relative, wanted in sorted(expected.items()):
        path = Path(worktree) / relative
        if not path.is_file():
            problems.append(f"frozen file {relative} is missing")
        elif sha256_file(path) != wanted:
            problems.append(f"frozen file {relative} differs from its recorded SHA-256")
    return problems


def project_names(worktree):
    """Top-level module names that belong to this worktree's source folders."""
    worktree = Path(worktree)
    names = {"evaluation", "stage2_capped"}
    source = worktree / "src"
    if source.is_dir():
        names |= {p.name for p in source.iterdir()
                  if p.is_dir() and not p.name.startswith(".") and not p.name.endswith(".egg-info")
                  and p.name != "__pycache__"}
    names |= {p.stem for p in (worktree / "scripts").glob("*.py")}
    return names


def module_inventory(worktree, modules=None):
    """(inventory, problems): for every imported project module, its path relative to the worktree
    and the SHA-256 of its bytes. A project module imported from outside the worktree, or whose bytes
    differ from the committed bytes, is a problem."""
    modules = sys.modules if modules is None else modules
    real = Path(worktree).resolve()
    names = project_names(real)
    inventory, problems = {}, []
    for name, module in sorted(modules.items()):
        file = getattr(module, "__file__", None)
        if name.split(".")[0] not in names or not file:
            continue
        path = Path(file).resolve()
        try:
            relative = str(path.relative_to(real))
        except ValueError:
            problems.append(f"{name} is imported from {path}, outside the worktree")
            continue
        inventory[relative] = sha256_file(path)
        try:
            committed = hashlib.sha256(committed_bytes(real, relative)).hexdigest()
        except CheckError:
            problems.append(f"{relative} is not committed at HEAD")
            continue
        if committed != inventory[relative]:
            problems.append(f"{relative} differs from the committed bytes")
    return inventory, problems


# ---- The diff gates -------------------------------------------------------

def diff_gate_hashes(worktree):
    """(hashes, problems): run the three gates against the worktree files and return the nine hashes
    a record names, per target the derived file, the frozen file and the allow-list file."""
    worktree = Path(worktree)
    hashes, problems = {}, []
    for target, allowlist_name in ALLOWLISTS.items():
        allowlist = worktree / "stage2_capped" / allowlist_name
        problems += [f"{target}: {p}" for p in diff_gate.check_gate(target, allowlist_path=allowlist, root=worktree)]
        entry = json.loads(allowlist.read_text())["targets"][target]
        hashes[target] = {"derived": sha256_file(worktree / entry["derived_file"]),
                          "frozen": sha256_file(worktree / entry["frozen_file"]),
                          "allowlist": sha256_file(allowlist)}
    return hashes, problems


# ---- Runtime and preflight ------------------------------------------------

def thread_variable_problems(environ=None):
    environ = os.environ if environ is None else environ
    return [f"{name} is set; the accepted pin has it unset" for name in THREAD_VARIABLES if environ.get(name)]


def _jax_backend():
    try:
        import jax
        return jax.default_backend()
    except Exception as error:  # noqa: BLE001 - a missing backend is reported, not raised
        return f"unavailable: {type(error).__name__}"


def runtime_values(environ=None, *, include_backend=False):
    """The runtime facts a launch records. The parent leaves the backend out (it never imports the
    numerical stack); a worker includes it and the backend must be cpu."""
    environ = os.environ if environ is None else environ
    values = {"python": sys.version, "python_executable": sys.executable, "platform": platform.platform(),
              "machine": platform.machine(), "pythonpath": environ.get("PYTHONPATH"),
              "threads": {name: environ.get(name) for name in THREAD_VARIABLES},
              "group9_root_source": admission.root_source(),
              "group9_root_variable": environ.get(admission.ROOT_ENVIRONMENT_VARIABLE)}
    if include_backend:
        values["jax_backend"] = _jax_backend()
    return values


def runtime_problems(recorded, current):
    """Fields of `recorded` (A0's runtime) that differ from `current`; the backend must be cpu."""
    problems = [f"runtime {key} differs from A0's" for key in sorted(recorded)
                if key != "jax_backend" and recorded[key] != current.get(key)]
    if recorded.get("jax_backend") != "cpu":
        problems.append(f"A0 ran on the {recorded.get('jax_backend')!r} backend, not cpu")
    if "jax_backend" in current and current["jax_backend"] != "cpu":
        problems.append(f"this process runs on the {current['jax_backend']!r} backend, not cpu")
    return problems


def mount_point(path):
    path = Path(path).resolve()
    while not os.path.ismount(path):
        path = path.parent
    return str(path)


def hardlink_problem(directory):
    """None if a hard link can be made in `directory`, else why not (the launch writes with hard links)."""
    directory = Path(directory)
    if not directory.is_dir():
        return f"{directory} is not an existing directory"
    descriptor, name = tempfile.mkstemp(dir=directory, prefix=".linktest-")
    os.close(descriptor)
    second = name + ".link"
    try:
        os.link(name, second)
    except OSError as error:
        return f"cannot make a hard link in {directory}: {error}"
    finally:
        for leftover in (second, name):
            try:
                os.unlink(leftover)
            except FileNotFoundError:
                pass
    return None


def preflight_problems(*, worktree, root, launch_folder, output_volume, environ=None, free_bytes=launch_io.GIB):
    """Everything about the machine and the intended root that needs no model and writes nothing under
    the root or the registry. Returns a list of problems."""
    environ = os.environ if environ is None else environ
    worktree, root = Path(worktree).resolve(), Path(root)
    problems = thread_variable_problems(environ)
    parent = root.parent
    if not parent.is_dir():
        problems.append(f"the parent folder of the output root, {parent}, does not exist")
    else:
        for place in (parent, Path(launch_folder)):
            message = hardlink_problem(place)
            if message:
                problems.append(message)
        try:
            if mount_point(parent) != output_volume:
                problems.append(f"the output root is on volume {mount_point(parent)}, the record names {output_volume}")
            elif os.stat(parent).st_dev != os.stat(output_volume).st_dev:
                problems.append("the output root's device differs from the named volume's device")
        except OSError as error:
            problems.append(f"cannot stat the output volume: {error}")
        try:
            launch_io.require_free(parent, free_bytes)
        except Exception as error:  # noqa: BLE001 - the frozen check raises its own types
            problems.append(f"free space: {error}")
    resolved = root.resolve() if root.exists() else (parent.resolve() / root.name)
    if resolved == worktree or worktree in resolved.parents:
        problems.append(f"the output root {resolved} lies inside the worktree")
    for tool in ("caffeinate", "git"):
        if shutil.which(tool, path=environ.get("PATH")) is None:
            problems.append(f"{tool} is not on PATH")
    entries = [p for p in (environ.get("PYTHONPATH") or "").split(os.pathsep) if p]
    if not entries or not all(os.path.isabs(p) for p in entries):
        problems.append("PYTHONPATH must be set and absolute (the supervisor passes no cwd or environment)")
    for name in LAUNCH_MODULES:
        try:
            importlib.import_module(name)
        except Exception as error:  # noqa: BLE001 - an import failure is a preflight problem
            problems.append(f"cannot import {name}: {type(error).__name__}: {error}")
    return problems


# ---- The audit hook -------------------------------------------------------

_ACTIVE = None
_INSTALLED = False
_LOCAL = threading.local()      # the recursion flag is per thread: a process-wide flag lets another thread's open escape (N05)


def _audit(event, args):
    guard = _ACTIVE
    if guard is None or event != "open" or getattr(_LOCAL, "busy", False) or not args:
        return
    target = args[0]
    if isinstance(target, int):
        return
    _LOCAL.busy = True
    try:
        guard._see(os.path.realpath(os.fsdecode(target)))
    except (TypeError, ValueError):
        pass
    finally:
        _LOCAL.busy = False


class AuditGuard:
    """Watches file opens (the audit event `open`) once enabled.

    Deny by default over a whole tree (the Group9 data tree, by real path): an open of any file under
    it other than the allowed paths (the four pinned dataset files of the study's data directory and
    the Landscape17 archive) is a violation. The guard first writes a violation record (written
    whole, then renamed into place) and then ends the process with `AUDIT_EXIT_CODE`, so a broad
    `except` cannot swallow it. Opens of the allowed files are listed; opens of data-bearing files
    (.npz, .zip) outside the tree are listed separately and are not violations. The hook sees file
    opens, not which archive members are decoded. A Python audit hook cannot be removed, so `enable`
    installs the process-wide function once and `disable` only switches it off; tests that can end
    the process run in a subprocess.
    """

    def __init__(self, tree_root, allowed_paths, violation_directory, violation_name=None):
        self.data_directory = os.path.realpath(tree_root)
        self.allowed = {os.path.realpath(path) for path in allowed_paths}
        self.violation_directory = Path(violation_directory)
        self.violation_name = violation_name or f"AUDIT_VIOLATION-{os.getpid()}"
        self.opened_in_data_directory, self.opened_elsewhere, self.violations = set(), set(), 0

    def enable(self):
        global _ACTIVE, _INSTALLED
        if not _INSTALLED:
            sys.addaudithook(_audit)
            _INSTALLED = True
        _ACTIVE = self
        return self

    def disable(self):
        global _ACTIVE
        if _ACTIVE is self:
            _ACTIVE = None

    def _inside(self, real):
        return real == self.data_directory or real.startswith(self.data_directory + os.sep)

    def _see(self, real):
        if self._inside(real):
            if real in self.allowed:
                self.opened_in_data_directory.add(real)
                return
            self.violations += 1
            self._violate(real)
        elif real.endswith(DATA_SUFFIXES):
            self.opened_elsewhere.add(real)

    def _violate(self, real):
        try:
            name = self.violation_name
            payload = json.dumps({"violation": "an open under the data tree outside the allowed files",
                                  "path": real, "pid": os.getpid()}).encode() + b"\n"
            os.makedirs(self.violation_directory, exist_ok=True)
            temporary = self.violation_directory / (name + ".partial")
            descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            os.write(descriptor, payload)
            os.fsync(descriptor)
            os.close(descriptor)
            os.rename(temporary, self.violation_directory / (name + ".json"))
        finally:
            os._exit(AUDIT_EXIT_CODE)

    def report(self):
        return {"data_directory": self.data_directory,
                "opened_in_data_directory": sorted(os.path.relpath(p, self.data_directory)
                                                   for p in self.opened_in_data_directory),
                "opened_data_files_elsewhere": sorted(self.opened_elsewhere),
                "violations": self.violations}


def audit_scope(group9=None):
    """(tree root, allowed paths): the Group9 data tree, and inside it exactly the four pinned files of
    the study's data directory and the Landscape17 archive the accepted launch records name. These are
    the only files the existing real-file tests of admission and the study inputs open under the tree."""
    group9 = admission.group9_root(group9)
    tree = group9 / "data"
    directory = admission.data_directory(group9)
    allowed = [directory / name for name in admission.DATASET_FILES]
    record = json.loads((group9 / admission.RECOVERY_RECORDS).read_text())
    allowed.append(admission.relocate(record["archive"], group9))
    return tree, allowed


def allowed_relative(tree, allowed):
    return sorted(os.path.relpath(os.path.realpath(p), os.path.realpath(tree)) for p in allowed)


def required_relative(tree, allowed, unit):
    """The files a process of `unit` must have opened, relative to the tree. A stage opens all five allowed files
    (the four pinned dataset files and the Landscape17 archive); A0 builds no study inputs, so it opens the four
    dataset files only. Measured on the real files: `admit_all` alone opens the four, the study inputs add the archive."""
    everything = allowed_relative(tree, allowed)
    if unit != "A0":
        return everything
    return [path for path in everything if os.path.basename(path) in admission.DATASET_FILES]


def audit_problems(report, required_relative_paths, allowed_relative_paths):
    """The completion decision's audit rule: zero violations, every required file opened, no other file opened
    (required is within opened is within allowed). An empty list is a problem: a hook that never ran must not pass."""
    problems = []
    if not isinstance(report, dict) or report.get("violations") != 0:
        problems.append("the audit log records a violation or is missing")
        return problems
    opened = set(report.get("opened_in_data_directory", []))
    missing = sorted(set(required_relative_paths) - opened)
    if missing:
        problems.append(f"files the process must open were not recorded as opened: {missing}")
    extra = sorted(opened - set(allowed_relative_paths))
    if extra:
        problems.append(f"files opened in the data tree outside the allowed set: {extra}")
    return problems
