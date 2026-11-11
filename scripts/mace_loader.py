"""Authenticate the private MACE packet and construct one admitted arm.

This module is intentionally import-safe without Torch or MACE.  Those heavy
dependencies are imported only after packet, launch and platform admission.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.machinery
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

MANIFEST_SHA256 = "3a872e1886c37b0e069e4ecaaa426bbab6a4a771db7c805f21741ee67e22717c"
MODEL_SHA256 = "fbd128215c0d74f5d9e5d3ccde778763221f987535dcc8ce4f4f8471a5b35407"
MODEL_BYTES = 23_397_415
MODEL_RELATIVE = Path("model/final_epoch.model")
MANIFEST_RELATIVE = Path("MANIFEST.json")
APPROVED_STATUSES = {"APPROVED", "ACCEPTED"}
CORE_PACKAGES = {"ase", "e3nn", "mace-torch", "numpy", "torch", "torch-ema", "torchmetrics"}
REQUIRED_SOURCES = {
    "mace.cli.run_train": ("mace-torch", "mace/cli/run_train.py"),
    "mace.tools": ("mace-torch", "mace/tools/__init__.py"),
    "mace.tools.arg_parser": ("mace-torch", "mace/tools/arg_parser.py"),
    "mace.tools.checkpoint": ("mace-torch", "mace/tools/checkpoint.py"),
    "mace.tools.train": ("mace-torch", "mace/tools/train.py"),
    "mace.tools.utils": ("mace-torch", "mace/tools/utils.py"),
    "torch_ema.ema": ("torch-ema", "torch_ema/ema.py"),
    "mace.calculators.mace": ("mace-torch", "mace/calculators/mace.py"),
}
REQUIRED_IMPLEMENTATION_FILES = {
    "scripts/mace_loader.py", "scripts/check_mace_parity.py",
    "scripts/supervise_mace_parity.py", "scripts/audit_mace_parity.py",
    "docs/run-mace-parity.md", "tests/test_mace_loader.py",
    "tests/test_mace_parity.py", "tests/test_mace_parity_supervisor.py",
    "tests/test_mace_parity_audit.py",
}


# Unchanged dependencies in accepted source snapshot
# 3fa5969e5a0c2fdeaebc76a05bace20799715203.
FROZEN_SOURCE_FILES = {
    "evaluation/__init__.py": {
        "bytes": 220, "sha256": "919da2d40e815006ac70cd1e8862b78db10ac72ed0a32ee8dba375b813ccf18e"},
    "evaluation/adapter.py": {
        "bytes": 14497, "sha256": "640a4afc31e971fa33876cb0b6a022416accd0a3c348a597bbebe62483ba65c5"},
    "scripts/mace_reference_adapter.py": {
        "bytes": 4694, "sha256": "c0aa26e964150817b382b0b4e3f9da7ef6dec1eaa94b0c275ebfd109d37c9759"},
}


LAUNCH_RESOURCES = {"timeout_seconds": 1800, "cleanup_seconds": 10,
                    "output_limit_bytes": 256 * 1024 * 1024,
                    "log_limit_bytes": 16 * 1024 * 1024}
EXECUTION_FIELDS = ("implementation_commit", "python_executable", "python_sha256",
                    "command", "packet_root", "attempt_root", "resources",
                    "wsl_admission_sha256", "species")
_RESERVED_APPROVAL_NAMES = {"launch-receipt.json", "supervisor-status.json", "supervisor-status.json.tmp",
                            "stdout.log",
                            "stderr.log", "admission-receipt.json", "parity-status.json",
                            "independent-audit.json", "predictions.npz", "predictions-partial.npz",
                            "private", "parity-status.json.tmp", "predictions.npz.tmp.npz",
                            "predictions-partial.npz.tmp.npz"}


class AdmissionError(RuntimeError):
    """A stable refusal raised before any real model construction."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _regular_file(root: Path, relative: Path) -> Path:
    if relative.is_absolute() or ".." in relative.parts:
        raise AdmissionError(f"unsafe packet path: {relative}")
    path = root / relative
    if path.is_symlink() or not path.is_file():
        raise AdmissionError(f"missing or non-regular packet member: {relative}")
    resolved_root = root.resolve()
    try:
        path.resolve().relative_to(resolved_root)
    except ValueError as exc:
        raise AdmissionError(f"packet member escapes root: {relative}") from exc
    return path


def _manifest_members(payload: Any) -> list[dict[str, Any]]:
    raw = payload.get("files", payload.get("members")) if isinstance(payload, dict) else None
    if isinstance(raw, dict):
        return [dict(value, path=key) if isinstance(value, dict) else {}
                for key, value in raw.items()]
    if isinstance(raw, list):
        return [entry for entry in raw if isinstance(entry, dict)]
    raise AdmissionError("manifest has no files/members inventory")


def verify_packet(packet_root: Path) -> dict[str, Any]:
    """Verify pinned manifest bytes, every member, and the primary model."""
    root = Path(packet_root)
    manifest_path = _regular_file(root, MANIFEST_RELATIVE)
    actual_manifest = _sha256(manifest_path)
    if actual_manifest != MANIFEST_SHA256:
        raise AdmissionError("manifest SHA-256 differs from the accepted identity")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdmissionError("manifest is not valid UTF-8 JSON") from exc
    members = _manifest_members(manifest)
    if len(members) != 11:
        raise AdmissionError(f"manifest must inventory 11 members, found {len(members)}")
    seen: set[str] = set()
    verified: list[dict[str, Any]] = []
    for entry in members:
        name = entry.get("path", entry.get("name"))
        digest = entry.get("sha256")
        size = entry.get("size", entry.get("bytes"))
        if not isinstance(name, str) or not name or name in seen:
            raise AdmissionError("manifest contains a missing or duplicate member path")
        if not isinstance(digest, str) or len(digest) != 64 or not isinstance(size, int):
            raise AdmissionError(f"manifest member metadata is malformed: {name!r}")
        seen.add(name)
        path = _regular_file(root, Path(name))
        if path.stat().st_size != size or _sha256(path) != digest.lower():
            raise AdmissionError(f"packet member identity mismatch: {name}")
        verified.append({"path": name, "bytes": size, "sha256": digest.lower()})
    model = _regular_file(root, MODEL_RELATIVE)
    if model.stat().st_size != MODEL_BYTES or _sha256(model) != MODEL_SHA256:
        raise AdmissionError("primary final EMA model identity mismatch")
    return {"status": "PACKET_VERIFIED", "manifest_sha256": actual_manifest,
            "member_count": len(verified), "members": verified,
            "model_sha256": MODEL_SHA256, "model_bytes": MODEL_BYTES}


def _hex_digest(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value.lower()))


def _approved_identity_inventory(admission: dict[str, Any]) -> dict[str, Any]:
    """Validate the complete recorded approval without consulting this host."""
    approved_platform = admission.get("platform")
    sources = admission.get("authenticated_source_comparison")
    declarations = admission.get("installed_packages")
    versions = admission.get("installed_package_versions")
    if (not isinstance(approved_platform, dict) or not isinstance(sources, list)
            or not isinstance(declarations, list) or not isinstance(versions, dict)):
        raise AdmissionError("approved WSL receipt lacks complete live identity fields")
    required_platform = {key: approved_platform.get(key)
                         for key in ("python", "platform", "architecture", "glibc")}
    platform_values = (required_platform["python"], required_platform["platform"],
                       required_platform["architecture"])
    if any(not isinstance(value, str) or not value for value in platform_values):
        raise AdmissionError("approved interpreter/platform binding is incomplete")
    macos_arm64 = (required_platform["platform"].startswith("macOS-")
                   and required_platform["architecture"] == "arm64")
    glibc = required_platform["glibc"]
    if not isinstance(glibc, str) or (glibc == "") != macos_arm64:
        raise AdmissionError("approved interpreter/platform binding is incomplete")

    def package_name(raw: Any) -> str:
        if (not isinstance(raw, str)
                or re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?", raw.strip()) is None):
            raise AdmissionError("approved package inventory has a malformed name")
        return re.sub(r"[-_.]+", "-", raw.strip().lower())

    normalized_versions: dict[str, str] = {}
    for raw_name, version in versions.items():
        name = package_name(raw_name)
        if (name in normalized_versions or not isinstance(version, str)
                or not version.strip() or version != version.strip()):
            raise AdmissionError("approved package version inventory is malformed or duplicated")
        normalized_versions[name] = version

    packages: dict[str, str] = {}
    for declaration in declarations:
        if not isinstance(declaration, str) or not declaration.strip():
            raise AdmissionError("approved package inventory is malformed")
        if "==" in declaration:
            raw_name, declared_version = declaration.split("==", 1)
        elif " @ " in declaration:
            raw_name, declared_version = declaration.split(" @ ", 1)[0], None
        else:
            raise AdmissionError(f"unsupported approved package declaration: {declaration}")
        name = package_name(raw_name)
        if name in packages or name not in normalized_versions:
            raise AdmissionError("approved package inventories are incomplete or contradictory")
        version = normalized_versions[name]
        if declared_version is not None and declared_version != version:
            raise AdmissionError(f"approved package inventories disagree: {name}")
        packages[name] = version
    if set(packages) != set(normalized_versions):
        raise AdmissionError("approved package inventories have different membership")
    if not CORE_PACKAGES.issubset(packages):
        raise AdmissionError("approved package inventory omits a mandatory core package")

    source_map: dict[str, dict[str, Any]] = {}
    for item in sources:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            raise AdmissionError("approved source inventory is malformed")
        name = item["name"]
        if name in source_map or name not in REQUIRED_SOURCES:
            raise AdmissionError("approved source inventory contains a duplicate or substituted source")
        distribution, relative = REQUIRED_SOURCES[name]
        if item.get("distribution") != distribution or item.get("relative_path") != relative:
            raise AdmissionError(f"approved source identity is noncanonical: {name}")
        if (not _hex_digest(item.get("actual_sha256"))
                or item.get("original_sha256") != item.get("actual_sha256")
                or item.get("matches_original") is not True):
            raise AdmissionError(f"approved source digest is incomplete or contradictory: {name}")
        source_map[name] = item
    if set(source_map) != set(REQUIRED_SOURCES):
        raise AdmissionError("approved source inventory is incomplete")
    return {"platform": required_platform, "packages": packages, "sources": source_map}


def _live_admission_check(admission: dict[str, Any]) -> dict[str, Any]:
    """Compare the running interpreter, packages and raw sources to approval."""
    inventory = _approved_identity_inventory(admission)
    actual_platform = {
        "python": platform.python_version(), "platform": platform.platform(),
        "architecture": platform.machine(), "glibc": platform.libc_ver()[1],
    }
    for key, actual in actual_platform.items():
        if inventory["platform"].get(key) != actual:
            raise AdmissionError(f"live platform differs from approved {key}")
    checked_packages: dict[str, str] = {}
    for name, expected in inventory["packages"].items():
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise AdmissionError(f"approved package is not installed: {name}") from exc
        if actual != expected:
            raise AdmissionError(f"live package differs from approval: {name}")
        checked_packages[name] = actual
    proposed_torch = admission.get("original_vs_proposed", {}).get("torch", {}).get("proposed")
    if proposed_torch is not None and checked_packages.get("torch") != proposed_torch:
        raise AdmissionError("live Torch version differs from approval")
    checked_sources = []
    for name, item in inventory["sources"].items():
        expected = item.get("actual_sha256")
        relative = item.get("relative_path")
        dist = importlib.metadata.distribution(item["distribution"])
        path = Path(dist.locate_file(relative)).resolve()
        if not path.is_file() or path.is_symlink() or _sha256(path) != expected:
            raise AdmissionError(f"live source differs from approval: {name}")
        checked_sources.append({"name": name, "path": str(path), "sha256": expected})
    return {"platform": actual_platform, "packages": checked_packages, "sources": checked_sources}



def _launch_approval_refs(launch: dict[str, Any]) -> list[dict[str, str]]:
    """Only distinct local approval files may be copied into the fresh attempt."""
    plan, reviews = launch.get("plan"), launch.get("reviews")
    if not isinstance(plan, dict) or not isinstance(reviews, list) or len(reviews) != 2:
        raise AdmissionError("launch requires a plan and two implementation reviews")
    refs = [{"path": launch.get("wsl_admission_receipt"),
             "sha256": launch.get("wsl_admission_sha256")}, plan, *reviews]
    seen = set()
    for ref in refs:
        if (not isinstance(ref, dict) or set(ref) != {"path", "sha256"}
                or not isinstance(ref["path"], str) or not ref["path"]
                or Path(ref["path"]).name != ref["path"] or ref["path"] in {".", ".."}
                or ref["path"].casefold() in seen or ref["path"].casefold() in _RESERVED_APPROVAL_NAMES
                or not _hex_digest(ref["sha256"])):
            raise AdmissionError("launch approval references are malformed, reserved or duplicated")
        seen.add(ref["path"].casefold())
    return refs


def _validate_launch_binding(launch: dict[str, Any], attempt_root: Path) -> dict[str, Any]:
    """Validate retained execution/approval facts without consulting this host."""
    try:
        if (type(launch.get("launch_schema")) is not int or launch["launch_schema"] != 1
                or launch.get("status") not in APPROVED_STATUSES):
            raise AdmissionError("launch schema or approval is incomplete")
        execution = {key: launch[key] for key in EXECUTION_FIELDS}
        if re.fullmatch(r"[0-9a-f]{40}", execution["implementation_commit"]) is None:
            raise AdmissionError("launch implementation identity is malformed")
        for key in ("python_executable", "packet_root", "attempt_root"):
            value = execution[key]
            if (not isinstance(value, str) or not Path(value).is_absolute()
                    or str(Path(value)) != value or ".." in Path(value).parts):
                raise AdmissionError(f"launch {key} must be a canonical absolute path")
        if not _hex_digest(execution["python_sha256"]):
            raise AdmissionError("launch executable digest is missing")
        command = execution["command"]
        if (not isinstance(command, list) or len(command) != 6
                or not all(isinstance(value, str) for value in command)
                or not Path(command[1]).is_absolute() or str(Path(command[1])) != command[1]
                or ".." in Path(command[1]).parts or Path(command[1]).name != "check_mace_parity.py"
                or command != [execution["python_executable"], command[1], "--packet-root",
                               execution["packet_root"], "--attempt-root", execution["attempt_root"]]):
            raise AdmissionError("launch command differs from the bounded worker contract")
        resources = execution["resources"]
        if (not isinstance(resources, dict) or resources != LAUNCH_RESOURCES
                or any(type(value) is not int for value in resources.values())):
            raise AdmissionError("launch resources differ from the fixed bounds")
        species = execution["species"]
        if not isinstance(species, list) or len(species) != 21 or not all(isinstance(x, str) for x in species):
            raise AdmissionError("launch species identity is malformed")
        refs = _launch_approval_refs(launch)
        documents = {}
        for ref in refs:
            path = _regular_file(attempt_root, Path(ref["path"]))
            if _sha256(path) != ref["sha256"]:
                raise AdmissionError(f"launch approval file digest differs: {ref['path']}")
            documents[ref["path"]] = json.loads(path.read_text(encoding="utf-8"))
        plan = documents[launch["plan"]["path"]]
        if plan != execution:
            changed = [key for key in EXECUTION_FIELDS if not isinstance(plan, dict) or plan.get(key) != execution[key]]
            raise AdmissionError(f"launch plan differs from execution identity: {changed}")
        reviewers = set()
        for ref in launch["reviews"]:
            review = documents[ref["path"]]
            reviewer = review.get("reviewer_id") if isinstance(review, dict) else None
            if (not isinstance(reviewer, str) or not reviewer.strip() or reviewer != reviewer.strip()
                    or reviewer in reviewers or review.get("status") not in APPROVED_STATUSES
                    or review.get("implementation_commit") != execution["implementation_commit"]
                    or review.get("plan_sha256") != launch["plan"]["sha256"]
                    or type(review.get("unresolved_findings")) is not int or review["unresolved_findings"] != 0):
                raise AdmissionError("launch implementation review is unresolved, duplicated or not bound")
            reviewers.add(reviewer)
        return execution
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise AdmissionError(f"launch binding is malformed: {exc}") from exc


def _validate_live_invocation(execution: dict[str, Any], packet_root: Path,
                              attempt_root: Path, *, worker: bool) -> None:
    expected = [sys.executable, str(Path(__file__).resolve().with_name("check_mace_parity.py")),
                "--packet-root", str(packet_root.resolve()), "--attempt-root", str(attempt_root.resolve())]
    if (execution["command"] != expected or execution["python_executable"] != sys.executable
            or execution["python_sha256"] != _sha256(Path(sys.executable))):
        raise AdmissionError("launch interpreter, command or roots differ from live invocation")
    if worker and (getattr(sys, "orig_argv", None) != expected or sys.argv != expected[1:]):
        raise AdmissionError("launch argv or interpreter options differ from the approved invocation")


def _validate_supervisor_binding(supervisor: dict[str, Any], execution: dict[str, Any],
                                 launch_sha256: str, *, running: bool) -> None:
    if (supervisor.get("status") != ("RUNNING" if running else "WORKER_COMPLETED")
            or supervisor.get("fresh_output") is not True
            or supervisor.get("synthetic_command") is not False
            or supervisor.get("launch_receipt_sha256") != launch_sha256
            or any(supervisor.get(key) != execution[key] for key in ("command", "packet_root", "attempt_root"))
            or any(type(supervisor.get(key)) is not int or supervisor[key] != value
                   for key, value in LAUNCH_RESOURCES.items())):
        raise AdmissionError("launch supervisor binding is missing or contradictory")
    if not running:
        for key, limit in (("output_bytes", LAUNCH_RESOURCES["output_limit_bytes"]),
                           ("stdout_bytes", LAUNCH_RESOURCES["log_limit_bytes"]),
                           ("stderr_bytes", LAUNCH_RESOURCES["log_limit_bytes"])):
            value = supervisor.get(key)
            if type(value) is not int or not 0 <= value <= limit:
                raise AdmissionError(f"launch supervisor observed byte count is invalid: {key}")


def _live_launch_context(launch: dict[str, Any], attempt_root: Path,
                         packet_root: Path) -> dict[str, Any]:
    execution = _validate_launch_binding(launch, attempt_root)
    _validate_live_invocation(execution, packet_root, attempt_root, worker=True)
    supervisor = json.loads(_regular_file(attempt_root, Path("supervisor-status.json")).read_text())
    _validate_supervisor_binding(supervisor, execution,
                                 _sha256(attempt_root / "launch-receipt.json"), running=True)
    allowed = {ref["path"] for ref in _launch_approval_refs(launch)} | {
        "launch-receipt.json", "supervisor-status.json", "stdout.log", "stderr.log"}
    if any(path.name not in allowed or path.is_symlink() or not path.is_file()
           for path in attempt_root.iterdir()):
        raise AdmissionError("launch attempt contains prior results or unexpected members")
    return execution


def _approved_launch(attempt_root: Path, *, packet_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    launch_path = _regular_file(attempt_root, Path("launch-receipt.json"))
    try:
        launch = json.loads(launch_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AdmissionError("launch receipt is not valid JSON") from exc
    if launch.get("status") not in APPROVED_STATUSES:
        raise AdmissionError("bounded launch is not explicitly approved")
    admission_name = launch.get("wsl_admission_receipt")
    admission_hash = launch.get("wsl_admission_sha256")
    if not isinstance(admission_name, str) or not isinstance(admission_hash, str):
        raise AdmissionError("launch receipt does not pin the WSL admission receipt")
    admission_path = _regular_file(attempt_root, Path(admission_name))
    if _sha256(admission_path) != admission_hash:
        raise AdmissionError("WSL admission receipt hash differs from launch record")
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    if admission.get("status") not in APPROVED_STATUSES:
        raise AdmissionError("WSL platform admission is not approved")
    identities = admission.get("identities", {})
    implementation_commit = identities.get("implementation_commit")
    if (not isinstance(implementation_commit, str) or len(implementation_commit) != 40
            or any(character not in "0123456789abcdef" for character in implementation_commit.lower())
            or launch.get("implementation_commit") != implementation_commit):
        raise AdmissionError("launch does not pin the approved implementation commit")
    implementation_files = identities.get("implementation_files")
    if (not isinstance(implementation_files, dict)
            or set(implementation_files) != REQUIRED_IMPLEMENTATION_FILES):
        raise AdmissionError("approved implementation file identities are missing")
    repo_root = Path(__file__).resolve().parent.parent
    try:
        live_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, check=True,
            text=True, capture_output=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise AdmissionError("cannot verify the live implementation revision") from exc
    if live_commit != implementation_commit:
        raise AdmissionError("live implementation revision differs from approval")
    for relative, receipt in implementation_files.items():
        path = repo_root / relative
        if (not isinstance(receipt, dict) or not path.is_file()
                or path.stat().st_size != receipt.get("bytes") or _sha256(path) != receipt.get("sha256")):
            raise AdmissionError(f"execution source differs from approval: {relative}")
    live = _live_admission_check(admission)
    live["launch_identity"] = _live_launch_context(launch, attempt_root, packet_root)
    return launch, admission, live



def _authenticated_bridge() -> tuple[Callable, dict[str, Any]]:
    """Execute only authenticated frozen source in the fresh parity worker.

    Ordinary imports can read stale bytecode despite correct source hashes.
    Compile the verified byte buffers explicitly; never read their .pyc files.
    Preloaded modules have unknown execution history and are refused.
    """
    root = Path(__file__).resolve().parent.parent
    modules = (("evaluation", "evaluation/__init__.py"),
               ("evaluation.adapter", "evaluation/adapter.py"),
               ("mace_reference_adapter", "scripts/mace_reference_adapter.py"))
    source_bytes = {}
    inserted = []
    try:
        for relative, expected in FROZEN_SOURCE_FILES.items():
            path = root / relative
            if any(part.is_symlink() for part in (path, *path.parents) if part != root):
                raise AdmissionError(f"frozen source uses a symlink: {relative}")
            if not path.is_file():
                raise AdmissionError(f"frozen source is missing or nonregular: {relative}")
            data = path.read_bytes()
            if len(data) != expected["bytes"] or hashlib.sha256(data).hexdigest() != expected["sha256"]:
                raise AdmissionError(f"frozen source differs from accepted baseline: {relative}")
            source_bytes[relative] = data
        if any(name in sys.modules for name, _ in modules):
            raise AdmissionError("frozen modules preloaded; use a fresh parity worker")
        for name, relative in modules:
            path = root / relative
            spec = importlib.util.find_spec(name)
            if (spec is None or not isinstance(spec.loader, importlib.machinery.SourceFileLoader)
                    or not isinstance(spec.origin, str) or Path(spec.origin).resolve() != path):
                raise AdmissionError(f"frozen module import origin differs: {name}")
            module = importlib.util.module_from_spec(spec)
            sys.modules[name] = module
            inserted.append(name)
            exec(compile(source_bytes[relative], str(path), "exec", dont_inherit=True), module.__dict__)
            if (Path(module.__file__).resolve() != path
                    or module.__spec__ is None or module.__spec__.origin != spec.origin):
                raise AdmissionError(f"frozen module execution origin differs: {name}")
            if "." in name:
                parent, child = name.rsplit(".", 1)
                setattr(sys.modules[parent], child, module)
        for relative, data in source_bytes.items():
            if (root / relative).read_bytes() != data:
                raise AdmissionError(f"frozen source changed during import: {relative}")
        bridge = sys.modules["mace_reference_adapter"].calculator_arm
        return bridge, {name: dict(value) for name, value in FROZEN_SOURCE_FILES.items()}
    except BaseException as exc:
        for name in reversed(inserted):
            if "." in name:
                parent, child = name.rsplit(".", 1)
                if parent in sys.modules:
                    vars(sys.modules[parent]).pop(child, None)
            sys.modules.pop(name, None)
        if isinstance(exc, AdmissionError) or not isinstance(exc, Exception):
            raise
        raise AdmissionError(f"frozen source authentication failed: {exc}") from exc


def _construct_calculator(snapshot: Path):
    import torch
    from mace.calculators import MACECalculator

    torch.set_num_threads(1)
    return MACECalculator(model_paths=[str(snapshot)], device="cpu",
                          default_dtype="float64", enable_cueq=False,
                          enable_oeq=False, compile_mode=None)


def _load_admitted_arm(packet_root: Path, attempt_root: Path, *,
                       constructor: Callable[[Path], Any],
                       expected_species: tuple[str, ...] | None = None) -> tuple[dict, dict]:
    packet = verify_packet(packet_root)
    launch, admission, live = _approved_launch(Path(attempt_root), packet_root=Path(packet_root))
    calculator_arm, frozen_sources = _authenticated_bridge()
    live = {**live, "frozen_source_files": frozen_sources}
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
    calculator = constructor(snapshot.resolve())
    after = _sha256(snapshot)
    if after != before:
        raise AdmissionError("model snapshot changed during construction")
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


def load_admitted_arm(packet_root: Path, attempt_root: Path,
                      *, expected_species: tuple[str, ...]) -> tuple[dict, dict]:
    """Production entry point; no constructor injection is accepted."""
    return _load_admitted_arm(Path(packet_root), Path(attempt_root),
                              constructor=_construct_calculator,
                              expected_species=expected_species)
