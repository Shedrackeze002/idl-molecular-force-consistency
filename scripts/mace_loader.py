"""Authenticate the private MACE packet and construct one admitted arm.

This module is intentionally import-safe without Torch or MACE.  Those heavy
dependencies are imported only after packet, launch and platform admission.
"""
from __future__ import annotations

import hashlib
import importlib.util
import importlib.machinery
import importlib.metadata
import json
import os
import platform
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
    "trust/mace_trusted_launcher.py", "tests/test_mace_trusted_launcher.py",
    "scripts/mace_bootstrap.py",
    "scripts/mace_loader.py", "scripts/check_mace_parity.py",
    "scripts/supervise_mace_parity.py", "scripts/audit_mace_parity.py",
    "docs/run-mace-parity.md", "tests/test_mace_loader.py",
    "tests/test_mace_parity.py", "tests/test_mace_parity_supervisor.py",
    "tests/test_mace_parity_audit.py",
}
FROZEN_DEPENDENCY_BLOBS = {
    "scripts/mace_reference_adapter.py": "f6f720e9767d43a2a3cb92e4dcf20b3dd1dcaf30",
    "evaluation/adapter.py": "80b61b5ba280ca7eca1c627fb2e2ce132a970ba7",
}
SOURCE_BASELINE = "3fa5969e5a0c2fdeaebc76a05bace20799715203"
IMPORT_INITIALIZERS = ("mace/__init__.py", "mace/calculators/__init__.py")
RESOURCE_LIMITS = {"seconds": 1800, "attempt_bytes": 256 * 1024 * 1024,
                   "per_log_bytes": 16 * 1024 * 1024, "cleanup_seconds": 10}


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


def _git_blob_bytes(repo_root: Path, relative: str, revision: str) -> bytes:
    """Read immutable commit-tree bytes directly, bypassing all clean filters."""
    try:
        return subprocess.run(
            ["git", "cat-file", "blob", f"{revision}:{relative}"], cwd=repo_root,
            check=True, capture_output=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError) as exc:
        raise AdmissionError(f"cannot authenticate Git object: {relative}") from exc


def _git_blob_id(repo_root: Path, relative: str, revision: str) -> str:
    try:
        value = subprocess.run(
            ["git", "rev-parse", f"{revision}:{relative}"], cwd=repo_root, check=True,
            text=True, capture_output=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise AdmissionError(f"cannot authenticate Git object: {relative}") from exc
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise AdmissionError(f"invalid Git object identity: {relative}")
    return value


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
    if any(not isinstance(value, str) or not value for value in required_platform.values()):
        raise AdmissionError("approved interpreter/platform binding is incomplete")

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
        name = raw_name.strip().lower().replace("_", "-")
        if not name or name in packages or not isinstance(versions.get(name), str):
            raise AdmissionError("approved package inventories are incomplete or contradictory")
        version = versions[name]
        if declared_version is not None and declared_version != version:
            raise AdmissionError(f"approved package inventories disagree: {name}")
        packages[name] = version
    if set(packages) != set(versions):
        raise AdmissionError("approved package declaration and version-map keys differ")
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
    initializers = admission.get("import_initializers")
    if (not isinstance(initializers, dict) or set(initializers) != set(IMPORT_INITIALIZERS)
            or any(not _hex_digest(value) for value in initializers.values())):
        raise AdmissionError("approved MACE initializer pins are missing or malformed")
    return {"platform": required_platform, "packages": packages, "sources": source_map,
            "import_initializers": dict(initializers)}


def _validated_launch_identity(launch: dict[str, Any], attempt_root: Path,
                               packet_root: Path, implementation_commit: str,
                               *, require_live_interpreter: bool) -> None:
    """Require the reviewed attempt, command, resources and review bindings."""
    attempt = Path(attempt_root).resolve()
    packet = Path(packet_root).resolve()
    interpreter = launch.get("python_executable")
    trusted_launcher = launch.get("trusted_launcher")
    trusted_policy = launch.get("trusted_policy")
    if (not isinstance(trusted_launcher, str) or not Path(trusted_launcher).is_absolute()
            or not isinstance(trusted_policy, str) or not Path(trusted_policy).is_absolute()):
        raise AdmissionError("launch does not pin external launcher and policy paths")
    expected_command = [interpreter, "-I", "-S",
                        trusted_launcher, "--entry", "worker", "--policy", trusted_policy]
    if (not isinstance(interpreter, str) or not Path(interpreter).is_absolute()
            or launch.get("command") != expected_command
            or launch.get("packet_root") != str(packet)
            or launch.get("attempt_root") != str(attempt)):
        raise AdmissionError("launch does not bind the exact interpreter, command and paths")
    if require_live_interpreter and Path(interpreter).resolve() != Path(sys.executable).resolve():
        raise AdmissionError("launch interpreter differs from the running worker")
    if launch.get("resource_limits") != RESOURCE_LIMITS:
        raise AdmissionError("launch resource limits differ from the bounded contract")
    if launch.get("source_baseline") != SOURCE_BASELINE:
        raise AdmissionError("launch source baseline differs from the accepted snapshot")
    reviews = launch.get("implementation_reviews")
    if (not isinstance(reviews, list) or len(reviews) < 2
            or any(not isinstance(review, dict) or review.get("status") != "RESOLVED"
                   or review.get("implementation_commit") != implementation_commit
                   or not isinstance(review.get("review_id"), str) or not review["review_id"]
                   or not isinstance(review.get("reviewer"), str) or not review["reviewer"]
                   or not isinstance(review.get("url"), str) or not review["url"]
                   for review in reviews)):
        raise AdmissionError("launch lacks two resolved reviews of the implementation commit")
    if (len({review["review_id"] for review in reviews}) != len(reviews)
            or len({review["reviewer"] for review in reviews}) != len(reviews)
            or len({review["url"] for review in reviews}) != len(reviews)):
        raise AdmissionError("launch implementation reviews are not distinct")


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
    live_inventory: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise AdmissionError("live package distribution lacks a normalized name")
        name = raw_name.strip().lower().replace("_", "-")
        if name in live_inventory and live_inventory[name] != distribution.version:
            raise AdmissionError(f"live package inventory has conflicting entries: {name}")
        live_inventory[name] = distribution.version
    if live_inventory != inventory["packages"]:
        raise AdmissionError("complete live package inventory differs from approval")
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
    distribution_roots: dict[str, str] = {}
    for name, item in inventory["sources"].items():
        expected = item.get("actual_sha256")
        relative = item.get("relative_path")
        dist = importlib.metadata.distribution(item["distribution"])
        distribution_roots[item["distribution"]] = str(Path(dist.locate_file("")).resolve())
        path = Path(dist.locate_file(relative)).resolve()
        if not path.is_file() or path.is_symlink() or _sha256(path) != expected:
            raise AdmissionError(f"live source differs from approval: {name}")
        checked_sources.append({"name": name, "path": str(path), "sha256": expected})
    for distribution in ("torch",):
        dist = importlib.metadata.distribution(distribution)
        distribution_roots[distribution] = str(Path(dist.locate_file("")).resolve())
    initializers = admission.get("import_initializers")
    if not isinstance(initializers, dict) or set(initializers) != set(IMPORT_INITIALIZERS):
        raise AdmissionError("approved MACE import initializer hashes are missing")
    dist = importlib.metadata.distribution("mace-torch")
    for relative in IMPORT_INITIALIZERS:
        path = Path(dist.locate_file(relative))
        if not _hex_digest(initializers[relative]) or _sha256(path) != initializers[relative]:
            raise AdmissionError(f"MACE import initializer differs from approval: {relative}")
    return {"platform": actual_platform, "packages": checked_packages, "sources": checked_sources,
            "distribution_roots": distribution_roots,
            "import_initializers": inventory["import_initializers"]}


def _require_standard_finders():
    allowed = (importlib.machinery.BuiltinImporter, importlib.machinery.FrozenImporter,
               importlib.machinery.PathFinder)
    if any(finder not in allowed for finder in sys.meta_path):
        raise AdmissionError("unexpected Python import finder before construction")


def validate_dependency_imports(names: tuple[str, ...]) -> None:
    """Resolve packages without executing their initializers, including children."""
    distributions = {"mace": "mace-torch", "torch": "torch", "numpy": "numpy", "ase": "ase"}
    for name in names:
        top = name.split(".")[0]
        dist = importlib.metadata.distribution(distributions[top])
        base = Path(dist.locate_file("")).resolve()
        search = None
        for index, part in enumerate(name.split(".")):
            fullname = ".".join(name.split(".")[:index + 1])
            expected_dir = base.joinpath(*fullname.split("."))
            expected_file = expected_dir / "__init__.py"
            if not expected_file.is_file():
                expected_file = expected_dir.with_suffix(".py")
            loaded = sys.modules.get(fullname)
            spec = (getattr(loaded, "__spec__", None) if loaded is not None else
                    importlib.machinery.PathFinder.find_spec(fullname, search))
            origin = getattr(loaded, "__file__", None) if loaded is not None else getattr(spec, "origin", None)
            if not isinstance(origin, str) or Path(origin).resolve() != expected_file.resolve():
                raise AdmissionError(f"module import origin differs from approval: {fullname}")
            locations = (getattr(loaded, "__path__", None) if loaded is not None else
                         getattr(spec, "submodule_search_locations", None))
            if expected_file.name == "__init__.py":
                if locations is None or [Path(p).resolve() for p in locations] != [expected_dir]:
                    raise AdmissionError(f"module package search path differs from approval: {fullname}")
                search = [str(expected_dir)]
            else:
                search = None


def _validate_import_origins(live: dict[str, Any]) -> None:
    """Refuse modules Python would import outside the approved distributions."""
    _require_standard_finders()
    roots = live.get("distribution_roots")
    if not isinstance(roots, dict):
        raise AdmissionError("live admission lacks approved import roots")
    for module_name, distribution in (("mace", "mace-torch"), ("torch", "torch")):
        raw_root = roots.get(distribution)
        if not isinstance(raw_root, str) or not raw_root:
            raise AdmissionError(f"live admission lacks import root: {distribution}")
        approved_root = Path(raw_root).resolve()
        loaded = sys.modules.get(module_name)
        loaded_origin = getattr(loaded, "__file__", None) if loaded is not None else None
        if loaded is not None:
            if not isinstance(loaded_origin, str):
                raise AdmissionError(f"loaded module has no verifiable origin: {module_name}")
            candidate = Path(loaded_origin).resolve()
        else:
            spec = importlib.util.find_spec(module_name)
            if spec is None or not isinstance(spec.origin, str):
                raise AdmissionError(f"cannot resolve approved module import: {module_name}")
            candidate = Path(spec.origin).resolve()
        try:
            candidate.relative_to(approved_root)
        except ValueError as exc:
            raise AdmissionError(f"module import origin differs from approval: {module_name}") from exc
    validate_dependency_imports(("mace.calculators.mace", "torch", "numpy", "ase.io"))


def _approved_launch(attempt_root: Path, packet_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
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
    frozen_dependencies = identities.get("frozen_dependencies")
    if (not isinstance(frozen_dependencies, dict)
            or set(frozen_dependencies) != set(FROZEN_DEPENDENCY_BLOBS)):
        raise AdmissionError("approved frozen bridge/evaluator identities are missing")
    _validated_launch_identity(launch, attempt_root, packet_root, implementation_commit,
                               require_live_interpreter=True)
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
        expected_bytes = _git_blob_bytes(repo_root, relative, implementation_commit)
        expected_blob = _git_blob_id(repo_root, relative, implementation_commit)
        if (not isinstance(receipt, dict) or not path.is_file()
                or receipt.get("git_blob") != expected_blob
                or path.read_bytes() != expected_bytes
                or receipt.get("bytes") != len(expected_bytes)
                or receipt.get("sha256") != hashlib.sha256(expected_bytes).hexdigest()):
            raise AdmissionError(f"execution source differs from approval: {relative}")
    for relative, expected_blob in FROZEN_DEPENDENCY_BLOBS.items():
        receipt = frozen_dependencies[relative]
        path = repo_root / relative
        expected_bytes = _git_blob_bytes(repo_root, relative, SOURCE_BASELINE)
        baseline_blob = _git_blob_id(repo_root, relative, SOURCE_BASELINE)
        if (not isinstance(receipt, dict) or receipt.get("baseline_commit") != SOURCE_BASELINE
                or expected_blob != baseline_blob or receipt.get("git_blob") != baseline_blob
                or not path.is_file() or path.read_bytes() != expected_bytes
                or receipt.get("bytes") != len(expected_bytes)
                or receipt.get("sha256") != hashlib.sha256(expected_bytes).hexdigest()):
            raise AdmissionError(f"frozen bridge/evaluator differs from approval: {relative}")
    return launch, admission, _live_admission_check(admission)


def _construct_calculator(snapshot: Path):
    _require_standard_finders()
    validate_dependency_imports(("mace.calculators.mace", "torch", "numpy", "ase.io"))
    import torch
    import mace
    validate_dependency_imports(("mace.calculators.mace",))
    import mace.calculators
    validate_dependency_imports(("mace.calculators.mace",))
    from mace.calculators.mace import MACECalculator
    validate_dependency_imports(("mace.calculators.mace",))
    if MACECalculator.__module__ != "mace.calculators.mace":
        raise AdmissionError("constructor is not defined in the approved MACE module")

    torch.set_num_threads(1)
    return MACECalculator(model_paths=[str(snapshot)], device="cpu",
                          default_dtype="float64", enable_cueq=False,
                          enable_oeq=False, compile_mode=None)


def _load_admitted_arm(packet_root: Path, attempt_root: Path, *,
                       constructor: Callable[[Path], Any],
                       expected_species: tuple[str, ...] | None = None) -> tuple[dict, dict]:
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
    calculator = constructor(snapshot.resolve())
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


def load_admitted_arm(packet_root: Path, attempt_root: Path,
                      *, expected_species: tuple[str, ...]) -> tuple[dict, dict]:
    """Production entry point; no constructor injection is accepted."""
    return _load_admitted_arm(Path(packet_root), Path(attempt_root),
                              constructor=_construct_calculator,
                              expected_species=expected_species)
