"""Authenticate the private MACE packet and construct one admitted arm.

This module is intentionally import-safe without Torch or MACE.  Those heavy
dependencies are imported only after packet, launch and platform admission.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
from pathlib import Path
from typing import Any, Callable

MANIFEST_SHA256 = "3a872e1886c37b0e069e4ecaaa426bbab6a4a771db7c805f21741ee67e22717c"
MODEL_SHA256 = "fbd128215c0d74f5d9e5d3ccde778763221f987535dcc8ce4f4f8471a5b35407"
MODEL_BYTES = 23_397_415
MODEL_RELATIVE = Path("model/final_epoch.model")
MANIFEST_RELATIVE = Path("MANIFEST.json")
APPROVED_STATUSES = {"APPROVED", "ACCEPTED"}


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


def _live_admission_check(admission: dict[str, Any]) -> dict[str, Any]:
    """Compare the running interpreter, packages and raw sources to approval."""
    approved_platform = admission.get("platform")
    sources = admission.get("authenticated_source_comparison")
    packages = admission.get("installed_packages")
    approved_versions = admission.get("installed_package_versions")
    if (not isinstance(approved_platform, dict) or not isinstance(sources, list)
            or not isinstance(packages, list) or not isinstance(approved_versions, dict)):
        raise AdmissionError("approved WSL receipt lacks complete live identity fields")
    actual_platform = {
        "python": platform.python_version(), "platform": platform.platform(),
        "architecture": platform.machine(), "glibc": platform.libc_ver()[1],
    }
    for key, actual in actual_platform.items():
        if approved_platform.get(key) != actual:
            raise AdmissionError(f"live platform differs from approved {key}")
    checked_packages: dict[str, str] = {}
    for declaration in packages:
        if not isinstance(declaration, str) or not declaration.strip():
            raise AdmissionError("approved package inventory is malformed")
        if "==" in declaration:
            name, expected = declaration.split("==", 1)
        elif " @ " in declaration:
            name, expected = declaration.split(" @ ", 1)[0], None
        else:
            raise AdmissionError(f"unsupported approved package declaration: {declaration}")
        try:
            actual = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError as exc:
            raise AdmissionError(f"approved package is not installed: {name}") from exc
        if expected is not None and actual != expected:
            raise AdmissionError(f"live package differs from approval: {name}")
        checked_packages[name] = actual
        normalized = name.lower().replace("_", "-")
        if approved_versions.get(normalized) != actual:
            raise AdmissionError(f"live package version differs from approval: {name}")
    proposed_torch = admission.get("original_vs_proposed", {}).get("torch", {}).get("proposed")
    if proposed_torch is not None and checked_packages.get("torch") != proposed_torch:
        raise AdmissionError("live Torch version differs from approval")
    checked_sources = []
    for item in sources:
        if not isinstance(item, dict) or not isinstance(item.get("distribution"), str):
            raise AdmissionError("approved source inventory is malformed")
        expected = item.get("actual_sha256")
        relative = item.get("relative_path")
        if not isinstance(expected, str) or len(expected) != 64 or not isinstance(relative, str):
            raise AdmissionError("approved source identity is malformed")
        dist = importlib.metadata.distribution(item["distribution"])
        path = Path(dist.locate_file(relative)).resolve()
        if not path.is_file() or path.is_symlink() or _sha256(path) != expected:
            raise AdmissionError(f"live source differs from approval: {item.get('name', relative)}")
        checked_sources.append({"name": item.get("name"), "path": str(path), "sha256": expected})
    return {"platform": actual_platform, "packages": checked_packages, "sources": checked_sources}


def _approved_launch(attempt_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
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
    if (not isinstance(implementation_commit, str)
            or launch.get("implementation_commit") != implementation_commit):
        raise AdmissionError("launch does not pin the approved implementation commit")
    implementation_files = identities.get("implementation_files")
    if not isinstance(implementation_files, dict):
        raise AdmissionError("approved implementation file identities are missing")
    repo_root = Path(__file__).resolve().parent.parent
    for relative, receipt in implementation_files.items():
        path = repo_root / relative
        if (not isinstance(receipt, dict) or not path.is_file()
                or path.stat().st_size != receipt.get("bytes") or _sha256(path) != receipt.get("sha256")):
            raise AdmissionError(f"execution source differs from approval: {relative}")
    return launch, admission, _live_admission_check(admission)


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
    launch, admission, live = _approved_launch(Path(attempt_root))
    species = launch.get("species")
    from ase.data import atomic_numbers
    if (not isinstance(species, list) or len(species) != 21
            or any(not isinstance(x, str) or x not in atomic_numbers for x in species)):
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
