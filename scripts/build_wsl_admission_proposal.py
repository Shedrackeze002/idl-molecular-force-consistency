"""Build the review-only WSL admission proposal without importing MACE or Torch."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_receipt(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def command(*args: str, cwd: Path | None = None) -> str:
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onboarding-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refresh-package-check", action="store_true")
    args = parser.parse_args()

    root = args.onboarding_root.resolve()
    repo = args.repo_root.resolve()
    packet = root / "mace-idris-handoff"
    accepted = load_json(packet / "ACCEPTED_IDENTITY.json")
    manifest = packet / "MANIFEST.json"
    model = packet / "model/final_epoch.model"
    env_python = root / ".venv-mace-wsl/bin/python"
    if args.refresh_package_check:
        check_command = [str(root / "tools/uv"), "pip", "check", "--python", str(env_python)]
        completed = subprocess.run(check_command, cwd=root, text=True, capture_output=True, check=False)
        (root / "package-check-after-tests.command.txt").write_text(
            " ".join(check_command) + "\n", encoding="utf-8")
        (root / "package-check-after-tests.stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (root / "package-check-after-tests.stderr.txt").write_text(completed.stderr, encoding="utf-8")
        (root / "package-check-after-tests.exit.txt").write_text(str(completed.returncode) + "\n", encoding="utf-8")
        if completed.returncode:
            raise SystemExit(completed.returncode)

    probe = r'''
import hashlib, importlib.metadata, json, pathlib, platform
accepted = json.loads(pathlib.Path(__import__('sys').argv[1]).read_text())
rows = []
for name, item in accepted['source_files'].items():
    dist = importlib.metadata.distribution(item['distribution'])
    path = pathlib.Path(dist.locate_file(item['relative_path'])).resolve()
    rows.append({
        'name': name,
        'distribution': item['distribution'],
        'relative_path': item['relative_path'],
        'path': str(path),
        'original_sha256': item['sha256'],
        'actual_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    })
versions = {name: importlib.metadata.version(name) for name in [
    'torch', 'pytest', 'iniconfig', 'pluggy', 'pygments'
]}
all_versions = {}
for dist in importlib.metadata.distributions():
    name = dist.metadata.get('Name')
    if name:
        all_versions[name.lower().replace('_', '-')] = dist.version
print(json.dumps({'python': platform.python_version(), 'platform': platform.platform(),
                  'sources': rows, 'versions': versions, 'all_versions': all_versions}))
'''
    runtime = json.loads(command(str(env_python), "-c", probe, str(packet / "ACCEPTED_IDENTITY.json")))
    for item in runtime["sources"]:
        item["matches_original"] = item["actual_sha256"] == item["original_sha256"]
        item["difference_explanation"] = (
            "No difference; raw installed source bytes match the authenticated Mac reference."
            if item["matches_original"]
            else "Difference requires independent review; not admitted by this proposal."
        )

    inventory_path = root / "environment-after-tests.txt"
    packages = [line for line in inventory_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    versions = {}
    for line in packages:
        if "==" in line:
            key, value = line.split("==", 1)
            versions[key.lower()] = value

    implementation_files = [
        "scripts/mace_loader.py",
        "scripts/check_mace_parity.py",
        "scripts/supervise_mace_parity.py",
        "scripts/audit_mace_parity.py",
        "docs/run-mace-parity.md",
        "tests/test_mace_loader.py",
        "tests/test_mace_parity.py",
        "tests/test_mace_parity_supervisor.py",
        "tests/test_mace_parity_audit.py",
    ]
    commit = command("git", "rev-parse", "HEAD", cwd=repo)
    cpu_model = "unknown"
    for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
        if line.lower().startswith("model name"):
            cpu_model = line.split(":", 1)[1].strip()
            break

    records = {}
    for name in [
        "wsl-requirements.lock",
        "requirements-tests.lock",
        "environment-after-tests.txt",
        "lock-build.txt",
        "package-install.txt",
        "package-check.txt",
        "package-check-after-tests.txt",
        "package-check-after-tests.command.txt",
        "package-check-after-tests.stdout.txt",
        "package-check-after-tests.stderr.txt",
        "package-check-after-tests.exit.txt",
        "packet-verification.txt",
        "mac-reference-comparison.json",
        "mac-reference-comparison-stderr.txt",
        "mac-reference-comparison-exit.txt",
        "wsl-system.txt",
        "source-commit.txt",
    ]:
        path = root / name
        records[name] = file_receipt(path)

    evidence_dir = args.output.parent
    evidence_dir.mkdir(parents=True, exist_ok=True)
    published_names = [
        "requirements-tests.lock", "environment-after-tests.txt",
        "package-check-after-tests.command.txt", "package-check-after-tests.stdout.txt",
        "package-check-after-tests.stderr.txt", "package-check-after-tests.exit.txt",
    ]
    for name in published_names:
        shutil.copyfile(root / name, evidence_dir / name)

    proposal = {
        "schema": 1,
        "status": "PROPOSED",
        "scope": "WSL admission and one bounded validation-only launch amendment; not an approval",
        "prohibitions": [
            "Do not edit or reinterpret the immutable Mac ACCEPTED_IDENTITY.json.",
            "Do not construct the real MACE model or run predictions until independent admission, implementation reviews, and explicit bounded launch authorization are recorded.",
        ],
        "identities": {
            "packet_archive": file_receipt(root / "mace-idris-handoff-2026-10-18-v2.tar.gz"),
            "extracted_manifest": file_receipt(manifest),
            "primary_final_ema_model": file_receipt(model),
            "runtime_lock": file_receipt(root / "wsl-requirements.lock"),
            "test_lock": file_receipt(root / "requirements-tests.lock"),
            "published_test_lock": file_receipt(evidence_dir / "requirements-tests.lock"),
            "implementation_commit": commit,
            "implementation_files": {name: file_receipt(repo / name) for name in implementation_files},
        },
        "platform": {
            "python": runtime["python"],
            "platform": runtime["platform"],
            "architecture": platform.machine(),
            "glibc": platform.libc_ver()[1],
            "cpu_model": cpu_model,
            "cpu_count": os.cpu_count(),
            "system_receipt": records["wsl-system.txt"],
        },
        "original_vs_proposed": {
            "scoring_platform": {
                "original": accepted["accepted_scoring_platform"],
                "proposed": runtime["platform"],
                "reason": "Required Windows/WSL CPU execution environment; requires separate review.",
            },
            "torch": {
                "original": accepted["accepted_versions"]["torch"],
                "proposed": runtime["versions"]["torch"],
                "reason": "Pinned official CPU wheel for WSL; requires separate review.",
            },
            "test_only_additions": {
                key: runtime["versions"][key] for key in ["pytest", "iniconfig", "pluggy", "pygments"]
            },
        },
        "installed_packages": packages,
        "installed_package_versions": runtime["all_versions"],
        "authenticated_source_comparison": runtime["sources"],
        "source_comparison_summary": {
            "count": len(runtime["sources"]),
            "all_match": all(item["matches_original"] for item in runtime["sources"]),
            "method": accepted["source_hash_method"],
        },
        "original_verifier": {
            "status": "FAIL",
            "exit_code": 1,
            "platform_difference_preserved": True,
            "output": load_json(root / "mac-reference-comparison.json"),
        },
        "numerical_amendment": {
            "input_scope": "Same 974 validation records; no training or test access.",
            "calculator": "Primary final EMA; CPU; one Torch thread; float64; cueq/oeq disabled; compile disabled.",
            "requests": {"construction": 1, "force_accessor": 974, "energy_accessor": 974},
            "prediction_tolerance": {"absolute": 1e-10, "relative": 0.0},
            "score_audit": "Four section-6 formulas; each absolute difference <= 1e-10; relative tolerance zero.",
            "resource_limits": {"seconds": 1800, "attempt_bytes": 268435456, "per_log_bytes": 16777216},
            "failure_policy": "Preserve partial outputs, timeout, and errors; no retry, fallback, or substitution.",
        },
        "receipts": records,
        "published_followup_evidence": {
            name: file_receipt(evidence_dir / name) for name in published_names
        },
    }
    args.output.write_text(json.dumps(proposal, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": proposal["status"], "output": str(args.output), "sha256": sha256(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
