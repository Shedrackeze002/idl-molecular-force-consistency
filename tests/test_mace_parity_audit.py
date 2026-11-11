import json
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pytest

import audit_mace_parity as module


def fixture_inputs():
    ids = ["a"]
    zeros = np.zeros((1, 21, 3), dtype=np.float64)
    return {"record_ids": ids, "expected_energy": np.zeros(1, dtype=np.float64),
            "expected_forces": zeros, "memberships": {"a": "13"},
            "connection_counts": {"13": 1}, "species": ("H",) * 21,
            "frames": {"a": {"positions": zeros[0], "species": ("H",) * 21,
                              "ref_energy": 0.0, "ref_forces": zeros[0]}}}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def approval_identity():
    from mace_loader import CORE_PACKAGES, REQUIRED_SOURCES
    packages = {name: "1" for name in CORE_PACKAGES}
    sources = [{"name": name, "distribution": distribution,
                "relative_path": relative, "actual_sha256": "c" * 64,
                "original_sha256": "c" * 64, "matches_original": True}
               for name, (distribution, relative) in REQUIRED_SOURCES.items()]
    platform = {"python": "3.12.13", "platform": "Linux-test",
                "architecture": "x86_64", "glibc": "2.35"}
    return platform, packages, sources


def write_supervisor_evidence(root, supervisor):
    """Build coherent saved evidence for the synthetic worker fixture."""
    for name in ("stdout.log", "stderr.log"):
        (root / name).touch()
    partial = root / "predictions-partial.npz"
    if not partial.exists():
        partial.write_bytes((root / "predictions.npz").read_bytes())
    status_path = root / "supervisor-status.json"
    other_bytes = sum(p.stat().st_size for p in root.rglob("*")
                      if p.is_file() and p not in {status_path, root / "independent-audit.json"})
    supervisor.update(status="WORKER_COMPLETED", exit_code=0, elapsed_seconds=1,
                      stdout_bytes=0, stderr_bytes=0, output_bytes=other_bytes)
    text = json.dumps(supervisor)
    while supervisor["output_bytes"] != other_bytes + len(text.encode()):
        supervisor["output_bytes"] = other_bytes + len(text.encode())
        text = json.dumps(supervisor)
    status_path.write_text(text)
    observation = {**supervisor, "publication": {
        "receipt_sha256": digest(status_path), "elapsed_seconds": 1,
    }}
    root.with_name(root.name + ".supervisor-observation.json").write_text(json.dumps(observation))


def write_attempt(root, status="PARITY_PASS"):
    root.mkdir()
    np.savez(root / "predictions.npz", record_ids=np.array(["a"]),
             energy=np.zeros(1, dtype=np.float64), forces=np.zeros((1, 21, 3), dtype=np.float64))
    private = root / "private"
    private.mkdir()
    snapshot = private / "final_epoch.model"
    snapshot.write_bytes(b"model")
    packet = {"status": "PACKET_VERIFIED", "manifest_sha256": "manifest",
              "model_sha256": digest(snapshot), "model_bytes": snapshot.stat().st_size}
    from mace_loader import REQUIRED_IMPLEMENTATION_FILES
    from test_mace_loader import accepted_frozen_inventory, write_launch_contract
    platform, packages, sources = approval_identity()
    wsl = {"status": "APPROVED", "identities": {
        "implementation_commit": "a" * 40,
        "implementation_files": {name: {"bytes": 1, "sha256": "b" * 64}
                                 for name in REQUIRED_IMPLEMENTATION_FILES}},
        "platform": platform,
        "installed_packages": [f"{name}=={version}" for name, version in packages.items()],
        "installed_package_versions": packages,
        "authenticated_source_comparison": sources}
    wsl_path = root / "wsl.json"
    wsl_path.write_text(json.dumps(wsl))
    launch = {"status": "APPROVED", "species": ["H"] * 21, "implementation_commit": "a" * 40,
              "wsl_admission_receipt": "wsl.json", "wsl_admission_sha256": digest(wsl_path)}
    execution = write_launch_contract(root, root.parent, launch)
    admission = root / "admission-receipt.json"
    admission.write_text(json.dumps({"status": "ADMITTED", "launch_receipt": launch,
        "wsl_admission": wsl, "packet": packet,
        "live_identity_comparison": {"platform": platform, "packages": packages,
            "frozen_source_files": accepted_frozen_inventory(), "launch_identity": execution,
            "sources": [{"name": item["name"], "path": f"/env/{item['relative_path']}",
                         "sha256": item["actual_sha256"]} for item in sources]},
        "constructor": {"device": "cpu", "torch_threads": 1,
                        "default_dtype": "float64", "enable_cueq": False,
                        "enable_oeq": False, "compile_mode": None},
        "model_snapshot": {"snapshot_path": str(snapshot),
                           "model_sha256": digest(snapshot),
                           "model_bytes": snapshot.stat().st_size, "epoch_index": 2047},
        "post_construction_sha256": digest(snapshot)}))
    parity = {"status": status, "record_count": 1, "force_component_count": 63,
              "max_abs_energy_eV": 0.0, "max_abs_force_eV_per_angstrom": 0.0,
              "per_record_max_energy_eV": [0.0],
              "per_record_max_force_eV_per_angstrom": [0.0],
              "construction_count": 1, "force_requests": 1, "energy_requests": 1,
              "packet": packet, "admission_receipt_sha256": digest(admission),
              "prediction_file_sha256": digest(root / "predictions.npz"),
              "final_model_snapshot": {"path": str(snapshot), "bytes": snapshot.stat().st_size,
                                       "sha256": digest(snapshot)}}
    (root / "parity-status.json").write_text(json.dumps(parity))
    supervisor = json.loads((root / "supervisor-status.json").read_text())
    write_supervisor_evidence(root, supervisor)
    return packet


def test_audit_recomputes_predictions_and_scores_without_model_import(tmp_path, monkeypatch):
    attempt = tmp_path / "attempt"
    packet = write_attempt(attempt)
    monkeypatch.setattr(module, "validate_inputs", lambda root: fixture_inputs())
    monkeypatch.setattr(__import__("mace_loader"), "verify_packet", lambda root: packet)
    result = module.audit_saved_parity(tmp_path, attempt)
    assert result["parity"]["status"] == "PARITY_PASS"
    assert result["score_status"] == "SCORE_NOT_REPRODUCED"


def test_saved_audit_refuses_version_map_only_noncore_pin(tmp_path, monkeypatch):
    attempt = tmp_path / "attempt"
    packet = write_attempt(attempt)
    monkeypatch.setattr(module, "validate_inputs", lambda root: fixture_inputs())
    monkeypatch.setattr(__import__("mace_loader"), "verify_packet", lambda root: packet)
    wsl_path = attempt / "wsl.json"
    wsl = json.loads(wsl_path.read_text())
    wsl["installed_package_versions"]["extra-package"] = "2"
    wsl_path.write_text(json.dumps(wsl))
    launch_path = attempt / "launch-receipt.json"
    launch = json.loads(launch_path.read_text())
    launch["wsl_admission_sha256"] = digest(wsl_path)
    launch_path.write_text(json.dumps(launch))
    admission_path = attempt / "admission-receipt.json"
    admission = json.loads(admission_path.read_text())
    admission.update(wsl_admission=wsl, launch_receipt=launch)
    admission_path.write_text(json.dumps(admission))
    status_path = attempt / "parity-status.json"
    status = json.loads(status_path.read_text())
    status["admission_receipt_sha256"] = digest(admission_path)
    status_path.write_text(json.dumps(status))
    with pytest.raises(ValueError, match="package|inventor"):
        module.audit_saved_parity(tmp_path, attempt)


def test_partial_worker_result_is_refused(tmp_path):
    attempt = tmp_path / "partial"
    write_attempt(attempt, "INCOMPLETE")
    with pytest.raises(ValueError, match="partial"):
        module.audit_saved_parity(tmp_path, attempt)


def test_invented_success_flag_is_refused(tmp_path, monkeypatch):
    attempt = tmp_path / "altered"
    packet = write_attempt(attempt, "PARITY_FAIL")
    monkeypatch.setattr(module, "validate_inputs", lambda root: fixture_inputs())
    monkeypatch.setattr(__import__("mace_loader"), "verify_packet", lambda root: packet)
    with pytest.raises(ValueError, match="claims"):
        module.audit_saved_parity(tmp_path, attempt)


def test_operationally_invalid_timeout_is_refused(tmp_path, monkeypatch):
    attempt = tmp_path / "timeout"
    packet = write_attempt(attempt)
    (attempt / "supervisor-status.json").write_text(json.dumps(
        {"status": "TIMEOUT", "exit_code": -9, "elapsed_seconds": 1801,
         "timeout_seconds": 1800}))
    monkeypatch.setattr(module, "validate_inputs", lambda root: fixture_inputs())
    monkeypatch.setattr(__import__("mace_loader"), "verify_packet", lambda root: packet)
    with pytest.raises(ValueError, match="operational supervisor"):
        module.audit_saved_parity(tmp_path, attempt)


def test_refused_admission_is_not_accepted_by_hash_only(tmp_path, monkeypatch):
    attempt = tmp_path / "refused"
    packet = write_attempt(attempt)
    admission = json.loads((attempt / "admission-receipt.json").read_text())
    admission["status"] = "REFUSED"
    (attempt / "admission-receipt.json").write_text(json.dumps(admission))
    parity = json.loads((attempt / "parity-status.json").read_text())
    parity["admission_receipt_sha256"] = digest(attempt / "admission-receipt.json")
    (attempt / "parity-status.json").write_text(json.dumps(parity))
    monkeypatch.setattr(module, "validate_inputs", lambda root: fixture_inputs())
    monkeypatch.setattr(__import__("mace_loader"), "verify_packet", lambda root: packet)
    with pytest.raises(ValueError, match="not ADMITTED"):
        module.audit_saved_parity(tmp_path, attempt)


def test_prediction_bytes_are_bound_to_worker_receipt(tmp_path, monkeypatch):
    attempt = tmp_path / "changed"
    packet = write_attempt(attempt)
    with np.load(attempt / "predictions.npz") as saved:
        np.savez(attempt / "predictions.npz", record_ids=saved["record_ids"],
                 energy=np.array([1e-12]), forces=saved["forces"])
    monkeypatch.setattr(module, "validate_inputs", lambda root: fixture_inputs())
    monkeypatch.setattr(__import__("mace_loader"), "verify_packet", lambda root: packet)
    with pytest.raises(ValueError, match="prediction bytes"):
        module.audit_saved_parity(tmp_path, attempt)


def test_per_record_claims_are_recomputed(tmp_path, monkeypatch):
    attempt = tmp_path / "per-record"
    packet = write_attempt(attempt)
    parity = json.loads((attempt / "parity-status.json").read_text())
    parity["per_record_max_force_eV_per_angstrom"] = [5e-11]
    (attempt / "parity-status.json").write_text(json.dumps(parity))
    saved_supervisor = json.loads((attempt / "supervisor-status.json").read_text())
    write_supervisor_evidence(attempt, saved_supervisor)
    monkeypatch.setattr(module, "validate_inputs", lambda root: fixture_inputs())
    monkeypatch.setattr(__import__("mace_loader"), "verify_packet", lambda root: packet)
    with pytest.raises(ValueError, match="claims"):
        module.audit_saved_parity(tmp_path, attempt)


@pytest.mark.parametrize("damage", ["platform", "packages", "sources", "constructor",
                                     "packet", "implementation"])
def test_incomplete_or_contradictory_admission_evidence_is_refused(tmp_path, monkeypatch, damage):
    attempt = tmp_path / damage
    packet = write_attempt(attempt)
    admission_path = attempt / "admission-receipt.json"
    admission = json.loads(admission_path.read_text())
    if damage == "platform":
        admission["live_identity_comparison"]["platform"]["python"] = "0.0"
    elif damage == "packages":
        admission["live_identity_comparison"]["packages"].pop("torch")
    elif damage == "sources":
        admission["live_identity_comparison"]["sources"][0]["sha256"] = "d" * 64
    elif damage == "constructor":
        admission["constructor"]["device"] = "cuda"
    elif damage == "packet":
        admission["packet"]["manifest_sha256"] = "invented"
    else:
        wsl_path = attempt / "wsl.json"
        wsl = json.loads(wsl_path.read_text())
        first = next(iter(wsl["identities"]["implementation_files"].values()))
        first["sha256"] = "not-hex"
        wsl_path.write_text(json.dumps(wsl))
        launch = json.loads((attempt / "launch-receipt.json").read_text())
        launch["wsl_admission_sha256"] = digest(wsl_path)
        (attempt / "launch-receipt.json").write_text(json.dumps(launch))
        admission["wsl_admission"] = wsl
        admission["launch_receipt"] = launch
    admission_path.write_text(json.dumps(admission))
    status = json.loads((attempt / "parity-status.json").read_text())
    status["admission_receipt_sha256"] = digest(admission_path)
    (attempt / "parity-status.json").write_text(json.dumps(status))
    monkeypatch.setattr(module, "validate_inputs", lambda root: fixture_inputs())
    monkeypatch.setattr(__import__("mace_loader"), "verify_packet", lambda root: packet)
    with pytest.raises(ValueError, match="identity|comparison|constructor|packet|implementation|source|package"):
        module.audit_saved_parity(tmp_path, attempt)


def test_actual_loader_receipt_flows_through_worker_and_auditor(tmp_path, monkeypatch):
    """Regression for the production receipt schema, with construction faked only."""
    from ase.calculators.calculator import Calculator, all_changes
    import check_mace_parity as worker
    import mace_loader as loader
    from test_mace_loader import isolate_frozen_imports, write_launch_contract, bind_worker_argv
    isolate_frozen_imports(monkeypatch)

    class ZeroCalculator(Calculator):
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {"energy": 0.0,
                            "forces": np.zeros((len(atoms), 3), dtype=np.float64)}

    packet_root = tmp_path / "packet"
    model = packet_root / loader.MODEL_RELATIVE
    model.parent.mkdir(parents=True)
    model.write_bytes(b"synthetic-model")
    packet = {"status": "PACKET_VERIFIED", "manifest_sha256": "manifest",
              "member_count": 11, "members": [], "model_sha256": digest(model),
              "model_bytes": model.stat().st_size}
    attempt = tmp_path / "integrated-attempt"
    attempt.mkdir()
    platform, packages, sources = approval_identity()
    repo = Path(loader.__file__).resolve().parent.parent
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    implementation_files = {name: {"bytes": (repo / name).stat().st_size, "sha256": digest(repo / name)}
                            for name in loader.REQUIRED_IMPLEMENTATION_FILES}
    wsl = {"status": "APPROVED", "identities": {
        "implementation_commit": commit, "implementation_files": implementation_files},
        "platform": platform,
        "installed_packages": [f"{name}=={version}" for name, version in packages.items()],
        "installed_package_versions": packages,
        "authenticated_source_comparison": sources}
    wsl_path = attempt / "wsl.json"
    wsl_path.write_text(json.dumps(wsl))
    launch = {"status": "APPROVED", "species": ["H"] * 21,
              "implementation_commit": commit, "wsl_admission_receipt": "wsl.json",
              "wsl_admission_sha256": digest(wsl_path)}
    write_launch_contract(attempt, packet_root, launch)
    bind_worker_argv(monkeypatch, launch)
    live = {"platform": platform, "packages": packages,
            "sources": [{"name": item["name"], "path": f"/env/{item['relative_path']}",
                         "sha256": item["actual_sha256"]} for item in sources]}

    monkeypatch.setattr(loader, "MODEL_SHA256", digest(model))
    monkeypatch.setattr(loader, "MODEL_BYTES", model.stat().st_size)
    monkeypatch.setattr(loader, "verify_packet", lambda root: packet)
    monkeypatch.setattr(loader, "_live_admission_check", lambda approval: live)
    monkeypatch.setattr(loader, "_construct_calculator", lambda snapshot: ZeroCalculator())
    monkeypatch.setattr(worker, "validate_inputs", lambda root: fixture_inputs())
    result = worker.run_parity(packet_root, attempt)
    assert result["status"] == "PARITY_PASS"
    receipt = json.loads((attempt / "admission-receipt.json").read_text())
    assert receipt["model_snapshot"]["model_sha256"] == digest(model)

    supervisor = json.loads((attempt / "supervisor-status.json").read_text())
    write_supervisor_evidence(attempt, supervisor)
    monkeypatch.setattr(module, "validate_inputs", lambda root: fixture_inputs())
    audited = module.audit_saved_parity(packet_root, attempt)
    assert audited["parity"]["status"] == "PARITY_PASS"


@pytest.mark.parametrize("damage", ["missing", "changed", "extra", "substituted"])
def test_saved_audit_requires_exact_frozen_sources(tmp_path, monkeypatch, damage):
    attempt = tmp_path / "attempt"
    packet = write_attempt(attempt)
    monkeypatch.setattr(module, "validate_inputs", lambda root: fixture_inputs())
    monkeypatch.setattr(__import__("mace_loader"), "verify_packet", lambda root: packet)
    receipt_path = attempt / "admission-receipt.json"
    receipt = json.loads(receipt_path.read_text())
    live = receipt["live_identity_comparison"]
    if damage == "missing":
        del live["frozen_source_files"]
    elif damage == "changed":
        live["frozen_source_files"]["evaluation/adapter.py"]["sha256"] = "0" * 64
    elif damage == "extra":
        live["frozen_source_files"]["extra.py"] = {"bytes": 0, "sha256": "0" * 64}
    else:
        live["frozen_source_files"]["other.py"] = live["frozen_source_files"].pop("evaluation/adapter.py")
    receipt_path.write_text(json.dumps(receipt))
    status_path = attempt / "parity-status.json"
    status = json.loads(status_path.read_text())
    status["admission_receipt_sha256"] = digest(receipt_path)
    status_path.write_text(json.dumps(status))
    with pytest.raises(ValueError, match="frozen"):
        module.audit_saved_parity(tmp_path, attempt)
    assert not (attempt / "independent-audit.json").exists()


@pytest.mark.parametrize("damage", ["command", "resources", "missing-flag", "synthetic", "fresh", "launch-hash", "negative-time", "nan-time", "live-identity", "review-semantic", "size-output", "size-stdout", "size-stderr", "size-bool"])
def test_saved_audit_requires_bound_launch(tmp_path, monkeypatch, damage):
    target = tmp_path / "attempt"; packet = write_attempt(target)
    monkeypatch.setattr(module, "validate_inputs", lambda root: fixture_inputs())
    monkeypatch.setattr(__import__("mace_loader"), "verify_packet", lambda root: packet)
    sp = target / "supervisor-status.json"; supervisor = json.loads(sp.read_text())
    if damage == "command": supervisor["command"][1] = "/other/check_mace_parity.py"
    elif damage == "resources": supervisor["timeout_seconds"] = 3600
    elif damage == "missing-flag": del supervisor["synthetic_command"]
    elif damage == "synthetic": supervisor["synthetic_command"] = True
    elif damage == "fresh": supervisor["fresh_output"] = False
    elif damage == "launch-hash": supervisor["launch_receipt_sha256"] = "0" * 64
    elif damage == "negative-time": supervisor["elapsed_seconds"] = -1
    elif damage == "nan-time": supervisor["elapsed_seconds"] = float("nan")
    elif damage == "size-output": supervisor["output_bytes"] = 268435457
    elif damage == "size-stdout": supervisor["stdout_bytes"] = 16777217
    elif damage == "size-stderr": supervisor["stderr_bytes"] = -1
    elif damage == "size-bool": supervisor["output_bytes"] = True
    else:
        ap=target / "admission-receipt.json"; admission=json.loads(ap.read_text())
        if damage == "live-identity": admission["live_identity_comparison"]["launch_identity"]["python_sha256"] = "0" * 64
        else:
            lp=target / "launch-receipt.json"; launch=json.loads(lp.read_text()); ref=launch["reviews"][0]
            rp=target / ref["path"]; review=json.loads(rp.read_text()); review["unresolved_findings"]=1
            rp.write_text(json.dumps(review)); ref["sha256"]=digest(rp); lp.write_text(json.dumps(launch))
            supervisor["launch_receipt_sha256"]=digest(lp); admission["launch_receipt"]=launch
        ap.write_text(json.dumps(admission)); pp=target / "parity-status.json"; parity=json.loads(pp.read_text()); parity["admission_receipt_sha256"]=digest(ap); pp.write_text(json.dumps(parity))
    sp.write_text(json.dumps(supervisor))
    with pytest.raises(ValueError, match="launch|supervisor|elapsed|review"):
        module.audit_saved_parity(tmp_path, target)


def test_saved_launch_audit_does_not_use_current_interpreter_or_worker(tmp_path, monkeypatch):
    import mace_loader as loader
    target=tmp_path / "attempt"; packet=write_attempt(target)
    monkeypatch.setattr(module, "validate_inputs", lambda root: fixture_inputs())
    monkeypatch.setattr(loader, "verify_packet", lambda root: packet)
    monkeypatch.setattr(loader.sys, "executable", "/unavailable/auditing-host/python")
    monkeypatch.setattr(loader, "__file__", "/unavailable/auditing-host/scripts/mace_loader.py")
    result=module.audit_saved_parity(tmp_path,target)
    assert result["parity"]["status"] == "PARITY_PASS"


@pytest.mark.parametrize("field", ["attempt_root", "species"])
def test_saved_audit_refuses_approval_that_contradicts_evidence(tmp_path, monkeypatch, field):
    from mace_loader import EXECUTION_FIELDS
    from test_mace_loader import refresh_execution_approvals

    attempt = tmp_path / "attempt"
    packet = write_attempt(attempt)
    inputs = fixture_inputs()
    inputs["species"] = ("C", "O", *("H",) * 19)
    launch_path = attempt / "launch-receipt.json"
    admission_path = attempt / "admission-receipt.json"
    status_path = attempt / "parity-status.json"
    launch = json.loads(launch_path.read_text())
    admission = json.loads(admission_path.read_text())
    status = json.loads(status_path.read_text())

    def save_approval():
        refresh_execution_approvals(attempt, launch)
        admission["launch_receipt"] = launch
        admission["live_identity_comparison"]["launch_identity"] = {
            key: launch[key] for key in EXECUTION_FIELDS
        }
        admission_path.write_text(json.dumps(admission))
        status["admission_receipt_sha256"] = digest(admission_path)
        status_path.write_text(json.dumps(status))
        saved_supervisor = json.loads((attempt / "supervisor-status.json").read_text())
        write_supervisor_evidence(attempt, saved_supervisor)

    monkeypatch.setattr(module, "validate_inputs", lambda root: inputs)
    monkeypatch.setattr(__import__("mace_loader"), "verify_packet", lambda root: packet)
    launch["species"] = list(inputs["species"])
    save_approval()
    assert module.audit_saved_parity(tmp_path, attempt)["parity"]["status"] == "PARITY_PASS"

    if field == "attempt_root":
        launch["attempt_root"] = str(tmp_path / "different-attempt")
        launch["command"][-1] = launch["attempt_root"]
        refusal = "approved attempt root differs"
    else:
        launch["species"][:2] = reversed(launch["species"][:2])
        refusal = "approved species order differs"
    save_approval()
    with pytest.raises(ValueError, match=refusal):
        module.audit_saved_parity(tmp_path, attempt)


@pytest.mark.parametrize("damage,reason", [
    ("redirect-snapshot", "symlink|private"),
    ("oversized-log", "log|stdout"),
    ("extra", "inventory|unexpected"),
])
def test_saved_audit_checks_actual_attempt_files(tmp_path, monkeypatch, damage, reason):
    import mace_loader as loader

    attempt = tmp_path / "attempt"
    packet = write_attempt(attempt)
    if damage == "redirect-snapshot":
        moved = (attempt / "private").rename(tmp_path / "redirected")
        (attempt / "private").symlink_to(moved, target_is_directory=True)
    elif damage == "oversized-log":
        with (attempt / "stdout.log").open("wb") as stream:
            stream.truncate(loader.LAUNCH_RESOURCES["log_limit_bytes"] + 1)
    else:
        (attempt / "unexpected.txt").touch()

    monkeypatch.setattr(loader, "verify_packet", lambda root: packet)
    monkeypatch.setattr(module, "validate_inputs", lambda root: fixture_inputs())
    with pytest.raises(ValueError, match=reason):
        module.audit_saved_parity(tmp_path, attempt)


@pytest.mark.parametrize("damage", ["missing", "late", "digest"])
def test_saved_audit_checks_publication_observation(tmp_path, monkeypatch, damage):
    import mace_loader as loader

    attempt = tmp_path / "attempt"
    packet = write_attempt(attempt)
    path = tmp_path / "attempt.supervisor-observation.json"
    if damage == "missing":
        path.unlink()
    else:
        observed = json.loads(path.read_text())
        if damage == "late":
            observed["publication"]["elapsed_seconds"] = 1801
        else:
            observed["publication"]["receipt_sha256"] = "0" * 64
        path.write_text(json.dumps(observed))
    monkeypatch.setattr(loader, "verify_packet", lambda root: packet)
    monkeypatch.setattr(module, "validate_inputs", lambda root: fixture_inputs())
    with pytest.raises(ValueError, match="supervisor publication"):
        module.audit_saved_parity(tmp_path, attempt)
