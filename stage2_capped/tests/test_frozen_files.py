"""M6: no frozen file changed. The twelve files the accepted evidence depends on keep the SHA-256 they had at the
base commit (recorded at M0); the source pin of the accepted launch records still matches in every required field."""
import hashlib
from pathlib import Path

import pytest

from stage2_capped import admission

ROOT = Path(__file__).resolve().parents[2]

FROZEN = {
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


def sha256(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


@pytest.mark.parametrize("path", sorted(FROZEN))
def test_each_frozen_file_has_its_recorded_hash(path):
    assert sha256(path) == FROZEN[path]


def test_the_frozen_set_is_the_twelve_files_of_the_accepted_run():
    assert len(FROZEN) == 12
    assert {p.split("/")[0] for p in FROZEN} == {"evaluation", "scripts"}
    assert all(len(h) == 64 for h in FROZEN.values())


def test_the_source_pin_matches_the_accepted_launch_pin_in_every_required_field():
    try:
        accepted = admission.accepted_source_pin()
    except admission.AdmissionError:
        pytest.skip("saved Group9 files are absent")
    from e3x_phase_common import source_pin
    report = admission.compare_pins(source_pin(), accepted)
    assert report["problems"] == [] and all(report["required_equal"].values())
