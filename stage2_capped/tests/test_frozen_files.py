"""M6: no frozen file changed. The twelve files the accepted evidence depends on keep the SHA-256 they had at the
base commit (recorded at M0); the source pin of the accepted launch records still matches in every required field."""
import hashlib
from pathlib import Path

import pytest

from stage2_capped import admission, launch_checks

ROOT = Path(__file__).resolve().parents[2]

FROZEN = launch_checks.FROZEN_FILES      # the dict moved to non-test code so a launch can check it at run time


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
