"""Shared fixtures for the shape-recovery evaluator tests.

This file is new. No existing test file or frozen artifact is modified.

Two kinds of fixtures live here:

- Real-input fixtures that read the public Landscape17 archive and the
  frozen split manifest once per pytest session. Set LANDSCAPE17_ARCHIVE to
  point at the zip when it is not checked out under data/.
- A synthetic mini-landscape (two minima, five connections across the three
  partitions, five stored frames per branch) used wherever the rule
  mechanics can be tested without the archive.
"""
import copy
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

# The evaluation package lives at the repository root while pyproject.toml
# pins pytest pythonpath to src/ and tests/ only and stays frozen. Adding
# the repo root to sys.path here keeps the frozen pyproject untouched.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from force_consistency.data.splits import canonical_hash
from force_consistency.data.types import PreparationPolicy


def _archive_path():
    override = os.environ.get("LANDSCAPE17_ARCHIVE")
    if override:
        return Path(override)
    return REPO / "data" / "Landscape17.zip"


@pytest.fixture(scope="session")
def aspirin_archive_path():
    return _archive_path()


@pytest.fixture(scope="session")
def aspirin_records(aspirin_archive_path):
    from force_consistency.data.xyz import read_records

    path = aspirin_archive_path
    if not path.exists():
        # The archive is a separately retained artifact, not a checkout
        # requirement. A contributor without it still gets a green run:
        # the real-data tests SKIP with the reason below instead of
        # failing at fixture setup, and everything synthetic runs.
        pytest.skip(
            f"Landscape17 archive not found at {path}; set LANDSCAPE17_ARCHIVE "
            f"to the frozen Landscape17.zip to run the real-data tests")
    return read_records(path, molecules=("aspirin",))


@pytest.fixture(scope="session")
def split_manifest():
    return json.loads((REPO / "manifests" / "landscape17-aspirin-split.json").read_text())


@pytest.fixture(scope="session")
def aspirin_start_set_primary(aspirin_records, split_manifest):
    # Built once per session; every real-data test reuses the same frozen
    # start set instead of paying for the symmetry-aware alignment again.
    from evaluation.start_set import build_start_set

    policy = PreparationPolicy(**split_manifest["policy"])
    return build_start_set(split_manifest, aspirin_records, policy, {"kind": "first_stored_frame"})


# ---------------------------------------------------------------------------
# Synthetic mini-landscape
# ---------------------------------------------------------------------------

# One shared three-atom graph (C, H, O) everywhere, so the symmetry-aware
# alignment has exactly one species-respecting mapping and stays well
# defined and cheap.
BASE_POSITIONS = [[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.0, 1.3, 0.0]]
# Nonrigid separation between the two minima: each atom moves by its own
# offset, so no rigid motion maps minimum 1 onto minimum 0, while the pair
# stays far above the 1e-5 reference tolerance (they are distinct equivalence
# classes) and close enough that a midway structure is within the 0.05
# endpoint tolerance of BOTH minima, which is what the ambiguous outcome
# test needs. RMSD between the minima is about 0.0624 angstrom.
MINIMUM_SEPARATION = [[0.06, 0.0, 0.0], [-0.06, 0.0, 0.0], [0.06, 0.03, 0.0]]

SPECIES = ("C", "H", "O")

# Branch frames run from near the transition state to near the branch's own
# minimum. frame_k = minimum + beta_k * (transition_state - minimum), so the
# RMSD to the minimum shrinks geometrically with beta along the order.
# Default ladder: five frames per branch.
DEFAULT_LADDER = (0.95, 0.8, 0.6, 0.3, 0.08)
# Two branches use other ladders so the sensitivity rule demonstrably picks
# a middle frame rather than the first or the last stored frame.
# Frame k sits at minimum + beta * (transition_state - minimum), so the RMSD
# to the branch minimum is beta times the TS-to-minimum RMSD. The sensitivity
# rule with fraction 0.10 therefore fires at the first frame with
# beta <= 0.10 / 0.90, which is about 0.0909.
LADDER_OVERRIDES = {
    ("01", "plus"): (0.3, 0.25, 0.08, 0.02),
    ("11", "minus"): (0.85, 0.5, 0.093, 0.07, 0.03, 0.01),
}

# (connection id, TS file name parts, endpoint_pair, target minimum per side).
# endpoint_pair is the manifest's positional filename layout, which the FIXED
# start rule never reads; the last column is the ground truth each branch was
# generated to relax toward and is what the synthetic committed-style endpoint
# classification below is built from (the real rule source is
# reports/landscape17-endpoints.json's branch-level reference_class).
CONNECTION_SPECS = (
    ("01", (0, 1), ["0", "1"], {"plus": 0, "minus": 1}),
    ("10", (1, 0), ["1", "0"], {"plus": 1, "minus": 0}),
    ("01_2", (0, 1, 2), ["0", "1"], {"plus": 0, "minus": 1}),
    ("00", (0, 0), ["0", "0"], {"plus": 0, "minus": 0}),
    ("11", (1, 1), ["1", "1"], {"plus": 1, "minus": 1}),
)

# All five connections are held out: three validation, two test. The start
# set tests read every branch, so the training partition stays empty here.
GROUPS = {"train": [], "validation": ["01", "10", "11"], "test": ["00", "01_2"]}


def _minimum_positions(index):
    base = np.array(BASE_POSITIONS, dtype=np.float64)
    if index == 0:
        return base
    return base + np.array(MINIMUM_SEPARATION, dtype=np.float64)


def _frozen(positions):
    positions = np.array(positions, dtype=np.float64)
    positions.setflags(write=False)
    return positions


@pytest.fixture
def synthetic_landscape():
    """Deterministic mini dataset shaped exactly like the frozen manifest."""
    from force_consistency.data.types import FrameRecord

    records = []
    minima = {}

    # The two reference minima.
    for number in (0, 1):
        path = f"Landscape17/aspirin/Conformations/coordsmin_{number}.xyz"
        records.append(FrameRecord(
            record_id=path, molecule="aspirin", kind="minimum", source_path=path,
            connection_id=None, branch=None, frame_index=None, species=SPECIES,
            positions=_frozen(_minimum_positions(number)), energy=-120.0 - number,
            forces=_frozen(np.zeros((3, 3), dtype=np.float64)), source_sha256="0" * 64,
        ))
        minima[number] = path

    # One transition state per connection, halfway in shape between the
    # minima (plus a tiny fixed jiggle so they are distinct records).
    jiggle = np.array([[0.02, -0.01, 0.0], [0.01, 0.02, 0.0], [-0.01, 0.0, 0.02]], dtype=np.float64)
    transition_states = {}
    for cid, parts, pair, _targets in CONNECTION_SPECS:
        name = "coordsts_" + "_".join(str(p) for p in parts) + ".xyz"
        path = f"Landscape17/aspirin/Conformations/{name}"
        positions = 0.5 * (_minimum_positions(0) + _minimum_positions(1)) + jiggle * len(parts)
        records.append(FrameRecord(
            record_id=path, molecule="aspirin", kind="transition_state", source_path=path,
            connection_id=None, branch=None, frame_index=None, species=SPECIES,
            positions=_frozen(positions), energy=-119.5,
            forces=_frozen(np.zeros((3, 3), dtype=np.float64)), source_sha256="0" * 64,
        ))
        transition_states[cid] = path

    # The stored branch frames.
    branches = {}
    # Precompute each connection's transition-state positions once.
    ts_positions_by_connection = {
        cid: 0.5 * (_minimum_positions(0) + _minimum_positions(1)) + jiggle * len(parts)
        for cid, parts, _pair, _targets in CONNECTION_SPECS
    }
    for cid, _parts, _pair, targets in CONNECTION_SPECS:
        branches[cid] = {}
        for side in ("plus", "minus"):
            ladder = LADDER_OVERRIDES.get((cid, side), DEFAULT_LADDER)
            minimum = _minimum_positions(targets[side])
            ts_positions = ts_positions_by_connection[cid]
            ids = []
            for k, beta in enumerate(ladder):
                path = f"Landscape17/aspirin/SteepestDescent/{cid}/coords{side}_{k:03d}.xyz"
                positions = minimum + beta * (ts_positions - minimum)
                records.append(FrameRecord(
                    record_id=path, molecule="aspirin", kind="path", source_path=path,
                    connection_id=cid, branch=side, frame_index=k, species=SPECIES,
                    positions=_frozen(positions), energy=-118.0 - 0.01 * k,
                    forces=_frozen(0.001 * (positions - minimum)), source_sha256=hashlib.sha256(path.encode()).hexdigest(),
                ))
                ids.append(path)
            branches[cid][side] = ids

    connections = {}
    for cid, _parts, pair, _targets in CONNECTION_SPECS:
        connections[cid] = {
            "molecule": "aspirin",
            "connection_id": cid,
            "branches": {side: list(branches[cid][side]) for side in ("plus", "minus")},
            "transition_state": transition_states[cid],
            "endpoint_pair": list(pair),
        }

    manifest = {
        # The digest of the SYNTHETIC payload this manifest describes
        # (review round 3, gap 2): a genuine fixture carries its own
        # provenance, never the approved production archive digest,
        # which would make this manifest a relabelled production
        # landscape. The digest covers the manifest's canonical
        # connection inventory, distinct from every fixture elsewhere
        # in the suite and stable across runs.
        "archive_sha256": canonical_hash({"connections": connections}),
        "connections": {"aspirin": connections},
        "groups": {"aspirin": {part: list(ids) for part, ids in GROUPS.items()}},
        "policy": asdict(PreparationPolicy()),
        # The explicit synthetic-fixture declaration (review round 2,
        # fixtures interface): the marker lives INSIDE the manifest, so
        # the fixture needs no extra call argument and never impersonates
        # the approved production split. Its reason is auditable.
        "synthetic": {"reason": "deterministic two-minimum mini landscape "
                                "for the frozen-rule test suite"},
    }
    manifest["identity"] = canonical_hash(
        {key: manifest[key] for key in ("archive_sha256", "connections", "groups", "policy")})

    # The fixture must stay genuinely synthetic (review round 3, gap 2):
    # if the runner's approved provenance ever changes, this manifest
    # must NOT start carrying it silently — the guard would then see the
    # approved digest here and refuse the fixture, so the disagreement
    # surfaces as this loud fixture error instead of a binding refusal.
    from evaluation.runner import APPROVED_ARCHIVE_SHA256, APPROVED_SPLIT_IDENTITY

    assert manifest["archive_sha256"] != APPROVED_ARCHIVE_SHA256, (
        "the synthetic fixture's archive digest now equals the approved "
        "production digest; rebuild it from the fixture's own payload")
    assert manifest["identity"] != APPROVED_SPLIT_IDENTITY, (
        "the synthetic fixture's identity now equals the approved "
        "production identity; rebuild it from the fixture's own payload")

    return {
        "records": records,
        "manifest": manifest,
        "policy": PreparationPolicy(),
        "minima": minima,
        "branches": branches,
    }


@pytest.fixture
def frozen_synthetic_manifest(synthetic_landscape):
    """Deep copy so tamper tests never mutate the shared landscape."""
    return copy.deepcopy(synthetic_landscape["manifest"])


@pytest.fixture
def synthetic_endpoint_classification(synthetic_landscape):
    """The synthetic branches' committed-style endpoint classification.

    Built from the ground truth each branch was generated to relax toward
    (CONNECTION_SPECS targets), not from the manifest's positional
    endpoint_pair, so the sensitivity rule is tested against the same
    branch-level reference_class source the real archive supplies in
    reports/landscape17-endpoints.json. Every synthetic branch is
    "matched", which is a property of the fixture, not of the real data.
    """
    classification = {}
    for cid, _parts, _pair, targets in CONNECTION_SPECS:
        for side in ("plus", "minus"):
            classification[f"aspirin/{cid}/{side}"] = {
                "record_id": f"Landscape17/aspirin/SteepestDescent/{cid}/coords{side}_000.xyz",
                "status": "matched",
                "reference_class": synthetic_landscape["minima"][targets[side]],
            }
    return classification


def build_minima_pool(landscape):
    """The synthetic reference-minimum records keyed by number alias."""
    by_id = {record.record_id: record for record in landscape["records"]}
    return {f"m{number}": by_id[path] for number, path in landscape["minima"].items()}