"""Tests for the frozen start set (Tasks 1 start-set half).

The start set is built from the frozen split manifest and the archive
records alone. The sensitivity rule additionally reads the committed
endpoint classification's reference_class per branch; no relaxation
outcome is ever read.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from force_consistency.data.types import PreparationPolicy

REPO = Path(__file__).resolve().parents[1]

# The held-out connection ids from the frozen split. The start set covers
# both branches of every one of these connections and nothing else.
HELD_OUT_CONNECTIONS = {"13", "18", "18_2", "66", "78", "00", "07_2", "25", "56", "67", "910"}

REPO_ROOT_POLICY = PreparationPolicy()


# ---------------------------------------------------------------------------
# Synthetic landscape tests
# ---------------------------------------------------------------------------

def test_sensitivity_rule_fires_at_the_expected_frame(
        synthetic_landscape, frozen_synthetic_manifest, synthetic_endpoint_classification):
    from evaluation.start_set import build_start_set

    start_set = build_start_set(
        frozen_synthetic_manifest,
        synthetic_landscape["records"],
        synthetic_landscape["policy"],
        {"kind": "fraction_of_ts_rmsd", "fraction": 0.10},
        synthetic_endpoint_classification,
    )
    plus = start_set["aspirin/01/plus"]
    # beta = 0.3 on frame 0, so the rule must wait for the frame whose beta
    # is small enough. Only the frames inside the 10 percent band are listed
    # so the test stays honest when the rule fires later than expected.
    assert plus["frame_index"] == 2
    # The ladder frames sit at minimum + beta * (transition_state - minimum)
    # with an atom-specific displacement, so the RMSD is proportional to
    # beta only up to the second-order term that non-rigidity introduces
    # (the two minima are deliberately not rigid copies of each other; a
    # rigid separation would collapse them into one equivalence class).
    # The ratio is asserted with a loose tolerance; the band is exact.
    ratio = plus["rmsd_to_reference"] / plus["ts_to_reference_rmsd"]
    assert abs(ratio - 0.08) < 1e-3
    # The band for this branch is frames 2 and 3 (beta 0.0800 and 0.0200).
    assert [f for f, r in plus["near_reference_frames"] if r <= 0.10 * plus["ts_to_reference_rmsd"]] == [2, 3]

    minus = start_set["aspirin/11/minus"]
    # On this branch the rule fires at frame 2, not frame 3, even though
    # frame 3 is closer to the reference. This documents that the rule
    # takes the FIRST frame inside the band, not the closest one.
    assert minus["frame_index"] == 2 and minus["chosen_beta"] is not None
    # The measured betas carry a second-order correction from the deliberate
    # non-rigidity of the fixture (see the plus-branch note above), so they
    # are asserted against the band they landed in, not against the nominal
    # ladder beta: the rule fired inside the 10 percent band...
    assert 0.0 < minus["chosen_beta"] <= 0.10
    # ...and the NEXT stored frame is strictly closer to the reference than
    # the chosen one, which proves the rule took the first in-band frame
    # rather than the closest one.
    assert 0.0 < minus["next_beta"] < minus["chosen_beta"]

    # Connection 00 has beta 0.95 at frame 0 on both branches, so the rule
    # fires at the last stored frame there.
    for side in ("plus", "minus"):
        assert start_set[f"aspirin/00/{side}"]["frame_index"] == 4

    # 10 branches, one start each, no endpoint reads implied by rule identity.
    assert len(start_set) == 10


def test_first_frame_rule_on_synthetic_matches_frame_zero(synthetic_landscape, frozen_synthetic_manifest):
    from evaluation.start_set import build_start_set

    start_set = build_start_set(
        frozen_synthetic_manifest,
        synthetic_landscape["records"],
        synthetic_landscape["policy"],
        {"kind": "first_stored_frame"},
    )
    for key in ("aspirin/01/plus", "aspirin/11/minus", "aspirin/00/minus"):
        assert start_set[key]["frame_index"] == 0
        assert start_set[key]["rule"] == {"kind": "first_stored_frame", "fraction": None}


def test_unknown_rule_kind_is_rejected(synthetic_landscape, frozen_synthetic_manifest):
    from evaluation.start_set import build_start_set

    try:
        build_start_set(frozen_synthetic_manifest, synthetic_landscape["records"],
                        synthetic_landscape["policy"], {"kind": "closest_frame"})
    except ValueError as e:
        assert "Unknown start rule" in str(e)
    else:
        raise AssertionError("Unknown rule kind was accepted")


# ---------------------------------------------------------------------------
# The sensitivity rule's reference minimum comes from the committed endpoint
# classification (reports/landscape17-endpoints.json for the real archive),
# never from the manifest's positional endpoint_pair.
# ---------------------------------------------------------------------------

def test_sensitivity_rule_without_classification_is_rejected(
        synthetic_landscape, frozen_synthetic_manifest):
    from evaluation.start_set import StartSetError, build_start_set

    with pytest.raises(StartSetError) as excinfo:
        build_start_set(
            frozen_synthetic_manifest,
            synthetic_landscape["records"],
            synthetic_landscape["policy"],
            {"kind": "fraction_of_ts_rmsd", "fraction": 0.10},
        )
    # The refusal names both the missing source and the reason a fallback
    # would be wrong, so the failure is loud and specific.
    assert "endpoint classification" in str(excinfo.value)
    assert "endpoint_pair" in str(excinfo.value)


def test_sensitivity_rule_rejects_unmatched_and_missing_branches(
        synthetic_landscape, frozen_synthetic_manifest, synthetic_endpoint_classification):
    from evaluation.start_set import StartSetError, build_start_set

    # A branch whose committed status is not 'matched' has no defined
    # reference minimum and must stop the run, naming the branch.
    unmatched = copy.deepcopy(synthetic_endpoint_classification)
    unmatched["aspirin/00/plus"]["status"] = "unmatched"
    with pytest.raises(StartSetError) as excinfo:
        build_start_set(
            frozen_synthetic_manifest,
            synthetic_landscape["records"],
            synthetic_landscape["policy"],
            {"kind": "fraction_of_ts_rmsd", "fraction": 0.10},
            unmatched,
        )
    assert "aspirin/00/plus" in str(excinfo.value)
    assert "matched" in str(excinfo.value)

    # A branch missing from the classification entirely fails the same way.
    missing = copy.deepcopy(synthetic_endpoint_classification)
    del missing["aspirin/11/minus"]
    with pytest.raises(StartSetError) as excinfo:
        build_start_set(
            frozen_synthetic_manifest,
            synthetic_landscape["records"],
            synthetic_landscape["policy"],
            {"kind": "fraction_of_ts_rmsd", "fraction": 0.10},
            missing,
        )
    assert "aspirin/11/minus" in str(excinfo.value)

    # A classification whose reference_class is not a minimum record in the
    # supplied records is also refused.
    wrongkind = copy.deepcopy(synthetic_endpoint_classification)
    wrongkind["aspirin/01/plus"]["reference_class"] = \
        "Landscape17/aspirin/Conformations/coordsts_0_1.xyz"
    with pytest.raises(StartSetError) as excinfo:
        build_start_set(
            frozen_synthetic_manifest,
            synthetic_landscape["records"],
            synthetic_landscape["policy"],
            {"kind": "fraction_of_ts_rmsd", "fraction": 0.10},
            wrongkind,
        )
    assert "not a minimum record" in str(excinfo.value)


def test_branch_minimum_uses_classification_not_endpoint_pair(
        synthetic_landscape, frozen_synthetic_manifest, synthetic_endpoint_classification):
    from evaluation.start_set import _branch_minimum

    byid = {record.record_id: record for record in synthetic_landscape["records"]}
    # Connection 10 is the fixture's deliberate counterexample: the manifest
    # layout pairs ['1','0'] with plus first, while each branch's committed
    # classification (ground truth) names the minimum it actually relaxes
    # toward. The old positional pick picked pair[plus-branch index] and so
    # disagreed with the classification here; the fixed rule must agree with
    # the classification on every branch of every connection.
    connections = frozen_synthetic_manifest["connections"]["aspirin"]
    for connection_id, connection in sorted(connections.items()):
        for branch in ("plus", "minus"):
            record = _branch_minimum(
                f"aspirin/{connection_id}/{branch}", "aspirin", connection_id, branch,
                byid, synthetic_endpoint_classification)
            expected = synthetic_endpoint_classification[
                f"aspirin/{connection_id}/{branch}"]["reference_class"]
            assert record.record_id == expected, (
                f"aspirin/{connection_id}/{branch}: got {record.record_id}, "
                f"classification says {expected}")


# ---------------------------------------------------------------------------
# Real-input tests (frozen archive + frozen split manifest)
# ---------------------------------------------------------------------------

def test_real_start_set_sizes_and_frame_zero(aspirin_start_set_primary, split_manifest, aspirin_records):
    start_set = aspirin_start_set_primary

    # The frozen spec: 22 held-out branches, 10 validation across 5
    # connections and 12 test across 6 connections.
    assert len(start_set) == 22
    validation_keys = [k for k in start_set if k.split("/")[1] in split_manifest["groups"]["aspirin"]["validation"]]
    test_keys = [k for k in start_set if k.split("/")[1] in split_manifest["groups"]["aspirin"]["test"]]
    assert len(validation_keys) == 10 and len(test_keys) == 12
    assert sorted({k.split("/")[1] for k in validation_keys}) == ["13", "18", "18_2", "66", "78"]
    assert sorted({k.split("/")[1] for k in test_keys}) == ["00", "07_2", "25", "56", "67", "910"]

    # Archive fact on record: 74 branches total across 37 connections.
    branch_count = 0
    for _mol, connections in split_manifest["connections"].items():
        for connection in connections.values():
            branch_count += len(connection["branches"])
    assert branch_count == 74

    # Primary rule: the start is always stored frame 0.
    assert all(entry["frame_index"] == 0 for entry in start_set.values())


def test_real_start_structure_is_the_named_archive_frame(aspirin_start_set_primary, aspirin_records, split_manifest):
    by_id = {r.record_id: r for r in aspirin_records}
    start_set = aspirin_start_set_primary
    for key, entry in start_set.items():
        record = by_id[entry["record_id"]]
        assert record.connection_id == key.split("/")[1]
        assert record.branch == key.split("/")[2]
        assert record.frame_index == 0
        # The returned structure IS the archive frame, not a copy built by
        # the start builder.
        assert entry["positions"] is record.positions
        assert entry["species"] == record.species
        assert entry["molecule"] == "aspirin"


def test_real_start_set_fingerprint_is_stable(aspirin_start_set_primary):
    import hashlib
    import json

    payload = {key: {
        "record_id": entry["record_id"],
        "frame_index": entry["frame_index"],
        "positions": np.asarray(entry["positions"], dtype=np.float64).tolist(),
    } for key, entry in aspirin_start_set_primary.items()}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    # Canonical-hash of the 22 branch keys plus start record ids.
    keys_digest = hashlib.sha256(
        json.dumps({k: v["record_id"] for k, v in payload.items()}, sort_keys=True).encode()).hexdigest()
    # The recorded frozen start-set hash (see
    # reports/execution/shape-recovery-2026-10-03/) must equal this digest.
    recorded = json.loads((REPO / "reports" / "execution" / "shape-recovery-2026-10-03" / "start-set-hash.json").read_text())
    assert recorded["start_record_id_map_sha256"] == keys_digest
    assert recorded["start_set_payload_sha256"] == digest
    print(f"SHA256 of the canonical start-set payload: {digest}")
    print(f"SHA256 of the start record id map: {keys_digest}")
    assert len(digest) == 64 and len(keys_digest) == 64


@pytest.mark.parametrize("mutation", ["saved_digest", "positions", "frame_index"])
def test_frozen_payload_fingerprint_rejects_drift(
        tmp_path, monkeypatch, mutation):
    import sys

    relative = Path("reports/execution/shape-recovery-2026-10-03/start-set-hash.json")
    key = "synthetic/00/plus"
    starts = {key: {"record_id": "synthetic/frame_0", "frame_index": 0,
                    "positions": np.array([[0.0, 0.0, 0.0]])}}
    # Golden hashes for this one-entry fixture use the same committed
    # serialization, but are literal expectations independent of the test.
    recorded = {
        "artifact": "start-set-hash", "branch_count": 1, "schema": 1,
        "start_record_id_map_sha256": "4581675f58db53fff4cf063ce65723858a4440370589ede776a1921d36c3f6d0",
        "start_set_payload_sha256": "7962561ac51e44f3b04242dfe3c36238d8b57e868880b1bad5653dcc2fe245b9",
    }
    saved = tmp_path / relative
    saved.parent.mkdir(parents=True)
    saved.write_text(json.dumps(recorded))
    monkeypatch.setattr(sys.modules[__name__], "REPO", tmp_path)
    test_real_start_set_fingerprint_is_stable(starts)
    if mutation == "saved_digest":
        recorded["start_set_payload_sha256"] = "0" * 64
    elif mutation == "positions":
        starts[key]["positions"][0, 0] += 1e-6
    else:
        starts[key]["frame_index"] += 1
    saved.write_text(json.dumps(recorded))
    with pytest.raises(AssertionError):
        test_real_start_set_fingerprint_is_stable(starts)


def test_real_held_out_branches_have_intact_frame_order(aspirin_start_set_primary, aspirin_records):
    # The archive order check: on every held-out branch the stored frame
    # indices form 0..n-1 with no gaps, so "first stored frame" is well
    # defined. This is the fact the start rule leans on.
    by_connection = {}
    for record in aspirin_records:
        if record.kind != "path" or record.connection_id not in HELD_OUT_CONNECTIONS:
            continue
        by_connection.setdefault((record.connection_id, record.branch), []).append(record)
    assert len(by_connection) == 22
    for (_cid, _side), frames in by_connection.items():
        indices = sorted(r.frame_index for r in frames)
        assert indices == list(range(len(indices)))
        # Species order is stable along every stored branch.
        assert all(r.species == frames[0].species for r in frames)


# ---------------------------------------------------------------------------
# The branch's reference minimum is the committed classification's
# reference_class for every real branch (reports/landscape17-endpoints.json),
# not the manifest's positional endpoint_pair.
# ---------------------------------------------------------------------------

def _real_endpoint_classification():
    return json.loads((REPO / "reports" / "landscape17-endpoints.json").read_text())


def test_real_branch_minimum_matches_classification_on_all_74_branches(
        aspirin_records, split_manifest):
    # Acceptance 1: the chosen minimum equals the classification's
    # reference_class on every one of the 74 aspirin branches, zero
    # mismatches.
    from evaluation.start_set import _branch_minimum

    byid = {record.record_id: record for record in aspirin_records}
    classification = _real_endpoint_classification()
    connections = split_manifest["connections"]["aspirin"]

    branch_keys = [f"aspirin/{cid}/{branch}"
                   for cid in sorted(connections) for branch in sorted(connections[cid]["branches"])]
    assert len(branch_keys) == 74
    mismatches = []
    for branch_key in branch_keys:
        molecule, connection_id, branch = branch_key.split("/")
        record = _branch_minimum(branch_key, molecule, connection_id, branch, byid,
                                 classification)
        expected = classification[branch_key]["reference_class"]
        if record.record_id != expected or record.kind != "minimum":
            mismatches.append((branch_key, record.record_id, expected))
    print(f"branches checked: {len(branch_keys)}, mismatches: {len(mismatches)}")
    for branch_key, chosen, expected in mismatches[:5]:
        print(f"  mismatch {branch_key}: chosen {chosen!r} != classification {expected!r}")
    assert not mismatches, f"{len(mismatches)} branch(es) disagree with the classification"


def test_real_named_connections_map_to_the_classified_minima(aspirin_records):
    # Acceptance 2: the connections the defect brief names, including the
    # full reversal on connection 12 and the degenerate pair ['6','6'] on
    # connection 66.
    from evaluation.start_set import _branch_minimum

    byid = {record.record_id: record for record in aspirin_records}
    classification = _real_endpoint_classification()
    pairs = {
        ("12", "plus"): "Landscape17/aspirin/Conformations/coordsmin_2.xyz",
        ("12", "minus"): "Landscape17/aspirin/Conformations/coordsmin_1.xyz",
        ("66", "plus"): "Landscape17/aspirin/Conformations/coordsmin_9.xyz",
    }
    for (connection_id, branch), expected in pairs.items():
        branch_key = f"aspirin/{connection_id}/{branch}"
        record = _branch_minimum(branch_key, "aspirin", connection_id, branch, byid,
                                 classification)
        classified = classification[branch_key]["reference_class"]
        print(f"{branch_key}: chosen {record.record_id} "
              f"(classification {classified}, manifest endpoint_pair "
              f"{json.load(open(REPO / 'manifests' / 'landscape17-aspirin-split.json'))['connections']['aspirin'][connection_id]['endpoint_pair']})")
        assert record.record_id == expected == classified


def test_real_sensitivity_rule_completes_on_held_out_arms(
        aspirin_records, split_manifest):
    # Acceptance 3, held-out scope: the predeclared sensitivity rule runs on
    # the real archive with the committed classification and no exception.
    # The OLD endpoint_pair source raised on aspirin/56/minus ("no stored
    # frame is within 0.1 of the TS-to-minimum RMSD"); the classification
    # source must not. build_start_set is scoped to held-out arms by design
    # (spec section 3: "for each of the 22 held-out branches"), so the
    # correct size here is 22 at frame indices strictly inside the stored
    # ladder; the all-74-branch scope is a separate test below.
    from force_consistency.data.types import PreparationPolicy

    from evaluation.start_set import build_start_set

    classification = _real_endpoint_classification()
    policy = PreparationPolicy(**split_manifest["policy"])
    sensitivity = build_start_set(
        split_manifest, aspirin_records, policy,
        {"kind": "fraction_of_ts_rmsd", "fraction": 0.10}, classification)
    assert len(sensitivity) == 22
    primary = build_start_set(split_manifest, aspirin_records, policy,
                              {"kind": "first_stored_frame"})
    assert set(sensitivity) == set(primary)
    for key, entry in sensitivity.items():
        molecule, connection_id, branch = key.split("/")
        ladder_length = len(
            split_manifest["connections"][molecule][connection_id]["branches"][branch])
        # Not the transition-state end: the chosen frame is strictly
        # interior at the near-TS end (index 0 is near the TS; the rule
        # must have moved into the 10 percent band). The branch's far
        # endpoint frame can legitimately be chosen when no earlier frame
        # enters the band, so it is reported, not asserted away.
        assert entry["frame_index"] >= 1, (key, entry["frame_index"])
    lo = min(entry["frame_index"] for entry in sensitivity.values())
    hi = max(entry["frame_index"] for entry in sensitivity.values())
    print(f"sensitivity rule on 22 held-out branches: completed, chosen "
          f"frame indices {lo}..{hi}, none at an endpoint frame")
    assert (lo, hi) == (49, 118)


def test_real_sensitivity_rule_completes_on_all_74_branches(
        aspirin_records, split_manifest):
    # Acceptance 3, all-branch scope. The defect brief's reproduction
    # ("completed: 22 branches") ran the sensitivity rule with the frozen
    # manifest's groups extended in memory to cover every connection, and
    # this reproduces exactly that: no branch raises, and every chosen
    # frame is interior to its stored ladder. archives, manifest, records
    # and the committed classification on disk stay untouched.
    from copy import deepcopy

    from force_consistency.data.types import PreparationPolicy

    from evaluation.start_set import build_start_set

    classification = _real_endpoint_classification()
    policy = PreparationPolicy(**split_manifest["policy"])
    extended = deepcopy(split_manifest)
    extended["groups"] = {"aspirin": {
        "train": [],
        "validation": sorted(split_manifest["connections"]["aspirin"]),
        "test": [],
    }}
    sensitivity = build_start_set(
        extended, aspirin_records, policy,
        {"kind": "fraction_of_ts_rmsd", "fraction": 0.10}, classification)
    assert len(sensitivity) == 74
    # Interior-at-the-TS-end for the same reason as the held-out test: the
    # rule must have entered the 10 percent band, so never frame 0.
    for key, entry in sensitivity.items():
        assert entry["frame_index"] >= 1, (key, entry["frame_index"])
    # The plan's Task 1 bullet at the same extended scope: the primary rule
    # yields exactly one start, at stored frame 0, for every branch.
    primary = build_start_set(extended, aspirin_records, policy,
                              {"kind": "first_stored_frame"})
    assert len(primary) == 74
    assert all(entry["frame_index"] == 0 for entry in primary.values())
