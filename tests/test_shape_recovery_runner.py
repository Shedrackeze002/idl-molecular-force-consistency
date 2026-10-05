"""Tests for the evaluation runner (review item A5) and the nonrigid
positive relaxation fixture (review item A6).

The runner composes the existing, individually tested pieces:
build_start_set output -> adapter arm -> relax -> classify_outcome.

Two disciplines are tested here:

- Partition authorisation: the test partition is evaluated only when an
  explicit authorisation flag is passed (the design spec's section 7);
  validation and train partitions need none.
- Recovery: the positive fixture now includes a NONRIGID start that sits
  outside the 0.05 angstrom match tolerance before relaxation and is
  matched only after relaxation, so the tests exercise recovery rather
  than passing for free on an already-aligned rigid translation.

The shared synthetic landscape in conftest keeps its minima ~0.06
angstrom apart so ambiguous territory exists (a midway point inside the
0.05 endpoint tolerance of both minima); its stored frame 0 therefore
sits only ~0.03 angstrom from its own minimum, INSIDE the 0.05
tolerance. A recovery fixture that must start OUTSIDE the tolerance
cannot be built from that landscape, so this module owns a second,
self-contained fixture landscape with a 0.5 angstrom-scale nonrigid
minimum separation: the minima stay distinct equivalence classes, the
start frames sit far outside the tolerance, and the ambiguity tests keep
running on the shared landscape.
"""
import copy
import numpy as np
import pytest

from force_consistency.data.types import PreparationPolicy

from conftest import SPECIES


# ---------------------------------------------------------------------------
# A6: a self-contained recovery landscape with a NONRIGID minimum pair at a
# 0.5 angstrom scale, so starts genuinely sit outside the 0.05 tolerance
# ---------------------------------------------------------------------------

_RECOVERY_BASE = np.array([[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 1.3, 0.0]],
                          dtype=np.float64)
# Non-rigid: the three rows carry three DIFFERENT offsets, so no rotation
# plus translation maps minimum 0 onto minimum 1 exactly and the pair are
# distinct equivalence classes under the frozen comparison.
_RECOVERY_SEPARATION = np.array([[0.35, 0.1, 0.0], [-0.35, -0.05, 0.0],
                                 [0.0, 0.45, 0.0]], dtype=np.float64)
# A side-specific nonrigid kick separating the two branch starts.
_RECOVERY_KICK = np.array([[0.05, 0.0, 0.0], [-0.05, 0.0, 0.0],
                           [0.0, 0.05, 0.0]], dtype=np.float64)

_CONNECTION_TARGETS = (("80", {"plus": 0, "minus": 0}),
                       ("81", {"plus": 1, "minus": 1}))


def recovery_minimum(index):
    return _RECOVERY_BASE + (0.0 if index == 0 else _RECOVERY_SEPARATION)


class RecoveryLandscape:
    """The minimum pair, its branch records and its frozen split manifest.

    Both branches of connection 80 commit to minimum 0; both branches of
    connection 81 commit to minimum 1. Each branch stores exactly one
    frame: its start, sitting near the midpoint between the minima with a
    side-specific kick, well outside the 0.05 angstrom tolerance.
    """

    def __init__(self):
        from force_consistency.data.types import FrameRecord

        self.minima = {}
        self.records = []
        for number in (0, 1):
            record_id = f"recovery/minimum_{number}"
            self.records.append(FrameRecord(
                record_id=record_id, molecule="aspirin", kind="minimum",
                source_path=record_id, connection_id=None, branch=None,
                frame_index=None, species=SPECIES,
                positions=self._frozen(recovery_minimum(number)),
                energy=-120.0 - number,
                forces=self._frozen(np.zeros((3, 3), dtype=np.float64)),
                source_sha256="0" * 64))
            self.minima[number] = record_id
        midpoint = 0.5 * (recovery_minimum(0) + recovery_minimum(1))
        connections = {}
        for cid, sides in _CONNECTION_TARGETS:
            branch_ids = {}
            for side, target in sides.items():
                record_id = f"recovery/{cid}/{side}/frame_0"
                kick = _RECOVERY_KICK if side == "plus" else -_RECOVERY_KICK
                self.records.append(FrameRecord(
                    record_id=record_id, molecule="aspirin", kind="path",
                    source_path=record_id, connection_id=cid, branch=side,
                    frame_index=0, species=SPECIES,
                    positions=self._frozen(midpoint + kick),
                    energy=-118.0,
                    forces=self._frozen(np.zeros((3, 3))),
                    source_sha256="0" * 64))
                branch_ids[side] = [record_id]
            connections[cid] = {
                "molecule": "aspirin", "connection_id": cid,
                "branches": branch_ids, "endpoint_pair": ["0", "1"],
            }
        self.manifest = {
            "archive_sha256": "0" * 64,
            "connections": {"aspirin": connections},
            "groups": {"aspirin": {"train": [], "validation": ["80"],
                                   "test": ["81"]}},
            "policy": None,
        }

    @staticmethod
    def _frozen(array):
        array = np.array(array, dtype=np.float64)
        array.setflags(write=False)
        return array

    def policy(self):
        return PreparationPolicy()

    def references(self):
        return {f"m{index}": record for index, record
                in enumerate(self.records) if record.kind == "minimum"}


@pytest.fixture
def recovery_landscape():
    return RecoveryLandscape()


@pytest.fixture
def frozen_recovery_manifest(recovery_landscape):
    return copy.deepcopy(recovery_landscape.manifest)


def _runner_procedure(**overrides):
    base = {
        "max_force_evaluations": 300,
        "initial_step_size": 0.2,
        "backtracking_factor": 0.5,
        "min_backtracking_floor": 1e-8,
        "convergence_tolerance_eV_per_A": 1e-2,
        "max_backtracking_attempts_per_step": 40,
    }
    base.update(overrides)
    return base


def _recovery_classification(landscape):
    classification = {}
    for cid, sides in _CONNECTION_TARGETS:
        for side, target in sides.items():
            classification[f"aspirin/{cid}/{side}"] = {
                "record_id": f"recovery/{cid}/{side}/frame_0",
                "status": "matched",
                "reference_class": landscape.minima[target],
            }
    return classification


def _bowl_arm_for(minimum, arm_id="recovery-bowl"):
    """A callable arm whose exact negative gradient pulls every atom
    toward the given minimum at strength 1 eV/angstrom^2."""
    from evaluation.adapter import callable_arm

    minimum = np.asarray(minimum, dtype=np.float64)

    def force(positions):
        positions = np.asarray(positions, dtype=np.float64)
        return minimum - positions

    return callable_arm(force, species=SPECIES, arm_id=arm_id)


def _recovery_start_set(landscape, manifest):
    from evaluation.start_set import build_start_set

    return build_start_set(
        manifest, landscape.records, landscape.policy(),
        {"kind": "first_stored_frame"})


# ---------------------------------------------------------------------------
# A5: the runner end to end over the recovery landscape
# ---------------------------------------------------------------------------

def test_runner_evaluates_validation_branches_end_to_end(
        recovery_landscape, frozen_recovery_manifest):
    from evaluation.runner import evaluate_starts

    full_start_set = _recovery_start_set(recovery_landscape, frozen_recovery_manifest)
    # The validation slice only: the runner refuses test-partition branches
    # without authorisation even inside a mixed set.
    start_set = {key: entry for key, entry in full_start_set.items()
                 if key.split("/")[1] == "80"}
    references = recovery_landscape.references()
    arm = _bowl_arm_for(recovery_minimum(0))
    result = evaluate_starts(
        start_set, arm, _runner_procedure(),
        _recovery_classification(recovery_landscape), references,
        recovery_landscape.policy(), split_manifest=frozen_recovery_manifest,
        seen_minimum=True,
    )
    # The validation connection's two branches are admitted; the test
    # connection's two branches are refused by default.
    assert result["branch_count"] == 2
    assert sorted(result["branches"]) == ["aspirin/80/minus", "aspirin/80/plus"]
    # Every branch relaxed and classified into exactly one category, with
    # the facts carried through; the exact per-atom gradient of the target
    # bowl lands each branch inside its own committed reference class.
    for row in result["branches"].values():
        assert row["partition"] == "validation"
        assert row["category"] == "matched_reference"
        assert row["status"] == "converged"
        assert isinstance(row["force_evaluation_count"], int)
        assert row["force_evaluation_count"] >= 2
        assert row["matching_status"] == "matched"
        assert row["rmsd_A"] is not None and row["rmsd_A"] <= 0.05
        assert row["nonfinite_force"] is False
        assert row["stop_reason"] is None
    # Per-connection aggregation: both branches counted together, and the
    # category counts sum to the evaluated branch count.
    assert set(result["connections"]) == {"aspirin/80"}
    for connection in result["connections"].values():
        assert connection["evaluated_branches"] == len(connection["branches"])
        assert sum(connection["counts"].values()) == connection["evaluated_branches"]
        assert connection["counts"]["matched_reference"] == 2


def test_runner_test_partition_is_refused_without_authorisation(
        recovery_landscape, frozen_recovery_manifest):
    from evaluation.runner import PartitionAuthorisationError, evaluate_starts

    start_set = _recovery_start_set(recovery_landscape, frozen_recovery_manifest)
    with pytest.raises(PartitionAuthorisationError) as excinfo:
        evaluate_starts(
            start_set, _bowl_arm_for(recovery_minimum(0)), _runner_procedure(),
            _recovery_classification(recovery_landscape),
            recovery_landscape.references(), recovery_landscape.policy(),
            split_manifest=frozen_recovery_manifest,
        )
    # The refusal NAMES the test-partition branches.
    message = str(excinfo.value)
    assert "aspirin/81/plus" in message and "aspirin/81/minus" in message


def test_runner_test_partition_is_permitted_with_authorisation(
        recovery_landscape, frozen_recovery_manifest):
    from evaluation.runner import evaluate_starts

    start_set = _recovery_start_set(recovery_landscape, frozen_recovery_manifest)
    arm = _bowl_arm_for(recovery_minimum(1))
    result = evaluate_starts(
        start_set, arm, _runner_procedure(),
        _recovery_classification(recovery_landscape),
        recovery_landscape.references(), recovery_landscape.policy(),
        split_manifest=frozen_recovery_manifest,
        authorise_test_partition=True,
    )
    # All four branches are evaluated once, each exactly once (the
    # authorisation is the recorded fact on the result).
    assert result["branch_count"] == 4
    assert result["authorise_test_partition"] is True
    partitions = {row["partition"] for row in result["branches"].values()}
    assert partitions == {"validation", "test"}
    test_branches = [key for key, row in result["branches"].items()
                     if row["partition"] == "test"]
    assert sorted(test_branches) == ["aspirin/81/minus", "aspirin/81/plus"]
    # The authorised test branches recovered their own committed minimum.
    for key in test_branches:
        assert result["branches"][key]["category"] == "matched_reference"


def test_runner_validation_needs_no_authorisation(
        recovery_landscape, frozen_recovery_manifest):
    from evaluation.runner import evaluate_starts

    start_set = _recovery_start_set(recovery_landscape, frozen_recovery_manifest)
    validation_only = {key: entry for key, entry in start_set.items()
                       if key.split("/")[1] == "80"}
    result = evaluate_starts(
        validation_only, _bowl_arm_for(recovery_minimum(0)),
        _runner_procedure(), _recovery_classification(recovery_landscape),
        recovery_landscape.references(), recovery_landscape.policy(),
        split_manifest=frozen_recovery_manifest,
    )
    # No authorisation flag at all: the validation branch run just works.
    assert result["branch_count"] == 2


def test_runner_missing_split_manifest_is_refused(
        recovery_landscape, frozen_recovery_manifest):
    from evaluation.runner import evaluate_starts

    start_set = _recovery_start_set(recovery_landscape, frozen_recovery_manifest)
    with pytest.raises(ValueError) as excinfo:
        evaluate_starts(
            start_set, _bowl_arm_for(recovery_minimum(0)), _runner_procedure(),
            _recovery_classification(recovery_landscape),
            recovery_landscape.references(), recovery_landscape.policy())
    assert "split_manifest is required" in str(excinfo.value)


def test_runner_arm_boundary_symbol_tuples_not_atomic_numbers(
        recovery_landscape, frozen_recovery_manifest):
    # The agreed boundary: the runner passes the START's own element-SYMBOL
    # tuple to the arm. The atomic-number encoding belongs behind the
    # adapter (arm_from_checkpoint converts a checkpoint's atomic numbers
    # to element symbols before any arm ever supplies a force). Prove the
    # boundary direction by recording what the arm actually receives.
    from evaluation.adapter import callable_arm
    from evaluation.runner import evaluate_starts

    start_set = _recovery_start_set(recovery_landscape, frozen_recovery_manifest)
    validation_only = {key: entry for key, entry in start_set.items()
                       if key.split("/")[1] == "80"}
    seen = []

    def force(positions):
        return np.zeros(np.asarray(positions).shape, dtype=np.float64)

    probe_arm = callable_arm(force, species=SPECIES, arm_id="boundary-probe")
    # Record the species kwarg exactly where the runner binds it: on the
    # arm's forces callable, which the runner's supplier calls as
    # arm["forces"](positions, species=<start's tuple>, record_id=<id>).
    arm_forces = probe_arm["forces"]

    def recording_forces(positions, species=None, record_id=None):
        seen.append((species, record_id))
        return arm_forces(positions, species=species, record_id=record_id)

    probe_arm["forces"] = recording_forces
    result = evaluate_starts(
        validation_only, probe_arm, _runner_procedure(),
        _recovery_classification(recovery_landscape),
        recovery_landscape.references(), recovery_landscape.policy(),
        split_manifest=frozen_recovery_manifest,
    )
    # Zero forces everywhere meet the declared rule on the first
    # evaluation, so both branches finish as matched_reference; the
    # boundary point stands regardless. The arm received the START's
    # SYMBOL tuple per branch start, never an atomic-number encoding, and
    # the record identity named the branch's own start record.
    assert result["branch_count"] == 2
    assert [entry[0] for entry in seen] == [SPECIES, SPECIES]
    assert all(isinstance(symbol, str) for entry in seen for symbol in entry[0])
    assert sorted(entry[1] for entry in seen) == [
        "recovery/80/minus/frame_0/relax", "recovery/80/plus/frame_0/relax"]
    assert result["arm"]["arm_id"] == "boundary-probe"


# ---------------------------------------------------------------------------
# A6: the pre-relaxation geometry facts the recovery claim needs
# ---------------------------------------------------------------------------

def _as_record(record_id, positions):
    from force_consistency.data.types import FrameRecord

    return FrameRecord(
        record_id=record_id, molecule="aspirin", kind="path",
        source_path=record_id, connection_id=None, branch=None,
        frame_index=None, species=SPECIES,
        positions=np.asarray(positions, dtype=np.float64), energy=0.0,
        forces=np.zeros_like(np.asarray(positions, dtype=np.float64)),
        source_sha256="0" * 64,
    )


def test_recovery_minima_are_distinct_nonrigid_classes(recovery_landscape):
    from force_consistency.data.geometry import align_records

    policy = recovery_landscape.policy()
    m0, m1 = recovery_landscape.references()["m0"], recovery_landscape.references()["m1"]
    separation = (np.asarray(m1.positions, dtype=np.float64)
                  - np.asarray(m0.positions, dtype=np.float64))
    # NOT rigid: the three rows carry three different offsets, so no
    # rotation+translation maps minimum 0 onto minimum 1 exactly; the
    # minima remain distinct equivalence classes.
    norms = np.linalg.norm(separation, axis=1)
    assert not np.allclose(norms, norms[0], atol=1e-12)
    # The frozen comparison agrees: each minimum does NOT match the other.
    for mine, theirs in ((m0, m1), (m1, m0)):
        aligned = align_records(_as_record("probe", mine.positions), theirs, policy)
        assert aligned.rmsd > 1e-5


def test_recovery_starts_sit_off_their_reference_before_relaxation(
        recovery_landscape, frozen_recovery_manifest):
    from force_consistency.data.geometry import align_records

    policy = recovery_landscape.policy()
    references = recovery_landscape.references()
    start_set = _recovery_start_set(recovery_landscape, frozen_recovery_manifest)
    for key, start in start_set.items():
        own_id = _recovery_classification(recovery_landscape)[key]["reference_class"]
        own = next(record for record in references.values()
                   if record.record_id == own_id)
        aligned = align_records(
            _as_record(start["record_id"], start["positions"]), own, policy)
        # THE review point: before relaxation every start sits OUTSIDE the
        # 0.05 angstrom match tolerance, so a later match cannot come from
        # the start already sitting on the reference.
        assert aligned.rmsd > 0.05, (key, aligned.rmsd)


def test_nonrigid_positive_start_becomes_matched_after_relaxation(
        recovery_landscape, frozen_recovery_manifest):
    from evaluation.runner import evaluate_starts
    from force_consistency.data.geometry import align_records

    policy = recovery_landscape.policy()
    start_set = _recovery_start_set(recovery_landscape, frozen_recovery_manifest)
    start = start_set["aspirin/80/plus"]
    m0 = recovery_landscape.references()["m0"]
    # Pre-relaxation: outside the tolerance.
    pre = align_records(_as_record(start["record_id"], start["positions"]), m0, policy)
    assert pre.rmsd > 0.05
    # Post-relaxation through the per-atom bowl toward the branch's own
    # committed minimum: matched_reference with a measured rmsd inside
    # the tolerance.
    result = evaluate_starts(
        {"aspirin/80/plus": start}, _bowl_arm_for(recovery_minimum(0)),
        _runner_procedure(), _recovery_classification(recovery_landscape),
        recovery_landscape.references(), policy,
        split_manifest=frozen_recovery_manifest,
    )
    row = result["branches"]["aspirin/80/plus"]
    assert row["category"] == "matched_reference"
    assert row["rmsd_A"] <= 0.05
    assert row["start_record_id"] == start["record_id"]