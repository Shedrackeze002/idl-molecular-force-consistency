"""Tests for the evaluation runner (review item A5) and the nonrigid
positive relaxation fixture (review item A6).

The runner composes the existing, individually tested pieces:
build_start_set output -> adapter arm -> relax -> classify_outcome.

Two disciplines are tested here:

- Partition authorisation: the test partition is evaluated only when an
  explicit authorisation flag is passed (the design spec's section 7);
  validation and train partitions need none.
- The split binding (review round 2): a manifest carrying the explicit
  "synthetic": {"reason": ...} declaration binds as an identified
  fixture; a manifest without it is production and must recompute to
  the approved split identity. Overlapping or incomplete partition
  membership and every unauthorised test branch are refused BEFORE any
  force call.
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
            # The explicit synthetic-fixture declaration (review round 2,
            # fixtures interface): the marker lives INSIDE the manifest, so
            # this test landscape never impersonates the approved
            # production split and needs no extra call argument.
            "synthetic": {"reason": "self-contained two-minimum recovery "
                                    "landscape for runner tests"},
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
# The split binding interface (review round 2, fixtures restored):
# synthetic manifests are identified INSIDE the manifest, production
# manifests are still identity-bound, and everything fails closed
# BEFORE any force call.
# ---------------------------------------------------------------------------

def _force_spy_arm(minimum, counter):
    """A bowl arm whose underlying supplier counts every real call."""
    arm = _bowl_arm_for(minimum)
    arm_forces = arm["forces"]

    def counting_forces(positions, species=None, record_id=None):
        counter["calls"] += 1
        return arm_forces(positions, species=species, record_id=record_id)

    arm["forces"] = counting_forces
    return arm


def _runner_call(landscape, manifest, **overrides):
    from evaluation.runner import evaluate_starts

    start_set = _recovery_start_set(landscape, manifest)
    kwargs = dict(
        start_set=start_set,
        arm=_bowl_arm_for(recovery_minimum(0)),
        procedure=_runner_procedure(),
        endpoint_classification=_recovery_classification(landscape),
        references=landscape.references(),
        policy=landscape.policy(),
        split_manifest=manifest,
    )
    kwargs.update(overrides)
    return evaluate_starts(**kwargs)


def _stripped_frozen_manifest(recovery_landscape):
    """A frozen manifest clone with the synthetic declaration removed."""
    manifest = copy.deepcopy(recovery_landscape.manifest)
    manifest.pop("synthetic")
    return manifest


def test_unmarked_manifest_is_refused_as_production_and_reaches_no_force_call(
        recovery_landscape):
    from evaluation.runner import SplitBindingError

    counter = {"calls": 0}
    with pytest.raises(SplitBindingError) as excinfo:
        _runner_call(
            recovery_landscape, _stripped_frozen_manifest(recovery_landscape),
            arm=_force_spy_arm(recovery_minimum(0), counter))
    message = str(excinfo.value)
    # The refusal names what is missing and points at the reviewed marker.
    assert "synthetic" in message
    assert "impersonating the approved manifest" in message
    assert counter["calls"] == 0


def test_marker_without_a_reason_is_refused(recovery_landscape):
    from evaluation.runner import SplitBindingError

    for bad_marker in (True, "yes", {}, {"reason": "   "}):
        manifest = copy.deepcopy(recovery_landscape.manifest)
        manifest["synthetic"] = bad_marker
        with pytest.raises(SplitBindingError) as excinfo:
            _runner_call(recovery_landscape, manifest)
        assert "reason" in str(excinfo.value) or "must be a dict" in str(excinfo.value)


def test_approved_manifest_relabelled_synthetic_is_refused(
        recovery_landscape, split_manifest):
    from evaluation.runner import SplitBindingError

    # Tamper route: the production manifest is relabelled as a synthetic
    # fixture, hoping the fixture path would skip the identity binding.
    tampered = copy.deepcopy(split_manifest)
    tampered["synthetic"] = {"reason": "definitely a fixture"}
    tampered.pop("identity")
    with pytest.raises(SplitBindingError) as excinfo:
        # The start set is built from the fixture landscape's own
        # manifest, whose records it names; the tampered production
        # manifest goes ONLY to the split binding, which is the thing
        # under test here.
        _runner_call(recovery_landscape,
                     copy.deepcopy(recovery_landscape.manifest),
                     split_manifest=tampered)
    message = str(excinfo.value)
    assert "recomputes to the approved identity" in message
    assert "must be bound as production" in message


def test_manifest_edited_after_approval_is_refused_before_any_force_call(
        recovery_landscape, split_manifest):
    from evaluation.runner import SplitBindingError

    # A production manifest whose partition groups were edited cannot
    # recompute to the frozen identity, so relabelling a test connection
    # into validation fails closed, and no force is ever called. Note
    # this copy has no synthetic declaration: stripping the shipped self
    # identity is not enough to pass it off as an unbound fixture.
    tampered = copy.deepcopy(split_manifest)
    tampered.pop("identity")
    tampered["groups"] = {"aspirin": {"train": [], "validation": ["78", "00"],
                                      "test": ["07_2", "25", "56", "67", "910"]}}
    counter = {"calls": 0}
    with pytest.raises(SplitBindingError) as excinfo:
        # The recovery landscape's records do not fit the real manifest,
        # so the run could not reach the force stage anyway; binding MUST
        # fail first, which is exactly this assertion. The count proves
        # the force supplier is untouched, not that the run is possible.
        # The start set comes from the fixture landscape's own manifest;
        # the edited production clone goes ONLY to the split binding.
        _runner_call(recovery_landscape,
                     copy.deepcopy(recovery_landscape.manifest),
                     arm=_force_spy_arm(recovery_minimum(0), counter),
                     split_manifest=tampered)
    assert "recomputes to identity" in str(excinfo.value)
    assert "refuses every other one" in str(excinfo.value)
    assert counter["calls"] == 0


def test_fixture_binding_is_echoed_on_the_result(recovery_landscape,
                                                frozen_recovery_manifest):
    result = _runner_call(
        recovery_landscape, frozen_recovery_manifest,
        start_set={"aspirin/80/plus": _recovery_start_set(
            recovery_landscape, frozen_recovery_manifest)["aspirin/80/plus"]})
    assert result["split_binding"] == "fixture"
    assert result["synthetic_fixture"] is True


# ---------------------------------------------------------------------------
# Review round 3, gap 2: the synthetic interface is bound to REAL
# provenance. The marker is an untrusted claim, so a marked manifest
# carrying production provenance (the approved archive digest, or the
# approved stored identity) is refused even when its partitions have
# been edited first, which is the route that defeated the old
# equal-the-approved-identity check.
# ---------------------------------------------------------------------------

def _edited_production_fixture_copy(split_manifest):
    """The reviewer's witness manifest, rebuilt fresh every call.

    Copy of the committed manifest, connection 25 moved from test to
    validation, then the synthetic declaration added. The stored
    identity is left as shipped: the witness keeps every approved
    provenance fact except the edited groups.
    """
    tampered = copy.deepcopy(split_manifest)
    groups = tampered["groups"]["aspirin"]
    groups["test"] = [c for c in groups["test"] if c != "25"]
    groups["validation"] = groups["validation"] + ["25"]
    tampered["synthetic"] = {"reason": "configuration smoke test"}
    return tampered


def _recovery_start_set_from_synth(landscape, manifest, classification):
    """build_start_set over the conftest synthetic landscape, rule FIXED."""
    from evaluation.start_set import build_start_set

    return build_start_set(
        manifest, landscape["records"], landscape["policy"],
        {"kind": "first_stored_frame"}, classification)


def _force_spy_arm_for_synthetic(counter):
    """A zero-force arm over the conftest SPECIES that counts real calls."""
    from evaluation.adapter import callable_arm

    def zero_force(positions):
        counter["calls"] += 1
        return np.zeros(np.asarray(positions).shape, dtype=np.float64)

    return callable_arm(zero_force, species=SPECIES, arm_id="synthetic-spy")


def test_edited_production_manifest_marked_synthetic_is_refused(
        recovery_landscape, split_manifest):
    # The witness from review round 3, finding 2: moving the test
    # connection changes the recomputed identity, so the old guard's
    # equality check no longer fired and the manifest bound as
    # 'fixture'. The synthetic path now refuses on the APPROVED archive
    # digest the copy still carries — a fact no partition edit removes.
    from evaluation.runner import SplitBindingError

    counter = {"calls": 0}
    tampered = _edited_production_fixture_copy(split_manifest)
    with pytest.raises(SplitBindingError) as excinfo:
        _runner_call(recovery_landscape,
                     copy.deepcopy(recovery_landscape.manifest),
                     arm=_force_spy_arm(recovery_minimum(0), counter),
                     split_manifest=tampered)
    message = str(excinfo.value)
    assert "APPROVED archive digest" in message
    assert "must be bound as production" in message
    assert counter["calls"] == 0


def test_edited_production_manifest_marked_synthetic_identity_removed_is_refused(
        recovery_landscape, split_manifest):
    # The variant: the same witness with the stored identity field
    # REMOVED. Removing a field must not help the relabelling; the
    # approved archive digest still identifies the copy as production
    # provenance and the binding still refuses before any force call.
    from evaluation.runner import SplitBindingError

    counter = {"calls": 0}
    tampered = _edited_production_fixture_copy(split_manifest)
    tampered.pop("identity", None)
    with pytest.raises(SplitBindingError) as excinfo:
        _runner_call(recovery_landscape,
                     copy.deepcopy(recovery_landscape.manifest),
                     arm=_force_spy_arm(recovery_minimum(0), counter),
                     split_manifest=tampered)
    message = str(excinfo.value)
    assert "APPROVED archive digest" in message
    assert counter["calls"] == 0


def test_control_marker_only_without_partition_edit_stays_refused(
        recovery_landscape, split_manifest):
    # The control from the review: the marker added with NO partition
    # edit. This passed before the round-3 change (the recomputed
    # identity still equalled the approved one) and must keep passing
    # after it: the manifest's own provenance says production.
    from evaluation.runner import SplitBindingError

    counter = {"calls": 0}
    tampered = copy.deepcopy(split_manifest)
    tampered["synthetic"] = {"reason": "configuration smoke test"}
    with pytest.raises(SplitBindingError) as excinfo:
        _runner_call(recovery_landscape,
                     copy.deepcopy(recovery_landscape.manifest),
                     arm=_force_spy_arm(recovery_minimum(0), counter),
                     split_manifest=tampered)
    message = str(excinfo.value)
    assert "bound as production" in message
    assert counter["calls"] == 0


def test_stored_approved_identity_refuses_even_with_digest_washed(
        recovery_landscape, split_manifest):
    # The second edit-proof fact on its own: a marked manifest whose
    # archive digest has been REPLACED (an attempt to wash the digest
    # specifically) is still refused when the stored identity field is
    # the approved one, so both production provenance facts
    # independently carry the refusal.
    from evaluation.runner import SplitBindingError

    counter = {"calls": 0}
    tampered = _edited_production_fixture_copy(split_manifest)
    tampered["archive_sha256"] = "washed-digest"
    with pytest.raises(SplitBindingError) as excinfo:
        _runner_call(recovery_landscape,
                     copy.deepcopy(recovery_landscape.manifest),
                     arm=_force_spy_arm(recovery_minimum(0), counter),
                     split_manifest=tampered)
    message = str(excinfo.value)
    assert "stores the APPROVED split identity" in message
    assert counter["calls"] == 0


def test_genuine_synthetic_fixture_still_binds_and_evaluates(
        synthetic_landscape, synthetic_endpoint_classification):
    # The flip side: a genuine fixture (its own digest, its own groups,
    # no approved provenance anywhere) keeps binding as 'fixture' and
    # keeps evaluating under the round-3 binding, so the guard against
    # relabelled production manifests did not close the fixture path.
    from evaluation.runner import evaluate_starts

    from conftest import build_minima_pool

    manifest = copy.deepcopy(synthetic_landscape["manifest"])
    start_set = _recovery_start_set_from_synth(
        synthetic_landscape, manifest, synthetic_endpoint_classification)
    counter = {"calls": 0}
    result = evaluate_starts(
        start_set, _force_spy_arm_for_synthetic(counter),
        _runner_procedure(), synthetic_endpoint_classification,
        build_minima_pool(synthetic_landscape), synthetic_landscape["policy"],
        split_manifest=manifest, authorise_test_partition=True)
    assert result["split_binding"] == "fixture"
    assert result["synthetic_fixture"] is True
    assert result["branch_count"] == len(start_set)
    assert counter["calls"] > 0
    for row in result["branches"].values():
        assert row["_audit_record"]["split_binding"] == "fixture"


def test_production_binding_with_policy_off_the_approved_one_is_refused(
        recovery_landscape, split_manifest):
    # A production binding whose EFFECTIVE preparation policy differs
    # from the approved manifest's policy is refused before any force
    # call, naming the first differing field with both values — being
    # an instance of PreparationPolicy is no longer enough.
    from dataclasses import replace

    from evaluation.runner import SplitBindingError

    approved = PreparationPolicy(**split_manifest["policy"])
    assert approved.endpoint_tolerance == 0.05
    drifted = replace(approved, endpoint_tolerance=0.1)
    counter = {"calls": 0}
    with pytest.raises(SplitBindingError) as excinfo:
        _runner_call(recovery_landscape,
                     copy.deepcopy(recovery_landscape.manifest),
                     split_manifest=split_manifest, policy=drifted,
                     arm=_force_spy_arm(recovery_minimum(0), counter))
    message = str(excinfo.value)
    assert "'endpoint_tolerance'" in message
    assert "0.1" in message and "0.05" in message
    assert "match the approved manifest's policy" in message
    assert counter["calls"] == 0


def test_production_binding_runs_the_real_archive_slice_end_to_end(
        aspirin_records, split_manifest, tmp_path):
    # The positive production path, end to end on the REAL archive: the
    # committed manifest under the approved policy binds 'approved',
    # evaluates a real validation connection through the runner, and
    # every saved audit record names the production binding.
    import json as _json

    from evaluation.adapter import callable_arm
    from evaluation.reporter import build_report, render
    from evaluation.runner import evaluate_starts
    from evaluation.start_set import build_start_set

    policy = PreparationPolicy(**split_manifest["policy"])
    start_set = build_start_set(
        split_manifest, aspirin_records, policy,
        {"kind": "first_stored_frame"})
    keys = ("aspirin/78/plus", "aspirin/78/minus")
    slice_set = {key: start_set[key] for key in keys}
    classification = _real_classification_payload()
    references = {}
    for key in keys:
        record_id = classification[key]["reference_class"]
        references[record_id] = next(
            record for record in aspirin_records
            if record.record_id == record_id)

    def zero_force(positions):
        return np.zeros(np.asarray(positions).shape, dtype=np.float64)

    arm = callable_arm(zero_force, species=slice_set[keys[0]]["species"],
                       arm_id="flat-zero")
    result = evaluate_starts(
        slice_set, arm, _runner_procedure(), classification, references,
        policy, split_manifest=copy.deepcopy(split_manifest),
        seen_minimum=True)
    # The production binding fired and the branches were evaluated.
    assert result["split_binding"] == "approved"
    assert result["synthetic_fixture"] is False
    assert result["branch_count"] == 2
    for row in result["branches"].values():
        assert row["partition"] == "validation"
        assert row["converged"] is True
        assert row["category"] == "unmatched_converged"
        assert row["_audit_record"]["split_binding"] == "approved"
    # The declaration survives into the SAVED per-branch evidence.
    report, diagnostics = build_report(result["branches"])
    out = render(report, diagnostics, tmp_path, stem="prod-binding")
    saved = _json.loads((tmp_path / "prod-binding.json").read_text())
    rows = {row["branch_key"]: row
            for row in saved["diagnostics"]["per_branch"]}
    assert sorted(rows) == sorted(keys)
    for key in keys:
        assert rows[key]["split_binding"] == "approved"


def test_fixture_run_saved_audit_names_the_fixture_binding(
        recovery_landscape, frozen_recovery_manifest, tmp_path):
    # The fixture's saved per-branch evidence names ITS binding too:
    # a reader of the diagnostics sees which contract each run ran under.
    import json as _json

    from evaluation.reporter import build_report, render

    result = _runner_call(recovery_landscape, frozen_recovery_manifest,
                          authorise_test_partition=True, seen_minimum=True)
    assert result["split_binding"] == "fixture"
    report, diagnostics = build_report(result["branches"])
    render(report, diagnostics, tmp_path, stem="fixture-binding")
    saved = _json.loads((tmp_path / "fixture-binding.json").read_text())
    rows = saved["diagnostics"]["per_branch"]
    assert rows
    for row in rows:
        assert row["split_binding"] == "fixture"


def _real_classification_payload():
    """The committed endpoint classification, loaded once, read-only."""
    import json
    from conftest import REPO

    return json.loads(
        (REPO / "reports" / "landscape17-endpoints.json").read_text())


# ---------------------------------------------------------------------------
# Partition overlap and completeness (review round 2): ambiguous or
# incomplete manifests are refused before any force call.
# ---------------------------------------------------------------------------

def test_connection_named_in_two_partitions_is_refused_by_name(
        recovery_landscape, frozen_recovery_manifest):
    from evaluation.runner import PartitionOverlapError

    counter = {"calls": 0}
    ambiguous = copy.deepcopy(frozen_recovery_manifest)
    # Connection 80 is now named by BOTH validation and test: a silent
    # first-match lookup would resolve it to validation, which is the
    # relabelling defect, so the manifest is refused explicitly.
    ambiguous["groups"]["aspirin"]["test"] = ["81", "80"]
    with pytest.raises(PartitionOverlapError) as excinfo:
        _runner_call(
            recovery_landscape, ambiguous, authorise_test_partition=True,
            arm=_force_spy_arm(recovery_minimum(0), counter))
    message = str(excinfo.value)
    assert "'80'" in message or '"80"' in message
    assert "validation" in message and "test" in message
    assert counter["calls"] == 0


def test_production_manifest_with_a_test_connection_moved_to_validation_is_refused(
        recovery_landscape, frozen_recovery_manifest, split_manifest):
    from evaluation.runner import SplitBindingError

    # The reviewer's counterexample, in production form: take the APPROVED
    # manifest, move a test connection into validation, and try to run it.
    # A marked synthetic fixture may arrange its own partitions, but a
    # production manifest must still recompute to the frozen identity, so
    # this is refused by the binding BEFORE any force call instead of being
    # scored under a validation label the frozen split never gave it.
    counter = {"calls": 0}
    moved = copy.deepcopy(split_manifest)
    moved.pop("identity")
    moved["groups"]["aspirin"]["validation"] = (
        moved["groups"]["aspirin"]["validation"] + ["00"])
    moved["groups"]["aspirin"]["test"] = [
        connection for connection in moved["groups"]["aspirin"]["test"]
        if connection != "00"]
    with pytest.raises(SplitBindingError) as excinfo:
        # The start set is built from the fixture landscape's own manifest,
        # whose records it names; the edited production clone goes ONLY to
        # the split binding, which is the guard under test here.
        _runner_call(
            recovery_landscape, copy.deepcopy(frozen_recovery_manifest),
            arm=_force_spy_arm(recovery_minimum(0), counter),
            split_manifest=moved)
    message = str(excinfo.value)
    assert "recomputes to identity" in message
    assert counter["calls"] == 0


def test_missing_partition_membership_is_refused_descriptively(
        recovery_landscape, frozen_recovery_manifest):
    counter = {"calls": 0}
    incomplete = copy.deepcopy(frozen_recovery_manifest)
    incomplete["groups"]["aspirin"]["validation"] = []
    start_set = _recovery_start_set(recovery_landscape, incomplete)
    # A branch whose connection is named by NO partition of the supplied
    # manifest: the run must refuse it by name, with an explicit message,
    # instead of raising a bare KeyError from the partition lookup.
    template = copy.deepcopy(next(iter(start_set.values())))
    template["molecule"] = "aspirin"
    template["connection_id"] = "99"
    template["branch"] = "plus"
    start_set["aspirin/99/plus"] = template
    with pytest.raises(ValueError) as excinfo:
        # The test-partition authorisation is given here so the branch
        # under test is the unlisted one, not the authorisation guard:
        # without it that guard fires before any connection is looked up
        # and the missing-partition path is never reached.
        _runner_call(
            recovery_landscape, incomplete, start_set=start_set,
            authorise_test_partition=True,
            arm=_force_spy_arm(recovery_minimum(0), counter))
    message = str(excinfo.value)
    assert "not named in any partition" in message
    assert "aspirin/99/plus" in message
    assert counter["calls"] == 0


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


# ---------------------------------------------------------------------------
# Item 1's own regressions: arm configuration errors, wrong shapes and
# nonfinite forces driven through the public evaluate_starts interface.
# ---------------------------------------------------------------------------

def test_wrong_species_callable_arm_raises_arm_configuration_error(
        recovery_landscape, frozen_recovery_manifest):
    from evaluation.adapter import ArmConfigurationError, callable_arm
    from evaluation.runner import evaluate_starts

    # An arm built for three DIFFERENT symbols: every species comparison
    # against the fixture graph fails by construction.
    wrong_species = ("N", "N", "N")
    underlying_calls = {"n": 0}

    def force(positions):
        underlying_calls["n"] += 1
        return np.zeros(np.asarray(positions).shape, dtype=np.float64)

    wrong_arm = callable_arm(force, species=wrong_species, arm_id="wrong-species")
    start_set = _recovery_start_set(recovery_landscape, frozen_recovery_manifest)
    validation_only = {key: entry for key, entry in start_set.items()
                       if key.split("/")[1] == "80"}
    with pytest.raises(ArmConfigurationError) as excinfo:
        evaluate_starts(
            validation_only, wrong_arm, _runner_procedure(),
            _recovery_classification(recovery_landscape),
            recovery_landscape.references(), recovery_landscape.policy(),
            split_manifest=frozen_recovery_manifest,
        )
    # The refusal is a configuration error, raised BEFORE any force call:
    # the arm's underlying supplier is never invoked and no branch is
    # scored anywhere in the returned structure.
    assert underlying_calls["n"] == 0
    message = str(excinfo.value)
    assert "species mismatch" in message
    assert "wrong-species" in message


def test_wrong_species_arm_refusal_is_loud_even_under_compatible_override(
        recovery_landscape, frozen_recovery_manifest):
    from evaluation.adapter import ArmConfigurationError, callable_arm
    from evaluation.runner import evaluate_starts

    # The "compatible" opt-out relaxes the runner's own pre-validation,
    # but the arm's contract still raises ArmConfigurationError on the
    # first call and relax routes it out of the scored path: a wrongly
    # configured arm is never scored, whatever the override says.
    def force(positions):
        return np.zeros(np.asarray(positions).shape, dtype=np.float64)

    wrong_arm = callable_arm(force, species=("N", "N", "N"),
                             arm_id="wrong-species-2")
    start_set = _recovery_start_set(recovery_landscape, frozen_recovery_manifest)
    validation_only = {key: entry for key, entry in start_set.items()
                       if key.split("/")[1] == "80"}
    with pytest.raises(ArmConfigurationError):
        evaluate_starts(
            validation_only, wrong_arm, _runner_procedure(),
            _recovery_classification(recovery_landscape),
            recovery_landscape.references(), recovery_landscape.policy(),
            split_manifest=frozen_recovery_manifest,
            arm_start_compatibility="compatible",
        )


def test_finite_wrong_shape_force_reports_invalid_force_shape_not_nonfinite(
        recovery_landscape, frozen_recovery_manifest):
    from evaluation.adapter import callable_arm
    from evaluation.runner import evaluate_starts

    # A FINITE force array of the wrong shape (transposed: (3, N) instead
    # of (N, 3)): a refusal, never physics, and never a nonfinite one.
    def transposed_force(positions):
        return np.zeros((np.asarray(positions).shape[1],
                         np.asarray(positions).shape[0]), dtype=np.float64)

    arm = callable_arm(transposed_force, species=SPECIES, arm_id="transposed")
    start_set = _recovery_start_set(recovery_landscape, frozen_recovery_manifest)
    validation_only = {key: entry for key, entry in start_set.items()
                       if key.split("/")[1] == "80"}
    result = evaluate_starts(
        validation_only, arm, _runner_procedure(),
        _recovery_classification(recovery_landscape),
        recovery_landscape.references(), recovery_landscape.policy(),
        split_manifest=frozen_recovery_manifest,
    )
    for row in result["branches"].values():
        assert row["status"] == "nonconverged"
        assert row["stop_reason"] == "invalid_force_shape"
        assert row["forces_finite"] is True
        assert row["nonfinite_force"] is False
        assert row["force_evaluation_count"] == 1
        assert row["category"] == "nonconverged"
        details = row["refusal_details"]
        assert len(details) == 1
        assert details[0]["reason"] == "invalid_force_shape"
        assert details[0]["arm_id"] == "transposed"
        assert "force shape" in details[0]["detail"]


def test_genuine_nan_force_reports_nonfinite_and_not_forces_finite(
        recovery_landscape, frozen_recovery_manifest):
    from evaluation.adapter import callable_arm
    from evaluation.runner import evaluate_starts

    # A genuine NaN inside an otherwise correctly shaped force array.
    def poisoned_force(positions):
        array = np.asarray(positions, dtype=np.float64).copy()
        array[..., 0] = np.nan
        return array

    arm = callable_arm(poisoned_force, species=SPECIES, arm_id="poisoned")
    start_set = _recovery_start_set(recovery_landscape, frozen_recovery_manifest)
    validation_only = {key: entry for key, entry in start_set.items()
                       if key.split("/")[1] == "80"}
    result = evaluate_starts(
        validation_only, arm, _runner_procedure(),
        _recovery_classification(recovery_landscape),
        recovery_landscape.references(), recovery_landscape.policy(),
        split_manifest=frozen_recovery_manifest,
    )
    for row in result["branches"].values():
        assert row["status"] == "nonconverged"
        assert row["stop_reason"] == "nonfinite_force"
        assert row["nonfinite_force"] is True
        assert row["forces_finite"] is False
        assert row["force_evaluation_count"] == 1
        assert row["category"] == "nonconverged"


def test_nonnumeric_supplier_return_is_a_recorded_refusal(
        recovery_landscape, frozen_recovery_manifest):
    from evaluation.adapter import callable_arm
    from evaluation.runner import evaluate_starts

    # A supplier returning a nonnumeric value: the arm's own contract
    # records the refusal, and the runner's accounting turns it into the
    # recorded branch failure instead of crashing the accounting.
    def text_force(positions):
        return "not a force array"

    arm = callable_arm(text_force, species=SPECIES, arm_id="texty")
    start_set = _recovery_start_set(recovery_landscape, frozen_recovery_manifest)
    validation_only = {key: entry for key, entry in start_set.items()
                       if key.split("/")[1] == "80"}
    result = evaluate_starts(
        validation_only, arm, _runner_procedure(),
        _recovery_classification(recovery_landscape),
        recovery_landscape.references(), recovery_landscape.policy(),
        split_manifest=frozen_recovery_manifest,
    )
    for row in result["branches"].values():
        assert row["status"] == "nonconverged"
        assert row["stop_reason"] == "invalid_force_shape"
        assert row["refusal_details"][0]["arm_id"] == "texty"
        assert "nonnumeric" in row["refusal_details"][0]["detail"]


def test_two_arms_with_identical_refusal_wording_stay_distinguishable(
        recovery_landscape, frozen_recovery_manifest):
    from evaluation.adapter import MalformedForceReturn, callable_arm
    from evaluation.runner import evaluate_starts

    # Two arms whose suppliers raise the SAME declared refusal wording
    # (MalformedForceReturn, the round-3 declared malformed-return type):
    # the recorded refusal details name WHICH arm refused, so the audit
    # trail cannot collapse the two runs into one anonymous fact.
    def refusing_force(positions):
        raise MalformedForceReturn("the same refusal wording")

    arm_w = callable_arm(refusing_force, species=SPECIES, arm_id="arm-with-shape")
    arm_s = callable_arm(refusing_force, species=SPECIES,
                         arm_id="arm-with-scalar")
    start_set = _recovery_start_set(recovery_landscape, frozen_recovery_manifest)
    validation_only = {key: entry for key, entry in start_set.items()
                       if key.split("/")[1] == "80"}
    first = evaluate_starts(
        validation_only, arm_w, _runner_procedure(),
        _recovery_classification(recovery_landscape),
        recovery_landscape.references(), recovery_landscape.policy(),
        split_manifest=frozen_recovery_manifest,
    )
    second = evaluate_starts(
        validation_only, arm_s, _runner_procedure(),
        _recovery_classification(recovery_landscape),
        recovery_landscape.references(), recovery_landscape.policy(),
        split_manifest=frozen_recovery_manifest,
    )
    first_details = [detail for row in first["branches"].values()
                     for detail in row["refusal_details"]]
    second_details = [detail for row in second["branches"].values()
                      for detail in row["refusal_details"]]
    assert all(detail["arm_id"] == "arm-with-shape" for detail in first_details)
    assert all(detail["arm_id"] == "arm-with-scalar" for detail in second_details)
    assert first_details[0]["arm_id"] != second_details[0]["arm_id"]


def test_arbitrary_supplier_value_error_propagates_and_scores_nothing(
        recovery_landscape, frozen_recovery_manifest):
    from evaluation.adapter import callable_arm
    from evaluation.runner import evaluate_starts

    # Review round 3, gap 3, end to end: a supplier whose INTERNAL
    # configuration is broken raises an ordinary ValueError. That is not
    # a refusal — it propagates out of evaluate_starts and scores
    # nothing, instead of being recorded as scored nonconverged branches.
    calls = {"n": 0}

    def force(positions):
        calls["n"] += 1
        raise ValueError("model configuration: missing radial basis parameter")

    arm = callable_arm(force, species=SPECIES, arm_id="raises-valueerror")
    start_set = _recovery_start_set(recovery_landscape, frozen_recovery_manifest)
    validation_only = {key: entry for key, entry in start_set.items()
                       if key.split("/")[1] == "80"}
    with pytest.raises(ValueError) as excinfo:
        evaluate_starts(
            validation_only, arm, _runner_procedure(),
            _recovery_classification(recovery_landscape),
            recovery_landscape.references(), recovery_landscape.policy(),
            split_manifest=frozen_recovery_manifest,
        )
    assert ("model configuration: missing radial basis parameter"
            in str(excinfo.value))
    # The first branch's attempt happened and nothing was scored: the
    # run aborted on the first undeclared supplier error.
    assert calls["n"] == 1


def test_mixed_species_starts_are_refused_with_zero_supplier_calls(
        recovery_landscape, frozen_recovery_manifest):
    from evaluation.adapter import ArmConfigurationError
    from evaluation.runner import evaluate_starts

    # Review round 3, gap 4, end to end in the DEFAULT strict mode: one
    # start in the set carries a species tuple the arm was not built
    # for. The plan-building preflight must refuse the whole set BEFORE
    # any force call: zero supplier calls, the branch and both species
    # tuples named in the error.
    counter = {"calls": 0}
    arm = _force_spy_arm(recovery_minimum(0), counter)
    start_set = _recovery_start_set(recovery_landscape, frozen_recovery_manifest)
    validation_only = {key: entry for key, entry in start_set.items()
                       if key.split("/")[1] == "80"}
    keys = sorted(validation_only)
    second = keys[1]
    entry = dict(validation_only[second])
    entry["species"] = ("N", "N", "N")
    validation_only[second] = entry
    with pytest.raises(ArmConfigurationError) as excinfo:
        evaluate_starts(
            validation_only, arm, _runner_procedure(),
            _recovery_classification(recovery_landscape),
            recovery_landscape.references(), recovery_landscape.policy(),
            split_manifest=frozen_recovery_manifest,
        )
    message = str(excinfo.value)
    assert "species mismatch" in message
    assert "('N', 'N', 'N')" in message
    assert "('C', 'H', 'O')" in message
    assert counter["calls"] == 0