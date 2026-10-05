"""The runner: one relaxation-and-classification pass over a frozen start set.

This is the composition the implementation plan promises (Task 3's
`evaluate_starts` interface): it drives every branch of a
`start_set.build_start_set` output through a single arm's force supplier
with the one frozen `relax` procedure, classifies every endpoint with
`outcomes.classify_outcome`, and reports the per-branch and
per-connection outcomes. It owns no physics and no scoring of its own:
every verdict comes from the pieces that already exist and are tested.

Arm boundary (explicit, per the review): the adapter's arms accept
element-symbol species tuples in the arm's fixed atom order and return
an info dict next to the forces. The runner passes each start's own
species tuple straight from the start set to the arm and checks the
info's status; it does NOT translate species into atomic numbers. The
atomic-number encoding belongs to the model interface behind
`arm_from_checkpoint` (the shared adapter agreement with Shedrack),
which converts checkpoints' atomic numbers to element symbols before
this runner ever sees a force. Nothing in this package bypasses the
adapter to talk to a model directly.

Partition authorisation (frozen in the design spec, section 7): the
test partition is evaluated once, after freeze, under a separate
authorisation from the one that approves the implementation. A start
set that contains test-partition branches is refused with the branch
names attached unless `authorise_test_partition=True` is passed by the
authorised caller. Train and validation partitions need no
authorisation: the fixture slice in the plan runs on training
connections only, and the start-set module also covers validation.
"""
import numpy as np

from evaluation.outcomes import classify_outcome
from evaluation.relax import relax

PARTITIONS = ("train", "validation", "test")
CATEGORIES = ("matched_reference", "unmatched_converged", "ambiguous",
              "unresolved", "nonconverged")


class PartitionAuthorisationError(ValueError):
    """Test-partition branches were requested without authorisation.

    The design spec's test discipline (section 7) reserves the test
    partition for one evaluation under a separate authorisation. This
    error carries the offending branch names so the refusal is
    auditable, and is not raised when the authorised flag is passed.
    """


def _split_key(branch_key):
    parts = branch_key.split("/")
    if len(parts) != 3:
        raise ValueError(
            f"branch key {branch_key!r} must be molecule/connection/branch")
    return parts[0], parts[1], parts[2]


def _connection_partition(branch_key, split_manifest):
    """The partition whose connection list names this branch's connection."""
    molecule, connection_id, _branch = _split_key(branch_key)
    groups = split_manifest.get("groups", {}).get(molecule) if isinstance(
        split_manifest, dict) else None
    if groups is None:
        raise ValueError(
            f"{branch_key!r}: the supplied manifest has no groups for "
            f"molecule {molecule!r}")
    for partition in PARTITIONS:
        if connection_id in groups.get(partition, []):
            return partition
    raise ValueError(
        f"{branch_key!r}: its connection is not named in any partition of "
        f"the supplied manifest")


def _own_reference(branch_key, endpoint_classification, references):
    """The branch's own reference minimum record, resolved from the pool.

    The classification supplies the branch's committed reference_class
    (a record id); the pool may be keyed by aliases (for example "m0"),
    so the record whose record_id equals the classified id is looked up.
    """
    entry = endpoint_classification.get(branch_key)
    if not isinstance(entry, dict) or entry.get("status") != "matched":
        observed = repr(entry.get("status")) if isinstance(entry, dict) else repr(entry)
        raise ValueError(
            f"{branch_key!r}: no 'matched' endpoint classification "
            f"(got {observed}); the runner can only evaluate a branch whose "
            f"reference minimum is committed")
    reference_class = entry.get("reference_class")
    for pool_key, record in references.items():
        if getattr(record, "record_id", None) == reference_class:
            return pool_key, record
    raise ValueError(
        f"{branch_key!r}: the classified reference {reference_class!r} is "
        f"not in the supplied reference pool")


def _final_record(branch_key, start_entry, positions):
    """The endpoint wrapped in a FrameRecord for the structural comparison."""
    from force_consistency.data.types import FrameRecord

    frozen = np.array(positions, dtype=np.float64, copy=True)
    frozen.setflags(write=False)
    record_id = f"{branch_key}/final"
    return FrameRecord(
        record_id=record_id, molecule=start_entry["molecule"], kind="path",
        source_path=record_id, connection_id=start_entry["connection_id"],
        branch=start_entry["branch"], frame_index=None,
        species=tuple(start_entry["species"]), positions=frozen,
        energy=0.0, forces=np.zeros_like(frozen), source_sha256="0" * 64,
    )


def _arm_force_supplier(arm, species, record_id):
    """Bind this branch's start identity into the arm's force callable."""
    if arm["status"] != "ready":
        raise ValueError(
            f"arm {arm.get('arm_id')!r} is not ready ({arm['status']!r}): "
            f"{arm.get('refusal_reason')}")

    def force_fn(positions):
        out, info = arm["forces"](positions, species=species, record_id=record_id)
        if info.get("status") != "ok":
            raise RuntimeError(
                f"arm {arm.get('arm_id')!r} returned a non-ok info without a "
                f"declared refusal exception: {info!r}")
        return out

    return force_fn


def evaluate_starts(start_set, arm, procedure, endpoint_classification,
                    references, policy, *, split_manifest=None,
                    seen_minimum=None, authorise_test_partition=False):
    """Relax and classify every branch of a start set under one arm.

    start_set: the dict `start_set.build_start_set` returns (branch key
        to start entry carrying the start record id, structure, species
        and molecule/connection/branch naming).
    arm: a ready arm dict from the adapter (analytic_fixture_arm or
        callable_arm today; arm_from_checkpoint through the same
        interface when a checkpoint exists).
    procedure: the frozen relaxation settings dict (relax validates it).
    endpoint_classification: the committed endpoint classification as a
        parsed dict; every branch key must be "matched" with a
        reference_class resolvable in the reference pool.
    references: the molecule's reference-minimum pool for the structural
        comparison (record id or alias keys, FrameRecord values).
    policy: the PreparationPolicy whose tolerances decide matching.
    split_manifest: the frozen split manifest, needed to know each
        branch's partition for the authorisation check. Required: the
        authorisation discipline is about partitions, so guessing one is
        not acceptable.
    seen_minimum: optional mapping of branch key (or constant) to the
        frozen manifest's seen_minimum flag, carried onto every outcome.
    authorise_test_partition: the separate authorisation for the test
        partition. False (default) refuses test-partition branches with
        the branch names in the error.

    Returns a dict with the per-branch and per-connection outcomes (see
    the module docstring for the arm boundary and the authorisation
    discipline).
    """
    if not isinstance(start_set, dict) or not start_set:
        raise ValueError("start_set must be the dict build_start_set returns")
    if split_manifest is None:
        raise ValueError(
            "split_manifest is required: the authorisation check reads each "
            "branch's partition from it and never guesses")
    if not isinstance(endpoint_classification, dict):
        raise ValueError(
            "endpoint_classification is required: every branch's own "
            "reference minimum comes from its committed reference_class")

    # First pass: partition and reference resolution for every branch,
    # BEFORE any force evaluation, so a refusal stops the run before
    # budget is spent on branches that must not be evaluated.
    plans = []
    refused = []
    for branch_key, start_entry in start_set.items():
        partition = _connection_partition(branch_key, split_manifest)
        if partition == "test" and not authorise_test_partition:
            refused.append(branch_key)
            continue
        if not isinstance(start_entry, dict):
            raise ValueError(f"{branch_key!r}: start entry must be a dict")
        pool_key, own = _own_reference(branch_key, endpoint_classification,
                                       references)
        plans.append((branch_key, start_entry, partition, pool_key, own))
    if refused:
        raise PartitionAuthorisationError(
            f"test-partition branches cannot be evaluated without the "
            f"separate authorisation the design spec reserves for the test "
            f"partition (evaluated once, after freeze): {sorted(refused)}. "
            f"Pass authorise_test_partition=True only when that "
            f"authorisation exists.")

    # Second pass: relax and classify the admitted branches.
    branches = {}
    for branch_key, start_entry, partition, _pool_key, own in plans:
        species = tuple(start_entry["species"])
        force_fn = _arm_force_supplier(arm, species,
                                       f"{start_entry['record_id']}/relax")
        relaxation = relax(start_entry["positions"], force_fn, procedure)
        final = _final_record(branch_key, start_entry, relaxation["final_positions"])
        flag = seen_minimum
        if isinstance(flag, dict):
            flag = flag.get(branch_key)
        if callable(flag):
            flag = flag(branch_key)
        outcome = classify_outcome(
            final, own, references, policy,
            budget_exhausted=relaxation["budget_exhausted"],
            converged=relaxation["converged"],
            forces_finite=relaxation["forces_finite"],
            seen_minimum=flag,
            stop_reason=relaxation["failure_reason"],
        )
        branches[branch_key] = {
            "branch_key": branch_key,
            "molecule": start_entry["molecule"],
            "connection_id": start_entry["connection_id"],
            "branch": start_entry["branch"],
            "partition": partition,
            "seen_minimum": outcome["seen_minimum"],
            "start_record_id": start_entry["record_id"],
            "final_record_id": final.record_id,
            "status": relaxation["status"],
            "category": outcome["category"],
            "converged": outcome["converged"],
            "budget_exhausted": outcome["budget_exhausted"],
            "forces_finite": outcome["forces_finite"],
            "nonfinite_force": outcome["nonfinite_force"],
            "stop_reason": outcome["stop_reason"],
            "force_evaluation_count": relaxation["force_evaluation_count"],
            "step_count": relaxation["step_count"],
            "matching_status": (outcome["matching"] or {}).get("status"),
            "accepted_classes": outcome["accepted_classes"],
            "rmsd_A": outcome["rmsd_A"],
        }

    connections = {}
    for branch_key, row in branches.items():
        connection_key = f"{row['molecule']}/{row['connection_id']}"
        entry = connections.setdefault(connection_key, {
            "connection_key": connection_key,
            "molecule": row["molecule"],
            "connection_id": row["connection_id"],
            "branches": [],
            "evaluated_branches": 0,
            "counts": {name: 0 for name in CATEGORIES},
        })
        entry["branches"].append(row["branch"])
        entry["evaluated_branches"] += 1
        entry["counts"][row["category"]] += 1
    for entry in connections.values():
        entry["branches"] = sorted(entry["branches"])

    return {
        "schema": 1,
        "arm": {"kind": arm.get("kind"), "arm_id": arm.get("arm_id"),
                "provides_energy": arm.get("provides_energy")},
        "procedure": dict(procedure),
        "authorise_test_partition": bool(authorise_test_partition),
        "branch_count": len(branches),
        "branches": branches,
        "connections": connections,
    }