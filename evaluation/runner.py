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

Per-branch audit record (review item 2): every branch outcome the
runner returns carries an "_audit_record" dict of plain-JSON evidence
for the reporter to retain: the arm identity actually used, the start
record id, the branch's own reference identity, the endpoint
coordinates, the RMSD and mapping facts, the structured failure facts
(including the refusal naming which arm refused), the force-evaluation
and step counts, the per-branch wall-clock timing, and the effective
settings the run actually used. The evidence is explicit names and
numbers only: no opaque object is stored as its repr, so the saved
diagnostics round-trip through json.loads without loss.

Partition authorisation (frozen in the design spec, section 7): the
test partition is evaluated once, after freeze, under a separate
authorisation from the one that approves the implementation. A start
set that contains test-partition branches is refused with the branch
names attached unless `authorise_test_partition=True` is passed by the
authorised caller. Train and validation partitions need no
authorisation: the fixture slice in the plan runs on training
connections only, and the start-set module also covers validation.
"""
import time

import numpy as np

from force_consistency.data.splits import canonical_hash
from force_consistency.data.source import SHA256 as ARCHIVE_SHA256
from force_consistency.data.types import PreparationPolicy

from evaluation.adapter import _validated_positions
from evaluation.outcomes import classify_outcome
from evaluation.relax import relax

PARTITIONS = ("train", "validation", "test")
CATEGORIES = ("matched_reference", "unmatched_converged", "ambiguous",
              "unresolved", "nonconverged")

# The manifest key that carries the explicit synthetic-fixture declaration
# (review round 2, fixtures): {"synthetic": {"reason": ...}} inside the
# manifest itself, so the public call pattern needs no extra argument and
# the identity binding stays default-on for every manifest that does not
# carry the marker.
MANIFEST_SYNTHETIC_KEY = "synthetic"

# The approved split (review round 2): what the target branch's own frozen
# shape recovery configuration declares (configs/aspirin-shape-recovery.json,
# with the archive digest force_consistency.data.source freezes). Production
# evaluation verifies the supplied manifest recomputes to this identity
# before any force call, so relabelling a manifest cannot admit a test
# connection under another partition's label. The identity covers the
# archive digest, the connection inventory, the partition groups and the
# preparation policy, so any of those edits changes the identity.
APPROVED_SPLIT_MANIFEST_PATH = "manifests/landscape17-aspirin-split.json"
APPROVED_SPLIT_IDENTITY = (
    "73d5475bfc1f1158da62b939f5431b1781db886443cece2cdb297c8dd645133a")
APPROVED_ARCHIVE_SHA256 = ARCHIVE_SHA256


class SplitBindingError(ValueError):
    """Production evaluation was handed a manifest that is not the approved one.

    The partition discipline only holds when the labels come from the
    reviewed split, so production evaluation fails closed when the
    manifest identity or its archive digest does not match the frozen
    declaration, naming which fact disagreed. Synthetic fixtures are not
    production evaluation: they are explicitly identified by the
    manifest's own "synthetic" declaration (reason included) instead of
    impersonating the approved split.
    """


class PartitionAuthorisationError(ValueError):
    """Test-partition branches were requested without authorisation.

    The design spec's test discipline (section 7) reserves the test
    partition for one evaluation under a separate authorisation. This
    error carries the offending branch names so the refusal is
    auditable, and is not raised when the authorised flag is passed.
    """


class PartitionOverlapError(ValueError):
    """A connection is named by more than one partition of the manifest.

    A first-match lookup would silently resolve the ambiguity toward the
    earlier partition, which is exactly the relabelling defect the review
    demonstrated, so the manifest is refused before any force call and
    the error names the connection and BOTH partitions.
    """


def _split_key(branch_key):
    parts = branch_key.split("/")
    if len(parts) != 3:
        raise ValueError(
            f"branch key {branch_key!r} must be molecule/connection/branch")
    return parts[0], parts[1], parts[2]


def _connection_partition(branch_key, split_manifest):
    """The partition whose connection list names this branch's connection.

    The manifest is required to be unambiguous: a connection named by two
    partitions would silently resolve to the first one in the PARTITIONS
    order, which is exactly the relabelling defect the review
    demonstrated, so the manifest is refused instead (review round 2). A
    connection named by none, and a molecule with no groups entry at all,
    are the same fail-closed case.
    """
    molecule, connection_id, _branch = _split_key(branch_key)
    if not isinstance(split_manifest, dict):
        raise ValueError(
            f"{branch_key!r}: the supplied split manifest is not a dict")
    groups_molecule = split_manifest.get("groups", {}).get(molecule)
    if not isinstance(groups_molecule, dict):
        raise ValueError(
            f"{branch_key!r}: the supplied manifest has no groups entry for "
            f"molecule {molecule!r}")
    matches = [partition for partition in PARTITIONS
               if connection_id in groups_molecule.get(partition, [])]
    if len(matches) > 1:
        raise PartitionOverlapError(
            f"{branch_key!r}: connection {connection_id!r} is named by "
            f"{len(matches)} partitions ({', '.join(matches)}); the supplied "
            f"manifest is ambiguous, so it is refused instead of guessing")
    if not matches:
        raise ValueError(
            f"{branch_key!r}: its connection is not named in any partition of "
            f"the supplied manifest")
    return matches[0]


def _validate_connection_partitions(start_set, split_manifest):
    """Fail closed on every ambiguous manifest BEFORE any force call.

    _connection_partition raises on the first ambiguous branch it sees,
    which would already be inside the per-branch first pass but after
    earlier branches passed their other checks. A manifest validation
    pass over every branch's connection here means an ambiguous manifest
    is reported in full, once, before any branch is admitted or any
    reference is resolved.
    """
    seen_connections = {}
    for branch_key in start_set:
        molecule, connection_id, _branch = _split_key(branch_key)
        connection_key = f"{molecule}/{connection_id}"
        if connection_key in seen_connections:
            continue
        seen_connections[connection_key] = _connection_partition(
            branch_key, split_manifest)
    return seen_connections


def _manifest_split_identity(split_manifest):
    """The manifest's identity, recomputed the way the preparation froze it.

    Covers the archive digest, the connection inventory, the partition
    groups and the preparation policy (the same four payloads
    force_consistency.data.splits.assign_groups hashes), so any edit to
    any of them changes the identity.
    """
    fields = tuple(split_manifest.get(key) for key in
                   ("archive_sha256", "connections", "groups", "policy"))
    if any(field is None for field in fields):
        missing = [key for key, field in
                   zip(("archive_sha256", "connections", "groups", "policy"),
                       fields) if field is None]
        raise SplitBindingError(
            f"the supplied split manifest is missing {', '.join(missing)}, "
            f"so no identity can be recomputed to bind it")
    return canonical_hash(dict(zip(
        ("archive_sha256", "connections", "groups", "policy"), fields)))


def _manifest_synthetic_declaration(split_manifest):
    """The manifest's own explicit synthetic-fixture declaration, or None.

    Reviewed interface (review round 2, fixtures): a synthetic fixture is
    identified INSIDE its manifest, by a "synthetic": {"reason": ...}
    declaration, never by a call-site flag. A truthy non-dict value is
    refused (the marker means one thing only), a dict value without a
    nonempty reason string is refused, and a manifest without the marker
    is not synthetic, so the escape hatch cannot be taken by accident.
    """
    if "synthetic" not in split_manifest:
        return None
    declaration = split_manifest.get("synthetic")
    if isinstance(declaration, dict):
        reason = declaration.get("reason")
        if isinstance(reason, str) and reason.strip():
            return declaration
        raise SplitBindingError(
            "the manifest's synthetic declaration carries no usable reason; "
            f"pass {{\"{MANIFEST_SYNTHETIC_KEY}\": {{\"reason\": <why this "
            "fixture exists>}}}")
    raise SplitBindingError(
        f"the manifest's synthetic marker is {declaration!r}; it must be a "
        f"dict with a nonempty 'reason' string, because an anonymous marker "
        f"cannot be audited")


def _computable_identity(split_manifest):
    """The manifest's identity when it CAN be recomputed, otherwise None.

    Fixture-side support for the relabel check inside
    _bind_split_manifest: a synthetic fixture may leave one of the four
    identity payloads (archive_sha256, connections, groups, policy)
    empty, so its identity is not recomputable and the
    relabelled-approved-manifest check has nothing to compare, which
    must skip that check alone, not make the fixture unusable. The
    production branch keeps calling the raising version directly, so a
    production manifest missing a field still fails closed.
    """
    fields = tuple(split_manifest.get(key) for key in
                   ("archive_sha256", "connections", "groups", "policy"))
    if any(field is None for field in fields):
        return None
    return canonical_hash(dict(zip(
        ("archive_sha256", "connections", "groups", "policy"), fields)))


def _bind_split_manifest(split_manifest, *, synthetic_declaration):
    """Bind the supplied manifest before any force call (review round 2).

    A manifest carrying no synthetic declaration is production: it must
    recompute to the exact approved pairing (the identity equals the
    frozen APPROVED_SPLIT_IDENTITY, the archive digest equals the frozen
    archive digest, and the caller bound a real PreparationPolicy object
    at the call site); anything else fails closed. A manifest carrying
    the explicit synthetic declaration binds as a fixture instead and is
    refused if it still carries the APPROVED identity: a production
    manifest that has been relabelled 'synthetic' must not be admitted
    as a fixture, it must go through the production binding.

    The synthetic marker lives INSIDE the manifest, so the existing
    public call pattern (split_manifest=..., authorise_test_partition=...)
    keeps working unchanged for both production and fixture runs, and a
    synthetic manifest missing the marker is refused: the escape hatch
    cannot be taken by accident or by omission.
    """
    if synthetic_declaration is None:
        identity = _computable_identity(split_manifest)
        if identity is None:
            # The manifest is neither marked as a fixture nor verifiable as
            # the approved one. Fail closed and name BOTH remedies, so the
            # refusal is actionable: bind the approved manifest, or declare
            # the fixture inside the manifest.
            missing = ", ".join(
                key for key in ("archive_sha256", "connections", "groups",
                                "policy")
                if split_manifest.get(key) is None)
            raise SplitBindingError(
                f"the supplied split manifest carries no "
                f"{MANIFEST_SYNTHETIC_KEY!r} declaration and its identity "
                f"cannot be recomputed, because {missing} is missing; "
                f"production evaluation is bound to the approved manifest "
                f"({APPROVED_SPLIT_MANIFEST_PATH}), so a manifest that cannot "
                f"prove it is that one is refused; a synthetic fixture must "
                f"carry its own explicit {MANIFEST_SYNTHETIC_KEY!r} declaration "
                f"instead of impersonating the approved manifest")
        if identity == APPROVED_SPLIT_IDENTITY:
            archive = split_manifest.get("archive_sha256")
            if archive != APPROVED_ARCHIVE_SHA256:
                raise SplitBindingError(
                    f"the supplied split manifest recomputes to the approved "
                    f"identity but carries archive digest {archive!r} instead of "
                    f"the approved {APPROVED_ARCHIVE_SHA256!r}")
            return "approved"
        raise SplitBindingError(
            f"the supplied split manifest recomputes to identity {identity!r}, "
            f"which is not the approved {APPROVED_SPLIT_IDENTITY!r} "
            f"({APPROVED_SPLIT_MANIFEST_PATH}); production evaluation is bound "
            "to that approved manifest and refuses every other one; a "
            "synthetic fixture must carry its own explicit "
            f"{MANIFEST_SYNTHETIC_KEY!r} declaration instead of impersonating "
            "the approved manifest")
    identity = _computable_identity(split_manifest)
    if identity is not None and identity == APPROVED_SPLIT_IDENTITY:
        raise SplitBindingError(
            f"the supplied manifest carries a synthetic declaration but "
            f"recomputes to the approved identity "
            f"({APPROVED_SPLIT_MANIFEST_PATH}); that manifest is production "
            f"evaluation and must be bound as production, not relabelled")
    return "fixture"


def _approved_policy():
    """The approved manifest's preparation policy, with the real data."""
    from force_consistency.data.types import PreparationPolicy as Policy
    return Policy()


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
    """Bind this branch's start identity into the arm's force callable.

    A supplier whose underlying callable returns something nonnumeric
    raises the declared in-compute ValueError here (np.asarray with an
    explicit dtype converts the malformed return into the same refusal
    fact callable_arm already declares), so the accounting records the
    attempt as a refused force instead of crashing.
    """
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
        array = np.asarray(out, dtype=np.float64)
        if array.size == 0:
            raise ValueError(
                f"arm {arm.get('arm_id')!r}: the force value {out!r} is "
                f"nonnumeric or empty, so no force array exists to compare")
        return array

    return force_fn


def _plain_json(value):
    """Convert one audit value to plain-JSON data, refusing opaque objects.

    numpy arrays become nested float lists, numpy scalars and booleans
    become Python numbers and bools, dicts and lists normalize
    recursively, and FrameRecord-like objects become explicit named
    fields (the same projection the reporter applies to its audit
    records). A value with no understood structure raises TypeError
    instead of being str()ed into an audit field, because a stringified
    repr inside a saved report pretends to be readable data.
    """
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float64).tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, dict):
        return {str(k): _plain_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    if hasattr(value, "record_id"):
        # A FrameRecord-like object: name its fields explicitly.
        species = getattr(value, "species", None)
        positions = getattr(value, "positions", None)
        return {
            "record_id": value.record_id,
            "molecule": getattr(value, "molecule", None),
            "kind": getattr(value, "kind", None),
            "source_path": getattr(value, "source_path", None),
            "connection_id": getattr(value, "connection_id", None),
            "branch": getattr(value, "branch", None),
            "frame_index": getattr(value, "frame_index", None),
            "species": ([_plain_json(item) for item in tuple(species)]
                        if species is not None else None),
            "positions": (np.asarray(positions, dtype=np.float64).tolist()
                          if positions is not None else None),
        }
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    raise TypeError(
        f"arm audit value of type {type(value).__name__} has no plain-JSON "
        f"representation")


def _arm_audit_record(arm):
    """The arm section of the audit record: explicit identity fields only.

    The adapter contract (review-agreed) fixes exactly four identity
    fields an arm dict carries beside its callables: kind, status,
    arm_id and provides_energy. Only non-callable ones are recorded, and
    the projection names those four explicitly rather than walking the
    dict, so an arm can never leak an unserializable callable or a
    private buffer into a saved report. An unexpected value keeps the
    record buildable: it is refused here, by _plain_json, so a
    non-JSON-representable arm fact fails loudly instead of being
    stringified.
    """
    record = {}
    for name in ("kind", "status", "arm_id", "provides_energy"):
        if name in arm:
            record[name] = _plain_json(arm[name])
    return record


def _branch_audit_record(start_entry, arm, procedure,
                         relaxation, outcome, own, elapsed_seconds):
    """The per-branch evidence dict the reporter retains verbatim.

    Every field is plain-JSON data under an explicit name: the arm
    identity actually used, the start identity, the branch's OWN
    reference record, the endpoint coordinates, the RMSD and mapping
    facts, the structured failure facts (refusal details naming which
    arm refused), the exact force-call and step counts, and this
    branch's own wall-clock timing. The failure dict distinguishes the
    structured refusal trail from the plain stop reason so the two stay
    separately auditable, and the effective settings are copied
    defensively so a later caller mutating the shared procedure dict
    cannot rewrite the saved report. The effective settings are
    recorded under "procedure" with the "effective" key naming the dict
    the branch actually ran under, the same shape the fixtures script
    records, so the saved report reads one convention for both callers.
    The branch's partition label is not duplicated here: the row the
    reporter already builds carries it.
    """
    failure = {
        "stop_reason": outcome["stop_reason"],
        "failure_reason": relaxation["failure_reason"],
        "forces_finite": bool(relaxation["forces_finite"]),
        "nonfinite_force": bool(outcome["nonfinite_force"]),
        "budget_exhausted": bool(outcome["budget_exhausted"]),
        "refusal_details": list(relaxation["refusal_details"]),
    }
    return {
        "arm": _arm_audit_record(arm),
        "start": {
            "record_id": start_entry["record_id"],
            "frame_index": start_entry.get("frame_index"),
            "relax_id": f"{start_entry['record_id']}/relax",
        },
        "reference": own,
        "final_positions": relaxation["final_positions"],
        "rmsd_A": outcome["rmsd_A"],
        "permutation": outcome["permutation"],
        "rotation": outcome["rotation"],
        "accepted_classes": outcome["accepted_classes"],
        "failure": failure,
        "force_evaluation_count": relaxation["force_evaluation_count"],
        "step_count": relaxation["step_count"],
        "timing_seconds": float(elapsed_seconds),
        "procedure": {"effective": dict(procedure)},
    }


def evaluate_starts(start_set, arm, procedure, endpoint_classification,
                    references, policy, *, split_manifest=None,
                    seen_minimum=None, authorise_test_partition=False,
                    arm_start_compatibility=None):
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
        not acceptable. Production evaluation must be paired with a
        real PreparationPolicy as policy, see split_binding.
    seen_minimum: optional mapping of branch key (or constant) to the
        frozen manifest's seen_minimum flag, carried onto every outcome.
    authorise_test_partition: the separate authorisation for the test
        partition. False (default) refuses test-partition branches with
        the branch names in the error.
    arm_start_compatibility: optional explicit override of how each
        branch's arm/start compatibility is checked. None (default) is
        the "strict" production discipline: every admitted branch's
        species tuple and atom count are validated against the arm
        BEFORE any force call, and an incompatible arm refuses the run
        with ArmConfigurationError instead of being scored. Pass the
        "compatible" string only under a reviewed, separately recorded
        amendment to that discipline (the explicit, reviewed opt-out
        interface; the flag is echoed on the result).

    The split binding (review round 2, with the fixtures interface
    restored) reads the manifest itself, so the public call pattern
    carries no extra argument. A manifest without the explicit
    "synthetic": {"reason": ...} declaration is production evaluation:
    policy must be a real PreparationPolicy and the manifest must
    recompute to the approved split identity and archive digest;
    anything else fails closed BEFORE any force call, naming what
    disagreed (relabelling a manifest cannot move a test connection
    into another partition's label). A manifest carrying the
    declaration binds as an explicitly identified synthetic fixture
    (a production manifest relabelled synthetic is refused; its own
    unambiguous groups stay enforced, so the overlap refusal still
    applies). The binding is echoed on the result.

    Returns a dict with the per-branch and per-connection outcomes (see
    the module docstring for the arm boundary, the authorisation
    discipline, and the per-branch audit record). Every branch outcome
    is directly passable to build_report: the "_audit_record" the
    reporter retains rides on each branch dict.
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

    # The reviewed split binding (review round 2, fixtures interface): the
    # manifest itself says whether it is a synthetic fixture, so the
    # identity binding is default-on and everything else fails closed
    # here, BEFORE any force call. The fixture marker lives INSIDE the
    # manifest; the call-site pattern carries no new argument.
    if not isinstance(split_manifest, dict):
        raise SplitBindingError(
            "the supplied split manifest is not a dict, so it cannot be "
            "bound at all")
    synthetic_declaration = _manifest_synthetic_declaration(split_manifest)
    bound = _bind_split_manifest(
        split_manifest, synthetic_declaration=synthetic_declaration)
    synthetic_fixture = bound == "fixture"
    if not synthetic_fixture and not isinstance(policy, PreparationPolicy):
        # Production requires the real policy object the reviewed run
        # used, supplied as policy, and the approved manifest here.
        raise SplitBindingError(
            f"production evaluation requires a real PreparationPolicy as "
            f"policy (got {type(policy).__name__!r}), paired with the "
            f"approved split manifest")

    # The manifest-wide partition validation BEFORE anything is admitted:
    # every branch's connection is resolved once, so an ambiguous or
    # incomplete manifest is reported before any branch is admitted, any
    # authorisation decision is taken, or any reference is resolved.
    _validate_connection_partitions(start_set, split_manifest)

    if arm_start_compatibility == "compatible":
        compatibility = "compatible"
    elif arm_start_compatibility is None:
        compatibility = "strict"
    else:
        raise ValueError(
            f"unknown arm_start_compatibility {arm_start_compatibility!r}; "
            f"pass None (the strict pre-validated discipline) or "
            f"'compatible' (the reviewed explicit opt-out)")

    # First pass: partition and reference resolution for every branch,
    # BEFORE any force evaluation, so a refusal stops the run before
    # budget is spent on branches that must not be evaluated. When the
    # strict compatibility discipline is on, it is checked in the same
    # pass, so an incompatible arm refuses the run before ANY branch
    # spends budget.
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
        if compatibility == "strict":
            # Arm/start compatibility before anything is scored: species
            # tuple, atom count and the start's own array shape, through
            # the adapter's own validator. This is the exact check the
            # arm would raise on its first call, moved ahead of every
            # force evaluation, so a misconfigured arm is a loud
            # configuration error instead of a scored branch outcome.
            _validated_positions(start_entry["positions"], arm["species"],
                                 None, f"runner arm {arm.get('arm_id')!r} "
                                 f"for {branch_key!r}")
        plans.append((branch_key, start_entry, partition, pool_key, own))
    if refused:
        raise PartitionAuthorisationError(
            f"test-partition branches cannot be evaluated without the "
            f"separate authorisation the design spec reserves for the test "
            f"partition (evaluated once, after freeze): {sorted(refused)}. "
            f"Pass authorise_test_partition=True only when that "
            f"authorisation exists.")

    # Second pass: relax and classify the admitted branches. Each branch
    # is timed individually (review item 2): the wall clock wraps this
    # branch's relax and classification alone, not the pass or the run,
    # so the saved timing is per-branch evidence rather than a
    # whole-run duration. Every outcome leaves the pass carrying a
    # structured "_audit_record" with the branch's seed-to-endpoint
    # evidence (review item 2), which the reporter retains into the
    # saved diagnostics.
    branches = {}
    for branch_key, start_entry, partition, _pool_key, own in plans:
        species = tuple(start_entry["species"])
        force_fn = _arm_force_supplier(arm, species,
                                       f"{start_entry['record_id']}/relax")
        branch_started = time.perf_counter()
        # The refusal record must name WHICH arm refused: relax reads the
        # arm only for its arm_id, so a refused force keeps the arm's
        # identity in refusal_details instead of collapsing to None.
        relaxation = relax(start_entry["positions"], force_fn, procedure,
                           arm=arm)
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
        elapsed = time.perf_counter() - branch_started
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
            "refusal_details": relaxation["refusal_details"],
            "_audit_record": _branch_audit_record(
                start_entry, arm, procedure,
                relaxation, outcome, own, elapsed),
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
        entry["branches"].append(row["branch_key"])
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
        "split_binding": bound,
        "synthetic_fixture": synthetic_fixture,
        "arm_start_compatibility": compatibility,
        "branch_count": len(branches),
        "branches": branches,
        "connections": connections,
    }