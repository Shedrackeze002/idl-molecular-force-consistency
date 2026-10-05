"""The frozen start set: which stored frame each arm relaxes from.

The rule reads the frozen split manifest, the archive records, and the
committed endpoint classification. No relaxation outcome is visible to
this module, so no outcome can leak backward into start selection.

Two rules are provided:

- first_stored_frame (primary): stored frame 0 of every held-out branch.
- fraction_of_ts_rmsd (predeclared sensitivity): the first stored frame
  whose RMSD to that branch's matched reference minimum is at or below a fraction
  of that branch's transition-state-to-minimum RMSD.

The reference minimum comes from the committed endpoint classification
(reference_class per branch, as frozen in reports/landscape17-endpoints.json
for the real archive), the same source the preparation package itself uses
(splits.py: "Endpoint classes, not filename pairs, determine reference
membership"). The manifest's endpoint_pair is never used: it is positional
filename layout, and it demonstrably disagrees with the committed classes on
26 of the 74 real aspirin branches (14 of 37 connections), comprising
reversals on 12 connections plus a degenerate same-minimum pair (connection
66, endpoint_pair ['6','6']) and one genuinely different pairing.
"""
from force_consistency.data.geometry import align_records
from force_consistency.data.splits import canonical_hash
from force_consistency.data.types import FrameRecord, PreparationPolicy


class StartSetError(ValueError):
    pass


KNOWN_RULES = ("first_stored_frame", "fraction_of_ts_rmsd")


def build_start_set(split_manifest, records, policy, rule, endpoint_classification=None):
    """Select the start structure for every held-out branch.

    split_manifest: the frozen split manifest as a parsed dict.
    records: every FrameRecord from the archive (or a subset that contains
        every held-out branch frame and every reference minimum).
    policy: a PreparationPolicy (the frozen preparation policy).
    rule: a dict naming the rule. Two kinds are allowed:
        {"kind": "first_stored_frame"} selects stored frame 0.
        {"kind": "fraction_of_ts_rmsd", "fraction": f} with 0 < f <= 1.
    endpoint_classification: the committed endpoint classification as a
        parsed dict (reports/landscape17-endpoints.json for the real
        archive), whose keys are branch keys and whose values carry the
        "status" and "reference_class" fields. The sensitivity rule maps
        every branch to its committed reference minimum through it, the
        same source the preparation package's eligibility step uses.
        It is REQUIRED for fraction_of_ts_rmsd (a StartSetError is raised
        without it) and is not consulted by first_stored_frame, which
        never reads a minimum.

    Returns a dict keyed by "molecule/connection/branch" holding the start
    record id, stored frame index, the rule identity, the start structure
    as the original archive FrameRecord, and the measured distances used.
    """
    if not isinstance(rule, dict) or "kind" not in rule:
        raise StartSetError("The start rule must be a dict with a 'kind' key")
    kind = rule["kind"]
    if kind not in KNOWN_RULES:
        raise StartSetError(f"Unknown start rule kind: {kind!r}")
    fraction = rule.get("fraction")
    if kind == "fraction_of_ts_rmsd":
        if not isinstance(fraction, (int, float)) or isinstance(fraction, bool) or not 0.0 < fraction <= 1.0:
            raise StartSetError("The fraction_of_ts_rmsd rule needs a fraction in (0, 1]")
        if not isinstance(endpoint_classification, dict):
            raise StartSetError(
                "The fraction_of_ts_rmsd rule needs the committed endpoint "
                "classification (the parsed reports/landscape17-endpoints.json "
                "or an equivalent synthetic one): a branch's reference minimum "
                "comes from its committed reference_class, and silently "
                "falling back to the manifest's endpoint_pair would resurrect "
                "the reversed-pairing defect")

    byid = {record.record_id: record for record in records}
    if len(byid) != len(records):
        raise StartSetError("Duplicate record identities in the record list")

    groups = split_manifest.get("groups", {})
    connections_manifest = split_manifest.get("connections", {})
    out = {}
    for molecule, partitions in sorted(groups.items()):
        held_out = []
        for partition in ("validation", "test"):
            for connection_id in partitions.get(partition, []):
                if connection_id not in held_out:
                    held_out.append(connection_id)
        for connection_id in sorted(held_out):
            connection = connections_manifest[molecule][connection_id]
            for branch in sorted(connection["branches"]):
                key = f"{molecule}/{connection_id}/{branch}"
                out[key] = _select_start(key, molecule, connection_id, connection,
                                         branch, byid, policy, dict(rule), fraction,
                                         endpoint_classification)
    return out


def start_set_hash(start_set):
    """A canonical hash of the start selection for the freeze record.

    It covers the branch keys and the chosen record ids, so any change of
    the selected frames changes the hash while byte-identical structure
    copies keep it stable.
    """
    payload = {key: {"record_id": entry["record_id"], "frame_index": entry["frame_index"]}
               for key, entry in sorted(start_set.items())}
    return canonical_hash(payload)


def _select_start(key, molecule, connection_id, connection, branch, byid, policy, identity,
                  fraction, endpoint_classification):
    branch_ids = connection["branches"][branch]
    frames = []
    for record_id in branch_ids:
        record = byid.get(record_id)
        if record is None:
            raise StartSetError(f"{key}: manifest frame {record_id} is missing from the records")
        if record.connection_id != connection_id or record.branch != branch:
            raise StartSetError(f"{key}: manifest frame {record_id} does not name this branch")
        frames.append(record)
    indices = [record.frame_index for record in frames]
    if indices != list(range(len(frames))):
        raise StartSetError(f"{key}: stored frame order is not contiguous from zero")

    if identity["kind"] == "first_stored_frame":
        identity["fraction"] = None
        chosen_index = 0
        chosen = frames[0]
        distances = {}
    else:
        chosen_index, distances = _sensitivity_choice(
            key, molecule, frames, connection, byid, policy, float(fraction),
            endpoint_classification)
        chosen = frames[chosen_index]
    entry = {
        "branch_key": key,
        "record_id": chosen.record_id,
        "frame_index": chosen_index,
        "rule": dict(identity),
        "positions": chosen.positions,
        "forces": chosen.forces,
        "energy": chosen.energy,
        "species": chosen.species,
        "structure": chosen,
        "molecule": molecule,
        "connection_id": connection_id,
        "branch": branch,
        "distances": distances,
    }
    # Sensitivity-rule extras are plain fields so the frozen payload stays
    # JSON friendly.
    for extra in ("rmsd_to_reference", "ts_to_reference_rmsd", "near_reference_frames",
                  "chosen_beta", "next_beta"):
        if extra in distances:
            entry[extra] = distances[extra]
    return entry


def _branch_minimum(key, molecule, connection_id, branch, byid, endpoint_classification):
    """The branch's reference minimum, identified by the committed classification.

    A branch relaxes toward the minimum its endpoint was actually classified
    against, so the reference is the committed classification's
    reference_class for that branch key ("molecule/connection/branch"),
    exactly as the preparation package resolves it (splits.py: "Endpoint
    classes, not filename pairs, determine reference membership").

    The classification must be supplied (build_start_set already enforces
    this for the sensitivity rule) and every branch it is asked about must
    be "matched" with a minimum reference_class; anything else raises a
    StartSetError naming the branch and the violated expectation.

    The manifest's endpoint_pair is intentionally NOT consulted: it is
    positional filename layout, it disagrees with the committed classes on
    26 of the 74 real aspirin branches (14 of 37 connections), comprising
    reversals on 12 connections plus a degenerate same-minimum pair
    (connection 66, endpoint_pair ['6','6']) and one genuinely different
    pairing, and a silent fallback to it would hide exactly that defect.
    """
    branch_key = f"{molecule}/{connection_id}/{branch}"
    if not isinstance(endpoint_classification, dict):
        raise StartSetError(
            f"{key}: the committed endpoint classification is required to "
            f"identify the branch's reference minimum; it was not supplied")
    entry = endpoint_classification.get(branch_key)
    if not isinstance(entry, dict) or entry.get("status") != "matched":
        observed = repr(entry.get("status")) if isinstance(entry, dict) else repr(entry)
        raise StartSetError(
            f"{key}: no 'matched' endpoint classification for {branch_key!r} "
            f"(got {observed}); a reference minimum is only defined for a "
            f"matched branch")
    reference_class = entry.get("reference_class")
    record = byid.get(reference_class) if isinstance(reference_class, str) else None
    if record is None or record.kind != "minimum":
        raise StartSetError(
            f"{key}: the classified reference {reference_class!r} for "
            f"{branch_key!r} is not a minimum record in the supplied records")
    return record


def _sensitivity_choice(key, molecule, frames, connection, byid, policy, fraction,
                        endpoint_classification):
    """First stored frame within fraction times the TS-to-minimum RMSD.

    Returns the chosen frame index and the measured distance summary. When
    no stored frame qualifies the rule raises: silently falling back to the
    last frame would hide a rule that is undefined on that branch.
    """
    minimum = _branch_minimum(key, molecule, frames[0].connection_id, frames[0].branch,
                              byid, endpoint_classification)
    ts = byid.get(connection["transition_state"])
    if ts is None:
        raise StartSetError(f"{key}: transition state is missing from the records")

    # RMSD after the symmetry-aware alignment of each frame to the branch's
    # own minimum, and of the transition state to the same minimum.
    frame_rmsds = []
    for record in frames:
        aligned = align_records(record, minimum, policy)
        if not aligned.search_complete:
            raise StartSetError(f"{key}: alignment to the reference minimum is incomplete")
        frame_rmsds.append(aligned.rmsd)
    ts_alignment = align_records(ts, minimum, policy)
    if not ts_alignment.search_complete:
        raise StartSetError(f"{key}: alignment of the transition state is incomplete")
    ts_rmsd = ts_alignment.rmsd

    threshold = fraction * ts_rmsd
    chosen_index = None
    for index, rmsd in enumerate(frame_rmsds):
        if rmsd <= threshold:
            chosen_index = index
            break
    if chosen_index is None:
        raise StartSetError(
            f"{key}: no stored frame is within {fraction} of the TS-to-minimum RMSD")

    # beta values in the synthetic notation are the frame's RMSD divided by
    # the TS RMSD, which is what the ladder tests reason about.
    near = [[index, rmsd] for index, rmsd in enumerate(frame_rmsds) if rmsd <= threshold]
    distances = {
        "rmsd_to_reference": frame_rmsds[chosen_index],
        "ts_to_reference_rmsd": ts_rmsd,
        "near_reference_frames": near,
        "chosen_beta": frame_rmsds[chosen_index] / ts_rmsd if ts_rmsd > 0 else None,
        "next_beta": (frame_rmsds[chosen_index + 1] / ts_rmsd
                      if chosen_index + 1 < len(frame_rmsds) and ts_rmsd > 0 else None),
    }
    return chosen_index, distances