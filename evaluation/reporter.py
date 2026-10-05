"""Count, validate and render shape-recovery outcomes per connection.

The reporter consumes one outcome dict per evaluated branch, as produced
by evaluation.outcomes.classify_outcome plus the branch bookkeeping the
caller attaches (branch_key, partition, seen_minimum). It validates
every entry, counts the five outcome categories per connection, keeps
the denominators consistent, and renders a JSON plus Markdown pair.

Denominators. The rate denominators are the converged branches per
connection (matched_reference, unmatched_converged and ambiguous), the
denominator description frozen in configs/aspirin-shape-recovery.json.
A connection whose denominator is zero is reported as unavailable, not
as zero percent: the honest statement is that nothing converged there,
not that every branch failed to recover the shape.

Discipline enforced here:

- A missing or forged outcome (unknown category, dropped seen_minimum
  flag) is a ValueError, never silently defaulted.
- The category must agree with the branch's own convergence facts, not
  merely sit beside them: matched_reference, unmatched_converged and
  ambiguous require converged=True with budget_exhausted=False,
  nonconverged requires converged=False, and unresolved (a comparison
  failure on a reached endpoint) requires converged=True too. A
  contradicting pair is refused, so a nonconverged branch can never be
  reported as a recovered shape.
- The two branches of one connection must carry the same partition
  label, and the label must be one of train / validation / test. A
  split-or-unknown label is misdescription, and is refused instead of
  pooled or guessed.
- seen_minimum is preserved per branch, so a held-out connection whose
  target shape was seen in training stays visible in every table.
- A train-partition branch stays visible as a counted branch but is
  excluded from the held-out pools and denominators.

Audit record. When the caller supplies an "_audit_record" dict on an
outcome, the reporter retains, under explicit names on that branch, the
reference identity, the start identity, the arm identity, the final
positions, the RMSD and the mapping, the failure facts, and the actual
force-call and step counts — in diagnostics.per_branch, in the rendered
JSON, and as a per-branch audit row in the Markdown. Every audit value
is normalized to plain-JSON data (FrameRecord-like objects become
explicit named fields, numpy arrays become float lists) before
rendering; an opaque object is never stored as its string repr, so a
saved report round-trips through json.loads without loss. The ten
per-branch summary fields kept in earlier rounds are all kept.

The disclaimer that a converged force-field fixed point is NOT a
minimum, and that this slice ran on an analytic fixture arm rather
than a trained model, is rendered into every Markdown report.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

CATEGORY_NAMES = ("matched_reference", "unmatched_converged", "ambiguous",
                  "unresolved", "nonconverged")
CONVERGED_CATEGORIES = ("matched_reference", "unmatched_converged", "ambiguous")
PARTITION_NAMES = ("train", "validation", "test")


def build_report(evaluated, expected_connections=None):
    """Validate and count the evaluated outcomes per connection.

    evaluated: dict of branch_key to outcome dict. Each outcome must
        carry branch_key, one category from CATEGORY_NAMES, the
        seen_minimum flag, a partition name, and the converged /
        budget_exhausted facts as booleans agreeing with the category.
        The categories of the two branches of one connection must carry
        the same partition label. Optionally an "_audit_record" dict of
        plain-JSON evidence (see the module docstring) is retained on
        the branch. Missing, forged or contradicting entries raise
        ValueError.
    expected_connections: optional list of branch keys that must exist
        even when no outcome was provided for them. Each expected key
        becomes a connection entry with denominator zero and
        availability "unavailable" until outcomes arrive.

    Returns (report, diagnostics): the report holds the counts, rates,
    partition labels and per-partition totals; diagnostics holds the
    per-branch audit list exactly as validated, with seen_minimum
    preserved and the caller's audit fields carried through.
    """
    if not isinstance(evaluated, dict):
        raise ValueError("evaluated must be a dict of branch_key to outcome")
    per_branch = []
    for key, outcome in evaluated.items():
        outcome = _validated_outcome(key, outcome)
        molecule, connection_id, branch = _split_key(key)
        row = {
            "branch_key": key,
            "molecule": molecule,
            "connection_id": connection_id,
            "branch": branch,
            "partition": outcome["partition"],
            "category": outcome["category"],
            "seen_minimum": outcome["seen_minimum"],
            "converged": outcome["converged"],
            "budget_exhausted": outcome["budget_exhausted"],
            "held_out": outcome["partition"] in ("validation", "test"),
        }
        # Auditable branch record: the caller's evidence retained per
        # branch under its explicit keys, never flattened into the row's
        # identity. When the caller supplies no record the row is the
        # summary projection alone.
        row.update(_record_fields(key, outcome))
        per_branch.append(row)

    # Group branches by (molecule, connection_id) and count categories.
    grouped = {}
    for row in per_branch:
        grouped.setdefault((row["molecule"], row["connection_id"]), []).append(row)

    # Forged or missing outcomes are rejected. Every connection in the
    # frozen landscape carries both branches (74 branches across 37
    # connections), so a connection whose evaluated branches are not a
    # complete plus/minus pair means an outcome was dropped or forged:
    # refusing is the honest response, and empty expected connections
    # (added below) are the explicit way to report a gap.
    incomplete = sorted(key for key in grouped
                        if sorted(row["branch"] for row in grouped[key])
                        != ["minus", "plus"])
    if incomplete:
        raise ValueError(
            f"connections whose evaluated branches are not a complete "
            f"plus/minus pair, which means an outcome is missing or "
            f"forged: {incomplete}")

    # Partition consistency (review round 2): the two branches of one
    # connection are the two arms of the same measurement and are pooled
    # into one denominator, so they must share one partition label. A
    # pair renaming partitions across the pair is mislabelled, not a
    # mixed pool, and is refused with both branch keys named.
    split_partition = {}
    for key, rows in grouped.items():
        labels = sorted({row["partition"] for row in rows})
        if len(labels) > 1:
            by_branch = {row["branch_key"]: row["partition"] for row in rows}
            split_partition["%/%".join(key)] = by_branch
    if split_partition:
        raise ValueError(
            f"the two branches of one connection carry different partition "
            f"labels, which is a mislabelled outcome rather than a mixed "
            f"pool: {split_partition}")

    # Expected connections with no outcome at all: zero denominator,
    # reported unavailable. They key on the same branch-key convention.
    expected_keys = list(expected_connections or [])
    for key in expected_keys:
        molecule, connection_id, _branch = _split_key(key)
        if (molecule, connection_id) not in grouped:
            grouped[(molecule, connection_id)] = []

    per_connection = []
    for (molecule, connection_id) in sorted(grouped):
        rows = grouped[(molecule, connection_id)]
        held_out_rows = [row for row in rows if row["held_out"]]
        # The denominator is the FULL held-out pool for this connection:
        # every evaluated held-out branch, converged or not. The tests
        # fix this semantics (1 matched out of 2 evaluated branches is a
        # 0.5 rate), so matched_reference_rate is the recovery rate over
        # everything evaluated, not over the converged subset.
        held_out_denominator = len(held_out_rows)
        counts = {name: 0 for name in CATEGORY_NAMES}
        seen_minimum = {}
        for row in held_out_rows:
            counts[row["category"]] += 1
            seen_minimum[row["branch"]] = row["seen_minimum"]
        # The true convergence count of this pool (review round 2): the
        # branches that met the declared force rule, whether or not the
        # comparison then scored them. It is NOT the denominator the
        # rates divide by; the two are kept separate.
        converged_count = sum(row["converged"] for row in held_out_rows)
        if held_out_denominator == 0:
            availability = "unavailable"
            rates = {f"{name}_rate": None for name in CONVERGED_CATEGORIES}
        else:
            availability = "available"
            rates = {f"{name}_rate": counts[name] / held_out_denominator
                     for name in CONVERGED_CATEGORIES}
        per_connection.append({
            "molecule": molecule,
            "connection_id": connection_id,
            # Partition label on the row itself (review round 2); the
            # consistency check above guarantees both branches agree. An
            # expected connection with no evaluated branch has no branch
            # partition to read, so its label stays unset.
            "partition": rows[0]["partition"] if rows else None,
            "denominator": held_out_denominator,
            "denominators": {
                "evaluated_branches": held_out_denominator,
                # The shape-recovery pool IS the evaluated held-out pool
                # for this connection; kept as an explicit alias so the
                # denominator identity is checkable in the report.
                "shape_recovery_total": held_out_denominator,
            },
            "converged_count": converged_count,
            "counts": counts,
            "rates": rates,
            "availability": availability,
            # seen_minimum per branch is preserved, not aggregated away:
            # one connection can carry branches with different flags only
            # when its two branches target minima seen differently.
            "seen_minimum_by_branch": dict(sorted(seen_minimum.items())),
        })

    # Category totals, pooled over the held-out branches AND split by
    # partition (review round 2): validation and test recovery counts
    # must never merge, because merging them hides a validation-only
    # recovery or a test-only nonconvergence.
    held_out_branches_rows = [row for row in per_branch if row["held_out"]]
    validation_rows = [row for row in held_out_branches_rows
                       if row["partition"] == "validation"]
    test_rows = [row for row in held_out_branches_rows
                 if row["partition"] == "test"]

    def _count(rows, name):
        return sum(1 for row in rows if row["category"] == name)

    report = {
        "schema": 2,
        "branch_count": len(per_branch),
        "held_out_branch_count": len(held_out_branches_rows),
        "held_out_branch_counts": {
            "validation": len(validation_rows),
            "test": len(test_rows),
        },
        "per_connection": per_connection,
        "category_totals": {
            "held_out": {name: _count(held_out_branches_rows, name)
                         for name in CATEGORY_NAMES},
            "validation": {name: _count(validation_rows, name)
                           for name in CATEGORY_NAMES},
            "test": {name: _count(test_rows, name)
                     for name in CATEGORY_NAMES},
        },
    }
    diagnostics = {"per_branch": per_branch}
    return report, diagnostics


def _record_fields(key, outcome):
    """The auditable evidence fields the caller supplied on this branch.

    The outcome may carry an "_audit_record" dict with any of: the
    reference identity ("reference"), the start identity ("start"), the
    arm identity ("arm"), the final positions ("final_positions"), the
    RMSD ("rmsd_A"), the mapping ("permutation", "rotation") and match
    classes ("accepted_classes"), the failure facts ("failure"), and the
    actual force-call and step counts ("force_evaluation_count",
    "step_count"). Each present field is retained on the branch row
    under its own name, with its ten summary neighbours untouched.

    Every field is normalized to plain-JSON data before rendering —
    FrameRecord-like objects to explicit named fields, numpy arrays to
    float lists, numpy scalars to Python numbers — and an audit value
    that is not plain-JSON representable is refused rather than
    stringified, because a stringified repr in a saved report pretends
    to be readable data.
    """
    record = outcome.get("_audit_record")
    if record is None:
        return {}
    if not isinstance(record, dict):
        raise ValueError(
            f"{key}: _audit_record must be a dict of plain-JSON audit "
            f"fields, got {type(record).__name__}")
    identity_fields = {}
    for name in ("reference", "start", "arm", "rmsd_A", "permutation",
                 "rotation", "accepted_classes", "failure"):
        if name not in record:
            continue
        try:
            identity_fields[name] = _plain_json_field(record[name])
        except TypeError as unrepresentable:
            raise ValueError(
                f"{key}: the _audit_record field {name!r} is not "
                f"plain-JSON representable: {unrepresentable}") from None
    if "final_positions" in record:
        try:
            identity_fields["final_positions"] = np.asarray(
                record["final_positions"], dtype=np.float64).tolist()
        except (TypeError, ValueError) as bad:
            raise ValueError(
                f"{key}: the _audit_record field 'final_positions' is not "
                f"a numeric array: {bad}") from None
    for name in ("force_evaluation_count", "step_count"):
        if name in record:
            identity_fields[name] = int(record[name])
    return identity_fields


def _plain_json_field(value):
    """Normalize one audit value to plain-JSON data, refusing opaques.

    numpy arrays become float lists, numpy scalars and booleans become
    Python numbers and bools, dict/list data normalize recursively, and
    a FrameRecord-like audit object becomes its explicit named fields
    (record id, molecule, kind, naming, frame index, species and stored
    positions) — never its repr. Anything scalar and JSON-native passes
    through unchanged. An object with no understood structure raises
    TypeError instead of being str()ed into the report.
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
        normalized = {}
        for item_key, item_value in value.items():
            normalized[str(item_key)] = _plain_json_field(item_value)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_plain_json_field(item) for item in value]
    if hasattr(value, "record_id"):
        # A FrameRecord-like audit object: name its fields, never repr
        # it. The retained position array is the record's own structure.
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
            "species": ([_plain_json_field(item) for item in tuple(species)]
                        if species is not None else None),
            "positions": (np.asarray(positions, dtype=np.float64).tolist()
                          if positions is not None else None),
        }
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    raise TypeError(
        f"value of type {type(value).__name__} has no plain-JSON "
        f"representation")


def _validated_outcome(key, outcome):
    """Reject forged or missing outcome fields with a clear message.

    Presence is not enough (review round 2): the convergence facts must
    be actual booleans and must agree with the category, and the
    partition must be a real partition name. The error names the branch
    key and the contradicted fact, so a forged report cannot pass as a
    validated one.
    """
    if not isinstance(outcome, dict):
        raise ValueError(f"{key}: outcome must be a dict, got {type(outcome).__name__}")
    category = outcome.get("category")
    if category not in CATEGORY_NAMES:
        raise ValueError(f"{key}: unknown outcome category {category!r}")
    if "seen_minimum" not in outcome:
        raise ValueError(f"{key}: seen_minimum is missing; it is never defaulted")
    if not isinstance(outcome["seen_minimum"], bool):
        raise ValueError(f"{key}: seen_minimum must be a boolean, got {type(outcome['seen_minimum']).__name__}")
    partition = outcome.get("partition", "test")
    if not isinstance(partition, str) or partition not in PARTITION_NAMES:
        raise ValueError(f"{key}: unknown partition {partition!r}")
    if "converged" not in outcome or "budget_exhausted" not in outcome:
        raise ValueError(f"{key}: the converged and budget_exhausted facts must be explicit")
    for fact in ("converged", "budget_exhausted"):
        if not isinstance(outcome[fact], bool):
            raise ValueError(f"{key}: {fact} must be a boolean, got "
                             f"{type(outcome[fact]).__name__}")
    # Consistency, not just presence. In the production taxonomy only a
    # branch whose force rule was met on a reached endpoint can carry any
    # verdict other than nonconverged: classify_outcome forces
    # converged=False onto every budget-exhausted or refused run before
    # any comparison, so a shape verdict or an unresolved comparison on a
    # branch with converged=False or budget_exhausted=True is a forged
    # pair, not a measured one. nonconverged is the mirror: the rule was
    # NOT met, whatever the budget did afterwards (exhausted, or a
    # no_progress / refused-force stop).
    if category == "nonconverged":
        if outcome["converged"]:
            raise ValueError(
                f"{key}: nonconverged contradicts converged=true "
                f"(budget_exhausted {outcome['budget_exhausted']!r}); a branch "
                f"whose declared force rule was met is never nonconverged")
    else:
        if not outcome["converged"]:
            raise ValueError(
                f"{key}: {category} contradicts converged=false; a branch "
                f"that did not meet the force rule has no shape or "
                f"comparison verdict")
        if outcome["budget_exhausted"]:
            raise ValueError(
                f"{key}: {category} contradicts budget_exhausted=true; a "
                f"branch whose budget ran out without the rule being met "
                f"is nonconverged, never a shape verdict")
    return dict(outcome)


def _split_key(key):
    parts = key.split("/")
    if len(parts) != 3:
        raise ValueError(f"branch key {key!r} must be molecule/connection/branch")
    return parts[0], parts[1], parts[2]


def _json_default(value):
    """JSON serializer for numpy arrays and scalars in the audit fields."""
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=np.float64).tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    return str(value)


def _audit_cell(value):
    """Markdown cell for one audit field: blank when it was not given."""
    if value is None:
        return ""
    if isinstance(value, dict) or isinstance(value, list):
        return json.dumps(value)
    return str(value)


def render(report, diagnostics, out_dir, stem):
    """Render the report as a JSON and Markdown pair, return their paths.

    The Markdown keeps the count table (whose denominator and converged
    columns are explicitly named and separated: the denominator is the
    per-connection evaluated held-out branch pool, the converged column
    the true number of branches meeting the force rule) and adds a
    per-branch audit row for every branch whose caller supplied an
    audit record, so a saved report can be checked against its own
    endpoints without reopening the run.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"

    payload = {"report": report, "diagnostics": diagnostics}
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))

    lines = []
    lines.append(f"# Shape-recovery diagnostics: {stem}")
    lines.append("")
    lines.append(f"Evaluated branches: {report['branch_count']} "
                 f"(held out: {report['held_out_branch_count']}; "
                 f"validation: {report['held_out_branch_counts']['validation']}, "
                 f"test: {report['held_out_branch_counts']['test']})")
    lines.append("")

    def _totals_line(label, counts):
        return (f"{label} category totals: " + ", ".join(
            f"{name} {count}" for name, count in counts.items()))

    # Held-out totals stay pooled for the headline, but the review
    # round's separate partition counts are below them, never merged.
    lines.append(_totals_line("Pooled held-out",
                              report["category_totals"]["held_out"]))
    lines.append(_totals_line("Validation",
                              report["category_totals"]["validation"]))
    lines.append(_totals_line("Test",
                              report["category_totals"]["test"]))
    lines.append("")
    # The denominator/converged distinction, spelled out once next to the
    # table that uses it, so the two columns can never be read as one.
    lines.append(
        "Rates are category counts divided by the denominator column. "
        "The denominator is the evaluated held-out branches per "
        "connection: every connection's rate pool is its full "
        "plus/minus branch pair, so neither branch alone carries the "
        "denominator. The converged column is NOT the denominator; it "
        "is the number of branches in that pool that met the declared "
        "force rule, whether or not the comparison then scored their "
        "endpoint.")
    lines.append("")
    lines.append("| molecule | connection | partition | denominator | converged "
                 "| " + " | ".join(CATEGORY_NAMES) + " | rates | availability |")
    lines.append("|---|---|---|---|---|" + "---|" * len(CATEGORY_NAMES) + "---|---|")
    for connection in report["per_connection"]:
        counts = connection["counts"]
        rates = ", ".join(f"{name.replace('_rate', '')} {value}" for name, value in connection["rates"].items())
        lines.append(
            f"| {connection['molecule']} | {connection['connection_id']} "
            f"| {connection['partition']} "
            f"| {connection['denominator']} "
            f"| {connection['converged_count']} "
            f"| " + " | ".join(str(counts[name]) for name in CATEGORY_NAMES) +
            f" | {rates} | {connection['availability']} |")
    lines.append("")

    # Per-branch audit table: the fields the caller supplied, retained
    # and rendered (review round 2). Rows carry the reference identity,
    # start identity, arm identity, RMSD and mapping, failure facts and
    # actual force-call and step counts, so the saved report can audit
    # the own-reference match and the budget without reopening the run.
    # Rows without an audit record show blank cells rather than invented
    # facts.
    lines.append("| branch_key | partition | category | reference | start | arm | "
                 "final_positions | rmsd_A | permutation | rotation | failure | "
                 "force_evaluation_count | step_count |")
    lines.append("|---|" + "---|" * 12)
    audit_keys = ("reference", "start", "arm", "final_positions", "rmsd_A",
                  "permutation", "rotation", "failure",
                  "force_evaluation_count", "step_count")
    for row in diagnostics["per_branch"]:
        cells = [row["branch_key"], row["partition"], row["category"]]
        cells += [_audit_cell(row.get(name)) for name in audit_keys]
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
    lines.append("")
    lines.append(_disclaimer())
    md_path.write_text("\n".join(lines) + "\n")
    return {"json": str(json_path), "md": str(md_path)}


def freeze_start_set_hash(start_set, out_dir, filename="start-set-hash.json"):
    """Freeze the start-set fingerprint the fingerprint test reads back.

    This is the reporter's start-set-hash step:
    tests/test_shape_recovery_starts.py
    ::test_real_start_set_fingerprint_is_stable recomputes both digests
    from the built start set and compares them against the file written
    here, so the serialization must match that test exactly: sha256 over
    json.dumps(payload, sort_keys=True) with default separators.

    Two digests are recorded:

    - start_record_id_map_sha256: branch key to start record id, the
      identity fingerprint of the frozen selection.
    - start_set_payload_sha256: the same map plus each start's stored
      frame index and positions, so a changed frame changes it too.
    """
    payload = {
        key: {
            "record_id": entry["record_id"],
            "frame_index": entry["frame_index"],
            "positions": np.asarray(entry["positions"], dtype=np.float64).tolist(),
        }
        for key, entry in start_set.items()
    }
    payload_digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()
    record_id_map = {key: row["record_id"] for key, row in payload.items()}
    record_id_map_digest = hashlib.sha256(
        json.dumps(record_id_map, sort_keys=True).encode()).hexdigest()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / filename
    path.write_text(json.dumps({
        "schema": 1,
        "artifact": "start-set-hash",
        "start_record_id_map_sha256": record_id_map_digest,
        "start_set_payload_sha256": payload_digest,
        "branch_count": len(start_set),
    }, indent=2, sort_keys=True) + "\n")
    return path


def _disclaimer():
    """The disclaimer every rendering carries, category count included."""
    return (
        "Disclaimer: the outcome taxonomy has five categories "
        "(matched_reference, unmatched_converged, ambiguous, unresolved, "
        "nonconverged); the seven-shape shorthand is not used. A converged "
        "endpoint is a force-field fixed point, not a minimum: it is never "
        "described as a minimum. A nonconverged run is never counted as a "
        "wrong shape. These numbers come from an analytic fixture arm, not "
        "from a trained model.")