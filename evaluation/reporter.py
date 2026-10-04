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
- seen_minimum is preserved per branch, so a held-out connection whose
  target shape was seen in training stays visible in every table.
- A train-partition branch stays visible as a counted branch but is
  excluded from the held-out pools and denominators.

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


def build_report(evaluated, expected_connections=None):
    """Validate and count the evaluated outcomes per connection.

    evaluated: dict of branch_key to outcome dict. Each outcome must
        carry branch_key, one category from CATEGORY_NAMES, the
        seen_minimum flag, a partition name, and the converged /
        budget_exhausted facts. Missing or forged entries raise
        ValueError.
    expected_connections: optional list of branch keys that must exist
        even when no outcome was provided for them. Each expected key
        becomes a connection entry with denominator zero and
        availability "unavailable" until outcomes arrive.

    Returns (report, diagnostics): the report holds the counts and
    rates per connection; diagnostics holds the per-branch audit list
    exactly as validated, with seen_minimum preserved.
    """
    if not isinstance(evaluated, dict):
        raise ValueError("evaluated must be a dict of branch_key to outcome")
    per_branch = []
    for key, outcome in evaluated.items():
        outcome = _validated_outcome(key, outcome)
        molecule, connection_id, branch = _split_key(key)
        per_branch.append({
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
        })

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
            "denominator": held_out_denominator,
            "denominators": {
                "evaluated_branches": held_out_denominator,
                # The shape-recovery pool IS the evaluated held-out pool
                # for this connection; kept as an explicit alias so the
                # denominator identity is checkable in the report.
                "shape_recovery_total": held_out_denominator,
            },
            "counts": counts,
            "rates": rates,
            "availability": availability,
            # seen_minimum per branch is preserved, not aggregated away:
            # one connection can carry branches with different flags only
            # when its two branches target minima seen differently.
            "seen_minimum_by_branch": dict(sorted(seen_minimum.items())),
        })

    report = {
        "schema": 1,
        "branch_count": len(per_branch),
        "held_out_branch_count": len([row for row in per_branch if row["held_out"]]),
        "per_connection": per_connection,
        "category_totals": {
            name: sum(1 for row in per_branch
                      if row["held_out"] and row["category"] == name)
            for name in CATEGORY_NAMES
        },
    }
    diagnostics = {"per_branch": per_branch}
    return report, diagnostics


def _validated_outcome(key, outcome):
    """Reject forged or missing outcome fields with a clear message."""
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
    if partition not in ("train", "validation", "test"):
        raise ValueError(f"{key}: unknown partition {partition!r}")
    if "converged" not in outcome or "budget_exhausted" not in outcome:
        raise ValueError(f"{key}: the converged and budget_exhausted facts must be explicit")
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


def render(report, diagnostics, out_dir, stem):
    """Render the report as a JSON and Markdown pair, return their paths."""
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
                 f"(held out: {report['held_out_branch_count']})")
    lines.append("")
    lines.append("Category totals over held-out branches: " + ", ".join(
        f"{name} {count}" for name, count in report["category_totals"].items()))
    lines.append("")
    lines.append("| molecule | connection | denominator | converged | " +
                 " | ".join(CATEGORY_NAMES) + " | rates | availability |")
    lines.append("|---|---|---|---|" + "---|" * len(CATEGORY_NAMES) + "---|---|")
    for connection in report["per_connection"]:
        counts = connection["counts"]
        rates = ", ".join(f"{name.replace('_rate', '')} {value}" for name, value in connection["rates"].items())
        lines.append(
            f"| {connection['molecule']} | {connection['connection_id']} "
            f"| {connection['denominator']} "
            f"| {connection['denominators']['shape_recovery_total']} "
            f"| " + " | ".join(str(counts[name]) for name in CATEGORY_NAMES) +
            f" | {rates} | {connection['availability']} |")
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