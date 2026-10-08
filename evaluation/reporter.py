"""Count, validate and render shape-recovery outcomes per connection.

The reporter consumes one outcome dict per evaluated branch, as produced
by evaluation.outcomes.classify_outcome plus the branch bookkeeping the
caller attaches (branch_key, partition, seen_minimum). It validates
every entry, counts the five outcome categories per connection, keeps
the denominators consistent, and renders a JSON plus Markdown pair.

Denominators. The rate denominators are the FULL evaluated held-out
branch pool per connection, every evaluated held-out branch whether
converged or not (the complete plus/minus pair), matching the denominator
description frozen in configs/aspirin-shape-recovery.json. A connection
whose denominator is zero is reported as unavailable, not as zero
percent: the honest statement is that nothing was evaluated there, not
that every branch failed to recover the shape. The connection's
converged_count is the number of those branches that met the declared
force rule; it is not the denominator and no rate divides by it.

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
- The caller's force-failure facts are reconciled with the same summary
  (review item 5): the outcome-level nonfinite_force and
  forces_finite flags, the stop reason, and the nested "_audit_record"
  failure section must agree with the category, the convergence and
  budget booleans, and with each other. A summary scored as recovered
  while its own audit records a failed, nonfinite or nonconverged run
  is refused with the branch key and the contradicting fact named; the
  reporter never repairs the pair, and audit facts the caller omits
  are never invented.
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
reference identity, the start identity, the arm identity, the effective
procedure (with any labelled synthetic override next to the configured
value), the final positions, the RMSD and the mapping, the failure
facts, the split binding the branch was evaluated under
("approved" or "fixture", review round 3, gap 2), and the actual
force-call and step counts, all in
diagnostics.per_branch, in the rendered JSON, and as a per-branch audit
row in the Markdown. Every audit value is normalized to plain-JSON data
(FrameRecord-like objects become explicit named fields, numpy arrays
become float lists) before rendering; an opaque object is never stored
as its string repr, so a saved report round-trips through json.loads
without loss. The ten per-branch summary fields kept in earlier rounds
are all kept.

The disclaimer that a converged force-field fixed point is NOT a
minimum is rendered into every Markdown report, next to a provenance
line naming the arm that actually produced the numbers (derived from
the branches' own audit records, or an explicit not-recorded sentence
when no record arrived).
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

    # Provenance from the actual records (review item 2): the arm that
    # produced the numbers is named from the branches' own audit
    # records, never from a hardcoded arm kind. Distinct arm sections
    # are compared on their full plain-JSON payload, branches carrying
    # no record contribute nothing, and a report with no record at all
    # renders an explicit not-recorded provenance instead of claiming
    # some arm it has no evidence for.
    arm_sections = []
    for row in per_branch:
        section = row.get("arm")
        if section is not None and section not in arm_sections:
            arm_sections.append(section)
    if len(arm_sections) == 1:
        provenance = {"source": "audit_record",
                      "arm": dict(arm_sections[0])}
    elif len(arm_sections) > 1:
        provenance = {"source": "audit_record",
                      "arm": {"mixed": list(arm_sections)}}
    else:
        provenance = {"source": "not_recorded",
                      "arm": None,
                      "note": ("no branch carried an audit record, so the "
                               "producing arm is not recorded here")}

    report = {
        "schema": 2,
        "branch_count": len(per_branch),
        "held_out_branch_count": len(held_out_branches_rows),
        "held_out_branch_counts": {
            "validation": len(validation_rows),
            "test": len(test_rows),
        },
        "provenance": provenance,
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
    arm identity ("arm"), the effective procedure ("procedure", with any
    labelled synthetic override next to the configured value), the final
    positions ("final_positions"), the RMSD ("rmsd_A"), the mapping
    ("permutation", "rotation") and match classes
    ("accepted_classes"), the failure facts ("failure"), the
    actual force-call and step counts ("force_evaluation_count",
    "step_count"), and the branch's own wall-clock timing in seconds
    ("timing_seconds", review item 2). Each present field is retained
    on the branch row under its own name, with its ten summary
    neighbours untouched.

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
    for name in ("reference", "start", "arm", "split_binding", "procedure",
                 "rmsd_A", "permutation", "rotation", "accepted_classes",
                 "failure", "timing_seconds"):
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
    # Force-failure reconciliation (review item 5, card B2). The check
    # above keeps the category honest against converged /
    # budget_exhausted. This block reconciles the remaining surfaces
    # the caller supplies: the outcome-level force flags (the runner
    # carries them on every branch outcome), the stop reason, and the
    # nested "_audit_record" failure section. Nothing here repairs a
    # value: each checked fact is read from a dict the caller supplied,
    # and every read is guarded so audit data the caller omitted is
    # never invented. numpy booleans are normalized with bool() so a
    # record built straight from a runner outcome validates.
    _reconcile_failure_facts(key, outcome)
    return dict(outcome)


def _bool_fact(value):
    """True/False for a bool (or numpy bool) fact, None when absent."""
    if value is None:
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def _reconcile_failure_facts(key, outcome):
    """Reject a branch whose force-failure facts contradict its summary.

    Three surfaces are checked, all of them facts the caller supplies:

    1. Summary versus the outcome-level failure facts: the
       nonfinite_force / forces_finite flags and the stop reason the
       runner carries on the branch outcome must agree with the
       category and the convergence facts (a nonfinite force forces
       category nonconverged with converged=False and forces_finite
       False; a stop reason on any category other than nonconverged
       claims a run stopped before the rule was met, which contradicts
       the shape verdict).
    2. Summary versus the nested audit record: a supplied
       _audit_record["failure"] section must agree with the branch
       summary. A branch recorded as recovered (matched_reference or
       unmatched_converged, converged=True) while its own audit says
       failed=True, forces_finite=False or converged=False, is the
       contradiction review item 5 reproduces and is refused with the
       contradicting field named.
    3. The nested audit section against itself: a nested dict that
       records failed=True with failure_reason=None, or a failed=False
       entry that carries a declared failure reason, is self
       contradictory and refused.

    The runner's nested audit section emits a different vocabulary
    from the fixture script's: stop_reason, failure_reason,
    nonfinite_force and refusal_details, and no "failed" key. Those
    facts are reconciled on the same axes without requiring "failed":
    the effective stop reason (whichever of stop_reason /
    failure_reason is present and non-null, at outcome level and in
    the nested section) must be a declared reason and must agree with
    its twin where both appear; a declared stop reason on a branch
    the summary counts as a reached endpoint is a contradiction;
    nonfinite_force=True with forces_finite=True is a contradiction;
    and nonfinite_force=True under a summary claiming converged is a
    contradiction. Absent optional keys mean "not stated", never
    "no contradiction": nothing is defaulted in, on either path.

    Only presence and plain booleans are read out of the nested
    section: nested failure data the caller omitted is not defaulted
    into a fact, so a caller that supplies no audit record, or one
    without a failure section, is validated exactly as before.
    """
    category = outcome["category"]
    converged = outcome["converged"]
    budget_exhausted = outcome["budget_exhausted"]

    # Surface 1: summary versus the outcome-level force-failure facts.
    nonfinite = _bool_fact(outcome.get("nonfinite_force"))
    forces_finite = _bool_fact(outcome.get("forces_finite"))
    if converged and nonfinite:
        raise ValueError(
            f"{key}: the summary records converged while nonfinite_force "
            f"is recorded True; a run terminated by a nonfinite force is "
            f"never reported as converged (the contradiction must be "
            f"rejected, not repaired)")
    if converged and forces_finite is False:
        raise ValueError(
            f"{key}: the summary records converged while forces_finite is "
            f"recorded False; a run whose force values were nonfinite is "
            f"never reported as converged (the contradiction must be "
            f"rejected, not repaired)")
    if nonfinite is True and forces_finite is True:
        raise ValueError(
            f"{key}: the outcome records nonfinite_force True while "
            f"forces_finite is True; a run terminated by a nonfinite force "
            f"never reports finite forces (the contradiction must be "
            f"rejected, not repaired)")
    stop_reason = outcome.get("stop_reason")
    failure_reason = outcome.get("failure_reason")
    declared_stops = ("budget_exhausted", "nonfinite_force",
                      "invalid_force_shape", "no_progress")
    if stop_reason is not None and category != "nonconverged":
        raise ValueError(
            f"{key}: the summary category {category!r} carries stop_reason "
            f"{stop_reason!r}; a stop reason before the rule was met "
            f"contradicts a verdict scored on a reached endpoint (the "
            f"contradiction must be rejected, not repaired)")
    if failure_reason is not None and category != "nonconverged":
        raise ValueError(
            f"{key}: the summary category {category!r} carries failure_reason "
            f"{failure_reason!r}; a stop reason before the rule was met "
            f"contradicts a verdict scored on a reached endpoint (the "
            f"contradiction must be rejected, not repaired)")
    effective_reason = next((name for name in (stop_reason, failure_reason)
                             if name is not None), None)
    if stop_reason is not None and failure_reason is not None \
            and stop_reason != failure_reason:
        raise ValueError(
            f"{key}: stop_reason {stop_reason!r} and failure_reason "
            f"{failure_reason!r} name different ways the run stopped; they "
            f"carry one fact and must agree (the contradiction must be "
            f"rejected, not repaired)")
    if effective_reason is not None and effective_reason not in declared_stops:
        raise ValueError(
            f"{key}: unknown stop_reason {effective_reason!r}; the declared "
            f"reasons are {', '.join(declared_stops)} or None")

    # Surfaces 2 and 3: the nested audit failure section.
    record = outcome.get("_audit_record")
    if not isinstance(record, dict):
        return
    nested = _bool_fact(record.get("failure"))
    if not isinstance(nested, dict):
        return
    failed = _bool_fact(nested.get("failed"))
    nested_reason = nested.get("failure_reason")
    nested_budget = _bool_fact(nested.get("budget_exhausted"))
    nested_converged = _bool_fact(nested.get("converged"))
    nested_finite = _bool_fact(nested.get("forces_finite"))
    nested_stop = nested.get("stop_reason")
    nested_failure_reason = nested.get("failure_reason")
    nested_nonfinite = _bool_fact(nested.get("nonfinite_force"))
    audit = (
        f"audit failure section {nested!s} against summary category "
        f"{category!r}, converged={converged!r}, budget_exhausted="
        f"{budget_exhausted!r}")
    # Surface 2: nested audit versus the summary. The nested failure
    # section repeats the run's own force facts; it must agree with the
    # summary the branch is counted under, whatever the category.
    contradiction = None
    if (failed is True
            and category in ("matched_reference", "unmatched_converged",
                             "ambiguous", "unresolved")):
        contradiction = (
            "the audit records failed=True while the summary claims a "
            "shape or comparison verdict")
    elif converged and nested_finite is False:
        contradiction = (
            "the audit records forces_finite=False while the summary "
            "claims converged")
    elif converged and nested_converged is False:
        contradiction = (
            "the audit records converged=False while the summary claims "
            "converged")
    elif (not converged) and nested_converged is True:
        contradiction = (
            "the audit records converged=True while the summary records "
            "converged=False")
    elif category == "nonconverged" and failed is False:
        contradiction = (
            "the audit records failed=False while the summary records the "
            "branch nonconverged")
    elif (not budget_exhausted) and nested_budget is True:
        contradiction = (
            "the audit records budget_exhausted=True while the summary "
            "records budget_exhausted=False")
    elif budget_exhausted and nested_budget is False:
        contradiction = (
            "the audit records budget_exhausted=False while the summary "
            "records budget_exhausted=True")
    if contradiction:
        raise ValueError(
            f"{key}: contradictory failure audit: {contradiction}: "
            f"{audit}")

    # Surface 3: the nested failure section against itself.
    if failed is True and nested_reason is None:
        raise ValueError(
            f"{key}: contradictory failure audit: the audit records "
            f"failed=True with failure_reason=None; a branch that failed "
            f"records why (the contradiction must be rejected, not "
            f"repaired)")
    if (failed is False and isinstance(nested_reason, str)
            and nested_reason in declared_stops):
        raise ValueError(
            f"{key}: contradictory failure audit: the audit records "
            f"failed=False with the declared failure reason "
            f"{nested_reason!r}; the run cannot have stopped on a declared "
            f"reason it never met and stayed unfailed (the contradiction "
            f"must be rejected, not repaired)")

    # Surface 2, runner vocabulary (review round 3): the runner's nested
    # section has no "failed" key, so the checks above cannot see its
    # contradictions. Reconcile the facts the runner DOES emit, both
    # against the summary and against each other, without defaulting any
    # absent key into a fact. An absent key means "not stated", which on
    # its own is no contradiction; the section against itself still
    # refuses a declared reason on an unfailed branch, because an
    # unfailed run cannot have stopped on a declared reason at all.
    nested_effective = next((name for name in (nested_stop,
                                               nested_failure_reason)
                             if name is not None), None)
    if nested_effective is not None and category != "nonconverged":
        raise ValueError(
            f"{key}: contradictory failure audit: the audit declares stop "
            f"reason {nested_effective!r} while the summary category "
            f"{category!r} records a reached endpoint; the two cannot "
            f"describe the same run ({audit})")
    if nested_effective is not None and nested_effective not in declared_stops:
        raise ValueError(
            f"{key}: contradictory failure audit: the audit carries the "
            f"unknown stop reason {nested_effective!r}; the declared "
            f"reasons are {', '.join(declared_stops)} or None ({audit})")
    if (nested_stop is not None and nested_failure_reason is not None
            and nested_stop != nested_failure_reason):
        raise ValueError(
            f"{key}: contradictory failure audit: the audit records "
            f"stop_reason {nested_stop!r} and failure_reason "
            f"{nested_failure_reason!r}; the two name one way of stopping "
            f"and must agree ({audit})")
    if converged and nested_nonfinite is True:
        raise ValueError(
            f"{key}: contradictory failure audit: the audit records "
            f"nonfinite_force=True while the summary claims converged "
            f"({audit})")
    if nested_nonfinite is True and nested_finite is True:
        raise ValueError(
            f"{key}: contradictory failure audit: the audit records "
            f"nonfinite_force=True with forces_finite=True; a run "
            f"terminated by a nonfinite force never reports finite forces "
            f"({audit})")


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
    # start identity, arm identity, the effective procedure with any
    # labelled synthetic override, RMSD and mapping, failure facts and
    # actual force-call and step counts, so the saved report can audit
    # the own-reference match, the actual force supplier and the budget
    # without reopening the run. Rows without an audit record show
    # blank cells rather than invented facts.
    lines.append("| branch_key | partition | category | reference | start | arm | "
                 "procedure | "
                 "final_positions | rmsd_A | permutation | rotation | failure | "
                 "force_evaluation_count | step_count |")
    lines.append("|---|" + "---|" * 13)
    audit_keys = ("reference", "start", "arm", "procedure", "final_positions",
                  "rmsd_A", "permutation", "rotation", "failure",
                  "force_evaluation_count", "step_count")
    for row in diagnostics["per_branch"]:
        cells = [row["branch_key"], row["partition"], row["category"]]
        cells += [_audit_cell(row.get(name)) for name in audit_keys]
        lines.append("| " + " | ".join(str(cell) for cell in cells) + " |")
    # The nonrigid positive fixture's measured evidence, when the caller
    # produced one: outside the tolerance before relaxation, matched
    # after. Blank (absent) for callers that did not run the fixture.
    nonrigid = diagnostics.get("nonrigid_positive_fixture")
    if nonrigid:
        outcome = nonrigid["outcome"]
        lines.append("")
        lines.append(
            "Nonrigid positive fixture: pre-relaxation rmsd_A "
            f"{float(nonrigid['pre_relaxation_rmsd_A']):.6f} is OUTSIDE "
            f"the {float(nonrigid['tolerance_A'])} angstrom match "
            f"tolerance; after relaxation the category is "
            f"{outcome['category']} with rmsd_A "
            f"{float(outcome['rmsd_A']):.6f}.")
    lines.append("")
    # The timing note, when the caller supplied one: it states where
    # per-branch timing IS recorded and why this artifact carries
    # timing_seconds as null, so a reader of the rendered report is never
    # left guessing why a listed fact is empty.
    timing_note = diagnostics.get("timing_note")
    if timing_note:
        lines.append(f"Timing: {timing_note}")
    lines.append(_provenance_line(report))
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
        "wrong shape.")


def _provenance_line(report):
    """The Markdown provenance sentence naming the ACTUAL producing arm.

    The arm comes from the report's own provenance record (review item
    2), which build_report derived from the branches' audit records: a
    single arm is named directly, a mixed set is listed in full, and a
    report with no audit record carries the explicit not-recorded
    sentence instead of a claimed arm.
    """
    provenance = report.get("provenance") or {}
    source = provenance.get("source")
    arm = provenance.get("arm")
    if source == "audit_record" and isinstance(arm, dict) and "mixed" not in arm:
        arm_id = arm.get("arm_id")
        kind = arm.get("kind")
        if arm_id is not None:
            return (f"Provenance: these numbers were produced by arm "
                    f"{arm_id!r} (kind {kind!r}).")
        return (f"Provenance: these numbers were produced by an arm of "
                f"kind {kind!r} that recorded no arm id.")
    if source == "audit_record" and isinstance(arm, dict):
        sections = [section for section in arm.get("mixed", [])
                    if isinstance(section, dict)]
        # The arm id alone can repeat across branches that ran under
        # DIFFERENT suppliers (the fixture slice has exactly one such
        # pair), so when the sections disagree on the supplier the
        # supplier is named too, and the rendered list is de-duplicated:
        # a provenance line that prints the same id twice tells the reader
        # nothing about what actually ran.
        suppliers = {section.get("force_supplier") for section in sections}
        listed = []
        for section in sections:
            entry = str(section.get("arm_id"))
            supplier = section.get("force_supplier")
            if len(suppliers) > 1 and supplier:
                entry = f"{entry} via {supplier!r}"
            if entry not in listed:
                listed.append(entry)
        return (f"Provenance: these numbers were produced by several arms "
                f"({'; '.join(listed)}).")
    return ("Provenance: the arm that produced these numbers is not "
            "recorded in this report.")