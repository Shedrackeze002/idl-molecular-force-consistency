"""Report generator for the Stage 2 movement-cap study (plan milestone M5).

The report is an analysis of saved result rows: the rows `stage2_capped.row_schema`
validates and the label records `stage2_capped.rules.analyze` produces. This module
loads no model, runs no relaxation, opens no test partition and needs no network. The
only file read is the accepted Stage 0 result set, read-only, for the historical
50-row recovery table.

Whether the supplied rows are synthetic or measured is a property of the rows, not of
this renderer, so the caller states it and the report prints it. The renderer never
infers it from the fact that no model is loaded.

Before any label is produced, the row inventory is checked as a whole: the approved
checkpoint, arm, seed and lambda grid, one checkpoint identity per grid cell, one
capped relaxer identity, binding agreement between each c = 1.00 A run and its control,
and the per-cap displacement bound. A table that is individually valid row by row but
mutually inconsistent is refused rather than labelled.

Every label prints the evidence the specification requires beside it: the interpretable
items out of 20, the connections and own minima they cover, the adequacy result, the
reference-start recoveries per checkpoint, the set E, the caps that are not evaluable
and why, the favoured arm including `direct`, the sweep caveats, the section 9 claim
limits, and the attribution status with its qualifying and non-qualifying flips.

Wording is fixed by the specification: "descriptive difference", never "effect" or
"significant", and every lambda = 1 table and caption carries the exact text
"exploratory, post-hoc, not MAE-matched". `render_document` refuses text that breaks
the wording rule, so a wording regression fails at render time and not only in a test.

The output is a dict of section name to markdown so a test can pin each printing rule
to one section, plus a writer that refuses to overwrite and produces identical bytes on
a second run.
"""
from __future__ import annotations

import re
from pathlib import Path

from stage2_capped import rules, stage0_compare
from stage2_capped.row_schema import CAP_VALUES, CONTROL_LABEL, EXPLORATORY_LABEL

# Section names, in the order the document prints them.
ROW_PROVENANCE = "row provenance"
STATUS = "report status"
PRIMARY = "primary comparison (lambda = 0.1)"
EXPLORATORY = "exploratory comparison (lambda = 1)"
ATTRIBUTION = "attribution"
PER_RUN_PRIMARY = "per-run evidence (lambda = 0.1)"
PER_RUN_EXPLORATORY = "per-run evidence (lambda = 1)"
REFERENCE_RUNS = "reference-start binding"
LIMITATIONS = "study limitations"
HISTORICAL = "historical 50-row recovery table"

SECTION_ORDER = (ROW_PROVENANCE, STATUS, PRIMARY, EXPLORATORY, ATTRIBUTION,
                 PER_RUN_PRIMARY, PER_RUN_EXPLORATORY, REFERENCE_RUNS, LIMITATIONS,
                 HISTORICAL)

CAP_TEXT = {"c0p05": "0.05 A", "c0p10": "0.10 A", "c0p25": "0.25 A", "c1p00": "1.00 A",
            CONTROL_LABEL: "no cap"}
CHECKPOINT_NAMES = ("energy_s0", "direct_s0", "direct_s1", "consistency_s0", "consistency_s1")
ARM_ORDER = ("energy", "direct", "consistency")
# The specification's own R-attribution phrase, quoted verbatim. It is the one
# allowed place where a string containing "effects" may appear.
ALLOWED_PHRASE = rules.NOT_ATTRIBUTABLE

# The approved grid: (arm, seed, consistency_lambda). Anything else is refused.
APPROVED_CHECKPOINTS = {
    ("energy", 0, None), ("direct", 0, None), ("direct", 1, None),
    ("consistency", 0, 0.1), ("consistency", 1, 0.1),
}
EXPLORATORY_CHECKPOINTS = {("consistency", 0, 1.0), ("consistency", 1, 1.0)}

# The section 9 claim limits, carried with the report. The sweep caveats alone are
# not the report's limitations.
CLAIM_LIMITS = (
    "every capped result is post-hoc: the caps, the lambda = 1 arm and the decision "
    "rules were fixed after the 0/10 result and the traces on the same validation data",
    "validation is reused: ten starts come from five connections, and six of the ten "
    "target two minima, so five connections overstates independence, and all ten "
    "own-reference minima are seen minima",
    "the interpretable items are a selected subset, the starts whose own minimum both "
    "arms reach from the reference, so conclusions apply only to that subset",
    "four caps over five to seven checkpoints are multiple looks at the same data",
    "one energy seed and two seeds per arm give no variance estimate beyond the "
    "placebo check, which is a crude screen, and the seed pairs share their seed value",
    "under a binding cap the step is the cap times the force direction, so force "
    "magnitude is discarded: if consistency training mainly repairs magnitude mismatch "
    "a cap can hide that benefit and a null does not refute it, and the results hold "
    "only for the capped procedure",
    "the 1.00 A cap is not shown to remove only jumps",
    "a recovery at a cap does not show the cap is optimal, a failure does not prove no "
    "nearby stationary point exists, and no energy-decrease condition is added",
    "standard optimizers such as FIRE and L-BFGS, and lambda = 10 recovery, remain "
    "untested and need their own reviewed plan",
)

EXPLORATORY_QUALIFICATION = (
    "lambda = 1 is not MAE-matched: equal-connection force MAE is 34.683 meV/A "
    "(33.723 to 35.643) against direct 38.186 (37.291 to 39.082), so the report gives "
    "an accuracy, cost and recovery frontier rather than a matched comparison, and "
    "lambda = 1 at c = infinity is a new post-hoc result"
)


class ReportError(ValueError):
    """The rows cannot support a report, or the rendered text breaks a rule."""


# ---- Small formatting helpers -------------------------------------------

def _num(value, digits=3, missing="-"):
    return missing if value is None else f"{value:.{digits}f}"


def _yes(value, missing="-"):
    return missing if value is None else ("yes" if value else "no")


def _arm(value):
    return value if value in ("direct", "consistency") else "neither arm"


def _table(header, rows):
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _caption(text, exploratory=False):
    """A table caption; every lambda = 1 caption carries the exploratory label."""
    return f"Table: {text}" + (f" ({EXPLORATORY_LABEL})" if exploratory else "")


def _bullet_list(items):
    return "\n".join(f"- {item}" for item in items)


# ---- Row inventory: refusals before any label ---------------------------

def _triple(row):
    return (row["arm"], row["seed"], row["consistency_lambda"])


def inventory_problems(rows):
    """Every way the row inventory is inconsistent as a whole, not row by row.

    A schema-valid row can still belong to a grid the study never ran, carry a second
    capped relaxer, or claim a binding its control contradicts. Any finding here is a
    refusal, produced before a label exists.
    """
    problems = []
    allowed = APPROVED_CHECKPOINTS | EXPLORATORY_CHECKPOINTS

    # 1. The approved checkpoint, arm, seed and lambda grid.
    unknown = sorted({_triple(row) for row in rows if _triple(row) not in allowed}, key=str)
    if unknown:
        problems.append("checkpoints outside the approved grid: "
                        + "; ".join(str(item) for item in unknown))

    # 2. One checkpoint identity per grid cell, so an identity cannot be split or shared.
    by_id, by_triple = {}, {}
    for row in rows:
        by_id.setdefault(row["checkpoint_id"], set()).add(_triple(row))
        by_triple.setdefault(_triple(row), set()).add(row["checkpoint_id"])
    for name, triples in sorted(by_id.items()):
        if len(triples) != 1:
            problems.append(f"checkpoint identity {name!r} carries several identities: "
                            + "; ".join(str(item) for item in sorted(triples)))
    for triple, names in sorted(by_triple.items(), key=lambda item: str(item[0])):
        if len(names) != 1:
            problems.append(f"identity {triple} is split across ids: "
                            + ", ".join(sorted(names)))

    # 3. One capped relaxer identity across every capped run.
    relaxed = {row["relaxer_sha256"] for row in rows if row["cap_label"] != CONTROL_LABEL}
    if len(relaxed) > 1:
        problems.append(f"capped rows carry {len(relaxed)} relaxer identities: "
                        + ", ".join(sorted(relaxed)))

    # 4. Binding agreement between each c = 1.00 A run and its control, and the
    #    control's own flag against its recorded displacement.
    by_key = {(row["cap_label"], _triple(row), row["start_kind"], row["start_id"]): row
              for row in rows}
    for key, row in sorted(by_key.items(), key=lambda item: str(item[0])):
        cap, triple, kind, start_id = key
        if cap == CONTROL_LABEL:
            # A cap can bind on a rejected trial, so a control may legitimately report
            # would_bind_at_1p0 with no accepted step above 1 A. Only the other direction
            # is a contradiction: an accepted step above 1 A without binding.
            if (row["max_accepted_displacement_A"] is not None and
                    row["max_accepted_displacement_A"] > 1.0 and
                    not row["would_bind_at_1p0"]):
                problems.append(
                    f"control run {triple} {kind} {start_id} claims "
                    f"would_bind_at_1p0=False but has an accepted step of "
                    f"{row['max_accepted_displacement_A']} A")
        elif cap == "c1p00":
            control = by_key.get((CONTROL_LABEL, triple, kind, start_id))
            if control is not None:
                bound = row["capped_trial_count"] > 0
                if bound != control["would_bind_at_1p0"]:
                    problems.append(
                        f"c1p00 run {triple} {kind} {start_id} records bound={bound} "
                        f"while its control says would_bind_at_1p0="
                        f"{control['would_bind_at_1p0']}")

    # 5. The per-cap displacement bound on capped runs.
    for row in rows:
        if row["cap_label"] == CONTROL_LABEL or row["max_accepted_displacement_A"] is None:
            continue
        if row["max_accepted_displacement_A"] > CAP_VALUES[row["cap_label"]] * (1 + 1e-12):
            problems.append(
                f"{row['cap_label']} run {_triple(row)} {row['start_kind']} "
                f"{row['start_id']} moved {row['max_accepted_displacement_A']} A, above its cap")
    return problems


# ---- Cap-level tables ----------------------------------------------------

def _reference_text(counts):
    total = counts.get("reference_starts", 0)
    parts = [f"{name} {counts[name]}/{total}" for name in CHECKPOINT_NAMES if name in counts]
    return ", ".join(parts)


def _adequacy_text(adequacy):
    if not adequacy:
        return "-"
    return (f"{adequacy['recovered_at_cap']} of {adequacy['starts']} at the cap, "
            f"{adequacy['recovered_at_control']} at the control "
            f"({'pass' if adequacy['passed'] else 'fail'})")


def _cap_note(record, sweep_label):
    """The note printed beside one cap's label.

    A zero-contrast cap gets the sentence the handover fixes word for word, keyed off
    the per-cap status rather than the sweep's `zero_contrast_caps` list (which also
    carries floor and gate caps). A placebo cap says it is not distinguishable from
    seed noise. Neither is ever printed as a null.
    """
    detail = record["detail"]
    status = detail.get("status")
    cap = record["cap_label"]
    if status == "zero_contrast":
        return (f"at {cap}, neither arm recovered any interpretable original-start "
                f"item (0 of {record['interpretable_items']})")
    if status == "placebo":
        return (f"at {cap}, not distinguishable from seed noise: the cap leaves E and "
                f"supports no direction")
    if status == "gate":
        return (f"at {cap}, uninformative (gate): the energy C seed 0 baseline does not "
                f"pass R-adequacy")
    if status == "floor":
        return f"at {cap}, uninformative (floor): " + "; ".join(record["detail"]["support"]["reasons"])
    return f"at {cap}, evaluable: the sweep's label applies ({sweep_label})"


def _comparison_text(comparison, exploratory):
    """One comparison: the cap table, the notes beside it, the sweep and the caveats."""
    sweep = comparison["sweep_record"]
    lines = []
    if exploratory:
        lines += [f"Comparison label: {EXPLORATORY_LABEL}", ""]
    lines += [_caption("cap-level labels and the evidence printed beside each one",
                       exploratory), ""]
    rows = []
    for record in comparison["cap_records"]:
        rows.append((f"{record['cap_label']} ({CAP_TEXT[record['cap_label']]})", record["label"],
                     f"{record['interpretable_items']} of 20",
                     record["connections_covered"], record["minima_covered"],
                     _adequacy_text(record["detail"]["adequacy"]),
                     _reference_text(record["reference_start_counts"]),
                     "yes" if record["cap_label"] in record["set_E"] else "no",
                     _arm(record["detail"].get("favors_own_items"))))
    lines.append(_table(("cap", "label", "interpretable items", "connections", "own minima",
                         "adequacy (energy C seed 0)", "reference recoveries", "in E",
                         "favours (own items)"), rows))
    lines.append("")
    lines += [f"Sweep label: {sweep['label']}", ""]
    detail = sweep["detail"]
    lines.append(_bullet_list([
        f"reason: {detail.get('reason', '-')}",
        "set E: " + (", ".join(detail.get("set_E") or []) or "empty"),
        "not evaluable: " + (", ".join(sweep["not_evaluable"]) or "none"),
        f"interpretable items in the sweep: {sweep['interpretable_items']} over "
        f"{sweep['connections_covered']} connections and {sweep['minima_covered']} own minima",
    ]))
    lines.append("")
    lines += ["Favoured arm", ""]
    lines.append(_bullet_list(_favoured_lines(comparison)))
    lines.append("")
    lines += ["Notes beside the labels", "", _bullet_list([_cap_note(r, sweep["label"])
                                                          for r in comparison["cap_records"]])]
    lines.append("")
    lines += ["Caveats carried with this comparison", "", _bullet_list(detail.get("caveats") or [])]
    return "\n".join(lines)


def _favoured_lines(comparison):
    """The favoured arm, always, including `direct`.

    A single-cap observation reads `r_direction_holds_at_the_cap` and
    `favors_at_the_cap`, because the sweep's own `favors` is null in that case.
    """
    sweep = comparison["sweep_record"]
    detail = sweep["detail"]
    label = sweep["label"]
    if label == rules.SINGLE_CAP:
        return [f"single-cap observation: the favoured arm at the cap is "
                f"{_arm(detail.get('favors_at_the_cap'))}",
                f"R-direction holds at the cap: {_yes(detail.get('r_direction_holds_at_the_cap'))}"]
    if label == rules.DIRECTION_STATED:
        return [f"the favoured arm is {_arm(detail.get('favors'))}"]
    own = detail.get("own_items") or {}
    if own:
        return [f"on its own items, {cap}: {_arm(values.get('favors'))}"
                for cap, values in own.items()]
    return ["no cap in E carries a favoured arm"]


# ---- Per-run evidence ----------------------------------------------------

def _checkpoint_order(row):
    arm = ARM_ORDER.index(row["arm"]) if row["arm"] in ARM_ORDER else len(ARM_ORDER)
    return (arm, row["seed"], row["consistency_lambda"] if row["consistency_lambda"] else -1.0)


def _is_exploratory_row(row):
    return row["consistency_lambda"] == 1.0


def _pooled_fraction(runs, field):
    """A fraction of accepted steps, pooled over accepted-step counts.

    Averaging per-run fractions equally would weight a five-step run like a 900-step
    run. A run with no accepted steps contributes nothing; if no run has an accepted
    step the pool is empty and the cell is blank rather than zero.
    """
    weighted = [(run[field], run["accepted_trial_count"]) for run in runs
                if run[field] is not None]
    steps = sum(count for _, count in weighted)
    if not weighted or steps == 0:
        return None
    return sum(fraction * count for fraction, count in weighted) / steps


def _grouped(rows, kind, exploratory):
    grouped = {}
    for row in rows:
        if row["start_kind"] != kind or _is_exploratory_row(row) != exploratory:
            continue
        grouped.setdefault((row["cap_label"], row["checkpoint_id"]), []).append(row)
    return grouped


def _cap_rows(grouped):
    caps = (CONTROL_LABEL,) + tuple(rules.CAP_ORDER)
    for cap in caps:
        keys = sorted((key for key in grouped if key[0] == cap),
                      key=lambda key: _checkpoint_order(grouped[key][0]))
        for key in keys:
            yield cap, grouped[key]


def _per_run_table(rows, exploratory):
    grouped = _grouped(rows, "original", exploratory)
    header = ("cap", "checkpoint", "runs", "accepted travel (A)", "attempted travel (A)",
              "rejected trials", "capped steps", "first capped attempt",
              "accepted steps > 0.25 A (pooled)", "accepted steps > 1.0 A (pooled)",
              "binds at c = 1.00 A")
    table_rows = []
    for cap, runs in _cap_rows(grouped):
        trials = sum(run["trial_count"] for run in runs)
        capped = sum(run["capped_trial_count"] for run in runs)
        firsts = [run["first_capped_attempt"] for run in runs
                  if run["first_capped_attempt"] is not None]
        binds = cap == "c1p00" and capped > 0
        table_rows.append((
            cap, runs[0]["checkpoint_id"], len(runs),
            _num(sum(run["accepted_travel_A"] for run in runs)),
            _num(sum(run["attempted_travel_A"] for run in runs)),
            sum(run["rejected_trial_count"] for run in runs),
            f"{capped} of {trials}" + (f" ({capped / trials:.3f})" if trials else ""),
            min(firsts) if firsts else "never",
            _num(_pooled_fraction(runs, "accepted_step_fraction_gt_0p25")),
            _num(_pooled_fraction(runs, "accepted_step_fraction_gt_1p0")),
            _yes(binds) if cap == "c1p00" else "-"))
    lines = [_caption("per cap and checkpoint for original starts: travel, rejections, "
                      "capped steps and the control step fractions", exploratory), "",
             _table(header, table_rows), "",
             "The step-fraction columns pool over accepted-step counts, so a run with no "
             "accepted steps contributes nothing."]
    return "\n".join(lines)


def _reference_run_table(rows, exploratory):
    """Reference-start binding, reported separately from the original starts."""
    grouped = _grouped(rows, "reference", exploratory)
    header = ("cap", "checkpoint", "reference runs", "capped steps",
              "first capped attempt", "binds at c = 1.00 A")
    table_rows = []
    for cap, runs in _cap_rows(grouped):
        trials = sum(run["trial_count"] for run in runs)
        capped = sum(run["capped_trial_count"] for run in runs)
        firsts = [run["first_capped_attempt"] for run in runs
                  if run["first_capped_attempt"] is not None]
        here = cap == "c1p00" and capped > 0
        table_rows.append((cap, runs[0]["checkpoint_id"], len(runs),
                           f"{capped} of {trials}", min(firsts) if firsts else "never",
                           _yes(here) if cap == "c1p00" else "-"))
    return "\n".join([_caption("reference-start binding by cap and checkpoint", exploratory),
                      "", _table(header, table_rows), ""])


def _binding_sentence(rows):
    """The c = 1.00 A duplication claim, over both start kinds.

    The claim is about the cap as a whole, so it may not be decided from original starts
    alone: a table whose only binding runs are reference starts would otherwise be read
    as if c = 1.00 A were the control.
    """
    binding = [row for row in rows
               if row["cap_label"] == "c1p00" and row["capped_trial_count"] > 0]
    if binding:
        kinds = ", ".join(sorted({row["start_kind"] for row in binding}))
        return (f"{len(binding)} run(s) bind at c = 1.00 A ({kinds} starts), so c1p00 is "
                f"not a duplicate of the control.")
    return ("No original-start or reference-start run binds at c = 1.00 A, so c1p00 "
            "duplicates the control.")


def _rows_have_lambda_one(rows):
    return any(_is_exploratory_row(row) for row in rows)


# ---- Attribution ---------------------------------------------------------

def _attribution_text(attribution):
    """R-attribution, matched on `base_status` and never on the suffixed `status`."""
    lines = [f"Status: {attribution['base_status']}", ""]
    if attribution.get("lambda1_included"):
        primary_only = attribution.get("supported_by_primary_runs_alone")
        if primary_only is None:
            primary_only = attribution["qualifying_primary"] >= rules.MIN_ATTRIBUTION_RUNS
        lines += [f"Exploratory warning: {EXPLORATORY_LABEL} runs are included in this "
                  f"status; primary runs alone support it: {_yes(primary_only)}", ""]
    items = [f"lambda = 1 rows included: {_yes(attribution.get('lambda1_included'))}",
             f"qualifying runs at c = 1.00 A: {len(attribution['qualifying_runs'])} "
             f"(primary {attribution['qualifying_primary']}, "
             f"exploratory {attribution['qualifying_lambda1']})"]
    lines.append(_bullet_list(items))
    lines += ["", "Qualifying runs", ""]
    if attribution["qualifying_runs"]:
        lines.append(_bullet_list([
            f"{run['checkpoint_id']} / {run['start_id']} "
            f"({'primary' if run['primary'] else 'exploratory'})"
            for run in attribution["qualifying_runs"]]))
    else:
        lines.append("- none")
    lines += ["", "Non-qualifying flips", ""]
    if attribution["non_qualifying_flips"]:
        lines.append(_bullet_list([
            f"{flip['checkpoint_id']} / {flip['start_id']} "
            f"({'primary' if flip['primary'] else 'exploratory'}): failed: "
            + "; ".join(flip["failed"])
            for flip in attribution["non_qualifying_flips"]]))
    else:
        lines.append("- none")
    lines += ["", "The condition is necessary, not sufficient."]
    return "\n".join(lines)


# ---- Provenance, limitations, historical table ---------------------------

def _provenance_text(provenance):
    """The row provenance, stated by the caller rather than inferred from rendering."""
    if provenance == "synthetic":
        return ("Row provenance: synthetic rows. No model produced them, so no measured "
                "model result is claimed from this table.")
    if provenance == "measured":
        return ("Row provenance: measured rows from a Stage 2 launch. The labels below "
                "report those measured results.")
    return ("Row provenance: not stated by the caller. This renderer does not decide "
            "whether the rows are synthetic or measured, so no claim is made either way.")


def _limitations_text(include_exploratory):
    items = list(CLAIM_LIMITS)
    if include_exploratory:
        items.append(EXPLORATORY_QUALIFICATION)
    return _bullet_list(items)


def _historical_text(saved):
    if saved is None:
        try:
            saved = stage0_compare.load_saved_original()
        except (FileNotFoundError, stage0_compare.SavedResultError) as error:
            return ("The accepted records are not present in this checkout, so the "
                    f"historical table is not reproduced here ({error}).")
    rows = []
    total_recovered = 0
    for model in stage0_compare.MODEL_ORDER:
        keys = sorted(key for key in saved if key[0] == model)
        recovered = sum(1 for key in keys if saved[key].get("category") == "matched_reference")
        total_recovered += recovered
        categories = {}
        for key in keys:
            categories[saved[key].get("category", "unknown")] = \
                categories.get(saved[key].get("category", "unknown"), 0) + 1
        rows.append((model, recovered, len(keys),
                     ", ".join(f"{name} {count}" for name, count in sorted(categories.items()))))
    lines = [_caption("the accepted 50-row original-start recovery table, read-only"), "",
             "Row provenance: measured accepted records, read-only.", "",
             _table(("checkpoint", "recovered", "of starts", "outcome categories"), rows), "",
             f"Total: {total_recovered} of {sum(row[2] for row in rows)} accepted "
             f"original-start runs recovered. This reproduces the accepted records before "
             f"any new number is produced."]
    return "\n".join(lines)


# ---- Top level -----------------------------------------------------------

def render_report(rows, saved_original=None, historical=True, provenance=None):
    """Section name to markdown, for one set of saved result rows.

    `provenance` is "synthetic" or "measured" and is printed as given; when it is None
    the report says so and claims nothing either way. `saved_original` may be the dict
    `stage0_compare.load_saved_original` returns; when it is None the loader is tried,
    and an absent accepted set is reported in words rather than ignored.
    """
    rows = list(rows)
    if not rows:
        raise ReportError("no rows: the report is an analysis of saved result rows, and "
                          "this set is empty")
    problems = inventory_problems(rows)
    if problems:
        raise ReportError("row inventory is inconsistent, so no label is produced: "
                          + "; ".join(problems))
    try:
        analysis = rules.analyze(rows)
    except rules.TableError as error:
        return {ROW_PROVENANCE: _provenance_text(provenance),
                STATUS: f"incomplete, not labelled: {error}"}
    except ValueError as error:  # a row that breaks the schema
        raise ReportError(f"rows cannot be labelled: {error}") from None

    sections = {ROW_PROVENANCE: _provenance_text(provenance),
                PRIMARY: _comparison_text(analysis["primary"], exploratory=False)}
    exploratory = analysis.get("exploratory")
    if exploratory is None:
        sections[EXPLORATORY] = ("not labelled: this table carries no lambda = 1 rows, so "
                                 "there is no exploratory comparison to report "
                                 f"({EXPLORATORY_LABEL})")
    elif exploratory.get("status") == "incomplete, not labelled":
        sections[EXPLORATORY] = (f"incomplete, not labelled: {exploratory.get('reason', '-')}"
                                 f" ({EXPLORATORY_LABEL})")
    else:
        sections[EXPLORATORY] = _comparison_text(exploratory, exploratory=True)
    sections[ATTRIBUTION] = _attribution_text(analysis["attribution"])
    sections[PER_RUN_PRIMARY] = _per_run_table(rows, exploratory=False)
    if _rows_have_lambda_one(rows):
        sections[PER_RUN_EXPLORATORY] = _per_run_table(rows, exploratory=True)
    sections[REFERENCE_RUNS] = (_reference_run_table(rows, exploratory=False) + "\n"
                                + _binding_sentence(rows))
    sections[LIMITATIONS] = _limitations_text(
        include_exploratory=_rows_have_lambda_one(rows))
    if historical:
        sections[HISTORICAL] = _historical_text(saved_original)
    return sections


def render_document(sections):
    """The whole report as one markdown document, with the wording rule enforced."""
    lines = ["# Stage 2 movement-cap report (plan milestone M5)", "",
             "An analysis of saved result rows. The renderer loads no model and runs no "
             "relaxation; the provenance of the supplied rows is stated in the row "
             "provenance section.", ""]
    for name in SECTION_ORDER:
        if name not in sections:
            continue
        heading = name[0].upper() + name[1:]
        lines += [f"## {heading}", "", sections[name].rstrip(), ""]
    for name in sections:
        if name not in SECTION_ORDER:
            raise ReportError(f"unknown report section {name!r}")
    text = "\n".join(lines).rstrip() + "\n"
    _check_wording(text)
    return text


def _check_wording(text):
    """Refuse the two words the specification bans, outside its own quoted phrase."""
    stripped = text.replace(ALLOWED_PHRASE, "")
    for word in ("effect", "significant"):
        if re.search(word, stripped, flags=re.IGNORECASE):
            raise ReportError(f"the report may not use the word {word!r} about the "
                              f"consistency-versus-direct comparison")


def write_report(sections, out_dir, stem="stage2-report"):
    """Write the rendered report under `out_dir`, refusing to overwrite."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.md"
    text = render_document(sections)
    with open(path, "x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return path
