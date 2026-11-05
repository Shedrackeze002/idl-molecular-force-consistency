"""Report generator for the Stage 2 movement-cap study (plan milestone M5).

The report is built from saved result rows only: the rows `stage2_capped.row_schema`
validates and the label records `stage2_capped.rules.analyze` produces. This module
loads no model, runs no relaxation, opens no test partition and needs no network. The
only file read is the accepted Stage 0 result set, read-only, for the historical
50-row recovery table.

Every label prints the evidence the specification requires beside it: the
interpretable items out of 20, the connections and own minima they cover, the
adequacy result, the reference-start recoveries per checkpoint, the set E, the caps
that are not evaluable and why, the favoured arm including `direct`, the three
caveats, and the attribution status with its qualifying and non-qualifying flips.

Wording is fixed by the specification: "descriptive difference", never "effect" or
"significant", and every lambda = 1 table and caption carries the exact text
"exploratory, post-hoc, not MAE-matched". `render_document` refuses text that breaks
the wording rule, so a wording regression fails at render time and not only in a
test.

The output is a dict of section name to markdown so a test can pin each printing
rule to one section, plus a writer that refuses to overwrite and produces identical
bytes on a second run.
"""
from __future__ import annotations

import re
from pathlib import Path

from stage2_capped import rules, stage0_compare
from stage2_capped.row_schema import CONTROL_LABEL, EXPLORATORY_LABEL

# Section names, in the order the document prints them.
PRIMARY = "primary comparison (lambda = 0.1)"
EXPLORATORY = "exploratory comparison (lambda = 1)"
ATTRIBUTION = "attribution"
PER_RUN_PRIMARY = "per-run evidence (lambda = 0.1)"
PER_RUN_EXPLORATORY = "per-run evidence (lambda = 1)"
HISTORICAL = "historical 50-row recovery table"
STATUS = "report status"

SECTION_ORDER = (STATUS, PRIMARY, EXPLORATORY, ATTRIBUTION, PER_RUN_PRIMARY,
                 PER_RUN_EXPLORATORY, HISTORICAL)

PREAMBLE = (
    "Built from saved result rows only. No model is loaded and no relaxation is run, "
    "so nothing here is a measured model result: every label is a reading of rows that "
    "were already saved."
)

CAP_TEXT = {"c0p05": "0.05 A", "c0p10": "0.10 A", "c0p25": "0.25 A", "c1p00": "1.00 A",
            CONTROL_LABEL: "no cap"}
CHECKPOINT_NAMES = ("energy_s0", "direct_s0", "direct_s1", "consistency_s0", "consistency_s1")
ARM_ORDER = ("energy", "direct", "consistency")
# The specification's own R-attribution phrase, quoted verbatim. It is the one
# allowed place where a string containing "effects" may appear.
ALLOWED_PHRASE = rules.NOT_ATTRIBUTABLE


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


def _per_run_table(rows, exploratory):
    grouped = {}
    for row in rows:
        if row["start_kind"] != "original" or _is_exploratory_row(row) != exploratory:
            continue
        grouped.setdefault((row["cap_label"], row["checkpoint_id"]), []).append(row)
    header = ("cap", "checkpoint", "runs", "accepted travel (A)", "attempted travel (A)",
              "rejected trials", "capped steps", "first capped attempt",
              "accepted steps > 0.25 A", "accepted steps > 1.0 A", "binds at c = 1.00 A")
    table_rows = []
    binds_anywhere = False
    caps = (CONTROL_LABEL,) + tuple(rules.CAP_ORDER)
    for cap in caps:
        keys = sorted((key for key in grouped if key[0] == cap),
                      key=lambda key: _checkpoint_order(grouped[key][0]))
        for key in keys:
            runs = grouped[key]
            trials = sum(run["trial_count"] for run in runs)
            capped = sum(run["capped_trial_count"] for run in runs)
            firsts = [run["first_capped_attempt"] for run in runs
                      if run["first_capped_attempt"] is not None]
            fractions_025 = [run["accepted_step_fraction_gt_0p25"] for run in runs
                             if run["accepted_step_fraction_gt_0p25"] is not None]
            fractions_10 = [run["accepted_step_fraction_gt_1p0"] for run in runs
                            if run["accepted_step_fraction_gt_1p0"] is not None]
            binds = cap == "c1p00" and capped > 0
            binds_anywhere = binds_anywhere or binds
            table_rows.append((
                cap, runs[0]["checkpoint_id"], len(runs),
                _num(sum(run["accepted_travel_A"] for run in runs)),
                _num(sum(run["attempted_travel_A"] for run in runs)),
                sum(run["rejected_trial_count"] for run in runs),
                f"{capped} of {trials}" + (f" ({capped / trials:.3f})" if trials else ""),
                min(firsts) if firsts else "never",
                _num(sum(fractions_025) / len(fractions_025)) if fractions_025 else "-",
                _num(sum(fractions_10) / len(fractions_10)) if fractions_10 else "-",
                _yes(binds) if cap == "c1p00" else "-"))
    lines = [_caption("per cap and checkpoint: travel, rejections, capped steps and the "
                      "control step fractions", exploratory), "", _table(header, table_rows), ""]
    if not binds_anywhere:
        lines.append("No run binds at c = 1.00 A, so c1p00 duplicates the control.")
    return "\n".join(lines)


def _rows_have_lambda_one(rows):
    return any(_is_exploratory_row(row) for row in rows)


# ---- Attribution ---------------------------------------------------------

def _attribution_text(attribution):
    """R-attribution, matched on `base_status` and never on the suffixed `status`."""
    lines = [f"Status: {attribution['base_status']}", ""]
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
    if attribution.get("status") != attribution["base_status"]:
        lines.append("The status line in the record carries the exploratory label because "
                     "lambda = 1 runs contribute; the plain value is printed above.")
    return "\n".join(lines)


# ---- Historical table ----------------------------------------------------

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
    lines = [_caption("the accepted 50-row original-start recovery table, read-only"),
             "",
             _table(("checkpoint", "recovered", "of starts", "outcome categories"), rows),
             "",
             f"Total: {total_recovered} of {sum(row[2] for row in rows)} accepted "
             f"original-start runs recovered. This reproduces the accepted records before "
             f"any new number is produced."]
    return "\n".join(lines)


# ---- Top level -----------------------------------------------------------

def render_report(rows, saved_original=None, historical=True):
    """Section name to markdown, for one set of saved result rows.

    `rows` are validated result rows. `saved_original` may be the dict
    `stage0_compare.load_saved_original` returns; when it is None the loader is
    tried, and an absent accepted set is reported in words rather than ignored.
    """
    rows = list(rows)
    if not rows:
        raise ReportError("no rows: the report is built from saved result rows, and this "
                          "set is empty")
    try:
        analysis = rules.analyze(rows)
    except rules.TableError as error:
        return {STATUS: f"incomplete, not labelled: {error}"}
    except ValueError as error:  # a row that breaks the schema
        raise ReportError(f"rows cannot be labelled: {error}") from None

    sections = {PRIMARY: _comparison_text(analysis["primary"], exploratory=False)}
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
    if historical:
        sections[HISTORICAL] = _historical_text(saved_original)
    return sections


def render_document(sections):
    """The whole report as one markdown document, with the wording rule enforced."""
    lines = ["# Stage 2 movement-cap report (plan milestone M5)", "", PREAMBLE, ""]
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
