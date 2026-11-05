"""M5: the Stage 2 report generator, on synthetic tables and the accepted records.

One test per printing rule the handover fixes, so a rule that is dropped is a
failing test rather than a review finding. Nothing here loads a model, relaxes
anything or opens the test partition.
"""
import copy
import json

import pytest

from stage2_capped import report, rules, stage0_compare
from stage2_capped.row_schema import EXPLORATORY_LABEL, LABELS
from stage2_capped.tests import table_builder as tb
from stage2_capped.tests.table_builder import (CAPS, CONS_FAVORED, DIRECT_FAVORED,
                                               NOISE_BREAKER, build_rows, merge)

CONS_CLEAN = merge(CONS_FAVORED, NOISE_BREAKER)
DIRECT_CLEAN = merge(DIRECT_FAVORED, NOISE_BREAKER)
NO_ENERGY = {"energy0": set()}
LAMBDA1_REC = {"lam1_cons0": {"s01", "s03", "s05"}, "lam1_cons1": {"s01", "s03", "s05"}}


# ---- Helpers -------------------------------------------------------------

def sections(**kwargs):
    return report.render_report(build_rows(**kwargs))


def document(**kwargs):
    return report.render_document(sections(**kwargs))


def section_of(name, **kwargs):
    return sections(**kwargs)[name]


def cap_rows(text):
    """The cap table's data rows, as lists of cells."""
    rows = []
    for line in text.splitlines():
        if line.startswith("| c") and " of 20 " in line:
            rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


# ---- Rule 1: everything printed beside every label -----------------------

def test_rule_1_every_cap_prints_the_evidence_beside_its_label():
    text = section_of(report.PRIMARY, rec=CONS_CLEAN)
    rows = cap_rows(text)
    assert [row[0].split()[0] for row in rows] == list(CAPS)
    for row in rows:
        assert row[1] in LABELS
        assert row[2] == "20 of 20"          # interpretable items out of 20
        assert row[3] == "5" and row[4] == "6"  # connections and own minima
        assert "1 of 10 at the cap" in row[5] and "pass" in row[5]
        assert "energy_s0 6/6" in row[6] and "consistency_s1 6/6" in row[6]
        assert row[7] == "yes"               # in E
    assert "set E: c0p05, c0p10, c0p25, c1p00" in text
    assert "not evaluable: none" in text
    assert "interpretable items in the sweep: 20 over 5 connections and 6 own minima" in text


def test_rule_1_the_cap_table_names_the_cap_in_angstrom():
    text = section_of(report.PRIMARY, rec=CONS_CLEAN)
    assert "c0p05 (0.05 A)" in text and "c1p00 (1.00 A)" in text


# ---- Rule 2: the favoured arm, always, including direct ------------------

def test_rule_2_a_stated_direction_prints_the_favoured_arm():
    assert "the favoured arm is consistency" in section_of(report.PRIMARY, rec=CONS_CLEAN)


def test_rule_2_the_favoured_arm_may_be_direct():
    text = section_of(report.PRIMARY, rec=DIRECT_CLEAN)
    assert "the favoured arm is direct" in text
    assert "the favoured arm is consistency" not in text


def test_rule_2_a_single_cap_reads_favors_at_the_cap_and_the_sweep_favours_is_null():
    per_cap = {cap: {**CONS_CLEAN, **NO_ENERGY} for cap in CAPS if cap != "c0p25"}
    text = section_of(report.PRIMARY, rec=CONS_CLEAN, per_cap=per_cap)
    assert "Sweep label: single-cap observation" in text
    assert "the favoured arm at the cap is consistency" in text
    assert "R-direction holds at the cap: yes" in text
    # and the sweep's own `favors` really is null for this shape
    analysis = rules.analyze(build_rows(rec=CONS_CLEAN, per_cap=per_cap))
    assert analysis["primary"]["sweep_record"]["detail"]["favors"] is None


# ---- Rule 3: not-evaluable caps and their notes beside a direction -------

def test_rule_3_a_stated_direction_prints_the_not_evaluable_caps_and_notes():
    per_cap = {"c0p05": {**CONS_CLEAN, **NO_ENERGY}}
    text = section_of(report.PRIMARY, rec=CONS_CLEAN, per_cap=per_cap)
    assert "Sweep label: direction stated" in text
    assert "at c0p05, uninformative (gate)" in text
    assert "c0p05" not in text.split("set E: ")[1].split("\n")[0]


# ---- Rule 4: zero contrast -----------------------------------------------

def test_rule_4_zero_contrast_prints_the_fixed_sentence_and_no_null_claim():
    text = section_of(report.PRIMARY, rec={})
    assert "at c0p05, neither arm recovered any interpretable original-start item (0 of 20)" in text
    assert "the arms do not differ" not in text
    assert "uninformative (floor or gate)" in text  # the label, not a null


# ---- Rule 5: placebo -----------------------------------------------------

def test_rule_5_a_placebo_cap_says_not_distinguishable_from_seed_noise():
    text = section_of(report.PRIMARY, rec={"direct1": {"s01", "s03", "s05"}})
    assert "not distinguishable from seed noise" in text
    assert "no direction (informative)" not in text


# ---- Rule 6: the three caveats ------------------------------------------

def test_rule_6_all_three_caveats_are_printed():
    text = section_of(report.PRIMARY, rec=CONS_CLEAN)
    for caveat in rules.CAVEATS:
        assert caveat in text


# ---- Rule 7: attribution ------------------------------------------------

def test_rule_7_attribution_prints_base_status_and_lambda1_inclusion():
    text = section_of(report.ATTRIBUTION, rec=CONS_CLEAN)
    assert "Status: supported" in text
    assert "lambda = 1 rows included: no" in text


def test_rule_7_non_qualifying_flips_are_listed_with_the_failed_clause():
    # No control run has an accepted step above 1 A, so the c1p00 flips fail that clause.
    text = section_of(report.ATTRIBUTION, rec=CONS_CLEAN, control_max_disp=0.5)
    assert "Non-qualifying flips" in text
    assert "the c = infinity run has no accepted step above 1.0 A" in text
    assert "Status: not excluded" in text


def test_rule_7_the_suffixed_status_is_never_printed_as_the_status():
    text = section_of(report.ATTRIBUTION, rec={**CONS_CLEAN, **LAMBDA1_REC},
                      names=tb.PRIMARY + tb.LAMBDA1)
    lines = [line for line in text.splitlines() if line.startswith("Status:")]
    assert len(lines) == 1
    assert EXPLORATORY_LABEL not in lines[0]


# ---- Rule 8: the lambda = 1 label in every table and caption -------------

def test_rule_8_every_exploratory_table_and_caption_carries_the_exact_text():
    text = section_of(report.EXPLORATORY, rec={**CONS_CLEAN, **LAMBDA1_REC},
                      names=tb.PRIMARY + tb.LAMBDA1)
    assert text.startswith(f"Comparison label: {EXPLORATORY_LABEL}")
    captions = [line for line in text.splitlines() if line.startswith("Table:")]
    assert captions
    for caption in captions:
        assert EXPLORATORY_LABEL in caption


def test_rule_8_the_exploratory_per_run_table_carries_the_label_too():
    text = section_of(report.PER_RUN_EXPLORATORY, rec={**CONS_CLEAN, **LAMBDA1_REC},
                      names=tb.PRIMARY + tb.LAMBDA1)
    captions = [line for line in text.splitlines() if line.startswith("Table:")]
    assert captions and all(EXPLORATORY_LABEL in c for c in captions)


def test_rule_8_a_table_without_lambda_one_rows_says_so_instead_of_inventing_one():
    text = section_of(report.EXPLORATORY, rec=CONS_CLEAN)
    assert text.startswith("not labelled")


# ---- Rule 9: wording ----------------------------------------------------

def test_rule_9_the_document_never_says_effect_or_significant():
    for kwargs in ({"rec": CONS_CLEAN}, {"rec": DIRECT_CLEAN}, {"rec": {}},
                   {"rec": CONS_CLEAN, "control_max_disp": 0.5},
                   {"rec": {**CONS_CLEAN, **LAMBDA1_REC}, "names": tb.PRIMARY + tb.LAMBDA1}):
        text = document(**kwargs)
        stripped = text.replace(rules.NOT_ATTRIBUTABLE, "")
        assert "effect" not in stripped.lower()
        assert "significant" not in stripped.lower()


def test_rule_9_the_wording_guard_refuses_a_banned_word():
    bad = {"report status": "this cap shows an effect"}
    with pytest.raises(report.ReportError):
        report.render_document(bad)


def test_rule_9_the_unit_is_five_connections_and_the_caveat_says_so():
    text = document(rec=CONS_CLEAN)
    assert "five connections, not ten independent starts" in text


# ---- Rule 10: per-run evidence ------------------------------------------

def test_rule_10_per_run_table_prints_travel_rejections_and_capped_fraction():
    text = section_of(report.PER_RUN_PRIMARY, rec=CONS_CLEAN)
    assert "accepted travel" in text and "attempted travel" in text
    assert "rejected trials" in text and "capped steps" in text
    assert "30 of 70 (0.429)" in text      # the fraction of capped steps
    assert "| 2 |" in text                 # the first capped attempt


def test_rule_10_controls_print_the_step_fractions_and_the_binding_flag():
    text = section_of(report.PER_RUN_PRIMARY, rec=CONS_CLEAN)
    control_line = next(line for line in text.splitlines()
                        if line.startswith("| cinf | energy0 "))
    assert "0.200" in control_line and "0.100" in control_line
    assert "no" not in control_line.split("|")[-2]  # cinf rows carry no binding claim


def test_rule_10_says_c1p00_duplicates_the_control_when_it_never_binds():
    text = section_of(report.PER_RUN_PRIMARY, rec=CONS_CLEAN, control_max_disp=0.5,
                      trials={(cap, name, start): 0
                              for cap in CAPS for name in tb.PRIMARY for start in tb.START_IDS})
    assert "No run binds at c = 1.00 A, so c1p00 duplicates the control." in text


# ---- Determinism, immutability, refusal ---------------------------------

def test_two_renders_are_byte_identical():
    first = document(rec=CONS_CLEAN)
    second = document(rec=CONS_CLEAN)
    assert first == second


def test_the_writer_produces_identical_bytes_and_refuses_to_overwrite(tmp_path):
    rendered = sections(rec=CONS_CLEAN)
    first = report.write_report(rendered, tmp_path / "a")
    second = report.write_report(sections(rec=CONS_CLEAN), tmp_path / "b")
    assert first.read_bytes() == second.read_bytes()
    with pytest.raises(FileExistsError):
        report.write_report(rendered, tmp_path / "a")


def test_reading_rows_never_mutates_them():
    rows = build_rows(rec=CONS_CLEAN)
    before = copy.deepcopy(rows)
    report.render_document(report.render_report(rows))
    assert rows == before


def test_an_empty_table_is_refused():
    with pytest.raises(report.ReportError):
        report.render_report([])


def test_an_incomplete_table_is_labelled_incomplete_not_reported_in_part():
    rows = [row for row in build_rows(rec=CONS_CLEAN)
            if not (row["cap_label"] == "c0p10" and row["checkpoint_id"] == "direct0")]
    rendered = report.render_report(rows)
    assert list(rendered) == [report.STATUS]
    assert rendered[report.STATUS].startswith("incomplete, not labelled")
    assert "missing cells" in rendered[report.STATUS]
    assert "c0p10" in rendered[report.STATUS]


def test_a_lambda_one_only_table_cannot_be_labelled():
    rows = [row for row in build_rows(rec={**CONS_CLEAN, **LAMBDA1_REC},
                                      names=tb.PRIMARY + tb.LAMBDA1)
            if row["consistency_lambda"] == 1.0]
    rendered = report.render_report(rows)
    assert list(rendered) == [report.STATUS]
    assert rendered[report.STATUS].startswith("incomplete, not labelled")


def test_an_incomplete_lambda_one_block_does_not_cost_the_primary_labels():
    rows = [row for row in build_rows(rec={**CONS_CLEAN, **LAMBDA1_REC},
                                      names=tb.PRIMARY + tb.LAMBDA1)
            if not (row["consistency_lambda"] == 1.0 and row["cap_label"] == "c0p10")]
    rendered = report.render_report(rows)
    assert rendered[report.PRIMARY].count("direction stated") >= 4
    assert rendered[report.EXPLORATORY].startswith("incomplete, not labelled")
    assert rendered[report.EXPLORATORY].endswith(f"({EXPLORATORY_LABEL})")


# ---- The historical table ----------------------------------------------

def test_the_historical_table_reproduces_fifty_rows_at_zero_of_ten():
    try:
        saved = stage0_compare.load_saved_original()
    except (FileNotFoundError, stage0_compare.SavedResultError) as error:
        pytest.skip(f"accepted records are absent: {error}")
    text = report.render_report(build_rows(rec=CONS_CLEAN), saved_original=saved)[
        report.HISTORICAL]
    assert len(saved) == 50
    rows = [line for line in text.splitlines() if line.startswith("| ") and "---" not in line][1:]
    assert len(rows) == 5
    for row in rows:
        cells = [cell.strip() for cell in row.strip("|").split("|")]
        assert cells[1] == "0" and cells[2] == "10"
    assert "Total: 0 of 50" in text


def test_absent_accepted_records_are_reported_in_words(monkeypatch):
    def missing(root=None):
        raise FileNotFoundError("no LAUNCH_RECORDS.json")
    monkeypatch.setattr(stage0_compare, "load_saved_original", missing)
    text = report.render_report(build_rows(rec=CONS_CLEAN))[report.HISTORICAL]
    assert "not present in this checkout" in text
    assert "0 of 50" not in text
