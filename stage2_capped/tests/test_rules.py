"""M2: decision rules (spec v4 section 8) on synthetic outcome tables."""
import copy
import json
import random
from types import SimpleNamespace

import pytest

from stage2_capped import rules
from stage2_capped.row_schema import EXPLORATORY_LABEL, LABELS
from stage2_capped.rules import (DIRECTION_STATED, NO_DIRECTION, SINGLE_CAP,
                                 UNINFORMATIVE, Table, TableError)
from stage2_capped.tests import table_builder as tb
from stage2_capped.tests.table_builder import (CAPS, CONS_FAVORED, DIRECT_FAVORED,
                                               NOISE_BREAKER, build_rows, merge)

CONS_CLEAN = merge(CONS_FAVORED, NOISE_BREAKER)
DIRECT_CLEAN = merge(DIRECT_FAVORED, NOISE_BREAKER)
NO_ENERGY = {"energy0": set()}


def run(**kwargs):
    return rules.evaluate_comparison(Table(build_rows(**kwargs)))


def cap_record(result, cap):
    return next(r for r in result["cap_records"] if r["cap_label"] == cap)


def only_cap(cap, **kwargs):
    """Gate-fail every cap but one (energy C recovers nothing there)."""
    per_cap = {c: {**kwargs.get("rec", {}), **NO_ENERGY} for c in CAPS if c != cap}
    return per_cap


# ---- Baseline: a clean consistency-favoring table ------------------------

def test_clean_table_states_a_direction_for_consistency():
    result = run(rec=CONS_CLEAN)
    sweep = result["sweep_record"]
    assert sweep["label"] == DIRECTION_STATED
    assert sweep["set_E"] == list(CAPS) and sweep["not_evaluable"] == []
    assert sweep["detail"]["favors"] == "consistency"
    assert sweep["interpretable_items"] == 20
    for cap in CAPS:
        record = cap_record(result, cap)
        assert record["label"] == DIRECTION_STATED
        assert record["interpretable_items"] == 20
        assert record["connections_covered"] == 5 and record["minima_covered"] == 6
        assert record["adequacy_passed"] is True
        assert record["reference_start_counts"]["direct_s0"] == 6


def test_the_rule_is_two_sided_a_stated_direction_may_favor_direct():
    result = run(rec=DIRECT_CLEAN)
    assert result["sweep_record"]["label"] == DIRECTION_STATED
    assert result["sweep_record"]["detail"]["favors"] == "direct"


def test_a_table_with_no_difference_gets_no_direction_informative():
    result = run(rec=NOISE_BREAKER)
    assert result["sweep_record"]["label"] == NO_DIRECTION
    assert result["sweep_record"]["set_E"] == list(CAPS)


# ---- R-adequacy (gate) ---------------------------------------------------

def test_gate_failure_is_uninformative_not_a_null():
    result = run(rec=CONS_CLEAN, per_cap={"c0p05": {**CONS_CLEAN, **NO_ENERGY}})
    record = cap_record(result, "c0p05")
    assert record["label"] == UNINFORMATIVE
    assert record["adequacy_passed"] is False
    assert record["detail"]["status"] == "gate"
    assert "c0p05" not in result["sweep_record"]["set_E"]
    assert "c0p05" not in result["sweep_record"]["not_evaluable"]  # adequacy failed


def test_adequacy_needs_a_recovery_that_exceeds_the_control():
    # Control recovers s01; the capped baseline also recovers only s01: not exceeded.
    result = run(rec=CONS_CLEAN, control={"energy0": {"s01"}},
                 per_cap={"c0p10": {**CONS_CLEAN, "energy0": {"s01"}}},
                 trials={("c0p10", "energy0", "s01"): 0})
    assert cap_record(result, "c0p10")["adequacy_passed"] is False
    # Two recoveries against one control recovery passes.
    result = run(rec=CONS_CLEAN, control={"energy0": {"s01"}},
                 per_cap={"c0p10": {**CONS_CLEAN, "energy0": {"s01", "s02"}}},
                 trials={("c0p10", "energy0", "s01"): 0})
    adequacy = cap_record(result, "c0p10")["detail"]["adequacy"]
    assert adequacy["passed"] and adequacy["recovered_at_cap"] == 2
    assert adequacy["recovered_at_control"] == 1


def test_adequacy_counts_the_control_from_the_rows_not_a_constant():
    result = run(rec=CONS_CLEAN, control={"energy0": {"s02", "s03"}})
    # The cap recovers only s01 (1 <= 2 control recoveries): gate fails everywhere.
    assert all(not cap_record(result, c)["adequacy_passed"] for c in CAPS)
    assert result["sweep_record"]["label"] == UNINFORMATIVE


def test_all_caps_gated_is_r_failure_uninformative():
    result = run(rec={**CONS_CLEAN, **NO_ENERGY})
    assert result["sweep_record"]["label"] == UNINFORMATIVE
    assert result["sweep_record"]["detail"]["r_failure"] is True
    assert result["sweep_record"]["set_E"] == []


# ---- R-floor and R-support -----------------------------------------------

def test_floor_items_are_left_out_and_listed():
    ref = {"direct0": {2, 3, 6, 7, 8, 9}}  # direct seed 0 cannot reach minimum 1
    result = run(rec=CONS_CLEAN, ref=ref)
    record = cap_record(result, "c0p05")
    assert record["interpretable_items"] == 17
    floors = record["detail"]["floor_items"]
    assert sorted((f["start_id"], f["pair"]) for f in floors) == [
        ("s01", 0), ("s03", 0), ("s05", 0)]
    assert all(f["direct_interpretable"] is False and f["consistency_interpretable"]
               for f in floors)


def test_an_item_needs_both_arms_of_its_seed_pair_interpretable():
    ref = {"cons1": {1, 3, 6, 7, 9}}  # consistency seed 1 cannot reach minimum 8
    record = cap_record(run(rec=CONS_CLEAN, ref=ref), "c0p05")
    floors = record["detail"]["floor_items"]
    assert {(f["start_id"], f["pair"]) for f in floors} == {
        ("s04", 1), ("s06", 1), ("s08", 1)}
    assert all(f["consistency_interpretable"] is False for f in floors)


def test_all_floor_is_uninformative_never_no_direction():
    result = run(rec=CONS_CLEAN, ref={name: set() for name in tb.PRIMARY})
    assert all(cap_record(result, c)["label"] == UNINFORMATIVE for c in CAPS)
    assert all(cap_record(result, c)["detail"]["status"] == "floor" for c in CAPS)
    assert result["sweep_record"]["label"] == UNINFORMATIVE
    assert sorted(result["sweep_record"]["not_evaluable"]) == sorted(CAPS)


@pytest.mark.parametrize("minima, reason", [
    ({3, 6}, "fewer than 3 connections"),      # s02 (a), s07 (d): two connections
    ({1}, "fewer than 2 own minima"),          # s01, s03, s05: three connections, one minimum
])
def test_support_failure_is_uninformative(minima, reason):
    ref = {name: set(minima) for name in tb.PRIMARY}
    result = run(rec=CONS_CLEAN, ref=ref)
    record = cap_record(result, "c0p05")
    assert record["label"] == UNINFORMATIVE and record["detail"]["status"] == "floor"
    assert any(reason in r for r in record["detail"]["support"]["reasons"])
    assert result["sweep_record"]["label"] == UNINFORMATIVE


def test_a_seed_pair_covering_one_connection_fails_support():
    # Direct seed 1 reaches only minimum 9 (s10, connection e): pair 1 has one connection.
    ref = {"direct1": {9}}
    record = cap_record(run(rec=CONS_CLEAN, ref=ref), "c0p05")
    support = record["detail"]["support"]
    assert support["pair_connections"]["pair_1"] == 1
    assert support["connections"] == 5 and support["minima"] >= 2
    assert record["label"] == UNINFORMATIVE
    assert any("pair_1" in r for r in support["reasons"])


def test_the_spec_example_three_connections_sharing_one_minimum_fails_support():
    # Minimum 1 is the own minimum of s01, s03, s05 (connections a, b, c).
    items = [{"start_id": s, "connection": c, "minimum": 1, "pair": p, "direct": 0,
              "consistency": 1} for s, c in (("s01", "a"), ("s03", "b"), ("s05", "c"))
             for p in (0, 1)]
    result = rules.support(items)
    assert not result["ok"] and result["connections"] == 3 and result["minima"] == 1


# ---- R-direction ---------------------------------------------------------

def item(conn, pair, direct, consistency, start="s", minimum=1):
    return {"start_id": f"{start}{conn}", "connection": conn, "minimum": minimum,
            "pair": pair, "direct": direct, "consistency": consistency}


CONNS = ["a", "b", "c", "d", "e"]


def test_direction_needs_three_connections_and_none_the_other_way():
    items = [item(c, p, 0, 1) for c in "abc" for p in (0, 1)]
    assert rules.direction(items, CONNS)["holds"]
    two = [item(c, p, 0, 1) for c in "ab" for p in (0, 1)]
    assert not rules.direction(two, CONNS)["holds"]
    opposed = items + [item("d", 0, 1, 0), item("d", 1, 1, 0)]
    result = rules.direction(opposed, CONNS)
    assert not result["holds"] and result["connections_favoring_direct"] == 1


def test_direction_is_two_sided_and_the_other_arm_vetoes_it_either_way():
    direct = [item(c, p, 1, 0) for c in "abc" for p in (0, 1)]
    result = rules.direction(direct, CONNS)
    assert result["holds"] and result["favors"] == "direct"
    # One connection favoring consistency blocks a direct-favoring direction.
    vetoed = direct + [item("d", 0, 0, 1), item("d", 1, 0, 1)]
    result = rules.direction(vetoed, CONNS)
    assert result["connections_favoring_direct"] == 3
    assert result["connections_favoring_consistency"] == 1
    assert not result["holds"] and result["favors"] is None


def test_net_zero_connections_favor_neither():
    items = [item("a", 0, 0, 1), item("a", 1, 1, 0)] + \
            [item(c, p, 0, 1) for c in "bc" for p in (0, 1)]
    result = rules.direction(items, CONNS)
    assert result["connection_sums"]["a"] == 0
    assert result["connections_favoring_consistency"] == 2 and not result["holds"]


def test_both_seed_pairs_need_a_nonzero_total_of_the_same_sign():
    base = [item(c, 0, 0, 2 // 2) for c in "abc"]
    # pair 1 total zero
    zero = base + [item("a", 1, 1, 1), item("b", 1, 0, 0)]
    assert rules.direction(zero, CONNS)["pair_totals"]["pair_1"] == 0
    assert not rules.direction(zero, CONNS)["holds"]
    # pair 1 total negative while three connections still sum positive
    items = [item(c, 0, 0, 1) for c in "abc"] + [item(c, 0, 0, 1, start="t") for c in "abc"] + \
            [item("a", 1, 1, 0), item("b", 1, 1, 0), item("c", 1, 1, 0)]
    result = rules.direction(items, CONNS)
    assert result["connections_favoring_consistency"] == 3
    assert result["connections_favoring_direct"] == 0
    assert result["pair_totals"] == {"pair_0": 6, "pair_1": -3}
    assert not result["holds"] and result["favors"] is None


def test_a_connection_without_items_favors_neither():
    items = [item(c, p, 0, 1) for c in "abc" for p in (0, 1)]
    assert rules.direction(items, CONNS)["connection_sums"]["e"] == 0


# ---- Placebo (seed-noise check) ------------------------------------------

def test_placebo_flags_a_cap_where_seed_contrasts_look_like_the_main_contrast():
    # Direct seed 1 recovers on connections a, b, c; direct seed 0 never does.
    rec = {"direct1": {"s01", "s03", "s05"}}
    result = run(rec=rec)
    record = cap_record(result, "c0p05")
    assert record["detail"]["status"] == "placebo"
    assert record["label"] == UNINFORMATIVE  # review F1: never an informative null
    assert "not distinguishable from seed noise" in record["detail"]["note"]
    assert record["detail"]["placebo"]["contrasts"]["direct"]["flagged"] is True
    assert result["sweep_record"]["set_E"] == []
    assert result["sweep_record"]["label"] == UNINFORMATIVE
    assert sorted(result["sweep_record"]["not_evaluable"]) == sorted(CAPS)


def test_placebo_applies_to_the_consistency_seeds_too():
    result = run(rec={"cons0": {"s02", "s04", "s06"}})
    assert cap_record(result, "c0p05")["detail"]["placebo"]["contrasts"][
        "consistency"]["flagged"] is True


def test_placebo_needs_three_connections_one_sided():
    rec = {"direct1": {"s01", "s03"}}  # two connections: not flagged
    record = cap_record(run(rec=rec), "c0p05")
    assert record["detail"]["placebo"]["flagged"] is False
    rec = {"direct1": {"s01", "s03", "s05"}, "direct0": {"s07"}}  # a, b, c vs d
    assert cap_record(run(rec=rec), "c0p05")["detail"]["placebo"]["flagged"] is False


def test_placebo_uses_only_starts_both_seeds_can_reach():
    # Direct seed 0 cannot reach minimum 1, so s01, s03, s05 drop out of its contrast.
    rec = {"direct1": {"s01", "s03", "s05"}}
    record = cap_record(run(rec=rec, ref={"direct0": {3, 6, 7, 8, 9}}), "c0p05")
    contrast = record["detail"]["placebo"]["contrasts"]["direct"]
    assert contrast["starts_used"] == 7 and contrast["flagged"] is False


def test_a_placebo_flagged_cap_never_contributes_to_a_stated_direction():
    # c0p05 shows the clean consistency contrast but direct seed 1 mimics it.
    rec = {**CONS_CLEAN, "direct0": set(), "direct1": {"s01", "s03", "s05"}}
    per_cap = {c: CONS_CLEAN for c in CAPS[1:]}
    result = run(rec=rec, per_cap={"c0p05": {**rec}, **per_cap})
    assert cap_record(result, "c0p05")["detail"]["status"] == "placebo"
    sweep = result["sweep_record"]
    assert "c0p05" not in sweep["set_E"] and "c0p05" in sweep["not_evaluable"]
    assert sweep["label"] == DIRECTION_STATED and sweep["set_E"] == list(CAPS[1:])
    # Dropping the flagged cap altogether leaves the conclusion unchanged.
    without = rules.evaluate_comparison(Table(build_rows(rec=CONS_CLEAN)), caps=CAPS[1:])
    assert without["sweep_record"]["label"] == sweep["label"]
    assert without["sweep_record"]["set_E"] == sweep["set_E"]


# ---- R-sweep on full tables ---------------------------------------------

def test_one_evaluable_cap_is_a_single_cap_observation_whether_or_not_direction_holds():
    per_cap = {c: NO_ENERGY for c in CAPS[1:]}
    holds = run(rec=CONS_CLEAN, per_cap={**per_cap, "c0p05": CONS_CLEAN})
    assert holds["sweep_record"]["label"] == SINGLE_CAP
    assert holds["sweep_record"]["detail"]["r_direction_holds_at_the_cap"] is True
    assert holds["sweep_record"]["set_E"] == ["c0p05"]
    assert holds["sweep_record"]["interpretable_items"] == 20
    quiet = run(rec=NOISE_BREAKER, per_cap={**per_cap, "c0p05": NOISE_BREAKER})
    assert quiet["sweep_record"]["label"] == SINGLE_CAP
    assert quiet["sweep_record"]["detail"]["r_direction_holds_at_the_cap"] is False
    assert cap_record(quiet, "c0p05")["label"] == SINGLE_CAP


def test_opposite_signs_across_caps_give_no_direction():
    per_cap = {"c0p05": CONS_CLEAN, "c0p10": CONS_CLEAN,
               "c0p25": DIRECT_CLEAN, "c1p00": DIRECT_CLEAN}
    result = run(rec=CONS_CLEAN, per_cap=per_cap)
    sweep = result["sweep_record"]
    assert sweep["label"] == NO_DIRECTION and len(sweep["set_E"]) == 4
    assert "opposite signs" in sweep["detail"]["reason"]


def test_direction_at_some_caps_only_is_no_direction_informative():
    per_cap = {"c0p05": CONS_CLEAN, "c0p10": NOISE_BREAKER,
               "c0p25": NOISE_BREAKER, "c1p00": NOISE_BREAKER}
    sweep = run(rec=NOISE_BREAKER, per_cap=per_cap)["sweep_record"]
    assert sweep["label"] == NO_DIRECTION
    assert "does not hold at every cap" in sweep["detail"]["reason"]


def test_adequacy_passing_caps_outside_e_are_not_evaluable():
    per_cap = {"c0p05": CONS_CLEAN, "c0p10": CONS_CLEAN, "c0p25": CONS_CLEAN,
               "c1p00": {**CONS_CLEAN, "direct0": set(),
                         "direct1": {"s01", "s03", "s05"}}}  # placebo-flagged
    result = run(rec=CONS_CLEAN, per_cap=per_cap)
    assert result["sweep_record"]["not_evaluable"] == ["c1p00"]
    assert cap_record(result, "c1p00")["adequacy_passed"] is True


# ---- R-sweep on item sets (assessments built directly) -------------------

def eligible(cap, items):
    return {"cap": cap, "status": rules.STATUS_ELIGIBLE, "items": items,
            "direction": rules.direction(items, CONNS)}


def mk(start, conn, minimum, pair, direct, consistency):
    return {"start_id": start, "connection": conn, "minimum": minimum, "pair": pair,
            "direct": direct, "consistency": consistency}


TABLE_STUB = SimpleNamespace(connections=CONNS)


def clean_items(conns="abc", minima=(1, 2, 3)):
    return [mk(f"{c}{m}", c, m, p, 0, 1) for c in conns for m in minima for p in (0, 1)]


def test_common_items_that_fail_support_are_uninformative():
    # Each cap's own items hold the direction (connections a, b, c and c, d, e); the
    # items interpretable at both caps are connection c only, which fails R-support.
    one = eligible("c0p05", clean_items("abc"))
    two = eligible("c0p10", clean_items("cde"))
    assert one["direction"]["holds"] and two["direction"]["holds"]
    label, detail, items, n = rules.sweep(TABLE_STUB, [one, two], 0.1)
    assert label == UNINFORMATIVE and n == 6
    assert detail["reason"].startswith("the items interpretable at every cap in E fail R-support")
    assert detail["common_items"]["support_ok"] is False


def test_direction_on_own_items_but_not_on_the_common_set_is_varies_with_item_set():
    # Items that make the direction hold at each cap are interpretable at one cap only;
    # the shared items are balanced and meet R-support.
    shared = [mk(f"s{c}{m}", c, m, p, 1, 1) for c in "abcd" for m in (1, 2) for p in (0, 1)]
    own_one = [mk(f"u{c}", c, 9, p, 0, 1) for c in "abc" for p in (0, 1)]
    own_two = [mk(f"v{c}", c, 8, p, 0, 1) for c in "abc" for p in (0, 1)]
    one = eligible("c0p05", shared + own_one)
    two = eligible("c0p10", shared + own_two)
    assert one["direction"]["holds"] and two["direction"]["holds"]
    label, detail, items, n = rules.sweep(TABLE_STUB, [one, two], 0.1)
    assert label == NO_DIRECTION and detail["reason"] == "varies with item set"
    assert n == len(shared) and detail["common_items"]["support_ok"] is True


def test_direction_stated_needs_the_common_set_to_hold_too():
    shared = clean_items("abcd", (1, 2))
    one = eligible("c0p05", shared + [mk("u", "e", 5, p, 0, 1) for p in (0, 1)])
    two = eligible("c0p10", shared + [mk("v", "e", 6, p, 0, 1) for p in (0, 1)])
    label, detail, _, n = rules.sweep(TABLE_STUB, [one, two], 0.1)
    assert label == DIRECTION_STATED and n == len(shared)
    assert detail["common_items"]["holds"] == {"c0p05": True, "c0p10": True}


def test_common_set_direction_must_name_the_same_arm_as_the_own_items():
    # Own items at both caps favor consistency; the shared items, which meet R-support,
    # favor direct. A direction on own items that flips on the common set is not stated.
    shared = [mk(f"s{c}{m}", c, m, p, 1, 0) for c in "abc" for m in (1, 2) for p in (0, 1)]
    own_one = [mk(f"u{c}{k}", c, 9, p, 0, 1) for c in "abc" for k in range(3) for p in (0, 1)]
    own_two = [mk(f"w{c}{k}", c, 8, p, 0, 1) for c in "abc" for k in range(3) for p in (0, 1)]
    one = eligible("c0p05", shared + own_one)
    two = eligible("c0p10", shared + own_two)
    assert one["direction"]["favors"] == "consistency" == two["direction"]["favors"]
    assert rules.direction(shared, CONNS)["favors"] == "direct"
    assert rules.support(shared)["ok"]
    label, detail, _, n = rules.sweep(TABLE_STUB, [one, two], 0.1)
    assert label == NO_DIRECTION and detail["reason"] == "varies with item set"
    assert n == len(shared)


def test_the_two_cap_condition_does_not_require_adjacent_caps():
    items = clean_items("abcd", (1, 2))
    label, _, _, _ = rules.sweep(TABLE_STUB, [eligible("c0p05", items),
                                              eligible("c1p00", items)], 0.1)
    assert label == DIRECTION_STATED


# ---- R-control -----------------------------------------------------------

def test_r_control_passes_only_on_a_complete_exact_match():
    assert rules.r_control(80, [])["verdict"] == "pass"
    assert rules.r_control(80, [])["proceed"] is True
    assert rules.r_control(79, [])["verdict"] == "stop: incomplete comparison"
    assert rules.r_control(0, [])["proceed"] is False


def test_r_control_first_mismatch_authorizes_one_frozen_rerun():
    result = rules.r_control(12, ["task-a"])
    assert result["verdict"] == "frozen re-run required" and not result["proceed"]
    assert "task-a" in result["action"]


def test_r_control_runtime_drift_and_code_error_both_stop():
    drift = rules.r_control(12, ["task-a"], {"task": "task-a", "matches_saved": False})
    assert drift["verdict"] == "stop: runtime drift" and not drift["proceed"]
    code = rules.r_control(12, ["task-a"], {"task": "task-a", "matches_saved": True})
    assert code["verdict"] == "stop: code error" and not code["proceed"]


def test_r_control_refuses_any_continuation_beyond_the_pre_authorized_rerun():
    assert rules.r_control(30, ["a", "b"])["verdict"] == "stop: unauthorized continuation"
    other = rules.r_control(12, ["a"], {"task": "b", "matches_saved": True})
    assert other["verdict"] == "stop: unauthorized continuation"
    assert not any(rules.r_control(80, m, r)["proceed"] for m, r in
                   [(["a"], None), (["a"], {"task": "a", "matches_saved": True}),
                    (["a", "b"], {"task": "a", "matches_saved": True})])


# ---- R-attribution -------------------------------------------------------

def attribution(**kwargs):
    kwargs.setdefault("energy_default", False)  # the energy baseline is not under test here
    return rules.r_attribution(Table(build_rows(**kwargs)))


def test_attribution_supported_with_two_qualifying_runs():
    rec = {"direct0": {"s01"}, "direct1": {"s02"}}
    result = attribution(per_cap={"c1p00": rec}, rec={})
    assert result["status"] == "supported" and result["qualifying_primary"] == 2
    assert result["qualifying_lambda1"] == 0 and "exploratory_label" not in result
    assert result["necessary_not_sufficient"] is True


def test_attribution_with_one_qualifying_run_is_not_excluded():
    result = attribution(per_cap={"c1p00": {"direct0": {"s01"}}}, rec={})
    assert result["status"] == "not excluded"


def test_attribution_when_the_capped_count_equals_the_control_has_nothing_to_attribute():
    same = {"direct0": {"s01"}}
    result = attribution(rec=same, control=same, energy_default=False,
                         trials={(c, "direct0", "s01"): 3 for c in CAPS})
    assert result["status"] == "no new recovery to attribute"
    assert all(n == 0 for n in result["new_recoveries_by_cap"].values())


def test_attribution_when_c1p00_never_binds_and_recoveries_appear_only_at_small_caps():
    small = {"direct0": {"s01", "s02"}}
    per_cap = {"c0p05": small, "c0p10": small, "c0p25": {}, "c1p00": {}}
    result = attribution(per_cap=per_cap, rec={},
                         trials={("c1p00", n, s): 0 for n in tb.PRIMARY for s in tb.START_IDS})
    assert result["status"] == rules.NOT_ATTRIBUTABLE
    assert result["new_recoveries_by_cap"]["c1p00"] == 0


def test_attribution_with_no_step_above_one_angstrom_is_not_supported():
    rec = {"direct0": {"s01"}, "direct1": {"s02"}}
    result = attribution(per_cap={"c1p00": rec}, rec={}, control_max_disp=0.5)
    assert result["status"] == "not excluded" and result["qualifying_primary"] == 0


def test_attribution_requires_each_recovery_run_to_have_bound():
    # A new recovery whose cap never bound cannot exist (it would equal its control),
    # so the table refuses it. The explicit check in r_attribution is a second guard:
    # forge the row after loading and the run must stop qualifying.
    rec = {"direct0": {"s01"}, "direct1": {"s02"}}
    with pytest.raises(TableError, match="never bound but differs"):
        Table(build_rows(per_cap={"c1p00": rec}, rec={}, energy_default=False,
                         trials={("c1p00", "direct1", "s02"): 0},
                         unbound_follows_control=False))
    table = Table(build_rows(per_cap={"c1p00": rec}, rec={}, energy_default=False))
    assert rules.r_attribution(table)["status"] == "supported"
    key = ("c1p00", ("direct", 1, None), "original", "s02")
    table.by_key[key] = dict(table.by_key[key], capped_trial_count=0,
                             first_capped_attempt=None, capped_fraction=0.0)
    forged = rules.r_attribution(table)
    assert forged["qualifying_primary"] == 1 and forged["status"] == "not excluded"


def test_attribution_counts_lambda_one_runs_but_labels_them_exploratory():
    per_cap = {"c1p00": {"lam1_cons0": {"s01"}, "lam1_cons1": {"s02"}}}
    result = attribution(per_cap=per_cap, rec={}, names=tb.PRIMARY + tb.LAMBDA1)
    assert result["base_status"] == "supported"
    assert result["status"].startswith("supported (") and EXPLORATORY_LABEL in result["status"]
    assert result["qualifying_lambda1"] == 2 and result["qualifying_primary"] == 0
    assert result["exploratory_label"] == EXPLORATORY_LABEL
    assert result["supported_by_primary_runs_alone"] is False
    mixed = attribution(per_cap={"c1p00": {"direct0": {"s01"}, "direct1": {"s02"},
                                           "lam1_cons0": {"s03"}}}, rec={},
                        names=tb.PRIMARY + tb.LAMBDA1)
    assert mixed["supported_by_primary_runs_alone"] is True


# ---- Table integrity -----------------------------------------------------

def test_duplicate_and_missing_rows_are_refused():
    rows = build_rows(rec=CONS_CLEAN)
    with pytest.raises(TableError, match="duplicate"):
        Table(rows + [rows[0]])
    missing = [r for r in rows if not (r["cap_label"] == "c0p10" and
                                       r["checkpoint_id"] == "direct1" and
                                       r["start_id"] == "s03")]
    with pytest.raises(TableError, match="incomplete table"):
        rules.evaluate_comparison(Table(missing))


def test_a_row_that_breaks_the_schema_is_refused():
    rows = build_rows(rec=CONS_CLEAN)
    rows[0] = dict(rows[0], recovered=not rows[0]["recovered"])
    with pytest.raises(TableError, match="schema"):
        Table(rows)


def test_a_table_of_the_wrong_shape_is_refused():
    rows = [r for r in build_rows(rec=CONS_CLEAN) if r["start_id"] != "s10"]
    with pytest.raises(TableError, match="10 original starts"):
        rules.evaluate_comparison(Table(rows))
    rows = [r for r in build_rows(rec=CONS_CLEAN) if r["start_id"] != "r9"]
    with pytest.raises(TableError, match="6 reference starts"):
        rules.evaluate_comparison(Table(rows))


def test_an_unbound_run_that_differs_from_its_control_is_a_harness_failure():
    with pytest.raises(TableError, match="never bound but differs"):
        Table(build_rows(rec={"direct0": {"s01"}}, trials={("c0p05", "direct0", "s01"): 0},
                         unbound_follows_control=False))


# ---- Lambda = 1, wording and output hygiene ------------------------------

def test_lambda_one_is_reported_the_same_way_and_labelled_exploratory():
    rows = build_rows(rec=merge(CONS_CLEAN, {"lam1_cons0": {"s01", "s03", "s05"},
                                              "lam1_cons1": {"s01", "s03", "s05"}}),
                      names=tb.PRIMARY + tb.LAMBDA1)
    out = rules.analyze(rows)
    assert set(out) == {"primary", "exploratory", "attribution"}
    for record in out["exploratory"]["cap_records"] + [out["exploratory"]["sweep_record"]]:
        assert record["detail"]["exploratory_label"] == EXPLORATORY_LABEL
    for record in out["primary"]["cap_records"] + [out["primary"]["sweep_record"]]:
        assert "exploratory_label" not in record["detail"]


def test_primary_comparison_ignores_lambda_one_rows():
    rows = build_rows(rec=CONS_CLEAN, names=tb.PRIMARY + tb.LAMBDA1)
    without = build_rows(rec=CONS_CLEAN)
    a = rules.analyze(without)["primary"]["sweep_record"]
    b = rules.analyze(rows)["primary"]["sweep_record"]
    assert a == b


def test_no_record_uses_effect_or_significant():
    for rec in (CONS_CLEAN, DIRECT_CLEAN, NOISE_BREAKER, {"direct1": {"s01", "s03", "s05"}}):
        out = rules.analyze(build_rows(rec=rec))
        text = json.dumps([out["primary"]["cap_records"], out["primary"]["sweep_record"]]).lower()
        assert "effect" not in text and "significan" not in text


def test_records_are_json_without_inf_or_nan():
    out = rules.analyze(build_rows(rec=CONS_CLEAN))
    for record in out["primary"]["cap_records"] + [out["primary"]["sweep_record"]]:
        json.dumps(record, allow_nan=False)


def test_analysis_is_deterministic_and_does_not_change_its_input():
    rows = build_rows(rec=CONS_CLEAN)
    before = copy.deepcopy(rows)
    first = rules.analyze(rows)
    second = rules.analyze(rows)
    assert rows == before
    assert json.dumps(first["primary"]["sweep_record"], sort_keys=True) == \
        json.dumps(second["primary"]["sweep_record"], sort_keys=True)


# ---- Seeded property test ------------------------------------------------

def random_rows(rng):
    """A random but schema-valid table. Modes mix noise, a favored arm, floors,
    gate failures and seed-noise structure."""
    def subset(prob):
        return {s for s in tb.START_IDS if rng.random() < prob}

    mode = rng.choice(["noise", "cons", "direct", "mixed"])
    per_cap, per_cap_ref = {}, {}
    for cap in CAPS:
        base = rng.choice([0.0, 0.15, 0.5])
        recs = {}
        for name in tb.PRIMARY + tb.LAMBDA1:
            recs[name] = subset(base)
        favored = subset(rng.choice([0.3, 0.6, 0.9]))
        if mode in ("cons", "mixed") and rng.random() < 0.8:
            recs["cons0"] |= favored
            recs["cons1"] |= favored if rng.random() < 0.9 else subset(0.5)
            recs["lam1_cons0"] |= favored
            recs["lam1_cons1"] |= favored
        if mode in ("direct", "mixed") and rng.random() < (0.8 if mode == "direct" else 0.4):
            recs["direct0"] |= favored
            recs["direct1"] |= favored
        recs["energy0"] = {"s01"} if rng.random() < 0.75 else set()
        per_cap[cap] = recs
        per_cap_ref[cap] = {name: {m for m in tb.MINIMA if rng.random() < rng.choice([0.3, 0.8, 1.0])}
                            for name in tb.PRIMARY + tb.LAMBDA1}
    control = {"energy0": subset(0.1), "direct0": subset(0.1)}
    # Some runs never bind (and then equal their control), exercising that branch.
    trials = {(cap, name, s): 0 for cap in CAPS for name in tb.PRIMARY + tb.LAMBDA1
              for s in tb.START_IDS if rng.random() < 0.03}
    return build_rows(per_cap=per_cap, per_cap_ref=per_cap_ref, control=control,
                      names=tb.PRIMARY + tb.LAMBDA1, trials=trials,
                      control_max_disp=rng.choice([0.5, 2.0]))


def check_invariants(result):
    records = result["cap_records"] + [result["sweep_record"]]
    for record in records:
        assert record["label"] in LABELS  # exactly one of the four spec labels
    sweep = result["sweep_record"]
    E = sweep["set_E"]
    by_cap = {r["cap_label"]: r for r in result["cap_records"]}
    for cap, record in by_cap.items():
        status = record["detail"]["status"]
        if status in ("gate", "floor", "zero_contrast"):
            assert record["label"] == UNINFORMATIVE  # never "no direction (informative)"
            assert cap not in E
        if status == "zero_contrast":
            assert record["detail"]["direction"]["zero_contrast"] is True
            assert cap in sweep["not_evaluable"]
        if status == "eligible":
            assert record["detail"]["direction"]["zero_contrast"] is False
        if status == "placebo":
            assert record["label"] == UNINFORMATIVE and cap not in E
            assert cap in sweep["not_evaluable"]
        if cap in E:
            assert status == "eligible" and record["adequacy_passed"]
    assert set(sweep["not_evaluable"]).isdisjoint(E)
    for cap in sweep["not_evaluable"]:
        assert by_cap[cap]["adequacy_passed"] is True
        assert by_cap[cap]["label"] != NO_DIRECTION  # "not evaluable" is never "informative"
    if sweep["label"] == NO_DIRECTION:
        assert E  # never produced by floor- or gate-driven cases
    if sweep["label"] == SINGLE_CAP:
        assert len(E) == 1
    if sweep["label"] == DIRECTION_STATED:
        assert len(E) >= 2 and sweep["detail"]["favors"] in ("consistency", "direct")
    if sweep["label"] == UNINFORMATIVE:
        assert not E or sweep["detail"]["common_items"]["support_ok"] is False
    for record in records:
        json.dumps(record, allow_nan=False)


def test_property_every_table_gets_exactly_one_label_and_the_rules_hold():
    rng = random.Random(20261102)
    seen = {"sweep": set(), "status": set()}
    for _ in range(400):
        rows = random_rows(rng)
        table = Table(rows)
        for lam in (rules.PRIMARY_LAMBDA, rules.EXPLORATORY_LAMBDA):
            result = rules.evaluate_comparison(table, lam)
            check_invariants(result)
            seen["sweep"].add(result["sweep_record"]["label"])
            seen["status"].update(r["detail"]["status"] for r in result["cap_records"])
            # A placebo-flagged or ineligible cap never changes the conclusion: the
            # sweep over E alone gives the same label and the same E.
            E = result["sweep_record"]["set_E"]
            if E:
                alone = rules.evaluate_comparison(table, lam, caps=tuple(E))["sweep_record"]
                assert alone["label"] == result["sweep_record"]["label"]
                assert alone["set_E"] == E
        attribution_result = rules.analyze(rows)["attribution"]
        assert attribution_result["base_status"] in rules.ATTRIBUTION_STATUSES
        if attribution_result["base_status"] == "supported":
            assert len(attribution_result["qualifying_runs"]) >= 2
        assert attribution_result["lambda1_included"] is True
        again = rules.evaluate_comparison(table, rules.PRIMARY_LAMBDA)
        assert json.dumps(again["sweep_record"], sort_keys=True) == json.dumps(
            rules.evaluate_comparison(Table(rows), rules.PRIMARY_LAMBDA)["sweep_record"],
            sort_keys=True)
    assert seen["sweep"] == set(LABELS), seen["sweep"]  # the generator reaches every label
    assert seen["status"] == {"gate", "floor", "zero_contrast", "placebo", "eligible"}


def test_property_counterfactual_removing_a_placebo_flagged_cap_changes_nothing():
    rng = random.Random(7)
    checked = 0
    for _ in range(900):
        table = Table(random_rows(rng))
        full = rules.evaluate_comparison(table)
        flagged = [r["cap_label"] for r in full["cap_records"]
                   if r["detail"]["status"] == "placebo"]
        if not flagged:
            continue
        kept = tuple(c for c in CAPS if c not in flagged)
        reduced = rules.evaluate_comparison(table, caps=kept)
        assert reduced["sweep_record"]["label"] == full["sweep_record"]["label"]
        assert reduced["sweep_record"]["set_E"] == full["sweep_record"]["set_E"]
        checked += 1
    assert checked >= 20


# ---- Scientific review, round 1 (SCIENTIFIC_SIGNOFF_rules_py.md) ----------

FLAGGED = {**CONS_CLEAN, "direct0": set(), "direct1": {"s01", "s03", "s05"}}


def test_f1_placebo_flagged_cap_records_agree_with_the_sweep():
    result = run(rec=FLAGGED)
    assert all(cap_record(result, c)["label"] == UNINFORMATIVE for c in CAPS)
    assert all(cap_record(result, c)["detail"]["status"] == "placebo" for c in CAPS)
    sweep = result["sweep_record"]
    assert sweep["label"] == UNINFORMATIVE and sweep["detail"]["r_failure"] is True
    assert sorted(sweep["not_evaluable"]) == sorted(CAPS)


def test_f2_zero_contrast_prints_the_recovery_totals():
    result = run(rec={})
    for cap in CAPS:
        d = cap_record(result, cap)["detail"]["direction"]
        assert d["zero_contrast"] is True
        assert d["recoveries_direct"] == 0 == d["recoveries_consistency"]
    detail = result["sweep_record"]["detail"]
    assert detail["zero_contrast_caps"] == list(CAPS)
    assert detail["per_cap"]["c0p05"]["recoveries_consistency"] == 0
    # Author's decision on review S8 (2 Nov): a 0 = 0 cap is uninformative, and it
    # leaves E (listed not evaluable), so the sweep has no evaluable cap.
    for cap in CAPS:
        record = cap_record(result, cap)
        assert record["label"] == UNINFORMATIVE and record["detail"]["status"] == "zero_contrast"
        assert "0 of 20" in record["detail"]["note"]
        assert "not a finding that the arms are alike" in record["detail"]["note"]
    assert result["sweep_record"]["label"] == UNINFORMATIVE
    assert result["sweep_record"]["set_E"] == []
    assert sorted(result["sweep_record"]["not_evaluable"]) == sorted(CAPS)
    clean = run(rec=CONS_FAVORED)
    assert clean["sweep_record"]["detail"]["zero_contrast_caps"] == []
    d = cap_record(clean, "c0p05")["detail"]["direction"]
    assert (d["recoveries_direct"], d["recoveries_consistency"]) == (0, 6)


def test_f3_the_grid_is_400_rows_primary_and_560_with_lambda_one():
    assert len(build_rows()) == 400
    assert len(build_rows(names=tb.PRIMARY + tb.LAMBDA1)) == 560


def test_f3_a_missing_control_row_is_refused_not_silently_dropped():
    rec = {"direct0": {"s01"}, "direct1": {"s02"}}
    rows = build_rows(per_cap={"c1p00": rec}, rec={}, energy_default=False)
    assert rules.r_attribution(Table(rows))["status"] == "supported"
    fewer = [r for r in rows if not (r["cap_label"] == "cinf" and
                                     r["checkpoint_id"] == "direct1" and r["start_id"] == "s02")]
    with pytest.raises(TableError, match="incomplete table"):
        rules.r_attribution(Table(fewer))
    missing_capped = [r for r in rows if not (r["cap_label"] == "c1p00" and
                                              r["checkpoint_id"] == "direct1" and
                                              r["start_id"] == "s02")]
    with pytest.raises(TableError, match="incomplete table"):
        rules.r_attribution(Table(missing_capped))


def test_f3_a_start_whose_minimum_has_no_reference_start_is_refused():
    rows = [dict(r, own_minimum=5) if r["start_id"] == "r9" else r
            for r in build_rows(rec=CONS_CLEAN)]
    with pytest.raises(TableError, match="no reference start"):
        rules.evaluate_comparison(Table(rows))
    # R-attribution never looks up reference rows, so the up-front shape check is
    # the only thing that refuses this table there.
    with pytest.raises(TableError, match="has own minimum 9 with no reference start"):
        rules.r_attribution(Table(rows))


def test_f4_an_incomplete_lambda_one_block_does_not_cost_the_primary_labels():
    rows = [r for r in build_rows(rec=CONS_CLEAN, names=tb.PRIMARY + tb.LAMBDA1)
            if not (r["checkpoint_id"] in tb.LAMBDA1 and r["cap_label"] != "cinf")]
    out = rules.analyze(rows)
    assert out["primary"]["sweep_record"]["label"] == DIRECTION_STATED
    assert out["exploratory"]["status"] == "incomplete, not labelled"
    assert "incomplete table" in out["exploratory"]["reason"]
    assert out["exploratory"]["exploratory_label"] == EXPLORATORY_LABEL
    assert out["attribution"]["lambda1_included"] is False


def test_f4_an_incomplete_primary_table_still_raises():
    rows = [r for r in build_rows(rec=CONS_CLEAN, names=tb.PRIMARY + tb.LAMBDA1)
            if not (r["checkpoint_id"] == "direct1" and r["cap_label"] == "c0p05"
                    and r["start_id"] == "s01")]
    with pytest.raises(TableError, match="incomplete table"):
        rules.analyze(rows)


def test_f5_lambda_one_only_recoveries_label_every_attribution_status():
    names = tb.PRIMARY + tb.LAMBDA1
    small = attribution(per_cap={"c0p05": {"lam1_cons0": {"s01"}}}, rec={}, names=names)
    assert small["base_status"] == rules.NOT_ATTRIBUTABLE
    assert small["status"].startswith(rules.NOT_ATTRIBUTABLE) and \
        EXPLORATORY_LABEL in small["status"]
    one = attribution(per_cap={"c1p00": {"lam1_cons0": {"s01"}}}, rec={}, names=names)
    assert one["base_status"] == "not excluded" and EXPLORATORY_LABEL in one["status"]
    primary_only = attribution(per_cap={"c1p00": {"direct0": {"s01"}}}, rec={}, names=names)
    assert primary_only["status"] == primary_only["base_status"] == "not excluded"
    assert "exploratory_label" not in primary_only


def test_f7_non_qualifying_flips_are_listed_with_the_failed_clause():
    rec = {"direct0": {"s01"}, "direct1": {"s02"}, "cons0": {"s03"}}
    rows = build_rows(per_cap={"c1p00": rec}, rec={}, energy_default=False)
    rows = [dict(r, max_accepted_displacement_A=0.5, accepted_step_fraction_gt_1p0=0.0,
                 would_bind_at_1p0=False)
            if (r["cap_label"] == "cinf" and r["checkpoint_id"] == "cons0" and
                r["start_id"] == "s03") else r for r in rows]
    result = rules.r_attribution(Table(rows))
    assert result["status"] == "supported" and len(result["qualifying_runs"]) == 2
    assert result["new_recoveries_by_cap"]["c1p00"] == 3
    (flip,) = result["non_qualifying_flips"]
    assert flip["checkpoint_id"] == "cons0" and flip["start_id"] == "s03"
    assert flip["failed"] == ["the c = infinity run has no accepted step above 1.0 A"]


def test_the_one_angstrom_boundary_is_strict():
    rec = {"direct0": {"s01"}, "direct1": {"s02"}}
    at = attribution(per_cap={"c1p00": rec}, rec={}, control_max_disp=1.0)
    assert at["qualifying_primary"] == 0 and at["status"] == "not excluded"
    above = attribution(per_cap={"c1p00": rec}, rec={}, control_max_disp=1.0000001)
    assert above["qualifying_primary"] == 2 and above["status"] == "supported"


def test_f9_the_sweep_record_carries_the_unit_of_analysis_caveats():
    caveats = " ".join(run(rec=CONS_CLEAN)["sweep_record"]["detail"]["caveats"])
    assert "five connections" in caveats and "not replication" in caveats
    assert "0.05 and 0.10 bind on the same first trials" in caveats


def test_f12_records_name_the_comparison_and_the_favored_arm():
    direct = run(rec=DIRECT_CLEAN)
    assert direct["sweep_record"]["detail"]["favors"] == "direct"
    assert direct["sweep_record"]["detail"]["comparison_lambda"] == 0.1
    assert cap_record(direct, "c0p05")["detail"]["favors_own_items"] == "direct"
    quiet = run(rec=NOISE_BREAKER)
    assert quiet["sweep_record"]["detail"]["favors"] is None
    # lambda = 1 mirrors the primary table, noise breaker included (direct arms carry it).
    rows = build_rows(rec=merge(CONS_CLEAN, {"lam1_cons0": {"s01", "s03", "s05", "s07"},
                                              "lam1_cons1": {"s01", "s03", "s05", "s09"}}),
                      names=tb.PRIMARY + tb.LAMBDA1)
    out = rules.analyze(rows)
    assert out["exploratory"]["sweep_record"]["detail"]["comparison_lambda"] == 1.0
    assert out["exploratory"]["sweep_record"]["detail"]["favors"] == "consistency"


def test_status_priority_is_gate_then_floor_then_placebo():
    rec = {"direct1": {"s01", "s03", "s05"}}
    ref = {name: {1} for name in tb.PRIMARY}  # three connections, one minimum: floor
    record = cap_record(run(rec=rec, ref=ref), "c0p05")
    assert record["detail"]["placebo"]["flagged"] is True
    assert record["detail"]["status"] == "floor"
    gated = cap_record(run(rec={**rec, **NO_ENERGY}, ref=ref), "c0p05")
    assert gated["detail"]["status"] == "gate"


def test_wording_also_covers_lambda_one_records_and_attribution():
    rows = build_rows(rec=merge(CONS_CLEAN, {"lam1_cons0": {"s01", "s03", "s05"},
                                              "lam1_cons1": {"s01", "s03", "s05"}}),
                      names=tb.PRIMARY + tb.LAMBDA1)
    out = rules.analyze(rows)
    records = json.dumps([out["exploratory"]["cap_records"],
                          out["exploratory"]["sweep_record"]]).lower()
    attribution_text = json.dumps(out["attribution"]).lower().replace(
        rules.NOT_ATTRIBUTABLE.lower(), "")  # the spec's own phrase, quoted verbatim
    for text in (records, attribution_text):
        assert "effect" not in text and "significan" not in text


def test_zero_contrast_priority_is_after_floor_and_before_placebo():
    # No interpretable items at all: 0 = 0 holds, but the floor status wins.
    floor = cap_record(run(rec={}, ref={name: set() for name in tb.PRIMARY}), "c0p05")
    assert floor["detail"]["direction"]["zero_contrast"] is True
    assert floor["detail"]["status"] == "floor"
    # Direct seed 1 recovers only starts that no consistency checkpoint can use as an
    # item (minimum 1 unreachable for consistency): the item totals are 0 = 0 while
    # the direct seed contrast looks like a placebo flag; zero contrast wins.
    rec = {"direct1": {"s01", "s03", "s05"}}
    ref = {"cons0": {3, 6, 7, 8, 9}, "cons1": {3, 6, 7, 8, 9}}
    record = cap_record(run(rec=rec, ref=ref), "c0p05")
    assert record["detail"]["placebo"]["flagged"] is True
    assert record["detail"]["direction"]["zero_contrast"] is True
    assert record["detail"]["status"] == "zero_contrast"
