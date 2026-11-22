"""M1: relax_capped equals the frozen relaxer at c = None / inf / huge, and
obeys the cap otherwise. Synthetic fields only; no model is loaded."""
import inspect

import numpy as np
import pytest

from evaluation.relax import relax
from stage2_capped import diff_gate
from stage2_capped.relax_capped import (_validate_movement_cap, effective_scale,
                                        relax_capped)
from stage2_capped.tests import fixtures as fx

HUGE_CAPS = [None, float("inf"), 1e300, 1e12, 1e6]
STUDY_CAPS = [0.05, 0.10, 0.25, 1.0]


def scenarios():
    """(name, start, force, procedure) fixtures, incl. budget exhaustion,
    loose tolerance, no_progress and every refusal mode."""
    start = fx.molecule(6, seed=1)
    target = fx.molecule(6, seed=2)
    out = [
        ("harmonic_soft", start, fx.harmonic(target, 0.4), fx.procedure()),
        ("harmonic_stiff", start, fx.harmonic(target, 40.0), fx.procedure()),
        ("valley", start, fx.anisotropic(target, 30.0, 0.3), fx.procedure()),
        ("pair", start * 0.8, fx.pair_potential(), fx.procedure()),
        ("far_start", start * 30.0, fx.harmonic(target, 2.0), fx.procedure()),
        ("budget_exhausted", start, fx.anisotropic(target, 30.0, 0.05),
         fx.procedure(max_force_evaluations=25)),
        ("budget_one", start, fx.harmonic(target, 0.4),
         fx.procedure(max_force_evaluations=1)),
        ("loose_tolerance", start, fx.harmonic(target, 0.4),
         fx.procedure(convergence_tolerance_eV_per_A=1e6)),
        ("loose_after_steps", start, fx.harmonic(target, 0.4),
         fx.procedure(convergence_tolerance_eV_per_A=2.0)),
        ("no_progress_tiny_step", start, fx.constant(0.5),
         fx.procedure(initial_step_size=1e-9)),
        ("opposing", start, fx.opposing(1.0), fx.procedure()),
        ("constant_force", start, fx.constant(3.0), fx.procedure(max_force_evaluations=60)),
        ("few_attempts", start, fx.harmonic(target, 400.0),
         fx.procedure(max_backtracking_attempts_per_step=2)),
        ("unbatched_vs_batched", start[None], fx.harmonic(target[None], 0.4), fx.procedure()),
    ]
    for name, force in fx.refusal_battery().items():
        out.append((f"refusal_{name}", start, force, fx.procedure()))
        out.append((f"refusal_later_{name}", start,
                    fx.flaky(fx.harmonic(target, 0.4), 3, lambda p, f=force: f(p)),
                    fx.procedure()))
    return out


SCENARIOS = scenarios()


def outcome_or_exception(function):
    try:
        return ("result", function())
    except Exception as error:  # compare refusals by type and text
        return ("raised", type(error).__name__, str(error))


def fresh(scenario_name):
    """Scenarios hold stateful forces, so build each call from scratch."""
    for name, start, force, proc in scenarios():
        if name == scenario_name:
            return start, force, proc
    raise KeyError(scenario_name)


@pytest.mark.parametrize("cap", HUGE_CAPS, ids=str)
@pytest.mark.parametrize("name", [s[0] for s in SCENARIOS])
def test_bitwise_equal_to_frozen_at_huge_caps(name, cap):
    start, force, proc = fresh(name)
    frozen = outcome_or_exception(lambda: relax(start, force, proc))
    start, force, proc = fresh(name)
    capped = outcome_or_exception(
        lambda: relax_capped(start, force, proc, movement_cap_A=cap))
    assert frozen[0] == capped[0]
    if frozen[0] == "raised":
        assert frozen == capped
    else:
        assert fx.bits(fx.frozen_fields(capped[1], frozen[1])) == fx.bits(frozen[1])


def test_scenarios_cover_the_outcome_modes():
    reasons = set()
    for name, start, force, proc in scenarios():
        result = relax(start, force, proc)
        reasons.add((result["status"], result["failure_reason"]))
    assert ("converged", None) in reasons
    for reason in ("budget_exhausted", "no_progress", "nonfinite_force",
                   "invalid_force_shape"):
        assert ("nonconverged", reason) in reasons


def test_default_cap_is_the_control():
    start, force, proc = fresh("pair")
    plain = relax_capped(start, force, proc)
    start, force, proc = fresh("pair")
    frozen = relax(start, force, proc)
    assert fx.bits(fx.frozen_fields(plain, frozen)) == fx.bits(frozen)


@pytest.mark.parametrize("name", ["harmonic_stiff", "valley", "pair", "far_start",
                                  "budget_exhausted", "no_progress_tiny_step"])
@pytest.mark.parametrize("cap", [None, 1e6] + STUDY_CAPS, ids=str)
def test_recorder_on_equals_recorder_off(name, cap):
    start, force, proc = fresh(name)
    off = relax_capped(start, force, proc, movement_cap_A=cap)
    start, force, proc = fresh(name)
    records = []
    on = relax_capped(start, force, proc, movement_cap_A=cap, recorder=records.append)
    assert fx.bits(on) == fx.bits(off)
    assert len(records) == on["force_evaluation_count"]
    assert [r["attempt_index"] for r in records] == list(range(1, len(records) + 1))


def test_recorder_mutation_cannot_change_the_run():
    start, force, proc = fresh("pair")
    clean = relax_capped(start, force, proc, movement_cap_A=0.1)

    def vandal(record):
        record["positions"][:] = 1e9

    start, force, proc = fresh("pair")
    touched = relax_capped(start, force, proc, movement_cap_A=0.1, recorder=vandal)
    assert fx.bits(touched) == fx.bits(clean)


def test_recorder_errors_propagate():
    start, force, proc = fresh("pair")

    def broken(record):
        raise RuntimeError("recorder failed")
    with pytest.raises(RuntimeError, match="recorder failed"):
        relax_capped(start, force, proc, movement_cap_A=0.1, recorder=broken)


@pytest.mark.parametrize("name", ["harmonic_soft", "harmonic_stiff", "valley", "pair",
                                  "far_start", "budget_exhausted"])
@pytest.mark.parametrize("cap", STUDY_CAPS, ids=str)
def test_every_trial_obeys_the_cap(name, cap):
    start, force, proc = fresh(name)
    records = []
    result = relax_capped(start, force, proc, movement_cap_A=cap, recorder=records.append)
    trials = [r for r in records if r["kind"] == "trial"]
    assert trials, "fixture must take at least one trial"
    for r in trials:
        assert r["max_displacement_A"] <= cap * (1 + 1e-12)
        # The recorded displacement is the real one: candidate minus parent.
        assert r["scale_effective"] <= r["scale_s"]
    assert result["cap_record"]["max_trial_displacement_A"] <= cap * (1 + 1e-12)


@pytest.mark.parametrize("cap", STUDY_CAPS, ids=str)
def test_rejected_trials_obey_the_cap_too(cap):
    start = fx.molecule(6, seed=1)
    target = fx.molecule(6, seed=2)
    records = []
    relax_capped(start, fx.harmonic(target, 400.0), fx.procedure(),
                 movement_cap_A=cap, recorder=records.append)
    rejected = [r for r in records if r["kind"] == "trial" and not r["accepted"]]
    assert rejected, "fixture must reject at least one trial"
    assert all(r["max_displacement_A"] <= cap * (1 + 1e-12) for r in rejected)


def test_cap_binds_exactly_when_s_f_exceeds_c():
    start, force, proc = fresh("far_start")
    records = []
    relax_capped(start, force, proc, movement_cap_A=0.25, recorder=records.append)
    for r in (x for x in records if x["kind"] == "trial"):
        predicted = r["scale_s"] * r["force_norm_max_f"] > 0.25
        # float rounding at the boundary is the only allowed difference
        if abs(r["scale_s"] * r["force_norm_max_f"] - 0.25) > 1e-9:
            assert r["capped"] == predicted
        assert r["capped"] == (r["scale_effective"] < r["scale_s"])


def test_effective_scale_formula_and_units():
    assert effective_scale(0.2, 10.0, None) == 0.2
    assert effective_scale(0.2, 10.0, 0.5) == 0.05          # c/f = 0.05 < s
    assert effective_scale(0.2, 1.0, 0.5) == 0.2            # c/f = 0.5 > s
    s_eff = effective_scale(0.2, 670.0, 0.05)
    assert s_eff * 670.0 <= 0.05 * (1 + 1e-12)
    # The v1 formula min(s, c/m) mixed units; the real move was c/s over the cap.
    assert 0.2 * 670.0 > 0.05


def test_no_step_test_runs_on_the_uncapped_step():
    # A force just above the tolerance with a floor-sized scale cannot move the
    # structure; the uncapped no-step test stops it regardless of the cap.
    start = fx.molecule(4, seed=3)
    proc = fx.procedure(initial_step_size=1e-9)
    for cap in (None, 0.05, 1.0):
        result = relax_capped(start, fx.constant(0.5), proc, movement_cap_A=cap)
        assert result["failure_reason"] == "no_progress"
        assert result["force_evaluation_count"] == 1


def test_backtracking_shrinks_the_real_step_not_s():
    # Stiff well, small cap: after a rejected capped trial the next candidate
    # must differ from the rejected one (s <- factor * s_eff).
    start = fx.molecule(6, seed=1)
    target = fx.molecule(6, seed=2)
    records = []
    relax_capped(start, fx.harmonic(target, 400.0), fx.procedure(),
                 movement_cap_A=0.05, recorder=records.append)
    trials = [r for r in records if r["kind"] == "trial"]
    assert any(r["capped"] and not r["accepted"] for r in trials), \
        "fixture must reject a capped trial"
    parents = {}
    for r in trials:
        parents.setdefault(r["step_index"], []).append(r)
    for group in parents.values():
        group.sort(key=lambda r: r["trial_in_step"])
        for before, after in zip(group, group[1:]):
            assert not np.array_equal(before["positions"], after["positions"])
            assert after["scale_s"] == max(0.5 * before["scale_effective"], 1e-8)


def test_accepted_step_size_is_the_applied_scale():
    start, force, proc = fresh("far_start")
    records = []
    result = relax_capped(start, force, proc, movement_cap_A=0.25,
                          recorder=records.append)
    accepted = [r for r in records if r["kind"] == "trial" and r["accepted"]]
    assert [s["accepted_step_size"] for s in result["steps"]] == \
        [r["scale_effective"] for r in accepted]
    assert [s["trial_count"] for s in result["steps"]] == \
        [r["trial_in_step"] for r in accepted]
    assert any(r["scale_effective"] < r["scale_s"] for r in accepted)


def test_growth_doubles_s_not_s_eff():
    start = fx.molecule(6, seed=1, spread=4.0)
    target = fx.molecule(6, seed=2, spread=4.0)
    records = []
    relax_capped(start, fx.harmonic(target, 0.05), fx.procedure(),
                 movement_cap_A=0.05, recorder=records.append)
    accepted = [r for r in records if r["kind"] == "trial" and r["accepted"]]
    assert len(accepted) >= 3
    for before, after in zip(accepted, accepted[1:]):
        if after["trial_in_step"] == 1:
            assert after["scale_s"] == min(2 * before["scale_s"], 1.0)


def test_floor_edge_is_documented_behavior():
    # f > c / (2 * floor): backtracking cannot shrink the real step any more.
    s_eff = effective_scale(1e-8, 1e8, 0.05)          # c/f = 5e-10 < floor
    after_backtrack = max(0.5 * s_eff, 1e-8)           # clamped up to the floor
    assert effective_scale(after_backtrack, 1e8, 0.05) == s_eff


def test_unrelated_exceptions_propagate_under_a_cap():
    start = fx.molecule(5, seed=4)
    for name, error in fx.UNRELATED.items():
        with pytest.raises(type(error)):
            relax_capped(start, fx.raises(error), fx.procedure(), movement_cap_A=0.1)
        with pytest.raises(type(error)):
            relax_capped(start, fx.flaky(fx.harmonic(start * 0, 0.4), 3,
                                         fx.raises(error)),
                         fx.procedure(), movement_cap_A=0.1)


@pytest.mark.parametrize("mode", list(fx.refusal_battery()))
@pytest.mark.parametrize("cap", [None] + STUDY_CAPS, ids=str)
def test_declared_refusals_are_recorded_under_a_cap(mode, cap):
    start = fx.molecule(5, seed=4)
    result = relax_capped(start, fx.refusal_battery()[mode], fx.procedure(),
                          movement_cap_A=cap)
    assert result["status"] == "nonconverged"
    assert result["failure_reason"] in ("nonfinite_force", "invalid_force_shape")
    assert result["force_evaluation_count"] == 1


@pytest.mark.parametrize("bad", [0, 0.0, -0.1, 1e-7, float("nan"), float("-inf"), True,
                                 False, "0.1", [0.1], np.array(0.1) if False else (0.1,)])
def test_bad_caps_are_refused(bad):
    start = fx.molecule(4, seed=3)
    with pytest.raises(ValueError):
        relax_capped(start, fx.harmonic(start * 0, 0.4), fx.procedure(), movement_cap_A=bad)


def test_cap_validation_values():
    assert _validate_movement_cap(None) is None
    assert _validate_movement_cap(float("inf")) is None
    assert _validate_movement_cap(1e-6) == 1e-6
    assert _validate_movement_cap(np.float64(0.1)) == 0.1
    assert _validate_movement_cap(1) == 1.0


def test_bad_cap_is_refused_before_any_force_call():
    calls = []

    def spy(positions):
        calls.append(1)
        return np.zeros_like(positions)
    with pytest.raises(ValueError):
        relax_capped(fx.molecule(4), spy, fx.procedure(), movement_cap_A=-1.0)
    assert calls == []


@pytest.mark.parametrize("cap", [None, 0.1])
def test_batch_size_other_than_one_is_refused(cap):
    start = np.stack([fx.molecule(4, seed=1), fx.molecule(4, seed=2)])
    with pytest.raises(ValueError, match="batch size 1"):
        relax_capped(start, fx.harmonic(0 * start, 0.4), fx.procedure(), movement_cap_A=cap)
    # the frozen relaxer still accepts a batch; the capped copy deliberately does not
    assert relax(start, fx.harmonic(0 * start, 0.4), fx.procedure())["status"]


def test_batch_of_one_and_unbatched_agree():
    start = fx.molecule(5, seed=6)
    target = fx.molecule(5, seed=7)
    a = relax_capped(start, fx.harmonic(target, 5.0), fx.procedure(), movement_cap_A=0.1)
    b = relax_capped(start[None], fx.harmonic(target[None], 5.0), fx.procedure(),
                     movement_cap_A=0.1)
    assert a["final_positions"].shape == start.shape
    assert b["final_positions"].shape == start[None].shape
    assert np.array_equal(a["final_positions"], b["final_positions"][0])


@pytest.mark.parametrize("cap", [None] + STUDY_CAPS, ids=str)
def test_determinism_rerun_is_bitwise_identical(cap):
    outputs = []
    for _ in range(2):
        start, force, proc = fresh("pair")
        records = []
        result = relax_capped(start, force, proc, movement_cap_A=cap,
                              recorder=records.append)
        outputs.append(fx.bits((result, records)))
    assert outputs[0] == outputs[1]


def test_cap_record_summary_matches_the_trace():
    start, force, proc = fresh("far_start")
    records = []
    result = relax_capped(start, force, proc, movement_cap_A=0.25,
                          recorder=records.append)
    summary = result["cap_record"]
    trials = [r for r in records if r["kind"] == "trial"]
    assert summary["movement_cap_A"] == 0.25
    assert summary["trial_count"] == len(trials)
    assert summary["capped_trial_count"] == sum(r["capped"] for r in trials) > 0
    assert summary["first_capped_attempt"] == next(r["attempt_index"] for r in trials
                                                   if r["capped"])
    assert summary["accepted_trial_count"] == result["step_count"]
    assert (summary["accepted_trial_count"] + summary["rejected_trial_count"]
            + summary["refused_trial_count"]) == len(trials)
    assert summary["accepted_travel_A"] == pytest.approx(
        sum(r["max_displacement_A"] for r in trials if r["accepted"]), rel=1e-12)
    assert summary["max_accepted_displacement_A"] == max(
        r["max_displacement_A"] for r in trials if r["accepted"])


def test_control_summary_has_no_capped_trials_and_none_cap():
    start, force, proc = fresh("far_start")
    result = relax_capped(start, force, proc, movement_cap_A=float("inf"))
    assert result["cap_record"]["movement_cap_A"] is None
    assert result["cap_record"]["capped_trial_count"] == 0
    assert result["cap_record"]["first_capped_attempt"] is None


def test_a_bound_cap_changes_the_far_start_run():
    start, force, proc = fresh("far_start")
    control = relax_capped(start, force, proc)
    start, force, proc = fresh("far_start")
    capped = relax_capped(start, force, proc, movement_cap_A=0.05)
    assert capped["cap_record"]["capped_trial_count"] > 0
    assert capped["force_evaluation_count"] != control["force_evaluation_count"]


def test_signature_adds_only_the_two_keywords():
    frozen = inspect.signature(relax)
    capped = inspect.signature(relax_capped)
    assert list(frozen.parameters) == ["start", "force_fn", "procedure", "arm"]
    assert list(capped.parameters) == ["start", "force_fn", "procedure", "arm",
                                       "movement_cap_A", "recorder"]


def test_diff_gate_passes_on_the_committed_files():
    diff_gate.assert_gate("relax_capped")


def test_diff_gate_catches_an_unlisted_change(tmp_path):
    root = tmp_path
    (root / "evaluation").mkdir()
    (root / "stage2_capped").mkdir()
    frozen = diff_gate.ROOT / "evaluation/relax.py"
    derived = diff_gate.ROOT / "stage2_capped/relax_capped.py"
    (root / "evaluation/relax.py").write_bytes(frozen.read_bytes())
    text = derived.read_text()
    tampered = text.replace("scale = min(scale * _STEP_GROWTH_FACTOR, _STEP_GROWTH_CAP)",
                            "scale = min(scale * _STEP_GROWTH_FACTOR, 2.0)")
    assert tampered != text
    (root / "stage2_capped/relax_capped.py").write_text(tampered)
    problems = diff_gate.check_gate("relax_capped", root=root)
    assert problems and any("hunk" in p for p in problems)


def test_diff_gate_catches_a_changed_frozen_file(tmp_path):
    (tmp_path / "evaluation").mkdir()
    (tmp_path / "stage2_capped").mkdir()
    frozen = (diff_gate.ROOT / "evaluation/relax.py").read_bytes()
    (tmp_path / "evaluation/relax.py").write_bytes(frozen + b"# edited\n")
    (tmp_path / "stage2_capped/relax_capped.py").write_bytes(
        (diff_gate.ROOT / "stage2_capped/relax_capped.py").read_bytes())
    assert any("pinned SHA-256" in p for p in diff_gate.check_gate("relax_capped",
                                                                    root=tmp_path))


def test_diff_gate_catches_a_missing_and_an_extra_hunk(tmp_path):
    import json
    allowlist = json.loads(diff_gate.ALLOWLIST.read_text())
    hunks = allowlist["targets"]["relax_capped"]["hunks"]
    short = json.loads(json.dumps(allowlist))
    short["targets"]["relax_capped"]["hunks"] = hunks[:-1]
    path = tmp_path / "short.json"
    path.write_text(json.dumps(short))
    assert diff_gate.check_gate("relax_capped", allowlist_path=path)
    bad = json.loads(json.dumps(allowlist))
    bad["targets"]["relax_capped"]["hunks"][0]["categories"] = ["not_allowed"]
    path2 = tmp_path / "bad.json"
    path2.write_text(json.dumps(bad))
    assert any("category" in p for p in diff_gate.check_gate("relax_capped",
                                                              allowlist_path=path2))


def test_every_listed_hunk_has_a_purpose_and_allowed_categories():
    import json
    entry = json.loads(diff_gate.ALLOWLIST.read_text())["targets"]["relax_capped"]
    assert entry["frozen_sha256"] == diff_gate.sha256_file(diff_gate.ROOT / "evaluation/relax.py")
    for hunk in entry["hunks"]:
        assert hunk["purpose"] and set(hunk["categories"]) <= set(entry["allowed_categories"])
