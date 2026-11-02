"""Synthetic result tables for the rules tests (no models).

Ten starts over five connections, mirroring the spec's shape: six of the ten
target two minima (minimum 1: s01, s03, s05; minimum 8: s04, s06, s08). Six
reference starts, one per minimum 1, 3, 6, 7, 8, 9.
"""
from stage2_capped.row_schema import (CAP_VALUES, CONTROL_LABEL, EXPLORATORY_LABEL,
                                      make_row)

CAPS = ("c0p05", "c0p10", "c0p25", "c1p00")
STARTS = [  # id, connection, branch, own minimum
    ("s01", "conn_a", "plus", 1), ("s02", "conn_a", "minus", 3),
    ("s03", "conn_b", "plus", 1), ("s04", "conn_b", "minus", 8),
    ("s05", "conn_c", "plus", 1), ("s06", "conn_c", "minus", 8),
    ("s07", "conn_d", "plus", 6), ("s08", "conn_d", "minus", 8),
    ("s09", "conn_e", "plus", 7), ("s10", "conn_e", "minus", 9)]
START_IDS = [s[0] for s in STARTS]
MINIMA = (1, 3, 6, 7, 8, 9)
CKPT = {"energy0": ("energy", 0, None), "direct0": ("direct", 0, None),
        "direct1": ("direct", 1, None), "cons0": ("consistency", 0, 0.1),
        "cons1": ("consistency", 1, 0.1), "lam1_cons0": ("consistency", 0, 1.0),
        "lam1_cons1": ("consistency", 1, 1.0)}
PRIMARY = ("energy0", "direct0", "direct1", "cons0", "cons1")
LAMBDA1 = ("lam1_cons0", "lam1_cons1")


def merge(*dicts):
    out = {}
    for d in dicts:
        for key, value in d.items():
            out[key] = set(out.get(key, set())) | set(value)
    return out


# Consistency recovers where direct does not, on three connections (a, b, c),
# identically in both seed pairs.
CONS_FAVORED = {"cons0": {"s01", "s03", "s05"}, "cons1": {"s01", "s03", "s05"}}
DIRECT_FAVORED = {"direct0": {"s01", "s03", "s05"}, "direct1": {"s01", "s03", "s05"}}
# Seed contrasts on connections d and e that point opposite ways in each arm, so
# the placebo check does not flag a table whose main contrast is clean. Both arms
# recover the same starts, so the arm contrast is unchanged.
NOISE_BREAKER = {"direct0": {"s07"}, "cons0": {"s07"}, "direct1": {"s09"}, "cons1": {"s09"}}


def _row(cap, name, kind, start_id, connection, branch, minimum, recovered, *,
         capped_trials, max_disp, launch="L"):
    arm, seed, lam = CKPT[name]
    control = cap == CONTROL_LABEL
    recovered = bool(recovered)
    stop = None if recovered else "budget_exhausted"
    trials = 7
    capped = 0 if control else min(capped_trials, trials)
    lam_one = lam == 1.0
    fields = {
        "launch_id": launch,
        "stage": ("0b" if control else "2") if lam_one else ("0" if control else "1"),
        "checkpoint_id": name, "arm": arm, "consistency_lambda": lam, "seed": seed,
        "start_kind": kind, "start_id": start_id, "connection": connection,
        "branch": branch, "own_minimum": minimum, "cap_label": cap,
        "movement_cap_A": CAP_VALUES[cap], "relaxer_sha256": "a" * 64,
        "exploratory_label": EXPLORATORY_LABEL if lam_one else None,
        "outcome_category": "matched_reference" if recovered else "nonconverged",
        "recovered": recovered, "stop_reason": stop, "converged": stop is None,
        "budget_exhausted": stop == "budget_exhausted", "step_count": 5,
        "force_evaluation_count": trials + 1,
        "final_max_force_component_eV_per_A": 0.005 if recovered else 0.2,
        "max_accepted_displacement_A": max_disp, "endpoint_connected": True,
        "first_disconnection_attempt": None, "trial_count": trials,
        "capped_trial_count": capped, "capped_fraction": capped / trials,
        "first_capped_attempt": 2 if capped else None, "accepted_trial_count": 5,
        "rejected_trial_count": 2, "refused_trial_count": 0, "accepted_travel_A": 1.0,
        "attempted_travel_A": 1.5, "elapsed_s": 0.1,
        "accepted_step_fraction_gt_0p25": 0.2 if control else None,
        "accepted_step_fraction_gt_1p0": (0.1 if max_disp > 1.0 else 0.0) if control else None,
        "would_bind_at_1p0": (max_disp > 1.0) if control else None,
        "trace_file": f"{cap}/{name}/{start_id}.npy", "trace_sha256": "b" * 64}
    return make_row(**fields)


def build_rows(rec=None, per_cap=None, ref=None, per_cap_ref=None, control=None,
               trials=None, names=PRIMARY, control_max_disp=2.0,
               unbound_follows_control=True, energy_default=True):
    """Rows for every (cap, checkpoint, start).

    rec / per_cap: original-start recoveries ({checkpoint name: set of start ids})
    for every capped cap, and per-cap replacements. ref / per_cap_ref: reference
    recoveries as sets of minima (default: every reference start recovers).
    control: control (c = infinity) original-start recoveries.
    trials: {(cap, name, start_id): capped trial count}, default 3 (the cap binds).
    """
    rec = dict(rec or {})
    if energy_default and "energy0" not in rec:
        rec["energy0"] = {"s01"}
    per_cap = {cap: dict(table) for cap, table in (per_cap or {}).items()}
    if energy_default:  # a per-cap table also gets the passing energy C baseline
        for table in per_cap.values():
            table.setdefault("energy0", {"s01"})
    ref = ref or {}
    per_cap_ref = per_cap_ref or {}
    control = control or {}
    trials = trials or {}
    rows = []
    for cap in (CONTROL_LABEL,) + CAPS:
        for name in names:
            for start_id, connection, branch, minimum in STARTS:
                if cap == CONTROL_LABEL:
                    recovered = start_id in control.get(name, set())
                    capped_trials, max_disp = 0, control_max_disp
                else:
                    table = per_cap.get(cap, rec)
                    recovered = start_id in table.get(name, set())
                    capped_trials = trials.get((cap, name, start_id), 3)
                    max_disp = 0.04
                    if capped_trials == 0 and unbound_follows_control:
                        recovered = start_id in control.get(name, set())
                        max_disp = control_max_disp
                rows.append(_row(cap, name, "original", start_id, connection, branch,
                                 minimum, recovered, capped_trials=capped_trials,
                                 max_disp=max_disp))
            for minimum in MINIMA:
                if cap == CONTROL_LABEL:
                    recovered = True
                else:
                    minima = per_cap_ref.get(cap, ref).get(name, set(MINIMA))
                    recovered = minimum in minima
                rows.append(_row(cap, name, "reference", f"r{minimum}", None, None,
                                 minimum, recovered, capped_trials=0 if cap == CONTROL_LABEL else 3,
                                 max_disp=0.04 if cap != CONTROL_LABEL else control_max_disp))
    return rows


def with_lambda1(rows_kwargs):
    """build_rows over the primary five plus the two lambda = 1 checkpoints."""
    return build_rows(**{**rows_kwargs, "names": PRIMARY + LAMBDA1})
