"""Copied second pass of evaluation.runner.evaluate_starts for the capped relaxer.

The loop body below is lines 811 to 861 of the frozen evaluation/runner.py, which
binds `relax` at module scope and has no injection point. Differences from the
frozen lines are limited to the hunks in allowlist_pass.json (the relax call
becomes relax_capped with the cap and a recorder, the row gains the run's
cap_record, and the function returns the rows); a diff gate checks this. Everything
that decides a row (force supplier, final record, classification, audit record) is
imported from the frozen modules, not copied.

The first pass of evaluate_starts (partition, reference and compatibility checks) and
run_model_pass's prevalidation are not copied: driver.prevalidate repeats the same
checks before any force call.
"""
import time

from evaluation.outcomes import classify_outcome
from evaluation.runner import _arm_force_supplier, _branch_audit_record, _final_record
from stage2_capped.relax_capped import relax_capped


def second_pass(plans, arm, procedure, references, policy, seen_minimum, bound, *,
                movement_cap_A, recorder_for):
    """Relax and classify the admitted branches under a movement cap (None: control).

    plans: (branch_key, start_entry, partition, pool_key, own_reference) tuples.
    recorder_for: branch_key -> the per-attempt recorder passed to relax_capped."""
    branches = {}
    for branch_key, start_entry, partition, _pool_key, own in plans:
        species = tuple(start_entry["species"])
        force_fn = _arm_force_supplier(arm, species,
                                       f"{start_entry['record_id']}/relax")
        branch_started = time.perf_counter()
        # The refusal record must name WHICH arm refused: relax reads the
        # arm only for its arm_id, so a refused force keeps the arm's
        # identity in refusal_details instead of collapsing to None.
        recorder = recorder_for(branch_key)
        relaxation = relax_capped(start_entry["positions"], force_fn, procedure,
                                  arm=arm, movement_cap_A=movement_cap_A,
                                  recorder=recorder)
        final = _final_record(branch_key, start_entry, relaxation["final_positions"])
        flag = seen_minimum
        if isinstance(flag, dict):
            flag = flag.get(branch_key)
        if callable(flag):
            flag = flag(branch_key)
        outcome = classify_outcome(
            final, own, references, policy,
            budget_exhausted=relaxation["budget_exhausted"],
            converged=relaxation["converged"],
            forces_finite=relaxation["forces_finite"],
            seen_minimum=flag,
            stop_reason=relaxation["failure_reason"],
        )
        elapsed = time.perf_counter() - branch_started
        branches[branch_key] = {
            "branch_key": branch_key,
            "molecule": start_entry["molecule"],
            "connection_id": start_entry["connection_id"],
            "branch": start_entry["branch"],
            "partition": partition,
            "seen_minimum": outcome["seen_minimum"],
            "start_record_id": start_entry["record_id"],
            "final_record_id": final.record_id,
            "status": relaxation["status"],
            "category": outcome["category"],
            "converged": outcome["converged"],
            "budget_exhausted": outcome["budget_exhausted"],
            "forces_finite": outcome["forces_finite"],
            "nonfinite_force": outcome["nonfinite_force"],
            "stop_reason": outcome["stop_reason"],
            "force_evaluation_count": relaxation["force_evaluation_count"],
            "step_count": relaxation["step_count"],
            "matching_status": (outcome["matching"] or {}).get("status"),
            "accepted_classes": outcome["accepted_classes"],
            "rmsd_A": outcome["rmsd_A"],
            "refusal_details": relaxation["refusal_details"],
            "_audit_record": _branch_audit_record(
                start_entry, arm, procedure,
                relaxation, outcome, own, elapsed, bound),
            "cap_record": relaxation["cap_record"],
        }
    return branches
