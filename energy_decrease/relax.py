"""Uncapped follow-up derived from the frozen Stage 2 relaxation function."""

import numpy as np

from evaluation.adapter import ForceShapeMismatch, MalformedForceReturn, NonfiniteValueError
from stage2_capped.relax_capped import (
    _NO_STEP_TOLERANCE_A,
    _REVERSAL_FLOOR_ABS,
    _STEP_GROWTH_CAP,
    _STEP_GROWTH_FACTOR,
    _atom_force_norm_max,
    _check_start,
    _validate_procedure,
    effective_scale,
)


def relax_energy(start, force_fn, procedure, *, arm=None, recorder=None, energy_fn=None):
    """Run the uncapped search with an optional strict energy decrease.

    The scalar energy callback receives one unbatched geometry. It is
    queried only for a force-eligible trial and its first incumbent.
    """
    cap, initial_scale, factor, floor, tol, max_attempts = _validate_procedure(procedure)
    movement_cap = None

    # An (N, 3) start is the unbatched one-molecule form. It is relaxed in
    # the internal batched shape and reported back unbatched, so callers
    # never receive or supply a phantom batch axis.
    start_array = np.asarray(start, dtype=np.float64)
    unbatched = start_array.ndim == 2
    if unbatched:
        start_array = start_array[np.newaxis, :, :]
    _check_start(start_array)
    if start_array.shape[0] != 1:
        raise ValueError("relax_capped needs batch size 1: the movement cap uses "
                         "the largest force norm over the whole batch")
    positions = start_array.copy()

    evals = 0
    known_force = None
    scale = initial_scale
    steps = []
    force_maximums = []
    forces_finite = True
    energy_calls = 0
    incumbent_energy = None
    energy_trials = []
    budget_exhausted = False
    converged = False
    failure_reason = None
    refusal_details = []
    arm_id = getattr(arm, "get", lambda _key, _default=None: None)("arm_id") if arm else None
    # Per-run cap bookkeeping and the read-only recorder hook. Neither feeds
    # back into the relaxation: log_attempt only reads loop state.
    cap_log = {"trial_count": 0, "capped_trial_count": 0,
               "first_capped_attempt": None, "accepted_trial_count": 0,
               "rejected_trial_count": 0, "refused_trial_count": 0,
               "max_accepted_displacement_A": None,
               "max_trial_displacement_A": None,
               "accepted_travel_A": 0.0, "attempted_travel_A": 0.0}
    step_index = 0
    trials = 0
    scale_applied = None
    step_force_norm = None
    step_displacement = None
    step_capped = False

    def log_attempt(kind, attempt_positions, force, accepted=None):
        # One call per force attempt, so the number of calls equals the
        # force-evaluation count. Reads loop state; changes none of it.
        refused = (failure_reason if force is None or failure_reason in
                   ("nonfinite_energy", "invalid_energy") else None)
        force_max = None if force is None else float(np.max(np.abs(force)))
        if kind == "trial":
            cap_log["trial_count"] += 1
            cap_log["attempted_travel_A"] += step_displacement
            previous = cap_log["max_trial_displacement_A"]
            if previous is None or step_displacement > previous:
                cap_log["max_trial_displacement_A"] = step_displacement
            if step_capped:
                cap_log["capped_trial_count"] += 1
                if cap_log["first_capped_attempt"] is None:
                    cap_log["first_capped_attempt"] = evals
            if refused is not None:
                cap_log["refused_trial_count"] += 1
            elif accepted:
                cap_log["accepted_trial_count"] += 1
                cap_log["accepted_travel_A"] += step_displacement
                previous = cap_log["max_accepted_displacement_A"]
                if previous is None or step_displacement > previous:
                    cap_log["max_accepted_displacement_A"] = step_displacement
            else:
                cap_log["rejected_trial_count"] += 1
        if recorder is not None:
            is_trial = kind == "trial"
            recorder({
                "attempt_index": evals,
                "kind": kind,
                "step_index": step_index if is_trial else 0,
                "trial_in_step": trials if is_trial else 0,
                "scale_s": scale if is_trial else None,
                "scale_effective": scale_applied if is_trial else None,
                "force_norm_max_f": step_force_norm if is_trial else None,
                "capped": step_capped if is_trial else False,
                "max_displacement_A": step_displacement if is_trial else None,
                "accepted": accepted if is_trial else None,
                "refused": refused,
                "converged_here": force_max is not None and force_max <= tol,
                "max_force_component": force_max,
                "positions": np.array(attempt_positions[0], dtype=np.float64),
            })

    def max_component(f):
        return float(np.max(np.abs(f)))

    def record_refusal(reason, detail, values_finite):
        # The refusal is recorded with its own detail line before the run
        # stops, so the audit trail carries WHICH arm refused and WHAT it
        # refused, not a flat reason string. The attempt that hit the
        # refusal has already been counted as the attempt it spent.
        # values_finite is False only for a genuine nonfinite value: a
        # wrongly shaped or malformed return carries no readable values,
        # so it refuses the run without labelling the values nonfinite,
        # and the outcome taxonomy derives its nonfinite_force flag from
        # exactly this one fact.
        nonlocal forces_finite, failure_reason
        if not values_finite:
            forces_finite = False
        failure_reason = reason
        refusal_details.append({
            "reason": reason,
            "arm_id": arm_id,
            "detail": str(detail),
            "force_evaluation_count": evals,
        })

    def call_force(pos):
        # One unit of budget per ATTEMPT: the count is spent before the
        # supplier runs, so a supplier that raises or refuses still has
        # its call accounted for.
        nonlocal evals
        if evals >= cap:
            return None
        evals += 1
        try:
            raw = force_fn(pos)
        except (NonfiniteValueError, MalformedForceReturn,
                ForceShapeMismatch) as refused:
            # Only the three DECLARED refusal exceptions are routed here
            # (review rounds 2 and 3): the arm's recorded nonfinite error
            # (NonfiniteValueError, a FloatingPointError carrying the
            # refusal as a recorded info field), and the two declared
            # malformed-return exceptions from the adapter's own contract
            # (MalformedForceReturn for a return that cannot become a
            # numeric array, ForceShapeMismatch for a returned shape that
            # does not mirror the positions). EVERY other exception keeps
            # propagating: ArmConfigurationError (species, atom count or
            # array rank mismatch — a configuration error, not an
            # outcome) and a supplier's own ValueError, TypeError or any
            # other undeclared type, because an internal model error or a
            # misconfiguration is not a refusal, it fails the run loudly
            # instead of being scored as a branch outcome (review round
            # 3, gap 3). Each routed refusal keeps its exception text as
            # the record detail. The nonfinite route alone touches the
            # values-finite fact: a malformed return carries no nonfinite
            # value.
            if isinstance(refused, FloatingPointError):
                record_refusal("nonfinite_force", refused, values_finite=False)
            else:
                record_refusal("invalid_force_shape", refused, values_finite=True)
            return None
        try:
            # The RETURNED VALUE's conversion is guarded separately from
            # the supplier call above (review round 4, gap 2): a plain
            # supplier that returns something nonnumeric reaches the same
            # declared malformed-return refusal the adapter's callable_arm
            # contract raises as MalformedForceReturn — recorded, named by
            # the arm when one was supplied, and never read as physics —
            # instead of aborting the caller with a bare conversion
            # ValueError. This guard wraps ONLY the conversion: exceptions
            # raised by force_fn itself stay in the catch above, so an
            # internal supplier error still propagates loudly instead of
            # being recorded as a refusal. There is no fallback value and
            # no convergence verdict on a malformed return.
            out = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as malformed:
            record_refusal("invalid_force_shape",
                           f"nonnumeric force return ({malformed})",
                           values_finite=True)
            return None
        if out.shape != np.asarray(pos).shape:
            # A returned force that does not mirror the positions shape is
            # a refusal, never physics. A scalar zero (broadcasting) is
            # excluded here rather than being read as convergence. The
            # values seen stay finite-labelled: only a genuine nonfinite
            # VALUE sets that fact (review round 2). The detail speaks
            # the caller's shapes: an unbatched (N, 3) start is relaxed
            # in the internal batched shape, and the audit record never
            # invents a phantom batch axis the caller did not supply.
            positions_label = (np.asarray(pos)[0].shape if unbatched
                               else np.asarray(pos).shape)
            record_refusal(
                "invalid_force_shape",
                f"force shape {out.shape} for positions {positions_label}",
                values_finite=True)
            return None
        if not np.isfinite(out).all():
            record_refusal(
                "nonfinite_force",
                "the returned force carries a nan or inf component",
                values_finite=False)
            return None
        return out

    def call_energy(pos):
        nonlocal energy_calls
        energy_calls += 1
        try:
            value = np.asarray(energy_fn(pos[0]))
        except NonfiniteValueError as error:
            record_refusal("nonfinite_energy", error, values_finite=True)
            return None
        if value.shape != () or value.dtype.kind not in "fiu":
            record_refusal("invalid_energy", "expected a real scalar energy",
                           values_finite=True)
            return None
        if not np.isfinite(value):
            record_refusal("nonfinite_energy", "energy is nonfinite", values_finite=True)
            return None
        return float(value)

    def accepts_energy(candidate):
        nonlocal incumbent_energy
        if incumbent_energy is None:
            incumbent_energy = call_energy(positions)
        trial_energy = call_energy(candidate) if incumbent_energy is not None else None
        if trial_energy is None:
            decision = "refused"
        elif trial_energy < incumbent_energy:
            decision = "accepted"
        elif trial_energy == incumbent_energy:
            decision = "equal"
        else:
            decision = "uphill"
        energy_trials.append({
            "attempt_index": evals,
            "incumbent_eV": incumbent_energy,
            "trial_eV": trial_energy,
            "decision": decision,
        })
        if decision == "accepted":
            incumbent_energy = trial_energy
        return decision == "accepted"

    while True:
        if failure_reason is not None:
            break
        if known_force is None:
            known_force = call_force(positions)
            log_attempt("initial", positions, known_force)
            if known_force is None:
                break
            force_maximums.append(max_component(known_force))

        # The declared rule, tested on the force the budget already bought.
        # No second condition is attached to it.
        if max_component(known_force) <= tol:
            converged = True
            break

        next_step = scale * known_force
        if max_component(next_step) <= _NO_STEP_TOLERANCE_A:
            # The structure cannot move at any admissible scale but the
            # force rule is not met: an honest stuck state, not a
            # converged one and not a budget one.
            failure_reason = failure_reason or "no_progress"
            break

        trials = 0
        accepted = False
        step_index = len(steps)
        while trials < max_attempts:
            if evals >= cap:
                # The budget is spent before an admissible trial step can
                # be evaluated, so there is genuinely no further evidence
                # to buy: the run is honestly nonconverged on budget.
                budget_exhausted = True
                failure_reason = failure_reason or "budget_exhausted"
                break
            step_force_norm = _atom_force_norm_max(known_force)
            scale_applied = effective_scale(scale, step_force_norm, movement_cap)
            step_capped = scale_applied < scale
            candidate = positions + scale_applied * known_force
            step_displacement = float(np.max(np.linalg.norm(candidate - positions,
                                                            axis=-1)))
            trial_force = call_force(candidate)
            trials += 1
            if trial_force is None:
                log_attempt("trial", candidate, None, accepted=False)
                break
            force_maximums.append(max_component(trial_force))
            # A2: the trial endpoint was evaluated within the allowed
            # budget. If its force satisfies the declared rule, the
            # result is established now, before any budget-exhaustion
            # claim; no extra evaluation is needed or taken.
            trial_converged = max_component(trial_force) <= tol
            force_eligible = trial_converged or (
                float(np.sum(trial_force * known_force)) >= -_REVERSAL_FLOOR_ABS
            )
            if force_eligible and energy_fn is not None:
                force_eligible = accepts_energy(candidate)
            if failure_reason is not None:
                log_attempt("trial", candidate, trial_force, accepted=False)
                break
            if trial_converged and force_eligible:
                positions = candidate
                known_force = trial_force
                steps.append({"accepted_step_size": scale_applied, "trial_count": trials})
                log_attempt("trial", candidate, trial_force, accepted=True)
                converged = True
                break
            if force_eligible:
                # The force at the candidate does not oppose the step, so
                # the step is accepted at this scale and the trial force
                # becomes the next step's direction.
                positions = candidate
                known_force = trial_force
                steps.append({"accepted_step_size": scale_applied, "trial_count": trials})
                log_attempt("trial", candidate, trial_force, accepted=True)
                # Growth doubles the scale up to the absolute cap, so the
                # final step into a separable well lands on its minimum.
                scale = min(scale * _STEP_GROWTH_FACTOR, _STEP_GROWTH_CAP)
                accepted = True
                break
            # The candidate force opposes the step: the step overshot far
            # enough to land past a fixed point. Shrink and try again.
            log_attempt("trial", candidate, trial_force, accepted=False)
            scale = scale_applied * factor
            if scale < floor:
                scale = floor
        if failure_reason is not None:
            break
        if converged:
            # Convergence was declared on the trial force just accepted:
            # the run is finished, whatever the accepted flag says.
            break
        if budget_exhausted:
            break
        if not accepted:
            failure_reason = failure_reason or "no_progress"
            break

    force_history = {
        "evaluation_count": evals,
        "maximum_seen_eV_per_A": max(force_maximums) if force_maximums else None,
        "force_maximums": list(force_maximums),
    }
    result = {
        "status": "converged" if converged else "nonconverged",
        "converged": converged,
        "budget_exhausted": budget_exhausted,
        "forces_finite": forces_finite,
        "failure_reason": failure_reason,
        # The audit trail rides on EVERY return path: an empty list when
        # nothing was refused, the recorded details when something was.
        "refusal_details": refusal_details,
        "final_positions": positions,
        "step_count": len(steps),
        "steps": steps,
        "force_evaluation_count": evals,
        "history": force_history,
        "force_history": force_history,
        "final_max_force_component_eV_per_A":
            max_component(known_force) if known_force is not None else None,
    }
    result["cap_record"] = dict(cap_log, movement_cap_A=movement_cap)
    # An unbatched (N, 3) start gets an unbatched (N, 3) endpoint back.
    if unbatched:
        result["final_positions"] = result["final_positions"][0]
    if energy_fn is not None:
        result["energy_record"] = {"requests": energy_calls, "trials": energy_trials}
    return result
