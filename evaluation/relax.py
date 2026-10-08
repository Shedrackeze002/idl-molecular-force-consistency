"""The frozen relaxation procedure: steepest descent driven by any arm.

One procedure is shared by every arm. It is deliberately simple and fully
deterministic:

- The step direction is the predicted force itself (steepest ascent in
  force space, which is descent in energy for a conservative arm).
- A proposed step is accepted only when the force at the proposed point
  does not oppose the step direction. A step whose endpoint force points
  backwards has overshot past a fixed point; it is rejected and the step
  scale is reduced by the backtracking factor. This "does the force agree
  with the step" test is the backtracking rule here.
- An accepted step is allowed to grow (doubled, capped at one angstrom per
  trial) so open approaches do not crawl. The cap is absolute, not tied to
  the initial step size, so the final step into a separable well lands on
  the minimum to floating point precision instead of hovering near it.
- The currently known force is carried across steps: the force evaluated
  at an accepted step's endpoint becomes the next step's direction. A
  run that never backtracks therefore spends exactly one evaluation per
  accepted step plus the first one.
- Budget: every force attempt consumes exactly one unit. The count is
  spent when the attempt is made, before the value is looked at, so a
  supplier that raises still has its call accounted for. The declared
  convergence rule is tested against the force the run already evaluated
  before any budget-exhaustion claim is made, so a result established
  within the allowed budget is reported as established, never thrown
  away because the budget happened to be spent on it. Budgets are never
  extended and never silently converted into success.
- Convergence: the declared rule only. A run converges exactly when the
  maximum absolute force component at an evaluated structure is at or
  below convergence_tolerance_eV_per_A. No secondary "fixed point" gate
  exists: the rule is the whole declaration, evaluated on the latest
  force the budget bought (first the start's own force, then each
  accepted step's endpoint force).
- Termination: the declared rule, budget exhaustion, a refused force
  evaluation, or backtracking exhaustion (no acceptable step scale, or a
  structure that cannot move at any admissible scale while the rule is
  unmet). The failure modes stay separate through "failure_reason":
  "no_progress" covers a run stuck above the force rule with no
  admissible step, which is a different endpoint from a budget that ran
  out or a force that was refused. A refused force ("nonfinite_force",
  "invalid_force_shape") is a recorded branch failure: the run reports
  it and returns, instead of aborting the caller or reading the refused
  array as physics. The caller scores nonconvergence without ever
  reading the endpoint as a shape. The two refusal facts stay
  independent (review round 2): only a genuinely nonfinite value sets
  "forces_finite" False (and the outcome's nonfinite_force flag), while
  a wrongly shaped or malformed return refuses the run with its own
  reason and leaves the values-seen-finite fact True, because a shape
  that cannot be compared carries no nonfinite value.

The convergence rule is a maximum per-atom force COMPONENT at or below the
tolerance in eV/angstrom. It is deliberately not a norm: the norm mixes
components, so a structure with one large stiff component could hide under
a small flexible average.

Positions are (B, N, 3) in angstrom, forces come back (B, N, 3) in
eV/angstrom. An (N, 3) start is accepted as the unbatched one-molecule
form and the endpoint is returned in the same unbatched shape.
"""
from evaluation.adapter import ArmConfigurationError
import numpy as np

# Dot products more negative than this absolute amount count as genuine
# opposition between the trial force and the step direction. Anything
# smaller in magnitude, including exact zero, is floating point noise or
# a perpendicular landing and is accepted.
_REVERSAL_FLOOR_ABS = 1e-12

# How much an accepted step is allowed to grow, relative to itself.
_STEP_GROWTH_FACTOR = 2.0

# The absolute cap on any single trial step, in angstrom. Growth stops
# here so a loose branch cannot jump across a tight well in one trial;
# the initial step size only sets where the growth starts.
_STEP_GROWTH_CAP = 1.0

# A proposed step whose largest force-component contribution falls below
# this (in eV/angstrom units, multiplied by the step scale) cannot change
# the structure by any measurable amount at any admissible scale. This is
# a TERMINATION record only: when the force rule is NOT met and even the
# largest admissible step cannot move the structure, the run stops with
# the recorded "no_progress" reason. It is never part of the convergence
# declaration, which is exactly and only the force rule.
_NO_STEP_TOLERANCE_A = 1e-9


def _validate_procedure(procedure):
    """Reject procedure settings the frozen rules do not allow."""
    required = ("max_force_evaluations", "initial_step_size", "backtracking_factor",
                "min_backtracking_floor", "convergence_tolerance_eV_per_A",
                "max_backtracking_attempts_per_step")
    for key in required:
        if key not in procedure:
            raise ValueError(f"The procedure is missing the setting {key!r}")
    cap = int(procedure["max_force_evaluations"])
    if cap <= 0:
        raise ValueError("max_force_evaluations must be a positive integer")
    initial = float(procedure["initial_step_size"])
    if not initial > 0.0:
        raise ValueError("initial_step_size must be positive")
    factor = float(procedure["backtracking_factor"])
    if not 0.0 < factor < 1.0:
        raise ValueError("backtracking_factor must lie strictly between 0 and 1")
    floor = float(procedure["min_backtracking_floor"])
    if not floor > 0.0:
        raise ValueError("min_backtracking_floor must be positive")
    tol = float(procedure["convergence_tolerance_eV_per_A"])
    if not tol > 0.0:
        raise ValueError("convergence_tolerance_eV_per_A must be positive")
    attempts = int(procedure["max_backtracking_attempts_per_step"])
    if attempts <= 0:
        raise ValueError("max_backtracking_attempts_per_step must be positive")
    return cap, initial, factor, floor, tol, attempts


def _check_start(positions):
    """A start must be finite and free of coincident atoms per molecule.

    Coincident atoms make the chemical graph degenerate and would make the
    symmetry-aware endpoint comparison ill defined. Relaxing from such a
    start would spend budget on a structure that cannot be scored, so the
    procedure refuses it instead.
    """
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError("start must have shape (batch, atoms, 3)")
    if not np.isfinite(positions).all():
        raise ValueError("start positions must all be finite")
    for row in positions:
        if row.shape[0] > 1:
            difference = np.linalg.norm(row[:, None, :] - row[None, :, :], axis=-1)
            # The pairwise matrix includes each atom against itself on the
            # diagonal, and that self distance is always exactly zero. A
            # zero self distance is not a coincident pair, so the diagonal
            # is excluded before the minimum is taken. Genuine overlaps
            # (two different atoms at the same point) still raise below.
            np.fill_diagonal(difference, np.inf)
            if float(np.min(difference)) <= 1e-10:
                raise ValueError("start has coincident atoms in one molecule")


def relax(start, force_fn, procedure, *, arm=None):
    """Relax one start structure under the frozen procedure.

    start: positions shaped (B, N, 3) or the unbatched (N, 3), in
        angstrom, finite, no two atoms of one molecule coincident.
    force_fn: callable mapping positions (B, N, 3) to forces (B, N, 3) in
        eV/angstrom. It is called on whole arrays, never per atom.
    procedure: the frozen settings dict, validated by _validate_procedure.
    arm: optional arm dict carrying the arm's identity. When supplied it
        is only read for its arm_id, so a refused force records WHICH arm
        refused instead of an anonymous refusal. Nothing else about the
        arm is consulted and no force is ever read from it; the force
        evaluation itself stays bound through force_fn alone.

    The returned dict carries:

    - status: "converged" or "nonconverged". Nothing else.
    - converged, budget_exhausted, forces_finite: the three orthogonal
      facts the outcome taxonomy needs.
    - failure_reason: which rule stopped the run (None on convergence):
      "budget_exhausted" when the budget ran out before the force rule
      was met, "nonfinite_force" or "invalid_force_shape" when a force
      evaluation was refused and recorded, or "no_progress" when the
      backtracking rule accepted no step or the structure sits stuck
      above the convergence rule.
    - refusal_details: [] on every run that saw no declared refusal. When
      a refusal stopped the run it names which refusal happened, from
      which arm (when the arm carries an arm_id) and at which attempt
      number, so two refusals of one wording from two different arms stay
      distinguishable in the audit trail instead of collapsing into one
      bare reason string.
    - final_positions: the endpoint in the start's own shape. Never NaN
      poisoned: the run stops at the last structure whose evaluation was
      finite, so the caller always receives a finite endpoint.
    - step_count and steps: how many accepted steps were taken, each with
      its accepted step size and the number of trial evaluations it cost.
    - force_evaluation_count: the exact number of force attempts. A
      refused evaluation (nonfinite or wrongly shaped) counts as the one
      budget unit its attempt spent. Budget accounting lives here and
      nowhere else.
    - history / force_history: the summary of every force evaluation seen.
    - final_max_force_component_eV_per_A: the maximum absolute force
      component at the endpoint, from the last evaluation.
    """
    cap, initial_scale, factor, floor, tol, max_attempts = _validate_procedure(procedure)

    # An (N, 3) start is the unbatched one-molecule form. It is relaxed in
    # the internal batched shape and reported back unbatched, so callers
    # never receive or supply a phantom batch axis.
    start_array = np.asarray(start, dtype=np.float64)
    unbatched = start_array.ndim == 2
    if unbatched:
        start_array = start_array[np.newaxis, :, :]
    _check_start(start_array)
    positions = start_array.copy()

    evals = 0
    known_force = None
    scale = initial_scale
    steps = []
    force_maximums = []
    forces_finite = True
    budget_exhausted = False
    converged = False
    failure_reason = None
    refusal_details = []
    arm_id = getattr(arm, "get", lambda _key, _default=None: None)("arm_id") if arm else None

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
            out = np.asarray(raw, dtype=np.float64)
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
        except (FloatingPointError, ValueError) as refused:
            # Only the two DECLARED refusals are routed here (review
            # round 2): a nonfinite value (FloatingPointError, including
            # the arm's recorded NonfiniteValueError), and a ValueError
            # raised INSIDE the supplier while computing, which the
            # adapter's contract documents as the malformed-return or
            # in-compute refusal fact (callable_arm's wrong output shape
            # raises ValueError under the same contract). A configuration
            # error (species, atom count or array rank mismatch:
            # ArmConfigurationError, a ValueError subclass) and every
            # undeclared error type escape this block and keep
            # propagating, so a wrongly configured arm is never scored as
            # a branch outcome, it fails the run loudly. Each routed
            # refusal keeps its exception text as the record detail. The
            # floating-point route alone touches the values-finite fact:
            # a malformed-return ValueError carries no nonfinite value.
            if isinstance(refused, ArmConfigurationError):
                raise
            if isinstance(refused, FloatingPointError):
                record_refusal("nonfinite_force", refused, values_finite=False)
            else:
                record_refusal("invalid_force_shape", refused, values_finite=True)
            return None

    while True:
        if failure_reason is not None:
            break
        if known_force is None:
            known_force = call_force(positions)
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
        while trials < max_attempts:
            if evals >= cap:
                # The budget is spent before an admissible trial step can
                # be evaluated, so there is genuinely no further evidence
                # to buy: the run is honestly nonconverged on budget.
                budget_exhausted = True
                failure_reason = failure_reason or "budget_exhausted"
                break
            candidate = positions + scale * known_force
            trial_force = call_force(candidate)
            trials += 1
            if trial_force is None:
                break
            force_maximums.append(max_component(trial_force))
            # A2: the trial endpoint was evaluated within the allowed
            # budget. If its force satisfies the declared rule, the
            # result is established now, before any budget-exhaustion
            # claim; no extra evaluation is needed or taken.
            if max_component(trial_force) <= tol:
                positions = candidate
                known_force = trial_force
                steps.append({"accepted_step_size": scale, "trial_count": trials})
                converged = True
                break
            agree = float(np.sum(trial_force * known_force))
            if agree >= -_REVERSAL_FLOOR_ABS:
                # The force at the candidate does not oppose the step, so
                # the step is accepted at this scale and the trial force
                # becomes the next step's direction.
                positions = candidate
                known_force = trial_force
                steps.append({"accepted_step_size": scale, "trial_count": trials})
                # Growth doubles the scale up to the absolute cap, so the
                # final step into a separable well lands on its minimum.
                scale = min(scale * _STEP_GROWTH_FACTOR, _STEP_GROWTH_CAP)
                accepted = True
                break
            # The candidate force opposes the step: the step overshot far
            # enough to land past a fixed point. Shrink and try again.
            scale = scale * factor
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
    # An unbatched (N, 3) start gets an unbatched (N, 3) endpoint back.
    if unbatched:
        result["final_positions"] = result["final_positions"][0]
    return result