"""The five-category outcome taxonomy for relaxation endpoints.

Every relaxation endpoint ends in exactly one category and categories never
collapse into each other:

- matched_reference: converged and matching the branch's own reference
  minimum class under the symmetry-aware comparison.
- unmatched_converged: converged to a force-field fixed point that is not
  the branch's own reference minimum.
- ambiguous: within tolerance of more than one reference equivalence class.
- unresolved: the comparison could not complete under the policy search
  limit (or the final structure is nonfinite). The branch stays visible.
- nonconverged: the budget was exhausted or a nonfinite force occurred
  before the convergence rule was met. A nonconverged run is never counted
  as a wrong shape.

The structural comparison itself is NOT reimplemented here. It reuses
classify_endpoint from the frozen force_consistency matching module, which
already returns matched / ambiguous / unmatched / unresolved plus the
reference class.
"""
import numpy as np

from force_consistency.data.matching import _reference_classes, classify_endpoint

CATEGORY_NAMES = ("matched_reference", "unmatched_converged", "ambiguous",
                  "unresolved", "nonconverged")


def classify_outcome(final, reference, references, policy, *,
                     budget_exhausted, converged, forces_finite=True,
                     seen_minimum=None, final_positions=None, nonfinite_force=False):
    """Classify one relaxation endpoint into exactly one category.

    final: the endpoint as a FrameRecord-like object with record_id,
        species and positions (positions in angstrom).
    reference: the branch's own reference minimum (FrameRecord).
    references: dict of reference minimum id to FrameRecord for the whole
        molecule (the equivalence domain for ambiguity).
    policy: the PreparationPolicy whose tolerances decide matching.
    budget_exhausted: True when the force-evaluation budget ran out before
        the convergence rule was met.
    converged: True when the relaxation reported its convergence rule met.
    forces_finite: True when every force evaluation in the run was finite.
    nonfinite_force: True when termination was triggered by a nonfinite
        force. It forces nonconverged even if a caller passed converged.
    seen_minimum: the seen_minimum flag from the frozen manifest, carried
        onto every result unchanged.
    final_positions: optional explicit endpoint array for the audit trail;
        defaults to final.positions.

    Returns a dict with exactly one category flag true, the reuse of the
    frozen matcher's verdict when a comparison was attempted, the measured
    rmsd, permutation and rotation where available, and the retained
    endpoint and seen_minimum flag.
    """
    if budget_exhausted is None or converged is None:
        raise ValueError("budget_exhausted and converged must be explicit booleans")

    if final_positions is None:
        final_positions = np.asarray(final.positions, dtype=np.float64)
    # The endpoint is always retained for audit, whatever the verdict. The
    # caller's array is carried through unchanged as the caller owns it.
    retained = np.asarray(final_positions, dtype=np.float64)

    # A nonfinite structure means the comparison cannot be honest.
    structure_finite = bool(np.isfinite(retained).all())

    if (not forces_finite) or nonfinite_force:
        category = "nonconverged"
        return _result(category, None, final=final, retained=retained,
                       seen_minimum=seen_minimum, budget_exhausted=budget_exhausted,
                       converged=False, forces_finite=False, nonfinite_force=True)

    if budget_exhausted or not converged:
        # The budget ran out or the run stopped without meeting the force
        # rule. Whatever shape the endpoint shows, it is not evidence:
        # a nonconverged run is never counted as a wrong shape. The
        # comparison is not attempted at all.
        category = "nonconverged"
        return _result(category, None, final=final, retained=retained,
                       seen_minimum=seen_minimum, budget_exhausted=True,
                       converged=False, forces_finite=True, nonfinite_force=False)

    if not structure_finite:
        # A finite run ended in a nonfinite structure. That is a
        # comparison failure, not a shape verdict.
        category = "unresolved"
        return _result(category, None, final=final, retained=retained,
                       seen_minimum=seen_minimum, budget_exhausted=budget_exhausted,
                       converged=True, forces_finite=True, nonfinite_force=False,
                       comparison={"status": "unresolved", "reason": "final_positions_nonfinite"})

    matching = classify_endpoint(final, references, policy)
    verdict = matching["status"]

    if verdict == "unresolved":
        category = "unresolved"
    elif verdict in ("matched", "ambiguous"):
        # classify_endpoint is pool level. Its verdict says which classes
        # the endpoint falls inside, but it has no concept of a branch's
        # own reference. The taxonomy needs that concept: the outcome is
        # matched_reference only when the endpoint ALSO matches a direct
        # comparison against the branch's own reference minimum, and the
        # ambiguity question is answered by the pool comparison. Two
        # comparisons, two jobs:
        #   - branch-own comparison ({own key: own record}): matched means
        #     the endpoint sits inside the branch's own reference class.
        #   - pool comparison (already computed above): matched means
        #     exactly one class accepted; ambiguous means more than one.
        own_id = getattr(reference, "record_id", None)
        if own_id is None:
            raise ValueError("the branch's own reference record has no record_id")
        # The reference pool may be keyed by aliases (for example "m0")
        # rather than by record_id, and classify_endpoint works in
        # record_id space. Resolve the pool key whose record carries the
        # branch reference's record_id, then name the class from that key.
        own_key = None
        for pool_key, pool_record in references.items():
            if getattr(pool_record, "record_id", None) == own_id:
                own_key = pool_key
                break
        if own_key is None:
            raise ValueError(
                f"the branch's own reference {own_id!r} is not in the reference pool")
        classes, class_unresolved = _reference_classes(references, policy)
        own_class = None
        for members in classes:
            if own_key in members:
                own_class = members
                break
        if class_unresolved or own_class is None:
            # The class table itself could not be completed under the
            # policy search limit, so no honest branch-own verdict exists.
            category = "unresolved"
        else:
            own = classify_endpoint(final, {own_key: reference}, policy)
            if own["status"] == "unresolved":
                category = "unresolved"
                matching = own
            elif verdict == "ambiguous":
                # Within tolerance of more than one reference class: the
                # pool decides this, whatever the branch-own verdict is.
                category = "ambiguous"
            elif own["status"] == "matched":
                category = "matched_reference"
            else:
                # Converged inside SOME class but not the branch's own:
                # a wrong shape, and the reported comparison is the
                # branch-own one (status unmatched, no reference class).
                category = "unmatched_converged"
                matching = own
    elif verdict == "unmatched":
        category = "unmatched_converged"
    else:
        raise ValueError(f"The frozen matcher returned an unknown status: {verdict!r}")

    return _result(category, matching, final=final, retained=retained,
                   seen_minimum=seen_minimum, budget_exhausted=budget_exhausted,
                   converged=converged, forces_finite=True, nonfinite_force=False)


def _result(category, matching, *, final, retained, seen_minimum,
            budget_exhausted, converged, forces_finite, nonfinite_force,
            comparison=None):
    rmsd = None
    permutation = None
    rotation = None
    accepted_classes = None
    if matching is not None:
        accepted_classes = matching.get("accepted_classes")
        representative = matching.get("mapping")
        if representative is not None:
            rmsd = representative.get("rmsd")
            permutation = representative.get("permutation")
            rotation = representative.get("rotation")
        scores = matching.get("scores") or {}
        if rmsd is None and accepted_classes:
            # Ambiguous verdicts carry no single representative mapping, so
            # report the best measured score among the accepted references.
            finite_scores = []
            for class_members in accepted_classes or []:
                if not isinstance(class_members, (list, tuple)):
                    class_members = [class_members]
                for key in class_members:
                    if scores.get(key) is not None:
                        finite_scores.append(float(scores[key]))
            if finite_scores:
                rmsd = min(finite_scores)
    result = {
        "final": final,
        "final_positions": retained,
        "seen_minimum": seen_minimum,
        "budget_exhausted": budget_exhausted,
        "converged": converged,
        "forces_finite": forces_finite,
        "nonfinite_force": nonfinite_force,
        "category": category,
        "matching": matching,
        "comparison": comparison if comparison is not None else (matching if matching is not None else None),
        "rmsd_A": rmsd,
        "permutation": permutation,
        "rotation": rotation,
        "accepted_classes": accepted_classes,
    }
    for name in CATEGORY_NAMES:
        result["is_" + name] = name == category
    return result