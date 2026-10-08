"""Arm adapters binding a force supplier to the shared evaluation interface.

An arm is a plain dict with the fields every consumer may rely on:

- kind: which constructor built it ("analytic_fixture", "callable",
  "checkpoint").
- status: "ready" when the arm can serve forces now, "unavailable" when
  a recorded refusal stands in the way.
- refusal_reason: None for a ready arm, a human-readable string when
  unavailable. The refusal is a RECORDED FIELD on the arm or on the
  raised error, never only an exception with no home.
- provides_energy: True only when a scalar energy in eV is available.
- species: the species encoding, a tuple of element symbols in the
  arm's fixed atom order.
- forces(positions, species=None, record_id=None): the force callable.
  Positions arrive unbatched (N, 3) for one molecule or batched
  (B, N, 3), in angstrom. Forces come back in the SAME shape, in
  eV/angstrom, and row i of the output belongs to row i of the input:
  the atom order is preserved by construction. An info dict is
  returned next to the forces and echoes the record identity, the
  species encoding and the atom_order_preserved flag.
- energy(positions, species=None, record_id=None): only meaningful for
  an energy-bearing arm; a direct-force arm refuses it.

Conventions this module fixes for the whole evaluator (the test
fixtures and the real archive records both agree with it, see
tests/analytic_fixtures.py and force_consistency.data.types.FrameRecord):

- A single molecule is an unbatched (N, 3) array. Batched input stays
  batched; the output shape always mirrors the input shape.
- Species are element-symbol tuples in the arm's atom order. A call
  that names a different species encoding is refused, not broadcast.
- A nonfinite force is a hard FloatingPointError carrying the refusal
  as a recorded info field. It is never silently zeroed or dropped.

Checkpoint stub wording (review round 2): the base repository DOES
carry checkpoint serialization and loading helpers —
`force_consistency.runs.save_checkpoint` and `load_checkpoint`, with the
msgpack format and provenance checks the runs package uses. What is
missing here is a USABLE checkpoint: no trained weights exist in this
export (runs/ is gitignored) and no export-specific weights format is
frozen, so arm_from_checkpoint NEVER opens a file and NEVER guesses a
reader: it returns an arm that reports an explicit unavailable status
with the recorded reason.
"""
from pathlib import Path

import numpy as np

UNITS = {"positions": "angstrom", "forces": "eV/angstrom", "energy": "eV"}


class ArmConfigurationError(ValueError):
    """An arm/start mismatch: configuration, not a model outcome.

    Species encoding, atom count and the start's own array shape live in
    the arm's construction dict and the start set. When they disagree, no
    force has been computed and no physics is in question, so the error is
    raised before any force call and propagates instead of being scored
    as a branch outcome. It stays a ValueError subclass so existing
    direct-callers' except ValueError clauses keep working.
    """


def _validated_positions(positions, arm_species, species, arm_name):
    """Check the arm/start compatibility before any physics.

    positions: unbatched (N, 3) or batched (B, N, 3), in angstrom.
    arm_species: the species tuple the arm was built with.
    species: the species tuple this call names, or None to use the
        arm's own.
    arm_name: which arm refused, for the error message.

    Every failure raised here is an ArmConfigurationError, because the
    mismatch lives in the assembled configuration (the arm was built for
    a different species tuple, atom count or rank than the start it was
    handed), not in any computed force. Raising it before the supplier
    runs is what keeps a misconfigured arm out of the scored outcomes.
    """
    array = np.asarray(positions, dtype=np.float64)
    if array.ndim == 2:
        # The unbatched (N, 3) form: N first, coordinates last.
        if array.shape[1] != 3:
            raise ArmConfigurationError(
                f"{arm_name}: positions must have shape (N, 3) or "
                f"(batch, N, 3), got {array.shape}")
        atoms = array.shape[0]
    elif array.ndim == 3:
        # The batched (B, N, 3) form: the atoms axis is the middle one.
        if array.shape[2] != 3:
            raise ArmConfigurationError(
                f"{arm_name}: positions must have shape (N, 3) or "
                f"(batch, N, 3), got {array.shape}")
        atoms = array.shape[1]
    else:
        raise ArmConfigurationError(
            f"{arm_name}: positions must have shape (N, 3) or (batch, N, 3), "
            f"got {array.shape}")
    if species is not None and tuple(species) != arm_species:
        raise ArmConfigurationError(
            f"{arm_name}: species mismatch: the arm encodes {arm_species}, "
            f"the call named {tuple(species)}")
    if atoms != len(arm_species):
        raise ArmConfigurationError(
            f"{arm_name}: shape mismatch: positions carry {atoms} atoms but "
            f"the arm encodes {len(arm_species)}: {arm_species}")
    return array


class NonfiniteValueError(FloatingPointError):
    """A nonfinite force or energy, carrying its refusal as a field.

    The refusal must be a RECORDED FIELD, not only an exception, so the
    error object itself carries the info dict for any caller that
    catches it. It is still a FloatingPointError for callers that only
    care about the type.
    """

    def __init__(self, message, info):
        super().__init__(message)
        self.info = info


def _require_finite_values(values, record_id, what):
    """Refuse any nonfinite value with a recorded field, never zero it."""
    if not np.isfinite(values).all():
        refusal = {
            "status": "unavailable",
            "reason": f"nonfinite {what}",
            "record_id": record_id,
            "policy": "the value is refused, never zeroed or dropped",
        }
        raise NonfiniteValueError(
            f"{what} from record {record_id!r} is not finite; the arm "
            f"reports it unavailable instead of zeroing it: {refusal}", refusal)


def _ok_info(record_id, arm_species, shape):
    return {
        "status": "ok",
        "record_id": record_id,
        "species": tuple(arm_species),
        "atom_order_preserved": True,
        "units": dict(UNITS),
        "shape": tuple(shape),
    }


def analytic_fixture_arm():
    """The analytic fixture arm: eight atoms in separable harmonic wells.

    Wraps the hand-computable fixture potential from the test support
    module, so the evaluator runs with known exact forces and energies
    without any trained model. The import is done here rather than at
    module level because the fixture potentials live beside the tests.
    """
    import analytic_fixtures as fixtures

    arm_species = tuple(fixtures.FIXTURE_SPECIES)
    force_fn = fixtures.fixture_molecule_potential()
    energy_and_forces_fn = fixtures.fixture_molecule_energy_and_forces()

    def forces(positions, species=None, record_id=None):
        array = _validated_positions(positions, arm_species, species,
                                     "analytic fixture arm")
        out = np.asarray(force_fn(array), dtype=np.float64)
        _require_finite_values(out, record_id, "force")
        return out, _ok_info(record_id, arm_species, out.shape)

    def energy(positions, species=None, record_id=None):
        array = _validated_positions(positions, arm_species, species,
                                     "analytic fixture arm")
        energy_fn = energy_and_forces_fn(array)[0]
        scalar = float(np.sum(energy_fn))
        _require_finite_values(np.array([scalar]), record_id, "energy")
        return scalar, _ok_info(record_id, arm_species, array.shape)

    def probe():
        # One displaced evaluation, batched: the probe reports the arm's
        # batched contract shape and a measured force magnitude.
        batched = (fixtures.FIXTURE_MINIMUM + 0.1)[np.newaxis, :, :]
        out = np.asarray(force_fn(batched), dtype=np.float64)
        return {
            "shape": tuple(batched.shape),
            "units": dict(UNITS),
            "force_order": "forces on the returned positions",
            "max_abs_force_component_eV_per_A": float(np.max(np.abs(out))),
        }

    return {
        "kind": "analytic_fixture",
        "status": "ready",
        "refusal_reason": None,
        "provides_energy": True,
        "arm_id": "analytic-fixture-8at-harmonic",
        "species": arm_species,
        "forces": forces,
        "energy": energy,
        "probe": probe,
    }


def callable_arm(force_fn, species, arm_id):
    """Wrap any plain force function as an arm.

    force_fn maps positions to forces in eV/angstrom. The wrapper
    refuses species mismatches and shape mismatches instead of
    broadcasting silently, so a wrongly shaped array can never be read
    as physics. It provides no energy: a direct-force arm is valid for
    shape recovery and refused for energy-based metrics, with the
    refusal recorded as a field.
    """
    arm_species = tuple(species)

    def forces(positions, species=None, record_id=None):
        array = _validated_positions(positions, arm_species, species,
                                     f"callable arm {arm_id!r}")
        raw = force_fn(array)
        try:
            out = np.asarray(raw, dtype=np.float64)
        except (TypeError, ValueError) as refused:
            # A return that cannot become a numeric array at all is a
            # refusal with its own recorded reason, not a crash in the
            # accounting: the refusal text names the fact so the audit
            # trail says WHY the attempt was refused.
            raise ValueError(
                f"callable arm {arm_id!r}: nonnumeric force return "
                f"({refused})") from refused
        if out.shape != array.shape:
            raise ValueError(
                f"callable arm {arm_id!r}: force shape mismatch: the force "
                f"function returned {out.shape} for positions {array.shape}")
        _require_finite_values(out, record_id, "force")
        return out, _ok_info(record_id, arm_species, out.shape)

    def energy(positions, species=None, record_id=None):
        raise LookupError(_energy_refusal(arm_id))

    return {
        "kind": "callable",
        "status": "ready",
        "refusal_reason": None,
        "provides_energy": False,
        "arm_id": str(arm_id),
        "species": arm_species,
        "forces": forces,
        "energy": energy,
        "force_fn": force_fn,
    }


def _energy_refusal(arm_id):
    """The recorded refusal text for energy metrics on a force-only arm."""
    return (f"arm {arm_id!r} provides no energy: shape recovery needs forces "
            f"only and stays valid, but energy-based metrics are refused for "
            f"this arm rather than approximated or zeroed")


def accepted_for_shape_recovery(arm):
    """Shape recovery needs forces only: any ready arm is accepted."""
    return arm["status"] == "ready"


def accepted_for_energy_metrics(arm):
    """(allowed, refusal) for energy-based metrics on this arm.

    The refusal is a recorded string the caller can put in its audit
    trail, never a bare exception with no home.
    """
    if arm["provides_energy"]:
        return True, None
    return False, _energy_refusal(arm.get("arm_id", "<unnamed>"))


def arm_from_checkpoint(path, contract):
    """Bind a checkpoint path into an arm, without guessing any format.

    The base repository has checkpoint serialization and loading
    helpers (force_consistency.runs.save_checkpoint / load_checkpoint),
    but no USABLE checkpoint is supplied in this export: runs/ is
    gitignored and no trained weights ship here, and this adapter binds
    no weights reader of its own. This constructor therefore never
    opens the file and never invents a reader. The returned arm records
    an explicit unavailable status and refusal reason: one when no
    checkpoint is present or reachable, the other when a path exists
    but no reader is bound here to interpret it. Both refusals raise
    LookupError if a consumer still asks the arm for physics.
    """
    path = Path(path)
    if not path.exists():
        refusal = (f"checkpoint not found at {path}: no trained checkpoint is "
                   f"present or reachable in the repository")
    else:
        refusal = (f"checkpoint at {path} is unusable: no frozen weights "
                   f"format exists, so the adapter refuses to interpret the "
                   f"file rather than guessing one")

    def forces(positions, species=None, record_id=None):
        raise LookupError(refusal)

    def energy(positions, species=None, record_id=None):
        raise LookupError(refusal)

    return {
        "kind": "checkpoint",
        "status": "unavailable",
        "refusal_reason": refusal,
        "provides_energy": False,
        "arm_id": f"checkpoint-{path.name}",
        "species": tuple(contract["species"]) if contract and "species" in contract else None,
        "contract": dict(contract) if contract else {},
        "forces": forces,
        "energy": energy,
    }