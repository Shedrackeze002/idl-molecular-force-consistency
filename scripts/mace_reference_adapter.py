"""Bridge an already constructed ASE calculator into the frozen evaluator.

This module does not load or admit checkpoints. Caller-supplied provenance
is preserved as a tag, never promoted to verified model identity. Use the
separate MACE environment for a future admitted MACE calculator.
"""
from copy import deepcopy
import json

import numpy as np
from ase import Atoms

from evaluation.adapter import (ArmConfigurationError, ForceShapeMismatch,
                                MalformedForceReturn, _ok_info,
                                _require_finite_values, _validated_positions)


def calculator_arm(calculator, species, *, arm_id, checkpoint_identity):
    """Preserve ASE angstrom/eV conventions and the supplied energy gauge.

    Each structure uses fresh nonperiodic Atoms, in the fixed species order.
    Force requests never explicitly request energy, although a calculator
    may compute it internally or reuse cached values. Request counts must
    therefore not be reported as model-forward counts.
    """
    try:
        arm_species = tuple(species)
        if not arm_species or any(not isinstance(s, str) for s in arm_species):
            raise ValueError('Nonempty element-symbol sequence required')
        atoms = Atoms(symbols=arm_species)  # No calculation.
        if np.any(atoms.numbers <= 0):
            raise ValueError('Real element symbols required; dummy atoms refused')
        if not isinstance(arm_id, str) or not arm_id.strip():
            raise ValueError('Nonempty arm_id required')
        if not isinstance(checkpoint_identity, dict) or not checkpoint_identity:
            raise ValueError('Nonempty caller provenance dict required')
        json.dumps(checkpoint_identity, allow_nan=False)
        identity = deepcopy(checkpoint_identity)
        if not {'forces', 'energy'} <= set(calculator.implemented_properties):
            raise ValueError('Calculator must support forces and energy')
    except (TypeError, ValueError, KeyError, AttributeError) as exc:
        raise ArmConfigurationError(str(exc)) from exc

    def positions_for_call(positions, species):
        try:
            array = _validated_positions(positions, arm_species, species, arm_id)
        except (TypeError, ValueError) as exc:
            raise ArmConfigurationError(str(exc)) from exc
        if array.ndim == 3 and array.shape[0] == 0:
            raise ArmConfigurationError('Empty position batch')
        if not np.isfinite(array).all():
            raise ArmConfigurationError('Positions must be finite')
        return array

    def atoms_for_call(array):
        return Atoms(symbols=arm_species, positions=array, pbc=False,
                     calculator=calculator)

    def info_for_call(record_id, shape):
        info = _ok_info(record_id, arm_species, shape)
        info['checkpoint_identity'] = deepcopy(identity)
        info['checkpoint_identity_verified'] = False
        return info

    def forces(positions, species=None, record_id=None):
        array = positions_for_call(positions, species)
        frames = array[None] if array.ndim == 2 else array
        result = []
        for frame in frames:
            raw = atoms_for_call(frame).get_forces(apply_constraint=False)
            try:
                value = np.asarray(raw, dtype=np.float64)
            except (TypeError, ValueError) as exc:
                raise MalformedForceReturn('Calculator returned nonnumeric forces') from exc
            if value.shape != frame.shape:
                raise ForceShapeMismatch('Calculator force shape differs from positions')
            _require_finite_values(value, record_id, 'force')
            result.append(value.copy())
        values = np.stack(result)
        output = values[0] if array.ndim == 2 else values
        return output, info_for_call(record_id, output.shape)

    def energy(positions, species=None, record_id=None):
        array = positions_for_call(positions, species)
        if array.ndim != 2:
            raise ArmConfigurationError('Energy requires one unbatched molecule')
        raw = atoms_for_call(array).get_potential_energy()
        value = np.asarray(raw, dtype=np.float64)
        if value.ndim != 0:
            raise ValueError('Calculator energy must be scalar')
        _require_finite_values(value, record_id, 'energy')
        return float(value), info_for_call(record_id, array.shape)

    return dict(kind='callable', status='ready', refusal_reason=None,
                provides_energy=True, arm_id=arm_id, species=arm_species,
                forces=forces, energy=energy,
                checkpoint_identity=deepcopy(identity),
                checkpoint_identity_verified=False)
