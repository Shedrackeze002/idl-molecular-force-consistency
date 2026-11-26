"""Adapter for suppliers that return paired molecular energies and forces."""

import numpy as np

from .adapter import (
    ForceShapeMismatch,
    _ok_info,
    _require_finite_values,
    _validated_positions,
)


def energy_force_arm(energy_force_fn, species, arm_id):
    """Wrap a callable that supplies paired scalar energies and forces.

    ``energy_force_fn`` receives positions shaped ``(N, 3)`` or
    ``(B, N, 3)``.  It returns one energy per configuration in eV and forces
    mirroring the input shape in eV/angstrom.  This small adapter is useful for
    analytic controls and for real model adapters that already expose a paired
    prediction call.
    """
    arm_species = tuple(species)

    def evaluate(positions, species=None, record_id=None):
        array = _validated_positions(
            positions, arm_species, species, f"energy-force arm {arm_id!r}")
        raw_energy, raw_forces = energy_force_fn(array)
        energy_values = np.asarray(raw_energy, dtype=np.float64)
        force_values = np.asarray(raw_forces, dtype=np.float64)
        expected_energy_shape = () if array.ndim == 2 else (array.shape[0],)
        if energy_values.shape != expected_energy_shape:
            raise ValueError(
                f"energy-force arm {arm_id!r}: energy shape mismatch: returned "
                f"{energy_values.shape}, expected {expected_energy_shape}")
        if force_values.shape != array.shape:
            raise ForceShapeMismatch(
                f"energy-force arm {arm_id!r}: force shape mismatch: returned "
                f"{force_values.shape}, expected {array.shape}")
        _require_finite_values(energy_values, record_id, "energy")
        _require_finite_values(force_values, record_id, "force")
        return energy_values, force_values, array

    def forces(positions, species=None, record_id=None):
        _, values, array = evaluate(positions, species, record_id)
        return values, _ok_info(record_id, arm_species, array.shape)

    def energy(positions, species=None, record_id=None):
        values, _, array = evaluate(positions, species, record_id)
        if array.ndim != 2:
            raise ValueError("energy() requires one unbatched configuration")
        return float(values), _ok_info(record_id, arm_species, array.shape)

    return {
        "kind": "energy_force_callable",
        "status": "ready",
        "refusal_reason": None,
        "provides_energy": True,
        "arm_id": str(arm_id),
        "species": arm_species,
        "forces": forces,
        "energy": energy,
        "energy_force_fn": energy_force_fn,
    }
