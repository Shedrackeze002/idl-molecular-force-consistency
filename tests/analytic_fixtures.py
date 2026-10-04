"""Analytic fixture potentials with known answers.

Used by the relaxation tests, the adapter tests and the reporter's
end-to-end slice. Everything is hand-computable: gradients (and so forces)
are exact, minima are exact, and nothing is stochastic.
"""
import numpy as np

# The exact minima the analytic fixtures converge to.
HARMONIC_MINIMUM = np.array([1.0, -0.5, 0.25], dtype=np.float64)
DOUBLE_WELL_MINIMA = [np.array([1.0, 0.0, 0.0], dtype=np.float64),
                      np.array([-1.0, 0.0, 0.0], dtype=np.float64)]

# A separable fixture molecule: eight atoms, each pulled toward its own
# target point by a harmonic well in every component. The minimum of the
# whole molecule is the exact target array, every force is exact, and the
# maximum force component at the start is easy to compute by hand. All rows
# are distinct, so the target array is a genuine nondegenerate structure.
FIXTURE_SPECIES = ("C", "C", "O", "O", "H", "H", "H", "H")
FIXTURE_MINIMUM = np.array([
    [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.4, 0.0], [0.0, -1.4, 0.0],
    [0.0, 0.0, 1.5], [0.0, 0.0, -1.5], [0.0, 0.0, 0.6], [0.0, 0.0, 1.9],
], dtype=np.float64)
FIXTURE_WELL_STRENGTH = 1.0  # eV per angstrom^2, per component per atom

HARMONIC_WELL_STRENGTH = 1.0
DOUBLE_WELL_CURVATURE = 1.0
DOUBLE_WELL_BARRIER_HEIGHT = 1.0


def harmonic_energy_and_forces(positions):
    """V = k/2 * sum((x - m)^2) on single-atom frames.

    Positions arrive as (B, N, 3); N is 1 for this fixture. The energy per
    frame is k/2 * |x - m|^2, the force on the atom is -grad V = k*(m - x).
    """
    positions = np.asarray(positions, dtype=np.float64)
    energy = 0.5 * HARMONIC_WELL_STRENGTH * np.sum(
        (positions - HARMONIC_MINIMUM) ** 2, axis=(-2, -1))
    forces = HARMONIC_WELL_STRENGTH * (HARMONIC_MINIMUM - positions)
    return energy, forces


def harmonic_force(positions):
    return harmonic_energy_and_forces(positions)[1]


def double_well_energy_and_forces(positions):
    """A one-dimensional double well per single-atom frame.

    V = k/4 * (x^2 - 1)^2 + k*b/2 * (y^2 + z^2), where x, y, z are the atom
    coordinates. Wells at (+1, 0, 0) and (-1, 0, 0), a barrier of height
    k/4 at the origin, and harmonic walls in y and z. The force is the
    exact negative gradient componentwise.
    """
    positions = np.asarray(positions, dtype=np.float64)
    x = positions[..., 0]
    y = positions[..., 1]
    z = positions[..., 2]
    energy = (DOUBLE_WELL_CURVATURE / 4.0) * (x * x - 1.0) ** 2 + (
        DOUBLE_WELL_CURVATURE * DOUBLE_WELL_BARRIER_HEIGHT / 2.0) * (y * y + z * z)
    fx = -DOUBLE_WELL_CURVATURE * x * (x * x - 1.0)
    fy = -DOUBLE_WELL_CURVATURE * DOUBLE_WELL_BARRIER_HEIGHT * y
    fz = -DOUBLE_WELL_CURVATURE * DOUBLE_WELL_BARRIER_HEIGHT * z
    forces = np.stack([fx, fy, fz], axis=-1)
    return energy, forces


def double_well_force(positions):
    return double_well_energy_and_forces(positions)[1]


def fixture_molecule_potential(offset=None):
    """Force function for the separable fixture molecule.

    Every atom feels a harmonic well with strength FIXTURE_WELL_STRENGTH in
    every component, centered on its row of FIXTURE_MINIMUM plus a shared
    (optional) offset. The force on atom i is
        k * (m_i + offset - x_i),
    so the exact fixed point is FIXTURE_MINIMUM + offset.
    """
    target = FIXTURE_MINIMUM + (np.zeros(3) if offset is None else np.asarray(offset, dtype=np.float64))

    def force(positions):
        positions = np.asarray(positions, dtype=np.float64)
        return FIXTURE_WELL_STRENGTH * (target - positions)

    return force


def fixture_molecule_energy_and_forces(offset=None):
    """Paired energy and force for the separable fixture molecule.

    The energy is the exact potential of the force function above, so an
    energy-bearing arm can be tested without any trained model.
    """
    target = FIXTURE_MINIMUM + (np.zeros(3) if offset is None else np.asarray(offset, dtype=np.float64))
    strength = FIXTURE_WELL_STRENGTH

    def energy_and_forces(positions):
        positions = np.asarray(positions, dtype=np.float64)
        energy = np.sum(0.5 * strength * (target - positions) ** 2, axis=(-2, -1))
        return energy, strength * (target - positions)

    return energy_and_forces


def constant_nonzero_force(value=0.05):
    """A constant nonzero force field: no fixed point exists anywhere."""
    array = np.full((3,), float(value), dtype=np.float64)

    def force(positions):
        positions = np.asarray(positions, dtype=np.float64)
        return np.array(np.broadcast_to(array, positions.shape), dtype=np.float64, copy=True)

    return force