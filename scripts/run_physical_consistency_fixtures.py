"""Generate the reviewed analytic-fixture physical-consistency record."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

# Keep direct ``python scripts/...`` execution reproducible without requiring
# the repository to be installed first.
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from evaluation.adapter import callable_arm, energy_force_arm
from evaluation.physical_consistency import (
    PhysicalConsistencyConfig,
    benchmark_inference,
    closed_loop_work,
    evaluate_physical_consistency,
)


FIXTURE_MINIMUM = np.array([
    [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 1.4, 0.0], [0.0, -1.4, 0.0],
    [0.0, 0.0, 1.5], [0.0, 0.0, -1.5], [0.0, 0.0, 0.6], [0.0, 0.0, 1.9],
], dtype=np.float64)
FIXTURE_SPECIES = ("C", "C", "O", "O", "H", "H", "H", "H")


def harmonic(positions):
    positions = np.asarray(positions, dtype=np.float64)
    energy = 0.5 * np.sum((positions - FIXTURE_MINIMUM) ** 2, axis=(-2, -1))
    return energy, FIXTURE_MINIMUM - positions


def square(target):
    offsets = np.array([
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0], [1.0, 0.0, 0.0],
    ])
    return np.stack([target + offset for offset in offsets])


def rotational(positions):
    positions = np.asarray(positions, dtype=np.float64)
    output = np.zeros_like(positions)
    output[..., 0] = -positions[..., 1]
    output[..., 1] = positions[..., 0]
    return output


def nonlinear_conservative(positions):
    """Force from E=x^3*y; useful for exposing coarse quadrature error."""
    positions = np.asarray(positions, dtype=np.float64)
    x = positions[..., 0]
    y = positions[..., 1]
    output = np.zeros_like(positions)
    output[..., 0] = -3.0 * x * x * y
    output[..., 1] = -(x ** 3)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    conservative = energy_force_arm(
        harmonic, species=FIXTURE_SPECIES, arm_id="analytic-harmonic")
    nonconservative = callable_arm(rotational, species=("C",), arm_id="rotational-force-only")
    nonlinear = callable_arm(
        nonlinear_conservative,
        species=("C",),
        arm_id="nonlinear-conservative-force-only",
    )
    rotational_loop = np.array([
        [[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]],
        [[-1.0, 0.0, 0.0]], [[0.0, -1.0, 0.0]],
        [[1.0, 0.0, 0.0]],
    ])
    nonlinear_loop = np.array([
        [[0.0, 0.0, 0.0]], [[1.0, 0.0, 0.0]],
        [[1.0, 1.0, 0.0]], [[0.0, 1.0, 0.0]],
        [[0.0, 0.0, 0.0]],
    ])
    report = {
        "purpose": "analytic fixture validation; not a trained-model result",
        "conservative": evaluate_physical_consistency(
            conservative,
            geometries={"displaced": FIXTURE_MINIMUM + 0.1},
            loops={"translated-square": square(FIXTURE_MINIMUM)},
        ),
        "nonconservative": evaluate_physical_consistency(
            nonconservative,
            geometries={"diamond-start": rotational_loop[0]},
            loops={"diamond": rotational_loop},
        ),
        "quadrature_refinement_control": {
            "purpose": ("demonstrate that coarse trapezoidal work can be nonzero "
                        "for a nonlinear conservative force field"),
            "coarse": closed_loop_work(
                nonlinear, nonlinear_loop, loop_id="unit-square-coarse"),
            "refined": closed_loop_work(
                nonlinear,
                nonlinear_loop,
                loop_id="unit-square-refined",
                config=PhysicalConsistencyConfig(loop_quadrature_subdivisions=8),
            ),
        },
        "timing_smoke": benchmark_inference(
            conservative, (FIXTURE_MINIMUM + 0.1)[None, ...],
            warmup_calls=3, measured_calls=20),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
