"""Physical-consistency and inference-cost evaluation for model arms.

The evaluator consumes the shared arm contract from :mod:`evaluation.adapter`.
It deliberately keeps two different questions separate:

* energy/force disagreement compares a supplied force with ``-grad(E)``;
* closed-loop work tests a force field directly and therefore also applies to
  force-only arms.

Neither sampled test proves global conservativity.  Results retain units,
evaluation counts, configuration, and explicit unavailable/failure states so
that missing energy is never silently replaced with zero.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from numbers import Integral
import platform
import statistics
import sys
import time
from typing import Callable, Mapping, Sequence

import numpy as np

SCHEMA_VERSION = "physical-consistency/v1"
UNITS = {
    "positions": "angstrom",
    "forces": "eV/angstrom",
    "energy": "eV",
    "loop_work": "eV",
    "force_disagreement": "eV/angstrom",
    "force_disagreement_squared": "eV^2/angstrom^2",
    "time": "second",
}


@dataclass(frozen=True)
class PhysicalConsistencyConfig:
    """Frozen numerical choices for a physical-consistency evaluation."""

    finite_difference_step_A: float = 1e-5
    loop_closure_tolerance_A: float = 1e-12

    def __post_init__(self):
        if (not math.isfinite(self.finite_difference_step_A)
                or self.finite_difference_step_A <= 0):
            raise ValueError("finite_difference_step_A must be finite and positive")
        if (not math.isfinite(self.loop_closure_tolerance_A)
                or self.loop_closure_tolerance_A < 0):
            raise ValueError("loop_closure_tolerance_A must be finite and nonnegative")


def _positions(value, *, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[-1] != 3 or array.shape[0] < 1:
        raise ValueError(f"{name} must have shape (N, 3), got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains nonfinite positions")
    return array


def _failed(metric: str, error: Exception) -> dict:
    info = getattr(error, "info", None)
    return {
        "metric": metric,
        "status": "failed",
        "reason": str(error),
        "exception_type": type(error).__name__,
        "refusal": dict(info) if isinstance(info, dict) else None,
        "units": dict(UNITS),
    }


def energy_force_disagreement(
    arm: Mapping,
    positions,
    *,
    record_id: str | None = None,
    config: PhysicalConsistencyConfig | None = None,
) -> dict:
    """Compare direct forces with a central-difference ``-grad(E)``.

    A force-only arm returns ``status='unavailable'``.  A supplier failure is
    recorded as ``status='failed'`` with the exception type and any structured
    refusal carried by the adapter.
    """
    config = config or PhysicalConsistencyConfig()
    point = _positions(positions, name="positions")
    if arm.get("status") != "ready":
        return {
            "metric": "energy_force_disagreement",
            "status": "unavailable",
            "reason": arm.get("refusal_reason") or "arm is not ready",
            "record_id": record_id,
            "units": dict(UNITS),
        }
    if not arm.get("provides_energy", False):
        return {
            "metric": "energy_force_disagreement",
            "status": "unavailable",
            "reason": (f"arm {arm.get('arm_id', '<unnamed>')!r} provides no energy; "
                       "energy-gradient comparison was not fabricated"),
            "record_id": record_id,
            "units": dict(UNITS),
        }

    step = config.finite_difference_step_A
    try:
        direct, _ = arm["forces"](point, record_id=record_id)
        direct = np.asarray(direct, dtype=np.float64)
        derived = np.empty_like(point)
        for atom in range(point.shape[0]):
            for component in range(3):
                plus = point.copy()
                minus = point.copy()
                plus[atom, component] += step
                minus[atom, component] -= step
                e_plus, _ = arm["energy"](plus, record_id=record_id)
                e_minus, _ = arm["energy"](minus, record_id=record_id)
                derived[atom, component] = -(float(e_plus) - float(e_minus)) / (2 * step)
        if direct.shape != point.shape or not np.isfinite(direct).all():
            raise ValueError("arm returned an invalid direct-force array")
        if not np.isfinite(derived).all():
            raise FloatingPointError("finite-difference force is nonfinite")
    except Exception as error:  # supplier errors become auditable metric results
        result = _failed("energy_force_disagreement", error)
        result["record_id"] = record_id
        return result

    residual = direct - derived
    absolute = np.abs(residual)
    squared = residual * residual
    return {
        "metric": "energy_force_disagreement",
        "status": "ok",
        "record_id": record_id,
        "atom_count": int(point.shape[0]),
        "component_count": int(point.size),
        "finite_difference_step_A": step,
        "force_evaluation_count": 1,
        "energy_evaluation_count": int(2 * point.size),
        "mae_eV_per_A": float(np.mean(absolute)),
        "rmse_eV_per_A": float(np.sqrt(np.mean(squared))),
        "max_abs_eV_per_A": float(np.max(absolute)),
        "mse_eV2_per_A2": float(np.mean(squared)),
        "units": dict(UNITS),
    }


def closed_loop_work(
    arm: Mapping,
    loop_positions,
    *,
    loop_id: str | None = None,
    config: PhysicalConsistencyConfig | None = None,
) -> dict:
    """Integrate ``F dot dr`` around an explicitly closed polygonal loop.

    The trapezoidal rule is used on every segment.  The first geometry must be
    repeated as the last geometry; the evaluator refuses to close a path
    silently because doing so would change the declared experiment.
    """
    config = config or PhysicalConsistencyConfig()
    loop = np.asarray(loop_positions, dtype=np.float64)
    if loop.ndim != 3 or loop.shape[-1] != 3 or loop.shape[0] < 4:
        raise ValueError("loop_positions must have shape (vertices>=4, N, 3)")
    if not np.isfinite(loop).all():
        raise ValueError("loop_positions contains nonfinite values")
    closure_error = float(np.max(np.abs(loop[0] - loop[-1])))
    if closure_error > config.loop_closure_tolerance_A:
        raise ValueError(
            f"loop is open by {closure_error} A; repeat the first vertex explicitly")
    if arm.get("status") != "ready":
        return {
            "metric": "closed_loop_work",
            "status": "unavailable",
            "reason": arm.get("refusal_reason") or "arm is not ready",
            "loop_id": loop_id,
            "units": dict(UNITS),
        }
    try:
        forces, _ = arm["forces"](loop, record_id=loop_id)
        forces = np.asarray(forces, dtype=np.float64)
        if forces.shape != loop.shape or not np.isfinite(forces).all():
            raise ValueError("arm returned an invalid force array for the loop")
    except Exception as error:
        result = _failed("closed_loop_work", error)
        result["loop_id"] = loop_id
        return result

    displacements = np.diff(loop, axis=0)
    segment_forces = 0.5 * (forces[:-1] + forces[1:])
    segment_work = np.sum(segment_forces * displacements, axis=(1, 2))
    work = float(np.sum(segment_work))
    return {
        "metric": "closed_loop_work",
        "status": "ok",
        "loop_id": loop_id,
        "atom_count": int(loop.shape[1]),
        "vertex_count": int(loop.shape[0]),
        "segment_count": int(loop.shape[0] - 1),
        "force_call_count": 1,
        "force_configuration_count": int(loop.shape[0]),
        "closure_error_A": closure_error,
        "signed_work_eV": work,
        "absolute_work_eV": abs(work),
        "segment_work_eV": [float(value) for value in segment_work],
        "units": dict(UNITS),
    }


def evaluate_physical_consistency(
    arm: Mapping,
    *,
    geometries: Mapping[str, np.ndarray] | None = None,
    loops: Mapping[str, np.ndarray] | None = None,
    config: PhysicalConsistencyConfig | None = None,
) -> dict:
    """Evaluate named geometries and loops and return a JSON-safe record."""
    config = config or PhysicalConsistencyConfig()
    geometry_results = [
        energy_force_disagreement(arm, value, record_id=name, config=config)
        for name, value in sorted((geometries or {}).items())
    ]
    loop_results = [
        closed_loop_work(arm, value, loop_id=name, config=config)
        for name, value in sorted((loops or {}).items())
    ]
    results = geometry_results + loop_results
    counts = {status: sum(item["status"] == status for item in results)
              for status in ("ok", "unavailable", "failed")}
    return {
        "schema_version": SCHEMA_VERSION,
        "arm": {
            "arm_id": arm.get("arm_id"),
            "kind": arm.get("kind"),
            "status": arm.get("status"),
            "provides_energy": bool(arm.get("provides_energy", False)),
            "species": list(arm.get("species") or ()),
        },
        "config": asdict(config),
        "units": dict(UNITS),
        "summary": {"result_count": len(results), **counts},
        "energy_force_disagreement": geometry_results,
        "closed_loop_work": loop_results,
        "claim_limit": ("Sampled geometries and loops can expose local inconsistency; "
                        "they do not prove global conservativity."),
    }


def _default_synchronize(value) -> None:
    method = getattr(value, "block_until_ready", None)
    if callable(method):
        method()


def benchmark_inference(
    arm: Mapping,
    positions,
    *,
    warmup_calls: int = 3,
    measured_calls: int = 20,
    synchronize: Callable[[object], None] | None = None,
) -> dict:
    """Measure force-inference wall time with warmup and synchronization.

    The caller may pass a backend-specific synchronization hook.  Warmup time
    is reported separately and never mixed into the measured distribution.
    """
    batch = np.asarray(positions, dtype=np.float64)
    if batch.ndim == 2:
        batch = batch[np.newaxis, ...]
    if batch.ndim != 3 or batch.shape[-1] != 3 or not np.isfinite(batch).all():
        raise ValueError("positions must have shape (N, 3) or (B, N, 3) and be finite")
    if (isinstance(warmup_calls, bool) or not isinstance(warmup_calls, Integral)
            or warmup_calls < 0):
        raise ValueError("warmup_calls must be a nonnegative integer")
    if (isinstance(measured_calls, bool) or not isinstance(measured_calls, Integral)
            or measured_calls < 1):
        raise ValueError("measured_calls must be a positive integer")
    if arm.get("status") != "ready":
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "unavailable",
            "reason": arm.get("refusal_reason") or "arm is not ready",
            "arm_id": arm.get("arm_id"),
            "units": dict(UNITS),
        }
    sync = synchronize or _default_synchronize

    warmup_started = time.perf_counter()
    try:
        for index in range(warmup_calls):
            output, _ = arm["forces"](batch, record_id=f"benchmark-warmup-{index}")
            sync(output)
        warmup_seconds = time.perf_counter() - warmup_started
        samples = []
        for index in range(measured_calls):
            started = time.perf_counter()
            output, _ = arm["forces"](batch, record_id=f"benchmark-measured-{index}")
            sync(output)
            samples.append(time.perf_counter() - started)
    except Exception as error:
        result = _failed("inference_timing", error)
        result.update(schema_version=SCHEMA_VERSION, arm_id=arm.get("arm_id"))
        return result

    ordered = sorted(samples)
    p95_index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    structures = int(batch.shape[0])
    median = float(statistics.median(samples))
    return {
        "schema_version": SCHEMA_VERSION,
        "metric": "inference_timing",
        "status": "ok",
        "arm_id": arm.get("arm_id"),
        "warmup_calls": int(warmup_calls),
        "warmup_seconds": float(warmup_seconds),
        "measured_calls": int(measured_calls),
        "structures_per_call": structures,
        "atom_count": int(batch.shape[1]),
        "synchronization": ("caller hook" if synchronize is not None
                            else "block_until_ready when exposed"),
        "mean_seconds_per_call": float(statistics.fmean(samples)),
        "median_seconds_per_call": median,
        "min_seconds_per_call": float(min(samples)),
        "p95_seconds_per_call": float(ordered[p95_index]),
        "median_seconds_per_structure": median / structures,
        "sample_seconds": [float(value) for value in samples],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "processor": platform.processor() or None,
        },
        "units": dict(UNITS),
        "claim_limit": ("Compare arms only with identical hardware, precision, batch, "
                        "warmup, synchronization, and software environment."),
    }
