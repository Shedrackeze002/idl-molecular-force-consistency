"""Measurement-only evidence from the A0 replay (plan v3 section 5 item 6, decision D10 point 3).

The gate of A0 is the frozen `adapter_replay`, unchanged. This module adds numbers and nothing else: it never
decides, never gates and never raises on a difference. For one restored arm it records

- batch 4: the 974 saved validation forces compared with the arm's forces in the saved shape (batches of four,
  the last of two): the largest absolute difference with its location, the largest scaled residual
  `|r - s| / (atol + rtol |s|)` (at most 1 means inside the frozen tolerance), the number of records equal to the
  saved forces bit for bit, and the RMS difference over the RMS saved force;
- batch 1: the same over all 974 records, one (N, 3) call each, through `evaluation.runner._arm_force_supplier`,
  the callable every relaxation uses, so the start geometries (which are validation records) are covered in the
  shape that produces every number of the study;
- the largest difference between the arm's own batch-1 and batch-4 forces;
- the batch-1 latency of each call: count, mean, median, p99, maximum.

The batch-1 latency is used by the post-A0 guard only; every other value triggers no action. Every failure inside
the measurement is recorded as an `error` string and never replaces or masks the replay result; the one exception
is the supervisor's deadline, which must still end the process.
"""
import time

import numpy as np

from stage2_capped import admission
from stage2_capped.driver import DeadlineReached


def scaled_residual(forces, saved, rtol, atol):
    """The largest `|r - s| / (atol + rtol |s|)`; at most 1 means every element is inside np.allclose."""
    return float(np.max(np.abs(forces - saved) / (atol + rtol * np.abs(saved))))


def comparison(forces, saved, rtol, atol):
    difference = np.abs(forces - saved)
    index = np.unravel_index(int(np.argmax(difference)), difference.shape)
    saved_rms = float(np.sqrt(np.mean(saved ** 2)))
    return {"max_abs_difference": float(difference[index]),
            "max_abs_location": {"record_index": int(index[0]), "atom": int(index[1]), "component": int(index[2])},
            "max_scaled_residual": scaled_residual(forces, saved, rtol, atol),
            "bit_identical_records": int(sum(np.array_equal(forces[i], saved[i]) for i in range(len(saved)))),
            "records": int(len(saved)),
            "rms_difference_over_rms_saved": float(np.sqrt(np.mean((forces - saved) ** 2)) / saved_rms) if saved_rms else None}


def latency_summary(seconds):
    values = np.asarray(seconds, dtype=np.float64)
    return {"count": int(values.size), "mean_seconds": float(values.mean()), "p50_seconds": float(np.percentile(values, 50)),
            "p99_seconds": float(np.percentile(values, 99)), "max_seconds": float(values.max())}


def _recorded(function):
    """Run one measurement; a failure becomes an `error` string, except the supervisor's deadline."""
    try:
        return function()
    except DeadlineReached:
        raise
    except Exception as error:  # noqa: BLE001 - evidence never fails the launch
        return {"error": f"{type(error).__name__}: {error}"}


def collect(arm, positions, saved, *, rtol=admission.REPLAY_TOLERANCE, atol=admission.REPLAY_TOLERANCE,
            batch=admission.REPLAY_BATCH, clock=time.perf_counter):
    """Evidence for one arm over `positions` (n, N, 3) against the `saved` forces (n, N, 3)."""
    from evaluation.runner import _arm_force_supplier
    positions, saved = np.asarray(positions, dtype=np.float64), np.asarray(saved, dtype=np.float64)
    evidence = {"saved_force_abs_max": float(np.max(np.abs(saved))) if saved.size else None}
    arrays = {}

    def batch_four():
        forces = np.empty_like(saved)
        for start in range(0, len(saved), batch):
            out, _ = arm["forces"](positions[start:start + batch], species=arm["species"], record_id="replay-evidence")
            forces[start:start + batch] = out
        arrays["batch4"] = forces
        return comparison(forces, saved, rtol, atol)

    def batch_one():
        supplier = _arm_force_supplier(arm, arm["species"], "replay-evidence")
        forces, seconds = np.empty_like(saved), []
        for index in range(len(saved)):
            started = clock()
            result = supplier(positions[index])
            seconds.append(clock() - started)
            forces[index] = result
        arrays["batch1"] = forces
        evidence["batch1_latency"] = latency_summary(seconds)
        return comparison(forces, saved, rtol, atol)

    evidence["batch4"] = _recorded(batch_four)
    evidence["batch1"] = _recorded(batch_one)
    if "batch1_latency" not in evidence:
        evidence["batch1_latency"] = {"error": "no batch-1 latency was measured"}
    if "batch4" in arrays and "batch1" in arrays:
        evidence["batch1_vs_batch4_max_abs"] = float(np.max(np.abs(arrays["batch1"] - arrays["batch4"])))
    return evidence
