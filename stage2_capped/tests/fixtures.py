"""Synthetic force fields and helpers shared by the Stage 2 tests (no models)."""
import struct

import numpy as np

from evaluation.adapter import (ArmConfigurationError, ForceShapeMismatch,
                                MalformedForceReturn, NonfiniteValueError)

PROCEDURE = {"max_force_evaluations": 1500, "initial_step_size": 0.2,
             "backtracking_factor": 0.5, "min_backtracking_floor": 1e-8,
             "convergence_tolerance_eV_per_A": 0.01,
             "max_backtracking_attempts_per_step": 40}


def procedure(**changes):
    return {**PROCEDURE, **changes}


def molecule(n_atoms=6, seed=0, spread=1.5):
    rng = np.random.default_rng(seed)
    return rng.normal(scale=spread, size=(n_atoms, 3))


def harmonic(target, stiffness):
    """F = -k (x - target). s*k > 2 overshoots, so steps get rejected."""
    def force(positions):
        return -stiffness * (np.asarray(positions) - target)
    return force


def anisotropic(target, stiffness_x, stiffness_rest):
    """Stiffer along x than along y, z (a narrow valley)."""
    weights = np.array([stiffness_x, stiffness_rest, stiffness_rest])

    def force(positions):
        return -weights * (np.asarray(positions) - target)
    return force


def pair_potential(strength=4.0):
    """Anharmonic pairwise forces: shared minimum at unit spacing."""
    def force(positions):
        x = np.asarray(positions)[0]
        out = np.zeros_like(x)
        for i in range(len(x)):
            for j in range(len(x)):
                if i == j:
                    continue
                d = x[i] - x[j]
                r = np.linalg.norm(d)
                out[i] += -strength * (r - 1.0) * d / r
        return out[None]
    return force


def constant(value):
    def force(positions):
        out = np.zeros_like(np.asarray(positions))
        out[..., 0] = value
        return out
    return force


def opposing(positions_shape_value=1.0):
    """Force that flips sign with every call: every trial looks like overshoot."""
    state = {"sign": 1.0}

    def force(positions):
        state["sign"] = -state["sign"]
        out = np.zeros_like(np.asarray(positions))
        out[..., 0] = state["sign"] * positions_shape_value
        return out
    return force


def returns(value):
    return lambda positions: value


def raises(exception):
    def force(positions):
        raise exception
    return force


def flaky(base, bad_call, bad):
    """Behaves as `base` except on call number `bad_call` (1-based)."""
    state = {"n": 0}

    def force(positions):
        state["n"] += 1
        if state["n"] == bad_call:
            return bad(positions)
        return base(positions)
    return force


def nan_return(positions):
    out = np.zeros_like(np.asarray(positions))
    out[..., 0, 0] = np.nan
    return out


def refusal_battery():
    """Every declared refusal mode plus unrelated exceptions."""
    return {
        "nan": returns_nan(), "inf": returns_inf(), "scalar": returns(0.0),
        "wrong_shape": lambda p: np.zeros((1, 2, 3)), "string": returns("force"),
        "nonfinite_error": raises(NonfiniteValueError("nonfinite", {"reason": "nonfinite"})),
        "malformed_error": raises(MalformedForceReturn("malformed")),
        "shape_error": raises(ForceShapeMismatch("shape")),
    }


def returns_nan():
    return lambda p: np.full(np.asarray(p).shape, np.nan)


def returns_inf():
    return lambda p: np.full(np.asarray(p).shape, np.inf)


UNRELATED = {"arm_config": ArmConfigurationError("species"), "value": ValueError("x"),
             "type": TypeError("x"), "runtime": RuntimeError("x")}


# ---- bitwise comparison ---------------------------------------------------

def bits(obj):
    """Canonical, bit-exact representation of nested results."""
    if isinstance(obj, np.ndarray):
        return ("array", obj.shape, str(obj.dtype), obj.tobytes())
    if isinstance(obj, (np.floating, float)):
        return ("float", struct.pack("<d", float(obj)))
    if isinstance(obj, (bool, np.bool_)):
        return ("bool", bool(obj))
    if isinstance(obj, (int, np.integer)):
        return ("int", int(obj))
    if isinstance(obj, dict):
        return ("dict", tuple((k, bits(v)) for k, v in sorted(obj.items())))
    if isinstance(obj, (list, tuple)):
        return ("seq", tuple(bits(v) for v in obj))
    return ("obj", obj)


def frozen_fields(result, frozen_result):
    """The comparison is of the frozen result's own fields only."""
    return {key: result[key] for key in frozen_result}


def without_cap_record(result):
    """The result minus the extra cap_record field (frozen fields only)."""
    return {key: value for key, value in result.items() if key != "cap_record"}


def run_both(start, force, proc, cap):
    from evaluation.relax import relax
    from stage2_capped.relax_capped import relax_capped
    frozen = relax(start, force, proc)
    capped = relax_capped(start, force, proc, movement_cap_A=cap)
    return frozen, capped
