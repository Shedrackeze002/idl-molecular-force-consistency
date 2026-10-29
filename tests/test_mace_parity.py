import json
import os
from pathlib import Path
import numpy as np
import pytest

from check_mace_parity import (EXPECTED_CONNECTION_COUNTS, InputValidationError,
                               TOLERANCE, _load_memberships,
                               compare_predictions, validate_inputs)


def bundle(order=("a", "b"), delta=0.0):
    energy = np.array([0.0, 2.0], dtype=np.float64)
    forces = np.zeros((2, 21, 3), dtype=np.float64)
    values = {"a": (energy[0], forces[0]), "b": (energy[1], forces[1])}
    return {"record_ids": list(order),
            "energy": np.array([values[key][0] for key in order], dtype=np.float64) + delta,
            "forces": np.array([values[key][1] for key in order], dtype=np.float64) + delta}


def test_permuted_storage_order_joins_exact_ids():
    result = compare_predictions(bundle(("b", "a")), bundle())
    assert result["status"] == "PARITY_PASS"
    assert result["record_count"] == 2


@pytest.mark.parametrize("delta,status", [(TOLERANCE * .9, "PARITY_PASS"),
                                           (TOLERANCE, "PARITY_PASS"),
                                           (np.nextafter(TOLERANCE, np.inf), "PARITY_FAIL")])
def test_absolute_bound_below_at_and_above(delta, status):
    expected = bundle()
    expected["energy"][:] = 0.0
    actual = {"record_ids": expected["record_ids"],
              "energy": expected["energy"] + delta,
              "forces": expected["forces"] + delta}
    assert compare_predictions(actual, expected)["status"] == status


def test_duplicate_missing_dtype_shape_and_nonfinite_refused():
    bad = bundle()
    bad["record_ids"] = ["a", "a"]
    with pytest.raises(InputValidationError):
        compare_predictions(bad, bundle())


def test_raw_surplus_rows_refused_before_id_join():
    actual = bundle(("a", "b"))
    actual["record_ids"] = ["a"]
    with pytest.raises(InputValidationError, match="raw shapes"):
        compare_predictions(actual, {"record_ids": ["a"], "energy": np.zeros(1, dtype=np.float64),
                                     "forces": np.zeros((1, 21, 3), dtype=np.float64)})


def test_atomic_npz_preserves_previous_file_on_write_failure(tmp_path, monkeypatch):
    import check_mace_parity as module
    path = tmp_path / "partial.npz"
    module._atomic_npz(path, record_ids=np.array(["a"]), energy=np.zeros(1),
                       forces=np.zeros((1, 21, 3)))
    original = path.read_bytes()
    monkeypatch.setattr(module.np, "savez", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(OSError):
        module._atomic_npz(path, record_ids=np.array(["a", "b"]), energy=np.zeros(2),
                           forces=np.zeros((2, 21, 3)))
    assert path.read_bytes() == original


def test_failed_accessor_attempts_are_counted(tmp_path, monkeypatch):
    import check_mace_parity as module
    import mace_loader
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    inputs = {"record_ids": ["a"], "frames": {"a": {"positions": np.zeros((21, 3)),
              "species": ("H",) * 21}}, "species": ("H",) * 21,
              "expected_energy": np.zeros(1), "expected_forces": np.zeros((1, 21, 3))}
    monkeypatch.setattr(module, "validate_inputs", lambda root: inputs)
    monkeypatch.setattr(mace_loader, "verify_packet", lambda root: {"manifest_sha256": "m", "model_sha256": "s"})
    arm = {"forces": lambda *args: (np.zeros((21, 3)), {}),
           "energy": lambda *args: (_ for _ in ()).throw(RuntimeError("energy failed"))}
    monkeypatch.setattr(mace_loader, "load_admitted_arm", lambda *args, **kwargs: (arm, {"status": "ADMITTED"}))
    result = module.run_parity(tmp_path, attempt)
    assert result["status"] == "INCOMPLETE"
    assert result["force_requests"] == 1 and result["energy_requests"] == 1
    bad = bundle()
    bad["energy"] = bad["energy"].astype(np.float32)
    with pytest.raises(InputValidationError):
        compare_predictions(bad, bundle())
    bad = bundle()
    bad["forces"][0, 0, 0] = np.nan
    with pytest.raises(InputValidationError):
        compare_predictions(bad, bundle())


def test_near_zero_does_not_gain_relative_tolerance():
    actual, expected = bundle(), bundle()
    actual["energy"][0] = np.nextafter(TOLERANCE, np.inf)
    assert compare_predictions(actual, expected)["status"] == "PARITY_FAIL"


def test_membership_schema_preserves_literal_18_2(tmp_path):
    path = tmp_path / "valid.memberships.json"
    path.write_text(json.dumps({"frame": {"groups": ["18_2"], "role": "path"}}))
    assert _load_memberships(path) == {"frame": "18_2"}
    path.write_text(json.dumps({"frame": {"groups": ["18", "18_2"]}}))
    with pytest.raises(InputValidationError, match="exactly one"):
        _load_memberships(path)


def test_verified_packet_inputs_when_available():
    packet = os.environ.get("MACE_PACKET_ROOT")
    if not packet:
        pytest.skip("private packet intentionally unavailable")
    result = validate_inputs(Path(packet))
    assert len(result["record_ids"]) == 974
    assert result["connection_counts"] == EXPECTED_CONNECTION_COUNTS
    assert "18_2" in set(result["memberships"].values())
