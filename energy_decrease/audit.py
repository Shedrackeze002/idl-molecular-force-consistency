"""Check completed recovery cases from their saved arrays and reduce paired results."""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import connected_components

from evaluation.outcomes import classify_outcome
from evaluation.runner import _final_record, _own_reference
from stage2_capped import admission, driver
from trace_validation_recovery import score_reference_control


def read(path):
    return json.loads(path.read_bytes())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_energy(row, accepted, force_stop_trials, mode):
    requests, trials = row["energy"]["requests"], row["energy"]["trials"]
    assert 0 <= requests <= row["force_evaluation_count"]
    assert [trial["attempt_index"] for trial in trials] == sorted({
        trial["attempt_index"] for trial in trials
    })
    if mode == "replay":
        assert not requests and not trials
        return
    assert force_stop_trials <= {trial["attempt_index"] for trial in trials}
    if not trials:
        assert not requests and not accepted
        return
    initial = trials[0]["incumbent_eV"]
    incumbent = initial
    assert requests == (1 if initial is None else 1 + len(trials))
    energy_accepts = set()
    for trial in trials:
        attempt = trial["attempt_index"]
        assert 2 <= attempt <= row["force_evaluation_count"]
        before, candidate = trial["incumbent_eV"], trial["trial_eV"]
        assert before == incumbent
        decision = trial["decision"]
        if decision == "refused":
            assert trial is trials[-1] and candidate is None
            assert row["outcome_category"] == "energy_evaluation_failure"
        else:
            assert math.isfinite(before) and math.isfinite(candidate)
            expected = ("accepted" if candidate < before else
                        "equal" if candidate == before else "uphill")
            assert decision == expected
        if decision == "accepted":
            incumbent = candidate
            energy_accepts.add(attempt)
    assert energy_accepts == accepted


def check_row(inputs, row, trace_path, model, mode, references):
    key = row["start_id"]
    original = row["start_kind"] == "original"
    if original:
        start = inputs.starts[key]
        _, own = _own_reference(key, inputs.classification, inputs.refs)
        start_positions = start["positions"]
    else:
        own = references[key]
        start_positions = own.positions
    assert Path(own.record_id).stem.removeprefix("coordsmin_") == row["target_minimum"]
    assert sha(trace_path) == row["trace_sha256"]
    with np.load(trace_path, allow_pickle=False) as trace:
        positions, accept, force, stop = [trace[k] for k in ("positions", "accepted", "max_force", "force_stop")]
    assert positions.shape == (len(force), *start_positions.shape)
    assert accept.shape == force.shape == stop.shape
    assert 1 <= len(force) == row["force_evaluation_count"] <= 1500
    assert np.all(np.isfinite(positions)) and np.array_equal(positions[0], start_positions)
    assert not accept[0]
    assert np.array_equal(stop, force <= inputs.procedure["convergence_tolerance_eV_per_A"])
    steps = np.flatnonzero(accept)
    endpoint_index = int(steps[-1]) if len(steps) else 0
    endpoint = positions[endpoint_index]
    assert len(steps) == row["step_count"]
    assert np.array_equal(endpoint, row["final_positions"])
    assert bool(stop[endpoint_index]) == row["converged"]
    if np.isfinite(force[endpoint_index]):
        assert force[endpoint_index] == row["final_max_force_component_eV_per_A"]
    assert bool(row["recovered"]) == (row["outcome_category"] == "matched_reference")
    low_force_trials = set((np.flatnonzero(stop[1:]) + 2).tolist())
    check_energy(row, set((steps + 1).tolist()), low_force_trials, mode)

    pairs = np.linalg.norm(endpoint[:, None] - endpoint[None, :], axis=-1)
    assert (connected_components(pairs < 5.0, directed=False, return_labels=False) == 1
            ) == row["endpoint_connected"]
    reason = row["stop_reason"]
    if reason in ("nonfinite_energy", "invalid_energy"):
        assert row["outcome_category"] == "energy_evaluation_failure"
        assert not row["converged"]
        return
    kwargs = {"converged": row["converged"], "budget_exhausted": row["budget_exhausted"],
              "forces_finite": reason != "nonfinite_force"}
    if not original and model == "mace":
        score = score_reference_control(
            dict(kwargs, final_positions=endpoint, failure_reason=reason,
                 final_max_force_component_eV_per_A=row["final_max_force_component_eV_per_A"]),
            own, inputs.policy)
        assert score["operational_force_converged_own_match"] == row["recovered"]
        rmsd = score["own_alignment"]["rmsd_A"]
        if rmsd is not None:
            assert math.isclose(rmsd, row["rmsd_A"], abs_tol=1e-10)
        return
    final = (_final_record(key, start, endpoint) if original else
             replace(own, record_id=own.record_id + "/audit-endpoint",
                     positions=endpoint, forces=np.zeros_like(own.forces)))
    score = classify_outcome(final, own, inputs.refs, inputs.policy, stop_reason=reason,
                             seen_minimum=inputs.seen[key] if original else True, **kwargs)
    assert score["category"] == row["outcome_category"]
    if score["rmsd_A"] is not None:
        assert math.isclose(score["rmsd_A"], row["rmsd_A"], abs_tol=1e-10)


def summarize(rows):
    original = [r for r in rows if r["start_kind"] == "original"]
    reference = [r for r in rows if r["start_kind"] == "reference"]
    return {
        "original_recovered": sum(r["recovered"] for r in original),
        "original_outcomes": dict(Counter(r["outcome_category"] for r in original)),
        "original_stop_reasons": dict(Counter(r["stop_reason"] or "force_stop" for r in original)),
        "original_disconnected": sum(not r["endpoint_connected"] for r in original),
        "reference_matches": sum(r["recovered"] for r in reference),
        "reference_matches_without_movement": sum(
            r["recovered"] and r["step_count"] == 0 for r in reference),
        "force_requests": sum(r["force_evaluation_count"] for r in rows),
        "energy_requests": sum(r["energy"]["requests"] for r in rows),
        "energy_decisions": dict(Counter(
            t["decision"] for r in rows for t in r["energy"]["trials"])),
        "elapsed_s": sum(r["elapsed_s"] for r in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group9", type=Path, required=True)
    parser.add_argument("--attempt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=("e3x", "mace"), default=("e3x", "mace"))
    args = parser.parse_args()
    inputs = driver.load_study_inputs(
        admission.file_sha256(args.group9 / admission.RECOVERY_RECORDS), args.group9)
    references = {f"min{Path(key).stem.removeprefix('coordsmin_')}": inputs.refs[key]
                  for key in inputs.reference_ids}
    result = {"status": "COMPLETE_AUDITED", "models": {}, "validation_connections": 5}
    for model in args.models:
        data, meta = {}, {}
        for mode in ("replay", "energy"):
            folder = args.attempt / f"{model}-{mode}"
            path = folder / "summary.json"
            run = read(path)
            assert (run["model"], run["mode"], run["status"]) == (model, mode, "complete")
            assert len(run["rows"]) == 16
            parity = read(folder / "preflight.json")
            assert parity["status"] == "verified"
            meta[mode] = {"run_sha256": sha(path), "historical_sha256": run["historical_sha256"],
                          "relaxer_sha256": run["relaxer_sha256"],
                          "max_replay_endpoint_difference_A": run["max_replay_endpoint_difference_A"],
                          "preflight_sha256": sha(folder / "preflight.json"),
                          "checkpoint": parity["checkpoint_identity"].get(
                              "checkpoint_sha256", parity["checkpoint_identity"].get("model_sha256"))}
            for index, row in enumerate(run["rows"]):
                assert read(folder / f"{index:02d}.json") == row
                check_row(inputs, row, folder / f"{index:02d}.npz", model, mode, references)
            data[mode] = {(r["start_kind"], r["start_id"]): r for r in run["rows"]}
            assert set(data[mode]) == ({("original", key) for key in inputs.starts} |
                                       {("reference", key) for key in references})
        assert meta["replay"]["checkpoint"] == meta["energy"]["checkpoint"]
        assert meta["replay"]["historical_sha256"] == meta["energy"]["historical_sha256"]
        assert meta["replay"]["relaxer_sha256"] == meta["energy"]["relaxer_sha256"]
        pairs = []
        for key in sorted(data["replay"]):
            item = {"start_kind": key[0], "start_id": key[1]}
            for mode in ("replay", "energy"):
                row = data[mode][key]
                item[mode] = {k: row[k] for k in (
                    "target_minimum", "outcome_category", "recovered", "stop_reason",
                    "rmsd_A", "endpoint_connected", "converged", "step_count",
                    "force_evaluation_count", "elapsed_s", "trace_sha256")}
                item[mode]["energy_requests"] = row["energy"]["requests"]
            pairs.append(item)
        flips = [p for p in pairs if p["replay"]["recovered"] != p["energy"]["recovered"]]
        originals = [p for p in pairs if p["start_kind"] == "original"]
        by_connection = {}
        for pair in originals:
            connection = pair["start_id"].split("/")[1]
            row = by_connection.setdefault(connection, {"replay": 0, "energy": 0})
            for mode in row:
                row[mode] += pair[mode]["recovered"]
        result["models"][model] = {**{
            mode: summarize(rows.values()) for mode, rows in data.items()
        }, "pairs": pairs, "recovery_flips": flips,
            "original_recoveries_by_connection": by_connection, "source_evidence": meta}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({model: {mode: data[mode] for mode in ("replay", "energy")}
                      for model, data in result["models"].items()}, indent=2))


if __name__ == "__main__":
    main()
