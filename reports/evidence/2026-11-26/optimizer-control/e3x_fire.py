"""Aspirin shape recovery with ASE FIRE in place of the custom force-following search.

Copy of e3x_lbfgs.py with only the optimizer changed: ase.optimize.FIRE with default settings
(downhill_check=False, so the step never reads energy) replaces LBFGS, and each run also records
the number of FIRE velocity resets (steps where v.f <= 0 zeroed the velocity and halved dt).

Same logic as run_opt2.py, but the checkpoints are loaded and verified with the team's own
loaders (energy_decrease.run setup_e3x / setup_mace, stage2_capped admission and arm provider,
run_validation_recovery.adapter_replay) and every endpoint is scored with the team's
evaluation.outcomes.classify_outcome and the reference pool and policy of
stage2_capped.driver.load_study_inputs. geom.sym_rmsd is a symmetry-aware cross-check.

Per start (as in e3x_lbfgs.py, with FIRE in place of LBFGS): the optimizer, using forces
only, runs to fmax=0.01 eV/A (ASE per-atom norm) within 1,500 force calls, the endpoint is
recorded, then the same optimizer continues to fmax=0.001 within the same total of 1,500 calls
and the endpoint is recorded again. The ASE calculator returns the arm's forces (float64, CPU)
and a constant 0.0 energy for every arm, so no model energy can reach the optimizer. A run stops
early if a force (or the structure) becomes non-finite.

Success uses the report's rule: max force component <= 0.01 eV/A and a match to the branch
target within 0.05 A (classify_outcome category matched_reference).

Usage (see e3x-fire-2026-11-26/SUMMARY.md for the exact environment):
  python e3x_fire.py run --model mace --group9 G --admission A --output DIR
  python e3x_fire.py run --model e3x  --group9 G --admission A --output DIR [--checkpoints ...]
  python e3x_fire.py summarize --root e3x-fire-2026-11-26
(compare-cloud is kept from e3x_lbfgs.py unchanged; it compares against the L-BFGS cloud rows.)
"""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

BUDGET = 1500
THRESHOLDS = (0.01, 0.001)
STEP_CAP = 15000       # FIRE only: total optimizer steps per run (10x the force-call budget)
FORCE_RULE = 0.01      # report convergence rule, max force component in eV/A
MATCH_RULE = 0.05      # report match tolerance in A (policy.endpoint_tolerance)
ADMISSION_SHA256 = {
    "e3x": "3a64d4f1942ab32ef7930bad9e4093034e4ba215d6e355d8c43023e0b2893dd5",
    "mace": "7bacb618c3bde93f4ef4f29c79b310d9896c3c9a8317f846bf9a17a25b152625",
}
MACE_ID = "mace-runA-final-ema"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def clean(value):
    """JSON-safe copy: numpy scalars and arrays become Python values, non-finite floats None."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


# ---------------------------------------------------------------- calculator

class NonfiniteStop(Exception):
    """A force evaluation returned (or was asked for) non-finite values."""


class BudgetOverrun(RuntimeError):
    """More than BUDGET force calls were requested; this is a driver bug."""


def make_calculator_class():
    from ase.calculators.calculator import Calculator, all_changes
    from evaluation.adapter import ArmConfigurationError, NonfiniteValueError

    class ArmCalculator(Calculator):
        """Forces from a team arm; energy is the constant 0.0 for every arm."""

        implemented_properties = ["energy", "forces"]

        def __init__(self, arm, species, record_id, budget=BUDGET):
            super().__init__()
            self.arm, self.species, self.record_id, self.budget = arm, tuple(species), record_id, budget
            self.calls = 0
            self.energy_requests = 0
            self.nonfinite = None

        def get_property(self, name, atoms=None, allow_calculation=True):
            if name in ("energy", "free_energy"):
                self.energy_requests += 1
            return super().get_property(name, atoms, allow_calculation)

        def calculate(self, atoms=None, properties=("forces",), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            if self.calls >= self.budget:
                raise BudgetOverrun(f"{self.record_id}: force call {self.calls + 1} exceeds {self.budget}")
            self.calls += 1
            positions = np.array(self.atoms.get_positions(), dtype=np.float64)
            if not np.isfinite(positions).all():
                self.nonfinite = "nonfinite_positions"
                raise NonfiniteStop(self.nonfinite)
            try:
                forces, info = self.arm["forces"](positions, species=self.species, record_id=self.record_id)
            except (NonfiniteValueError, ArmConfigurationError) as error:
                if isinstance(error, ArmConfigurationError) and "finite" not in str(error):
                    raise
                self.nonfinite = f"arm refused: {type(error).__name__}: {error}"
                raise NonfiniteStop(self.nonfinite) from error
            if info.get("status") != "ok":
                raise RuntimeError(f"{self.record_id}: non-ok force info {info!r}")
            forces = np.array(forces, dtype=np.float64)
            if forces.shape != positions.shape:
                raise RuntimeError(f"{self.record_id}: force shape {forces.shape}")
            if not np.isfinite(forces).all():
                self.nonfinite = "nonfinite_force"
                raise NonfiniteStop(self.nonfinite)
            self.results = {"energy": 0.0, "forces": forces}

    return ArmCalculator


# ---------------------------------------------------------------- loading

def setup_e3x_all(args, checkpoint_ids):
    """setup_e3x from the combined-search run.py, for each requested checkpoint."""
    from e3x_phase_common import SPLIT
    from run_validation_recovery import adapter_replay
    from stage2_capped import admission, driver
    from stage2_capped.launch_arms import real_arm_provider

    report = admission.admit_all(args.group9)
    accepted = json.loads(Path(args.admission).read_bytes())
    if not accepted["verdict"]["passed"] or report["identities"] != accepted["admission"]["identities"]:
        raise ValueError("E3x checkpoint identities differ from the accepted replay")
    inputs = driver.load_study_inputs(report["launch_records_sha256"], args.group9)
    items = {row["id"]: row for row in admission.load_bindings(args.group9)}
    directory = admission.data_directory(args.group9)
    for cid in checkpoint_ids:
        item = items[cid]
        arm = real_arm_provider([item], directory, admission.PinPolicy())(cid)
        parity = adapter_replay(arm, item["score_directory"], directory, SPLIT)
        if parity["status"] != "verified":
            raise ValueError(f"{cid}: parity {parity['status']}")
        yield cid, inputs, arm, parity, report["identities"][cid]


def setup_mace_one(args):
    from energy_decrease.run import setup_mace
    inputs, arm, parity = setup_mace(args)
    yield MACE_ID, inputs, arm, parity, parity["checkpoint_identity"]


# ---------------------------------------------------------------- scoring

def label_of(key):
    return Path(key).stem.replace("coordsmin_", "min")


def min_number(key):
    return Path(key).stem.removeprefix("coordsmin_")


def score_endpoint(task, inputs, positions, forces, *, calls, nonfinite):
    from evaluation.outcomes import classify_outcome
    from evaluation.runner import _final_record
    from scipy.sparse.csgraph import connected_components
    import geom

    kind, key, own, start = task
    finite_structure = bool(np.isfinite(positions).all())
    max_component = float(np.abs(forces).max()) if forces is not None else None
    max_norm = float(np.linalg.norm(forces, axis=1).max()) if forces is not None else None
    forces_finite = nonfinite is None
    report_converged = bool(forces_finite and max_component is not None and max_component <= FORCE_RULE)
    budget_exhausted = bool(not report_converged and calls >= BUDGET)
    if not forces_finite:
        stop_reason = "nonfinite_force"
    elif report_converged:
        stop_reason = None
    else:
        stop_reason = "budget_exhausted" if budget_exhausted else "no_progress"
    final = (_final_record(key, start, positions) if start is not None
             else replace(own, record_id=own.record_id + "/trace-final",
                          positions=np.array(positions, dtype=np.float64), forces=np.zeros_like(own.forces)))
    scored = classify_outcome(
        final, own, inputs.refs, inputs.policy, budget_exhausted=budget_exhausted,
        converged=report_converged, forces_finite=forces_finite,
        nonfinite_force=not forces_finite, stop_reason=stop_reason,
        seen_minimum=inputs.seen[key] if start is not None else True)
    row = {
        "outcome_category": scored["category"], "team_rmsd_A": scored["rmsd_A"],
        "recovered": scored["category"] == "matched_reference",
        "report_converged": report_converged, "budget_exhausted": budget_exhausted,
        "stop_reason": stop_reason, "forces_finite": forces_finite, "nonfinite_detail": nonfinite,
        "max_abs_force_component_eV_per_A": max_component, "max_atom_force_norm_eV_per_A": max_norm,
    }
    species = list(own.species)
    if finite_structure:
        target, n_iso = geom.sym_rmsd(species, positions, np.asarray(own.positions))
        nearest, nearest_rmsd = None, None
        for ref_key, ref in sorted(inputs.refs.items()):
            value, _ = geom.sym_rmsd(species, positions, np.asarray(ref.positions))
            if value is not None and (nearest_rmsd is None or value < nearest_rmsd):
                nearest, nearest_rmsd = min_number(ref_key), value
        pairs = np.linalg.norm(positions[:, None] - positions[None, :], axis=-1)
        connected = bool(connected_components(pairs < 5.0, directed=False, return_labels=False) == 1)
    else:
        target, n_iso, nearest, nearest_rmsd, connected = None, 0, None, None, False
    row.update({
        "sym_rmsd_to_target_A": target, "sym_isomorphisms": n_iso,
        "sym_recovered_crosscheck": bool(report_converged and target is not None and target <= MATCH_RULE),
        "nearest_minimum": nearest, "nearest_minimum_sym_rmsd_A": nearest_rmsd,
        "endpoint_connected_5A": connected,
    })
    if kind == "reference":
        # The start is the DFT minimum itself, so this is the distance from the model's own
        # (FIRE) minimum to the DFT minimum.
        row["own_minimum_rmsd_from_dft_A"] = target
    return row


# ---------------------------------------------------------------- one start

def run_start(checkpoint_id, model, inputs, arm, task, ArmCalculator):
    from ase import Atoms
    from ase.optimize import FIRE

    kind, key, own, start = task
    record_id = key if start is not None else key + "/reference-control"
    positions = np.array(start["positions"] if start is not None else own.positions, dtype=np.float64)
    atoms = Atoms(symbols=list(own.species), positions=positions.copy())
    calc = ArmCalculator(arm, own.species, record_id)
    atoms.calc = calc
    opt = FIRE(atoms, logfile=None)
    # Defaults; downhill_check off, so the step never reads energy.
    defaults = dict(dt=0.1, maxstep=0.2, dtmax=1.0, Nmin=5, finc=1.1, fdec=0.5, astart=0.1, fa=0.99, a=0.1,
                    downhill_check=False, position_reset_callback=None)
    if any(getattr(opt, name) != value for name, value in defaults.items()):
        raise ValueError("FIRE is not at its defaults")
    # A velocity reset (v.f <= 0) is the only branch of FIRE.step that lowers dt (dt *= fdec);
    # otherwise dt stays or grows to at most dtmax. So count the steps where dt went down.
    resets = {"count": 0, "dt": opt.dt}

    def count_reset():
        if opt.dt < resets["dt"]:
            resets["count"] += 1
        resets["dt"] = opt.dt

    opt.attach(count_reset, interval=1)
    began = time.perf_counter()
    stopped = None
    rows = []
    for fmax in THRESHOLDS:
        stage_began = time.perf_counter()
        ase_converged = False
        # FIRE differs from LBFGS in one way that matters for the budget: after many velocity
        # resets its time step dt (halved at each reset) can become so small that a step leaves the
        # positions unchanged, so the step needs no new force call. The budget stays 1,500 force
        # calls; total optimizer steps are capped at STEP_CAP as a safety stop for a stalled run.
        # The run is driven in chunks that can never exceed the remaining force calls; re-entering
        # run() re-checks convergence on cached forces only, so the trajectory is unchanged.
        while stopped is None:
            steps = max(min(BUDGET - calc.calls - (1 if calc.calls == 0 else 0), STEP_CAP - opt.nsteps), 0)
            try:
                ase_converged = bool(opt.run(fmax=fmax, steps=steps))
            except NonfiniteStop:
                stopped = calc.nonfinite
                break
            if ase_converged or steps == 0:
                break
        endpoint = np.array(atoms.get_positions(), dtype=np.float64)
        if stopped is None:
            if calc.check_state(atoms):
                raise RuntimeError(f"{record_id}: endpoint forces are not the cached ones")
            forces = np.array(calc.results["forces"], dtype=np.float64)
        else:
            forces = None
        elapsed = time.perf_counter() - began
        row = {
            "model": model, "checkpoint_id": checkpoint_id, "start_id": key if start is not None else label_of(key),
            "start_kind": kind, "target_minimum": min_number(own.record_id), "fmax_norm_eV_per_A": fmax,
            "force_calls": calc.calls, "optimizer_steps": int(opt.nsteps), "ase_converged": ase_converged,
            "energy_property_requests": calc.energy_requests, "energy_value_returned": 0.0,
            "optimizer": "FIRE", "fire_velocity_resets": resets["count"], "fire_final_dt": opt.dt,
            "fire_steps_without_force_call": int(opt.nsteps) - max(calc.calls - 1, 0),
            "step_cap_reached": int(opt.nsteps) >= STEP_CAP,
            "wall_s": elapsed, "stage_wall_s": time.perf_counter() - stage_began,
        }
        score_began = time.perf_counter()
        row.update(score_endpoint(task, inputs, endpoint, forces, calls=calc.calls, nonfinite=stopped))
        row["scoring_s"] = time.perf_counter() - score_began
        row["final_positions"] = endpoint.tolist()
        rows.append(clean(row))
    if calc.calls > BUDGET:
        raise BudgetOverrun(record_id)
    return rows


# ---------------------------------------------------------------- commands

def command_run(args):
    if sha(args.admission) != ADMISSION_SHA256[args.model]:
        raise ValueError("Retained admission digest differs")
    args.admission_sha256 = ADMISSION_SHA256[args.model]
    args.output.mkdir(parents=False, exist_ok=False)
    from energy_decrease.run import schedule
    import ase
    # The calculator class imports evaluation.adapter, which the MACE bridge must load itself
    # (it refuses a preloaded copy), so the class is built only after the first checkpoint loads.
    ArmCalculator = None
    loader =(setup_mace_one(args) if args.model == "mace"
              else setup_e3x_all(args, args.checkpoints))
    environment = {"python": sys.executable, "ase": ase.__version__, "argv": sys.argv,
                   "PYTHONPATH": os.environ.get("PYTHONPATH"), "cwd": os.getcwd(),
                   "script_sha256": sha(__file__), "geom_sha256": sha(HERE / "geom.py")}
    write_new(args.output / "environment.json", environment)
    with (args.output / "rows.jsonl").open("x", encoding="utf-8") as stream:
        for checkpoint_id, inputs, arm, parity, identity in loader:
            if ArmCalculator is None:
                ArmCalculator = make_calculator_class()
            folder = args.output / checkpoint_id
            folder.mkdir()
            write_new(folder / "parity.json", clean({"parity": parity, "identity": identity}))
            print(f"{checkpoint_id}: parity {parity['status']} "
                  f"(max force diff {parity['maximum_absolute_force_difference']:.2e})", flush=True)
            tasks = list(schedule(inputs))
            if len(tasks) != 16:
                raise ValueError("Expected ten original starts and six known-minimum controls")
            for index, task in enumerate(tasks):
                rows = run_start(checkpoint_id, args.model, inputs, arm, task, ArmCalculator)
                write_new(folder / f"{index:02d}.json", {"rows": rows})
                for row in rows:
                    stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
                stream.flush()
                a, b = rows
                print(f"{checkpoint_id} {index + 1:02d}/16 {a['start_id']}: "
                      f"0.01 {a['outcome_category']} ({a['force_calls']}) | "
                      f"0.001 {b['outcome_category']} ({b['force_calls']})", flush=True)
    print("DONE", flush=True)


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def command_compare_cloud(args):
    """Validation: the team's matcher on the local MACE L-BFGS run against the cloud results."""
    cloud = {}
    for line in Path(args.cloud).read_text().splitlines():
        if line.startswith("{"):
            row = json.loads(line)
            if row["optimizer"] == "LBFGS":
                cloud[row["start"], row["fmax_norm"]] = row
    local = {(r["start_id"], r["fmax_norm_eV_per_A"]): r for r in load_rows(args.rows)
             if r["start_kind"] == "original"}
    if set(cloud) != set(local):
        raise ValueError(f"Start sets differ: {sorted(set(cloud) ^ set(local))}")
    report = {"per_start": [], "counts": {}}
    agree = True
    for fmax in THRESHOLDS:
        cloud_set = sorted(k for (k, f), r in cloud.items() if f == fmax and r["recovered"])
        team_set = sorted(k for (k, f), r in local.items() if f == fmax and r["recovered"])
        sym_set = sorted(k for (k, f), r in local.items() if f == fmax and r["sym_recovered_crosscheck"])
        report["counts"][str(fmax)] = {"cloud": len(cloud_set), "team_matcher": len(team_set),
                                       "local_sym_rmsd": len(sym_set),
                                       "team_missed": sorted(set(k for (k, f) in local if f == fmax) - set(team_set)),
                                       "cloud_missed": sorted(set(k for (k, f) in cloud if f == fmax) - set(cloud_set))}
        agree &= cloud_set == team_set
    for (key, fmax), c in sorted(cloud.items()):
        r = local[key, fmax]
        report["per_start"].append({
            "start": key, "fmax": fmax, "cloud_recovered": c["recovered"], "team_recovered": r["recovered"],
            "team_category": r["outcome_category"], "cloud_rmsd": c["rmsd_to_target"],
            "team_rmsd": r["team_rmsd_A"], "local_sym_rmsd": r["sym_rmsd_to_target_A"],
            "cloud_calls": c["force_calls"], "local_calls": r["force_calls"],
            "cloud_maxF": c["max_force_component"], "local_maxF": r["max_abs_force_component_eV_per_A"]})
    expected = report["counts"]["0.01"]["team_matcher"] == 7 and report["counts"]["0.001"]["team_matcher"] == 9 \
        and report["counts"]["0.001"]["team_missed"] == ["aspirin/13/minus"]
    report["agrees_with_cloud"] = bool(agree and expected)
    write_new(args.output, clean(report))
    for item in report["per_start"]:
        print(json.dumps(clean(item)))
    print(json.dumps(report["counts"], indent=1))
    print("AGREES" if report["agrees_with_cloud"] else "DISAGREES")
    return 0 if report["agrees_with_cloud"] else 1


def command_summarize(args):
    root = args.root
    order = ("energy-0", "direct-0", "consistency-0.1-0", "direct-1", "consistency-0.1-1",
             "consistency-1-0", "consistency-1-1")
    groups = {}
    for path in sorted(root.glob("*/rows.jsonl")):
        for row in load_rows(path):
            groups.setdefault(row["checkpoint_id"], []).append(row)
    summary = {"budget": BUDGET, "thresholds": THRESHOLDS, "force_rule_eV_per_A": FORCE_RULE,
               "match_rule_A": MATCH_RULE, "checkpoints": {}}
    for cid, rows in groups.items():
        entry = {}
        for fmax in THRESHOLDS:
            rs = [r for r in rows if r["fmax_norm_eV_per_A"] == fmax]
            orig = [r for r in rs if r["start_kind"] == "original"]
            refs = [r for r in rs if r["start_kind"] == "reference"]
            own = [r["own_minimum_rmsd_from_dft_A"] for r in refs if r["own_minimum_rmsd_from_dft_A"] is not None]
            entry[str(fmax)] = {
                "original_recovered": sum(r["recovered"] for r in orig), "original_count": len(orig),
                "original_recovered_ids": sorted(r["start_id"] for r in orig if r["recovered"]),
                "original_categories": dict(sorted(
                    (c, sum(r["outcome_category"] == c for r in orig)) for c in {r["outcome_category"] for r in orig})),
                "original_sym_crosscheck_recovered": sum(r["sym_recovered_crosscheck"] for r in orig),
                "reference_kept": sum(r["recovered"] for r in refs), "reference_count": len(refs),
                "reference_categories": dict(sorted(
                    (c, sum(r["outcome_category"] == c for r in refs)) for c in {r["outcome_category"] for r in refs})),
                "team_vs_sym_disagreements": sorted(
                    f"{r['start_id']}" for r in rs if r["recovered"] != r["sym_recovered_crosscheck"]),
                "nonfinite_stops": sum(not r["forces_finite"] for r in rs),
                "disconnected_endpoints": sum(not r["endpoint_connected_5A"] for r in rs),
                "budget_exhausted": sum(r["budget_exhausted"] for r in rs),
                "ase_converged": sum(r["ase_converged"] for r in rs),
                "force_calls_min": min(r["force_calls"] for r in rs),
                "force_calls_max": max(r["force_calls"] for r in rs),
                "own_minimum_rmsd_from_dft_A": {r["start_id"]: r["own_minimum_rmsd_from_dft_A"] for r in refs},
                "median_own_minimum_rmsd_from_dft_A": statistics.median(own) if own else None,
                "energy_property_requests_total": sum(r["energy_property_requests"] for r in rs),
                "optimizer_wall_s_total": sum(r["stage_wall_s"] for r in rs),
                "endpoint_connected_fraction": sum(r["endpoint_connected_5A"] for r in rs) / len(rs),
                "reference_report_converged": sum(r["report_converged"] for r in refs),
                "reference_connected": sum(r["endpoint_connected_5A"] for r in refs),
                "fire_velocity_resets": {"min": min(r["fire_velocity_resets"] for r in rs),
                                         "median": statistics.median(r["fire_velocity_resets"] for r in rs),
                                         "max": max(r["fire_velocity_resets"] for r in rs)},
            }
        summary["checkpoints"][cid] = entry
    write_new(root / "summary.json", clean(summary))

    def cell(cid, fn):
        return " / ".join(fn(summary["checkpoints"][cid][str(f)]) for f in THRESHOLDS)

    lines = ["| Checkpoint | Recovered at 0.01 | Recovered at 0.001 | Known minima kept (0.01 / 0.001) "
             "| Force calls (min-max) | Median own-minimum RMSD from DFT minimum, A (0.01 / 0.001) "
             "| Endpoint connected at 5 A, fraction of runs (0.01 / 0.001) |",
             "|---|---|---|---|---|---|---|"]
    ids = [c for c in order if c in summary["checkpoints"]] + \
          [c for c in summary["checkpoints"] if c not in order]
    for cid in ids:
        e = summary["checkpoints"][cid]
        a, b = e["0.01"], e["0.001"]
        lines.append(
            f"| {cid} | {a['original_recovered']}/{a['original_count']} | {b['original_recovered']}/{b['original_count']} "
            f"| {a['reference_kept']}/{a['reference_count']} / {b['reference_kept']}/{b['reference_count']} "
            f"| {min(a['force_calls_min'], b['force_calls_min'])}-{max(a['force_calls_max'], b['force_calls_max'])} "
            f"| {cell(cid, lambda x: 'n/a' if x['median_own_minimum_rmsd_from_dft_A'] is None else format(x['median_own_minimum_rmsd_from_dft_A'], '.3f'))} "
            f"| {cell(cid, lambda x: format(x['endpoint_connected_fraction'], '.2f'))} |")
    with (root / "TABLE.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--model", choices=("e3x", "mace"), required=True)
    run.add_argument("--group9", type=Path, required=True)
    run.add_argument("--admission", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--checkpoints", nargs="+", default=[
        "energy-0", "direct-0", "consistency-0.1-0", "direct-1", "consistency-0.1-1",
        "consistency-1-0", "consistency-1-1"])
    compare = sub.add_parser("compare-cloud")
    compare.add_argument("--rows", type=Path, required=True)
    compare.add_argument("--cloud", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    summarize = sub.add_parser("summarize")
    summarize.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        command_run(args)
    elif args.command == "compare-cloud":
        sys.exit(command_compare_cloud(args))
    else:
        command_summarize(args)


if __name__ == "__main__":
    main()
