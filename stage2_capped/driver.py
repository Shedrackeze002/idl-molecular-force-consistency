"""Driver for the Stage 2 movement-cap study (plan M4; spec v4 sections 5, 7, 10).

Runs the fixed schedule of relaxations for one stage and writes one result row and
one binary trace per run. No model is chosen or loaded here: arms come from an
`arm_provider(model_id)` (the restore fork in a real launch, a synthetic arm in the
tests). Nothing here starts a stage by itself; launch scripts (M7, M8) call it.

What a run does:
- Original starts go through the copied second pass (pass_copy.second_pass); the
  checks the frozen first pass and run_model_pass make before any force call are
  repeated by `prevalidate`.
- Reference starts follow the paired-trace harness (relax from the reference
  geometry, classify with seen_minimum True).
- Stage 0 compares every control run with the saved result as it finishes and stops
  at the first mismatch (stage_gate.Stage0Controller); the single pre-authorized
  frozen re-run is `run_frozen_rerun`.
- Capped stages check each run against its control: the prefix-equality gate.
- Each run's attempt count is added to the stage's attempt ceiling, the output root is
  checked against the storage ceiling before every run, and a stop of any kind leaves
  the rows in place with an INCOMPLETE marker.
- A stage runs under a StageLedger and the fixed schedule; explicit task lists and a
  missing ledger exist for tests only and are refused unless a test flag is set. A finished
  stage writes STAGE_MANIFEST.json and then STAGE_COMPLETE.json.
"""
import hashlib
import json
import re
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from stage2_capped import admission, launch_io, prefix_gate, row_schema, stage0_compare
from stage2_capped.pass_copy import second_pass
from stage2_capped.relax_capped import relax_capped
from stage2_capped.stage_gate import StageOrderError
from stage2_capped.stage0_compare import (OWN_IDS, compare_original, compare_reference,
                                          minimum_number, original_task_id, reference_task_id)

WORKTREE = Path(__file__).resolve().parents[1]
SAVED_FILE_COUNT = 110       # 50 saved original rows, and relaxation.json and result.json for each of 30 references
CAPS = (0.05, 0.10, 0.25, 1.0)
MAX_FORCE_EVALUATIONS = 1500
GRAPH_CUTOFF_A = 5.0
A0_WALL_SECONDS = 300.0
STORAGE_LIMIT_BYTES = 2 ** 30
ROW_RESERVE_BYTES = 65536
# The six procedure values of the accepted relaxation (configs/aspirin-shape-recovery.json).
FROZEN_PROCEDURE = {"max_force_evaluations": 1500, "initial_step_size": 0.2, "backtracking_factor": 0.5,
                    "min_backtracking_floor": 1e-08, "convergence_tolerance_eV_per_A": 0.01,
                    "max_backtracking_attempts_per_step": 40}
assert MAX_FORCE_EVALUATIONS == FROZEN_PROCEDURE["max_force_evaluations"]


def run_reserve_bytes(n_atoms):
    """Space reserved before each run: one binary trace record per force attempt at the budget
    (the trace dtype's own size), plus the row."""
    return MAX_FORCE_EVALUATIONS * row_schema.trace_dtype(n_atoms).itemsize + ROW_RESERVE_BYTES

STAGE_SPECS = {
    "0": {"models": admission.PRIMARY_IDS, "caps": (None,), "runs": 80,
          "attempt_ceiling": 120_000, "wall_seconds": 600.0},
    "0b": {"models": admission.LAMBDA1_IDS, "caps": (None,), "runs": 32,
           "attempt_ceiling": 48_000, "wall_seconds": 300.0},
    "1": {"models": admission.PRIMARY_IDS, "caps": CAPS, "runs": 320,
          "attempt_ceiling": 480_000, "wall_seconds": 2100.0},
    "2": {"models": admission.LAMBDA1_IDS, "caps": CAPS, "runs": 128,
          "attempt_ceiling": 192_000, "wall_seconds": 900.0},
}
CONTROL_STAGES = ("0", "0b")

MODEL_IDENTITY = {
    "energy-0": ("energy", 0, None),
    "direct-0": ("direct", 0, None), "direct-1": ("direct", 1, None),
    "consistency-0.1-0": ("consistency", 0, 0.1), "consistency-0.1-1": ("consistency", 1, 0.1),
    "consistency-1-0": ("consistency", 0, 1.0), "consistency-1-1": ("consistency", 1, 1.0),
}
assert set(MODEL_IDENTITY) == set(admission.ALL_IDS)


class DriverError(RuntimeError):
    """Base class for refusals raised by the driver."""


class AttemptCeilingExceeded(DriverError):
    pass


class DeadlineReached(DriverError):
    pass


class PrefixGateFailure(DriverError):
    pass


class MissingControl(DriverError):
    pass


class Stage0Stop(DriverError):
    """Stage 0 stopped on a mismatch or an incomplete comparison."""

    def __init__(self, verdict, task_id=None, problems=None):
        super().__init__(f"Stage 0 stopped: {verdict['verdict']}")
        self.verdict, self.task_id, self.problems = verdict, task_id, problems


# ---- Study inputs --------------------------------------------------------

class StudyInputs:
    """The validation starts, reference pool, policy and split data of the accepted run."""

    def __init__(self, starts, refs, policy, manifest, classification, seen, procedure,
                 reference_ids=OWN_IDS):
        self.starts, self.refs, self.policy = starts, refs, policy
        self.manifest, self.classification = manifest, classification
        self.seen, self.procedure, self.reference_ids = seen, procedure, tuple(reference_ids)


def _evaluation_pin(root):
    files = list((root / "evaluation").glob("*.py")) + [root / "scripts" / "run_validation_recovery.py"]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}


def load_study_inputs(expected_launch_records_sha256, group9=None, worktree=None):
    """Read the accepted recovery inputs (relocated), bind every one to its recorded
    hash and to the accepted evaluation sources, and build the starts. The launch records
    file must have the SHA-256 the launch record names; its evaluation block must equal the
    shape-recovery config, and its relaxation block the six frozen procedure values. Reads
    the validation members and the minima of the archive only; the test partition is never
    opened (validation_inputs records test_branch_members_read: 0)."""
    from evaluation.validation_controls import validation_inputs
    group9 = admission.group9_root(group9)
    worktree = Path(worktree or WORKTREE)
    record_bytes = (group9 / admission.RECOVERY_RECORDS).read_bytes()
    if hashlib.sha256(record_bytes).hexdigest() != expected_launch_records_sha256:
        raise admission.AdmissionError("the recovery launch records differ from the SHA-256 the launch record names")
    record = json.loads(record_bytes)
    config = json.loads((worktree / "configs" / "aspirin-shape-recovery.json").read_text())
    if record["evaluation"] != config:
        raise admission.AdmissionError("the recovery launch records' evaluation block differs from the config")
    if record["evaluation"]["relaxation"] != FROZEN_PROCEDURE:
        raise admission.AdmissionError("the relaxation procedure differs from the six frozen values")
    for path, expected in record["inputs"].items():
        file = admission.relocate(path, group9)
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise admission.AdmissionError(f"frozen input changed: {file}")
    if _evaluation_pin(worktree) != record["evaluation_pin"]:
        raise admission.AdmissionError("evaluation sources differ from the accepted pin")
    archive = admission.relocate(record["archive"], group9)
    manifest = json.loads(admission.relocate(record["manifest"], group9).read_text())
    classification = json.loads(admission.relocate(record["classification"], group9).read_text())
    starts, refs, policy, binding = validation_inputs(archive, manifest, classification)
    if binding != record["start_binding"]:
        raise admission.AdmissionError("frozen validation starts or member binding differs")
    prepared = json.loads((admission.data_directory(group9) / "manifest.json").read_text())
    train_ids = set(prepared["partitions"]["train"]["record_ids"])
    seen = {key: classification[key]["reference_class"] in train_ids for key in starts}
    return StudyInputs(starts, refs, policy, manifest, classification, seen,
                       record["evaluation"]["relaxation"])


# ---- Schedule ------------------------------------------------------------

def build_schedule(stage, branch_keys, reference_ids=OWN_IDS, check_count=True):
    """The ordered tasks of a stage: per model, per cap, the original starts sorted by
    branch key, then the reference starts in minimum order."""
    spec = STAGE_SPECS[stage]
    tasks = []
    for model_id in spec["models"]:
        for cap in spec["caps"]:
            for key in sorted(branch_keys):
                tasks.append({"stage": stage, "model_id": model_id, "cap": cap,
                              "kind": "original", "key": key})
            for reference in reference_ids:
                tasks.append({"stage": stage, "model_id": model_id, "cap": cap,
                              "kind": "reference", "key": reference})
    if check_count and len(tasks) != spec["runs"]:
        raise DriverError(f"stage {stage} schedule has {len(tasks)} runs, expected {spec['runs']}")
    return tasks


def task_id(task):
    base = (original_task_id(task["model_id"], task["key"]) if task["kind"] == "original"
            else reference_task_id(task["model_id"], task["key"]))
    return base if task["cap"] is None else f"{base}@{row_schema.cap_label(task['cap'])}"


def start_id_of(task):
    return task["key"] if task["kind"] == "original" else f"min{minimum_number(task['key'])}"


# ---- Prevalidation (before any force call) -------------------------------

def prevalidate(starts, arm, procedure, classification, refs, policy, manifest):
    """The checks run_model_pass and the frozen first pass make before any force
    evaluation: arm ready, split binding and approved policy, partitions, procedure, reference
    resolution, start/arm compatibility, and batch size 1. Returns (plans, bound)."""
    from evaluation.adapter import _validated_positions
    from evaluation.relax import _check_start, _validate_procedure
    from evaluation.runner import (PreparationPolicy, SplitBindingError, _approved_policy,
                                   _bind_split_manifest, _connection_partition,
                                   _manifest_synthetic_declaration, _own_reference,
                                   _policy_mismatch_field, _validate_connection_partitions)
    if arm["status"] != "ready":
        raise ValueError(f"arm {arm.get('arm_id')!r} is not ready ({arm['status']!r}): {arm.get('refusal_reason')}")
    bound = _bind_split_manifest(manifest, synthetic_declaration=_manifest_synthetic_declaration(manifest))
    if bound != "fixture":
        if not isinstance(policy, PreparationPolicy):
            raise SplitBindingError("production evaluation requires a real PreparationPolicy")
        mismatch = _policy_mismatch_field(policy, _approved_policy())
        if mismatch is not None:
            raise SplitBindingError(f"preparation policy differs from the approved one at {mismatch[0]!r}")
    _validate_connection_partitions(starts, manifest)
    _validate_procedure(procedure)
    plans = []
    for key, entry in sorted(starts.items()):
        if _connection_partition(key, manifest) != "validation":
            raise ValueError("Only validation branches admitted")
        pool_key, own = _own_reference(key, classification, refs)
        array = _validated_positions(entry["positions"], arm["species"], entry["species"], arm["arm_id"])
        if array.ndim != 2:
            raise ValueError("relaxations run at batch size 1: each start must be one (N, 3) molecule")
        _check_start(array[None])
        plans.append((key, entry, "validation", pool_key, own))
    return plans, bound


# ---- Row ingredients -----------------------------------------------------

def component_count(positions, cutoff=GRAPH_CUTOFF_A):
    """Number of connected components of the interaction graph (pairs closer than the
    cutoff are linked), as the frozen trace summary computes it."""
    p = np.asarray(positions).reshape(-1, 3)
    n = len(p)
    close = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1) < cutoff
    unseen, count = set(range(n)), 0
    while unseen:
        count += 1
        todo = [min(unseen)]
        member = set()
        while todo:
            i = todo.pop()
            if i in member:
                continue
            member.add(i)
            todo.extend(j for j in range(n) if close[i, j] and j not in member and j != i)
        unseen -= member
    return count


def connectivity(records, final_positions):
    """(endpoint connected, first attempt whose structure is disconnected). The query
    covers every attempt: the start, accepted and rejected trials."""
    first = next((r["attempt_index"] for r in records if component_count(r["positions"]) > 1), None)
    return component_count(final_positions) == 1, first


def accepted_step_fractions(records):
    steps = [r["max_displacement_A"] for r in records if r["kind"] == "trial" and r["accepted"] is True]
    if not steps:
        return 0.0, 0.0
    return (sum(s > 0.25 for s in steps) / len(steps), sum(s > 1.0 for s in steps) / len(steps))


def final_max_force(records):
    """Largest force component at the last accepted state (the start if none accepted)."""
    last = [r for r in records if r["kind"] == "initial" or r["accepted"] is True][-1]
    value = last["max_force_component"]
    return None if value is None or not np.isfinite(value) else float(value)


def relaxer_sha256():
    from stage2_capped import relax_capped as module
    return hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()


def make_result_row(*, context, task, category, stop_reason, converged, budget_exhausted, step_count,
                    force_evaluation_count, connection, branch, own_minimum, cap_record, records,
                    final_positions, elapsed, trace_file, trace_sha256):
    arm, seed, lam = MODEL_IDENTITY[task["model_id"]]
    cap = task["cap"]
    label = row_schema.cap_label(cap)
    connected, first_disconnection = connectivity(records, final_positions)
    count = cap_record["trial_count"]
    fields = dict(
        launch_id=context.launch_id, stage=task["stage"], checkpoint_id=task["model_id"], arm=arm,
        consistency_lambda=lam, seed=seed, start_kind=task["kind"], start_id=start_id_of(task),
        connection=connection, branch=branch, own_minimum=own_minimum, cap_label=label,
        movement_cap_A=row_schema.cap_value(label), relaxer_sha256=context.relaxer_sha256,
        exploratory_label=row_schema.EXPLORATORY_LABEL if lam == 1.0 else None,
        outcome_category=category, recovered=category == "matched_reference",
        stop_reason=stop_reason, converged=converged, budget_exhausted=budget_exhausted,
        step_count=step_count, force_evaluation_count=force_evaluation_count,
        final_max_force_component_eV_per_A=final_max_force(records),
        max_accepted_displacement_A=cap_record["max_accepted_displacement_A"],
        endpoint_connected=connected, first_disconnection_attempt=first_disconnection,
        trial_count=count, capped_trial_count=cap_record["capped_trial_count"],
        capped_fraction=(cap_record["capped_trial_count"] / count) if count else None,
        first_capped_attempt=cap_record["first_capped_attempt"],
        accepted_trial_count=cap_record["accepted_trial_count"],
        rejected_trial_count=cap_record["rejected_trial_count"],
        refused_trial_count=cap_record["refused_trial_count"],
        accepted_travel_A=cap_record["accepted_travel_A"],
        attempted_travel_A=cap_record["attempted_travel_A"], elapsed_s=float(elapsed),
        accepted_step_fraction_gt_0p25=None, accepted_step_fraction_gt_1p0=None,
        would_bind_at_1p0=None, trace_file=trace_file, trace_sha256=trace_sha256)
    if cap is None:
        fractions = accepted_step_fractions(records)
        fields.update(accepted_step_fraction_gt_0p25=fractions[0],
                      accepted_step_fraction_gt_1p0=fractions[1],
                      would_bind_at_1p0=prefix_gate.would_bind(records, 1.0))
    return row_schema.make_row(**fields)


# ---- One run -------------------------------------------------------------

class Context:
    """Everything a stage needs besides the stage itself. The root must have been made by
    launch_io.fresh_root for this launch id, must lie outside the worktree, and must have the
    free space the storage ceiling asks for."""

    def __init__(self, root, launch_id, inputs, arm_provider, *, storage_limit=STORAGE_LIMIT_BYTES):
        self.root, self.launch_id, self.inputs = Path(root), launch_id, inputs
        launch_io.require_launch_root(self.root, launch_id)
        resolved, worktree = self.root.resolve(), WORKTREE.resolve()
        if resolved == worktree or worktree in resolved.parents:
            raise DriverError(f"the output root {resolved} lies inside the worktree")
        launch_io.require_free(self.root, storage_limit)
        self.arm_provider = arm_provider
        self.guard = launch_io.StorageGuard(self.root, storage_limit)
        self.relaxer_sha256 = relaxer_sha256()
        self.reserve_bytes = run_reserve_bytes(max(np.asarray(e["positions"]).shape[-2]
                                                   for e in inputs.starts.values()))
        self._arms, self._plans = {}, {}

    def arm(self, model_id):
        if model_id not in self._arms:
            self._arms[model_id] = self.arm_provider(model_id)
        return self._arms[model_id]

    def plans(self, model_id):
        if model_id not in self._plans:
            i = self.inputs
            plans, bound = prevalidate(i.starts, self.arm(model_id), i.procedure, i.classification,
                                       i.refs, i.policy, i.manifest)
            self._plans[model_id] = ({plan[0]: plan for plan in plans}, bound)
        return self._plans[model_id]


def run_original(context, task):
    """One original start through the copied second pass. Returns (frozen-shaped row,
    attempt records, elapsed seconds)."""
    inputs = context.inputs
    plans, bound = context.plans(task["model_id"])
    arm = context.arm(task["model_id"])
    records = []
    started = time.perf_counter()
    rows = second_pass([plans[task["key"]]], arm, inputs.procedure, inputs.refs, inputs.policy,
                       inputs.seen, bound, movement_cap_A=task["cap"], recorder_for=lambda key: records.append)
    elapsed = time.perf_counter() - started
    row = rows[task["key"]]
    if "checkpoint_identity" in arm:
        row["_audit_record"]["arm"]["checkpoint_identity"] = arm["checkpoint_identity"]
    return row, records, elapsed


def classify_reference(context, task, result):
    """The outcome category of one reference relaxation, as the paired-trace harness scores
    it. Shared by run_reference and the frozen re-run, so both classify the same way."""
    from evaluation.outcomes import classify_outcome
    inputs = context.inputs
    own = inputs.refs[task["key"]]
    final = replace(own, record_id=own.record_id + "/trace-final", positions=result["final_positions"],
                    forces=np.zeros_like(own.forces))
    return classify_outcome(final, own, inputs.refs, inputs.policy,
                            budget_exhausted=result["budget_exhausted"], converged=result["converged"],
                            forces_finite=result["forces_finite"], stop_reason=result["failure_reason"],
                            seen_minimum=True)


def run_reference(context, task):
    """One reference start, as the paired-trace harness runs it. Returns (relaxation
    result, classification outcome, attempt records, elapsed seconds)."""
    from evaluation.runner import _arm_force_supplier
    inputs = context.inputs
    own = inputs.refs[task["key"]]
    arm = context.arm(task["model_id"])
    records = []
    started = time.perf_counter()
    supplier = _arm_force_supplier(arm, own.species, own.record_id + "/reference-control")
    result = relax_capped(own.positions, supplier, inputs.procedure, arm=arm,
                          movement_cap_A=task["cap"], recorder=records.append)
    outcome = classify_reference(context, task, result)
    return result, outcome, records, time.perf_counter() - started


def execute_task(context, task):
    """Run one task, write its trace and row, return what the comparators need."""
    context.guard.check(next_bytes=context.reserve_bytes)
    paths = launch_io.output_paths(context.root, task["stage"], task["model_id"],
                                   row_schema.cap_label(task["cap"]), task["kind"], start_id_of(task))
    common = dict(context=context, task=task)
    if task["kind"] == "original":
        raw, records, elapsed = run_original(context, task)
        entry = context.inputs.starts[task["key"]]
        own_minimum = minimum_number(context.plans(task["model_id"])[0][task["key"]][4].record_id)
        outcome = dict(category=raw["category"], stop_reason=raw["stop_reason"], converged=raw["converged"],
                       budget_exhausted=raw["budget_exhausted"], step_count=raw["step_count"],
                       force_evaluation_count=raw["force_evaluation_count"], connection=entry["connection_id"],
                       branch=entry["branch"], own_minimum=own_minimum, cap_record=raw["cap_record"],
                       final_positions=raw["_audit_record"]["final_positions"])
        compared = raw
    else:
        result, scored, records, elapsed = run_reference(context, task)
        outcome = dict(category=scored["category"], stop_reason=result["failure_reason"],
                       converged=result["converged"], budget_exhausted=result["budget_exhausted"],
                       step_count=result["step_count"], force_evaluation_count=result["force_evaluation_count"],
                       connection=None, branch=None, own_minimum=minimum_number(task["key"]),
                       cap_record=result["cap_record"], final_positions=result["final_positions"])
        compared = result
    category = outcome["category"]
    n_atoms = np.asarray(records[0]["positions"]).shape[0]
    paths["trace"].parent.mkdir(parents=True, exist_ok=True)
    try:
        trace = row_schema.trace_from_records(records, n_atoms)
    except (ValueError, OverflowError) as error:
        raise DriverError(f"{task_id(task)}: the attempt records do not form a valid trace: {error}") from error
    trace_sha = row_schema.write_trace(paths["trace"], trace)
    try:
        row = make_result_row(**common, records=records, elapsed=elapsed, trace_file=paths["relative_trace"],
                              trace_sha256=trace_sha, **outcome)
    except (ValueError, OverflowError) as error:
        raise DriverError(f"{task_id(task)}: the result row is invalid: {error}") from error
    launch_io.write_row(paths["row"], row)
    return {"row": row, "records": records, "compared": compared, "category": category, "paths": paths}


# ---- Control index and the prefix-equality gate --------------------------

class ControlIndex:
    """Control (c = infinity) attempt records by (model, start kind, start id)."""

    def __init__(self):
        self.records = {}

    @classmethod
    def from_roots(cls, roots, stages=CONTROL_STAGES, *, allow_custom=False):
        """Read the control traces of every stage folder present under the roots. A folder is
        refused unless it holds STAGE_COMPLETE.json (a full run of the fixed schedule, unless
        allow_custom for tests) and no INCOMPLETE.json; its cinf rows must be as many as the stage's
        runs, every row must carry the current relaxer's SHA-256, every trace must match its row
        (hash and length), and no start may appear twice."""
        index = cls()
        for root in roots:
            root = Path(root)
            for stage in stages:
                directory = root / "rows" / f"stage{stage}"
                if not directory.is_dir():
                    continue
                if (directory / "INCOMPLETE.json").exists():
                    raise MissingControl(f"{directory} is marked INCOMPLETE")
                marker = directory / "STAGE_COMPLETE.json"
                if not marker.is_file():
                    raise MissingControl(f"{directory} has no STAGE_COMPLETE.json")
                summary = json.loads(marker.read_text())
                if not summary.get("complete"):
                    if not (allow_custom and summary.get("custom_tasks")):
                        raise MissingControl(f"{directory} is not a complete run of the fixed schedule")
                paths = sorted(directory.glob("*/cinf/*.json"))
                if len(paths) != summary["runs"]:
                    raise MissingControl(f"{directory}: {len(paths)} control rows, the stage recorded {summary['runs']}")
                for path in paths:
                    row = row_schema.loads_row(path.read_text())
                    if row["relaxer_sha256"] != relaxer_sha256():
                        raise MissingControl(f"{path}: the row was made by a different relaxer")
                    trace = row_schema.verify_trace_link(row, root)
                    key = (row["checkpoint_id"], row["start_kind"], row["start_id"])
                    if key in index.records:
                        raise MissingControl(f"two control rows for {key}")
                    index.records[key] = row_schema.records_from_trace(trace)
        return index

    def get(self, task):
        key = (task["model_id"], task["kind"], start_id_of(task))
        if key not in self.records:
            raise MissingControl(f"no control trace for {key}")
        return self.records[key]


def check_against_control(task, records, control_index):
    """The prefix-equality gate for one capped run; raises PrefixGateFailure."""
    report = prefix_gate.check_prefix(control_index.get(task), records, task["cap"])
    if report["problems"]:
        raise PrefixGateFailure(f"{task_id(task)}: " + "; ".join(report["problems"]))
    return report


# ---- Stage ---------------------------------------------------------------

def control_summary(traces):
    """Stage 0 and 0b: how often accepted steps exceed 0.25 and 1.0 A, and whether c = 1.0
    would bind anywhere (if not, c1p00 duplicates the control and is reported as such)."""
    fractions = [accepted_step_fractions(records) for records in traces.values()]
    return {"runs": len(traces),
            "mean_accepted_step_fraction_gt_0p25": float(np.mean([f[0] for f in fractions])) if fractions else None,
            "mean_accepted_step_fraction_gt_1p0": float(np.mean([f[1] for f in fractions])) if fractions else None,
            "runs_where_c1p00_would_bind": sum(prefix_gate.would_bind(r, 1.0) for r in traces.values()),
            "c1p00_duplicates_control": prefix_gate.duplicates_control(list(traces.values()), 1.0)}


def _json_bytes(payload):
    return (json.dumps(payload, indent=1, allow_nan=False, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path, payload):
    launch_io.atomic_write_bytes(path, _json_bytes(payload))


def _count(values):
    counts = {}
    for value in values:
        key = "none" if value is None else str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def stage_manifest(stage, context, tasks, rows, *, custom, controller, match_lines, saved_hashes, control):
    """What a finished stage records about itself: the run counts, each model's arm id and
    checkpoint identity, the relaxer's SHA-256, outcome and stop-reason counts, refused trials, and for
    Stage 0 the verdict, the compared count, the saved-file hashes and one match line per task."""
    arms = {}
    for model_id in dict.fromkeys(t["model_id"] for t in tasks):
        arm = context.arm(model_id)
        arms[model_id] = {"arm_id": arm["arm_id"], "checkpoint_identity": arm.get("checkpoint_identity")}
    manifest = {"stage": stage, "launch_id": context.launch_id, "custom_tasks": custom,
                "runs": len(rows), "relaxer_sha256": context.relaxer_sha256, "arms": arms,
                "outcome_categories": _count(r["outcome_category"] for r in rows),
                "stop_reasons": _count(r["stop_reason"] for r in rows),
                "refused_trial_count": sum(r["refused_trial_count"] for r in rows)}
    if control is not None:
        manifest["control"] = control
    if stage == "0":
        manifest["stage0"] = {"verdict": controller.verdict, "compared": controller.compared,
                              "matches": match_lines, "saved_file_hashes": saved_hashes}
    return manifest


def run_stage(stage, context, *, ledger=None, saved=None, controller=None, control_index=None,
              deadline=None, tasks=None, saved_hashes=None, allow_unledgered=False,
              allow_custom_tasks=False):
    """Run one stage to completion or to its first stop.

    ledger: the StageLedger (start and finish are recorded); required unless allow_unledgered
    (tests). saved: (saved original rows, saved reference items) for Stage 0. controller: the
    Stage0Controller, which must expect all 80 runs. saved_hashes: the saved-file hashes the
    launch record names, kept in the Stage 0 manifest. control_index: required for capped stages.
    deadline: a time.perf_counter() value. tasks: an explicit task list; tests only, and refused
    unless allow_custom_tasks. A custom run writes complete: false in STAGE_COMPLETE.json.

    Every refusal that needs no force call, including building every model's arm and plans, is
    raised before the ledger starts the stage."""
    spec = STAGE_SPECS[stage]
    if ledger is None and not allow_unledgered:
        raise DriverError("a stage runs under a StageLedger (allow_unledgered is for tests only)")
    if tasks is not None and not allow_custom_tasks:
        raise DriverError("an explicit task list is for tests only; a launch runs the fixed schedule")
    custom = tasks is not None
    if stage == "0":
        if saved is None or controller is None:
            raise DriverError("Stage 0 needs the saved results and a Stage0Controller")
        if not custom and controller.expected_runs != spec["runs"]:
            raise DriverError(f"Stage 0 compares {spec['runs']} runs; the controller expects {controller.expected_runs}")
        if not custom and (not isinstance(saved_hashes, dict) or len(saved_hashes) != SAVED_FILE_COUNT
                           or not all(isinstance(v, str) and len(v) == 64 for v in saved_hashes.values())):
            raise DriverError(f"Stage 0 needs the SHA-256 of all {SAVED_FILE_COUNT} saved files (saved_hashes), "
                              "the ones the launch record names")
    if stage in ("1", "2") and control_index is None:
        raise DriverError("capped stages need the control index")
    if ledger is not None and ledger.blockers(stage):
        raise StageOrderError("; ".join(ledger.blockers(stage)))      # before any arm is built
    tasks = tasks if custom else build_schedule(stage, sorted(context.inputs.starts),
                                                context.inputs.reference_ids)
    if stage == "0":
        for kind, table in (("original", saved[0]), ("reference", saved[1])):
            missing = [task_id(t) for t in tasks if t["kind"] == kind and (t["model_id"], t["key"]) not in table]
            if missing:
                raise DriverError(f"{len(missing)} scheduled Stage 0 tasks have no saved {kind} result, "
                                  f"first {missing[0]}")
    for model_id in dict.fromkeys(t["model_id"] for t in tasks):
        context.plans(model_id)                       # every arm built and checked before any force call
        try:
            json.dumps(context.arm(model_id).get("checkpoint_identity"), allow_nan=False)
        except (TypeError, ValueError) as error:      # the stage manifest must be writable
            raise DriverError(f"{model_id}: the checkpoint identity is not plain JSON: {error}") from error
    stage_directory = context.root / "rows" / f"stage{stage}"
    if stage_directory.exists():
        raise FileExistsError(f"{stage_directory} exists; a spent stage directory is never reused")
    if ledger is not None:
        ledger.start(stage)
    stage_directory.mkdir(parents=True, exist_ok=False)
    completed, rows, match_lines, attempts, current, control_traces = [], [], [], 0, None, {}
    try:
        for task in tasks:
            current = task_id(task)
            if deadline is not None and time.perf_counter() >= deadline:
                raise DeadlineReached("the stage's numerical deadline was reached")
            executed = execute_task(context, task)
            attempts += executed["row"]["force_evaluation_count"]
            if attempts > spec["attempt_ceiling"]:
                raise AttemptCeilingExceeded(f"{attempts} attempts exceed the stage ceiling "
                                             f"{spec['attempt_ceiling']}")
            if stage in CONTROL_STAGES:
                control_traces[current] = executed["records"]
            if stage == "0":
                if task["kind"] == "original":
                    problems = compare_original(saved[0][(task["model_id"], task["key"])],
                                                executed["compared"])["problems"]
                else:
                    problems = compare_reference(saved[1][(task["model_id"], task["key"])],
                                                 executed["compared"], executed["category"])["problems"]
                verdict = controller.observe(current, problems)
                if verdict is not None:
                    _write_json(stage_directory / "STAGE0_MISMATCH.json",
                                {"task": current, "problems": problems, "verdict": verdict,
                                 "compared_before": controller.compared})
                    raise Stage0Stop(verdict, current, problems)
                match_lines.append({"task": current, "matched": True})
            if stage in ("1", "2"):
                check_against_control(task, executed["records"], control_index)
            completed.append(current)                 # only a run that passed every gate is completed
            rows.append(executed["row"])
        if stage == "0":
            verdict = controller.finish()
            if not verdict["proceed"]:
                raise Stage0Stop(verdict)
        if not custom and len(completed) != spec["runs"]:
            raise DriverError(f"stage {stage} completed {len(completed)} runs, expected {spec['runs']}")
        control = control_summary(control_traces) if stage in CONTROL_STAGES else None
        manifest = _json_bytes(stage_manifest(stage, context, tasks, rows, custom=custom, controller=controller,
                                              match_lines=match_lines, saved_hashes=saved_hashes,
                                              control=control))
        launch_io.atomic_write_bytes(stage_directory / "STAGE_MANIFEST.json", manifest)
        summary = {"stage": stage, "runs": len(completed), "force_evaluations": attempts,
                   "attempt_ceiling": spec["attempt_ceiling"], "complete": not custom, "custom_tasks": custom,
                   "manifest_sha256": hashlib.sha256(manifest).hexdigest()}
        if control is not None:
            summary["control"] = control
        _write_json(stage_directory / "STAGE_COMPLETE.json", summary)
        if ledger is not None:
            ledger.finish(stage, True)
        return summary
    except BaseException as exc:
        launch_io.mark_incomplete(stage_directory, f"{type(exc).__name__}: {exc}", completed, current)
        if ledger is not None and ledger.state.get(stage) == "running":
            ledger.finish(stage, False)
        raise


def run_frozen_rerun(context, task, saved, controller):
    """The single pre-authorized re-run of the first mismatching Stage 0 task with the
    frozen relax in the same runtime. Returns the controller's verdict."""
    from evaluation.relax import relax
    from evaluation.runner import _arm_force_supplier, evaluate_starts
    controller.authorize_rerun(task_id(task))       # spend the one re-run before any force call
    inputs = context.inputs
    arm = context.arm(task["model_id"])
    if task["kind"] == "original":
        result = evaluate_starts({task["key"]: inputs.starts[task["key"]]}, arm, inputs.procedure,
                                 inputs.classification, inputs.refs, inputs.policy,
                                 split_manifest=inputs.manifest, seen_minimum=inputs.seen,
                                 authorise_test_partition=False)
        problems = compare_original(saved[0][(task["model_id"], task["key"])],
                                    result["branches"][task["key"]])["problems"]
    else:
        own = inputs.refs[task["key"]]
        supplier = _arm_force_supplier(arm, own.species, own.record_id + "/reference-control")
        relaxed = relax(own.positions, supplier, inputs.procedure, arm=arm)
        problems = compare_reference(saved[1][(task["model_id"], task["key"])], relaxed,
                                     classify_reference(context, task, relaxed)["category"])["problems"]
    return controller.record_frozen_rerun(task_id(task), not problems)
