"""Stage gating and the Stage 0 mismatch procedure (spec v4 sections 5 and 7). Pure state
machines: no I/O, no model.

Stage start conditions (spec section 5):
- A0 (admission only, no relaxation): none.
- Stage 0 (control, primary five): A0 complete.
- Stage 0b (control, lambda = 1): Stage 0 complete.
- Stage 1 (capped, primary five): Stage 0 complete.
- Stage 2 (capped, lambda = 1): Stage 1 complete, Stage 0b complete, and the A0 replay passed.
A stage runs at most once per launch: there is no retry and no resume, so a stage
that is not pending can never start again. One stage runs at a time.

Stage 0 mismatch procedure (fixed now, spec section 7): fail fast at the first
mismatch. The launch record pre-authorizes exactly one re-run of that single task
with the frozen relax in the same runtime. Frozen differs from the saved result:
runtime drift, stop. Frozen matches the saved result but c = infinity does not:
code error, stop. Any tolerance, further re-run or continuation needs a new reviewed
launch record. The verdicts are rules.r_control's.
"""
from stage2_capped import rules

STAGE_ORDER = ("A0", "0", "0b", "1", "2")
PENDING, RUNNING, COMPLETE, FAILED = "pending", "running", "complete", "failed"

PREREQUISITES = {"A0": (), "0": ("A0",), "0b": ("0",), "1": ("0",), "2": ("1", "0b")}


class StageOrderError(RuntimeError):
    """A stage was started out of order, twice, or while another was running."""


class Stage0Halted(RuntimeError):
    """Stage 0 has already stopped; nothing may continue under this record."""


class StageLedger:
    def __init__(self):
        self.state = {stage: PENDING for stage in STAGE_ORDER}
        self.a0_replay_passed = False
        self.events = []

    def blockers(self, stage):
        """Why `stage` cannot start now (empty list means it can)."""
        if stage not in PREREQUISITES:
            return [f"unknown stage {stage!r}"]
        reasons = []
        if self.state[stage] != PENDING:
            reasons.append(f"stage {stage} is {self.state[stage]}; a stage runs once, with no retry or resume")
        running = [s for s, st in self.state.items() if st == RUNNING]
        if running:
            reasons.append(f"stage {running[0]} is still running")
        for needed in PREREQUISITES[stage]:
            if self.state[needed] != COMPLETE:
                reasons.append(f"stage {needed} is not complete ({self.state[needed]})")
        if stage == "2" and not self.a0_replay_passed:
            reasons.append("the A0 replay has not passed")
        return reasons

    def start(self, stage):
        reasons = self.blockers(stage)
        if reasons:
            raise StageOrderError("; ".join(reasons))
        self.state[stage] = RUNNING
        self.events.append((stage, "started"))

    def finish(self, stage, ok, *, replay_passed=None):
        if self.state.get(stage) != RUNNING:
            raise StageOrderError(f"stage {stage} is not running")
        if stage == "A0":
            ok = bool(ok and replay_passed is True)
            self.a0_replay_passed = ok
        self.state[stage] = COMPLETE if ok else FAILED
        self.events.append((stage, "complete" if ok else "failed"))

    def snapshot(self):
        return {"state": dict(self.state), "a0_replay_passed": self.a0_replay_passed,
                "events": [list(e) for e in self.events]}


class Stage0Controller:
    """Counts compared tasks and applies the fail-fast and single-re-run rules."""

    def __init__(self, expected_runs=rules.CONTROL_RUNS):
        self.expected_runs = expected_runs
        self.compared = 0
        self.mismatches = []
        self.mismatch_problems = None
        self.rerun_done = False
        self.halted = False
        self.verdict = None

    def observe(self, task_id, problems):
        """Record one compared task. Returns None while everything matches; at the first
        mismatch returns (and keeps) the 'frozen re-run required' verdict and halts."""
        if self.halted:
            raise Stage0Halted("Stage 0 already stopped; a new reviewed launch record is needed")
        if problems:
            self.mismatches.append(task_id)
            self.mismatch_problems = list(problems)
            self.halted = True
            self.verdict = rules.r_control(self.compared, self.mismatches,
                                           expected_runs=self.expected_runs)
            return self.verdict
        self.compared += 1
        return None

    def record_frozen_rerun(self, task_id, matches_saved):
        """The one pre-authorized re-run of the mismatching task with the frozen relax.
        Always leaves Stage 0 stopped."""
        if not self.halted or not self.mismatches:
            raise Stage0Halted("no mismatch to re-run")
        if self.rerun_done:
            raise Stage0Halted("the single pre-authorized re-run has been used")
        if task_id != self.mismatches[0]:
            raise Stage0Halted("the re-run covers only the first mismatching task")
        self.rerun_done = True
        self.verdict = rules.r_control(self.compared, self.mismatches,
                                       {"task": task_id, "matches_saved": bool(matches_saved)},
                                       expected_runs=self.expected_runs)
        return self.verdict

    def finish(self):
        """The final Stage 0 verdict: pass only if all runs compared and none differ."""
        if self.halted:
            return self.verdict
        self.verdict = rules.r_control(self.compared, [], expected_runs=self.expected_runs)
        return self.verdict
