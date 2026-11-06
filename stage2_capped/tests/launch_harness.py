"""A launch harness for the end-to-end tests: a committed scratch tree, records A and B with their approvals, and a
way to run the real parent (`launch_test_parent`) as a subprocess from that tree, as a launch would."""
import json
import os
import subprocess
import sys
from pathlib import Path

from stage2_capped import admission, launch_checks, launch_record
from stage2_capped.tests.launch_fixtures import build_record, write_approval, write_record


class Harness:
    def __init__(self, scratch, tmp_path, *, ceilings=None):
        self.scratch, self.tmp = Path(scratch), Path(tmp_path)
        self.folder = self.tmp / "launch"
        self.records = self.folder / "records"
        self.registry = self.folder / "registry"
        self.group9 = admission.group9_root()
        volume = launch_checks.mount_point(self.tmp)
        self.root_a, self.root_b = self.tmp / "outA", self.tmp / "outB"
        self.record_a = build_record(self.scratch, kind="A", launch_id="A-e2e", root=self.root_a, volume=volume,
                                     launch_folder=self.folder, ceilings=ceilings)
        self.path_a, self.sha_a = write_record(self.records, self.record_a)
        self.approval_a = write_approval(self.records, self.sha_a, "approval-a")
        self.record_b = build_record(self.scratch, kind="B", launch_id="B-e2e", root=self.root_b, volume=volume,
                                     launch_folder=self.folder, ceilings=ceilings,
                                     a0={"record_a_sha256": self.sha_a, "a0_launch_id": "A-e2e",
                                         "a0_output_root": str(self.root_a)})
        self.path_b, self.sha_b = write_record(self.records, self.record_b)
        self.approval_b = write_approval(self.records, self.sha_b, "approval-b")
        self.fixture_dir = self.tmp / "fixture"
        self.fixture_dir.mkdir()

    # -- running the parent ------------------------------------------------------------------------------------
    def environment(self, scenario=None):
        paths = [str(self.scratch), str(self.scratch / "src"), str(self.scratch / "scripts")]
        return {"PYTHONPATH": os.pathsep.join(paths), "PYTHONDONTWRITEBYTECODE": "1", "PATH": os.environ["PATH"],
                "HOME": os.environ.get("HOME", ""), "LAUNCH_TEST_DIR": str(self.fixture_dir),
                "LAUNCH_TEST_SCENARIO": json.dumps(scenario or {})}

    def run(self, *argv, scenario=None, timeout=600):
        return subprocess.run([sys.executable, "-m", "stage2_capped.tests.launch_test_parent", *argv], cwd=self.scratch,
                              env=self.environment(scenario), capture_output=True, text=True, timeout=timeout)

    def run_production(self, *argv, timeout=900):
        """The real command line, `python -m stage2_capped.launch`, with its production dependencies (the real
        model-free `admit_all`; no fixture is reachable from here)."""
        return subprocess.run([sys.executable, "-m", "stage2_capped.launch", *argv], cwd=self.scratch,
                              env=self.environment(), capture_output=True, text=True, timeout=timeout)

    def a0_arguments(self, command="a0"):
        head = ["check", "a0"] if command == "check" else ["a0"]
        return [*head, "--record", str(self.path_a), "--approval", str(self.approval_a), "--record-sha256", self.sha_a,
                "--group9-root", str(self.group9)]

    def stage_arguments(self, unit, *, a0=None, stage0=None, stage0b=None, stage1=None, record=None, check=False):
        path, approval, sha = (self.path_b, self.approval_b, self.sha_b) if record is None else record
        argv = (["check"] if check else []) + ["stage", unit, "--record", str(path), "--approval", str(approval),
                                                "--record-sha256", sha, "--group9-root", str(self.group9)]
        for flag, value in (("a0-result-sha256", a0), ("stage0-sha256", stage0), ("stage0b-sha256", stage0b),
                            ("stage1-sha256", stage1)):
            if value:
                argv += [f"--{flag}", value]
        return argv

    # -- reading what a launch left ---------------------------------------------------------------------------------
    def a0_result_sha(self):
        return launch_record.sha256_bytes((self.root_a / "A0_RESULT.json").read_bytes())

    def marker_sha(self, stage):
        return launch_record.sha256_bytes((self.root_b / "rows" / f"stage{stage}" / "STAGE_COMPLETE.json").read_bytes())

    def events(self, record_sha=None):
        out = []
        for path in sorted(self.registry.glob("*.json")):
            if record_sha is None or path.name.startswith(record_sha):
                out.append((path.name, json.loads(path.read_text())))
        return out

    def run_a0_and_stage0(self, scenario=None):
        first = self.run(*self.a0_arguments(), scenario=scenario)
        assert first.returncode == 0, first.stdout + first.stderr
        second = self.run(*self.stage_arguments("0", a0=self.a0_result_sha()), scenario=scenario)
        assert second.returncode == 0, second.stdout + second.stderr
        return first, second
