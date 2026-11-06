"""Plan v3 section 9: the real arm provider hands each model its own binding and loads nothing at import."""
import subprocess
import sys

import pytest

from stage2_capped import launch_arms
from stage2_capped.tests.launch_fixtures import REPO

ITEMS = [{"id": "energy-0", "run_directory": "/runs/e", "launch_record": {"x": 1}, "score_directory": "/scores/e",
          "acceptance": {"best_update": 1}},
         {"id": "direct-0", "run_directory": "/runs/d", "launch_record": {"x": 2}, "score_directory": "/scores/d",
          "acceptance": {"best_update": 2}}]


def test_each_model_is_restored_with_its_own_binding_and_the_shared_policy():
    calls = []
    policy, directory = object(), "/data/processed/aspirin"
    provider = launch_arms.real_arm_provider(ITEMS, directory, policy, restore=lambda **kw: calls.append(kw) or kw["run_directory"])
    assert provider("direct-0") == "/runs/d" and provider("energy-0") == "/runs/e"
    assert calls[0] == {"run_directory": "/runs/d", "launch_record": {"x": 2}, "score_directory": "/scores/d",
                        "acceptance": {"best_update": 2}, "data_directory": directory, "pin_policy": policy}
    assert calls[1]["acceptance"] == {"best_update": 1} and calls[1]["pin_policy"] is policy


def test_an_unknown_model_is_refused_before_any_restore():
    calls = []
    provider = launch_arms.real_arm_provider(ITEMS, "/d", object(), restore=lambda **kw: calls.append(kw))
    with pytest.raises(ValueError, match="no accepted binding"):
        provider("consistency-9-9")
    assert calls == []


def test_importing_the_provider_module_loads_no_restore_code():
    done = subprocess.run([sys.executable, "-c", "import sys; import stage2_capped.launch_arms as m; "
                           "print('stage2_capped.restore_fork' in sys.modules)"], capture_output=True, cwd=REPO,
                          env={"PYTHONPATH": f"{REPO}:{REPO / 'src'}:{REPO / 'scripts'}", "PYTHONDONTWRITEBYTECODE": "1",
                                "PATH": "/usr/bin:/bin"})
    assert done.stdout.strip() == b"False", done.stderr.decode()[-300:]


def test_the_default_restore_is_the_forks_checkpoint_arm_when_a_model_is_first_asked_for(monkeypatch):
    import stage2_capped.restore_fork as fork
    seen = []
    monkeypatch.setattr(fork, "e3x_checkpoint_arm", lambda **kw: seen.append(kw["run_directory"]) or "arm")
    assert launch_arms.real_arm_provider(ITEMS, "/d", object())("energy-0") == "arm" and seen == ["/runs/e"]
