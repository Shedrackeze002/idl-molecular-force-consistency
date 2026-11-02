"""Re-export the repository's synthetic-landscape fixtures (tests/conftest.py) for the
Stage 2 tests, without importing that file under the name `conftest`."""
import importlib.util
import sys
from pathlib import Path

_REPO_CONFTEST = Path(__file__).resolve().parents[2] / "tests" / "conftest.py"
_spec = importlib.util.spec_from_file_location("repo_tests_conftest", _REPO_CONFTEST)
_module = importlib.util.module_from_spec(_spec)
sys.modules["repo_tests_conftest"] = _module
_spec.loader.exec_module(_module)

synthetic_landscape = _module.synthetic_landscape
frozen_synthetic_manifest = _module.frozen_synthetic_manifest
synthetic_endpoint_classification = _module.synthetic_endpoint_classification


# ---- STAGE2_REQUIRE_REAL_FILES=1 turns every skip into a failure ----------
# Tests that read the saved Group9 files skip when those files are absent. A launch
# record may only quote a test count that ran against the real files, so with this
# variable set a skipped test is a failed one (the count then reads "N passed, 0 skipped").
import os  # noqa: E402

import pytest  # noqa: E402


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    report = outcome.get_result()
    if os.environ.get("STAGE2_REQUIRE_REAL_FILES") == "1" and report.skipped:
        report.outcome = "failed"
        report.longrepr = ("skipped, and STAGE2_REQUIRE_REAL_FILES=1 requires the saved Group9 files: "
                           f"{item.nodeid}")
