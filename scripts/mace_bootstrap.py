"""Isolated, site-hook-free entry point for production parity commands."""
from pathlib import Path
from importlib.machinery import ModuleSpec
from importlib.util import spec_from_file_location
import runpy
import sys


class ReviewedProjectModules:
    """A path-entry finder exposing only named project files, never a directory."""

    def __init__(self, scripts, entry):
        self.entry = entry
        self.files = {name: scripts / f"{name}.py" for name in (
            "mace_loader", "mace_reference_adapter", "check_mace_parity")}
        self.files["evaluation.adapter"] = scripts.parent / "evaluation/adapter.py"

    def find_spec(self, fullname, target=None):
        if fullname == "evaluation":
            # The bridge needs only the pinned adapter. Do not execute an
            # unpinned initializer or expose other evaluation directory files.
            spec = ModuleSpec(fullname, loader=self, is_package=True)
            spec.submodule_search_locations = [self.entry]
            return spec
        path = self.files.get(fullname)
        return spec_from_file_location(fullname, path) if path is not None else None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        # The evaluation package is a sealed container for evaluation.adapter.
        # A concrete loader also prevents a later installed package from winning
        # over this entry, as it could with a namespace-package candidate.
        if module.__name__ != "evaluation":
            raise ImportError("unexpected project package")


def main():
    if not sys.flags.isolated or not sys.flags.no_site:
        raise SystemExit("MACE commands require python -I -S scripts/mace_bootstrap.py")
    if len(sys.argv) < 3 or sys.argv[1] != "--entry":
        raise SystemExit("expected --entry supervisor|worker|audit")
    scripts = Path(__file__).resolve().parent
    entries = {"supervisor": "supervise_mace_parity.py", "worker": "check_mace_parity.py",
               "audit": "audit_mace_parity.py"}
    if sys.argv[2] not in entries:
        raise SystemExit("unknown parity entry point")
    # -S prevents sitecustomize, usercustomize and executable .pth files. Add the
    # selected venv's packages explicitly; never call site.main()/addsitedir().
    venv = Path(sys.executable).absolute().parent.parent
    packages = (venv / "Lib/site-packages" if sys.platform == "win32" else
                venv / f"lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages")
    if not (venv / "pyvenv.cfg").is_file() or not packages.is_dir():
        raise SystemExit("a dedicated virtual environment with site-packages is required")
    # PathFinder can resolve the exact project names above, but cannot search
    # scripts/root/src for e3nn, scipy, numpy or any other dependency.
    project_entry = str(scripts / ".reviewed-module-map")
    finder = ReviewedProjectModules(scripts, project_entry)
    sys.path_importer_cache[project_entry] = finder
    sys.path.extend([project_entry, str(packages)])
    target = scripts / entries[sys.argv[2]]
    sys.argv = [str(target), *sys.argv[3:]]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    raise SystemExit("Direct bootstrap invocation is refused; use the external trusted launcher")
