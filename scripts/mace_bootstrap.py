"""Isolated, site-hook-free entry point for production parity commands."""
from pathlib import Path
import runpy
import sys


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
    sys.path.extend([str(packages), str(scripts), str(scripts.parent), str(scripts.parent / "src")])
    target = scripts / entries[sys.argv[2]]
    sys.argv = [str(target), *sys.argv[3:]]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
