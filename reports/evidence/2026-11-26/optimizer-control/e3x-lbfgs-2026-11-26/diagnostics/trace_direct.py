"""Scratch diagnostic: step-by-step L-BFGS trace on one known-minimum control (not a deliverable)."""
import sys, json, argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, sys.argv[1])
import e3x_lbfgs as E
from energy_decrease.run import schedule

class A: pass
args = A(); args.group9 = Path(sys.argv[2]); args.admission = Path(sys.argv[3])
ids = sys.argv[4].split(",")
for cid, inputs, arm, parity, identity in E.setup_e3x_all(args, ids):
    AC = E.make_calculator_class()
    tasks = [t for t in schedule(inputs) if t[0] == "reference"][:2]
    for kind, key, own, start in tasks:
        from ase import Atoms
        from ase.optimize import LBFGS
        x0 = np.array(own.positions, dtype=np.float64)
        atoms = Atoms(symbols=list(own.species), positions=x0.copy())
        calc = AC(arm, own.species, key + "/trace"); atoms.calc = calc
        opt = LBFGS(atoms, logfile=None)
        log = []
        def obs():
            f = calc.results["forces"]; x = atoms.get_positions()
            s = opt.state.s[-1] if getattr(opt.state, "s", None) else None
            y = opt.state.y[-1] if getattr(opt.state, "y", None) else None
            ys = float(np.dot(np.ravel(s), np.ravel(y))) if s is not None and y is not None else None
            pairs = np.linalg.norm(x[:, None] - x[None], axis=-1)
            log.append((opt.nsteps, calc.calls, float(np.abs(f).max()),
                        float(np.linalg.norm(x - x0, axis=1).max()), float(pairs.max()), ys))
        opt.attach(obs, interval=1)
        try:
            opt.run(fmax=0.01, steps=299)
        except E.NonfiniteStop as e:
            print("nonfinite", e)
        neg = sum(1 for r in log if r[5] is not None and r[5] < 0)
        print(f"{cid} {Path(key).stem}: initial maxF {log[0][2]:.4f}, steps {opt.nsteps}, negative y.s updates {neg}")
        for r in log:
            if r[0] in (0, 1, 2, 5, 10, 20, 40, 60, 80, 100, 150, 200, 250, 299) or r is log[-1]:
                print(f"  step {r[0]:4d} calls {r[1]:4d} maxF {r[2]:10.4f} maxdisp {r[3]:8.3f} maxpair {r[4]:8.3f} y.s {r[5]}")
