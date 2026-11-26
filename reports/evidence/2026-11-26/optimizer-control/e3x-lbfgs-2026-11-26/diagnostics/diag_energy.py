"""Scratch diagnostic: energy-0 model energies at starts, L-BFGS endpoints and the saved custom-search
(combined, cap 0.05) endpoints. Energy is read only here, after the runs; the optimizer never saw it."""
import sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, sys.argv[1])
import e3x_lbfgs as E
from energy_decrease.run import schedule

class A: pass
args = A(); args.group9 = Path(sys.argv[2]); args.admission = Path(sys.argv[3])
rows = [json.loads(l) for l in Path(sys.argv[4]).read_text().splitlines() if l.strip()]
combined = Path(sys.argv[5])
out = []
for cid, inputs, arm, parity, identity in E.setup_e3x_all(args, ["energy-0"]):
    tasks = list(schedule(inputs))
    for index, (kind, key, own, start) in enumerate(tasks):
        label = key if start is not None else E.label_of(key)
        x0 = np.array(start["positions"] if start is not None else own.positions)
        rid = "diag"
        e = lambda x: arm["energy"](np.asarray(x, dtype=np.float64), species=own.species, record_id=rid)[0]
        fmax = lambda x: float(np.abs(arm["forces"](np.asarray(x, dtype=np.float64), species=own.species, record_id=rid)[0]).max())
        mine = {r["fmax_norm_eV_per_A"]: r for r in rows if r["checkpoint_id"] == cid and r["start_id"] == label}
        saved = json.loads((combined / f"{index:02d}.json").read_text())
        assert saved["start_id"] == label, (saved["start_id"], label)
        rec = {"start": label, "kind": kind, "E_start": e(x0)}
        for f, r in sorted(mine.items(), reverse=True):
            x = np.array(r["final_positions"])
            rec[f"E_lbfgs_{f}"] = e(x) if r["endpoint_connected_5A"] else None
        xc = np.array(saved["final_positions"])
        rec["E_custom"] = e(xc); rec["custom_category"] = saved["outcome_category"]; rec["custom_maxF"] = fmax(xc)
        out.append(rec)
        def d(k):
            return "   n/a" if rec.get(k) is None else f"{(rec[k] - rec['E_start']) * 1000:8.2f}"
        print(f"{label:20s} dE meV vs start: lbfgs0.01 {d('E_lbfgs_0.01')} lbfgs0.001 {d('E_lbfgs_0.001')} "
              f"custom {d('E_custom')} ({rec['custom_category']}, maxF {rec['custom_maxF']:.4f})")
Path(sys.argv[6]).write_text(json.dumps(out, indent=1) + "\n")
