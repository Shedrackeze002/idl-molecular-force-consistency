"""Fixed-atom-order Kabsch RMSD of each known-minimum control endpoint (fmax 0.001 row) from its DFT
starting minimum, for both optimizer runs. The DFT start is taken from the MACE validation rows whose
0.01 endpoint used one force call (no step was taken, so the endpoint is the start exactly).
Run with the Group9 venv from output/research/optimizer-control: python e3x-fire-2026-11-26/diagnostics/control_drift.py"""
import json, statistics, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
import geom
out = {}
for opt in ("e3x-lbfgs-2026-11-26", "e3x-fire-2026-11-26"):
    dft = {}
    for line in (HERE / opt / "mace-runA-validation/rows.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["start_kind"] == "reference" and r["fmax_norm_eV_per_A"] == 0.01 and r["force_calls"] == 1:
            dft[r["start_id"]] = np.array(r["final_positions"])
    assert len(dft) == 6, sorted(dft)
    out[opt] = {}
    for line in (HERE / opt / "e3x/rows.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["start_kind"] == "reference" and r["fmax_norm_eV_per_A"] == 0.001:
            out[opt].setdefault(r["checkpoint_id"], {})[r["start_id"]] = {
                "fixed_order_rmsd_from_dft_A": float(geom.kabsch_rmsd(np.array(r["final_positions"]), dft[r["start_id"]])),
                "connected_5A": r["endpoint_connected_5A"], "max_force": r["max_abs_force_component_eV_per_A"],
                "sym_rmsd_from_dft_A": r["sym_rmsd_to_target_A"], "category": r["outcome_category"]}
    for cid, d in out[opt].items():
        v = [x["fixed_order_rmsd_from_dft_A"] for x in d.values()]
        print(f"{opt} {cid:18s} {min(v):6.2f}-{max(v):6.2f} A (median {statistics.median(v):.2f}); "
              f"connected {sum(x['connected_5A'] for x in d.values())}/6")
(Path(__file__).with_name("control_drift.json")).write_text(json.dumps(out, indent=1) + "\n")
