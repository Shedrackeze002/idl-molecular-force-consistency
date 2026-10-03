from dataclasses import replace
from pathlib import Path
import numpy as np
import pytest

from force_consistency.data.xyz import parse_xyz

ROOT = Path(__file__).resolve().parents[1]


def records(group="010", pair="0_10"):
    root = ROOT / "data/landscape17_sample/aspirin"
    out = []
    for branch in ["plus", "minus"]:
        p = root / f"SteepestDescent/01/coords{branch}_000.xyz"
        out.append(parse_xyz(p.read_text(), source_path=f"Landscape17/aspirin/SteepestDescent/{group}/coords{branch}_000.xyz", molecule="aspirin"))
    for i in sorted(set(pair.split('_')[:2])):
        p = root / "Conformations/coordsmin_0.xyz"
        out.append(parse_xyz(p.read_text(), source_path=f"Landscape17/aspirin/Conformations/coordsmin_{i}.xyz", molecule="aspirin"))
    p = root / "Conformations/coordsts_0_1.xyz"
    out.append(parse_xyz(p.read_text(), source_path=f"Landscape17/aspirin/Conformations/coordsts_{pair}.xyz", molecule="aspirin"))
    return out


@pytest.mark.parametrize("group,pair", [("010","0_10"),("310","3_10"),("01_2","0_1_2"),("00","0_0")])
def test_string_identifiers_and_ts_lookup(group, pair):
    from force_consistency.data.audit import inventory_connections
    inv = inventory_connections(records(group,pair))
    c = inv.connections["aspirin"][group]
    assert c.connection_id == group
    assert c.transition_state.endswith(f"coordsts_{pair}.xyz")
    assert set(c.branches) == {"plus","minus"}


@pytest.mark.parametrize("problem", ["missing", "duplicate", "gap", "order", "ambiguous"])
def test_broken_connection_rejected(problem):
    from force_consistency.data.audit import inventory_connections, AuditError
    rs = records()
    if problem == "missing": rs = rs[1:]
    if problem == "duplicate": rs.append(rs[0])
    if problem == "gap": rs.append(replace(rs[0], record_id="gap", frame_index=2))
    if problem == "order": rs.append(replace(rs[0], record_id="order", frame_index=1, species=rs[0].species[::-1]))
    if problem == "ambiguous": rs.append(replace(rs[-1],record_id="Landscape17/aspirin/Conformations/coordsts_01_0.xyz",source_path="Landscape17/aspirin/Conformations/coordsts_01_0.xyz"))
    with pytest.raises(AuditError): inventory_connections(rs)


def test_units_need_evidence_and_explicit_scales():
    from force_consistency.data.units import verify_units, convert_record, UnitError
    with pytest.raises(UnitError): verify_units({})
    with pytest.raises(UnitError): verify_units({"source_units": {"energy":"kcal/mol"}})
    r = records()[0]
    evidence = {"source_units":{"positions":"angstrom","energy":"eV","forces":"eV/angstrom"},
                "sources":[{"url":"https://example.test/methods", "sha256":"a"*64,"location":"calculator energy/force conversion"}],
                "archive_sha256":"3870ecfd6a5404c6654efba22f003f645e2f4dffffad1b5a22770d3437b3027b"}
    contract = verify_units(evidence)
    out = convert_record(r, contract)
    np.testing.assert_array_equal(out.forces,r.forces)
    assert out.energy == r.energy
