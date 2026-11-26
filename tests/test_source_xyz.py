import io
import zipfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "data/landscape17_sample/aspirin/SteepestDescent/01/coordsplus_000.xyz"


def parse(text):
    from force_consistency.data.xyz import parse_xyz
    return parse_xyz(text, source_path="Landscape17/aspirin/SteepestDescent/01/coordsplus_000.xyz", molecule="aspirin")


def test_real_schema_and_reordered_fields():
    text = EXAMPLE.read_text()
    a = parse(text)
    lines = text.splitlines()
    lines[1] = lines[1].replace("species:S:1:pos:R:3:forces:R:3", "forces:R:3:species:S:1:pos:R:3")
    lines[2:] = [" ".join(row.split()[4:] + row.split()[:4]) for row in lines[2:]]
    b = parse("\n".join(lines))
    np.testing.assert_array_equal(a.positions, b.positions)
    np.testing.assert_array_equal(a.forces, b.forces)
    assert a.species == b.species and a.energy == b.energy
    assert a.connection_id == "01" and a.frame_index == 0 and a.branch == "plus"
    assert a.positions.dtype == np.float64


@pytest.mark.parametrize("mutation", ["count", "periodic", "duplicate", "missing", "shape", "nan", "truncated", "extra"])
def test_reject_bad_records(mutation):
    from force_consistency.data.xyz import SchemaError
    s = EXAMPLE.read_text()
    if mutation == "count": s = s.replace("21\n", "20\n", 1)
    if mutation == "periodic": s = s.replace('F F F', 'T F F')
    if mutation == "duplicate": s = s.replace("pos:R:3", "species:R:3")
    if mutation == "missing": s = s.replace("forces:R:3", "velocity:R:3")
    if mutation == "shape": s = s.replace("forces:R:3", "forces:R:2")
    if mutation == "nan": s = s.replace("2.29217661", "nan", 1)
    if mutation == "truncated": s = "\n".join(s.splitlines()[:-1])
    if mutation == "extra": s += "unexpected row\n"
    with pytest.raises(SchemaError): parse(s)


@pytest.mark.parametrize("name,mode", [("../outside.xyz", 0), ("/absolute.xyz", 0), ("C:/outside.xyz", 0), ("link.xyz", 0o120777 << 16)])
def test_unsafe_archive_rejected(name, mode):
    from force_consistency.data.source import validate_members, SourceError
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        entry = zipfile.ZipInfo(name); entry.external_attr = mode
        z.writestr(entry, "payload")
    with zipfile.ZipFile(io.BytesIO(b.getvalue())) as z, pytest.raises(SourceError): validate_members(z)


def test_source_checksum_rejects_corruption(tmp_path):
    from force_consistency.data.source import verify_archive, SourceError
    p = tmp_path / "bad.zip"; p.write_bytes(b"not Landscape17")
    with pytest.raises(SourceError): verify_archive(p)


def test_all_local_sample_records():
    from force_consistency.data.xyz import parse_xyz
    files = list((ROOT / "data/landscape17_sample").rglob("*.xyz"))
    assert len(files) == 405
    for p in files:
        mol = p.relative_to(ROOT / "data/landscape17_sample").parts[0]
        r = parse_xyz(p.read_text(), source_path="Landscape17/"+str(p.relative_to(ROOT / "data/landscape17_sample")), molecule=mol)
        assert len(r.species) == {"aspirin": 21, "paracetamol": 20}[mol]
def test_correct_size_wrong_checksum_is_rejected(tmp_path):
    from force_consistency.data.source import SIZE,SourceError,verify_archive
    import pytest
    p=tmp_path/'wrong.zip'
    with p.open('wb') as f: f.truncate(SIZE)
    with pytest.raises(SourceError,match='[Cc]hecksum|SHA'):
        verify_archive(p)
