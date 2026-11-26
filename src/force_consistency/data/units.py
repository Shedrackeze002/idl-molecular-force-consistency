from dataclasses import dataclass, replace
from typing import Mapping

from .source import SHA256
from .types import FrameRecord


class UnitError(ValueError):
    pass


@dataclass(frozen=True)
class UnitContract:
    positions: float
    energy: float
    forces: float
    evidence: dict


def verify_units(evidence: Mapping[str, object]) -> UnitContract:
    expected = {"positions":"angstrom", "energy":"eV", "forces":"eV/angstrom"}
    if evidence.get("source_units") != expected or evidence.get("archive_sha256") != SHA256:
        raise UnitError("Unresolved or unsupported source units/archive version")
    sources = evidence.get("sources", [])
    if not sources or any(not s.get("url", "").startswith("https://") or len(s.get("sha256", "")) != 64 or not s.get("location") for s in sources):
        raise UnitError("Missing versioned unit evidence")
    return UnitContract(1.0, 1.0, 1.0, dict(evidence))


def convert_record(record: FrameRecord, units: UnitContract) -> FrameRecord:
    p = record.positions * units.positions; f = record.forces * units.forces
    p.setflags(write=False); f.setflags(write=False)
    return replace(record, positions=p, forces=f, energy=record.energy*units.energy)
