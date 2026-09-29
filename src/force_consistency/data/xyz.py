import hashlib
import re
import shlex
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np

from .source import verify_archive
from .types import FrameRecord


class SchemaError(ValueError):
    pass


COMPOSITIONS = {"aspirin": Counter(C=9, H=8, O=4), "paracetamol": Counter(C=8, H=9, N=1, O=2)}


def parse_xyz(text: str, *, source_path: str, molecule: str) -> FrameRecord:
    try:
        lines = text.strip().splitlines()
        n = int(lines[0])
        if n <= 0 or len(lines) != n + 2:
            raise SchemaError("Atom count or record length mismatch")
        metadata = {}
        for part in shlex.split(lines[1]):
            key, sep, value = part.partition("=")
            if not sep or key in metadata:
                raise SchemaError("Malformed or duplicate header key")
            metadata[key] = value
        if metadata.get("pbc", "").upper().split() != ["F", "F", "F"]:
            raise SchemaError("Only explicitly nonperiodic records are supported")
        schema = metadata["Properties"].split(":")
        if len(schema) % 3:
            raise SchemaError("Invalid Properties schema")
        fields, width = {}, 0
        for i in range(0, len(schema), 3):
            name, kind, count = schema[i:i+3]; count = int(count)
            if name in fields or kind not in {"S", "R", "I", "L"} or count < 1:
                raise SchemaError("Invalid or duplicate property")
            fields[name] = (kind, slice(width, width+count)); width += count
        for name, kind, count in [("species", "S", 1), ("pos", "R", 3), ("forces", "R", 3)]:
            actual, sl = fields[name]
            if actual != kind or sl.stop-sl.start != count:
                raise SchemaError(f"Invalid {name} field")
        rows = [line.split() for line in lines[2:]]
        if any(len(row) != width for row in rows):
            raise SchemaError("Atom row does not match declared schema")
        species = tuple(row[fields["species"][1]][0] for row in rows)
        if molecule not in COMPOSITIONS or Counter(species) != COMPOSITIONS[molecule]:
            raise SchemaError("Unexpected molecular composition")
        for kind, sl in fields.values():
            for row in rows:
                for value in row[sl]:
                    if kind == "R" and not np.isfinite(float(value)):
                        raise SchemaError("Nonfinite numeric property")
                    if kind == "I": int(value)
                    if kind == "L" and value.upper() not in {"T", "F", "TRUE", "FALSE", "0", "1"}:
                        raise SchemaError("Malformed logical property")
        pos = np.array([row[fields["pos"][1]] for row in rows], dtype=np.float64)
        force = np.array([row[fields["forces"][1]] for row in rows], dtype=np.float64)
        energy = float(metadata["energy"])
        if not np.isfinite(energy): raise SchemaError("Nonfinite energy")
        p = PurePosixPath(source_path)
        connection = branch = index = None
        if "SteepestDescent" in p.parts:
            k = p.parts.index("SteepestDescent")
            connection = p.parts[k+1]
            match = re.fullmatch(r"coords(plus|minus)_(\d+)\.xyz", p.name)
            if not match: raise SchemaError("Unknown branch filename")
            branch, index = match.group(1), int(match.group(2)); record_kind = "path"
        elif re.fullmatch(r"coordsmin_\d+\.xyz", p.name): record_kind = "minimum"
        elif re.fullmatch(r"coordsts_\d+_\d+(?:_\d+)?\.xyz", p.name): record_kind = "transition_state"
        else: raise SchemaError("Unknown reference filename")
        pos.setflags(write=False); force.setflags(write=False)
        return FrameRecord(source_path, molecule, record_kind, source_path, connection, branch,
                           index, species, pos, energy, force, hashlib.sha256(text.encode()).hexdigest())
    except (KeyError, IndexError, TypeError, ValueError) as e:
        if isinstance(e, SchemaError): raise
        raise SchemaError(f"Malformed XYZ {source_path}: {e}") from e


def read_records(archive: Path, molecules: tuple[str, ...] = ("aspirin", "paracetamol")) -> list[FrameRecord]:
    verify_archive(archive)
    records = []
    with zipfile.ZipFile(archive) as z:
        for name in sorted(z.namelist()):
            parts = PurePosixPath(name).parts
            if len(parts) >= 4 and parts[0] == "Landscape17" and parts[1] in molecules and name.endswith(".xyz"):
                if parts[2] in {"Conformations", "SteepestDescent"}:
                    records.append(parse_xyz(z.read(name).decode("utf-8"), source_path=name, molecule=parts[1]))
    return records
