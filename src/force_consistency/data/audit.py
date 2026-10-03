import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence

from .types import FrameRecord


class AuditError(ValueError):
    pass


@dataclass(frozen=True)
class Connection:
    molecule: str
    connection_id: str
    branches: dict[str, tuple[str, ...]]
    transition_state: str
    endpoint_pair: tuple[str, str]


@dataclass(frozen=True)
class ConnectionInventory:
    connections: dict[str, dict[str, Connection]]
    minima: dict[str, dict[str, str]]
    counts: dict[str, dict[str, int]]


def inventory_connections(records: Sequence[FrameRecord]) -> ConnectionInventory:
    if len({r.record_id for r in records}) != len(records):
        raise AuditError("Duplicate source record identities")
    groups, references, minima, counts = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list)), defaultdict(dict), defaultdict(Counter)
    for r in records:
        counts[r.molecule][r.kind] += 1
        if r.kind == "path": groups[r.molecule][r.connection_id].append(r)
        elif r.kind == "minimum":
            name = re.fullmatch(r"coordsmin_(\d+)\.xyz", r.source_path.rsplit("/", 1)[-1])
            if not name: raise AuditError("Invalid minimum identity")
            if name[1] in minima[r.molecule]: raise AuditError("Ambiguous minimum identity")
            minima[r.molecule][name[1]] = r.record_id
        else:
            m = re.fullmatch(r"coordsts_(\d+)_(\d+)(?:_(\d+))?\.xyz", r.source_path.rsplit("/", 1)[-1])
            if not m: raise AuditError("Invalid transition-state identity")
            key = m[1]+m[2]+("_"+m[3] if m[3] else "")
            references[r.molecule][key].append((r, (m[1], m[2])))
    output = {}
    for mol, bygroup in groups.items():
        if set(bygroup) != set(references[mol]): raise AuditError(f"Unpaired pathway/TS groups in {mol}")
        output[mol] = {}
        for group, rs in sorted(bygroup.items()):
            hits = references[mol][group]
            if len(hits) != 1: raise AuditError(f"Ambiguous TS for {mol}/{group}")
            ts, pair = hits[0]
            if any(i not in minima[mol] for i in pair): raise AuditError(f"Missing reference minima for {mol}/{group}")
            branches = {}
            for branch in ("plus", "minus"):
                seq = sorted((r for r in rs if r.branch == branch), key=lambda r: r.frame_index)
                if not seq or [r.frame_index for r in seq] != list(range(len(seq))):
                    raise AuditError(f"Missing/duplicate/gapped branch {mol}/{group}/{branch}")
                if any(r.species != seq[0].species for r in seq) or ts.species != seq[0].species:
                    raise AuditError(f"Unstable species order {mol}/{group}/{branch}")
                branches[branch] = tuple(r.record_id for r in seq)
            output[mol][group] = Connection(mol, group, branches, ts.record_id, pair)
    return ConnectionInventory(output, dict(minima), {m: dict(c) for m,c in counts.items()})
