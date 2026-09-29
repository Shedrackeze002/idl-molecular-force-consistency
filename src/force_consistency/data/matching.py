from dataclasses import dataclass
from typing import Sequence

import networkx as nx
import numpy as np

from .geometry import align_records,candidate_pairs,descriptor
from .types import FrameRecord,PreparationPolicy


@dataclass(frozen=True)
class DuplicateAudit:
    pairs: tuple[tuple[str,str],...]
    diagnostics: tuple[dict,...]
    unresolved: tuple[dict,...]
    label_conflicts: tuple[dict,...]


def audit_duplicates(records: Sequence[FrameRecord],policy: PreparationPolicy) -> DuplicateAudit:
    pairs=[];diagnostics=[];unresolved=[];conflicts=[]
    for i,j in candidate_pairs(records,policy.duplicate_tolerance):
        a,b=records[i],records[j];r=align_records(a,b,policy)
        ids=(a.record_id,b.record_id)
        if not r.search_complete:
            unresolved.append({"records":ids,"reason":r.reason});continue
        if r.rmsd>policy.duplicate_tolerance: continue
        pairs.append(ids)
        exact=np.array_equal(a.positions,b.positions[r.permutation])
        fd=float(np.max(np.abs(a.forces@r.rotation-b.forces[r.permutation])))
        ed=abs(a.energy-b.energy)
        d={"records":ids,"rmsd":r.rmsd,"exact_stored_positions":exact,"energy_difference_eV":ed,"aligned_force_difference_eV_per_A":fd}
        diagnostics.append(d)
        if exact and (ed>policy.exact_energy_tolerance or np.max(np.abs(a.forces-b.forces[r.permutation]))>policy.exact_force_tolerance): conflicts.append(d)
    return DuplicateAudit(tuple(pairs),tuple(diagnostics),tuple(unresolved),tuple(conflicts))


def _reference_classes(references: dict[str,FrameRecord],policy):
    graph=nx.Graph();graph.add_nodes_from(sorted(references));scores={};unresolved=[]
    keys=sorted(references)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            # Noncandidate reference pairs cannot be geometrically equivalent.
            if np.linalg.norm(descriptor(references[a])-descriptor(references[b]))>len(references[a].species)*policy.reference_tolerance+1e-10: continue
            r=align_records(references[a],references[b],policy)
            if not r.search_complete: unresolved.append((a,b));continue
            scores[(a,b)]=r.rmsd
            if r.rmsd<=policy.reference_tolerance: graph.add_edge(a,b)
    classes=[]
    for component in nx.connected_components(graph):
        members=sorted(component)
        if any(scores.get((a,b),float('inf'))>policy.reference_tolerance for i,a in enumerate(members) for b in members[i+1:]):
            unresolved.append(tuple(members))
        classes.append(members)
    return sorted(classes),unresolved


def classify_endpoint(end: FrameRecord,references: dict[str,FrameRecord],policy: PreparationPolicy,*,classes=None) -> dict:
    if classes is None: classes,reference_unresolved=_reference_classes(references,policy)
    else: reference_unresolved=[]
    scores={};mappings={};incomplete=[]
    for key,ref in sorted(references.items()):
        bound=len(end.species)*policy.endpoint_tolerance+1e-10
        if np.linalg.norm(descriptor(end)-descriptor(ref))>bound:
            scores[key]=None;continue
        r=align_records(end,ref,policy)
        scores[key]=r.rmsd if np.isfinite(r.rmsd) else None
        if not r.search_complete: incomplete.append({"reference":key,"reason":r.reason});continue
        if r.rmsd<=policy.endpoint_tolerance:
            mappings[key]={"permutation":r.permutation.tolist(),"rotation":r.rotation.tolist(),"translation":r.translation.tolist(),"rmsd":r.rmsd}
    accepted=[c[0] for c in classes if any(k in mappings for k in c)]
    status="unresolved" if incomplete or reference_unresolved else ("matched" if len(accepted)==1 else "ambiguous" if accepted else "unmatched")
    representative=min((k for k in mappings if any(k in c and c[0] in accepted for c in classes)),key=lambda k:(mappings[k]['rmsd'],k),default=None)
    return {"record_id":end.record_id,"status":status,"accepted_classes":accepted,"reference_class":accepted[0] if status=='matched' else None,"reference_record":representative,"mapping":mappings.get(representative),"scores":scores,"incomplete":incomplete,"reference_unresolved":reference_unresolved}


def map_endpoints(inventory,records: Sequence[FrameRecord],policy: PreparationPolicy) -> dict:
    byid={r.record_id:r for r in records};out={}
    for mol,groups in inventory.connections.items():
        refs={rid:byid[rid] for rid in inventory.minima[mol].values()}
        classes,unresolved=_reference_classes(refs,policy)
        for group,c in groups.items():
            for branch,ids in c.branches.items():
                result=classify_endpoint(byid[ids[-1]],refs,policy,classes=classes)
                if unresolved: result['status']='unresolved';result['reference_unresolved']=unresolved
                out[f'{mol}/{group}/{branch}']=result
    return out
