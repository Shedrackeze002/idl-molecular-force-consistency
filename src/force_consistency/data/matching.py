from dataclasses import dataclass
from typing import Sequence

import networkx as nx
import numpy as np

from .geometry import align_records,candidate_pairs,descriptor,chemical_graph
from .types import FrameRecord,PreparationPolicy


@dataclass(frozen=True)
class DuplicateAudit:
    pairs: tuple[tuple[str,str],...]
    diagnostics: tuple[dict,...]
    unresolved: tuple[dict,...]
    label_conflicts: tuple[dict,...]


def exact_coordinate_permutation(a,b,policy):
    """Stored-coordinate identity is independent of an O(3) symmetry winner."""
    slots={}
    for i,(species,position) in enumerate(zip(b.species,b.positions)):
        slots.setdefault((species,tuple(position)),[]).append(i)
    order=[]
    for species,position in zip(a.species,a.positions):
        choices=slots.get((species,tuple(position)))
        if not choices: return None
        order.append(choices.pop(0))
    order=np.array(order,dtype=int)
    if len(a.species)!=len(b.species): return None
    ag=nx.to_numpy_array(chemical_graph(a,policy),nodelist=range(len(order)))
    bg=nx.to_numpy_array(chemical_graph(b,policy),nodelist=order)
    return order if np.array_equal(ag,bg) else None


def audit_duplicates(records: Sequence[FrameRecord],policy: PreparationPolicy) -> DuplicateAudit:
    pairs=[];diagnostics=[];unresolved=[];conflicts=[]
    for i,j in candidate_pairs(records,policy.duplicate_tolerance):
        a,b=records[i],records[j];r=align_records(a,b,policy)
        ids=(a.record_id,b.record_id)
        if not r.search_complete:
            unresolved.append({"records":ids,"reason":r.reason});continue
        if r.rmsd>policy.duplicate_tolerance: continue
        pairs.append(ids)
        exact_order=exact_coordinate_permutation(a,b,policy)
        exact=exact_order is not None
        fd=float(np.max(np.abs(a.forces@r.rotation-b.forces[r.permutation])))
        ed=abs(a.energy-b.energy)
        exact_fd=float(np.max(np.abs(a.forces-b.forces[exact_order]))) if exact else None
        d={"records":ids,"rmsd":r.rmsd,"exact_stored_positions":exact,"exact_permutation":exact_order.tolist() if exact else None,"exact_force_difference_eV_per_A":exact_fd,"energy_difference_eV":ed,"aligned_force_difference_eV_per_A":fd}
        diagnostics.append(d)
        if exact and (ed>policy.exact_energy_tolerance or exact_fd>policy.exact_force_tolerance): conflicts.append(d)
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
