"""Connection-level assignments and record eligibility, never label-based splits."""
from dataclasses import asdict, dataclass
from collections import defaultdict
import hashlib
import json
import networkx as nx
import numpy as np

from .source import SHA256


class PreparationError(ValueError):
    pass


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True)
class SplitManifest:
    groups: dict
    connections: dict
    policy: dict
    identity: str

    def payload(self):
        return {'groups':self.groups,'connections':self.connections,'policy':self.policy,'identity':self.identity,'archive_sha256':SHA256}


@dataclass(frozen=True)
class EligibilityManifest:
    partitions: dict
    memberships: dict
    excluded: dict


def assign_groups(inventory,policy):
    groups={};connections={}
    for mol,cs in sorted(inventory.connections.items()):
        counts={'aspirin':(26,5,6),'paracetamol':(5,2,2)}.get(mol)
        if counts is None or len(cs)!=sum(counts): raise PreparationError(f'Unexpected connection count: {mol}')
        shuffled=np.random.Generator(np.random.PCG64(policy.seed)).permutation(sorted(cs)).tolist()
        a,b,_=counts
        groups[mol]={'train':shuffled[:a],'validation':shuffled[a:a+b],'test':shuffled[a+b:]}
        connections[mol]={g:asdict(c) for g,c in sorted(cs.items())}
    payload={'groups':groups,'connections':connections,'policy':asdict(policy),'archive_sha256':SHA256}
    return SplitManifest(groups,connections,asdict(policy),canonical_hash(payload))


def apply_eligibility(records,splits,duplicates,endpoints):
    byid={r.record_id:r for r in records}
    if len(byid)!=len(records): raise PreparationError('Duplicate record IDs')
    selected=set(splits.groups)
    if {r.molecule for r in records}!=selected: raise PreparationError('Mixed molecule records')
    memberships=defaultdict(lambda:{'groups':[],'role':None,'seen_in_training':False})
    owners={};priority={'train':0,'validation':1,'test':2}
    for mol,parts in splits.groups.items():
        for part,gs in parts.items():
            for g in gs:
                c=splits.connections[mol][g]
                for rid in [c['transition_state'],*sum((list(v) for v in c['branches'].values()),[])]:
                    if rid not in byid: raise PreparationError('Missing connection record')
                    owners[rid]=part;memberships[rid]['groups']=[g];memberships[rid]['role']=byid[rid].kind
                # Endpoint classes, not filename pairs, determine reference membership.
                for b in ('plus','minus'):
                    end=endpoints.get(f'{mol}/{g}/{b}',{})
                    if end.get('status')!='matched': raise PreparationError(f'Unresolved endpoint {mol}/{g}/{b}')
                    rid=end['reference_class']
                    if rid not in byid or byid[rid].kind!='minimum': raise PreparationError('Missing endpoint reference')
                    if g not in memberships[rid]['groups']: memberships[rid]['groups'].append(g)
                    memberships[rid]['role']='minimum'
                    if rid not in owners or priority[part]<priority[owners[rid]]: owners[rid]=part
    for rid,part in owners.items(): memberships[rid]['seen_in_training']=part=='train'
    relevant=set(byid)
    if any(set(d['records'])&relevant for d in duplicates.unresolved+duplicates.label_conflicts):
        raise PreparationError('Unresolved duplicate certification or conflicting exact labels')
    graph=nx.Graph();graph.add_nodes_from(owners)
    graph.add_edges_from((a,b) for a,b in duplicates.pairs if a in owners and b in owners)
    excluded={}
    for component in nx.connected_components(graph):
        highest=min(priority[owners[rid]] for rid in component)
        for rid in component:
            if priority[owners[rid]]>highest: excluded[rid]='duplicate_of_higher_priority'
    partitions={p:[] for p in ('train','validation','validation_minima_diagnostics','test')}
    for rid,part in sorted(owners.items()):
        if rid in excluded: continue
        target='validation_minima_diagnostics' if part=='validation' and byid[rid].kind=='minimum' else part
        partitions[target].append(rid)
    for mol,parts in splits.groups.items():
        for part,gs in parts.items():
            for g in gs:
                c=splits.connections[mol][g];eligible=set(partitions[part])
                if not eligible.intersection([c['transition_state'],*sum((list(v) for v in c['branches'].values()),[])]):
                    raise PreparationError(f'Missing group coverage {mol}/{g}')
                if part!='train':
                    for b,ids in c['branches'].items():
                        if not any(rid in eligible and np.max(np.abs(byid[rid].forces))>splits.policy['nonstationary_force'] for rid in ids):
                            raise PreparationError(f'Missing branch coverage {mol}/{g}/{b}')
    return EligibilityManifest(partitions,{rid:{**v,'groups':sorted(v['groups'])} for rid,v in sorted(memberships.items())},excluded)
