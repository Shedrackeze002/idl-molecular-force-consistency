"""Label-blind frozen training-path selection for the reviewed coverage controls."""
from dataclasses import replace
import re

import numpy as np

from .data.splits import canonical_hash


_PATTERN=re.compile(r'^Landscape17/([^/]+)/SteepestDescent/([^/]+)/coords(minus|plus)_(\d+)\.xyz$')


def numeric_path(record_id):
    match=_PATTERN.fullmatch(record_id)
    if match is None:
        raise ValueError(f'Ambiguous path record: {record_id}')
    molecule,group,branch,frame=match.groups()
    return molecule,group,branch,int(frame)


def take_subset(partition,record_ids):
    ids=tuple(record_ids)
    if len(set(ids))!=len(ids) or len(set(partition.record_ids))!=len(partition.record_ids):
        raise ValueError('Duplicate record IDs')
    lookup={r:i for i,r in enumerate(partition.record_ids)}
    if any(r not in lookup for r in ids):
        raise ValueError('Record outside training partition')
    idx=np.array([lookup[r] for r in ids],dtype=int)
    arrays={}
    for name in ('positions','energy_residual','forces','atomic_numbers'):
        arrays[name]=getattr(partition,name)[idx].copy()
        arrays[name].flags.writeable=False
    return replace(partition,**arrays,record_ids=ids,memberships={r:partition.memberships[r] for r in ids})


def select_coverage_subsets(partition,manifest,anchor_ids,seed=20260930):
    ids=list(partition.record_ids);anchor=list(anchor_ids)
    if len(set(ids))!=len(ids) or len(set(anchor))!=16:
        raise ValueError('Duplicate IDs or invalid anchor count')
    molecule=manifest['molecule']
    groups=manifest['splits']['groups'][molecule]['train']
    if len(set(groups))!=len(groups) or '01' not in groups or len(groups)<8:
        raise ValueError('Insufficient unambiguous training groups')
    ordered=['01',*np.random.Generator(np.random.PCG64(seed)).permutation(sorted(g for g in groups if g!='01'))[:7].tolist()]
    pools={g:{b:[] for b in ('minus','plus')} for g in ordered}
    lookup={r:i for i,r in enumerate(ids)}
    for rid in ids:
        membership=partition.memberships[rid]
        if membership['role']!='path':
            continue
        mol,g,b,frame=numeric_path(rid)
        if mol!=molecule or g not in groups or membership['groups']!=[g]:
            raise ValueError('Path outside training role/group or mixed membership')
        if g in pools:
            pools[g][b].append((frame,rid))
    for branches in pools.values():
        for b,pool in branches.items():
            pool.sort()
            if not pool or len({frame for frame,r in pool})!=len(pool):
                raise ValueError('Empty or ambiguous numeric branch pool')
            branches[b]=[rid for frame,rid in pool]
    numerators={};denominators={}
    if any(r not in lookup or partition.memberships[r]['role']!='path' for r in anchor):
        raise ValueError('Anchor outside training path records')
    for b in ('minus','plus'):
        selected=[r for r in anchor if numeric_path(r)[1:3]==('01',b)]
        if len(selected)!=8 or any(r not in pools['01'][b] for r in selected):
            raise ValueError('Anchor branch membership mismatch')
        numerators[b]=[pools['01'][b].index(r) for r in selected]
        denominators[b]=len(pools['01'][b])-1
        if denominators[b]<=0: raise ValueError('Insufficient anchor pool')
    selection={};small=[];large=[]
    for i,g in enumerate(ordered):
        selection[g]={}
        for b in ('minus','plus'):
            pool=pools[g][b];den=denominators[b]
            ranks=[(2*r*(len(pool)-1)+den)//(2*den) for r in numerators[b]]
            if len(set(ranks))!=8:
                raise ValueError(f'Insufficient pool or colliding ranks: {g}/{b}')
            mapped=[pool[r] for r in ranks];chosen=mapped[i]
            selection[g][b]={'pool_count':len(pool),'ranks':ranks,'record_ids':mapped,'small_set_record_id':chosen}
            small.append(chosen);large.extend(mapped)
    subsets={}
    for name,records in [('narrow16',anchor),('diverse16',small),('diverse128',large)]:
        subsets[name]={'record_ids':records,'indices':[lookup[r] for r in records],
                       'identity':canonical_hash(records),'anchor_fit_overlap_ids':[r for r in anchor if r in records]}
    if not set(anchor)<=set(large) or not set(small)<=set(large):
        raise ValueError('Subset nesting mismatch')
    return {'ordered_groups':ordered,'rank_numerators':numerators,'rank_denominators':denominators,
            'selection':selection,'subsets':subsets,'selection_seed':seed,
            'prepared_identity':manifest['identity'],'split_hash':manifest['split_hash'],'anchor_identity':canonical_hash(anchor)}


def verify_coverage_subsets(selected,planned):
    if canonical_hash({k:v for k,v in planned.items() if k!='identity'})!=planned['identity']:
        raise ValueError('Frozen subset manifest identity mismatch')
    for key,value in selected.items():
        if planned.get(key)!=value:
            raise ValueError(f'Frozen subset mismatch: {key}')
