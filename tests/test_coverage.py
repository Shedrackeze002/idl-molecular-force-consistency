from dataclasses import replace
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from force_consistency.coverage import select_coverage_subsets, verify_coverage_subsets, take_subset
from force_consistency.data.prepare import load_training_partition


def frozen_case():
    planned=json.loads(Path('manifests/aspirin-data-coverage-subsets-planned.json').read_text())
    p,m,n=load_training_partition('data/processed/aspirin',expected_split_hash=planned['split_hash'])
    return p,m,n,planned


def test_exact_nested_selection_and_rank_distribution():
    p,m,n,planned=frozen_case()
    anchor=planned['subsets']['narrow16']['record_ids']
    selected=select_coverage_subsets(p,m,anchor)
    verify_coverage_subsets(selected,planned)
    assert selected['ordered_groups']==['01','23','77','58_2','44','22','78_2','01_2']
    a,b,c=[selected['subsets'][k]['record_ids'] for k in ('narrow16','diverse16','diverse128')]
    assert [len(set(x)) for x in (a,b,c)]==[16,16,128]
    assert set(a)<=set(c) and set(b)<=set(c) and len(set(a)&set(b))==2
    assert selected['selection']['01_2']['minus']['ranks']==[0,15,31,46,63,78,94,109]
    assert selected['selection']['01_2']['plus']['ranks']==[1,14,27,40,54,67,80,94]
    for g in selected['ordered_groups']:
        assert sum(p.memberships[r]['groups']==[g] for r in b)==2
        assert sum(p.memberships[r]['groups']==[g] for r in c)==16
    labels=replace(p,energy_residual=np.full_like(p.energy_residual,999.),forces=-777*p.forces)
    assert select_coverage_subsets(labels,m,anchor)==selected
    subset=take_subset(p,b)
    assert subset.record_ids==tuple(b) and not subset.positions.flags.writeable
    bad=copy.deepcopy(planned);bad['subsets']['diverse16']['identity']='stale'
    with pytest.raises(ValueError): verify_coverage_subsets(selected,bad)


def test_train_only_source_and_rank_failures(monkeypatch):
    original=np.load
    def guarded(path,*args,**kw):
        assert Path(path).name=='train.npz'
        return original(path,*args,**kw)
    monkeypatch.setattr(np,'load',guarded)
    p,m,n,planned=frozen_case();anchor=planned['subsets']['narrow16']['record_ids']
    selected=select_coverage_subsets(p,m,anchor)
    # Every malformed selection fails rather than choosing replacement support.
    for role,groups in [('minimum',['01']),('path',['heldout']),('path',['01','23'])]:
        memberships=copy.deepcopy(p.memberships);memberships[anchor[0]]={'role':role,'groups':groups}
        with pytest.raises(ValueError): select_coverage_subsets(replace(p,memberships=memberships),m,anchor)
    with pytest.raises(ValueError): select_coverage_subsets(p,m,[anchor[0]]*16)
    with pytest.raises(ValueError): take_subset(p,[anchor[0],anchor[0]])
    with pytest.raises(ValueError): take_subset(p,['outside'])
    with pytest.raises(ValueError): select_coverage_subsets(replace(p,record_ids=(*p.record_ids[:-1],p.record_ids[0])),m,anchor)
    keep=[r for r in p.record_ids if '/23/' not in r or r.endswith('000.xyz')]
    with pytest.raises(ValueError,match='rank|pool'): select_coverage_subsets(take_subset(p,keep),m,anchor)
    from force_consistency.coverage import numeric_path
    assert numeric_path('Landscape17/aspirin/SteepestDescent/01/coordsminus_2.xyz')[-1]==2
    assert numeric_path('Landscape17/aspirin/SteepestDescent/01/coordsminus_10.xyz')[-1]==10
