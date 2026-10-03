from dataclasses import replace
import json
from pathlib import Path
import numpy as np
import pytest

from force_consistency.data.audit import Connection, ConnectionInventory
from force_consistency.data.matching import DuplicateAudit
from force_consistency.data.types import FrameRecord, PreparationPolicy


def fixture(molecule='aspirin'):
    groups={}; records=[]; endpoints={}
    names=['010','310','00']+[f'g{i:02}' for i in range(34)]
    if molecule=='paracetamol': names=names[:9]
    def record(rid,kind,group=None,branch=None):
        return FrameRecord(rid,molecule,kind,rid,group,branch,0,('C','H'),np.array([[0.,0,0],[1,0,0]]),float(len(records)),np.ones((2,3)),rid)
    for i in range(4): records.append(record(f'{molecule}/min{i}','minimum'))
    for g in names:
        ids={b:(f'{molecule}/{g}/{b}',) for b in ('plus','minus')}
        ts=f'{molecule}/{g}/ts'
        groups[g]=Connection(molecule,g,ids,ts,('0','0'))
        for b in ids:
            records.append(record(ids[b][0],'path',g,b))
            endpoints[f'{molecule}/{g}/{b}']={'status':'matched','reference_class':f'{molecule}/min0','reference_record':f'{molecule}/min0'}
        records.append(record(ts,'transition_state',g))
    inv=ConnectionInventory({molecule:groups},{molecule:{str(i):f'{molecule}/min{i}' for i in range(4)}},{})
    return records,inv,endpoints


UNIT_EVIDENCE=json.loads((Path(__file__).resolve().parents[1]/'manifests/landscape17-unit-evidence.json').read_text())

def empty_duplicates(): return DuplicateAudit((),(),(),())


def test_counts_disjoint_deterministic_string_ids():
    from force_consistency.data.splits import assign_groups
    _,inv,_=fixture();p=PreparationPolicy();s=assign_groups(inv,p)
    a=s.groups['aspirin']
    assert list(map(len,(a['train'],a['validation'],a['test'])))==[26,5,6]
    expected=np.random.Generator(np.random.PCG64(p.seed)).permutation(sorted(inv.connections['aspirin']))
    assert a['train']==expected[:26].tolist()
    assert len(set(sum(a.values(),[])))==37 and '010' in sum(a.values(),[])
    assert s.identity==assign_groups(inv,p).identity


def test_minimum_memberships_follow_geometry_and_precedence():
    from force_consistency.data.splits import assign_groups,apply_eligibility
    rs,inv,ends=fixture();s=assign_groups(inv,PreparationPolicy());parts=s.groups['aspirin']
    for partition,number in [('validation',1),('test',2)]:
        for g in parts[partition]:
            for b in ('plus','minus'): ends[f'aspirin/{g}/{b}']['reference_class']=f'aspirin/min{number}';ends[f'aspirin/{g}/{b}']['reference_record']=f'aspirin/min{number}'
    # One minimum shared between validation and test is diagnostic once.
    ends[f"aspirin/{parts['test'][0]}/plus"].update(reference_class='aspirin/min1',reference_record='aspirin/min1')
    e=apply_eligibility(rs,s,empty_duplicates(),ends)
    assert e.partitions['validation_minima_diagnostics']==['aspirin/min1']
    assert 'aspirin/min0' in e.partitions['train']
    assert e.memberships['aspirin/min0']['seen_in_training']
    assert len(e.memberships['aspirin/min0']['groups'])==26
    assert all(next(r.kind for r in rs if r.record_id==rid)!='minimum' for rid in e.partitions['validation'])
    assert 'aspirin/min2' not in e.partitions['train']+e.partitions['validation_minima_diagnostics']
    for g in parts['train']:
        assert all(r in e.partitions['train'] for r in [inv.connections['aspirin'][g].transition_state,*sum((list(v) for v in inv.connections['aspirin'][g].branches.values()),[])])


def test_duplicates_and_branch_coverage_gate():
    from force_consistency.data.splits import assign_groups,apply_eligibility,PreparationError
    rs,inv,ends=fixture();s=assign_groups(inv,PreparationPolicy());a=s.groups['aspirin']
    pair=(f"aspirin/{a['train'][0]}/plus",inv.connections['aspirin'][a['validation'][0]].transition_state)
    e=apply_eligibility(rs,s,DuplicateAudit((pair,),(),(),()),ends)
    assert pair[0] in e.partitions['train'] and pair[1] not in e.partitions['validation']
    assert e.excluded=={pair[1]:'duplicate_of_higher_priority'}
    pair=(pair[0],f"aspirin/{a['validation'][0]}/plus")
    with pytest.raises(PreparationError,match='coverage'): apply_eligibility(rs,s,DuplicateAudit((pair,),(),(),()),ends)
    bad=[replace(r,forces=np.zeros_like(r.forces)) if r.record_id==pair[1] else r for r in rs]
    with pytest.raises(PreparationError,match='coverage'): apply_eligibility(bad,s,empty_duplicates(),ends)


def test_normalization_label_isolation_and_no_test_loading(tmp_path,monkeypatch):
    from force_consistency.data.prepare import export_prepared,load_training_bundle
    rs,inv,ends=fixture();kwargs=dict(inventory=inv,endpoints=ends,duplicates=empty_duplicates(),policy=PreparationPolicy(),molecule='aspirin')
    m=export_prepared(rs,unit_evidence=UNIT_EVIDENCE,directory=tmp_path/'a',**kwargs)
    held=set(m['splits']['groups']['aspirin']['validation']+m['splits']['groups']['aspirin']['test'])
    changed=[replace(r,energy=1e9,forces=r.forces*1e6) if r.connection_id in held else r for r in rs]
    n=export_prepared(changed,unit_evidence=UNIT_EVIDENCE,directory=tmp_path/'b',**kwargs)
    assert m['normalization']==n['normalization']
    original=np.load
    def guarded(path,*a,**kw):
        assert 'test.npz' not in str(path)
        return original(path,*a,**kw)
    monkeypatch.setattr(np,'load',guarded)
    bundle=load_training_bundle(tmp_path/'a',expected_split_hash=m['split_hash'])
    assert not hasattr(bundle,'test') and bundle.train.positions.dtype==np.float64
    with pytest.raises(ValueError): load_training_bundle(tmp_path/'a',expected_split_hash='stale')
    path=tmp_path/'a/manifest.json';payload=json.loads(path.read_text());payload['policy_hash']='stale';path.write_text(json.dumps(payload))
    with pytest.raises(ValueError): load_training_bundle(tmp_path/'a',expected_split_hash=m['split_hash'])


def test_molecule_gates_isolation_and_repeat_identity(tmp_path):
    from force_consistency.data.prepare import export_prepared
    from force_consistency.data.splits import PreparationError
    rs,inv,ends=fixture();ps,pi,pe=fixture('paracetamol')
    allinv=ConnectionInventory(inv.connections|pi.connections,inv.minima|pi.minima,{})
    pe[next(iter(pe))]['status']='unmatched'
    kw=dict(inventory=allinv,endpoints=ends|pe,duplicates=empty_duplicates(),policy=PreparationPolicy())
    m=export_prepared(rs+ps,unit_evidence=UNIT_EVIDENCE,directory=tmp_path/'a',molecule='aspirin',**kw)
    changed=[replace(r,energy=np.nan) for r in ps]
    n=export_prepared(rs+changed,unit_evidence=UNIT_EVIDENCE,directory=tmp_path/'b',molecule='aspirin',**kw)
    assert m==n
    with pytest.raises(PreparationError,match='endpoint'): export_prepared(rs+ps,unit_evidence=UNIT_EVIDENCE,directory=tmp_path/'failed',molecule='paracetamol',**kw)
    assert not (tmp_path/'failed/manifest.json').exists()


def test_training_bundle_rejects_mixed_molecule_and_stale_archive(tmp_path):
    from force_consistency.data.prepare import export_prepared,load_training_bundle,canonical_hash
    rs,inv,ends=fixture();m=export_prepared(rs,unit_evidence=UNIT_EVIDENCE,directory=tmp_path,inventory=inv,endpoints=ends,duplicates=empty_duplicates(),policy=PreparationPolicy(),molecule='aspirin')
    for key,value in [('archive_sha256','stale'),('molecule','paracetamol')]:
        bad=dict(m);bad[key]=value;bad.pop('identity');bad['identity']=canonical_hash(bad)
        (tmp_path/'manifest.json').write_text(json.dumps(bad))
        with pytest.raises(ValueError): load_training_bundle(tmp_path,expected_split_hash=m['split_hash'])
