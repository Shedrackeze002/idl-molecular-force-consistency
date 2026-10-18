"""Synthetic controls for the fixed saved-endpoint diagnostic; no real fits."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest


def module():
    assert importlib.util.find_spec('diagnose_validation_recovery'), 'Diagnostic script missing'
    return __import__('diagnose_validation_recovery')


def entries():
    keys=[f'aspirin/{g}/{b}' for g in ('13','18','18_2','66','78') for b in ('minus','plus')]
    out=[];p=np.array([[0.,0,0],[1.,0,0],[0.,1.,0]])
    for role in ('start','endpoint','reference'):
        for i in range(11 if role=='reference' else 10):
            out.append({'role':role,'record_id':f'{role}-{i}','branch_key':None if role=='reference' else keys[i],
                'positions':(p+i*.01).tolist(),'species':['C','H','H'],
                'reference_forces':np.zeros((3,3)).tolist() if role=='reference' else None,
                'official_category':'nonconverged' if role=='endpoint' else None,
                'recorded_converged':False if role=='endpoint' else None,
                'own_reference_id':f'reference-{i%11}' if role!='reference' else None})
    return out


def test_force_metrics_have_physical_arithmetic_and_raw_net_force():
    m=module();p=np.array([[0.,0,0],[1.,0,0],[0.,1.,0]]);f=np.array([[1.,2,0],[0.,0,3],[0.,0,0]])
    x=m.force_metrics(p,f)
    assert x['max_abs_component_eV_per_A']==3
    assert x['rms_component_eV_per_A']==pytest.approx(np.sqrt(14/9))
    np.testing.assert_allclose(x['net_force_eV_per_A'],[1,2,3])
    np.testing.assert_allclose(x['centroid_torque_eV'],np.cross(p-p.mean(0),f).sum(0))
    assert f.sum()!=0, 'Test must catch silently recentered forces'


@pytest.mark.parametrize('matrix',[np.array([[0.,-1,0],[1.,0,0],[0.,0,1]]),np.diag([-1.,1,1])])
def test_torque_translation_and_axial_rotation_covariance(matrix):
    m=module();p=np.array([[0.,0,0],[1.,0,0],[0.,1.,0]]);f=np.array([[1.,2,0],[0.,0,3],[0.,0,0]])
    old=m.force_metrics(p,f);new=m.force_metrics(p@matrix+[3,-4,2],f@matrix)
    assert new['rms_component_eV_per_A']==pytest.approx(old['rms_component_eV_per_A'])
    np.testing.assert_allclose(new['net_force_eV_per_A'],np.array(old['net_force_eV_per_A'])@matrix,atol=1e-14)
    np.testing.assert_allclose(new['centroid_torque_eV'],np.linalg.det(matrix)*np.array(old['centroid_torque_eV'])@matrix,atol=1e-14)


@pytest.mark.parametrize('bad',[np.ones((3,2)),np.full((3,3),np.nan),np.empty((0,3))])
def test_bad_force_or_position_refused(bad):
    m=module()
    with pytest.raises(ValueError):m.force_metrics(np.ones((3,3)),bad)


def test_fixed_input_counts_and_hash_identity():
    m=module();items=entries();m.validate_entries(items,('C','H','H'))
    changed=json.loads(json.dumps(items));changed[0]['positions'][0][0]+=.1
    assert m.entries_digest(items)!=m.entries_digest(changed)
    for damaged in [items[:-1],items+[items[0]],items[:1]*31]:
        with pytest.raises(ValueError):m.validate_entries(damaged,('C','H','H'))
    changed[0]['branch_key']='aspirin/test/plus'
    with pytest.raises(ValueError):m.validate_entries(changed,('C','H','H'))


def test_all_entries_prevalidated_before_model_calls(tmp_path):
    m=module();items=entries();items[-1]['positions'][0][0]=float('nan');calls=[]
    arm={'species':('C','H','H'),'forces':lambda *a,**k:calls.append(1)}
    with pytest.raises(ValueError):m.probe_batches(arm,items,{},tmp_path/'out')
    assert not calls and not (tmp_path/'out').exists()


def synthetic_arm(calls,*,fail_at=None):
    def forces(p,**kw):
        calls.append(len(p))
        if len(calls)==fail_at:raise RuntimeError('synthetic failure after completed batch')
        return np.asarray(p)+.02,{}
    return {'species':('C','H','H'),'forces':forces,'checkpoint_identity':{'checkpoint_sha256':'synthetic'}}


def expectations(items):return {x['record_id']:np.array(x['positions'])+.02 for x in items if x['role']=='start'}


def test_exact155_molecular_probes40_supplier_calls_and_raw_persistence(tmp_path):
    m=module();items=entries();total=[]
    for cid in m.MODEL_ORDER:
        calls=[];result=m.probe_batches(synthetic_arm(calls),items,expectations(items),tmp_path/cid)
        assert calls==[4]*7+[3] and result['molecular_evaluations']==31 and result['supplier_calls']==8
        assert len(list((tmp_path/cid).glob('batch-*.json')))==8
        raw=result['records'];assert len(raw)==31
        refs=[x for x in raw if x['role']=='reference'];ends=[x for x in raw if x['role']=='endpoint']
        assert all(x['label_force_metrics']['max_abs_component_eV_per_A']==0 for x in refs)
        assert all(x['prediction_error_metrics'] is not None for x in refs)
        assert all(x['reference_forces'] is None and x['prediction_error_metrics'] is None for x in ends)
        assert all(x['official_category']=='nonconverged' for x in ends)
        assert all(x['predicted_forces_eV_per_A']==(np.array(x['positions'])+.02).tolist() for x in raw)
        total.extend(calls)
    assert sum(total)==155 and len(total)==40


def test_partial_failure_retains_batches_and_attempt_accounting(tmp_path):
    m=module();items=entries();calls=[]
    with pytest.raises(RuntimeError,match='after completed'):m.probe_batches(synthetic_arm(calls,fail_at=2),items,expectations(items),tmp_path/'out')
    assert calls==[4,4]
    assert len(list((tmp_path/'out').glob('batch-*.json')))==1
    failed=json.loads((tmp_path/'out/INCOMPLETE.json').read_text())
    assert failed['supplier_calls_attempted']==2 and failed['molecular_evaluations_attempted']==8
    assert failed['molecular_evaluations_completed']==4 and not (tmp_path/'out/result.json').exists()


def test_start_parity_failure_stops_no_retry_with_raw_output(tmp_path):
    m=module();items=entries();expected=expectations(items);expected['start-0']=np.zeros((3,3));calls=[]
    with pytest.raises(ValueError,match='parity'):m.probe_batches(synthetic_arm(calls),items,expected,tmp_path/'out')
    assert calls==[4] and len(list((tmp_path/'out').glob('batch-*.json')))==1
    assert not (tmp_path/'out/result.json').exists()


def test_own_geometry_is_frozen_and_unavailable_rmsd_is_not_numeric(synthetic_landscape):
    m=module();records=synthetic_landscape['records'];own=next(r for r in records if r.kind=='minimum')
    refs={own.record_id:own};same=m.geometry_comparison(own,own,refs,synthetic_landscape['policy'])
    assert same['own']['status']=='matched' and same['own_alignment']['rmsd_A']==pytest.approx(0.)
    moved=replace(own,record_id='changed',positions=own.positions*20)
    x=m.geometry_comparison(moved,own,refs,synthetic_landscape['policy'])
    assert x['own_alignment']['rmsd_A'] is None and x['own_alignment']['reason']=='graph_mismatch'
    assert x['own']['status']=='unmatched' and x['fixed_order_edges_removed']
    assert 'not a chemical bond-breaking claim' in x['interpretation']


def test_incomplete_alignment_and_history_gap_explicit(monkeypatch,synthetic_landscape):
    m=module();from force_consistency.data.geometry import AlignmentResult
    own=next(r for r in synthetic_landscape['records'] if r.kind=='minimum')
    monkeypatch.setattr(m,'align_records',lambda *a:AlignmentResult(.01,np.arange(len(own.species)),np.eye(3),np.zeros(3),False,'search_limit'))
    x=m.geometry_comparison(own,own,{own.record_id:own},synthetic_landscape['policy'])
    assert x['own_alignment']['rmsd_A'] is None and not x['own_alignment']['search_complete']
    assert m.history_availability({'_audit_record':{'force_evaluation_count':12}})['status']=='unavailable'


def xyz_text():
    species=['C']*9+['H']*8+['O']*4
    header='Properties=species:S:1:pos:R:3:forces:R:3 energy=-1 pbc="F F F"'
    return '21\n'+header+'\n'+'\n'.join(f'{s} {i} 0 0 0 0 0' for i,s in enumerate(species))+'\n'


def test_archive_reads_only_exact21_members_and_checks_hash(tmp_path,monkeypatch):
    m=module();names=m.allowed_member_names();p=tmp_path/'fixture.zip';payload=xyz_text().encode()
    with zipfile.ZipFile(p,'w') as z:
        for name in names:z.writestr(name,payload)
        z.writestr('Landscape17/aspirin/SteepestDescent/test/coordsplus_000.xyz',payload)
    monkeypatch.setattr(m,'verify_archive',lambda p:None);opened=[];actual=zipfile.ZipFile.read
    def guarded(self,name,*a,**kw):
        assert name in names;opened.append(name);return actual(self,name,*a,**kw)
    monkeypatch.setattr(zipfile.ZipFile,'read',guarded)
    bound={n:m.digest(payload) for n in names};records=m.read_selected_archive(p,bound)
    assert set(opened)==set(names) and len(records)==len(opened)==21
    bad={**bound};bad[next(iter(bad))]='changed'
    with pytest.raises(ValueError,match='member'):m.read_selected_archive(p,bad)
    with pytest.raises(ValueError):m.read_selected_archive(p,{**bound,'test-member':'x'})


def driver_fixture(tmp_path,monkeypatch):
    m=module();out=tmp_path/'out';record={'output':str(out),'margin_seconds':1.,'global_wall_seconds':900.,'geometry_wall_seconds':300.,'model_wall_seconds':120.,'automatic_retry':False}
    p=tmp_path/'records';m.atomic_json(p,record);reviews={}
    for i in range(2):
        r=tmp_path/f'review{i}';r.write_text(str(i));reviews[str(r)]=m.digest(r.read_bytes())
    a={'status':'accepted_for_diagnosis','records_sha256':m.digest(p.read_bytes()),'review_sha256':reviews};m.atomic_json(tmp_path/'approval',a)
    monkeypatch.setattr(m,'assert_imports',lambda:None);monkeypatch.setattr(m,'verify_record',lambda *a:None);monkeypatch.setattr(m,'require_storage',lambda *a:None)
    return m,p,out,tmp_path/'approval',a


def test_supervisor_failure_stops_once_without_retry(tmp_path,monkeypatch):
    m,p,out,a,_=driver_fixture(tmp_path,monkeypatch);calls=[]
    def failed(*args,**kw):calls.append(1);raise subprocess.TimeoutExpired('synthetic timeout',.01)
    monkeypatch.setattr(m,'invoke',failed)
    with pytest.raises(subprocess.TimeoutExpired):m.run_diagnosis(p,out,a)
    assert calls==[1] and (out/'STOPPED.json').exists() and not (out/'COMPLETE.json').exists()


@pytest.mark.parametrize('count',[0,1])
def test_two_review_bindings_required_before_outputs(tmp_path,monkeypatch,count):
    m,p,out,a,approval=driver_fixture(tmp_path,monkeypatch);approval['review_sha256']=dict(list(approval['review_sha256'].items())[:count]);a.write_text(json.dumps(approval))
    with pytest.raises(ValueError):m.run_diagnosis(p,out,a)
    assert not out.exists()


def test_fresh_cli_child_imports_with_existing_environment(tmp_path):
    m=module();from e3x_phase_common import child_environment
    p=subprocess.run([sys.executable,m.__file__,'--help'],cwd=tmp_path,env=child_environment(),text=True,capture_output=True,timeout=10)
    assert p.returncode==0,p.stderr


def saved_input_fixture(tmp_path,monkeypatch):
    m=module();from dataclasses import asdict
    from force_consistency.data.xyz import parse_xyz
    from force_consistency.data.types import PreparationPolicy
    from e3x_phase_common import save_json
    raw={n:parse_xyz(xyz_text(),source_path=n,molecule='aspirin') for n in m.allowed_member_names()}
    refs=sorted(k for k,v in raw.items() if v.kind=='minimum');classification={}
    rows={}
    for i,key in enumerate(sorted(m.EXPECTED_KEYS)):
        group,side=key.split('/')[1:];rid=f'Landscape17/aspirin/SteepestDescent/{group}/coords{side}_000.xyz';own=raw[refs[i%11]]
        classification[key]={'reference_class':own.record_id}
        rows[key]={'branch_key':key,'start_record_id':rid,'final_record_id':key+'/final','category':'nonconverged','converged':False,
            'stop_reason':'budget_exhausted','force_evaluation_count':1500,'step_count':749,
            '_audit_record':{'start':{'record_id':rid},'reference':{'record_id':own.record_id,'species':list(own.species),'positions':own.positions.tolist()},'final_positions':(own.positions+.01).tolist()}}
    save_json(tmp_path/'classification',classification)
    for cid in m.MODEL_ORDER:
        p=tmp_path/('pass-'+cid);p.mkdir();save_json(p/'result.json',{'branches':rows})
    prior={'output':str(tmp_path),'archive':'synthetic','classification':str(tmp_path/'classification'),
        'start_binding':{'reference_ids':refs},'evaluation':{'preparation_policy':asdict(PreparationPolicy())}}
    save_json(tmp_path/'prior',prior);monkeypatch.setattr(m,'read_selected_archive',lambda *a:raw)
    return m,{'prior_records':str(tmp_path/'prior'),'archive_members':{}},raw


def test_loader_binds_actual_policy_schema_and_frozen_entries(tmp_path,monkeypatch):
    m,r,raw=saved_input_fixture(tmp_path,monkeypatch)
    prior,models,branches,_,refs,policy=m.load_inputs(r)
    assert len(models)==5 and all(len(items)==31 for items in models.values()) and len(refs)==11
    assert policy.endpoint_tolerance==.05 and all(x['role']=='reference' for x in models['energy-0'][-11:])
    bound={**r,'entries':models};m.load_inputs(bound)
    bad=json.loads(json.dumps(models));bad['direct-0'][0]['positions'][0][0]+=1
    with pytest.raises(ValueError,match='entries differ'):m.load_inputs({**r,'entries':bad})


def test_geometry_keeps_official_nonconvergence_and_marks_trace_missing(tmp_path,monkeypatch):
    m,r,_=saved_input_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(m,'geometry_comparison',lambda *a:{'own':{'status':'matched'}})
    m.geometry_worker(r,tmp_path/'geometry')
    result=json.loads((tmp_path/'geometry/geometry.json').read_text())
    assert result['model_force_calls']==0 and len(result['rows'])==50
    assert all(x['official_category']=='nonconverged' and x['history']['status']=='unavailable' for x in result['rows'])


def test_endpoint_ground_truth_must_not_be_invented():
    m=module();items=entries();items[10]['reference_forces']=np.zeros((3,3)).tolist()
    with pytest.raises(ValueError,match='reference labels'):m.validate_entries(items,('C','H','H'))


def test_actual_timeout_retains_first_completed_force_batch(tmp_path):
    m=module();from benchmark_e3x_budget import invoke
    script=tmp_path/'timeout_fixture.py';root=Path(m.__file__).resolve().parents[1];out=tmp_path/'probe'
    script.write_text("import sys,time\nsys.path[:0]="+repr([str(root),str(root/'tests')])+"\nfrom test_recovery_diagnosis import entries,expectations,synthetic_arm\nimport diagnose_validation_recovery as m\nitems=entries();calls=[];arm=synthetic_arm(calls);actual=arm['forces']\ndef delayed(*a,**kw):\n if calls:time.sleep(30)\n return actual(*a,**kw)\narm['forces']=delayed\nm.probe_batches(arm,items,expectations(items),"+repr(str(out))+")\n")
    with pytest.raises(subprocess.TimeoutExpired):invoke([sys.executable,str(script)],tmp_path/'timeout.txt',5.)
    assert len(list(out.glob('batch-*.json')))==1 and len(list(out.glob('attempt-*.json')))==2
    assert len(json.loads((out/'batch-000.json').read_text())['records'])==4
    assert not (out/'result.json').exists()


def test_summary_requires_all_fixed_models_and_support(tmp_path):
    m=module();items=entries();models={cid:m.probe_batches(synthetic_arm([]),items,expectations(items),tmp_path/cid) for cid in m.MODEL_ORDER}
    geometry={'complete':True,'rows':[{}]*50};summary=m.diagnosis_summary(geometry,models)
    assert summary['molecular_force_evaluations']==155 and summary['supplier_calls']==40 and not summary['cause_established']
    del models[m.MODEL_ORDER[-1]]
    with pytest.raises(ValueError):m.diagnosis_summary(geometry,models)


def test_frozen_input_hash_refused_before_prior_restoration(tmp_path,monkeypatch):
    m=module();accept=tmp_path/'accept';accept.write_text('synthetic');bad=tmp_path/'input';bad.write_text('changed');calls=[]
    monkeypatch.setattr(m,'PRIOR_ACCEPTANCE',m.digest(accept.read_bytes()));monkeypatch.setattr(m,'_verify_records',lambda *a:calls.append(1))
    record={'order':list(m.MODEL_ORDER),'model_wall_seconds':120.,'geometry_wall_seconds':300.,'global_wall_seconds':900.,'automatic_retry':False,
        'prior_acceptance':str(accept),'inputs':{str(bad):'not-the-hash'}}
    with pytest.raises(ValueError,match='input changed'):m.verify_record(record)
    assert not calls
