"""Fixed-schedule and inference controls, with synthetic forces only."""
import importlib.util
from dataclasses import replace
import copy
from pathlib import Path

import numpy as np
import pytest

from evaluation.relax import relax
from force_consistency.data.types import FrameRecord, PreparationPolicy
from test_relax_trace import P, START


def module():
    assert importlib.util.find_spec('trace_validation_recovery'), 'Trace driver missing'
    return __import__('trace_validation_recovery')


def reference():
    return FrameRecord('ref','aspirin','minimum','ref',None,None,None,
                       ('C','H'),START.copy(),0.,np.zeros_like(START),'0'*64)


def test_exact_schedule_and_deduplicated_fixed_references():
    m=module();tasks=m.fixed_schedule(m.OWN_IDS)
    assert len(tasks)==34
    assert [(x['model_id'],x['branch_key']) for x in tasks[:4]]==[
        ('direct-0','aspirin/18/plus'),('consistency-0.1-0','aspirin/18/plus'),
        ('energy-0','aspirin/18/plus'),('energy-0','aspirin/13/plus')]
    assert len({(x['model_id'],x['reference_id']) for x in tasks[4:]})==30
    with pytest.raises(ValueError):m.fixed_schedule(m.OWN_IDS[:-1])
    with pytest.raises(ValueError):m.fixed_schedule(m.OWN_IDS+['test-reference'])


@pytest.mark.parametrize('field', ['step_count','force_evaluation_count','failure_reason',
                                 'converged','budget_exhausted','forces_finite','category','endpoint'])
def test_each_reproduction_mismatch_stops_next_task(field):
    m=module();result=relax(START,lambda p:np.zeros_like(p),P)
    original={**copy.deepcopy(result),'category':'matched_reference'}
    original['_audit_record']={'final_positions':START.tolist(),'failure':{'failure_reason':None}}
    mutated=copy.deepcopy(result);category='matched_reference'
    if field=='category':category='unmatched_converged'
    elif field=='endpoint':mutated['final_positions'][0,0]+=2e-8
    elif field=='failure_reason':mutated[field]='no_progress'
    elif field in ('converged','budget_exhausted','forces_finite'):mutated[field]=not mutated[field]
    else:mutated[field]+=1
    called=[]
    def execute(task):
        called.append(task)
        m.reproduction_gate(mutated,category,original)
    with pytest.raises(ValueError):m.run_schedule([1,2],execute)
    assert called==[1]


@pytest.mark.parametrize('force,cap,expected', [(0.,6,True),(1.,1,False)])
def test_reference_control_joint_force_geometry_rule(force,cap,expected):
    m=module();r=reference();out=relax(r.positions,lambda p:np.full_like(p,force),P|{'max_force_evaluations':cap})
    score=m.score_reference_control(out,r,PreparationPolicy())
    assert score['operational_force_converged_own_match']==expected
    assert score['geometry']['status']=='matched'
    assert score['nearby_stationary_point_absence_established'] is False


def test_converged_nonmatch_retains_geometry_failure():
    m=module();r=reference();out=relax(START,lambda p:np.zeros_like(p),P)
    out['final_positions']=np.array([[0.,0,0],[10.,0,0]])
    score=m.score_reference_control(out,r,PreparationPolicy())
    assert not score['operational_force_converged_own_match']
    assert score['geometry']['status']=='unmatched'


def test_source_inventory_unchanged_by_additive_scripts(tmp_path,monkeypatch):
    module();import run_validation_recovery as old
    (tmp_path/'evaluation').mkdir();(tmp_path/'scripts').mkdir()
    (tmp_path/'evaluation/relax.py').write_text('original')
    script=tmp_path/'scripts/run_validation_recovery.py';script.write_text('old driver')
    monkeypatch.setattr(old,'ROOT',tmp_path);monkeypatch.setattr(old,'__file__',str(script))
    expected=old.evaluation_pin()
    (tmp_path/'scripts/recovery_trace.py').write_text('new helper')
    assert old.evaluation_pin()==expected
    (tmp_path/'evaluation/relax.py').write_text('changed')
    assert old.evaluation_pin()!=expected
    (tmp_path/'evaluation/relax.py').unlink()
    assert old.evaluation_pin()!=expected


def test_launch_approval_requires_two_exact_bound_reviews(tmp_path):
    import hashlib,json
    m=module();record=tmp_path/'records.json';record.write_text('{}')
    a=tmp_path/'review-a.md';a.write_text('independent engineering approval')
    b=tmp_path/'review-b.md';b.write_text('independent scientific approval')
    h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    approval={'status':'accepted_for_paired_trace','records_sha256':h(record),
              'review_sha256':{str(a):h(a),str(b):h(b)}}
    m.verify_launch_approval(record,approval)
    approval['review_sha256'].pop(str(b))
    with pytest.raises(ValueError):m.verify_launch_approval(record,approval)
    approval['review_sha256'][str(b)]=h(b);b.write_text('changed')
    with pytest.raises(ValueError):m.verify_launch_approval(record,approval)


@pytest.mark.parametrize('field,value', [('max_force_attempts',51001),('numerical_wall_seconds',601.),
    ('max_output_bytes',2**31),('main_test_access',True),('automatic_retry',True)])
def test_expanded_caps_or_test_access_rejected_before_inputs(field,value):
    m=module();record=dict(tasks=m.fixed_schedule(m.OWN_IDS),own_ids=m.OWN_IDS,
        max_force_attempts=51000,numerical_wall_seconds=600.,finalization_wall_seconds=60.,
        max_output_bytes=2**30,main_test_access=False,automatic_retry=False,extra_energy_queries=0)
    record[field]=value
    with pytest.raises(ValueError,match='schedule/caps'):m.verify_package(record)


def test_parent_watchdog_terminates_synthetic_worker(tmp_path):
    import sys,time
    m=module();started=time.perf_counter()
    result=m.bounded_process([sys.executable,'-c','import time; time.sleep(10)'],tmp_path/'worker.log',.1,.1)
    assert result['timed_out'] and result['returncode']!=0
    assert time.perf_counter()-started<3
