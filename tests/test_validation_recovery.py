"""Fixed A2/paired reporting and durable branch failure semantics."""
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
GROUPS=['13','18','18_2','66','78']


def module():
    assert importlib.util.find_spec('run_validation_recovery'), 'validation driver missing'
    return __import__('run_validation_recovery')


def outcome_payload(count,*,complete=True):
    keys=[f'aspirin/{g}/{b}' for g in GROUPS for b in ['minus','plus']]
    branches={k:{'branch_key':k,'partition':'validation','seen_minimum':False,
        'category':'matched_reference' if i<count else 'nonconverged',
        'converged':i<count,'budget_exhausted':i>=count} for i,k in enumerate(keys)}
    return {'complete':complete,'branches':branches}


@pytest.mark.parametrize('recovered,control,expected',[(0,0,'failed'),(1,0,'passed'),(1,1,'failed'),(2,1,'passed'),(10,10,'failed')])
def test_literal_a2_rule(recovered,control,expected):
    m=module();a=m.evaluate_a2(outcome_payload(recovered),{'own_geometry_status':'verified','own_geometry_count':control,'nearest_context_status':'verified','nearest_own_class_count':8})
    assert a['status']==expected and a['recovered_branches']==recovered
    assert a['nearest_reference_context']==('relaxation adds no evidence beyond geometric proximity at this accuracy' if recovered<=8 else 'recovery exceeds nearest-reference context count')


def test_incomplete_or_unverified_a2_withheld_and_nearest_not_extra_gate():
    m=module();control={'own_geometry_status':'unavailable','own_geometry_count':None,'nearest_context_status':'unavailable','nearest_own_class_count':None}
    assert m.evaluate_a2(outcome_payload(2),control)['status']=='withheld'
    control.update(own_geometry_status='verified',own_geometry_count=0)
    assert m.evaluate_a2(outcome_payload(2),control)['status']=='passed'
    assert m.evaluate_a2(outcome_payload(2,complete=False),control)['status']=='withheld'
    damaged=outcome_payload(2);damaged['branches'].pop(next(iter(damaged['branches'])))
    assert m.evaluate_a2(damaged,control)['status']=='withheld'


def test_pair_denominators_seed_aggregation_and_five_differences():
    m=module();reports={'energy-0':outcome_payload(1),'direct-0':outcome_payload(2),'direct-1':outcome_payload(4),'consistency-0.1-0':outcome_payload(4),'consistency-0.1-1':outcome_payload(4)}
    control={'own_geometry_status':'verified','own_geometry_count':0,'nearest_context_status':'verified','nearest_own_class_count':8}
    result=m.recovery_summary(reports,control,m.evaluate_a2(reports['energy-0'],control))
    assert result['status']=='complete_validation_comparison'
    assert result['arms']['direct']['seed_rates']==[.2,.4] and result['arms']['direct']['mean_rate']==pytest.approx(.3)
    assert result['arms']['consistency-0.1']['mean_rate']==.4
    assert result['paired_connection_differences']=={'13':0.,'18':.5,'18_2':0.,'66':0.,'78':0.}
    assert result['mean_paired_difference']==pytest.approx(.1)
    assert result['independent_connection_count']==5 and result['seeds_per_arm']==2
    # Genuine nonconvergence remains in the 10-branch denominator.
    assert result['models']['direct-0']['branch_denominator']==10
    del reports['direct-1']
    assert m.recovery_summary(reports,control,{'status':'passed'})['arms']['direct']['status']=='withheld'


def test_a2_failure_keeps_fixed_descriptive_results():
    m=module();reports={cid:outcome_payload(0) for cid in m.MODEL_ORDER}
    control={'own_geometry_status':'verified','own_geometry_count':0,'nearest_context_status':'verified','nearest_own_class_count':8}
    summary=m.recovery_summary(reports,control,m.evaluate_a2(reports['energy-0'],control))
    assert summary['status']=='complete_descriptive_A2_failed' and summary['mean_paired_difference']==0.
    assert summary['scientific_main_hypothesis_established'] is False


def synthetic_pass_inputs(land,manifest,classification):
    from evaluation.start_set import build_start_set
    from evaluation.adapter import callable_arm
    starts=build_start_set(manifest,land['records'],land['policy'],{'kind':'first_stored_frame'},partitions=('validation',))
    refs={r.record_id:r for r in land['records'] if r.kind=='minimum'}
    arm=callable_arm(lambda p:np.zeros_like(p),next(iter(starts.values()))['species'],'synthetic-zero')
    procedure={'max_force_evaluations':2,'initial_step_size':.2,'backtracking_factor':.5,'min_backtracking_floor':1e-8,'convergence_tolerance_eV_per_A':.01,'max_backtracking_attempts_per_step':40}
    return starts,arm,procedure,classification,refs,land['policy'],manifest


def test_first_completed_branch_survives_later_operational_failure(tmp_path,monkeypatch,synthetic_landscape,frozen_synthetic_manifest,synthetic_endpoint_classification):
    m=module();inputs=synthetic_pass_inputs(synthetic_landscape,frozen_synthetic_manifest,synthetic_endpoint_classification)
    actual=m.evaluate_starts;calls=[]
    def failing(*args,**kw):
        calls.append(1)
        if len(calls)==2:raise RuntimeError('synthetic later failure')
        return actual(*args,**kw)
    monkeypatch.setattr(m,'evaluate_starts',failing)
    with pytest.raises(RuntimeError,match='later failure'):m.run_model_pass(*inputs,seen_minimum=False,output=tmp_path/'pass')
    raw=list((tmp_path/'pass/raw-branches').glob('*.json'));assert len(raw)==1
    row=json.loads(raw[0].read_text());assert row['force_evaluation_count']==1 and row['_audit_record']['final_positions']
    assert (tmp_path/'pass/INCOMPLETE.json').exists() and not (tmp_path/'pass/result.json').exists()


def test_all_starts_validated_before_any_force_calls(tmp_path,synthetic_landscape,frozen_synthetic_manifest,synthetic_endpoint_classification):
    m=module();inputs=list(synthetic_pass_inputs(synthetic_landscape,frozen_synthetic_manifest,synthetic_endpoint_classification));calls=[]
    from evaluation.adapter import callable_arm
    inputs[1]=callable_arm(lambda p:(calls.append(1),np.zeros_like(p))[1],('C','H','O'),'counted')
    bad_key=sorted(inputs[0])[-1];inputs[0][bad_key]={**inputs[0][bad_key],'species':('H','H','O')}
    with pytest.raises(ValueError):m.run_model_pass(*inputs,seen_minimum=False,output=tmp_path/'pass')
    assert not calls and not list((tmp_path/'pass').rglob('*.json'))


def test_nonfinite_serialization_visible_without_zero_replacement():
    m=module();out=m.safe_json({'positions':np.array([[np.nan,np.inf,2.]]),'category':'unresolved'})
    assert out['positions']==[[None,None,2.]] and len(out['_nonfinite_values'])==2
    assert out['category']=='unresolved';json.dumps(out,allow_nan=False)


def test_worker_timeout_retains_atomic_completed_row(tmp_path):
    m=module();from benchmark_e3x_budget import invoke
    # An actual child process is terminated after publishing its first row.
    script=tmp_path/'timeout_fixture.py';out=tmp_path/'pass'
    root=Path(m.__file__).resolve().parents[1]
    script.write_text("import sys,time,numpy as np\nsys.path[:0]="+repr([str(root),str(root/'tests')])+"\nimport run_validation_recovery as m\nfrom conftest import synthetic_landscape,synthetic_endpoint_classification\nfrom evaluation.start_set import build_start_set\nfrom evaluation.adapter import callable_arm\nland=synthetic_landscape.__wrapped__()\nmanifest=land['manifest']\nclassification=synthetic_endpoint_classification.__wrapped__(land)\nstarts=build_start_set(manifest,land['records'],land['policy'],{'kind':'first_stored_frame'},partitions=('validation',))\nrefs={r.record_id:r for r in land['records'] if r.kind=='minimum'}\narm=callable_arm(lambda p:np.zeros_like(p),('C','H','O'),'synthetic-timeout')\nactual=m.evaluate_starts\ncalls=0\ndef delayed(*a,**kw):\n global calls\n calls+=1\n if calls==2:time.sleep(30)\n return actual(*a,**kw)\nm.evaluate_starts=delayed\nprocedure={'max_force_evaluations':2,'initial_step_size':.2,'backtracking_factor':.5,'min_backtracking_floor':1e-8,'convergence_tolerance_eV_per_A':.01,'max_backtracking_attempts_per_step':40}\nm.run_model_pass(starts,arm,procedure,classification,refs,land['policy'],manifest,seen_minimum=False,output="+repr(str(out))+")\n")
    with pytest.raises(subprocess.TimeoutExpired):invoke([sys.executable,str(script)],tmp_path/'timeout.txt',5.)
    rows=list((out/'raw-branches').glob('*.json'));assert len(rows)==1
    assert json.loads(rows[0].read_text())['force_evaluation_count']==1
    assert not (out/'result.json').exists()


def test_cli_imports_in_fresh_child_with_frozen_child_environment(tmp_path):
    m=module();from e3x_phase_common import child_environment
    p=subprocess.run([sys.executable,str(Path(m.__file__).resolve()),'--help'],env=child_environment(),cwd=tmp_path,capture_output=True,text=True,timeout=15)
    assert p.returncode==0,p.stderr


def test_wrong_import_refused_before_output(tmp_path,monkeypatch):
    m=module()
    def wrong():raise ValueError('Wrong imported trainer')
    monkeypatch.setattr(m,'assert_imports',wrong)
    with pytest.raises(ValueError,match='Wrong imported trainer'):m.run_validation(tmp_path/'missing',tmp_path/'out',tmp_path/'approval')
    assert not (tmp_path/'out').exists()


def test_nonfinite_failure_metadata_does_not_add_a_phantom_branch(tmp_path,monkeypatch,synthetic_landscape,frozen_synthetic_manifest,synthetic_endpoint_classification):
    m=module();inputs=synthetic_pass_inputs(synthetic_landscape,frozen_synthetic_manifest,synthetic_endpoint_classification);actual=m.evaluate_starts;calls=[]
    def unresolved(*args,**kw):
        result=actual(*args,**kw);calls.append(1)
        if len(calls)==1:
            row=next(iter(result['branches'].values()));row['category']='unresolved';row['rmsd_A']=None
            row['_audit_record']['final_positions']=np.full((3,3),np.nan)
        return result
    monkeypatch.setattr(m,'evaluate_starts',unresolved)
    m.run_model_pass(*inputs,seen_minimum=False,output=tmp_path/'pass')
    saved=json.loads((tmp_path/'pass/result.json').read_text())
    assert len(saved['branches'])==len(inputs[0]) and '_nonfinite_values' not in saved['branches']
    assert saved['_nonfinite_values']


def driver_fixture(tmp_path,monkeypatch):
    m=module();output=tmp_path/'out'
    record={'order':list(m.MODEL_ORDER),'models':{cid:{} for cid in m.MODEL_ORDER},'output':str(output),
        'control_wall_seconds':300.,'global_wall_seconds':12375.,'required_free_storage_bytes':1,
        'start_binding':{},'phase_reconciliation':{'margin_seconds':1.}}
    m.save_json(tmp_path/'records',record);reviews={}
    for i in range(2):
        p=tmp_path/f'review-{i}';p.write_text(str(i));reviews[str(p)]=m.digest(p.read_bytes())
    approval={'status':'accepted_for_validation_recovery','records_sha256':m.digest((tmp_path/'records').read_bytes()),'review_sha256':reviews}
    monkeypatch.setattr(m,'assert_imports',lambda:None);monkeypatch.setattr(m,'_verify_records',lambda *a:None);monkeypatch.setattr(m,'require_storage',lambda *a:None)
    return m,record,approval,output


def test_driver_fixed_order_and_no_outcome_based_gate_skips(tmp_path,monkeypatch):
    m,record,approval,out=driver_fixture(tmp_path,monkeypatch);calls=[]
    def invoked(command,log,timeout):
        stage=command[command.index('--worker')+1];cid=command[command.index('--model')+1] if '--model' in command else None
        dest=Path(command[command.index('--output')+1]);dest.mkdir();calls.append((stage,cid))
        if stage=='control':m.save_json(dest/'control.json',{'own_geometry_status':'verified','own_geometry_count':0,'nearest_context_status':'verified','nearest_own_class_count':8})
        elif stage=='preflight':m.save_json(dest/'replay.json',{'status':'verified'})
        else:m.save_json(dest/'result.json',outcome_payload(0))
    monkeypatch.setattr(m,'invoke',invoked);m.save_json(tmp_path/'approval',approval)
    result=m.run_validation(tmp_path/'records',out,tmp_path/'approval')
    assert calls==[('control',None)]+[('preflight',c) for c in m.MODEL_ORDER]+[('pass',c) for c in m.MODEL_ORDER]
    assert result['A2']['status']=='failed' and result['status']=='complete_descriptive_A2_failed'


def test_failed_preflight_stops_without_retry_or_relaxation(tmp_path,monkeypatch):
    m,record,approval,out=driver_fixture(tmp_path,monkeypatch);calls=[]
    def invoked(command,log,timeout):
        stage=command[command.index('--worker')+1];calls.append(stage)
        if stage=='preflight':raise RuntimeError('synthetic replay failure')
        dest=Path(command[command.index('--output')+1]);dest.mkdir();m.save_json(dest/'control.json',{})
    monkeypatch.setattr(m,'invoke',invoked);m.save_json(tmp_path/'approval',approval)
    with pytest.raises(RuntimeError,match='replay failure'):m.run_validation(tmp_path/'records',out,tmp_path/'approval')
    assert calls==['control','preflight'] and (out/'STOPPED.json').exists()
    assert not (out/'COMPLETE.json').exists()


@pytest.mark.parametrize('count',[0,1])
def test_two_record_reviews_required_before_output(tmp_path,monkeypatch,count):
    m,record,approval,out=driver_fixture(tmp_path,monkeypatch);approval['review_sha256']=dict(list(approval['review_sha256'].items())[:count]);m.save_json(tmp_path/'approval',approval)
    with pytest.raises(ValueError,match='Two independent'):m.run_validation(tmp_path/'records',out,tmp_path/'approval')
    assert not out.exists()


@pytest.mark.parametrize('margin',[-1.,float('nan'),True])
def test_invalid_budget_reconciliation_refused_before_output(tmp_path,monkeypatch,margin):
    m,record,approval,out=driver_fixture(tmp_path,monkeypatch);record['phase_reconciliation']['margin_seconds']=margin
    (tmp_path/'records').write_text(json.dumps(record));approval['records_sha256']=m.digest((tmp_path/'records').read_bytes());m.save_json(tmp_path/'approval',approval)
    with pytest.raises(ValueError,match='reconciliation'):m.run_validation(tmp_path/'records',out,tmp_path/'approval')
    assert not out.exists()


def test_record_mapping_keeps_only_accepted_recipe_and_pair(tmp_path):
    m=module();from e3x_phase_common import fixed_config
    models={};chosen={};source=m.source_pin()
    for cid in m.MODEL_ORDER:
        arm='energy' if cid=='energy-0' else ('direct' if cid.startswith('direct') else 'consistency');run=tmp_path/cid;run.mkdir()
        cfg=fixed_config('C',tmp_path/'data',run,200000,100.,arm=arm,strength=.1 if arm=='consistency' else 0.,seed=int(cid[-1]))
        launch={'candidate':'C' if cid=='energy-0' else cid,'config':cfg,'source_pin':source};lp=run/'launch.json';m.save_json(lp,launch)
        score={'status':'verified','best_checkpoint_sha256':'a'*64,'best_update':184500};m.save_json(run/'score.json',score)
        models[cid]={'run_directory':str(run),'launch':str(lp),'score_directory':str(run)}
        if cid!='energy-0':chosen[cid]={'checkpoint':str(run/'best.msgpack'),'checkpoint_sha256':'a'*64,'best_update':184500,'score_sha256':m.digest((run/'score.json').read_bytes())}
    m.save_json(tmp_path/'controlled',{'status':'ACCEPTED_TWO_SEED_VALIDATION_COMPARISON','selected_consistency_lambda':.1,'matched_pair_checkpoint_identities':chosen})
    m.save_json(tmp_path/'energy',{'status':'ACCEPTED_SINGLE_SEED_VALIDATION_A1_PASS','selected_candidate':'C','selected_checkpoint_sha256':'a'*64,'selected_update':184500})
    for n in ['archive','manifest','classification']:(tmp_path/n).write_text('{}')
    config={'evaluation':m.read_json(m.ROOT/'configs/aspirin-shape-recovery.json'),'models':models,**{k:str(tmp_path/k) for k in ['archive','manifest','classification']}}
    acceptances={k:str(tmp_path/k) for k in ['controlled','energy']}
    record=m.recovery_records(acceptances,config,tmp_path/'output')
    assert record['order']==list(m.MODEL_ORDER) and len(record['models'])==5 and record['global_wall_seconds']==12375.
    assert all(v['preflight_wall_seconds']==120. and v['pass_wall_seconds']==1800. for v in record['models'].values())
    lp=Path(models['direct-0']['launch']);wrong=m.read_json(lp);wrong['config']['batch_size']=8;m.save_json(lp,wrong)
    with pytest.raises(ValueError,match='recipe'):m.recovery_records(acceptances,config,tmp_path/'output')
