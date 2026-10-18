"""New nine-slot mappings and stage accounting; supervisor primitives have prior tests."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def module():
    assert importlib.util.find_spec('launch_e3x_comparison'), 'comparison launcher missing'
    return __import__('launch_e3x_comparison')


def inputs(tmp_path):
    costs={a:{'fit_seconds':f,'score_seconds':s,'seconds':f+s} for a,f,s in [('energy',400.,20.),('direct',200.,10.),('consistency',600.,30.)]}
    budget={'updates':200000,'reserve_multiplier':1.25,'per_recipe':{'C':costs},'io':{'retained_bytes':100}}
    identities={a:{str(seed):{'initial_parameter_sha256':('energy' if a=='energy' else 'dual')+str(seed),'initial_state_sha256':('energy' if a=='energy' else 'dual')+'state'+str(seed)} for seed in (0,1)} for a in ('energy','direct','consistency')}
    selected={'candidate':'C','config':{'data_directory':str(tmp_path/'data')},'fit_wall_seconds':500.,'dataset_pin':{'same':'data'},'source_pin':{'same':'source'},'initial_identities':identities,'external_scripts':{'test':'pin'}}
    return budget,selected


def test_exact_nine_mapping_caps_seed_pairing_and_energy_last(tmp_path):
    m=module();b,s=inputs(tmp_path);records=m.comparison_records(b,s,tmp_path/'output')
    expected=[('direct',0,0.),('consistency',0,1.),('direct',1,0.),('consistency',1,1.),('consistency',0,.1),('consistency',1,.1),('consistency',0,10.),('consistency',1,10.),('energy',1,0.)]
    assert [(r['config']['arm'],r['config']['seed'],r['config']['consistency_strength']) for r in records.values()]==expected
    assert len({r['config']['run_directory'] for r in records.values()})==9
    energy=list(records.values())[-1];assert energy['fit_wall_seconds']==500.==s['fit_wall_seconds']
    for r in records.values():
        cfg=r['config'];assert cfg['max_updates']==200000 and cfg['model']['features']==64 and cfg['learning_rate']==.003 and cfg['batch_size']==4
        assert cfg['loss_weights']=={'energy':0.014584365670390719,'force':1.9854156343296092}
        assert r['initial_identity']==s['initial_identities'][cfg['arm']][str(cfg['seed'])]
        assert r['fit_wall_seconds']+r['score_wall_seconds']==r['slot_wall_seconds']
    assert list(records.values())[0]['initial_identity']==list(records.values())[1]['initial_identity']
    assert list(records.values())[2]['initial_identity']==list(records.values())[3]['initial_identity']
    assert list(records.values())[0]['initial_identity']!=list(records.values())[2]['initial_identity']
    m.validate_comparison_records(records,b,s,tmp_path/'output')
    wrong=copy.deepcopy(records);next(iter(wrong.values()))['config']['max_updates']=50000
    with pytest.raises(ValueError):m.validate_comparison_records(wrong,b,s,tmp_path/'output')
    b['reserve_multiplier']=1.
    with pytest.raises(ValueError):m.comparison_records(b,s,tmp_path/'output')


def test_primary_score_failure_keeps_completed_fit_and_stops_without_next_fit(tmp_path,monkeypatch):
    m=module();r={'candidate':'direct-0','config':{'run_directory':str(tmp_path/'fit')},'fit_wall_seconds':10.,'score_wall_seconds':2.,'slot_wall_seconds':12.,'diagnostic_wall_seconds':180.}
    calls=[]
    def invoke(command,log,timeout):
        calls.append(command)
        if '--train-worker' not in command:raise RuntimeError('synthetic scoring failure')
    monkeypatch.setattr(m,'invoke',invoke)
    monkeypatch.setattr(m,'completed_fit',lambda *args:{'updates':200000,'inventory':{}})
    state=m.execute_slot(r,tmp_path,100000000000.)
    assert state['fit']=='completed' and state['score']=='failed' and state['diagnostic']=='pending'
    assert len(calls)==2 and sum('--train-worker' in c for c in calls)==1
    assert state['automatic_retry'] is False
    assert state['fit_artifacts_unchanged'] is True


def test_expired_deadline_does_not_consume_unstarted_slot(tmp_path,monkeypatch):
    m=module();r={'candidate':'direct-0','config':{'run_directory':str(tmp_path/'fit')},'fit_wall_seconds':10.,'score_wall_seconds':2.,'slot_wall_seconds':12.,'diagnostic_wall_seconds':180.}
    calls=[];monkeypatch.setattr(m,'invoke',lambda *a:calls.append(a))
    with pytest.raises(TimeoutError):m.execute_slot(r,tmp_path,-1.)
    assert not calls and not list(tmp_path.iterdir())


def test_failed_scorer_still_audits_completed_fit_bytes(tmp_path,monkeypatch):
    m=module();fit=tmp_path/'fit';fit.mkdir();saved=fit/'checkpoint';saved.write_text('original')
    r={'candidate':'direct-0','config':{'run_directory':str(fit)},'fit_wall_seconds':10.,'score_wall_seconds':2.,'slot_wall_seconds':12.,'diagnostic_wall_seconds':180.}
    original=m.run_inventory(fit)
    monkeypatch.setattr(m,'completed_fit',lambda *a:{'updates':200000,'inventory':original})
    def invoke(command,*args):
        if '--train-worker' not in command:
            saved.write_text('changed');raise RuntimeError('synthetic scorer failed after mutation')
    monkeypatch.setattr(m,'invoke',invoke);state=m.execute_slot(r,tmp_path,100000000000.)
    assert state['fit']=='completed' and state['score']=='failed'
    assert state['fit_artifacts_unchanged'] is False
    assert 'artifacts changed' in state['artifact_audit_failure']


def test_failed_fit_is_consumed_without_scoring_or_replacement(tmp_path,monkeypatch):
    m=module();r={'candidate':'direct-0','config':{'run_directory':str(tmp_path/'fit')},'fit_wall_seconds':10.,'score_wall_seconds':2.,'slot_wall_seconds':12.,'diagnostic_wall_seconds':180.}
    calls=[]
    def invoke(command,*args):calls.append(command);raise RuntimeError('synthetic fit failure')
    monkeypatch.setattr(m,'invoke',invoke);state=m.execute_slot(r,tmp_path,100000000000.)
    assert state['fit']=='failed' and state['fit_slot_consumed'] is True and state['score']=='pending' and len(calls)==1


def score(value,update=500):return {'status':'verified','equal_connection_force_mae_eV_per_A':value,'best_update':update}


def test_matching_uses_two_seeds_only_force_mae_and_lower_lambda_tie():
    m=module();scores={'direct-0':score(.04),'direct-1':score(.044),'consistency-1-0':score(.041),'consistency-1-1':score(.045),'consistency-0.1-0':score(.041),'consistency-0.1-1':score(.045),'consistency-10-0':score(.02),'consistency-10-1':score(.021)}
    result=m.matching_decision(scores);assert result['selected_lambda']==.1 and result['matched_lambdas']==[.1,1.]
    # Diagnostic errors/energy fields cannot influence matching.
    scores['consistency-0.1-0']['diagnostic_status']='failed';scores['consistency-0.1-0']['energy_mae_eV']=999
    assert m.matching_decision(scores)['selected_lambda']==.1
    del scores['direct-1'];assert m.matching_decision(scores)['selected_lambda'] is None
    # Range criterion rules out a narrow pair even inside the 10% mean band.
    scores['direct-1']=score(.04);assert m.matching_decision(scores)['selected_lambda'] is None


def test_initial_checkpoint_ties_are_disclosed_without_posthoc_exclusion():
    m=module();scores={m.slot_id(a,s,l):score(.06,0) for a,s,l in m.ORDER if a!='energy'}
    result=m.matching_decision(scores)
    assert result['selected_lambda']==.1 and result['matched_lambdas']==[.1,1.,10.]
    assert result['status']=='matched_with_initial_checkpoints'
    assert result['direct']['best_updates']==[0,0]
    assert len(result['initial_checkpoint_candidates'])==8
    assert result['learned_comparison_claim_supported'] is False


def test_wrong_import_rejected_before_any_output(tmp_path,monkeypatch):
    m=module()
    def wrong():raise ValueError('Wrong imported trainer')
    monkeypatch.setattr(m,'assert_imports',wrong)
    with pytest.raises(ValueError,match='Wrong imported trainer'):m.run_comparison(tmp_path/'missing',tmp_path/'out',tmp_path/'approval')
    assert not (tmp_path/'out').exists()


def driver_fixture(tmp_path,monkeypatch):
    m=module();budget,selected=inputs(tmp_path);output=tmp_path/'out'
    records=m.comparison_records(budget,selected,output)
    envelope={'budget':budget,'selected_record':selected,'records':records,'frozen_inputs':{},'label_report':str(tmp_path/'labels'),
        'global_wall_seconds':sum(r['slot_wall_seconds']+1.25*r['diagnostic_wall_seconds'] for r in records.values()),'phase_reconciliation':{'margin_seconds':1.}}
    m.save_json(tmp_path/'records',envelope);m.save_json(tmp_path/'labels',{});(tmp_path/'data').mkdir();m.save_json(tmp_path/'data/manifest.json',{})
    reviews={}
    for i in range(2):
        path=tmp_path/f'review-{i}';path.write_text(str(i));reviews[str(path)]=m.digest(path.read_bytes())
    approval={'status':'accepted_for_comparison','records_sha256':m.digest((tmp_path/'records').read_bytes()),'external_scripts':selected['external_scripts'],'review_sha256':reviews}
    for name,value in [('assert_imports',lambda:None),('source_pin',lambda:selected['source_pin']),('dataset_pin',lambda *a:selected['dataset_pin']),('external_scripts',lambda:selected['external_scripts']),('label_coverage',lambda *a:True),('require_storage',lambda *a:True)]:monkeypatch.setattr(m,name,value)
    return m,envelope,approval,output


@pytest.mark.parametrize('count',[0,1])
def test_two_filled_record_reviews_required_before_output(tmp_path,monkeypatch,count):
    m,env,approval,out=driver_fixture(tmp_path,monkeypatch)
    approval['review_sha256']=dict(list(approval['review_sha256'].items())[:count]);m.save_json(tmp_path/'approval',approval)
    with pytest.raises(ValueError,match='Two independent'):m.run_comparison(tmp_path/'records',out,tmp_path/'approval')
    assert not out.exists()


def test_driver_keeps_literal_order_after_sorted_json_roundtrip(tmp_path,monkeypatch):
    m,env,approval,out=driver_fixture(tmp_path,monkeypatch);calls=[]
    def execute(r,*args):
        calls.append(r['candidate']);return {'candidate':r['candidate'],'fit':'completed','score':'completed','diagnostic':'completed','primary_score':score(.04)}
    monkeypatch.setattr(m,'execute_slot',execute);m.save_json(tmp_path/'approval',approval)
    m.run_comparison(tmp_path/'records',out,tmp_path/'approval')
    assert calls==[m.slot_id(*x) for x in m.ORDER]==m.read_json(out/'SESSION.json')['order']
    assert m.read_json(out/'COMPLETE.json')['A1_redecided'] is False


@pytest.mark.parametrize('margin',[-1.,float('nan')])
def test_bad_reconciliation_rejected_before_output(tmp_path,monkeypatch,margin):
    m,env,approval,out=driver_fixture(tmp_path,monkeypatch);env['phase_reconciliation']['margin_seconds']=margin
    (tmp_path/'records').write_text(json.dumps(env));approval['records_sha256']=m.digest((tmp_path/'records').read_bytes());m.save_json(tmp_path/'approval',approval)
    with pytest.raises(ValueError,match='reconciliation'):m.run_comparison(tmp_path/'records',out,tmp_path/'approval')
    assert not out.exists()
