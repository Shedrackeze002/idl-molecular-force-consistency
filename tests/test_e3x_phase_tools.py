"""Guard budget arithmetic, outcome eligibility and full host-work measurement."""
import importlib.util
import sys
from pathlib import Path
import json
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))


def tool(name):
    assert importlib.util.find_spec(name) is not None, f'{name} implementation missing'
    return __import__(name)


def cells():
    return [{'id':f'{w}-{a}','width':w,'arm':a,'status':'complete',
             'full_update_seconds':[.01]*100,'kernel_update_seconds':[.001]*100,
             'validation_seconds':[1.,2.,1.], 'initialization_seconds':1.,
             'compilation_seconds':2.,'warmup_seconds':.1}
            for w in (32,64) for a in ('energy','direct','consistency')]


def io():
    return {'validation_base_bytes':2,'validation_event_bytes':10,'loss_base_bytes':2,
            'loss_row_bytes':3,'checkpoint_base_bytes':20,'checkpoint_event_bytes':5,
            'fixed_output_bytes':100,'seconds_per_byte':.000001,'fixed_event_seconds':.01,
            'reload_and_score_seconds':2.}


def test_io_counts_initial_scheduled_and_duplicate_final():
    b=tool('estimate_e3x_budget');x=b.io_accounting(1000,io())
    # three validations; losses at 500/1000/final; LAST likewise; BEST at 0/500/1000.
    assert x['writes']=={'validation':3,'losses':3,'last':3,'best':3}
    assert x['bytes_by_format']['validation']==3*2+10*(1+2+3)
    assert x['bytes_by_format']['losses']==3*2+3*(500+1000+1000)
    assert x['bytes_by_format']['checkpoints']==6*20+5*((2+3+3)+(1+2+3))
    assert x['retained_bytes'] < x['cumulative_bytes']


def test_ledger_twelve_slots_no_double_count_and_reserve():
    b=tool('estimate_e3x_budget');x=b.forecast(cells(),io(),50000,aux_seconds=7)
    assert len(x['slots'])==12 and len(set(x['slots']))==12
    assert x['reserved_total_seconds']==pytest.approx(1.25*(7+sum(x['slots'].values())))
    assert x['reserved_baseline_seconds']==pytest.approx(1.25*sum(x['slots'][n] for n in ('energy-A-0','energy-B-0','energy-C-0')))
    assert x['per_recipe']['A']['energy']['ordinary_update_seconds']==pytest.approx(.01)


def test_missing_duplicate_incomplete_nonfinite_timing_refused():
    b=tool('estimate_e3x_budget')
    for bad in (cells()[:-1],cells()+[cells()[0]]):
        with pytest.raises(ValueError):b.forecast(bad,io(),50000)
    for key,val in [('status','failed'),('full_update_seconds',[float('nan')]*100)]:
        bad=cells();bad[0][key]=val
        with pytest.raises(ValueError):b.forecast(bad,io(),50000)


def test_update_rule_largest_grid_and_empty_set():
    b=tool('estimate_e3x_budget');ts=cells();i=io()
    ceiling=b.choose_updates(ts,i,total_seconds=1e9,baseline_seconds=1e9)
    assert ceiling['updates']==200000
    boundary=b.forecast(ts,i,50500)
    chosen=b.choose_updates(ts,i,total_seconds=boundary['reserved_total_seconds'],baseline_seconds=1e9)
    assert chosen['updates']==50500
    with pytest.raises(ValueError,match='feasible'):b.choose_updates(ts,i,total_seconds=1,baseline_seconds=1)
    baseline=b.forecast(ts,i,50000)['reserved_baseline_seconds']
    assert b.choose_updates(ts,i,total_seconds=1e9,baseline_seconds=baseline)['updates']==50000


def test_full_update_timer_includes_host_and_synchronization():
    t=tool('benchmark_e3x_budget');clock=[0.];events=[]
    def work():clock[0]+=9;events.append('host');return 'result',.1
    def sync(result):assert result=='result';clock[0]+=4;events.append('sync')
    result=t.measure_update(work,sync,clock=lambda:clock[0])
    assert result['full_seconds']==13 and result['kernel_seconds']==.1
    assert events==['host','sync']
    assert tool('estimate_e3x_budget').conservative_update([13.]*100)==13


def test_fresh_output_and_wrong_pin_refused(tmp_path):
    c=tool('e3x_phase_common');p=tmp_path/'run';c.fresh_directory(p)
    with pytest.raises(FileExistsError):c.fresh_directory(p)
    with pytest.raises(ValueError):c.require_equal_pin({'code_fingerprint':'a'},{'code_fingerprint':'b'})


def test_equal_connection_rescore_and_zero_force_not_pooled():
    s=tool('evaluate_e3x_adequacy');ids=['a','b','c'];groups=['g1','g2','g2']
    forces=np.array([[[3.,3.,3.]],[[1.,1.,1.]],[[1.,1.,1.]]]);pred=np.zeros_like(forces)
    x=s.prediction_metrics(ids,groups,np.zeros(3),pred,np.zeros(3),forces)
    assert x['equal_connection_force_mae_eV_per_A']==2
    assert x['pooled_force_mae_eV_per_A']==pytest.approx(5/3)
    assert x['zero_force_equal_connection_mae_eV_per_A']==2
    with pytest.raises(ValueError):s.prediction_metrics(['a','a','c'],groups,np.zeros(3),pred,np.zeros(3),forces)
    pred[0,0,0]=np.nan
    with pytest.raises(ValueError):s.prediction_metrics(ids,groups,np.zeros(3),pred,np.zeros(3),forces)


def record(cid='A',score=.05,**changes):
    r={'candidate':cid,'status':'verified','best_update':500,'updates':50000,'max_updates':50000,
       'stop_reason':'max_updates','record_count':974,'connection_count':5,'source_matches':True,
       'best_reload_verified':True,'zero_force_crosscheck':True,'equal_connection_force_mae_eV_per_A':score}
    return {**r,**changes}


def test_once_only_a1_boundary_slots_and_ties():
    s=tool('evaluate_e3x_adequacy')
    assert s.evaluate_a1([record('C',.05),record('B',.05),record('A',.05)],50000,True)['selected_candidate']=='A'
    assert s.evaluate_a1([record('A',.0500001),record('B',status='failed'),record('C',best_update=0)],50000,True)['a1_passed'] is False
    for changes in ({'best_update':0},{'updates':49999},{'source_matches':False},{'record_count':973},{'zero_force_crosscheck':False},{'best_reload_verified':False}):
        assert s.evaluate_a1([record('A',**changes),record('B',status='failed'),record('C',status='failed')],50000,True)['a1_passed'] is False
    for bad in ([record('A')],[record('A'),record('A'),record('C')],[record('A'),record('B'),record('C'),record('D')]):
        with pytest.raises(ValueError):s.evaluate_a1(bad,50000,True)
    with pytest.raises(ValueError):s.evaluate_a1([record('A'),record('B'),record('C')],50000,False)


def test_real_synthetic_checkpoint_scorer_full_arrays_and_pin_refusal(tmp_path,monkeypatch):
    # Synthetic prepared records only; never the scientific dataset or a candidate slot.
    from dataclasses import asdict
    from force_consistency.data.prepare import export_prepared
    from force_consistency.data.types import PreparationPolicy
    from force_consistency.models.backbone import ModelConfig
    from force_consistency.train import RunConfig,train
    from test_splits_prepare import fixture,empty_duplicates,UNIT_EVIDENCE
    records,inv,ends=fixture();data=tmp_path/'data'
    manifest=export_prepared(records,directory=data,inventory=inv,endpoints=ends,duplicates=empty_duplicates(),policy=PreparationPolicy(),molecule='aspirin',unit_evidence=UNIT_EVIDENCE)
    common=tool('e3x_phase_common');scorer=tool('evaluate_e3x_adequacy')
    monkeypatch.setattr(common,'SPLIT',manifest['split_hash']);monkeypatch.setattr(scorer,'SPLIT',manifest['split_hash'])
    generated=common.fixed_config('A',data,tmp_path/'run',50000,180)
    generated.update(expected_split_hash=manifest['split_hash'],max_updates=2,validation_interval=1,model=asdict(ModelConfig(features=4,iterations=1,radial_basis=4,energy_readout_init='zero')))
    config=RunConfig(**generated)
    opened=[];real_load=np.load
    def watched_load(file,*args,**kwargs):
        name=str(file if isinstance(file,(str,Path)) else getattr(file,'name',file))
        assert Path(name).name!='test.npz','test partition opened'
        opened.append(name)
        return real_load(file,*args,**kwargs)
    monkeypatch.setattr(np,'load',watched_load)
    with pytest.raises(AssertionError,match='test partition'):watched_load(data/'test.npz')
    result=train(config);assert result['status']=='passed',result
    launch={'candidate':'A','config':generated,'source_pin':common.source_pin(),'dataset_pin':common.dataset_pin(data),
            'initial_identity':{k:result['provenance'][k] for k in ('initial_parameter_sha256','initial_state_sha256')}}
    scored=scorer.score_best(tmp_path/'run',launch,tmp_path/'score')
    assert scored['zero_force_crosscheck'] and scored['best_reload_verified']
    with np.load(tmp_path/'score/best-validation.npz',allow_pickle=False) as arrays:
        assert arrays['record_ids'].tolist()==manifest['partitions']['validation']['record_ids']
        assert arrays['forces'].shape==arrays['reference_forces'].shape
    assert any(n.endswith('/validation.npz') for n in opened) and not any(n.endswith('/test.npz') for n in opened)
    benchmark=tool('benchmark_e3x_budget');monkeypatch.setattr(benchmark,'SPLIT',manifest['split_hash'])
    with pytest.raises(ValueError,match='full dataset'):benchmark.setup(data,32,'energy')
    assert not any(n.endswith('/test.npz') for n in opened)
    launch['source_pin']['code_fingerprint']='wrong'
    with pytest.raises(ValueError,match='pin'):scorer.score_best(tmp_path/'run',launch,tmp_path/'wrong-score')


def test_label_coverage_wrong_branch_split_or_identity_refused():
    c=tool('e3x_phase_common');groups={'train':[f't{i}' for i in range(26)],'validation':[f'v{i}' for i in range(5)],'test':[f'x{i}' for i in range(6)]}
    manifest={'identity':'m','splits':{'groups':{'aspirin':groups}}}
    connections={n:{'split':s,'branches':{'plus':{'pairs':1},'minus':{'pairs':1}}} for s in ('train','validation') for n in groups[s]}
    report={'manifest_identity':'m','test_connections_skipped':groups['test'],'connections':connections}
    assert c.label_coverage(report,manifest)
    report['connections']['t0']['branches']['minus']['pairs']=0
    with pytest.raises(ValueError):c.label_coverage(report,manifest)
    report['connections']['t0']['branches']['minus']['pairs']=1;report['manifest_identity']='wrong'
    with pytest.raises(ValueError):c.label_coverage(report,manifest)


def test_io_fixture_metadata_allowed_and_missing_cost_refused():
    b=tool('estimate_e3x_budget');model={**io(),'scope':'synthetic fixture','fixture_sizes':{'a':10}}
    assert b.forecast(cells(),model,50000)['updates']==50000
    model['seconds_per_byte']=float('nan')
    with pytest.raises(ValueError):b.forecast(cells(),model,50000)


def test_generated_launch_config_matches_trainer_saved_full_schema(tmp_path):
    from dataclasses import asdict
    from force_consistency.train import RunConfig
    c=tool('e3x_phase_common');generated=c.fixed_config('A',tmp_path/'data',tmp_path/'run',50000,1000)
    assert generated==asdict(RunConfig(**generated))


def test_timeout_resistant_worker_killed_and_reaped(tmp_path,monkeypatch):
    import subprocess,signal
    t=tool('benchmark_e3x_budget');signals=[]
    class Child:
        pid=999999
        killed=False
        def poll(self):return -9 if self.killed else None
        def wait(self,timeout):
            if self.killed:return -9
            raise subprocess.TimeoutExpired('mock',timeout)
    child=Child()
    monkeypatch.setattr(t.subprocess,'Popen',lambda *a,**kw:child)
    def kill(pid,sig):signals.append(sig);child.killed=sig==signal.SIGKILL
    monkeypatch.setattr(t.os,'killpg',kill)
    with pytest.raises(subprocess.TimeoutExpired):t.invoke(['mock'],tmp_path/'worker.txt',3)
    assert signal.SIGKILL in signals and child.poll() is not None


def test_global_and_slot_deadline_caps_scoring_remainder():
    m=tool('launch_e3x_candidates');assert hasattr(m,'remaining_seconds'),'shared deadline guard missing'
    assert m.remaining_seconds(100,91,clock=lambda:90)==1
    with pytest.raises(TimeoutError):m.remaining_seconds(100,91,clock=lambda:91)


def test_record_set_rejects_different_U_or_budget_before_slots():
    m=tool('launch_e3x_candidates');assert hasattr(m,'validate_record_set'),'common-U boundary missing'
    data={r:{'candidate':r,'config':{'max_updates':50000,'seed':0,'arm':'energy'},'budget_sha256':'same','source_pin':{'same':1},'dataset_pin':{'same':1}} for r in ('A','B','C')}
    assert m.validate_record_set(data)==50000
    data['B']['config']['max_updates']=50500
    with pytest.raises(ValueError):m.validate_record_set(data)
    data['B']['config']['max_updates']=50000;data['B']['budget_sha256']='different'
    with pytest.raises(ValueError):m.validate_record_set(data)


def test_complete_history_and_first_best_selection_audit():
    s=tool('evaluate_e3x_adequacy');assert hasattr(s,'verify_selection_history'),'BEST history audit missing'
    h=[{'update':u,'equal_connection_force_mae_eV_per_A':score,'record_count':974,'per_group':{f'g{i}':{} for i in range(5)}} for u,score in [(0,.0625),(500,.04),(1000,.04)]]
    assert s.verify_selection_history(h,1000,500,500,.04)==h[1]
    for bad,selected in [(h,1000),(h[:-1],500),(h+[h[-1]],500)]:
        with pytest.raises(ValueError):s.verify_selection_history(bad,1000,500,selected,.04)
    h[1]['equal_connection_force_mae_eV_per_A']=float('nan')
    with pytest.raises(ValueError):s.verify_selection_history(h,1000,500,500,.04)


def test_numerical_helper_context_is_explicit_cpu():
    import jax
    b=tool('benchmark_e3x_budget');assert hasattr(b,'cpu_context'),'CPU context missing'
    with b.cpu_context():assert jax.config.jax_default_device.platform=='cpu'


def test_checkpoint_calibration_uses_production_atomic_api(tmp_path):
    from flax import serialization
    b=tool('benchmark_e3x_budget');assert hasattr(b,'calibrate_checkpoint'),'production API calibration missing'
    state={'params':{'x':np.ones(3,dtype=np.float64)},'update':0};control={'best_state':serialization.to_bytes(state),'validation_history':[]}
    result=b.calibrate_checkpoint(tmp_path/'fixture',state,{'synthetic':True},control)
    assert result['bytes']==(tmp_path/'fixture').stat().st_size
    assert result['serialization_seconds']>=0 and result['write_seconds']>=0


def test_storage_guard_refuses_capacity_shortfall_before_launch(tmp_path,monkeypatch):
    import shutil
    m=tool('launch_e3x_candidates');assert hasattr(m,'require_storage'),'fresh capacity check missing'
    monkeypatch.setattr(m.shutil,'disk_usage',lambda p:shutil._ntuple_diskusage(10,10,0))
    with pytest.raises(OSError):m.require_storage(tmp_path,1)


def test_launcher_caps_actual_scorer_invocation_to_baseline_remainder(tmp_path,monkeypatch):
    m=tool('launch_e3x_candidates');b=tool('benchmark_e3x_budget');c=tool('e3x_phase_common')
    clock=[0.];score_timeouts=[];root=tmp_path/'baseline';data=tmp_path/'data';data.mkdir();c.save_json(data/'manifest.json',{})
    labels=tmp_path/'labels.json';c.save_json(labels,{})
    monkeypatch.setattr(m.time,'perf_counter',lambda:clock[0])
    monkeypatch.setattr(m,'source_pin',lambda:{'same':1});monkeypatch.setattr(m,'dataset_pin',lambda p:{'same':1})
    monkeypatch.setattr(m,'label_coverage',lambda *args:True)
    monkeypatch.setattr(m,'require_storage',lambda *args:{'free_bytes':10,'required_bytes':1})
    records={r:{'candidate':r,'config':c.fixed_config(r,data,root/r,50000,50000),
       'source_pin':{'same':1},'dataset_pin':{'same':1},'budget_sha256':'same','initial_identity':{},
       'forecast_seconds':48000,'fit_wall_seconds':50000,'score_wall_seconds':10000,'slot_wall_seconds':60000,
       'required_remaining_storage_bytes':1} for r in ('A','B','C')}
    path=tmp_path/'records.json';c.save_json(path,records)
    approval=tmp_path/'approval.json';c.save_json(approval,{'status':'accepted_for_candidates','source_pin':{'same':1},'launch_records_sha256':c.digest(path.read_bytes())})
    def fake_invoke(command,log,timeout,**kw):
        if '--train-worker' in command:clock[0]=43199.6
        else:
            score_timeouts.append(timeout);out=Path(command[command.index('--output')+1]);out.mkdir()
            c.save_json(out/'score.json',record('C',.04));clock[0]=43200
    monkeypatch.setattr(b,'invoke',fake_invoke)
    result=m.launch_candidates(path,root,approval,labels)
    assert score_timeouts==pytest.approx([.4])
    assert result['consumed_slots']==3


def test_supervisor_normal_and_nonzero_workers(tmp_path):
    b=tool('benchmark_e3x_budget')
    b.invoke([sys.executable,'-c','print("ok")'],tmp_path/'normal.txt',3)
    assert 'ok' in (tmp_path/'normal.txt').read_text()
    with pytest.raises(RuntimeError,match='exit 7'):b.invoke([sys.executable,'-c','raise SystemExit(7)'],tmp_path/'bad.txt',3)


def test_child_imports_frozen_trainer_even_with_wrong_parent_pythonpath(tmp_path,monkeypatch):
    b=tool('benchmark_e3x_budget');c=tool('e3x_phase_common')
    old_root=c.ROOT.parents[1]
    monkeypatch.setenv('PYTHONPATH',str(old_root/'src'))
    log=tmp_path/'import-witness.txt'
    b.invoke([sys.executable,'-c','import force_consistency.train as t;print(t.__file__)'],log,10)
    assert log.read_text().strip()==str(c.ROOT/'src/force_consistency/train.py')


def test_parent_import_mismatch_stops_before_any_output_or_approval_read(tmp_path,monkeypatch):
    m=tool('launch_e3x_candidates')
    def wrong_import():raise ValueError('Wrong imported trainer')
    monkeypatch.setattr(m,'assert_imports',wrong_import)
    with pytest.raises(ValueError,match='Wrong imported trainer'):
        m.launch_candidates(tmp_path/'missing-records',tmp_path/'never-created',tmp_path/'missing-approval',tmp_path/'missing-labels')
    assert not (tmp_path/'never-created').exists()


# Regression: raw authenticated state identity must survive template dictionary ordering.
def best_identity_fixture(tmp_path):
    from flax import serialization
    from force_consistency.runs import save_checkpoint
    state={'params':{'z':np.array([1.25],dtype=np.float64),'a':np.array([-2.],dtype=np.float64)},
           'optimizer':{'count':np.array(7,dtype=np.int32),'moment':np.array([.3],dtype=np.float64)},
           'update':7,'sampler_rng_json':'{"seed":0}'}
    payload=serialization.to_bytes(state);prov={'synthetic':'nonzero-state-order-regression'}
    path=tmp_path/'best.msgpack';save_checkpoint(path,state,prov,run_control={'best_state':payload,'best_update':7})
    return path,prov,state,payload


def test_embedded_best_identity_accepts_nonzero_state_with_template_key_reordering(tmp_path):
    from flax import serialization
    from force_consistency.runs import load_checkpoint
    import hashlib
    path,prov,state,payload=best_identity_fixture(tmp_path)
    template={**state,'params':{'a':state['params']['a'],'z':state['params']['z']}}
    restored=load_checkpoint(path,prov,template=template)
    assert restored['update']==7
    for k in ('a','z'):np.testing.assert_array_equal(restored['params'][k],state['params'][k])
    assert serialization.to_bytes(restored)!=payload, 'must reproduce the old false-rejection condition'
    scorer=tool('evaluate_e3x_adequacy')
    assert hasattr(scorer,'verify_embedded_best'),'authenticated raw-state verifier missing'
    result=scorer.verify_embedded_best(path,prov)
    assert result['payload_sha256']==hashlib.sha256(payload).hexdigest()
    assert result['embedded_best_state_sha256']==result['payload_sha256']
    assert result['exact_original_payload_match'] is True
    envelope=serialization.msgpack_restore(path.read_bytes())
    assert result['control_sha256']==hashlib.sha256(envelope['control_payload']).hexdigest()


@pytest.mark.parametrize('new_value',[9.,np.nextafter(1.25,np.inf)])
def test_embedded_best_identity_rejects_different_internally_authenticated_state(tmp_path,new_value):
    from flax import serialization
    import hashlib
    path,prov,state,payload=best_identity_fixture(tmp_path)
    envelope=serialization.msgpack_restore(path.read_bytes());control=serialization.msgpack_restore(envelope['control_payload'])
    other={**state,'params':{**state['params'],'z':np.array([new_value],dtype=np.float64)}}
    control['best_state']=serialization.to_bytes(other)
    envelope['control_payload']=serialization.msgpack_serialize(control)
    envelope['control_sha256']=hashlib.sha256(envelope['control_payload']).hexdigest()
    path.write_bytes(serialization.msgpack_serialize(envelope))
    scorer=tool('evaluate_e3x_adequacy')
    assert hasattr(scorer,'verify_embedded_best'),'authenticated raw-state verifier missing'
    with pytest.raises(ValueError,match='BEST embedded state mismatch'):scorer.verify_embedded_best(path,prov)


@pytest.mark.parametrize('corruption',['payload','control','provenance'])
def test_embedded_best_identity_rejects_failed_authentication(tmp_path,corruption):
    from flax import serialization
    path,prov,state,payload=best_identity_fixture(tmp_path)
    envelope=serialization.msgpack_restore(path.read_bytes())
    if corruption=='payload':envelope['payload']+=b'bad'
    elif corruption=='control':envelope['control_payload']+=b'bad'
    else:prov={'synthetic':'wrong-provenance'}
    path.write_bytes(serialization.msgpack_serialize(envelope))
    scorer=tool('evaluate_e3x_adequacy')
    assert hasattr(scorer,'verify_embedded_best'),'authenticated raw-state verifier missing'
    with pytest.raises(ValueError):scorer.verify_embedded_best(path,prov)
