"""Checks protect ancestry, nonzero-moment schedules and honest saved outcomes."""
import copy
import json
from pathlib import Path
import numpy as np
import pytest
from force_consistency import late_rate as lr


def config():
    return json.loads(Path('configs/aspirin-late-rate.json').read_text())


@pytest.mark.parametrize('arm,update,wanted',[
    ('constant',6001,.001),('constant',8001,.001),('constant',10000,.001),
    ('late_decay',6001,.0003),('late_decay',8000,.0003),
    ('late_decay',8001,.0001),('late_decay',10000,.0001)])
def test_rate_at_actual_committed_update(arm,update,wanted):
    assert lr.rate_for_update(arm,update)==wanted


@pytest.mark.parametrize('arm,update',[('unknown',7000),('constant',6000),('late_decay',10001)])
def test_schedule_cannot_escape_reviewed_interval(arm,update):
    with pytest.raises(ValueError):lr.rate_for_update(arm,update)


@pytest.mark.parametrize('damage',['rate','weight','end','score','pins','parent'])
def test_changed_reviewed_contract_is_rejected(damage):
    c=config()
    if damage=='rate':c['learning_rates']['late_decay'][0][1]=.0005
    if damage=='weight':c['loss_weights']['energy']=1.
    if damage=='end':c['end_update']=11000
    if damage=='score':c['score_updates'].remove(8001)
    if damage=='pins':c['source_pins_sha256']='0'*64
    if damage=='parent':c['parent_directory']='other'
    with pytest.raises(ValueError):lr.validate_contract(c)


def test_file_pin_guard_rejects_changed_actual_bytes(tmp_path):
    p=tmp_path/'source';p.write_text('original')
    pins={str(p):lr.digest(p)};lr.verify_pins(pins)
    p.write_text('modified')
    with pytest.raises(ValueError):lr.verify_pins(pins)


def test_residual_decomposition_preserves_raw_energy_gate():
    r=lr.residual_summary(np.array([1.,3.]),np.zeros(2),np.ones((2,1,3)),np.zeros((2,1,3)))
    assert r['energy_mse_eV2']==5.
    assert r['signed_mean_energy_error_eV']==2.
    assert r['centered_energy_error_variance_eV2']==1.
    assert r['mean_energy_error_squared_eV2']==4.
    assert r['force_mse_eV2_per_A2']==1.


def test_saved_only_analysis_reconciles_groups_and_spike():
    c=config();lr.validate_contract(c)
    r=lr.saved_observations(Path(c['parent_directory']))
    assert r['new_training_updates']==r['new_prediction_frames']==0
    assert r['spike']['sampled_peak_update']==2000
    assert len(r['endpoint_per_record'])==128
    assert r['endpoint']['energy_mse_eV2']==pytest.approx(.007645024080897856)
    assert r['per_group_reconstruction_max_error']<1e-12


def tiny_prepared(start=6000):
    import jax
    from test_coverage import seven_case
    from force_consistency.loss_balance import build_weighted_step
    from force_consistency.coverage_runs import coverage_metrics
    p,m,b,n,s=seven_case();base=coverage_metrics(s['params'],p,p,m,3)
    step=build_weighted_step(m,n,.001,lr.WEIGHTS);step.prepare(s['params'],s['optimizer'],[b])
    for _ in range(3):
        pp,oo,_=step(s['params'],s['optimizer'],[b]);jax.block_until_ready((pp,oo));s={**s,'params':pp,'optimizer':oo}
    s={**s,'update':start,'optimizer':(s['optimizer'][0]._replace(count=np.asarray(start,dtype=np.int32)),*s['optimizer'][1:])}
    c=config();c.update(start_update=start,end_update=start+3,score_updates=list(range(start,start+4)),microbatch_size=3,finalization_reserve_seconds=.01)
    prepared={'state':s,'fitting':p,'anchor':p,'model':m,'norm':n,'baselines':base,
              'source_pins':{},'code':{'code_files':{'src/force_consistency/late_rate.py':lr.digest('src/force_consistency/late_rate.py')}}}
    return c,prepared


def tiny_probe(root,timeout):
    import subprocess,sys
    subprocess.run([sys.executable,'-c','import sys; from force_consistency.late_rate import _probe_saved; _probe_saved(sys.argv[1])',str(root.resolve())],check=True,capture_output=True,timeout=timeout)
    return lr.read(root/'checkpoint-verification.json')


@pytest.mark.parametrize('start',[6000,7999])
def test_nonzero_moments_survive_rate_boundary_and_real_child(tmp_path,start):
    import jax
    from test_coverage import close_trees
    from force_consistency.loss_balance import weighted_loss
    from force_consistency.metrics import partition_batch
    from force_consistency.width_rate_runs import independent_adam_step
    from force_consistency.runs import load_checkpoint
    c,p=tiny_prepared(start);wanted=copy.deepcopy(p['state']);full=partition_batch(p['fitting'],np.arange(7))
    gradient=jax.jit(jax.grad(lambda pp:weighted_loss(pp,full,p['model'],p['norm'],lr.WEIGHTS)[0]))
    for u in range(start+1,start+4):
        g=gradient(wanted['params']);pp,oo=independent_adam_step(wanted['params'],wanted['optimizer'],g,lr.rate_for_update('late_decay',u))
        wanted={**wanted,'params':pp,'optimizer':oo,'update':u}
    before=lr.state_hashes(p['state'])
    out=lr._run_arm(c,'late_decay',p,tmp_path/'arm',lr.time.perf_counter()+120,probe_runner=tiny_probe)
    assert out['execution_valid'],out.get('failure_reason')
    assert out['additional_updates']==3 and out['additional_frame_exposures']==21
    assert out['persisted_additional_updates']==3
    assert out['checkpoint_verification']['adam_count']==start+3
    got=load_checkpoint(tmp_path/'arm/snapshots'/f'{start+3}.msgpack',lr.read(tmp_path/'arm/provenance.json'),template=p['state'])
    close_trees((got['params'],got['optimizer']),(wanted['params'],wanted['optimizer']))
    assert lr.state_hashes(p['state'])==before
    with pytest.raises(FileExistsError):lr._run_arm(c,'late_decay',p,tmp_path/'arm',lr.time.perf_counter()+120,probe_runner=tiny_probe)


def test_interrupted_unscored_work_keeps_persisted_receipt(tmp_path,monkeypatch):
    c,p=tiny_prepared();c['end_update']=6005;c['score_updates']=[6000,6001,6005]
    real=lr.build_weighted_step
    def factory(*args):
        step=real(*args);calls=0
        def wrapped(*a):
            nonlocal calls
            calls+=1
            # warmup, oracle proposal, real6001, real6002, interrupted6003
            if calls==5:raise KeyboardInterrupt('unreturned numerical operation')
            return step(*a)
        wrapped.prepare=step.prepare;return wrapped
    monkeypatch.setattr(lr,'build_weighted_step',factory)
    out=lr._run_arm(c,'constant',p,tmp_path/'arm',lr.time.perf_counter()+120,probe_runner=tiny_probe)
    assert not out['execution_valid']
    assert out['additional_updates']==2 and out['persisted_additional_updates']==1
    assert out['attempted_additional_updates']==3 and out['pending_update_unknown']
    assert lr.read(tmp_path/'arm/commit.json')['update']==6001
    assert lr.read(tmp_path/'arm/phase.json')['status']=='failed'


def test_orphan_checkpoint_does_not_advance_persisted_count(tmp_path,monkeypatch):
    c,p=tiny_prepared();real=lr.atomic_json;root=tmp_path/'arm'
    def fail(path,value):
        if Path(path)==root/'commit.json' and value['update']==6001:raise OSError('receipt publication failure')
        return real(path,value)
    monkeypatch.setattr(lr,'atomic_json',fail)
    out=lr._run_arm(c,'constant',p,root,lr.time.perf_counter()+120,probe_runner=tiny_probe)
    assert not out['execution_valid'] and out['additional_updates']==1
    assert out['persisted_additional_updates']==0
    assert lr.read(root/'commit.json')['update']==6000
    assert (root/'snapshots/6001.msgpack').exists()


def test_probe_rejects_prediction_preserving_wrong_count(tmp_path):
    from force_consistency.runs import load_checkpoint,save_checkpoint
    c,p=tiny_prepared();root=tmp_path/'arm'
    out=lr._run_arm(c,'constant',p,root,lr.time.perf_counter()+120,probe_runner=tiny_probe)
    assert out['execution_valid']
    cp=root/'snapshots/6003.msgpack';prov=lr.read(root/'provenance.json')
    s=load_checkpoint(cp,prov,template=p['state']);s['optimizer']=(s['optimizer'][0]._replace(count=np.asarray(6002,dtype=np.int32)),*s['optimizer'][1:])
    save_checkpoint(cp,s,prov)
    with pytest.raises(ValueError):lr._probe_saved(root)


def test_parent_ancestry_and_predictions_restore():
    p=lr.load_parent(config())
    assert int(p['state']['update'])==int(p['state']['optimizer'][0].count)==6000
    assert len(p['fitting'].record_ids)==128 and len(p['anchor'].record_ids)==16
    assert p['source_prediction_max_error']==0.
    assert p['parent_source_reproduction_frames']==p['parent_endpoint_scoring_frames']==144


def test_current_code_identity_guard_rejects_changed_code(tmp_path):
    p=tmp_path/'code.py';p.write_text('original')
    code={'code_files':{str(p):lr.digest(p)}};lr.verify_code(code)
    p.write_text('changed')
    with pytest.raises(ValueError):lr.verify_code(code)


@pytest.mark.parametrize('damage',['interior','oracle','summary','child','input'])
def test_saved_report_refuses_missing_or_disagreeing_evidence(tmp_path,damage):
    c,p=tiny_prepared();root=tmp_path/'arm'
    out=lr._run_arm(c,'constant',p,root,lr.time.perf_counter()+120,probe_runner=tiny_probe)
    assert out['execution_valid']
    assert lr._validated_arm(root,c,check_parent=False)['execution_valid']
    if damage=='interior':
        receipt=lr.read(root/'commit.json');hp=root/receipt['history'];history=lr.read(hp);history.pop(1);lr.atomic_json(hp,history)
        receipt['history_sha256']=lr.digest(hp);lr.atomic_json(root/'commit.json',receipt)
    if damage=='oracle':(root/'oracle-6001.json').unlink()
    if damage=='summary':
        s=lr.read(root/'summary.json');s['history'][-1]['metrics']['fitting'][lr.MSE_KEYS[1]]*=2;lr.atomic_json(root/'summary.json',s)
    if damage=='child':
        v=lr.read(root/'checkpoint-verification.json');v['binding']['config_identity']='other';lr.atomic_json(root/'checkpoint-verification.json',v)
    if damage=='input':(root/'fitting-inputs.npz').write_bytes(b'changed')
    with pytest.raises((ValueError,FileNotFoundError)):lr._validated_arm(root,c,check_parent=False)


def test_endpoint_contrast_orientation_and_zero_denominator():
    result=lr.metric_contrast(4.,3.)
    assert result=={'late_decay_minus_constant':-1.,'late_decay_divided_by_constant':.75,'ratio_unavailable_reason':None}
    assert lr.metric_contrast(0.,1.)['ratio_unavailable_reason']=='constant denominator is zero'


def test_saved_report_recognizes_valid_execution_with_failed_learning(tmp_path):
    c,p=tiny_prepared();root=tmp_path/'arm'
    out=lr._run_arm(c,'constant',p,root,lr.time.perf_counter()+120,probe_runner=tiny_probe)
    assert out['execution_valid']
    report=lr._validated_arm(root,c,check_parent=False)
    assert report['execution_valid'] and report['additional_updates']==3
    assert not report['learning_accepted']


def test_partial_report_preserves_known_computed_and_persisted_work(tmp_path):
    c=config();root=tmp_path/'matrix';root.mkdir();lr.atomic_json(root/'config.json',c)
    lr.atomic_json(root/'code.json',{'code_files':{}})
    for arm in c['arms']:
        p=root/arm;p.mkdir();lr.atomic_json(p/'summary.json',{'status':'failed','execution_valid':False,
            'additional_updates':2,'additional_frame_exposures':256,'persisted_additional_updates':1,
            'attempted_additional_updates':3,'pending_update_unknown':True})
    lr.atomic_json(root/'raw-artifacts.json',{str(p.relative_to(root)):lr.digest(p) for p in root.rglob('*') if p.is_file()})
    r=lr.build_report(root)
    assert not r['execution_valid'] and not r['contrasts_available']
    assert r['additional_updates']==4 and r['additional_frame_exposures']==512
    assert r['reported_persisted_additional_updates']==2 and r['any_pending_work_unknown']


def test_saved_constant_arm_cannot_be_used_as_decay_control(tmp_path):
    c,p=tiny_prepared();root=tmp_path/'arm'
    assert lr._run_arm(c,'constant',p,root,lr.time.perf_counter()+120,probe_runner=tiny_probe)['execution_valid']
    with pytest.raises(ValueError):lr._validated_arm(root,c,expected_arm='late_decay',check_parent=False)


@pytest.mark.parametrize('damage',['checkpoint','baseline','inputs','start'])
def test_saved_parent_binding_rejects_consistent_wrong_ancestry(damage):
    c=config();root=Path(c['parent_directory']);pp=lr.read(root/'provenance.json');receipt=lr.read(root/'commit.json')
    p={**{k:pp[k] for k in ('model','normalization','inputs','original_baselines')},
       'source_pins':lr.read(c['source_pins']),'initial_state_hashes':receipt['state_hashes'],
       'parent_checkpoint_sha256':receipt['checkpoint_sha256'],'parent_provenance_sha256':lr.digest(root/'provenance.json')}
    lr._verify_parent_binding(c,p)
    if damage=='checkpoint':p['parent_checkpoint_sha256']='0'*64
    if damage=='baseline':p['original_baselines']['fitting'][lr.MSE_KEYS[1]]*=2
    if damage=='inputs':p['inputs']['fitting']['record_ids'][0]='other'
    if damage=='start':p['initial_state_hashes']['params']='0'*64
    with pytest.raises(ValueError):lr._verify_parent_binding(c,p)


def test_reporting_overrun_invalidates_persisted_scientific_contrast(tmp_path,monkeypatch):
    c=config()
    monkeypatch.chdir(tmp_path);Path('reports').mkdir();root=tmp_path/'run';root.mkdir()
    r={'execution_valid':True,'contrasts_available':True,'endpoint_contrasts':{'fake':'contrast'},'verification_errors':[]}
    lr.atomic_json('reports/late-rate-diagnostics.json',r);Path('reports/late-rate-diagnostics.md').write_text('provisional')
    monkeypatch.setattr(lr.time,'perf_counter',lambda:6001.)
    closed=lr._finalize(c,root,0.,r)
    assert not closed['execution_valid'] and not closed['contrasts_available']
    assert lr.read('reports/late-rate-diagnostics.json')['endpoint_contrasts'] is None
    assert not lr.read(root/'finalization.json')['within_global_cap']


@pytest.mark.parametrize('damage',['nan_reference','contract_arm'])
def test_child_rejects_nonfinite_reference_or_wrong_contract_arm(tmp_path,damage):
    c,p=tiny_prepared();root=tmp_path/'arm'
    assert lr._run_arm(c,'constant',p,root,lr.time.perf_counter()+120,probe_runner=tiny_probe)['execution_valid']
    contract=lr.read(root/'checkpoint-contract.json')
    if damage=='nan_reference':
        with np.load(root/'reference.npz') as ref:values={k:np.array(ref[k]) for k in ref.files}
        values['fitting_forces'][0,0,0]=np.nan;np.savez(root/'reference.npz',**values)
        contract['reference_sha256']=lr.digest(root/'reference.npz')
    else:contract['arm']='late_decay'
    lr.atomic_json(root/'checkpoint-contract.json',contract)
    with pytest.raises((ValueError,FloatingPointError)):lr._probe_saved(root)


def test_valid_finalization_closes_after_report_and_evidence(tmp_path,monkeypatch):
    c=config();monkeypatch.chdir(tmp_path);Path('reports').mkdir();root=tmp_path/'run';root.mkdir()
    r={'execution_valid':True,'contrasts_available':True,'endpoint_contrasts':{'saved':'contrast'},'verification_errors':[]}
    lr.atomic_json('reports/late-rate-diagnostics.json',r);Path('reports/late-rate-diagnostics.md').write_text('verified report')
    monkeypatch.setattr(lr.time,'perf_counter',lambda:100.)
    assert lr._finalize(c,root,0.,r)['execution_valid']
    closed=lr.read(root/'finalization.json')
    assert closed['within_global_cap'] and closed['elapsed_seconds']==100.
    assert closed['report_sha256']==lr.digest('reports/late-rate-diagnostics.json')
    assert closed['evidence_prefix_bytes']>0


def test_inconsistent_executable_fingerprint_is_rejected(tmp_path):
    p=tmp_path/'code';p.write_text('same recorded code')
    with pytest.raises(ValueError):lr.verify_code({'code_files':{str(p):lr.digest(p)},'code_fingerprint':'0'*64})


def test_complete_report_reconciles_prediction_work_categories(tmp_path):
    c,p=tiny_prepared();root=tmp_path/'arm'
    assert lr._run_arm(c,'constant',p,root,lr.time.perf_counter()+120,probe_runner=tiny_probe)['execution_valid']
    r=lr._validated_arm(root,c,check_parent=False)
    assert r['work']['scoring_prediction_frames']==56
    assert r['work']['reference_prediction_frames']==14
    assert r['child_work']['prediction_frames']==14
    s=lr.read(root/'summary.json');s['scoring_prediction_frames']=0;lr.atomic_json(root/'summary.json',s)
    with pytest.raises(ValueError):lr._validated_arm(root,c,check_parent=False)
