import copy
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import subprocess
import jax
import numpy as np
import pytest
from force_consistency import convergence_runs as cr
from force_consistency.coverage_runs import _state_hashes, coverage_metrics, MSE_KEYS
from force_consistency.diagnostics import sha256
from force_consistency.metrics import partition_batch
from force_consistency.loss_balance import build_weighted_step
from force_consistency.runs import load_checkpoint, save_checkpoint
from test_coverage import seven_case, close_trees, arm_config


def config():return json.loads(Path('configs/aspirin-convergence-budget.json').read_text())

@pytest.mark.parametrize('key,value',[('end_update',6001),('start_update',0),('additional_updates',3999),('continuation_wall_seconds',1201),('total_wall_seconds',3901),('terminal_window_updates',100),('small_window_relative_change',.1),('learning_rate',.002),('early_stopping',True)])
def test_frozen_settings_fail_before_arrays(monkeypatch,key,value):
    c=config();c[key]=value
    monkeypatch.setattr(cr,'optimizer_preflight',lambda *a:pytest.fail('parent loader reached'))
    with pytest.raises(ValueError):cr.convergence_preflight(c)


def test_omitted_stale_and_rehashed_bad_optimizer_controls_before_arrays(tmp_path,monkeypatch):
    monkeypatch.setattr(cr,'optimizer_preflight',lambda *a:pytest.fail('parent loader reached'))
    c=config();c['optimizer_diagnostic_sha256'].pop('cases.json')
    with pytest.raises(ValueError):cr.convergence_preflight(c)
    c=config();c['optimizer_diagnostic_compact_sha256']='0'*64
    with pytest.raises(ValueError):cr.convergence_preflight(c)
    c=config();old=Path(c['optimizer_diagnostic_run']);new=tmp_path/'source';shutil.copytree(old,new);c['optimizer_diagnostic_run']=str(new)
    s=json.loads((new/'summary.json').read_text());s['execution_valid']=False;(new/'summary.json').write_text(json.dumps(s));c['optimizer_diagnostic_sha256']['summary.json']=sha256(new/'summary.json')
    with pytest.raises(ValueError):cr.convergence_preflight(c)


def test_real_preflight_original_baselines_counts_oracles_and_source_identity():
    c=config();fit,anchor,model,norm,states,sources,records=cr.convergence_preflight(c)
    assert (len(fit.record_ids),len(anchor.record_ids))==(128,16)
    for name,state in states.items():
        assert int(state['update'])==int(state['optimizer'][0].count)==2000
        assert _state_hashes(state)==records[name]['state_hashes_before']
        assert sources['arms'][name]['initial_metrics']==sources['historical_reference']['arms']['diverse128']['summary']['initial_metrics']
    cr.verify_convergence_sources(c)


def fixture():
    p,m,b,n,s=seven_case();c={**arm_config(n),'start_update':2,'end_update':5,'additional_updates':3,'continuation_wall_seconds':120.,'loss_weights':{'energy':.25,'force':1.75},'arm':'balanced'}
    baseline=coverage_metrics(s['params'],p,p,m,3)
    import optax
    from force_consistency.models.energy import energy_and_forces
    tx=optax.adam(.001)
    def loss(params):
        e,f=energy_and_forces(params,b,model=m)
        return .25*jax.numpy.mean((e-b['energy'])**2)/.2**2+1.75*jax.numpy.mean((f-b['forces'])**2)/.1**2
    valuegrad=jax.jit(jax.value_and_grad(loss))
    def step(params,opt,batches):
        value,gradient=valuegrad(params);delta,nextopt=tx.update(gradient,opt,params)
        return optax.apply_updates(params,delta),nextopt,{'scaled_loss':value}
    chunks=[b]
    states=[copy.deepcopy(s)]
    for _ in range(5):
        pp,oo,_=step(s['params'],s['optimizer'],chunks);jax.block_until_ready((pp,oo));s={**s,'params':pp,'optimizer':oo,'update':int(s['update'])+1};states.append(s)
    oracle=coverage_metrics(states[3]['params'],p,p,m,3)
    return p,m,n,c,states[2],states[5],baseline,oracle


def test_saved_moment_continuation_matches_uninterrupted_without_rebase(tmp_path):
    p,m,n,c,s,wanted,base,oracle=fixture();before=_state_hashes(s)
    out=cr.run_convergence_arm(c,s,p,p,m,n,base,oracle,tmp_path/'arm',{'engineering_fixture':True})
    assert out['execution_valid'],out.get('failure_reason')
    assert (out['updates'],out['additional_updates'],out['additional_frame_exposures'],out['frame_exposures'])==(5,3,21,35)
    assert out['warmup_frame_exposures']==7 and out['warmup_source_unchanged']
    assert [x['update'] for x in out['history']]==[2,3,4,5]
    assert out['original_baselines']==base and out['oracle_verified']
    got=load_checkpoint(tmp_path/'arm/last.msgpack',out['provenance'],template=s)
    close_trees(got['params'],wanted['params']);close_trees(got['optimizer'],wanted['optimizer'])
    assert _state_hashes(s)==before
    v=json.loads((tmp_path/'arm/checkpoint-verification.json').read_text());assert v['adam_count']==5 and v['checkpoint_sha256']==sha256(tmp_path/'arm/last.msgpack')
    with pytest.raises(ValueError,match='not empty'):cr.run_convergence_arm(c,s,p,p,m,n,base,oracle,tmp_path/'arm',{})

@pytest.mark.parametrize('damage',['count','normalization'])
def test_fresh_process_rejects_prediction_preserving_checkpoint_corruption(tmp_path,damage):
    p,m,n,c,s,wanted,base,oracle=fixture();root=tmp_path/'arm'
    out=cr.run_convergence_arm(c,s,p,p,m,n,base,oracle,root,{})
    assert out['execution_valid']
    changed=load_checkpoint(root/'last.msgpack',out['provenance'],template=s)
    if damage=='count':changed['optimizer']=(changed['optimizer'][0]._replace(count=changed['optimizer'][0].count-1),*changed['optimizer'][1:])
    else:changed['normalization_json']=json.dumps({**asdict(n),'energy_offset_eV':123.})
    save_checkpoint(root/'last.msgpack',changed,out['provenance'])
    proc=subprocess.run([str(Path('.venv/bin/python').resolve()),'-m','force_consistency.convergence_runs','--probe',str(root)],capture_output=True)
    assert proc.returncode!=0

@pytest.mark.parametrize('kind',['interrupt','badcount','nonfinite','oracle','probe','history_write','checkpoint_write','wall'])
def test_partial_failure_retains_verified_state_and_counters(tmp_path,monkeypatch,kind):
    p,m,n,c,s,wanted,base,oracle=fixture();root=tmp_path/kind
    if kind=='oracle':oracle=copy.deepcopy(oracle);oracle['fitting'][MSE_KEYS[1]]*=2
    if kind=='wall':c['continuation_wall_seconds']=.00001
    real_factory=cr.build_weighted_step
    if kind in ('interrupt','badcount','nonfinite'):
        def factory(*a):
            step=real_factory(*a);calls=0
            def wrapped(*args):
                nonlocal calls
                calls+=1
                if calls==3:
                    if kind=='interrupt':raise KeyboardInterrupt('interrupted in pending update')
                    pp,oo,mm=step(*args)
                    if kind=='badcount':return pp,args[1],mm
                    return jax.tree.map(lambda x:x*np.nan,pp),oo,mm
                return step(*args)
            return wrapped
        monkeypatch.setattr(cr,'build_weighted_step',factory)
    if kind=='probe':monkeypatch.setattr(cr.subprocess,'run',lambda *a,**k:(_ for _ in ()).throw(ValueError('probe failure')))
    if kind in ('history_write','checkpoint_write'):
        target='atomic_json' if kind=='history_write' else 'save_checkpoint';real=getattr(cr,target);calls=0
        def fail(path,*a,**k):
            nonlocal calls
            selected=Path(path).name=='history.json' if kind=='history_write' else Path(path).suffix=='.msgpack'
            if selected:
                calls+=1
                if calls==2:raise OSError('injected persistence failure')
            return real(path,*a,**k)
        monkeypatch.setattr(cr,target,fail)
    out=cr.run_convergence_arm(c,s,p,p,m,n,base,oracle,root,{})
    assert not out['execution_valid'] and out['status'] in ('failed','incomplete')
    assert out['additional_frame_exposures']==out['additional_updates']*7
    assert out['frame_exposures']==out['updates']*7
    if kind in ('interrupt','badcount','nonfinite'):
        assert out['updates']==3 and out['attempted_updates']==2 and out['pending_update_unknown']
    if (root/'snapshots/2.msgpack').exists():
        keep=load_checkpoint(root/'snapshots/2.msgpack',out['provenance'],template=s);assert int(keep['update'])==2
    assert _state_hashes(s)==out['source_state_hashes_before']==out['source_state_hashes_after']


def test_controller_interrupt_blocks_later_arms_and_keeps_failure(tmp_path):
    calls=[]
    def runner(name,path):
        calls.append(name);raise KeyboardInterrupt('test interrupt')
    out=cr.execute_convergence_matrix(tmp_path,runner)
    assert calls==['replay']
    assert out['replay']['status']=='failed' and not out['replay']['execution_valid']
    assert all(out[n]['status']=='blocked' and out[n]['additional_frame_exposures']==0 for n in ('balanced','force-only'))
    assert (tmp_path/'replay/controller-failure.json').exists()


def test_cli_execution_valid_is_distinct_from_learning(tmp_path,monkeypatch):
    from force_consistency.cli import main
    path=tmp_path/'c.json';path.write_text('{}')
    monkeypatch.setattr(cr,'diagnose_convergence',lambda c:{'status':'completed','execution_valid':True,'learning_accepted':False,'full_pilot':'pending'})
    assert main(['diagnose-convergence-budget','--config',str(path)])==0


def test_controller_summary_write_failure_is_independently_recorded_and_blocks(tmp_path,monkeypatch):
    real=cr.atomic_json
    def fail(path,value):
        if Path(path)==tmp_path/'replay/summary.json':raise OSError('summary device failure')
        return real(path,value)
    monkeypatch.setattr(cr,'atomic_json',fail)
    def runner(name,path):
        path.mkdir();return {'status':'completed','execution_valid':True,'additional_updates':1,'additional_frame_exposures':7}
    out=cr.execute_convergence_matrix(tmp_path,runner)
    assert not out['replay']['execution_valid']
    assert (tmp_path/'replay/controller-failure.json').exists()
    assert out['balanced']['status']==out['force-only']['status']=='blocked'
