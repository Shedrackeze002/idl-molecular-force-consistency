"""Transactional persistence, independent mixed-derivative Adam and failure routing."""
import copy
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import jax
import numpy as np
import pytest
from force_consistency import width_rate_runs as wr
from force_consistency.width_rate_contract import resolve_commit,digest
from force_consistency.coverage_runs import _state_hashes,coverage_metrics
from force_consistency.metrics import partition_batch
from force_consistency.loss_balance import build_weighted_step
from force_consistency.models.energy import energy_and_forces
from test_coverage import seven_case,close_trees
from test_width_rate_preflight import config


def fixture():
    p,m,b,n,s=seven_case();c=config();c.update(max_updates=3,score_updates=[0,1,2,3],microbatch_size=3,model=asdict(m.config))
    case={'name':'tiny','features':4,'learning_rate':.001,'wall_seconds':120.}
    base=coverage_metrics(s['params'],p,p,m,3)
    return p,m,b,n,s,c,case,base

@pytest.mark.parametrize('failure',['checkpoint','history','receipt'])
def test_transaction_interruption_preserves_matched_previous_receipt(tmp_path,monkeypatch,failure):
    p,m,b,n,s,c,case,base=fixture();wr.commit_state(tmp_path,s,[{'update':0}],{},n)
    before=(tmp_path/'commit.json').read_bytes();receipt,_=resolve_commit(tmp_path)
    old={key:digest(tmp_path/receipt[key]) for key in ('checkpoint','history')}
    step=build_weighted_step(m,n,.001,c['loss_weights']);pp,oo,_=step(s['params'],s['optimizer'],[b]);jax.block_until_ready((pp,oo));nextstate={**s,'params':pp,'optimizer':oo,'update':1}
    target='save_checkpoint' if failure=='checkpoint' else 'atomic_json';real=getattr(wr,target)
    def fail(path,*a,**kw):
        selected=Path(path).suffix=='.msgpack' if failure=='checkpoint' else (Path(path).name=='commit.json' if failure=='receipt' else Path(path).parent.name=='histories')
        if selected:raise OSError('injected write failure')
        return real(path,*a,**kw)
    monkeypatch.setattr(wr,target,fail)
    with pytest.raises(OSError):wr.commit_state(tmp_path,nextstate,[{'update':0},{'update':1}],{},n)
    assert (tmp_path/'commit.json').read_bytes()==before
    actual,history=resolve_commit(tmp_path)
    assert actual['update']==history[-1]['update']==0
    assert all(digest(tmp_path/actual[k])==old[k] for k in old)


def test_committed_versions_cannot_be_overwritten(tmp_path):
    p,m,b,n,s,c,case,base=fixture();wr.commit_state(tmp_path,s,[{'update':0}],{},n)
    receipt,_=resolve_commit(tmp_path);before=(tmp_path/receipt['history']).read_bytes()
    with pytest.raises(ValueError):wr.commit_state(tmp_path,s,[{'update':0,'fabricated':True}],{},n)
    assert (tmp_path/receipt['history']).read_bytes()==before


def test_independent_numpy_adam_matches_mixed_force_energy_multistep():
    p,m,b,n,state,c,case,base=fixture();chunks=[partition_batch(p,np.arange(i,min(i+3,7))) for i in (0,3,6)]
    step=build_weighted_step(m,n,.001,c['loss_weights']);params=state['params'];opt=state['optimizer']
    weights=c['loss_weights']
    def loss(pp):
        e,f=energy_and_forces(pp,b,model=m)
        return weights['energy']*jax.numpy.mean((e-b['energy'])**2)/n.energy_loss_scale_eV**2+weights['force']*jax.numpy.mean((f-b['forces'])**2)/n.force_loss_scale_eV_per_A**2
    grad=jax.jit(jax.grad(loss))
    for count in (1,2,3):
        gradient=grad(params);rp,ro=wr.independent_adam_step(params,opt,gradient,.001)
        ap,ao,_=step(params,opt,chunks);jax.block_until_ready((ap,ao))
        close_trees(ap,rp);close_trees(ao,ro)
        assert int(ao[0].count)==count
        params,opt=ap,ao

@pytest.mark.parametrize('features',[16,32])
@pytest.mark.parametrize('rate',[.001,.0003])
def test_approved_width_rate_first_step_oracle(features,rate):
    p,m,b,n,s,c,case,base=fixture()
    from force_consistency.models.backbone import ModelConfig
    from force_consistency.models.energy import EnergyModel
    from force_consistency.train import initialize_state
    model=EnergyModel(ModelConfig(**{**config()['model'],'features':features}));state=initialize_state(model,b,n,0,rate)
    chunks=[partition_batch(p,np.arange(i,min(i+3,7))) for i in (0,3,6)]
    before=_state_hashes(state)
    oracle=wr.first_step_oracle(state,chunks,b,model,n,rate,c['loss_weights'],wr.ExecutionContext(120.))
    assert oracle['receipt']['status']=='passed' and oracle['receipt']['fullbatch_reference_frames']==7
    assert int(oracle['state']['optimizer'][0].count)==1
    assert _state_hashes(state)==before


def test_matrix_learning_failure_continues_but_execution_failure_blocks(tmp_path):
    c=config();calls=[]
    def runner(case):
        calls.append(case['name']);return {'status':'completed','executed':True,'execution_valid':True,'fit_gate':{'accepted':False},'anchor_gate':{'accepted':False}}
    out=wr.execute_width_rate_matrix(tmp_path/'learning',c,runner,lambda:None,wr.ExecutionContext(120.))
    assert len(calls)==4 and out['execution_valid'] and out['full_pilot']=='pending'
    calls.clear()
    def failed(case):calls.append(case['name']);return {'status':'failed','executed':True,'execution_valid':False}
    out=wr.execute_width_rate_matrix(tmp_path/'failed',c,failed,lambda:None,wr.ExecutionContext(120.))
    assert len(calls)==1 and not out['execution_valid']
    assert [a['status'] for a in out['arms'].values()]==['failed','blocked','blocked','blocked']


def test_matrix_source_failure_and_summary_write_failure_retained(tmp_path,monkeypatch):
    c=config();calls=[]
    def verify():raise ValueError('source changed')
    out=wr.execute_width_rate_matrix(tmp_path/'source',c,lambda case:calls.append(case),verify,wr.ExecutionContext(120.))
    assert not calls and all(x['status']=='blocked' for x in out['arms'].values())
    real=wr.atomic_json
    def badwrite(path,value):
        if Path(path).name=='summary.json':raise OSError('summary unavailable')
        return real(path,value)
    monkeypatch.setattr(wr,'atomic_json',badwrite)
    out=wr.execute_width_rate_matrix(tmp_path/'write',c,lambda case:{'status':'completed','executed':True,'execution_valid':True},lambda:None,wr.ExecutionContext(120.))
    assert not out['execution_valid'] and (tmp_path/'write/controller-failure.json').exists()


def tiny_probe(root,context):
    context.check_budget()
    subprocess.run([sys.executable,'-c','from force_consistency.width_rate_runs import _probe_saved; _probe_saved('+repr(str(root.resolve()))+')'],check=True,capture_output=True,timeout=context.remaining_seconds())
    return json.loads((root/'checkpoint-verification.json').read_text())


def test_tiny_fresh_runner_transaction_and_exhaustive_probe(tmp_path):
    p,m,b,n,s,c,case,base=fixture();before=_state_hashes(s)
    out=wr._run_arm(c,case,s,p,p,m,n,base,{},tmp_path/'tiny',{'engineering_fixture':True},wr.ExecutionContext(120.).for_arm(120.,.25),probe_runner=tiny_probe)
    assert out['execution_valid'],out.get('failure_reason')
    assert (out['updates'],out['attempted_updates'],out['persisted_update'],out['frame_exposures'])==(3,3,3,21)
    assert out['warmup_frame_exposures']==7 and out['warmup_source_unchanged']
    assert [r['update'] for r in out['history']]==[0,1,2,3]
    receipt,history=resolve_commit(tmp_path/'tiny');assert receipt['update']==3 and history==out['history']
    assert out['charged_elapsed_seconds']==pytest.approx(out['elapsed_seconds']+.25)
    assert _state_hashes(s)==before and out['checkpoint_verification']['fitting_record_count']==7
    with pytest.raises(ValueError):wr._run_arm(c,case,s,p,p,m,n,base,{},tmp_path/'tiny',{},wr.ExecutionContext(120.))
    with pytest.raises(ValueError):wr.run_width_rate_arm(c,case,s,p,p,m,n,base,{},tmp_path/'bad',{},wr.ExecutionContext(120.))

@pytest.mark.parametrize('damage',['count','normalization','features','dtype','checkpoint_sha'])
def test_probe_rejects_prediction_preserving_state_corruption(tmp_path,damage):
    p,m,b,n,s,c,case,base=fixture();root=tmp_path/'tiny'
    out=wr._run_arm(c,case,s,p,p,m,n,base,{},root,{},wr.ExecutionContext(120.).for_arm(120.,0.),probe_runner=tiny_probe)
    assert out['execution_valid']
    from force_consistency.runs import load_checkpoint,save_checkpoint
    receipt,h=resolve_commit(root);state=load_checkpoint(root/receipt['checkpoint'],out['provenance'],template=s)
    if damage=='count':state['optimizer']=(state['optimizer'][0]._replace(count=state['optimizer'][0].count-1),*state['optimizer'][1:])
    elif damage=='normalization':state['normalization_json']='{}'
    elif damage=='dtype':state['params']=jax.tree.map(lambda x:x.astype('float32'),state['params'])
    elif damage=='features':
        saved=json.loads((root/'case.json').read_text());saved['features']=8;(root/'case.json').write_text(json.dumps(saved))
    else:receipt['checkpoint_sha256']='0'*64
    if damage in ('count','normalization'):
        save_checkpoint(root/receipt['checkpoint'],state,out['provenance']);receipt['checkpoint_sha256']=digest(root/receipt['checkpoint'])
    if damage=='dtype':
        # Checkpoint writer itself rejects low precision.
        with pytest.raises(ValueError):save_checkpoint(root/receipt['checkpoint'],state,out['provenance'])
        return
    (root/'commit.json').write_text(json.dumps(receipt))
    with pytest.raises((ValueError,KeyError)):wr._probe_saved(root)

@pytest.mark.parametrize('kind',['interrupt','nonfinite','replay','timeout','global_cap'])
def test_arm_failure_retains_partial_truth(tmp_path,monkeypatch,kind):
    p,m,b,n,s,c,case,base=fixture();historical={};now=[0.]
    context=wr.ExecutionContext(120.,clock=lambda:now[0]).for_arm(120.,0.)
    real=wr.build_weighted_step
    if kind in ('interrupt','nonfinite','global_cap'):
        def factory(*a):
            step=real(*a);calls=0
            def wrapped(*args):
                nonlocal calls
                calls+=1
                if calls==4:
                    if kind=='interrupt':raise KeyboardInterrupt('pending update')
                    if kind=='global_cap':now[0]=130.
                    pp,oo,mm=step(*args)
                    if kind=='nonfinite':pp=jax.tree.map(lambda x:x*np.nan,pp)
                    return pp,oo,mm
                return step(*args)
            wrapped.prepare=step.prepare;return wrapped
        monkeypatch.setattr(wr,'build_weighted_step',factory)
    if kind=='replay':historical={0:copy.deepcopy(base)};historical[0]['fitting']['force_mse_eV2_per_A2']*=2
    def badprobe(root,context):raise subprocess.TimeoutExpired('probe',.01)
    out=wr._run_arm(c,case,s,p,p,m,n,base,historical,tmp_path/kind,{},context,probe_runner=badprobe if kind=='timeout' else tiny_probe)
    assert not out['execution_valid'] and out['status'] in ('failed','incomplete')
    assert out['frame_exposures']==out['updates']*7 and out['updates']<=3
    receipt,h=resolve_commit(tmp_path/kind);assert receipt['update']==h[-1]['update']==out['persisted_update']
    if kind in ('interrupt','nonfinite'):assert out['pending_update_unknown'] and out['attempted_updates']>out['updates']


def test_controller_preserves_completed_work_when_post_arm_source_check_fails(tmp_path):
    c=config();calls=[0]
    def verify():
        calls[0]+=1
        if calls[0]==2:raise ValueError('source mutated after completed work')
    out=wr.execute_width_rate_matrix(tmp_path,c,lambda case:{'status':'completed','executed':True,'execution_valid':True,'updates':6000,'frame_exposures':768000},verify,wr.ExecutionContext(120.))
    first=out['arms']['width16-lr1e-3']
    assert first['updates']==6000 and first['frame_exposures']==768000 and not first['execution_valid']
    assert all(a['status']=='blocked' for a in list(out['arms'].values())[1:])


def test_probe_timeout_uses_minimum_remaining_allowance_and_never_launches_after_cap(tmp_path,monkeypatch):
    now=[0.];global_context=wr.ExecutionContext(10.,clock=lambda:now[0]);now[0]=2.;context=global_context.for_arm(6.,1.)
    seen=[]
    def proc(*a,**kw):seen.append(kw['timeout']);(tmp_path/'checkpoint-verification.json').write_text('{}')
    monkeypatch.setattr(wr.subprocess,'run',proc)
    wr._launch_probe(tmp_path,context);assert seen==[5.]
    now[0]=7.
    with pytest.raises(wr.WallBudget):wr._launch_probe(tmp_path,context)
    assert seen==[5.]


def test_cli_dispatches_frozen_width_rate_configuration(monkeypatch):
    from force_consistency.cli import main
    monkeypatch.setattr(wr,'diagnose_width_rate',lambda c:{'status':'completed','execution_valid':True,'full_pilot':'pending'})
    assert main(['diagnose-width-rate','--config','configs/aspirin-width-rate-diagnostic.json'])==0


def test_oracle_receipt_bounds_are_independently_checkable():
    p,m,b,n,s,c,case,base=fixture();batches=[partition_batch(p,np.arange(i,min(i+3,7))) for i in (0,3,6)]
    oracle=wr.first_step_oracle(s,batches,b,m,n,.001,c['loss_weights'],wr.ExecutionContext(120.))
    assert 0<=oracle['receipt']['state_maximum_tolerance_fraction']<=1
    assert 0<=oracle['receipt']['prediction_maximum_tolerance_fraction']<=1


def test_nested_integer_history_provenance_roundtrips_in_fresh_process(tmp_path):
    p,m,b,n,s,c,case,base=fixture()
    provenance={'historical':{0:base,100:base,200:base,1000:base,2000:base,2001:base},'source':'engineering-only'}
    out=wr._run_arm(c,case,s,p,p,m,n,base,{},tmp_path/'tiny',provenance,wr.ExecutionContext(120.).for_arm(120.,0.),probe_runner=tiny_probe)
    assert out['execution_valid'],out.get('failure_reason')
    saved=json.loads((tmp_path/'tiny/provenance.json').read_text())
    assert set(saved['historical'])=={'0','100','200','1000','2000','2001'}


def test_failed_probe_preserves_known_reference_work_and_unknown_verification_work(tmp_path):
    p,m,b,n,s,c,case,base=fixture()
    def failed(root,context):raise subprocess.CalledProcessError(1,'probe',stderr=b'checkpoint metadata failure')
    out=wr._run_arm(c,case,s,p,p,m,n,base,{},tmp_path/'tiny',{},wr.ExecutionContext(120.).for_arm(120.,0.),probe_runner=failed)
    assert not out['execution_valid'] and out['updates']==out['persisted_update']==3
    assert out['reference_prediction_frames']==14
    assert out['probe_work_status']=='unknown' and out['probe_seconds']>=0
