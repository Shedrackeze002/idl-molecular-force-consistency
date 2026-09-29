from dataclasses import replace
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from force_consistency.data.prepare import Normalization
from force_consistency.models.energy import EnergyModel,make_batch,energy_and_forces
from force_consistency.models.backbone import ModelConfig
from force_consistency.train import initialize_state


def small_case():
    from test_training_contract import bundle
    p=bundle().train
    positions=p.positions[:4].copy()
    positions[:,1,0]+=np.arange(4)*.1
    positions[:,2,1]+=np.arange(4)*.03
    p=replace(p,positions=positions,energy_residual=p.energy_residual[:4],forces=p.forces[:4],atomic_numbers=p.atomic_numbers[:4],record_ids=p.record_ids[:4],memberships={r:p.memberships[r] for r in p.record_ids[:4]})
    model=EnergyModel(ModelConfig(features=4,iterations=1,radial_basis=4,energy_readout_init='zero'))
    b=make_batch(p.positions,p.atomic_numbers,p.energy_residual,p.forces)
    norm=Normalization(0.,1.,1.)
    return p,model,b,norm,initialize_state(model,b,norm,0,.001)


def test_train_only_loader_with_holdouts_absent(tmp_path,monkeypatch):
    from test_splits_prepare import fixture,empty_duplicates,UNIT_EVIDENCE
    from force_consistency.data.types import PreparationPolicy
    from force_consistency.data.prepare import export_prepared,load_training_partition
    rs,inv,ends=fixture()
    m=export_prepared(rs,directory=tmp_path,inventory=inv,endpoints=ends,duplicates=empty_duplicates(),policy=PreparationPolicy(),molecule='aspirin',unit_evidence=UNIT_EVIDENCE)
    for name in ('validation','test','validation_minima_diagnostics'):
        (tmp_path/f'{name}.npz').unlink()
    original=np.load
    def guarded(path,*args,**kw):
        assert Path(path).name=='train.npz'
        return original(path,*args,**kw)
    monkeypatch.setattr(np,'load',guarded)
    p,manifest,norm=load_training_partition(tmp_path,expected_split_hash=m['split_hash'])
    assert len(p.record_ids)==len(m['partitions']['train']['record_ids'])
    assert manifest==json.loads(json.dumps(m)) and not p.forces.flags.writeable
    with pytest.raises(ValueError,match='split'):
        load_training_partition(tmp_path,expected_split_hash='stale')


def test_subset_identity_membership_and_index_gates():
    from force_consistency.diagnostics import verified_subset
    from force_consistency.data.splits import canonical_hash
    p,_,_,_,_=small_case()
    m={'molecule':'synthetic','splits':{'groups':{'synthetic':{'train':['a','b']}}}}
    ids=list(p.record_ids);s={'record_ids':ids,'indices':list(range(4)),'identity':canonical_hash(ids)}
    assert verified_subset(p,m,s,s['identity']).record_ids==p.record_ids
    for changed in ({**s,'indices':[1,0,2,3]},{**s,'record_ids':ids[::-1]}, {**s,'record_ids':[ids[0]]*4}):
        with pytest.raises(ValueError): verified_subset(p,m,changed,s['identity'])
    bad=replace(p,memberships={r:{**p.memberships[r],'role':'minimum'} for r in ids})
    with pytest.raises(ValueError): verified_subset(bad,m,s,s['identity'])


def test_nonzero_head_design_sign_axes_and_atom_bias():
    from force_consistency.diagnostics import readout_design,with_head,head_coefficients,verify_design
    p,model,b,_,state=small_case();coef=np.array([.2,-.1,.03,.4,.07])
    params=with_head(state['params'],coef)
    x,g=readout_design(state['params'],p,model.config)
    assert x.shape==(4,5) and g.shape==(4,3,3,5)
    assert np.all(x[:,-1]==3) and np.all(g[...,-1]==0)
    e,f=energy_and_forces(params,b,model=model)
    np.testing.assert_allclose(x@coef,e,atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(g@coef,np.asarray(f).reshape(4,3,3),atol=1e-10,rtol=1e-10)
    np.testing.assert_array_equal(head_coefficients(params),coef)
    assert verify_design(x,g,params,p,model)['maximum_absolute_error']<1e-10
    with pytest.raises(ValueError,match='reconstruction'):
        verify_design(x,-g,params,p,model)


def test_weighted_joint_solve_and_force_only_bias():
    from force_consistency.diagnostics import solve_readout
    # 2 energy rows vs 6 force components: separate MSEs, not pooled residual SSE.
    x=np.array([[1.,3.],[2.,3.]])
    g=np.array([[1.,0.],[1.,0.],[1.,0.],[1.,0.],[1.,0.],[1.,0.]])
    energy=np.array([2.,4.]);force=np.zeros(6);norm=Normalization(0.,2.,.5)
    joint=solve_readout(x,g,energy,force,norm,mode='joint',rcond=1e-10)
    oracle=np.linalg.lstsq(np.vstack([x/(2*np.sqrt(2)),g/(.5*np.sqrt(6))]),np.r_[energy/(2*np.sqrt(2)),force/(.5*np.sqrt(6))],rcond=1e-10)[0]
    np.testing.assert_allclose(joint['coefficients'],oracle,atol=1e-12)
    forceonly=solve_readout(x,g,energy,force,norm,mode='force_only',rcond=1e-10)
    np.testing.assert_allclose(forceonly['coefficients'],[0.,1.],atol=1e-12)
    assert not np.allclose(joint['coefficients'],forceonly['coefficients'])


def control_config():
    return {'seed':0,'learning_rate':.001,'validation_interval':1,'max_updates':2,'wall_seconds':120.,'gate_fraction':.1,'backend':'cpu','model':{'features':4,'iterations':1,'radial_basis':4,'energy_readout_init':'zero'}}


def test_teacher_is_nonzero_representable_centered_and_readonly():
    from force_consistency.diagnostics import build_teacher,readout_design,head_coefficients
    p,model,b,norm,state=small_case();original=p.forces.copy()
    target,teacher=build_teacher(state['params'],p,model,23,.1)
    assert abs(target.energy_residual.mean())<1e-12
    assert np.mean(target.forces**2)>1e-12 and np.var(target.energy_residual)>1e-24
    x,g=readout_design(state['params'],p,model.config);c=head_coefficients(teacher)
    np.testing.assert_allclose(x@c,target.energy_residual,atol=1e-10)
    np.testing.assert_allclose(g@c,target.forces,atol=1e-10)
    np.testing.assert_array_equal(p.forces,original)
    assert not target.forces.flags.writeable
    with pytest.raises(ValueError,match='Degenerate'): build_teacher(state['params'],p,model,23,0.)


def test_control_cap_history_checkpoint_probe_without_data_files(tmp_path):
    from force_consistency.diagnostics import run_control
    p,model,b,norm,state=small_case()
    result=run_control(control_config(),p,model,norm,state,tmp_path/'run',{'test':'control'},'real_DFT')
    assert result['status']=='incomplete' and result['updates']==2 and result['stop_reason']=='max_updates'
    assert [row['update'] for row in result['history']]==[0,1,2]
    assert result['checkpoint_new_process_verified'] and result['batch_size']==4
    assert result['provenance']['input_arrays']['positions']['shape']==[4,3,3]
    with pytest.raises(ValueError,match='not empty'):
        run_control(control_config(),p,model,norm,state,tmp_path/'run',{},'real_DFT')


def test_nonfinite_update_retains_failure_history(tmp_path,monkeypatch):
    import force_consistency.diagnostics as d
    p,model,b,norm,state=small_case()
    @jax.jit
    def bad(params,opt,batch):
        return jax.tree.map(lambda x:x*jnp.nan,params),opt,{'scaled_loss':jnp.array(np.nan)}
    monkeypatch.setattr(d,'build_step',lambda *args:bad)
    result=d.run_control(control_config(),p,model,norm,state,tmp_path/'run',{},'synthetic_teacher')
    assert result['status']=='failed' and result['updates']==0 and 'Nonfinite' in result['failure_reason']
    assert json.loads((tmp_path/'run/summary.json').read_text())['status']=='failed'
    assert json.loads((tmp_path/'run/history.json').read_text())[0]['update']==0


def test_teacher_failure_blocks_real_without_executing_it(tmp_path):
    from force_consistency.diagnostics import run_after_teacher
    called=[]
    result=run_after_teacher({'accepted':False,'status':'incomplete'},lambda:called.append(True),tmp_path/'real')
    assert called==[] and result['status']=='blocked' and result['updates']==0
    assert json.loads((tmp_path/'real/summary.json').read_text())==result


def test_source_hash_preflight_failure_saved_and_no_training(tmp_path):
    from force_consistency.diagnostics import diagnose
    config={**control_config(),'run_directory':str(tmp_path/'failed'),'source_run':str(tmp_path/'missing'),
            'source_hashes':{'last.msgpack':'incorrect'}}
    result=diagnose(config)
    assert result['status']=='failed' and result['teacher']['status']=='blocked' and result['real']['status']=='blocked'
    assert (tmp_path/'failed/summary.json').exists()
    with pytest.raises(ValueError,match='not empty'): diagnose(config)


def test_historical_state_settings_reject_stale_norm_model_update():
    from dataclasses import asdict
    from force_consistency.diagnostics import verify_source_settings
    p,model,b,norm,state=small_case();c={**control_config(),'expected_split_hash':'split','expected_learning_subset_hash':'subset'}
    original={**c,'mode':'learning_check','batch_size':4}
    manifest={'identity':'data','split_hash':'split'}
    prov={'dataset_identity':'data','split_hash':'split','normalization':asdict(norm)}
    good={**state,'update':2000}
    verify_source_settings(c,original,prov,manifest,norm,good)
    for bad in ({**good,'update':1999},{**good,'normalization_json':'{}'}):
        with pytest.raises(ValueError): verify_source_settings(c,original,prov,manifest,norm,bad)
    with pytest.raises(ValueError): verify_source_settings(c,{**original,'model':{**c['model'],'cutoff_A':4}},prov,manifest,norm,good)
    with pytest.raises(ValueError): verify_source_settings(c,original,{**prov,'dataset_identity':'stale'},manifest,norm,good)


def test_historical_hash_mismatch_aborts_before_array_loading(tmp_path,monkeypatch):
    import force_consistency.diagnostics as d
    source=tmp_path/'source';source.mkdir()
    names=['last.msgpack','learning-subset.json','config.json','provenance.json']
    for n in names: (source/n).write_text('{}')
    hashes={n:d.sha256(source/n) for n in names};hashes['last.msgpack']='stale'
    monkeypatch.setattr(d,'load_training_partition',lambda *a,**kw:pytest.fail('opened arrays after stale source'))
    result=d.diagnose({**control_config(),'run_directory':str(tmp_path/'run'),'source_run':str(source),'source_hashes':hashes})
    assert result['status']=='failed' and 'source identity' in result['failure_reason']


def test_diagnostic_cli_integration_without_holdout_arrays(tmp_path):
    from dataclasses import asdict
    from test_splits_prepare import fixture,empty_duplicates,UNIT_EVIDENCE
    from force_consistency.data.types import PreparationPolicy
    from force_consistency.data.prepare import export_prepared,load_training_partition
    from force_consistency.data.splits import canonical_hash
    from force_consistency.diagnostics import sha256
    from force_consistency.runs import save_checkpoint
    from force_consistency.cli import main
    rs,inv,ends=fixture()
    # Vary geometry for a nondegenerate synthetic teacher; no scientific labels.
    rs=[replace(r,positions=r.positions+np.array([[0.,0,0],[i*.002,0,0]])) for i,r in enumerate(rs)]
    data=tmp_path/'data';m=export_prepared(rs,directory=data,inventory=inv,endpoints=ends,duplicates=empty_duplicates(),policy=PreparationPolicy(),molecule='aspirin',unit_evidence=UNIT_EVIDENCE)
    for name in ('validation','test','validation_minima_diagnostics'): (data/f'{name}.npz').unlink()
    p,_,norm=load_training_partition(data,expected_split_hash=m['split_hash'])
    indices=[i for i,r in enumerate(p.record_ids) if p.memberships[r]['role']=='path'][:16]
    ids=[p.record_ids[i] for i in indices];sid=canonical_hash(ids);source=tmp_path/'source';source.mkdir()
    c={**control_config(),'max_updates':1,'batch_size':16,'source_run':str(source),'data_directory':str(data),'run_directory':str(tmp_path/'run'),
       'expected_prepared_identity':m['identity'],'expected_split_hash':m['split_hash'],'expected_learning_subset_hash':sid,'teacher_seed':23,'teacher_kernel_std':.1,'lstsq_rcond':1e-10}
    original={**c,'mode':'learning_check','batch_size':4}
    (source/'config.json').write_text(json.dumps(original));(source/'learning-subset.json').write_text(json.dumps({'indices':indices,'record_ids':ids,'identity':sid}))
    prov={'dataset_identity':m['identity'],'split_hash':m['split_hash'],'normalization':asdict(norm)}
    (source/'provenance.json').write_text(json.dumps(prov))
    model=EnergyModel(ModelConfig(**c['model']));b=make_batch(p.positions[indices],p.atomic_numbers[indices])
    state=initialize_state(model,b,norm,0,.001)
    # Fixture's synthetic historical counter exercises update identity, not training claims.
    save_checkpoint(source/'last.msgpack',{**state,'update':2000},prov)
    c['source_hashes']={n:sha256(source/n) for n in ('last.msgpack','learning-subset.json','config.json','provenance.json')}
    config=tmp_path/'config.json';config.write_text(json.dumps(c))
    assert main(['diagnose-learning','--config',str(config)])==0
    result=json.loads((tmp_path/'run/summary.json').read_text())
    assert result['execution_valid'] and result['teacher']['checkpoint_new_process_verified']
    assert result['teacher']['updates']==1 and result['real']['status']=='blocked' and result['full_pilot']=='pending'
