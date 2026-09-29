from dataclasses import replace
from pathlib import Path
import json
import subprocess
import sys
import numpy as np
import jax
import jax.numpy as jnp
import pytest

from force_consistency.data.prepare import PreparedBundle,PreparedPartition,Normalization


def bundle():
    def partition(n,prefix,groups):
        p=np.repeat(np.array([[[0.,0,0],[1.1,.2,.1],[.2,1.25,-.3]]]),n,axis=0)
        ids=tuple(f'{prefix}{i}' for i in range(n));mem={rid:{'groups':[groups[i]],'role':'path','seen_in_training':prefix=='tr'} for i,rid in enumerate(ids)}
        return PreparedPartition(p,np.arange(n,dtype=float)*.01,np.ones_like(p)*.05,np.repeat([[6,1,8]],n,axis=0),ids,mem)
    train=partition(6,'tr',['a']*3+['b']*3);val=partition(11,'v',['a']*3+['b']*8)
    return PreparedBundle(train,val,partition(0,'d',[]),{'molecule':'synthetic','splits':{'groups':{'synthetic':{'train':['a','b']}}}},Normalization(0.,1.,1.))


def test_validation_exhaustive_partial_batches_and_weighting():
    from force_consistency.metrics import validate
    b=bundle();f=np.zeros_like(b.validation.forces);f[:3]=1;f[3:]=3
    b=replace(b,validation=replace(b.validation,forces=f))
    seen=[]
    def pred(_,batch):
        seen.append(len(batch['energy']));return jnp.zeros_like(batch['energy']),jnp.zeros_like(batch['forces'])
    m=validate(None,b,None,4,predict=pred)
    assert seen==[4,4,3] and m['evaluated_ids']==list(b.validation.record_ids)
    assert m['pooled_force_mae_eV_per_A']==pytest.approx(27/11)
    assert m['equal_connection_force_mae_eV_per_A']==2
    assert m['force_component_count']==99 and m['diagnostics']['evaluated_ids']==[]
    assert m['per_group']['a']['force_component_count']==27


def test_uniform_group_sampler_and_resume_rng():
    from force_consistency.train import sample_training_batch,new_sampler_state
    b=bundle();state=new_sampler_state(0);batch,nextstate=sample_training_batch(b,state,4)
    again,againstate=sample_training_batch(b,state,4)
    np.testing.assert_array_equal(batch['record_indices'],again['record_indices']);assert nextstate==againstate and nextstate!=state
    second,end=sample_training_batch(b,nextstate,4)
    resumed,resumedend=sample_training_batch(b,json.loads(json.dumps(nextstate)),4)
    np.testing.assert_array_equal(second['record_indices'],resumed['record_indices']);assert end==resumedend
    counts=np.bincount(np.array(sample_training_batch(b,state,2000)[0]['record_indices']),minlength=6)
    assert 850<counts[:3].sum()<1150


def test_checkpoint_rejects_wrong_identity_corruption_and_dtype(tmp_path):
    from force_consistency.runs import save_checkpoint,load_checkpoint
    s={'params':{'x':jnp.ones(2,dtype=jnp.float64)},'optimizer':{'m':jnp.zeros(2)},'update':2,'sampler_rng_json':'{}','model_rng':jnp.array([1,2],dtype=jnp.uint32),'normalization_json':'{}'}
    p=tmp_path/'checkpoint.msgpack';prov={'split_hash':'ok'}
    save_checkpoint(p,s,prov);loaded=load_checkpoint(p,prov,template=s)
    np.testing.assert_array_equal(loaded['params']['x'],s['params']['x'])
    with pytest.raises(ValueError,match='provenance'): load_checkpoint(p,{'split_hash':'wrong'},template=s)
    with pytest.raises(ValueError,match='float64'): save_checkpoint(p,{**s,'params':{'x':jnp.ones(2,dtype=jnp.float32)}},prov)
    raw=bytearray(p.read_bytes());raw[len(raw)//2]^=1;p.write_bytes(raw)
    with pytest.raises(ValueError): load_checkpoint(p,prov,template=s)


def test_optimizer_resume_matches_uninterrupted_and_new_process(tmp_path):
    from force_consistency.models.energy import EnergyModel,energy_and_forces
    from force_consistency.models.backbone import ModelConfig
    from force_consistency.train import initialize_state,build_step,sample_training_batch
    from force_consistency.runs import save_checkpoint,load_checkpoint
    b=bundle();model=EnergyModel(ModelConfig(features=4,iterations=1,radial_basis=4))
    initbatch,_=sample_training_batch(b,np.random.PCG64(0).state,4)
    state=initialize_state(model,initbatch,b.normalization,0,.001);step=build_step(model,b.normalization,.001)
    def advance(s):
        batch,rng=sample_training_batch(b,json.loads(s['sampler_rng_json']),4)
        params,opt,metrics=step(s['params'],s['optimizer'],batch)
        return {**s,'params':params,'optimizer':opt,'update':s['update']+1,'sampler_rng_json':json.dumps(rng,sort_keys=True)}
    one=advance(state);p=tmp_path/'checkpoint.msgpack';prov={'test':'cpu-continuity'};save_checkpoint(p,one,prov)
    resumed=advance(load_checkpoint(p,prov,template=state));full=advance(one)
    for a,c in zip(jax.tree.leaves(resumed),jax.tree.leaves(full)):
        if hasattr(a,'dtype'): np.testing.assert_allclose(a,c,atol=1e-10,rtol=1e-10)
        else: assert a==c
    np.savez(tmp_path/'batch.npz',**{k:np.array(v) for k,v in initbatch.items()})
    script='''import json,sys,numpy as np,jax
from force_consistency.models.energy import EnergyModel,energy_and_forces
from force_consistency.models.backbone import ModelConfig
from force_consistency.data.prepare import Normalization
from force_consistency.train import initialize_state
from force_consistency.runs import load_checkpoint
root=sys.argv[1];batch={k:jax.numpy.asarray(v) for k,v in np.load(root+'/batch.npz').items()}
model=EnergyModel(ModelConfig(features=4,iterations=1,radial_basis=4));template=initialize_state(model,batch,Normalization(0,1,1),0,.001)
state=load_checkpoint(root+'/checkpoint.msgpack',{'test':'cpu-continuity'},template=template)
e,f=energy_and_forces(state['params'],batch,model=model);np.savez(root+'/probe.npz',energy=e,forces=f)
'''
    subprocess.run([sys.executable,'-c',script,str(tmp_path)],check=True,capture_output=True)
    e,f=energy_and_forces(one['params'],initbatch,model=model)
    with np.load(tmp_path/'probe.npz') as probe:
        np.testing.assert_allclose(e,probe['energy'],atol=1e-10,rtol=1e-10);np.testing.assert_allclose(f,probe['forces'],atol=1e-10,rtol=1e-10)


def test_backend_requirement_and_nonfinite_update():
    from force_consistency.runs import require_backend,assert_finite
    assert require_backend('cpu').platform=='cpu'
    if not any(d.platform=='gpu' for d in jax.devices()):
        with pytest.raises(ValueError,match='GPU'): require_backend('gpu')
    with pytest.raises(ValueError): require_backend('imaginary')
    with pytest.raises(FloatingPointError): assert_finite({'loss':np.nan})
