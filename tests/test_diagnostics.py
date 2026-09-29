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
