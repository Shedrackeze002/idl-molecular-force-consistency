"""Catch factory wiring, weighted energy regressions and unsafe arm reloads."""
from dataclasses import asdict,replace
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from force_consistency.data.prepare import Normalization
from force_consistency.models.backbone import ModelConfig
from force_consistency.models.energy import EnergyModel,energy_and_forces,make_batch
from force_consistency.train import RunConfig,validate_run_configuration,initialize_state,build_step,normalized_loss


def config(**kwargs):
    return RunConfig(data_directory='unused',expected_split_hash='unused',run_directory='unused',**kwargs)


def test_default_energy_config_and_fixed_weight_validation():
    c=config();assert getattr(c,'arm',None)=='energy', 'Arm support is not implemented'
    assert c.consistency_strength==0 and c.loss_weights=={'energy':1.,'force':1.}
    validate_run_configuration(c)


@pytest.mark.parametrize('changes',[{'arm':'bogus'},{'arm':'direct','consistency_strength':1.},{'arm':'energy','consistency_strength':1.},{'arm':'consistency','consistency_strength':-.1},{'arm':'consistency','consistency_strength':float('nan')},{'loss_weights':{'energy':0.,'force':0.}},{'loss_weights':{'energy':1.}},{'arm':'direct','model':{'max_degree':0}}])
def test_bad_arm_objective_contracts_refused(changes):
    with pytest.raises(ValueError):validate_run_configuration(config(**changes))


def test_energy_default_loss_gradients_and_step_match_original_equation():
    c=config();assert hasattr(c,'loss_weights'), 'Fixed objective weights not implemented'
    model=EnergyModel(ModelConfig(features=4,iterations=1,radial_basis=4));b=make_batch(np.array([[[0.,0,0],[1.1,.2,.1],[.2,1.25,-.3]]]),np.array([[6,1,8]]),energy=[.3],forces=np.ones((1,3,3))*.1)
    n=Normalization(0.,2.,3.);s=initialize_state(model,b,n,0,.001)
    def old_loss(p):
        e,f=energy_and_forces(p,b,model=model)
        return jnp.mean((e-b['energy'])**2)/4+jnp.mean((f-b['forces'])**2)/9
    def new_loss(p):return normalized_loss(p,b,model,n)[0]
    np.testing.assert_array_equal(old_loss(s['params']),new_loss(s['params']))
    for a,d in zip(jax.tree.leaves(jax.grad(old_loss)(s['params'])),jax.tree.leaves(jax.grad(new_loss)(s['params']))):np.testing.assert_array_equal(a,d)
    import optax
    opt=optax.adam(.001);updates,state=opt.update(jax.grad(old_loss)(s['params']),s['optimizer'],s['params']);expected=optax.apply_updates(s['params'],updates)
    actual,actualopt,_=build_step(model,n,.001)(s['params'],s['optimizer'],b)
    for a,d in zip(jax.tree.leaves((expected,state)),jax.tree.leaves((actual,actualopt))):np.testing.assert_allclose(a,d,rtol=1e-12,atol=1e-12)
    value,_=normalized_loss(s['params'],b,model,n,loss_weights={'energy':.25,'force':1.75});e,f=energy_and_forces(s['params'],b,model=model)
    np.testing.assert_allclose(value,.25*jnp.mean((e-b['energy'])**2)/4+1.75*jnp.mean((f-b['forces'])**2)/9,rtol=1e-15)


def test_direct_consistency_zero_same_initial_state_and_step():
    from force_consistency import train as t
    assert hasattr(t,'model_and_predictor'), 'Shared trainer factory is not implemented'
    model_cfg=asdict(ModelConfig(features=4,iterations=1,radial_basis=4,energy_readout_init='zero'))
    direct=config(arm='direct',model=model_cfg);cons=config(arm='consistency',model=model_cfg)
    m,predict=t.model_and_predictor(direct);m2,_=t.model_and_predictor(cons)
    b=make_batch(np.array([[[0.,0,0],[1.1,.2,.1],[.2,1.25,-.3]]]),np.array([[6,1,8]]),energy=[.3],forces=np.ones((1,3,3))*.1);n=Normalization(0.,1.,1.)
    a=initialize_state(m,b,n,0,.001);d=initialize_state(m2,b,n,0,.001)
    for x,y in zip(jax.tree.leaves(a),jax.tree.leaves(d)):
        if hasattr(x,'dtype'):np.testing.assert_array_equal(x,y)
        else:assert x==y
    aa=t.build_step(m,n,.001,arm='direct',loss_weights={'energy':.25,'force':1.75})(a['params'],a['optimizer'],b)
    dd=t.build_step(m2,n,.001,arm='consistency',consistency_strength=0.,loss_weights={'energy':.25,'force':1.75})(d['params'],d['optimizer'],b)
    for x,y in zip(jax.tree.leaves(aa),jax.tree.leaves(dd)):np.testing.assert_array_equal(x,y)
    assert predict(aa[0],b,model=m)[1].shape==(3,3)


@pytest.mark.parametrize('arm,strength',[('energy',0.),('direct',0.),('consistency',.1)])
def test_synthetic_training_both_checkpoint_roles_fresh_reload_and_wrong_weights(tmp_path,arm,strength):
    from force_consistency import train as t
    assert hasattr(t,'model_and_predictor'), 'Shared trainer factory is not implemented'
    from force_consistency.data.prepare import export_prepared
    from test_splits_prepare import fixture,empty_duplicates,UNIT_EVIDENCE
    from force_consistency.data.types import PreparationPolicy
    records,inv,ends=fixture();prepared=tmp_path/'data'
    manifest=export_prepared(records,directory=prepared,inventory=inv,endpoints=ends,duplicates=empty_duplicates(),policy=PreparationPolicy(),molecule='aspirin',unit_evidence=UNIT_EVIDENCE)
    c=RunConfig(data_directory=str(prepared),expected_split_hash=manifest['split_hash'],run_directory=str(tmp_path/'run'),mode='integration',arm=arm,consistency_strength=strength,loss_weights={'energy':.25,'force':1.75},max_updates=2,validation_interval=2,wall_seconds=180,model=asdict(ModelConfig(features=4,iterations=1,radial_basis=4,energy_readout_init='zero')))
    result=t.train(c);assert result['status']=='passed',result
    assert result['updates']==2 and result['checkpoint_new_process_verified'] and result['best_checkpoint_new_process_verified']
    assert result['provenance']['arm']==arm and result['provenance']['loss_weights']==c.loss_weights
    assert len(result['provenance']['initial_parameter_sha256'])==64
    assert len(result['provenance']['initial_state_sha256'])==64
    for role in ['last','best']:
        filename='checkpoint-verification.json' if role=='last' else 'best-checkpoint-verification.json'
        v=json.loads((tmp_path/'run'/filename).read_text());assert v['role']==role and v['status']=='passed'
    configpath=tmp_path/'run/config.json';changed=json.loads(configpath.read_text());changed['loss_weights']={'energy':1.,'force':1.};configpath.write_text(json.dumps(changed))
    with pytest.raises(ValueError,match='identity'):t.checkpoint_probe(tmp_path/'run')


def test_paired_initialization_identity_persistable_and_seed_sensitive():
    from force_consistency import train as t
    assert hasattr(t,'initialization_identity'), 'Initial identities are not retained in provenance'
    b=make_batch(np.array([[[0.,0,0],[1.1,.2,.1],[.2,1.25,-.3]]]),np.array([[6,1,8]]));n=Normalization(0.,1.,1.)
    mc=asdict(ModelConfig(features=4,iterations=1,radial_basis=4,energy_readout_init='zero'))
    m,_=t.model_and_predictor(config(arm='direct',model=mc));cm,_=t.model_and_predictor(config(arm='consistency',model=mc))
    a=t.initialize_state(m,b,n,0,.001);c=t.initialize_state(cm,b,n,0,.001);other=t.initialize_state(m,b,n,1,.001)
    identity=t.initialization_identity(a)
    assert identity==t.initialization_identity(c)
    assert identity['initial_parameter_sha256']!=t.initialization_identity(other)['initial_parameter_sha256']
    assert identity['initial_state_sha256']!=t.initialization_identity(other)['initial_state_sha256']
    # Optimizer/RNG/update changes must affect state identity even with unchanged params.
    advanced={**a,'update':1}
    assert identity['initial_parameter_sha256']==t.initialization_identity(advanced)['initial_parameter_sha256']
    assert identity['initial_state_sha256']!=t.initialization_identity(advanced)['initial_state_sha256']


def test_legacy_energy_loss_auxiliary_tree_contract_is_unchanged():
    model=EnergyModel(ModelConfig(features=4,iterations=1,radial_basis=4));b=make_batch(np.array([[[0.,0,0],[1.1,.2,.1]]]),np.array([[6,1]]));n=Normalization(0.,1.,1.)
    state=initialize_state(model,b,n,0,.001)
    _,metrics=normalized_loss(state['params'],b,model,n)
    assert set(metrics)=={'energy_mse_eV2','force_mse_eV2_per_A2'}
    _,_,metrics=build_step(model,n,.001)(state['params'],state['optimizer'],b)
    assert set(metrics)=={'energy_mse_eV2','force_mse_eV2_per_A2','scaled_loss'}
