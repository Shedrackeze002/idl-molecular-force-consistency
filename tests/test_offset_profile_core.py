"""Catch atom scaling, local centering, biased Adam and mutable restoration."""
import copy
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

def api():
    spec=importlib.util.spec_from_file_location('offset_profile_core',Path(__file__).resolve().parents[1]/'scripts/offset_profile_core.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module

def params():
    return {'body':np.array([2.,3.]),'energy_head':{'0+':{'kernel':np.array([[.5]]),'bias':np.array([.2])}}}

@pytest.mark.parametrize('shift,delta',[(0.,0.),(4.,-4/21),(-7.,1/3)])
def test_projection_uses_global_mean_and_atomic_bias(shift,delta):
    m=api();p=params();before=copy.deepcopy(p);errors=np.array([-3.,-1.,1.,3.])+shift
    projected,r=m.project_bias(p,errors,np.full(4,21))
    assert r['delta_bias']==pytest.approx(delta)
    assert r['predicted_post_mean_eV']==pytest.approx(0.,abs=1e-14)
    assert r['predicted_post_mse_eV2']==pytest.approx(5.)
    assert r['centered_variance_eV2']==pytest.approx(5.)
    assert float(projected['energy_head']['0+']['bias'][0])==pytest.approx(.2+delta)
    np.testing.assert_array_equal(p['energy_head']['0+']['bias'],before['energy_head']['0+']['bias'])
    np.testing.assert_array_equal(projected['body'],before['body'])
    np.testing.assert_array_equal(projected['energy_head']['0+']['kernel'],before['energy_head']['0+']['kernel'])

@pytest.mark.parametrize('errors,atoms', [([],[]),([1.,2.],[21,20]),([np.nan],[21]),([np.inf],[21]),([1.],[0]),([1.,2.],[21])])
def test_projection_rejects_invalid_support(errors,atoms):
    with pytest.raises(ValueError):api().project_bias(params(),np.array(errors),np.array(atoms))

def test_between_chunk_residuals_not_removed_by_local_centering():
    # Full residual variance is5, while averaging each2-element chunk variance gives1.
    _,r=api().project_bias(params(),np.array([-3.,-1.,1.,3.]),np.full(4,21))
    assert r['predicted_post_mse_eV2']==5.
    assert r['predicted_post_mse_eV2']!=1.

def test_restoration_decodes_saved_state_without_parameter_initializer(tmp_path,monkeypatch):
    from test_coverage import seven_case,close_trees
    from force_consistency.runs import save_checkpoint
    from force_consistency import train
    _,_,_,_,state=seven_case();path=tmp_path/'zero.msgpack';save_checkpoint(path,state,{})
    def forbidden(*a,**kw):raise AssertionError('initializer called')
    monkeypatch.setattr(train,'initialize_state',forbidden)
    restored=api().restore_state(path,{})
    close_trees((restored['params'],restored['optimizer'],restored['model_rng']),
                (state['params'],state['optimizer'],state['model_rng']))
    assert restored['update']==state['update']
    assert restored['sampler_rng_json']==state['sampler_rng_json']
    assert restored['normalization_json']==state['normalization_json']

def direct_call(kind,count,fn,*args,on_returned=lambda x:None):
    import jax
    from force_consistency.runs import assert_finite
    result=fn(*args);jax.block_until_ready(result);on_returned(result);assert_finite(result)
    return result

def fixture(count):
    import jax
    import jax.numpy as jnp
    from test_coverage import seven_case
    from force_consistency.metrics import partition_batch
    p,model,full,norm,state=seven_case();m=api()
    pp=copy.deepcopy(state['params']);pp['energy_head']['0+']['kernel']=jnp.full_like(pp['energy_head']['0+']['kernel'],.13)
    energy=model.apply({'params':pp},full['atomic_numbers'],full['positions'],full['dst_idx'],full['src_idx'],full['batch_segments'],7)
    pp,_=m.project_bias(pp,np.asarray(energy)-p.energy_residual,np.full(7,3))
    mu=jax.tree.map(lambda x:jnp.full_like(x,.03),state['optimizer'][0].mu)
    nu=jax.tree.map(lambda x:jnp.full_like(x,.02),state['optimizer'][0].nu)
    if count:
        opt=state['optimizer'][0]._replace(count=jnp.asarray(count,dtype=jnp.int32),mu=m.set_bias(mu,[0.]),nu=m.set_bias(nu,[0.]))
        state={**state,'optimizer':(opt,state['optimizer'][1]),'update':count}
    state={**state,'params':pp}
    batches=[partition_batch(p,np.arange(i,min(i+3,7))) for i in range(0,7,3)]
    return m,p,model,full,norm,state,batches

@pytest.mark.parametrize('count',[0,3])
def test_real_profile_step_matches_global_objective_and_independent_adam(count):
    import jax
    import jax.numpy as jnp
    from force_consistency.models.energy import energy_and_forces
    from force_consistency.width_rate_runs import independent_adam_step
    from test_coverage import close_trees
    m,p,model,full,norm,state,batches=fixture(count);weights={'energy':.2,'force':1.8}
    before=m.full_hash(state);events=[]
    def objective(pp):
        e,f=energy_and_forces(pp,full,model=model);err=e-full['energy']
        return weights['energy']*jnp.mean((err-jnp.mean(err))**2)/norm.energy_loss_scale_eV**2+weights['force']*jnp.mean((f-full['forces'])**2)/norm.force_loss_scale_eV_per_A**2
    wanted_grad=jax.jit(jax.grad(objective))(state['params']);wanted_grad=m.set_bias(wanted_grad,[0.])
    rp,ro=independent_adam_step(state['params'],state['optimizer'],wanted_grad,.0003)
    re,rf=energy_and_forces(rp,full,model=model)
    wanted_bias=float(rp['energy_head']['0+']['bias'][0])-float(jnp.mean(re-full['energy']))/3
    rp=m.set_bias(rp,[wanted_bias])
    got,g,r=m.profile_step(state,batches,model,norm,weights,.0003,direct_call,lambda event:events.append(event))
    close_trees(g,wanted_grad);close_trees((got['params'],got['optimizer']),(rp,ro))
    ge,gf=energy_and_forces(got['params'],full,model=model)
    np.testing.assert_allclose(gf,rf,atol=1e-12,rtol=0)
    assert float(jnp.mean(ge-full['energy']))==pytest.approx(0.,abs=1e-12)
    assert got['update']==count+1 and int(got['optimizer'][0].count)==count+1
    assert m.full_hash(state)==before
    assert float(got['optimizer'][0].mu['energy_head']['0+']['bias'][0])==0.
    assert float(got['optimizer'][0].nu['energy_head']['0+']['bias'][0])==0.
    assert r['projection']['predicted_post_mean_eV']==pytest.approx(0.,abs=1e-14)
    oracle=m.profile_oracle(state,got,g,full,model,norm,weights,.0003,direct_call,lambda event:events.append(event))
    assert oracle['status']=='passed'
    assert oracle['incoming_nonzero_moments']==bool(count)
    assert m.full_hash(state)==before

def test_nonzero_bias_moment_rejected_before_any_numerical_call():
    m,p,model,full,norm,state,batches=fixture(3)
    state['optimizer']=(state['optimizer'][0]._replace(mu=m.set_bias(state['optimizer'][0].mu,[.1])),state['optimizer'][1])
    def forbidden(*a,**kw):raise AssertionError('scheduled numerical call')
    with pytest.raises(ValueError,match='bias moment'):
        m.profile_step(state,batches,model,norm,{'energy':.2,'force':1.8},.0003,forbidden,lambda e:None)
