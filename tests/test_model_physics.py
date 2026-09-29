import jax
jax.config.update('jax_enable_x64',True)
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.fixture(scope='module')
def system():
    from force_consistency.models.backbone import ModelConfig
    from force_consistency.models.energy import EnergyModel,make_batch,energy_and_forces
    model=EnergyModel(ModelConfig(energy_readout_init='zero'))
    p=np.array([[0.,0,0],[1.1,.2,.1],[.2,1.25,-.3],[-.2,.3,1.4]])
    z=np.array([6,1,8,1]);b=make_batch(p[None],z[None])
    params=model.init(jax.random.PRNGKey(19),b['atomic_numbers'],b['positions'],b['dst_idx'],b['src_idx'],b['batch_segments'],1)['params']
    # A deterministic nonzero head avoids vacuous zero-output physics checks.
    params['energy_head']=jax.tree.map(lambda x:jax.random.normal(jax.random.PRNGKey(191),x.shape,dtype=jnp.float64)*.1,params['energy_head'])
    predict=jax.jit(lambda par,b:energy_and_forces(par,b,model=model))
    return model,p,z,params,predict


def test_float64_scalar_vector_features(system):
    from force_consistency.models.energy import make_batch
    model,p,z,params,_=system;b=make_batch(p[None],z[None])
    assert all(v.dtype==jnp.float64 for v in jax.tree.leaves(params))
    _,out=model.apply({'params':params},b['atomic_numbers'],b['positions'],b['dst_idx'],b['src_idx'],b['batch_segments'],1,mutable=['intermediates'])
    x=out['intermediates']['features'][0]
    assert x.shape[-2]==4 and np.linalg.norm(x[...,0,:])>0 and np.linalg.norm(x[...,1:4,:])>0


def test_o3_translation_permutation_and_batch(system):
    from force_consistency.models.energy import make_batch
    _,p,z,par,pred=system;e,f=pred(par,make_batch(p[None],z[None]))
    q,_=np.linalg.qr(np.array([[.2,.7,.5],[.9,.3,.8],[.5,.3,.1]]));q[:,0]*=-1
    order=np.array([2,0,3,1]);trans=(p@q+np.array([3,-2,.4]))[order]
    et,ft=pred(par,make_batch(trans[None],z[order][None]))
    np.testing.assert_allclose(e,et,atol=1e-8,rtol=1e-7)
    np.testing.assert_allclose(np.array(f).reshape(-1,3)[order]@q,np.array(ft).reshape(-1,3),atol=1e-8,rtol=1e-7)
    eb,fb=pred(par,make_batch(np.stack([p,trans]),np.stack([z,z[order]])))
    np.testing.assert_allclose(eb,np.concatenate([e,et]),atol=1e-8,rtol=1e-7)
    np.testing.assert_allclose(np.array(fb),np.concatenate([f,ft]),atol=1e-8,rtol=1e-7)


def test_finite_difference_convergence_net_force_torque(system):
    from force_consistency.models.energy import make_batch
    _,p,z,par,pred=system;_,f=pred(par,make_batch(p[None],z[None]));errors=[]
    for h in (1e-3,1e-4,1e-5):
        fd=np.zeros_like(p)
        for i in range(len(p)):
            for d in range(3):
                a=p.copy();b=p.copy();a[i,d]+=h;b[i,d]-=h
                fd[i,d]=-float((pred(par,make_batch(a[None],z[None]))[0][0]-pred(par,make_batch(b[None],z[None]))[0][0])/(2*h))
        errors.append(float(np.max(np.abs(fd-f))))
    assert min(errors)<=1e-5 and errors[1]<errors[0]
    np.testing.assert_allclose(np.sum(f,axis=0),0,atol=1e-8)
    np.testing.assert_allclose(np.cross(p-p.mean(0),f).sum(0),0,atol=1e-8)


def test_force_loss_parameter_derivatives_finite_nonzero(system):
    from force_consistency.models.energy import make_batch,energy_and_forces
    model,p,z,par,_=system;b=make_batch(p[None],z[None])
    g=jax.jit(jax.grad(lambda params:jnp.mean((energy_and_forces(params,b,model=model)[1]-.1)**2)))(par)
    leaves=jax.tree.leaves(g)
    assert all(np.isfinite(x).all() for x in leaves) and sum(float(jnp.sum(x*x)) for x in leaves)>1e-12


def test_cutoff_matching_limits(system):
    from force_consistency.models.energy import make_batch
    _,_,_,par,pred=system;z=np.array([[6,8]]);gaps=[]
    for h in (.1,.03,.01):
        a=make_batch(np.array([[[0.,0,0],[5-h,0,0]]]),z)
        b=make_batch(np.array([[[0.,0,0],[5+h,0,0]]]),z)
        ea,fa=pred(par,a);eb,fb=pred(par,b)
        gaps.append(float(np.max(np.abs(ea-eb))+np.max(np.abs(fa-fb))))
    assert gaps[-1]<1e-8 and gaps[-1]<=gaps[0]


def test_reject_float32_inputs(system):
    from force_consistency.models.energy import make_batch
    _,p,z,_,_=system
    with pytest.raises(ValueError,match='float64'): make_batch(p[None].astype(np.float32),z[None])


def test_zero_readout_is_explicit_and_can_learn_forces():
    from force_consistency.models.backbone import ModelConfig
    from force_consistency.models.energy import EnergyModel,make_batch,energy_and_forces
    from force_consistency.train import initialize_state,build_step
    from force_consistency.data.prepare import Normalization
    model=EnergyModel(ModelConfig(features=4,iterations=1,radial_basis=4,energy_readout_init='zero'))
    p=np.array([[[0.,0,0],[1.1,.2,.1],[.2,1.25,-.3]]]);b=make_batch(p,np.array([[6,1,8]]),np.array([.2]),-.1*p)
    state=initialize_state(model,b,Normalization(0,1,1),0,.001)
    e,f=energy_and_forces(state['params'],b,model=model)
    np.testing.assert_array_equal(e,0);np.testing.assert_array_equal(f,0)
    g=jax.grad(lambda params:jnp.mean((energy_and_forces(params,b,model=model)[1]-b['forces'])**2))(state['params'])
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(g))
    assert sum(float(jnp.sum(x*x)) for x in jax.tree.leaves(g['energy_head']))>0
    params,_,metrics=build_step(model,Normalization(0,1,1),.001)(state['params'],state['optimizer'],b)
    assert np.isfinite(metrics['scaled_loss']) and np.linalg.norm(energy_and_forces(params,b,model=model)[1])>0
