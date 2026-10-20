"""Synthetic configuration, capacity and nonvacuous symmetry checks."""
from dataclasses import asdict
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from experiments.representation_study.protocol import make_configs,choose_scalar_width,count_parameters


def test_fixed_configs_and_parameter_matching():
    configs=make_configs('/unused','/outputs',73)
    assert set(configs)=={'E64-0','E64-1','SP-0','SP-1','S64-0','S64-1'}
    for key,c in configs.items():
        assert (c['backend'],c['arm'],c['max_updates'],c['validation_interval'],c['batch_size'],c['learning_rate'])==('cpu','energy',200000,500,4,.003)
        assert c['seed']==int(key[-1]) and c['model']['energy_readout_init']=='zero'
        assert c['loss_weights']=={'energy':.014584365670390719,'force':1.9854156343296092}
    assert configs['E64-0']['model']['max_degree']==1 and configs['SP-0']['model']['max_degree']==0
    assert choose_scalar_width({72:995,73:1005,74:1030},1000)==72
    with pytest.raises(ValueError,match='1%'):choose_scalar_width({8:10},1000)
    assert 'S64-0' not in make_configs('/unused','/outputs',64)


def test_shape_only_reference_count():
    c=make_configs('/unused','/outputs',64)['E64-0']
    assert count_parameters(c)==43521


@pytest.mark.parametrize('degree',[0,1])
def test_nonzero_synthetic_physics_and_force_gradients(degree):
    from force_consistency.models.backbone import ModelConfig
    from force_consistency.models.energy import EnergyModel,make_batch,energy_and_forces
    p=np.array([[0.,0,0],[1.1,.2,.1],[.2,1.25,-.3],[-.2,.3,1.4]])
    z=np.array([6,1,8,1]);model=EnergyModel(ModelConfig(features=4,max_degree=degree))
    b=make_batch(p[None],z[None]);params=model.init(jax.random.PRNGKey(19),b['atomic_numbers'],b['positions'],b['dst_idx'],b['src_idx'],b['batch_segments'],1)['params']
    predict=jax.jit(lambda par,b:energy_and_forces(par,b,model=model));e,f=predict(params,b)
    assert all(v.dtype==jnp.float64 for v in jax.tree.leaves(params)) and np.linalg.norm(f)>1e-8
    _,out=model.apply({'params':params},b['atomic_numbers'],b['positions'],b['dst_idx'],b['src_idx'],b['batch_segments'],1,mutable=['intermediates'])
    assert out['intermediates']['features'][0].shape[-2]==(degree+1)**2
    q,_=np.linalg.qr(np.array([[.2,.7,.5],[.9,.3,.8],[.5,.3,.1]]))
    for transform in (np.eye(3),q,q@np.diag([-1.,1.,1.])):
        ix=np.array([2,0,3,1]);et,ft=predict(params,make_batch((p@transform+np.array([3,-2,.4]))[ix][None],z[ix][None]))
        np.testing.assert_allclose(e,et,atol=1e-8,rtol=1e-7)
        np.testing.assert_allclose(np.asarray(f)[ix]@transform,ft,atol=1e-8,rtol=1e-7)
    np.testing.assert_allclose(np.sum(f,axis=0),0,atol=1e-8)
    np.testing.assert_allclose(np.cross(p-p.mean(0),f).sum(0),0,atol=1e-8)
    h=1e-5;a=p.copy();a[1,2]+=h;c=p.copy();c[1,2]-=h
    fd=-(predict(params,make_batch(a[None],z[None]))[0][0]-predict(params,make_batch(c[None],z[None]))[0][0])/(2*h)
    assert abs(float(fd-f[1,2]))<1e-5
    grad=jax.jit(jax.grad(lambda par:jnp.mean((energy_and_forces(par,b,model=model)[1]-.1)**2)))(params)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(grad))
    assert sum(float(jnp.sum(x*x)) for x in jax.tree.leaves(grad))>1e-12
