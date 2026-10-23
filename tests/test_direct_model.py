"""Catch wrong polar basis/parity, atom mapping, and vacuous zero-head tests."""
import importlib.util
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from force_consistency.models.backbone import ModelConfig
from force_consistency.models.energy import make_batch


def components():
    assert importlib.util.find_spec('force_consistency.models.direct') is not None, 'Direct force model is not implemented'
    from force_consistency.models.direct import DualHeadModel,direct_energy_and_forces
    return DualHeadModel,direct_energy_and_forces


def fixture(nonzero=True):
    Model,predict=components();model=Model(ModelConfig(features=4,iterations=1,radial_basis=4,energy_readout_init='zero'))
    p=np.array([[0.,0,0],[1.1,.2,.1],[.2,1.25,-.3],[-.2,.3,1.4]])
    z=np.array([6,1,8,1]);b=make_batch(p[None],z[None])
    params=model.init(jax.random.PRNGKey(19),b['atomic_numbers'],b['positions'],b['dst_idx'],b['src_idx'],b['batch_segments'],1)['params']
    if nonzero:
        for key in ['energy_head','force_head']:
            params[key]=jax.tree.map(lambda x:jax.random.normal(jax.random.PRNGKey(191),x.shape,dtype=jnp.float64)*.1,params[key])
    return model,predict,p,z,b,params


def test_direct_zero_initialization_dtype_and_can_learn():
    model,pred,_,_,b,params=fixture(False);e,f=pred(params,b,model=model)
    assert e.shape==(1,) and f.shape==(4,3)
    assert all(x.dtype==jnp.float64 for x in jax.tree.leaves(params))
    np.testing.assert_array_equal(e,0);np.testing.assert_array_equal(f,0)
    grads=jax.grad(lambda par:jnp.mean((pred(par,b,model=model)[1]-.1)**2))(params)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(grads))
    assert sum(float(jnp.sum(x*x)) for x in jax.tree.leaves(grads['force_head']))>0


@pytest.mark.parametrize('reflection',[False,True])
def test_direct_nonzero_rotation_parity_translation_and_permutation(reflection):
    model,pred,p,z,b,params=fixture();predict=jax.jit(lambda par,b:pred(par,b,model=model));e,f=predict(params,b)
    assert np.linalg.norm(f)>1e-5
    q,_=np.linalg.qr(np.array([[.2,.7,.5],[.9,.3,.8],[.5,.3,.1]]))
    if np.linalg.det(q)<0:q[:,0]*=-1
    if reflection:q[:,0]*=-1
    order=np.array([2,0,3,1]);p2=(p@q+np.array([3.,-2,.4]))[order];z2=z[order]
    e2,f2=predict(params,make_batch(p2[None],z2[None]))
    np.testing.assert_allclose(e2,e,atol=1e-9,rtol=1e-8)
    np.testing.assert_allclose(f2,np.asarray(f)[order]@q,atol=1e-9,rtol=1e-8)
    eb,fb=predict(params,make_batch(np.stack([p,p2]),np.stack([z,z2])))
    np.testing.assert_allclose(eb,np.concatenate([e,e2]),atol=1e-9,rtol=1e-8)
    np.testing.assert_allclose(fb,np.concatenate([f,f2]),atol=1e-9,rtol=1e-8)


def test_pinned_cartesian_degree_one_axes():
    import e3x
    # Our head requires Cartesian [x,y,z] ordering, not [y,z,x].
    got=e3x.so3.spherical_harmonics(jnp.eye(3,dtype=jnp.float64),max_degree=1,cartesian_order=True)
    np.testing.assert_allclose(got[:,1:4],np.eye(3),atol=1e-15)
    model,pred,_,_,b,params=fixture()
    (_,force),inter=model.apply({'params':params},b['atomic_numbers'],b['positions'],b['dst_idx'],b['src_idx'],b['batch_segments'],1,mutable=['intermediates'])
    features=inter['intermediates']['force_readout'][0]
    np.testing.assert_array_equal(force,features[:,0,1:4,0])


def test_direct_degree_zero_and_float32_parameters_refused():
    Model,pred=components();_,_,_,_,b,params=fixture()
    with pytest.raises(ValueError,match='degree'):
        model=Model(ModelConfig(max_degree=0));model.init(jax.random.PRNGKey(1),b['atomic_numbers'],b['positions'],b['dst_idx'],b['src_idx'],b['batch_segments'],1)
    model,_,_,_,b,_=fixture()
    with pytest.raises(ValueError,match='float64'):pred(jax.tree.map(lambda x:x.astype(jnp.float32),params),b,model=model)
