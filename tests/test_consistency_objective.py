"""Catch force signs, normalization, detached derivatives and lambda-zero work."""
import importlib.util
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from force_consistency.data.prepare import Normalization
from force_consistency.models.energy import make_batch


def api():
    assert importlib.util.find_spec('force_consistency.objectives') is not None, 'Weighted objectives are not implemented'
    from force_consistency import objectives
    return objectives


class AnalyticDual:
    def apply(self,variables,z,p,dst,src,segments,size):
        v=variables['params'];s=v['backbone'];e=jax.ops.segment_sum(.5*v['energy']*s*jnp.sum(p*p,axis=1),segments,num_segments=size)
        return e,-v['force']*s*p


def case():
    b=make_batch(np.array([[[1.,0,0],[0,2.,0]]]),np.array([[1,8]]))
    return b,AnalyticDual(),Normalization(0.,2.,3.)


def test_weighted_losses_follow_hand_computed_scales_and_components():
    o=api();b,m,n=case();p={k:jnp.array(1.) for k in ['energy','force','backbone']}
    value,metrics=o.direct_loss(p,b,m,n,{'energy':.25,'force':1.75})
    assert float(value)==pytest.approx(.25*6.25/4+1.75*(5/6)/9)
    assert float(metrics['energy_mse_eV2'])==pytest.approx(6.25)
    assert float(metrics['force_mse_eV2_per_A2'])==pytest.approx(5/6)


def test_consistency_conservative_zero_and_nonconservative_literal():
    o=api();b,m,n=case();p={k:jnp.array(1.) for k in ['energy','force','backbone']};w={'energy':1.,'force':1.}
    _,metrics=o.consistency_loss(p,b,m,n,.7,w)
    assert float(metrics['consistency_mse_eV2_per_A2'])==pytest.approx(0.,abs=1e-15)
    p['force']=jnp.array(.4);loss,metrics=o.consistency_loss(p,b,m,n,.7,w);ordinary,_=o.direct_loss(p,b,m,n,w)
    assert float(loss-ordinary)==pytest.approx(.7*.6**2*5/(6*9))
    np.testing.assert_allclose(o.energy_derived_forces(p,b,model=m),-np.asarray(b['positions']),atol=1e-15)


def test_lambda_zero_exact_loss_gradients_and_no_coordinate_derivative(monkeypatch):
    o=api();b,m,n=case();p={'energy':jnp.array(1.2),'force':jnp.array(.4),'backbone':jnp.array(.8)};w={'energy':.25,'force':1.75}
    a=lambda p:o.direct_loss(p,b,m,n,w)[0]
    def forbidden(*args,**kwargs):raise AssertionError('lambda zero performed coordinate derivative')
    monkeypatch.setattr(o,'energy_derived_forces',forbidden)
    c=lambda p:o.consistency_loss(p,b,m,n,0.,w)[0]
    np.testing.assert_array_equal(a(p),c(p))
    for x,y in zip(jax.tree.leaves(jax.grad(a)(p)),jax.tree.leaves(jax.grad(c)(p))):np.testing.assert_array_equal(x,y)


def test_consistency_parameter_gradients_match_finite_differences():
    o=api();b,m,n=case();p={'energy':jnp.array(1.2),'force':jnp.array(.4),'backbone':jnp.array(.8)};w={'energy':.25,'force':1.75}
    f=lambda p:o.consistency_loss(p,b,m,n,.7,w)[0];g=jax.grad(f)(p)
    for key in p:
        h=1e-5;plus={**p,key:p[key]+h};minus={**p,key:p[key]-h};fd=(f(plus)-f(minus))/(2*h)
        assert abs(float(g[key]))>1e-6
        np.testing.assert_allclose(g[key],fd,rtol=1e-8,atol=1e-9)


@pytest.mark.parametrize('weights',[{'energy':-1.,'force':1.},{'energy':float('nan'),'force':1.},{'energy':1.,'force':float('inf')},{'energy':0.,'force':0.},{'energy':1.},{'energy':1.,'force':1.,'other':1.}])
def test_invalid_weights_refused(weights):
    o=api();b,m,n=case();p={k:jnp.array(1.) for k in ['energy','force','backbone']}
    with pytest.raises(ValueError,match='weight'):o.direct_loss(p,b,m,n,weights)


@pytest.mark.parametrize('strength',[-.1,float('nan'),float('inf')])
def test_invalid_consistency_strength_refused(strength):
    o=api();b,m,n=case();p={k:jnp.array(1.) for k in ['energy','force','backbone']}
    with pytest.raises(ValueError,match='strength'):o.consistency_loss(p,b,m,n,strength,{'energy':1.,'force':1.})


def test_real_dual_head_consistency_gradient_reaches_all_three_parameter_groups():
    from test_direct_model import fixture
    o=api();model,_,_,_,batch,params=fixture(nonzero=True)
    from jax.flatten_util import ravel_pytree
    flat,unravel=ravel_pytree(params)
    f=jax.jit(lambda vector:o.consistency_loss(unravel(vector),batch,model,Normalization(0.,1.,1.),.7,{'energy':.25,'force':1.75})[0])
    g=jax.jit(jax.grad(f))(flat)
    # Choose one active leaf coordinate per real model group, not zero head biases.
    offset=0;indices=[]
    for group in sorted(params):
        group_flat,_=ravel_pytree(params[group]);section=np.asarray(g[offset:offset+len(group_flat)])
        index=offset+int(np.argmax(np.abs(section)))
        assert abs(float(g[index]))>1e-10,group
        indices.append(index);offset+=len(group_flat)
    assert len(indices)==3
    for index in indices:
        h=1e-5;fd=(f(flat.at[index].add(h))-f(flat.at[index].add(-h)))/(2*h)
        np.testing.assert_allclose(g[index],fd,rtol=2e-5,atol=1e-8)
