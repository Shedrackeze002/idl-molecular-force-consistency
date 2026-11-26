import copy
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from test_coverage import seven_case,close_trees
from force_consistency.gradient_diagnostics import split_gradients,gradient_geometry,verify_directional_derivatives,analyze_snapshot
from force_consistency.coverage_runs import _state_hashes
from force_consistency.train import normalized_loss
from force_consistency.diagnostics import with_head


def witness():
    p,m,b,n,s=seven_case();s=copy.deepcopy(s)
    s['params']=with_head(s['params'],np.array([.03,-.02,.01,.04,.005]))
    return p,m,b,n,s


def test_uneven_chunks_match_original_joint_and_split_derivative():
    p,m,b,n,s=witness();before=_state_hashes(s)
    losses,ge,gf,gj=split_gradients(s['params'],p,m,n,3)
    (want,_),joint=jax.jit(jax.value_and_grad(lambda pp:normalized_loss(pp,b,m,n),has_aux=True))(s['params'])
    close_trees(gj,joint);close_trees(gj,jax.tree.map(lambda a,b:a+b,ge,gf))
    assert losses['energy']+losses['force']==pytest.approx(float(want),abs=1e-10)
    assert sum(float(jnp.linalg.norm(v)) for v in jax.tree.leaves(gf['Backbone_0']))>1e-6
    assert _state_hashes(s)==before

@pytest.mark.parametrize('e,f,cosine,ratio,dot',[(np.array([1.,0]),np.array([2.,0]),1.,2.,2.),(np.array([1.,0]),np.array([-2.,0]),-1.,2.,-2.),(np.array([1.,0]),np.array([0.,2]),0.,2.,0.),(np.zeros(2),np.ones(2),None,None,0.)])
def test_geometry_known_directions(e,f,cosine,ratio,dot):
    g=gradient_geometry({'x':e},{'x':f})['whole']
    assert g['cosine']==cosine and g['force_over_energy_norm']==ratio and g['dot']==dot
    assert g['energy_raw_joint_descent_slope']==pytest.approx(-(e@e+dot))
    assert g['force_raw_joint_descent_slope']==pytest.approx(-(f@f+dot))


def test_finite_differences_retain_both_steps_detect_wrong_sign_and_preserve_state():
    p,m,b,n,s=witness();before=_state_hashes(s)
    _,ge,gf,_=split_gradients(s['params'],p,m,n,3)
    result=verify_directional_derivatives(s['params'],p,m,n,ge,gf)
    assert result['valid']
    for k in ('energy','force'):
        assert [r['step'] for r in result[k]['steps']]==[1e-4,1e-5]
        assert any(r['valid'] for r in result[k]['steps'])
    wrong=verify_directional_derivatives(s['params'],p,m,n,jax.tree.map(lambda x:-x,ge),gf)
    assert not wrong['valid'] and not wrong['energy']['valid']
    assert _state_hashes(s)==before


def test_zero_head_structural_blocks_and_deterministic_fallback():
    p,m,b,n,s=seven_case();d=analyze_snapshot(s,p,m,n,{'microbatch_size':3})
    assert d['verification_valid'] and d['state_hashes_before']==d['state_hashes_after']
    assert d['geometry']['backbone']['force_norm']==0
    assert d['geometry']['head_bias']['force_norm']==0
    assert d['geometry']['backbone']['cosine'] is None
    zeros=jax.tree.map(jnp.zeros_like,s['params'])
    a=verify_directional_derivatives(s['params'],p,m,n,zeros,zeros)
    c=verify_directional_derivatives(s['params'],p,m,n,zeros,zeros)
    assert a==c and a['energy']['direction_kind']=='seeded_rademacher'
    assert not a['valid']  # A manufactured zero gradient must fail against nonzero true objective derivative.


def test_snapshot_retains_gradient_observations_before_fd_budget_stop():
    from force_consistency.coverage_runs import WallBudget
    p,m,b,n,s=seven_case();saved=[]
    def retain(value):saved.append(copy.deepcopy(value))
    def budget():
        if saved:raise WallBudget('stop after retained gradients')
    with pytest.raises(WallBudget):
        analyze_snapshot(s,p,m,n,{'microbatch_size':3,'_check_budget':budget,'_save_partial':retain})
    assert saved[0]['losses']['joint']>0 and saved[0]['joint_gradient_identity']['valid']
    assert 'geometry' in saved[0]
