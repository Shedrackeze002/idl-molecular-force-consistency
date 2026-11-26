import copy
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from force_consistency.optimizer_diagnostics import adam_directions,direction_geometry,directional_fd,analyze_optimizer_snapshot,tree_norm
from force_consistency.coverage_runs import _state_hashes,coverage_metrics,WallBudget
from test_gradient_diagnostics import witness
from test_coverage import close_trees


def scalar_state(g=None,mu=None):
    p={'x':jnp.array([.3,-.7]),'tiny':jnp.array([.1])};tx=optax.adam(.001)
    opt=tx.init(p);g=g or {'x':jnp.array([.2,-.4]),'tiny':jnp.array([1e-12])}
    mu=mu or {'x':jnp.array([.07,-.02]),'tiny':jnp.array([0.])}
    nu={'x':jnp.array([.09,.05]),'tiny':jnp.array([0.])}
    return p,(opt[0]._replace(count=jnp.array(2000,jnp.int32),mu=mu,nu=nu),opt[1]),g


def test_saved_moments_count_bias_correction_epsilon_and_equal_norm_controls():
    p,o,g=scalar_state();dirs,v=adam_directions(p,o,g,.001)
    expected=jax.tree.map(lambda gg,m,n:-.001*(.9*np.asarray(m)+.1*np.asarray(gg))/(1-.9**2001)/(np.sqrt((.999*np.asarray(n)+.001*np.asarray(gg)**2)/(1-.999**2001))+1e-8),g,o[0].mu,o[0].nu)
    close_trees(dirs['stateful_adam']['delta'],expected)
    assert v['valid'] and v['next_count']==2001 and v['reset_count']==1 and int(o[0].count)==2000
    norm=tree_norm(expected)
    for name in ('reset_adam_matched_norm','steepest_matched_norm'):assert dirs[name]['norm']==pytest.approx(norm)
    dot=sum(float(np.vdot(a,b)) for a,b in zip(jax.tree.leaves(g),jax.tree.leaves(dirs['steepest_matched_norm']['delta']),strict=True))
    assert dot<0


def test_zero_current_gradient_does_not_erase_saved_momentum():
    p,o,g=scalar_state();zero=jax.tree.map(jnp.zeros_like,g)
    dirs,v=adam_directions(p,o,zero,.001)
    assert dirs['stateful_adam']['norm']>0 and dirs['steepest_matched_norm']['delta'] is None and dirs['reset_adam_matched_norm']['delta'] is None
    dirs,v=adam_directions(p,optax.adam(.001).init(p),zero,.001)
    assert dirs['stateful_adam']['norm']==0 and v['next_count']==1


def test_quadratic_direction_slopes_and_both_fd_steps_detect_wrong_derivative():
    p={'x':jnp.array([.3,-.7])};ge={'x':2*p['x']};gf={'x':6*p['x']};d={'x':jnp.array([-.01,.02])}
    geometry=direction_geometry(ge,gf,d,{'energy':.25,'force':1.75})['whole']
    assert geometry['weighted_slope']==pytest.approx(.25*float(ge['x']@d['x'])+1.75*float(gf['x']@d['x']))
    evaluate=lambda pp:np.array([np.sum(np.asarray(pp['x'])**2),3*np.sum(np.asarray(pp['x'])**2)])
    fd=directional_fd(p,d,ge,gf,evaluate)
    assert fd['valid'] and all(len(v['steps'])==2 for v in fd['objectives'].values())
    wrong=directional_fd(p,d,jax.tree.map(lambda x:-x,ge),gf,evaluate)
    assert not wrong['valid'] and not wrong['objectives']['energy']['valid']
    assert directional_fd(p,jax.tree.map(jnp.zeros_like,d),ge,gf,evaluate)['status']=='not_applicable_zero_direction'


def test_snapshot_all_cells_probes_counts_and_immutable_real_mixed_derivatives():
    part,m,b,n,s=witness();before=_state_hashes(s);baseline=coverage_metrics(s['params'],part,part,m,3)
    c={'microbatch_size':3,'learning_rate':.001,'_baselines':baseline}
    saved=[];c['_save_partial']=lambda d:saved.append(copy.deepcopy(d))
    result=analyze_optimizer_snapshot(s,part,part,m,n,{'energy':.25,'force':1.75},c)
    assert result['verification_valid'] and result['state_hashes_before']==result['state_hashes_after']==before
    assert int(s['update'])==0 and int(s['optimizer'][0].count)==0
    assert result['counts']['fd_cells']==12 and result['counts']['fd_steps']==24 and result['counts']['probes']==6
    assert result['compute']['completed_frame_evaluations']['fd']==168
    assert result['compute']['completed_frame_evaluations']['probe']==84
    assert result['training_frame_exposures']==0 and saved
    assert all(op['status']=='completed' for op in result['compute']['operations'])
    assert result['weighted_gradient_identity']['valid'] and result['builder_identity']['valid']


def test_budget_retains_started_operation_without_invented_completed_work():
    part,m,b,n,s=witness();saved=[]
    def keep(d):saved.append(copy.deepcopy(d))
    def check():
        if saved:raise WallBudget('stop before numerical work')
    with pytest.raises(WallBudget):analyze_optimizer_snapshot(s,part,part,m,n,{'energy':0.,'force':2.},{'microbatch_size':3,'learning_rate':.001,'_baselines':{},'_save_partial':keep,'_check_budget':check})
    assert saved and not saved[-1]['compute']['operations']
