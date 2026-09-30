import copy
from functools import partial
import json
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from test_coverage import seven_case,close_trees,arm_config
from test_gradient_diagnostics import witness
from force_consistency.loss_balance import validate_weights,weighted_loss,accumulated_weighted_loss_and_grad,build_weighted_step
from force_consistency.coverage_runs import _state_hashes,build_accumulated_step,run_coverage_arm
from force_consistency.models.energy import energy_and_forces
from force_consistency.metrics import partition_batch


def oracle(params,batch,model,norm,a,b):
    e,f=energy_and_forces(params,batch,model=model)
    em=jnp.sum((e-batch['energy'])**2)/batch['energy'].size
    fm=jnp.sum((f-batch['forces'])**2)/batch['forces'].size
    return a*em/norm.energy_loss_scale_eV**2+b*fm/norm.force_loss_scale_eV_per_A**2


@pytest.mark.parametrize('weights',[{'energy':-1,'force':3},{'energy':0,'force':0},{'energy':float('nan'),'force':1},{'energy':1,'force':float('inf')}])
def test_invalid_weighted_objectives_reject(weights):
    with pytest.raises(ValueError):validate_weights(weights)


def test_weighted_uneven_chunks_derivative_and_single_adam_application():
    p,m,b,n,s=witness();before=_state_hashes(s);weights={'energy':.25,'force':1.75}
    batches=[partition_batch(p,np.arange(i,min(i+3,7))) for i in (0,3,6)]
    (value,metrics),grad=accumulated_weighted_loss_and_grad(s['params'],batches,m,n,weights)
    want,wg=jax.jit(jax.value_and_grad(lambda pp:oracle(pp,b,m,n,.25,1.75)))(s['params'])
    assert float(value)==pytest.approx(float(want),abs=1e-10);close_trees(grad,wg)
    step=build_weighted_step(m,n,.001,weights);step.prepare(s['params'],s['optimizer'],batches)
    pp,oo,mm=step(s['params'],s['optimizer'],batches)
    tx=optax.adam(.001);up,wo=tx.update(wg,s['optimizer'],s['params'])
    close_trees(pp,optax.apply_updates(s['params'],up));close_trees(oo,wo)
    assert int(oo[0].count)==1 and _state_hashes(s)==before
    assert mm['scaled_loss']==pytest.approx(float(want))


def test_force_only_mixed_derivatives_and_no_bias_update():
    p,m,b,n,s=witness();w={'energy':0.,'force':2.}
    (v,_),g=jax.value_and_grad(lambda pp:weighted_loss(pp,b,m,n,w),has_aux=True)(s['params'])
    assert sum(float(jnp.linalg.norm(x)) for x in jax.tree.leaves(g['Backbone_0']))>1e-6
    assert np.asarray(g['energy_head']['0+']['bias'])[0]==0
    pp,oo,_=build_weighted_step(m,n,.001,w)(s['params'],s['optimizer'],[b])
    np.testing.assert_array_equal(pp['energy_head']['0+']['bias'],s['params']['energy_head']['0+']['bias'])
    assert int(oo[0].count)==1


def test_one_one_step_exactly_preserves_baseline():
    p,m,b,n,s=seven_case();chunks=[partition_batch(p,np.arange(i,min(i+3,7))) for i in (0,3,6)]
    baseline=build_accumulated_step(m,n,.001);weighted=build_weighted_step(m,n,.001,{'energy':1.,'force':1.})
    a=baseline(s['params'],s['optimizer'],chunks);w=weighted(s['params'],s['optimizer'],chunks)
    for aa,ww in zip(jax.tree.leaves(a),jax.tree.leaves(w),strict=True):np.testing.assert_array_equal(aa,ww)


def test_actual_injection_matches_weighted_step_and_checkpoint(tmp_path):
    p,m,b,n,s=seven_case();weights={'energy':.25,'force':1.75};c={**arm_config(n),'loss_weights':weights}
    factory=partial(build_weighted_step,weights=weights)
    result=run_coverage_arm(c,p,p,tmp_path/'weighted',{},step_factory=factory)
    assert result['execution_valid'] and result['updates']==2 and result['checkpoint_new_process_verified']
    from force_consistency.runs import load_checkpoint
    from force_consistency.train import initialize_state
    prov=json.loads((tmp_path/'weighted/provenance.json').read_text())
    initial=initialize_state(m,partition_batch(p,np.arange(3)),n,0,.001)
    state=load_checkpoint(tmp_path/'weighted/last.msgpack',prov,template=initial)
    chunks=[partition_batch(p,np.arange(i,min(i+3,7))) for i in (0,3,6)];step=factory(m,n,.001)
    pp,oo=initial['params'],initial['optimizer']
    for _ in range(2):pp,oo,_=step(pp,oo,chunks)
    close_trees(state['params'],pp);close_trees(state['optimizer'],oo)
    assert json.loads((tmp_path/'weighted/config.json').read_text())['loss_weights']==weights


def test_builder_freezes_weights_against_caller_mutation():
    p,m,b,n,s=seven_case();weights={'energy':.25,'force':1.75}
    step=build_weighted_step(m,n,.001,weights);weights.update(energy=2.,force=0.)
    _,_,metrics=step(s['params'],s['optimizer'],[b])
    assert metrics['scaled_loss']==pytest.approx(float(oracle(s['params'],b,m,n,.25,1.75)),abs=1e-10)
