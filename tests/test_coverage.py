from dataclasses import replace
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from force_consistency.coverage import select_coverage_subsets, verify_coverage_subsets, take_subset
from force_consistency.data.prepare import load_training_partition


def frozen_case():
    planned=json.loads(Path('manifests/aspirin-data-coverage-subsets-planned.json').read_text())
    p,m,n=load_training_partition('data/processed/aspirin',expected_split_hash=planned['split_hash'])
    return p,m,n,planned


def test_exact_nested_selection_and_rank_distribution():
    p,m,n,planned=frozen_case()
    anchor=planned['subsets']['narrow16']['record_ids']
    selected=select_coverage_subsets(p,m,anchor)
    verify_coverage_subsets(selected,planned)
    assert selected['ordered_groups']==['01','23','77','58_2','44','22','78_2','01_2']
    a,b,c=[selected['subsets'][k]['record_ids'] for k in ('narrow16','diverse16','diverse128')]
    assert [len(set(x)) for x in (a,b,c)]==[16,16,128]
    assert set(a)<=set(c) and set(b)<=set(c) and len(set(a)&set(b))==2
    assert selected['selection']['01_2']['minus']['ranks']==[0,15,31,46,63,78,94,109]
    assert selected['selection']['01_2']['plus']['ranks']==[1,14,27,40,54,67,80,94]
    for g in selected['ordered_groups']:
        assert sum(p.memberships[r]['groups']==[g] for r in b)==2
        assert sum(p.memberships[r]['groups']==[g] for r in c)==16
    labels=replace(p,energy_residual=np.full_like(p.energy_residual,999.),forces=-777*p.forces)
    assert select_coverage_subsets(labels,m,anchor)==selected
    subset=take_subset(p,b)
    assert subset.record_ids==tuple(b) and not subset.positions.flags.writeable
    bad=copy.deepcopy(planned);bad['subsets']['diverse16']['identity']='stale'
    with pytest.raises(ValueError): verify_coverage_subsets(selected,bad)


def test_train_only_source_and_rank_failures(monkeypatch):
    original=np.load
    def guarded(path,*args,**kw):
        assert Path(path).name=='train.npz'
        return original(path,*args,**kw)
    monkeypatch.setattr(np,'load',guarded)
    p,m,n,planned=frozen_case();anchor=planned['subsets']['narrow16']['record_ids']
    selected=select_coverage_subsets(p,m,anchor)
    # Every malformed selection fails rather than choosing replacement support.
    for role,groups in [('minimum',['01']),('path',['heldout']),('path',['01','23'])]:
        memberships=copy.deepcopy(p.memberships);memberships[anchor[0]]={'role':role,'groups':groups}
        with pytest.raises(ValueError): select_coverage_subsets(replace(p,memberships=memberships),m,anchor)
    with pytest.raises(ValueError): select_coverage_subsets(p,m,[anchor[0]]*16)
    with pytest.raises(ValueError): take_subset(p,[anchor[0],anchor[0]])
    with pytest.raises(ValueError): take_subset(p,['outside'])
    with pytest.raises(ValueError): select_coverage_subsets(replace(p,record_ids=(*p.record_ids[:-1],p.record_ids[0])),m,anchor)
    keep=[r for r in p.record_ids if '/23/' not in r or r.endswith('000.xyz')]
    with pytest.raises(ValueError,match='rank|pool'): select_coverage_subsets(take_subset(p,keep),m,anchor)
    from force_consistency.coverage import numeric_path
    assert numeric_path('Landscape17/aspirin/SteepestDescent/01/coordsminus_2.xyz')[-1]==2
    assert numeric_path('Landscape17/aspirin/SteepestDescent/01/coordsminus_10.xyz')[-1]==10


def seven_case():
    from force_consistency.models.energy import EnergyModel,make_batch
    from force_consistency.models.backbone import ModelConfig
    from force_consistency.data.prepare import Normalization,PreparedPartition
    from force_consistency.train import initialize_state
    positions=np.tile(np.array([[0.,0.,0.],[.7,.1,0.],[.2,.8,.1]]),(7,1,1))
    positions[:,1,0]+=np.arange(7)*.07;positions[:,2,1]+=np.arange(7)*.025
    energy=np.linspace(-.1,.3,7);forces=np.arange(63).reshape(7,3,3)*.0007-.02
    ids=tuple(f'tr{i}' for i in range(7));z=np.tile([1,6,8],(7,1))
    p=PreparedPartition(positions,energy,forces,z,ids,{r:{'role':'path','groups':['a' if i<3 else 'b']} for i,r in enumerate(ids)})
    model=EnergyModel(ModelConfig(features=4,iterations=1,radial_basis=4,energy_readout_init='zero'))
    norm=Normalization(0.,.2,.1);b=make_batch(positions,z,energy,forces)
    return p,model,b,norm,initialize_state(model,b,norm,0,.001)


def close_trees(left,right):
    import jax
    for a,b in zip(jax.tree.leaves(left),jax.tree.leaves(right),strict=True):
        np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-10)


def test_accumulated_step_matches_full_batch():
    import jax
    import optax
    from force_consistency.train import normalized_loss,build_step
    from force_consistency.coverage_runs import accumulated_loss_and_grad,build_accumulated_step
    from force_consistency.metrics import partition_batch
    p,model,b,norm,state=seven_case();params=state['params'];opt=state['optimizer']
    chunks=[partition_batch(p,np.arange(i,min(i+3,7))) for i in (0,3,6)]
    full_grad=jax.jit(jax.value_and_grad(lambda pp,bb:normalized_loss(pp,bb,model,norm),has_aux=True))
    accumulated=build_accumulated_step(model,norm,.001);full=build_step(model,norm,.001)
    for _ in range(2):
        expected=full_grad(params,b)
        actual=accumulated_loss_and_grad(params,chunks,model,norm)
        close_trees(actual,expected)
        ap,ao,am=accumulated(params,opt,chunks);fp,fo,fm=full(params,opt,b)
        close_trees((ap,ao,am),(fp,fo,fm))
        assert int(ao[0].count)==int(opt[0].count)+1
        params,opt=fp,fo
    assert any(np.linalg.norm(x)>0 for x in jax.tree.leaves(actual[1]))


def test_existing_step_objective_unchanged():
    import jax
    import jax.numpy as jnp
    import optax
    from force_consistency.train import normalized_loss,build_step
    from force_consistency.models.energy import energy_and_forces
    p,model,b,norm,state=seven_case()
    def old_loss(pp):
        e,f=energy_and_forces(pp,b,model=model)
        em=jnp.mean((e-b['energy'])**2);fm=jnp.mean((f-b['forces'])**2)
        return em/norm.energy_loss_scale_eV**2+fm/norm.force_loss_scale_eV_per_A**2,{'energy_mse_eV2':em,'force_mse_eV2_per_A2':fm}
    (value,metrics),grad=jax.value_and_grad(old_loss,has_aux=True)(state['params'])
    close_trees(jax.value_and_grad(lambda pp:normalized_loss(pp,b,model,norm),has_aux=True)(state['params']),((value,metrics),grad))
    tx=optax.adam(.001);updates,opt=tx.update(grad,state['optimizer'],state['params'])
    pp,oo,mm=build_step(model,norm,.001)(state['params'],state['optimizer'],b)
    close_trees((pp,oo,mm),(optax.apply_updates(state['params'],updates),opt,{**metrics,'scaled_loss':value}))
    # Force mixed derivatives with a nonzero energy head must reach the backbone.
    first=pp
    fg=jax.grad(lambda x:normalized_loss(x,b,model,norm)[1]['force_mse_eV2_per_A2'])(first)
    assert sum(float(jnp.linalg.norm(x)) for x in jax.tree.leaves(fg['Backbone_0']))>0


def test_exhaustive_metrics_counts_baselines_and_overlap():
    from force_consistency.coverage_runs import coverage_metrics,gate_result
    p,model,b,norm,state=seven_case();fit=take_subset(p,p.record_ids[:3])
    result=coverage_metrics(state['params'],fit,p,model,microbatch_size=2)
    assert result['fitting']['record_count']==3 and result['anchor']['record_count']==7
    assert result['anchor']['force_component_count']==63
    assert result['anchor_fitted']['record_count']==3 and result['anchor_not_fitted']['record_count']==4
    assert result['anchor']['record_ids']==list(p.record_ids)
    assert result['anchor']['energy_mse_eV2']==pytest.approx(np.mean(p.energy_residual**2))
    assert result['anchor']['force_mse_eV2_per_A2']==pytest.approx(np.mean(p.forces**2))
    assert result['anchor']['equal_group']['energy_mse_eV2']==pytest.approx((np.mean(p.energy_residual[:3]**2)+np.mean(p.energy_residual[3:]**2))/2)
    empty=coverage_metrics(state['params'],p,p,model,2)['anchor_not_fitted']
    assert empty['record_count']==0 and empty['energy_mse_eV2'] is None
    assert not gate_result(empty,empty,.1)['accepted']
    current={'energy_mse_eV2':.005,'force_mse_eV2_per_A2':.002}
    initial={'energy_mse_eV2':.1,'force_mse_eV2_per_A2':.01}
    assert gate_result(current,initial,.1)['fraction_remaining']['energy_mse_eV2']==pytest.approx(.05)
    assert not gate_result(current,initial,.1)['accepted']


def arm_config(norm):
    from dataclasses import asdict
    return {'backend':'cpu','seed':0,'learning_rate':.001,'validation_interval':1,'max_updates':2,'wall_seconds':120.,'gate_fraction':.1,
            'microbatch_size':3,'early_stopping':False,'normalization':asdict(norm),
            'model':{'features':4,'iterations':1,'radial_basis':4,'energy_readout_init':'zero'}}


def test_arm_fixed_cap_probe_baselines_and_no_resume(tmp_path):
    from force_consistency.coverage_runs import run_coverage_arm
    p,model,b,norm,state=seven_case();config=arm_config(norm)
    fit=take_subset(p,p.record_ids[:3]);root=tmp_path/'run'
    result=run_coverage_arm(config,fit,p,root,{'engineering_fixture':True})
    assert result['execution_valid'] and result['updates']==2 and result['stop_reason']=='max_updates'
    assert result['checkpoint_new_process_verified'] and result['frame_exposures']==6
    assert [r['update'] for r in result['history']]==[0,1,2]
    assert result['initial_metrics']['anchor']['force_mse_eV2_per_A2']==pytest.approx(np.mean(p.forces**2))
    saved=json.loads((root/'checkpoint-verification.json').read_text())
    assert saved['fitting_record_count']==3 and saved['anchor_record_count']==7
    assert saved['maximum_absolute_error']<=1e-10
    # Fresh initialization must match even with a different fitting shape.
    other=run_coverage_arm({**config,'max_updates':1},p,p,tmp_path/'other',{'expected_initial_hashes':result['initial_state_hashes']})
    assert other['initial_state_hashes']==result['initial_state_hashes'] and other['execution_valid']
    with pytest.raises(ValueError,match='not empty'): run_coverage_arm(config,fit,p,root,{})
    invalid=run_coverage_arm({**config,'resume_checkpoint':'forbidden'},fit,p,tmp_path/'resume',{})
    assert invalid['status']=='failed' and invalid['updates']==0


def test_wall_and_nonfinite_failures_keep_last_finite_state(tmp_path,monkeypatch):
    import force_consistency.coverage_runs as cr
    import jax
    import jax.numpy as jnp
    p,model,b,norm,state=seven_case();config=arm_config(norm)
    wall=cr.run_coverage_arm({**config,'wall_seconds':.001},p,p,tmp_path/'wall',{})
    assert not wall['execution_valid'] and wall['status']=='incomplete' and wall['stop_reason']=='wall_budget' and wall['updates']==0
    def bad(params,opt,batches):
        return jax.tree.map(lambda x:x*jnp.nan,params),opt,{'scaled_loss':jnp.array(np.nan)}
    monkeypatch.setattr(cr,'build_accumulated_step',lambda *a:bad)
    failed=cr.run_coverage_arm(config,p,p,tmp_path/'bad',{})
    assert failed['status']=='failed' and failed['updates']==0 and 'Nonfinite' in failed['failure_reason']
    assert json.loads((tmp_path/'bad/history.json').read_text())[0]['update']==0


def test_learning_gate_never_shortens_fixed_cap(tmp_path):
    from force_consistency.coverage_runs import run_coverage_arm
    p,model,b,norm,state=seven_case()
    from force_consistency.diagnostics import with_head
    from force_consistency.models.energy import energy_and_forces
    teacher=with_head(state['params'],np.array([.01,-.02,.03,.01,.005]))
    e,f=energy_and_forces(teacher,b,model=model)
    p=replace(p,energy_residual=np.asarray(e),forces=np.asarray(f).reshape(p.forces.shape))
    c={**arm_config(norm),'max_updates':10,'gate_fraction':.99}
    result=run_coverage_arm(c,p,p,tmp_path/'easy',{})
    assert result['updates']==10 and len(result['history'])==11
    assert result['first_observed_fit_passing_scheduled_update']<10
    assert result['initial_metrics']==result['history'][0]['metrics']
