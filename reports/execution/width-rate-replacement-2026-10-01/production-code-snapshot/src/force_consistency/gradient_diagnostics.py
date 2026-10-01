"""Frozen, scaled-objective parameter derivatives; no optimizer updates."""
from functools import lru_cache
import jax
import jax.numpy as jnp
import numpy as np
from .metrics import partition_batch
from .models.energy import energy_and_forces
from .coverage_runs import accumulated_loss_and_grad,_state_hashes
from .runs import assert_finite


@lru_cache(maxsize=8)
def _objectives(model,normalization):
    @jax.jit
    def values(params,batch):
        e,f=energy_and_forces(params,batch,model=model)
        return jnp.array([jnp.mean((e-batch['energy'])**2)/normalization.energy_loss_scale_eV**2,
                          jnp.mean((f-batch['forces'])**2)/normalization.force_loss_scale_eV_per_A**2])
    return values,tuple(jax.jit(jax.value_and_grad(lambda p,b,i=i:values(p,b)[i])) for i in range(2))


def _batches(partition,size):
    if not isinstance(size,int) or size<=0 or not partition.record_ids:
        raise ValueError('invalid diagnostic microbatches')
    return [partition_batch(partition,np.arange(i,min(i+size,len(partition.record_ids))))
            for i in range(0,len(partition.record_ids),size)]


def split_gradients(params,partition,model,normalization,microbatch_size=16,check_budget=lambda:None):
    batches=_batches(partition,microbatch_size);total=len(partition.record_ids)
    _,functions=_objectives(model,normalization);results=[]
    for fn in functions:
        combined=None
        for batch in batches:
            check_budget()
            result=fn(params,batch);jax.block_until_ready(result);assert_finite(result)
            weighted=jax.tree.map(lambda x:x*len(batch['energy'])/total,result)
            combined=weighted if combined is None else jax.tree.map(lambda a,b:a+b,combined,weighted)
        results.append(combined)
    check_budget()
    (joint_value,_),joint=accumulated_loss_and_grad(params,batches,model,normalization)
    assert_finite((joint_value,joint))
    return {'energy':float(results[0][0]),'force':float(results[1][0]),'joint':float(joint_value)},results[0][1],results[1][1],joint


def _geometry(e,f):
    el=[np.asarray(v,dtype=float).ravel() for v in jax.tree.leaves(e)];fl=[np.asarray(v,dtype=float).ravel() for v in jax.tree.leaves(f)]
    if len(el)!=len(fl) or any(a.shape!=b.shape for a,b in zip(el,fl)):
        raise ValueError('gradient trees differ')
    en2=sum(float(a@a) for a in el);fn2=sum(float(b@b) for b in fl);dot=sum(float(a@b) for a,b in zip(el,fl))
    en=en2**.5;fn=fn2**.5
    return {'parameter_count':sum(a.size for a in el),'energy_norm':en,'force_norm':fn,'dot':dot,
            'cosine':dot/(en*fn) if en and fn else None,'force_over_energy_norm':fn/en if en else None,
            'energy_raw_joint_descent_slope':-(en2+dot),'force_raw_joint_descent_slope':-(fn2+dot)}


def gradient_geometry(energy_grad,force_grad):
    assert_finite((energy_grad,force_grad))
    if jax.tree.structure(energy_grad)!=jax.tree.structure(force_grad):raise ValueError('gradient trees differ')
    result={'whole':_geometry(energy_grad,force_grad)}
    if 'Backbone_0' in energy_grad:result['backbone']=_geometry(energy_grad['Backbone_0'],force_grad['Backbone_0'])
    if 'energy_head' in energy_grad:
        for block,key in [('head_kernel','kernel'),('head_bias','bias')]:
            result[block]=_geometry(energy_grad['energy_head']['0+'][key],force_grad['energy_head']['0+'][key])
    epath,_=jax.tree_util.tree_flatten_with_path(energy_grad);fpath,_=jax.tree_util.tree_flatten_with_path(force_grad)
    result['leaves']={jax.tree_util.keystr(path):_geometry(a,b) for (path,a),(_,b) in zip(epath,fpath,strict=True)}
    return result


def verify_directional_derivatives(params,partition,model,normalization,energy_grad,force_grad,
                                   steps=(1e-4,1e-5),seed=20260930,microbatch_size=16,
                                   zero_tolerance=1e-12,atol=1e-8,rtol=1e-4,check_budget=lambda:None,on_progress=lambda d:None):
    batches=_batches(partition,microbatch_size);values,_=_objectives(model,normalization)
    total=len(partition.record_ids);rng=np.random.default_rng(seed);result={}
    def evaluate(pp):
        out=np.zeros(2)
        for b in batches:
            check_budget()
            out+=np.asarray(values(pp,b))*len(b['energy'])/total
        if not np.isfinite(out).all():raise ValueError('nonfinite FD objective')
        return out
    for i,(name,grad) in enumerate([('energy',energy_grad),('force',force_grad)]):
        norm=_geometry(grad,grad)['energy_norm']
        if norm>zero_tolerance:
            direction=jax.tree.map(lambda x:x/norm,grad);kind='unit_gradient'
        else:
            raw=jax.tree.map(lambda x:jnp.asarray(rng.choice([-1.,1.],size=x.shape)),params)
            size=sum(x.size for x in jax.tree.leaves(raw));direction=jax.tree.map(lambda x:x/np.sqrt(size),raw);kind='seeded_rademacher'
        autodiff=sum(float(np.vdot(a,b)) for a,b in zip(jax.tree.leaves(grad),jax.tree.leaves(direction),strict=True))
        records=[]
        result[name]={'direction_kind':kind,'autodiff_directional_derivative':autodiff,'steps':records,'status':'running'}
        for step in steps:
            check_budget()
            if not np.isfinite(step) or step<=0:raise ValueError('invalid FD step')
            plus=jax.tree.map(lambda p,d:p+step*d,params,direction);minus=jax.tree.map(lambda p,d:p-step*d,params,direction)
            fd=float((evaluate(plus)[i]-evaluate(minus)[i])/(2*step))
            records.append({'step':float(step),'finite_difference':fd,'absolute_error':abs(fd-autodiff),
                            'valid':bool(np.isclose(fd,autodiff,atol=atol,rtol=rtol))})
            on_progress(result)
        result[name]={'direction_kind':kind,'autodiff_directional_derivative':autodiff,'steps':records,
                      'valid':any(r['valid'] for r in records)}
    result['valid']=all(result[k]['valid'] for k in ('energy','force'))
    return result


def analyze_snapshot(state,partition,model,normalization,config):
    before=_state_hashes(state);size=config.get('microbatch_size',16)
    check=config.get('_check_budget',lambda:None);save=config.get('_save_partial',lambda d:None)
    losses,ge,gf,gj=split_gradients(state['params'],partition,model,normalization,size,check)
    atol=config.get('gradient_identity_atol',1e-10);rtol=config.get('gradient_identity_rtol',1e-8)
    deltas=jax.tree.map(lambda a,b,c:c-a-b,ge,gf,gj)
    identity=all(np.allclose(c,np.asarray(a)+np.asarray(b),atol=atol,rtol=rtol)
                 for a,b,c in zip(jax.tree.leaves(ge),jax.tree.leaves(gf),jax.tree.leaves(gj),strict=True))
    partial={'losses':losses,'geometry':gradient_geometry(ge,gf),'joint_gradient_identity':{'valid':bool(identity),
                'max_absolute_difference':max(float(np.max(abs(x))) for x in jax.tree.leaves(deltas))},
             'record_ids':list(partition.record_ids),'state_hashes_before':before}
    save(partial);check()
    def progress(d):
        partial['finite_differences']=d;save(partial)
    fd=verify_directional_derivatives(state['params'],partition,model,normalization,ge,gf,
        steps=tuple(config.get('fd_steps',(1e-4,1e-5))),seed=config.get('direction_seed',20260930),microbatch_size=size,
        zero_tolerance=config.get('direction_zero_tolerance',1e-12),atol=config.get('fd_atol',1e-8),rtol=config.get('fd_rtol',1e-4),check_budget=check,on_progress=progress)
    after=_state_hashes(state)
    return {'record_ids':list(partition.record_ids),'record_count':len(partition.record_ids),
            'force_component_count':int(partition.forces.size),'update':int(state['update']),'losses':losses,
            'geometry':gradient_geometry(ge,gf),'joint_gradient_identity':{'valid':bool(identity),
                'max_absolute_difference':max(float(np.max(abs(x))) for x in jax.tree.leaves(deltas))},
            'finite_differences':fd,'state_hashes_before':before,'state_hashes_after':after,
            'verification_valid':bool(identity and fd['valid'] and before==after)}
