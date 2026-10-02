"""Global fitting-bias profiling. Imported only after production pure guards."""
import copy
from functools import lru_cache
import hashlib
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization
from flax.core import FrozenDict,freeze,unfreeze
from force_consistency.runs import _read_checkpoint,assert_finite,assert_parameter_dtype
from force_consistency.loss_balance import _weighted_gradient,validate_weights

def full_hash(state):
    return hashlib.sha256(serialization.to_bytes(state)).hexdigest()

def bias(tree):return tree['energy_head']['0+']['bias']

def set_bias(tree,value):
    result=unfreeze(tree) if isinstance(tree,FrozenDict) else copy.deepcopy(tree)
    result['energy_head']['0+']['bias']=jnp.asarray(value,dtype=jnp.float64).reshape(1)
    return freeze(result) if isinstance(tree,FrozenDict) else result

def project_bias(params,errors,atom_counts):
    e=np.asarray(errors,dtype=float);n=np.asarray(atom_counts)
    if (e.ndim!=1 or not e.size or n.shape!=e.shape or not np.isfinite(e).all()
            or not np.isfinite(n).all() or np.any(n<=0) or np.any(n!=n.astype(int)) or np.any(n!=n[0])):
        raise ValueError('Projection requires finite nonempty constant-atom support')
    b=np.asarray(bias(params))
    if b.shape!=(1,) or b.dtype!=np.dtype('float64') or not np.isfinite(b).all():
        raise ValueError('Invalid atomic head bias')
    mean=float(np.mean(e));delta=-mean/int(n[0]);post=e+n*delta
    result=set_bias(params,b+delta)
    return result,{'record_count':len(e),'atom_count':int(n[0]),'bias_before':float(b[0]),
        'bias_after':float(np.asarray(bias(result))[0]),'delta_bias':delta,
        'preprojection_mean_eV':mean,'preprojection_mse_eV2':float(np.mean(e**2)),
        'centered_variance_eV2':float(np.var(e)),'predicted_post_mean_eV':float(np.mean(post)),
        'predicted_post_mse_eV2':float(np.mean(post**2)),
        'post_statistics_scope':'Algebraic prediction from constant atomic-bias shift; not a fresh model evaluation'}

def restore_state(path,provenance):
    raw=serialization.msgpack_restore(_read_checkpoint(path,provenance)['payload'])
    if set(raw)!= {'params','optimizer','update','sampler_rng_json','model_rng','normalization_json'}:
        raise ValueError('Restored state fields differ')
    old=raw['optimizer']
    if set(old)!= {'0','1'} or set(old['0'])!= {'count','mu','nu'} or old['1']!={}:
        raise ValueError('Restored Adam tuple differs')
    convert=lambda t:jax.tree.map(jnp.asarray,t)
    state={**raw,'params':convert(raw['params']),'model_rng':jnp.asarray(raw['model_rng']),
        'optimizer':(optax.ScaleByAdamState(count=jnp.asarray(old['0']['count']),mu=convert(old['0']['mu']),nu=convert(old['0']['nu'])),optax.EmptyState())}
    assert_finite(state);assert_parameter_dtype(state['params'])
    return state

@lru_cache(maxsize=8)
def energy_kernel(model):
    return jax.jit(lambda p,b:model.apply({'params':p},b['atomic_numbers'],b['positions'],b['dst_idx'],b['src_idx'],b['batch_segments'],len(b['energy'])))

@lru_cache(maxsize=8)
def adam_kernel(rate):
    tx=optax.adam(rate)
    @jax.jit
    def apply(params,optimizer,gradient):
        updates,nextopt=tx.update(gradient,optimizer,params)
        return optax.apply_updates(params,updates),nextopt
    return apply

def validate_bias_moments(state):
    if any(np.any(np.asarray(bias(t))!=0) for t in (state['optimizer'][0].mu,state['optimizer'][0].nu)):
        raise ValueError('Nonzero energy bias moment')
    if int(state['optimizer'][0].count)!=int(state['update']):raise ValueError('Adam/update count differs')

def calibrate(params,batches,model,call,retain):
    errors=[];atoms=[];offset=0
    for batch in batches:
        first=offset
        def retained(e):retain({'event':'calibration_batch','first_index':first,
            'data':{'energy':e,'target':batch['energy']}})
        e=call('calibration_energy',len(batch['energy']),energy_kernel(model),params,batch,on_returned=retained)
        errors.append(np.asarray(e)-np.asarray(batch['energy']))
        atoms.append(np.bincount(np.asarray(batch['batch_segments']),minlength=len(batch['energy'])))
        offset+=len(batch['energy'])
    error=np.concatenate(errors);counts=np.concatenate(atoms)
    projected,receipt=project_bias(params,error,counts)
    retain({'event':'projection','data':{'errors':error,'atom_counts':counts},'receipt':receipt})
    if abs(receipt['predicted_post_mean_eV'])>1e-10:raise ValueError('Projection expected mean differs from zero')
    return projected,receipt

def prepare(state,batches,model,norm,weights,rate,check):
    coeff=validate_weights(weights);seen=set()
    for batch in batches:
        if len(batch['energy']) not in seen:
            check();_weighted_gradient(model,norm,coeff).lower(state['params'],batch).compile()
            check();energy_kernel(model).lower(state['params'],batch).compile();seen.add(len(batch['energy']))
    check();adam_kernel(rate).lower(state['params'],state['optimizer'],jax.tree.map(jnp.zeros_like,state['params'])).compile();check()

def profile_step(state,batches,model,norm,weights,rate,call,retain):
    validate_bias_moments(state);assert_finite(state);before=full_hash(state)
    if not batches or any(not len(b['energy']) for b in batches):raise ValueError('Empty profile microbatch')
    count=sum(len(b['energy']) for b in batches);gradient=None;value=0.;metrics={'energy_mse_eV2':0.,'force_mse_eV2_per_A2':0.}
    coeff=validate_weights(weights);offset=0
    for batch in batches:
        first=offset
        result=call('gradient',len(batch['energy']),_weighted_gradient(model,norm,coeff),state['params'],batch,
            on_returned=lambda returned:retain({'event':'gradient_batch','first_index':first,'data':returned}))
        (loss,rawmetrics),g=result;weight=len(batch['energy'])/count;value+=float(loss)*weight
        for key in metrics:metrics[key]+=float(rawmetrics[key])*weight
        piece=jax.tree.map(lambda x:x*weight,g)
        gradient=piece if gradient is None else jax.tree.map(lambda x,y:x+y,gradient,piece)
        retain({'event':'partial_gradient','first_index':first,'data':gradient})
        offset+=len(batch['energy'])
    unmasked=float(np.asarray(bias(gradient))[0]);retain({'event':'gradient_total','data':gradient})
    if abs(unmasked)>1e-10:raise ValueError('Incoming fitting bias gradient is not at calibrated optimum')
    gradient=set_bias(gradient,[0.])
    pp,oo=call('adam_proposal',0,adam_kernel(rate),state['params'],state['optimizer'],gradient,
        on_returned=lambda returned:retain({'event':'adam_candidate','data':returned}))
    if not np.array_equal(np.asarray(bias(pp)),np.asarray(bias(state['params']))):raise ValueError('Adam changed masked bias')
    projected,projection=calibrate(pp,batches,model,call,retain)
    candidate={**state,'params':projected,'optimizer':oo,'update':int(state['update'])+1}
    validate_bias_moments(candidate);assert_finite(candidate);assert_parameter_dtype(candidate['params'])
    if before!=full_hash(state):raise ValueError('Profile step mutated incoming state')
    return candidate,gradient,{'incoming_update':int(state['update']),'outgoing_update':int(candidate['update']),
        'incoming_full_hash':before,'outgoing_full_hash':full_hash(candidate),
        'unmasked_weighted_bias_gradient':unmasked,'scaled_weighted_loss':value,'incoming_raw_metrics':metrics,
        'projection':projection}

@lru_cache(maxsize=8)
def reference_gradient(model,norm,coeff):
    from force_consistency.models.energy import energy_and_forces
    a,b=coeff
    def objective(p,batch):
        e,f=energy_and_forces(p,batch,model=model);r=e-batch['energy']
        return a*jnp.mean((r-jnp.mean(r))**2)/norm.energy_loss_scale_eV**2+b*jnp.mean((f-batch['forces'])**2)/norm.force_loss_scale_eV_per_A**2
    return jax.jit(jax.value_and_grad(objective))

def profile_oracle(incoming,candidate,actual_gradient,fullbatch,model,norm,weights,rate,call,retain):
    from force_consistency.coverage_runs import _predict
    from force_consistency.width_rate_runs import independent_adam_step,_trees_close
    before=full_hash(incoming);n=len(fullbatch['energy'])
    value,ref=call('oracle_gradient',n,reference_gradient(model,norm,validate_weights(weights)),incoming['params'],fullbatch,
        on_returned=lambda returned:retain({'event':'oracle_gradient','data':returned}))
    ref=set_bias(ref,[0.]);gradient_error=_trees_close(actual_gradient,ref)
    rp,ro=call('oracle_numpy_adam',0,independent_adam_step,incoming['params'],incoming['optimizer'],ref,rate,
        on_returned=lambda returned:retain({'event':'oracle_numpy_adam','data':returned}))
    re,rf=call('oracle_prediction_reference',n,_predict(model),rp,fullbatch,
        on_returned=lambda returned:retain({'event':'oracle_prediction_reference','data':returned}))
    atoms=np.bincount(np.asarray(fullbatch['batch_segments']),minlength=n)
    if not n or np.any(atoms!=atoms[0]) or atoms[0]<=0:raise ValueError('Oracle needs constant atom count')
    error=np.asarray(re)-np.asarray(fullbatch['energy']);mean=float(np.mean(error));delta=-mean/int(atoms[0])
    rp=set_bias(rp,np.asarray(bias(rp))+delta)
    corrected=np.asarray(re)+atoms*delta
    state_error=_trees_close((candidate['params'],candidate['optimizer']),(rp,ro))
    ae,af=call('oracle_prediction_actual',n,_predict(model),candidate['params'],fullbatch,
        on_returned=lambda returned:retain({'event':'oracle_prediction_actual','data':returned}))
    prediction_error=_trees_close((ae,af),(corrected,rf))
    retain({'event':'oracle_complete','data':{'reference_gradient':ref,'actual_gradient':actual_gradient,
        'reference_params':rp,'reference_optimizer':ro,'reference_energy_raw':re,'reference_energy_adjusted':corrected,
        'reference_forces':rf,'actual_energy':ae,'actual_forces':af},
        'receipt':{'reference_bias_delta':delta,'reference_raw_mean_eV':mean}})
    if before!=full_hash(incoming):raise ValueError('Oracle mutated incoming state')
    return {'status':'passed','incoming_update':int(incoming['update']),'outgoing_update':int(candidate['update']),
        'incoming_nonzero_moments':any(np.any(np.asarray(x)!=0) for x in jax.tree.leaves(incoming['optimizer'][0].mu)),
        'gradient_maximum_error':gradient_error,'state_maximum_error':state_error,'prediction_maximum_error':prediction_error,
        'independent_scaled_profiled_loss':float(value),'atol':1e-10,'rtol':1e-8,
        'incoming_full_hash':before,'candidate_full_hash':full_hash(candidate),'reference_proposal_discarded':True}
