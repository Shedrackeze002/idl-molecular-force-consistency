"""Isolated fixed-weight objectives; the baseline loss and default updates stay intact."""
from functools import lru_cache
import jax
import numpy as np
import optax
from .train import normalized_loss
from .coverage_runs import accumulated_loss_and_grad,build_accumulated_step


def validate_weights(weights):
    if set(weights)!= {'energy','force'}:raise ValueError('Weights require energy and force')
    a,b=float(weights['energy']),float(weights['force'])
    if not np.isfinite([a,b]).all() or min(a,b)<0 or a+b<=0:
        raise ValueError('Weights must be finite nonnegative and not both zero')
    return a,b


def weighted_loss(params,batch,model,normalization,weights):
    a,b=validate_weights(weights)
    value,metrics=normalized_loss(params,batch,model,normalization)
    if (a,b)==(1.,1.):return value,metrics
    return (a*metrics['energy_mse_eV2']/normalization.energy_loss_scale_eV**2+
            b*metrics['force_mse_eV2_per_A2']/normalization.force_loss_scale_eV_per_A**2),metrics


@lru_cache(maxsize=16)
def _weighted_gradient(model,normalization,coefficients):
    a,b=coefficients
    return jax.jit(jax.value_and_grad(lambda p,batch:weighted_loss(p,batch,model,normalization,{'energy':a,'force':b}),has_aux=True))


def accumulated_weighted_loss_and_grad(params,batches,model,normalization,weights):
    coefficients=validate_weights(weights)
    if coefficients==(1.,1.):return accumulated_loss_and_grad(params,batches,model,normalization)
    if not batches or any(len(b['energy'])==0 for b in batches):raise ValueError('Empty weighted microbatch')
    total=sum(len(b['energy']) for b in batches);combined=None
    for batch in batches:
        result=_weighted_gradient(model,normalization,coefficients)(params,batch);jax.block_until_ready(result)
        weighted=jax.tree.map(lambda x:x*len(batch['energy'])/total,result)
        combined=weighted if combined is None else jax.tree.map(lambda a,b:a+b,combined,weighted)
    return combined


def build_weighted_step(model,normalization,learning_rate,weights):
    coefficients=validate_weights(weights)
    if coefficients==(1.,1.):return build_accumulated_step(model,normalization,learning_rate)
    weights={'energy':coefficients[0],'force':coefficients[1]}
    tx=optax.adam(learning_rate)
    @jax.jit
    def apply(params,opt,gradient):
        updates,nextopt=tx.update(gradient,opt,params)
        return optax.apply_updates(params,updates),nextopt
    def prepare(params,opt,batches):
        sizes=set()
        for batch in batches:
            if len(batch['energy']) not in sizes:
                _weighted_gradient(model,normalization,coefficients).lower(params,batch).compile()
                sizes.add(len(batch['energy']))
        apply.lower(params,opt,jax.tree.map(lambda x:x*0,params)).compile()
    def step(params,opt,batches):
        (value,metrics),gradient=accumulated_weighted_loss_and_grad(params,batches,model,normalization,weights)
        pp,oo=apply(params,opt,gradient)
        return pp,oo,{**metrics,'scaled_loss':value}
    step.prepare=prepare
    return step
