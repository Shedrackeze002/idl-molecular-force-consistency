"""Bounded training-only probes; never baseline or held-out evaluation."""
from copy import deepcopy
from dataclasses import asdict,replace
from pathlib import Path
import hashlib
import json
import subprocess
import sys
import time

import jax
import jax.numpy as jnp
import numpy as np

from .data.prepare import (PreparedPartition,Normalization,array_identity,
                           load_training_partition,write_json)
from .data.splits import canonical_hash
from .models.backbone import Backbone,ModelConfig
from .models.energy import EnergyModel,make_batch,energy_and_forces
from .runs import assert_finite,code_provenance,load_checkpoint,require_backend,save_checkpoint
from .train import initialize_state,build_step


def verified_subset(partition,manifest,source,expected_hash):
    ids=source['record_ids'];indices=source['indices']
    if canonical_hash(ids)!=expected_hash or source['identity']!=expected_hash or len(set(ids))!=len(ids):
        raise ValueError('Subset identity mismatch')
    if len(indices)!=len(ids) or any(not isinstance(i,int) or i<0 or i>=len(partition.record_ids) for i in indices):
        raise ValueError('Subset indices mismatch')
    if [partition.record_ids[i] for i in indices]!=ids: raise ValueError('Subset indices mismatch')
    groups=manifest['splits']['groups'][manifest['molecule']]['train']
    if any(partition.memberships[r]['role']!='path' or not partition.memberships[r]['seen_in_training'] or not set(partition.memberships[r]['groups']).issubset(groups) for r in ids):
        raise ValueError('Subset training path membership mismatch')
    arrays={name:getattr(partition,name)[indices].copy() for name in ('positions','energy_residual','forces','atomic_numbers')}
    for a in arrays.values(): a.setflags(write=False)
    return PreparedPartition(**arrays,record_ids=tuple(ids),memberships={r:partition.memberships[r] for r in ids})


def head_coefficients(params):
    head=params['energy_head']['0+']
    return np.r_[np.asarray(head['kernel']).reshape(-1),np.asarray(head['bias']).reshape(-1)]


def with_head(params,coefficients):
    out=deepcopy(params);c=jnp.asarray(coefficients,dtype=jnp.float64)
    out['energy_head']['0+']['kernel']=c[:-1,None]
    out['energy_head']['0+']['bias']=c[-1:]
    return out


def readout_design(params,partition,config):
    n=partition.positions.shape[1];dst,src=np.where(~np.eye(n,dtype=bool))
    body=Backbone(config)
    def features(positions,z):
        x=body.apply({'params':params['Backbone_0']},z,positions,jnp.asarray(dst),jnp.asarray(src))
        return x[:,0,0,:].sum(axis=0)
    x=jax.jit(jax.vmap(features))(jnp.asarray(partition.positions),jnp.asarray(partition.atomic_numbers))
    jac=jax.jit(jax.vmap(jax.jacrev(features,argnums=0)))(jnp.asarray(partition.positions),jnp.asarray(partition.atomic_numbers))
    x,jac=jax.device_get((x,jac))
    g=-jac.transpose(0,2,3,1)
    return np.c_[x,np.full(len(x),n,dtype=np.float64)],np.concatenate([g,np.zeros((*g.shape[:-1],1))],axis=-1)


def verify_design(x,g,params,partition,model):
    b=make_batch(partition.positions,partition.atomic_numbers)
    e,f=energy_and_forces(params,b,model=model);e,f=jax.device_get((e,f))
    c=head_coefficients(params);ep=x@c;fp=g@c
    assert_finite((ep,fp,e,f))
    if not (np.allclose(ep,e,atol=1e-10,rtol=1e-10) and np.allclose(fp.reshape(-1,3),f,atol=1e-10,rtol=1e-10)):
        raise ValueError('Readout reconstruction mismatch')
    return {'maximum_absolute_error':max(float(np.max(np.abs(ep-e))),float(np.max(np.abs(fp.reshape(-1,3)-f)))),'atol':1e-10,'rtol':1e-10}


def mse(energy,forces,target_energy,target_forces):
    return {'energy_mse_eV2':float(np.mean((energy-target_energy)**2)),
            'force_mse_eV2_per_A2':float(np.mean((forces-target_forces)**2))}


def solve_readout(x,g,energy,forces,normalization,*,mode,rcond):
    g=g.reshape(-1,x.shape[1]);forces=forces.reshape(-1)
    ef=normalization.energy_loss_scale_eV*np.sqrt(energy.size)
    ff=normalization.force_loss_scale_eV_per_A*np.sqrt(forces.size)
    if mode=='joint': a=np.vstack([x/ef,g/ff]);y=np.r_[energy/ef,forces/ff]
    elif mode=='force_only': a=g[:,:-1]/ff;y=forces/ff
    else: raise ValueError('Unknown readout solve')
    coefficients,_,rank,s=np.linalg.lstsq(a,y,rcond=rcond)
    if mode=='force_only': coefficients=np.r_[coefficients,np.mean(energy-x[:,:-1]@coefficients)/x[0,-1]]
    assert_finite(coefficients)
    metrics=mse(x@coefficients,g@coefficients,energy,forces)
    return {'mode':mode,'coefficients':coefficients.tolist(),'rank':int(rank),'singular_values':s.tolist(),
            'untruncated_condition_number':float(s[0]/s[-1]) if s[-1]>0 else None,
            'coefficient_l2':float(np.linalg.norm(coefficients)),'rcond':rcond,**metrics,
            'scaled_loss':metrics['energy_mse_eV2']/normalization.energy_loss_scale_eV**2+metrics['force_mse_eV2_per_A2']/normalization.force_loss_scale_eV_per_A**2}
