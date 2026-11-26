"""Exhaustive primary validation; reference minima never select checkpoints."""
import jax
import numpy as np
from .models.energy import make_batch,energy_and_forces
from .runs import assert_finite


def partition_batch(partition,indices):
    indices=np.asarray(indices,dtype=int)
    return make_batch(partition.positions[indices],partition.atomic_numbers[indices],partition.energy_residual[indices],partition.forces[indices])


def evaluate_partition(params,partition,predict,batch_size,*,primary):
    ids=[];per_group={};force_sum=0.;energy_sum=0.;force_count=0;details=[]
    for start in range(0,len(partition.record_ids),batch_size):
        ix=np.arange(start,min(start+batch_size,len(partition.record_ids)))
        b=partition_batch(partition,ix);e,f=predict(params,b)
        e,f=jax.device_get((e,f));assert_finite((e,f))
        n=partition.positions.shape[1];f=np.asarray(f).reshape(-1,n,3)
        for k,i in enumerate(ix):
            rid=partition.record_ids[i];ef=float(abs(e[k]-partition.energy_residual[i]));ff=float(np.abs(f[k]-partition.forces[i]).sum());count=int(f[k].size)
            ids.append(rid);force_sum+=ff;energy_sum+=ef;force_count+=count
            if primary:
                mem=partition.memberships[rid]
                if mem['role']=='minimum' or len(mem['groups'])!=1: raise ValueError('Invalid primary validation membership')
                group=mem['groups'][0];g=per_group.setdefault(group,{'force_absolute_sum':0.,'force_component_count':0,'energy_absolute_sum':0.,'record_count':0})
                g['force_absolute_sum']+=ff;g['force_component_count']+=count;g['energy_absolute_sum']+=ef;g['record_count']+=1
            else: details.append({'record_id':rid,'energy_mae_eV':ef,'force_mae_eV_per_A':ff/count})
    if len(ids)!=len(set(ids)): raise ValueError('Duplicate validation records')
    for g in per_group.values(): g['force_mae_eV_per_A']=g['force_absolute_sum']/g['force_component_count']
    return {'evaluated_ids':ids,'record_count':len(ids),'force_absolute_sum':force_sum,'force_component_count':force_count,
            'pooled_force_mae_eV_per_A':force_sum/force_count if force_count else None,'energy_mae_eV':energy_sum/len(ids) if ids else None,
            'equal_connection_force_mae_eV_per_A':float(np.mean([g['force_mae_eV_per_A'] for g in per_group.values()])) if per_group else None,
            'per_group':dict(sorted(per_group.items())),'record_errors':details}


def validate(params,bundle,model,batch_size,*,predict=None):
    if batch_size<1: raise ValueError('Invalid validation batch size')
    if predict is None: predict=jax.jit(lambda params,b:energy_and_forces(params,b,model=model))
    m=evaluate_partition(params,bundle.validation,predict,batch_size,primary=True)
    m['diagnostics']=evaluate_partition(params,bundle.validation_minima_diagnostics,predict,batch_size,primary=False)
    return m
