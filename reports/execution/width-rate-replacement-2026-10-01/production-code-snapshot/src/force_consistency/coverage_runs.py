"""Full-subset gradients and exhaustive, training-only coverage diagnostics."""
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from flax import serialization
import jax
import jax.numpy as jnp
import numpy as np
import optax

from .coverage import take_subset
from .data.prepare import Normalization,PreparedPartition,array_identity,write_json
from .data.splits import canonical_hash
from .metrics import partition_batch
from .models.backbone import ModelConfig
from .models.energy import EnergyModel,energy_and_forces
from .runs import assert_finite,load_checkpoint,require_backend,save_checkpoint
from .train import build_step,initialize_state,normalized_loss

MSE_KEYS=('energy_mse_eV2','force_mse_eV2_per_A2')


@lru_cache(maxsize=8)
def _gradient(model,normalization):
    return jax.jit(jax.value_and_grad(lambda p,b:normalized_loss(p,b,model,normalization),has_aux=True))


@lru_cache(maxsize=8)
def _predict(model):
    return jax.jit(lambda p,b:energy_and_forces(p,b,model=model))


def accumulated_loss_and_grad(params,batches,model,normalization):
    if not batches or any(len(b['energy'])==0 for b in batches):
        raise ValueError('Empty gradient microbatch')
    total=sum(len(b['energy']) for b in batches)
    combined=None
    # All sequential chunks see precisely the same incoming parameter tree.
    for batch in batches:
        result=_gradient(model,normalization)(params,batch)
        jax.block_until_ready(result)
        weight=len(batch['energy'])/total
        weighted=jax.tree.map(lambda x:x*weight,result)
        combined=weighted if combined is None else jax.tree.map(lambda a,b:a+b,combined,weighted)
    return combined


def build_accumulated_step(model,normalization,learning_rate):
    single=build_step(model,normalization,learning_rate)
    optimizer=optax.adam(learning_rate)
    @jax.jit
    def apply(params,optstate,grads):
        updates,newopt=optimizer.update(grads,optstate,params)
        return optax.apply_updates(params,updates),newopt
    compiled_single=None
    def prepare(params,optstate,batches):
        nonlocal compiled_single
        if len(batches)==1:
            compiled_single=single.lower(params,optstate,batches[0]).compile()
        else:
            # Compile each unique chunk shape without changing any parameters.
            sizes=set()
            for batch in batches:
                if len(batch['energy']) not in sizes:
                    _gradient(model,normalization).lower(params,batch).compile()
                    sizes.add(len(batch['energy']))
            apply.lower(params,optstate,jax.tree.map(jnp.zeros_like,params)).compile()
    def step(params,optstate,batches):
        # Preserve the original single-full-batch compilation for the replay.
        if len(batches)==1:
            return (compiled_single or single)(params,optstate,batches[0])
        (value,metrics),grads=accumulated_loss_and_grad(params,batches,model,normalization)
        newparams,newopt=apply(params,optstate,grads)
        return newparams,newopt,{**metrics,'scaled_loss':value}
    step.prepare=prepare
    return step


def _partition_predictions(params,partition,model,microbatch_size):
    energies=[];forces=[]
    for start in range(0,len(partition.record_ids),microbatch_size):
        idx=np.arange(start,min(start+microbatch_size,len(partition.record_ids)))
        e,f=jax.device_get(_predict(model)(params,partition_batch(partition,idx)))
        assert_finite((e,f));energies.append(e);forces.append(f.reshape(len(idx),*partition.forces.shape[1:]))
    if not energies: return np.empty((0,)),np.empty_like(partition.forces)
    return np.concatenate(energies),np.concatenate(forces)


def _score(partition,energy,forces):
    ids=list(partition.record_ids);count=len(ids)
    result={'record_count':count,'energy_count':count,'force_component_count':int(partition.forces.size),'record_ids':ids}
    if count:
        em=float(np.mean((energy-partition.energy_residual)**2));fm=float(np.mean((forces-partition.forces)**2))
        result.update(energy_mse_eV2=em,force_mse_eV2_per_A2=fm,energy_rmse_eV=em**.5,force_rmse_eV_per_A=fm**.5,
                      zero_energy_mse_eV2=float(np.mean(partition.energy_residual**2)),mean_energy_mse_eV2=float(np.var(partition.energy_residual)),
                      zero_force_mse_eV2_per_A2=float(np.mean(partition.forces**2)))
    else:
        result.update({k:None for k in (*MSE_KEYS,'energy_rmse_eV','force_rmse_eV_per_A','zero_energy_mse_eV2','mean_energy_mse_eV2','zero_force_mse_eV2_per_A2')})
    return result


def _group_score(partition,energy,forces):
    result=_score(partition,energy,forces)
    groups=sorted({g for r in partition.record_ids for g in partition.memberships[r]['groups']})
    per={}
    for group in groups:
        indices=[i for i,r in enumerate(partition.record_ids) if group in partition.memberships[r]['groups']]
        per[group]=_score(take_subset(partition,[partition.record_ids[i] for i in indices]),energy[indices],forces[indices])
    result['per_group']=per
    result['equal_group']={k:float(np.mean([m[k] for m in per.values()])) if per else None for k in MSE_KEYS}
    return result


def coverage_metrics(params,fitting,anchor,model,microbatch_size=16):
    if microbatch_size<=0: raise ValueError('Invalid microbatch size')
    ef,ff=_partition_predictions(params,fitting,model,microbatch_size)
    ea,fa=_partition_predictions(params,anchor,model,microbatch_size)
    result={'fitting':_group_score(fitting,ef,ff),'anchor':_group_score(anchor,ea,fa)}
    overlap=set(fitting.record_ids)
    for name,fitted in [('anchor_fitted',True),('anchor_not_fitted',False)]:
        idx=[i for i,r in enumerate(anchor.record_ids) if (r in overlap)==fitted]
        result[name]=_group_score(take_subset(anchor,[anchor.record_ids[i] for i in idx]),ea[idx],fa[idx])
    result['anchor_fitted_flags']={r:r in overlap for r in anchor.record_ids}
    return result


def gate_result(current,initial,fraction):
    ratios={k:current.get(k)/initial[k] if current.get(k) is not None and initial.get(k) is not None and initial[k]>0 else None for k in MSE_KEYS}
    return {'accepted':all(r is not None and np.isfinite(r) and r<fraction for r in ratios.values()),'fraction_remaining':ratios,
            'threshold_mse':{k:initial[k]*fraction if initial.get(k) is not None else None for k in MSE_KEYS},'gate_fraction':fraction}


def _arrays(partition):
    return {k:getattr(partition,k) for k in ('positions','energy_residual','forces','atomic_numbers')}


def _state_hashes(state):
    return {key:hashlib.sha256(serialization.to_bytes(state[key])).hexdigest() for key in ('params','optimizer')}


class WallBudget(Exception):
    pass


def run_coverage_arm(config,fitting,anchor,root,provenance,*,step_factory=None):
    start=time.perf_counter();root=Path(root)
    if root.exists() and any(root.iterdir()): raise ValueError('Run directory is not empty')
    root.mkdir(parents=True,exist_ok=True)
    c=dict(config);history=[];times=[];eval_times=[];state=None;initial=None
    summary={'status':'failed','executed':True,'execution_valid':False,'updates':0,'history':history,
             'checkpoint_new_process_verified':False,'fit_gate':{'accepted':False},'anchor_gate':{'accepted':False}}
    prov={**provenance,'config_hash':canonical_hash(c),'normalization':c['normalization'],
          'inputs':{name:{'arrays':{k:array_identity(v) for k,v in _arrays(p).items()},'record_ids':list(p.record_ids),'memberships':p.memberships}
                    for name,p in [('fitting',fitting),('anchor',anchor)]}}
    write_json(root/'config.json',c);write_json(root/'provenance.json',prov)
    for name,p in [('fitting',fitting),('anchor',anchor)]: np.savez(root/f'{name}-inputs.npz',**_arrays(p))
    def check_budget(reserve=0.):
        if time.perf_counter()-start+reserve>=c['wall_seconds']: raise WallBudget('Per-arm wall budget reached')
    try:
        if c.get('resume_checkpoint') or c.get('early_stopping',False): raise ValueError('No resume or learning early stop permitted')
        if min(c['max_updates'],c['microbatch_size'],c['validation_interval'],c['wall_seconds'],c['learning_rate'])<=0 or not 0<c['gate_fraction']<1:
            raise ValueError('Invalid coverage settings')
        check_budget();device=require_backend(c['backend'])
        if device.platform!='cpu': raise ValueError('Coverage protocol requires CPU')
        with jax.default_device(device):
            model=EnergyModel(ModelConfig(**c['model']));norm=Normalization(**c['normalization'])
            batches=[partition_batch(fitting,np.arange(i,min(i+c['microbatch_size'],len(fitting.record_ids))))
                     for i in range(0,len(fitting.record_ids),c['microbatch_size'])]
            if not batches or not anchor.record_ids: raise ValueError('Empty fitting or anchor subset')
            state=initialize_state(model,batches[0],norm,c['seed'],c['learning_rate']);assert_finite(state)
            summary['initialization_seconds']=time.perf_counter()-start
            summary['initial_state_hashes']=_state_hashes(state)
            if provenance.get('expected_initial_hashes',summary['initial_state_hashes'])!=summary['initial_state_hashes']:
                raise ValueError('Fresh parameter/optimizer initialization differs across arms')
            check_budget()
            def evaluate():
                nonlocal initial
                t=time.perf_counter();metrics=coverage_metrics(state['params'],fitting,anchor,model,c['microbatch_size'])
                eval_times.append(time.perf_counter()-t)
                if initial is None:
                    initial=metrics;summary['initial_metrics']=initial
                    baseline=provenance.get('anchor_initial_mse',initial['anchor'])
                    for k in MSE_KEYS:
                        if not np.isclose(initial['anchor'][k],baseline[k],atol=1e-10,rtol=1e-8): raise ValueError('Immutable anchor baseline mismatch')
                    expected_fit=provenance.get('fitting_initial_mse',initial['fitting'])
                    for k in MSE_KEYS:
                        if not np.isclose(initial['fitting'][k],expected_fit[k],atol=1e-10,rtol=1e-8): raise ValueError('Immutable fitting baseline mismatch')
                    summary['anchor_initial_mse']={k:baseline[k] for k in MSE_KEYS}
                own=gate_result(metrics['fitting'],initial['fitting'],c['gate_fraction'])
                common=gate_result(metrics['anchor'],summary['anchor_initial_mse'],c['gate_fraction'])
                row={'update':int(state['update']),'elapsed_seconds':time.perf_counter()-start,
                     'frame_exposures':int(state['update'])*len(fitting.record_ids),'metrics':metrics,'fit_gate':own,'anchor_gate':common,
                     'fractions_from_own_initial':{name:gate_result(m,initial[name],c['gate_fraction'])['fraction_remaining'] for name,m in metrics.items() if name!='anchor_fitted_flags'}}
                if history and history[-1]['update']==row['update']: history[-1]=row
                else: history.append(row)
                for name,gate in [('fit',own),('anchor',common)]:
                    key=f'first_observed_{name}_passing_scheduled_update'
                    if gate['accepted'] and summary.get(key) is None: summary[key]=int(state['update'])
                summary.update(final_metrics=metrics,fit_gate=own,anchor_gate=common)
                write_json(root/'history.json',history)
                return row
            evaluate();check_budget(10+max(eval_times))
            step=(step_factory or build_accumulated_step)(model,norm,c['learning_rate'])
            t=time.perf_counter()
            if hasattr(step,'prepare'): step.prepare(state['params'],state['optimizer'],batches)
            summary['compilation_seconds']=time.perf_counter()-t
            check_budget(10+max(eval_times))
            t=time.perf_counter();warm=step(state['params'],state['optimizer'],batches)
            jax.block_until_ready(warm);assert_finite(warm)
            summary['warmup_seconds']=time.perf_counter()-t
            summary['warmup_frame_exposures']=len(fitting.record_ids)
            stop='max_updates'
            while state['update']<c['max_updates']:
                check_budget(10+max(eval_times))
                t=time.perf_counter();params,opt,metrics=step(state['params'],state['optimizer'],batches)
                jax.block_until_ready((params,opt,metrics));assert_finite((params,opt,metrics));times.append(time.perf_counter()-t)
                state={**state,'params':params,'optimizer':opt,'update':int(state['update'])+1};summary['updates']=int(state['update'])
                if state['update']%c['validation_interval']==0 or state['update']==250:
                    check_budget();row=evaluate();save_checkpoint(root/'last.msgpack',state,prov)
                    print(json.dumps({'arm':root.name,'update':state['update'],'fitting_force_mse':row['metrics']['fitting'][MSE_KEYS[1]],
                                      'anchor_force_mse':row['metrics']['anchor'][MSE_KEYS[1]],'elapsed_seconds':row['elapsed_seconds']}),flush=True)
            if history[-1]['update']!=state['update']: check_budget();evaluate()
            save_checkpoint(root/'last.msgpack',state,prov);check_budget()
            references={}
            for name,p in [('fitting',fitting),('anchor',anchor)]:
                check_budget();e,f=_partition_predictions(state['params'],p,model,c['microbatch_size'])
                references[f'{name}_energy']=e;references[f'{name}_forces']=f
            np.savez(root/'reference.npz',**references);check_budget()
            subprocess.run([sys.executable,'-m','force_consistency.coverage_runs','--probe',str(root.resolve())],check=True,capture_output=True,
                           timeout=max(.001,c['wall_seconds']-(time.perf_counter()-start)))
            check_budget()
            summary.update(status='completed',execution_valid=True,stop_reason=stop,checkpoint_new_process_verified=True)
    except (WallBudget,subprocess.TimeoutExpired) as exc:
        summary.update(status='incomplete',stop_reason='wall_budget',failure_reason=f'{type(exc).__name__}: {exc}')
    except Exception as exc:
        summary.update(status='failed',stop_reason='failure',failure_reason=f'{type(exc).__name__}: {exc}')
        if isinstance(exc,subprocess.CalledProcessError): summary['probe_stderr']=exc.stderr.decode()[-3000:]
    if state is not None:
        save_checkpoint(root/'last.msgpack',state,prov)
    summary.update(elapsed_seconds=time.perf_counter()-start,steady_update_seconds_total=sum(times),
                   steady_update_seconds_mean=float(np.mean(times)) if times else None,evaluation_seconds_total=sum(eval_times),
                   evaluation_passes=len(eval_times),evaluation_frame_count=len(eval_times)*(len(fitting.record_ids)+len(anchor.record_ids)),
                   frame_exposures=summary['updates']*len(fitting.record_ids),fitting_frame_count=len(fitting.record_ids),
                   microbatches_per_update=(len(fitting.record_ids)+c['microbatch_size']-1)//c['microbatch_size'],provenance=prov)
    write_json(root/'summary.json',summary)
    return summary


def coverage_probe(root):
    """Exhaustive saved-input-only checkpoint check in a fresh process."""
    root=Path(root);c=json.loads((root/'config.json').read_text());prov=json.loads((root/'provenance.json').read_text())
    if canonical_hash(c)!=prov['config_hash']: raise ValueError('Probe configuration identity mismatch')
    partitions={}
    for name in ('fitting','anchor'):
        with np.load(root/f'{name}-inputs.npz',allow_pickle=False) as file: arrays={k:file[k] for k in file.files}
        metadata=prov['inputs'][name]
        if {k:array_identity(v) for k,v in arrays.items()}!=metadata['arrays']: raise ValueError('Probe input identity mismatch')
        partitions[name]=PreparedPartition(**arrays,record_ids=tuple(metadata['record_ids']),memberships=metadata['memberships'])
    device=require_backend(c['backend'])
    with jax.default_device(device):
        model=EnergyModel(ModelConfig(**c['model']));p=partitions['fitting']
        template=initialize_state(model,partition_batch(p,np.arange(min(c['microbatch_size'],len(p.record_ids)))),Normalization(**c['normalization']),c['seed'],c['learning_rate'])
        state=load_checkpoint(root/'last.msgpack',prov,template=template);errors=[]
        with np.load(root/'reference.npz',allow_pickle=False) as reference:
            for name,p in partitions.items():
                e,f=_partition_predictions(state['params'],p,model,c['microbatch_size'])
                for predicted,key in [(e,f'{name}_energy'),(f,f'{name}_forces')]:
                    expected=reference[key];assert_finite(expected)
                    if predicted.shape!=expected.shape: raise ValueError('Probe prediction shape mismatch')
                    errors.append(float(np.max(np.abs(predicted-expected))))
        error=max(errors)
        if not np.isfinite(error) or error>1e-10: raise ValueError('New-process prediction mismatch')
    write_json(root/'checkpoint-verification.json',{'status':'passed','maximum_absolute_error':error,'tolerance':1e-10,
               'update':int(state['update']),'fitting_record_count':len(partitions['fitting'].record_ids),'anchor_record_count':len(partitions['anchor'].record_ids)})


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--probe',required=True)
    coverage_probe(parser.parse_args().probe)
