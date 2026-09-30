"""Single-device seeded baseline with a full-validation selection contract."""
from dataclasses import asdict,dataclass,field
from pathlib import Path
import json
import subprocess
import sys
import time
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import serialization

from .data.prepare import load_training_bundle,write_json
from .data.splits import canonical_hash
from .metrics import partition_batch,validate
from .models.backbone import ModelConfig
from .models.energy import EnergyModel,energy_and_forces
from .runs import assert_finite,code_provenance,load_checkpoint,load_run_control,require_backend,save_checkpoint


@dataclass(frozen=True)
class RunConfig:
    data_directory: str
    expected_split_hash: str
    run_directory: str
    mode: str='pilot'
    backend: str='cpu'
    seed: int=0
    batch_size: int=4
    learning_rate: float=.001
    validation_interval: int=100
    max_updates: int=1000
    wall_seconds: float=3600
    model: dict=field(default_factory=lambda:asdict(ModelConfig()))
    resume_checkpoint: str|None=None
    expected_learning_subset_hash: str|None=None


def new_sampler_state(seed): return np.random.PCG64(seed).state


def verify_learning_subset(ids,expected_hash):
    identity=canonical_hash(ids)
    if expected_hash is not None and identity!=expected_hash: raise ValueError('Learning subset identity mismatch')
    return identity


def validate_run_configuration(config):
    if config.mode not in ('pilot','learning_check','integration') or min(config.max_updates,config.batch_size,config.validation_interval,config.wall_seconds,config.learning_rate)<=0: raise ValueError('Invalid run configuration')
    if config.mode=='learning_check' and config.resume_checkpoint: raise ValueError('Learning continuation requires immutable original acceptance baseline; resume disabled')


def sample_training_batch(bundle,rng_state,batch_size):
    rng=np.random.Generator(np.random.PCG64());rng.bit_generator.state=rng_state
    groups=bundle.manifest['splits']['groups'][bundle.manifest['molecule']]['train']
    pools={g:[i for i,rid in enumerate(bundle.train.record_ids) if g in bundle.train.memberships[rid]['groups']] for g in groups}
    if not groups or any(not pool for pool in pools.values()): raise ValueError('Missing training-group coverage')
    indices=[int(rng.choice(pools[groups[int(rng.integers(len(groups)))]])) for _ in range(batch_size)]
    batch=partition_batch(bundle.train,indices);batch['record_indices']=jnp.array(indices)
    return batch,rng.bit_generator.state


def initialize_state(model,batch,normalization,seed,learning_rate):
    key=jax.random.PRNGKey(seed);key,initkey=jax.random.split(key)
    params=model.init(initkey,batch['atomic_numbers'],batch['positions'],batch['dst_idx'],batch['src_idx'],batch['batch_segments'],batch['energy'].shape[0])['params']
    return {'params':params,'optimizer':optax.adam(learning_rate).init(params),'update':0,'sampler_rng_json':json.dumps(new_sampler_state(seed),sort_keys=True),
            'model_rng':key,'normalization_json':json.dumps(asdict(normalization),sort_keys=True)}


def normalized_loss(params,batch,model,normalization):
    e,f=energy_and_forces(params,batch,model=model)
    emse=jnp.mean((e-batch['energy'])**2);fmse=jnp.mean((f-batch['forces'])**2)
    return emse/normalization.energy_loss_scale_eV**2+fmse/normalization.force_loss_scale_eV_per_A**2,{'energy_mse_eV2':emse,'force_mse_eV2_per_A2':fmse}


def build_step(model,normalization,learning_rate):
    optimizer=optax.adam(learning_rate)
    def loss(params,batch):
        return normalized_loss(params,batch,model,normalization)
    @jax.jit
    def step(params,optstate,batch):
        (value,metrics),grads=jax.value_and_grad(loss,has_aux=True)(params,batch)
        updates,optstate=optimizer.update(grads,optstate,params)
        return optax.apply_updates(params,updates),optstate,{**metrics,'scaled_loss':value}
    return step


def subset_mse(params,partition,indices,predict,batch_size):
    es=fs=0.;ec=fc=0
    for start in range(0,len(indices),batch_size):
        b=partition_batch(partition,indices[start:start+batch_size]);e,f=predict(params,b);e,f=jax.device_get((e,f));assert_finite((e,f))
        es+=float(np.sum((e-np.asarray(b['energy']))**2));fs+=float(np.sum((f-np.asarray(b['forces']))**2));ec+=np.asarray(e).size;fc+=np.asarray(f).size
    return {'energy_mse_eV2':es/ec,'force_mse_eV2_per_A2':fs/fc}


def learning_indices(bundle):
    groups=bundle.manifest['splits']['groups'][bundle.manifest['molecule']]['train'];group=min(groups)
    eligible=[i for i,rid in enumerate(bundle.train.record_ids) if bundle.train.memberships[rid]['role']=='path' and group in bundle.train.memberships[rid]['groups']]
    if len(eligible)<16: raise ValueError('Learning check requires 16 eligible path frames')
    # Numeric frame order within each plus/minus branch, not lexicographic filenames.
    eligible.sort(key=lambda i:(bundle.train.record_ids[i].split('/')[-1].split('_')[0],int(bundle.train.record_ids[i].split('_')[-1].split('.')[0])))
    return np.array(eligible,dtype=int)[np.linspace(0,len(eligible)-1,16,dtype=int)].tolist()


def checkpoint_probe(run_directory):
    root=Path(run_directory);config=RunConfig(**json.loads((root/'config.json').read_text()));bundle=load_training_bundle(config.data_directory,expected_split_hash=config.expected_split_hash)
    device=require_backend(config.backend)
    with jax.default_device(device):
        model=EnergyModel(ModelConfig(**config.model));b=partition_batch(bundle.train,np.arange(min(config.batch_size,len(bundle.train.record_ids))))
        template=initialize_state(model,b,bundle.normalization,config.seed,config.learning_rate)
        state=load_checkpoint(root/'last.msgpack',json.loads((root/'provenance.json').read_text()),template=template)
        e,f=energy_and_forces(state['params'],b,model=model)
        with np.load(root/'checkpoint-reference.npz') as expected:
            error=max(float(np.max(np.abs(e-expected['energy']))),float(np.max(np.abs(f-expected['forces']))))
        if error>1e-10: raise ValueError('New-process checkpoint prediction mismatch')
        write_json(root/'checkpoint-verification.json',{'status':'passed','maximum_prediction_difference':error,'tolerance':1e-10,'update':int(state['update']),'backend':config.backend})


def train(config):
    if isinstance(config,dict): config=RunConfig(**config)
    start=time.perf_counter();root=Path(config.run_directory)
    if root.exists() and any(root.iterdir()): raise ValueError('Run directory is not empty')
    root.mkdir(parents=True,exist_ok=True);write_json(root/'config.json',asdict(config))
    summary={'status':'failed','mode':config.mode,'updates':0,'backend_requested':config.backend}
    try:
        validate_run_configuration(config)
        device=require_backend(config.backend);bundle=load_training_bundle(config.data_directory,expected_split_hash=config.expected_split_hash)
        with jax.default_device(device):
            model=EnergyModel(ModelConfig(**config.model));b,_=sample_training_batch(bundle,new_sampler_state(config.seed),config.batch_size)
            state=initialize_state(model,b,bundle.normalization,config.seed,config.learning_rate)
            provenance={**code_provenance(Path(__file__).resolve().parents[2]),'dataset_identity':bundle.manifest['identity'],'split_hash':bundle.manifest['split_hash'],'policy_hash':bundle.manifest['policy_hash'],'unit_evidence_hash':bundle.manifest['unit_evidence_hash'],
                        'config_computation_hash':canonical_hash({k:v for k,v in asdict(config).items() if k not in ('data_directory','run_directory','resume_checkpoint','max_updates','wall_seconds')}),
                        'backend':device.platform,'device_kind':device.device_kind,'precision':'float64','parameter_count':sum(v.size for v in jax.tree.leaves(state['params'])),'normalization':asdict(bundle.normalization)}
            write_json(root/'provenance.json',provenance)
            restored_control=None
            if config.resume_checkpoint:
                state=load_checkpoint(config.resume_checkpoint,provenance,template=state)
                restored_control=load_run_control(config.resume_checkpoint,provenance)
            initialization_seconds=time.perf_counter()-start
            step=build_step(model,bundle.normalization,config.learning_rate);predict=jax.jit(lambda p,b:energy_and_forces(p,b,model=model))
            t=time.perf_counter();compiled_step=step.lower(state['params'],state['optimizer'],b).compile();compile_seconds=time.perf_counter()-t
            warm=time.perf_counter();jax.block_until_ready(compiled_step(state['params'],state['optimizer'],b));warmup_seconds=time.perf_counter()-warm
            shapes={config.batch_size}
            for partition in (bundle.validation,bundle.validation_minima_diagnostics):
                if len(partition.record_ids)%config.batch_size: shapes.add(len(partition.record_ids)%config.batch_size)
            for size in sorted(shapes):
                probe=partition_batch(bundle.train,np.arange(size));t=time.perf_counter();predict.lower(state['params'],probe).compile();compile_seconds+=time.perf_counter()-t
            subset=learning_indices(bundle) if config.mode=='learning_check' else None
            if subset:
                ids=[bundle.train.record_ids[i] for i in subset]
                subset_hash=verify_learning_subset(ids,config.expected_learning_subset_hash)
                write_json(root/'learning-subset.json',{'indices':subset,'record_ids':ids,'identity':subset_hash})
            initial_mse=subset_mse(state['params'],bundle.train,subset,predict,config.batch_size) if subset else None
            baseline={'zero_force_mse_eV2_per_A2':float(np.mean(bundle.train.forces[subset]**2)),
                      'zero_energy_residual_mse_eV2':float(np.mean(bundle.train.energy_residual[subset]**2)),
                      'subset_mean_energy_mse_eV2':float(np.var(bundle.train.energy_residual[subset]))} if subset else None
            learning_history=[{'update':int(state['update']),**initial_mse}] if subset else []
            best=restored_control['best_score'] if restored_control else float('inf')
            best_update=restored_control['best_update'] if restored_control else None
            best_payload=restored_control['best_state'] if restored_control else None
            history=restored_control['validation_history'] if restored_control else []
            losses=[];times=[];validation_times=[]
            def control():
                return {'best_score':best,'best_update':best_update,'best_state':best_payload,'validation_history':history}
            if restored_control:
                selected=serialization.from_bytes(state,best_payload)
                # A selected checkpoint cannot inherit history from its future.
                selected_control={**control(),'validation_history':[h for h in history if h['update']<=best_update]}
                save_checkpoint(root/'best.msgpack',selected,provenance,run_control=selected_control)
            def evaluate():
                nonlocal best,best_update,best_payload
                t=time.perf_counter();m=validate(state['params'],bundle,model,config.batch_size,predict=predict);validation_times.append(time.perf_counter()-t)
                if m['record_count']==0: raise ValueError('Empty primary validation')
                m['update']=int(state['update'])
                if history and history[-1]['update']==state['update']: history[-1]=m
                else: history.append(m)
                write_json(root/'validation.json',history)
                score=m['equal_connection_force_mae_eV_per_A']
                if score<best:
                    best=score;best_update=int(state['update']);best_payload=serialization.to_bytes(state)
                    save_checkpoint(root/'best.msgpack',state,provenance,run_control=control())
            evaluate();stop='max_updates';accepted=False
            while state['update']<config.max_updates:
                # Leave room for exhaustive final validation, reload and file writes.
                reserve=max(validation_times,default=0)+10
                if time.perf_counter()-start+reserve>=config.wall_seconds: stop='wall_budget';break
                rng=np.random.Generator(np.random.PCG64());rng.bit_generator.state=json.loads(state['sampler_rng_json'])
                if subset:
                    indices=rng.choice(subset,size=config.batch_size,replace=True);batch=partition_batch(bundle.train,indices);batch['record_indices']=jnp.asarray(indices);new_rng=rng.bit_generator.state
                else: batch,new_rng=sample_training_batch(bundle,rng.bit_generator.state,config.batch_size)
                t=time.perf_counter();params,opt,metrics=compiled_step(state['params'],state['optimizer'],batch);jax.block_until_ready((params,opt,metrics));times.append(time.perf_counter()-t)
                assert_finite((params,opt,metrics));state={**state,'params':params,'optimizer':opt,'update':state['update']+1,'sampler_rng_json':json.dumps(new_rng,sort_keys=True)}
                losses.append({'update':int(state['update']),**{k:float(v) for k,v in metrics.items()}})
                if state['update']%config.validation_interval==0:
                    evaluate();write_json(root/'losses.json',losses);save_checkpoint(root/'last.msgpack',state,provenance,run_control=control())
                    print(json.dumps({'update':state['update'],'loss':losses[-1]['scaled_loss'],'validation_force_mae':history[-1]['equal_connection_force_mae_eV_per_A'],'elapsed_seconds':time.perf_counter()-start}),flush=True)
                    if subset:
                        mse=subset_mse(state['params'],bundle.train,subset,predict,config.batch_size)
                        learning_history.append({'update':int(state['update']),**mse});write_json(root/'learning-history.json',learning_history)
                        if all(mse[k]<.1*initial_mse[k] for k in mse): accepted=True;stop='learning_acceptance';break
            if history[-1]['update']!=state['update']: evaluate()
            save_checkpoint(root/'last.msgpack',state,provenance,run_control=control());write_json(root/'losses.json',losses)
            probe=partition_batch(bundle.train,np.arange(min(config.batch_size,len(bundle.train.record_ids))));e,f=predict(state['params'],probe);np.savez(root/'checkpoint-reference.npz',energy=np.array(e),forces=np.array(f))
            subprocess.run([sys.executable,'-m','force_consistency.train','--probe',str(root.resolve())],check=True,capture_output=True)
            mse=subset_mse(state['params'],bundle.train,subset,predict,config.batch_size) if subset else None
            if subset: accepted=all(mse[k]<.1*initial_mse[k] for k in mse)
            if subset and learning_history[-1]['update']!=state['update']: learning_history.append({'update':int(state['update']),**mse})
            if subset: write_json(root/'learning-history.json',learning_history)
            engineering_success=(accepted if subset else state['update']>=100 if config.mode=='pilot' else state['update']>0)
            stats=device.memory_stats()
            summary={**summary,'status':'passed' if engineering_success else 'incomplete','updates':int(state['update']),'stop_reason':stop,'device_kind':device.device_kind,'backend':device.platform,
                     'initialization_seconds':initialization_seconds,'compilation_seconds':compile_seconds,'warmup_execution_seconds':warmup_seconds,'steady_update_seconds_mean':float(np.mean(times)) if times else None,
                     'steady_update_seconds_total':sum(times),'validation_seconds_total':sum(validation_times),'best_update':best_update,'best_equal_connection_force_mae_eV_per_A':best,
                     'initial_validation':history[0],'final_validation':history[-1],'learning_initial_mse':initial_mse,'learning_final_mse':mse,'learning_accepted':accepted if subset else None,
                     'learning_baselines':baseline,'learning_history':learning_history,
                     'checkpoint_new_process_verified':True,'precision':'float64','parameter_count':provenance['parameter_count'],'provenance':provenance,
                     'peak_device_memory_bytes':stats.get('peak_bytes_in_use') if stats else None,'memory_collection_method':'JAX device.memory_stats; unavailable when backend returns None',
                     'resume_scope':'Pinned CPU environment tested to 1e-10; GPU/cross-platform bitwise reproducibility unverified'}
    except Exception as exc:
        summary['failure_reason']=f'{type(exc).__name__}: {exc}'
    summary['elapsed_seconds']=time.perf_counter()-start
    summary['gpu_hours']=summary['elapsed_seconds']/3600 if summary.get('backend')=='gpu' else None
    write_json(root/'summary.json',summary)
    return summary


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--probe',required=True);checkpoint_probe(p.parse_args().probe)
