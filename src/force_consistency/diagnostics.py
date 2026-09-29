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


def input_arrays(partition):
    return {k:getattr(partition,k) for k in ('positions','atomic_numbers','energy_residual','forces')}


def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def append_evidence(config,phase,event,details):
    if config.get('evidence_log'):
        with Path(config['evidence_log']).open('a') as file:
            file.write(f'\n## DIAGNOSTIC-001{phase} — {event} (30 September 2026)\n\n')
            file.write('Training-only engineering diagnosis; full pilot remains pending.\n\n```json\n')
            file.write(json.dumps(details,sort_keys=True,indent=2,allow_nan=False)+'\n```\n')


def build_teacher(params,partition,model,seed,std):
    kernel=np.asarray(jax.random.normal(jax.random.PRNGKey(seed),(model.config.features,),dtype=jnp.float64))*std
    teacher=with_head(params,np.r_[kernel,0.])
    b=make_batch(partition.positions,partition.atomic_numbers)
    e,_=energy_and_forces(teacher,b,model=model)
    teacher=with_head(params,np.r_[kernel,-float(e.mean())/partition.positions.shape[1]])
    e,f=jax.device_get(energy_and_forces(teacher,b,model=model))
    e=np.array(e);f=np.array(f).reshape(partition.forces.shape)
    assert_finite((e,f))
    if np.sqrt(np.mean(f**2))<=1e-12 or np.var(e)<=1e-24: raise ValueError('Degenerate synthetic teacher targets')
    e.setflags(write=False);f.setflags(write=False)
    return replace(partition,energy_residual=e,forces=f),teacher


def control_probe(root):
    """Fresh process, saved control arrays only; no prepared loader."""
    root=Path(root);c=json.loads((root/'config.json').read_text());prov=json.loads((root/'provenance.json').read_text())
    device=require_backend(c['backend'])
    with np.load(root/'inputs.npz',allow_pickle=False) as file: a={k:file[k] for k in file.files}
    if {k:array_identity(v) for k,v in a.items()}!=prov['input_arrays']: raise ValueError('Probe input identity mismatch')
    with jax.default_device(device):
        b=make_batch(a['positions'],a['atomic_numbers'],a['energy_residual'],a['forces'])
        model=EnergyModel(ModelConfig(**c['model']))
        template=initialize_state(model,b,Normalization(**c['normalization']),c['seed'],c['learning_rate'])
        state=load_checkpoint(root/'last.msgpack',prov,template=template)
        e,f=jax.device_get(energy_and_forces(state['params'],b,model=model))
        with np.load(root/'reference.npz',allow_pickle=False) as expected:
            error=max(float(np.max(np.abs(e-expected['energy']))),float(np.max(np.abs(f-expected['forces']))))
        if not np.isfinite(error) or error>1e-10: raise ValueError('New-process prediction mismatch')
    write_json(root/'checkpoint-verification.json',{'status':'passed','maximum_absolute_error':error,'tolerance':1e-10,'update':int(state['update'])})


def run_control(config,partition,model,normalization,state,root,provenance,label_type):
    root=Path(root)
    if root.exists() and any(root.iterdir()): raise ValueError('Run directory is not empty')
    root.mkdir(parents=True,exist_ok=True);start=time.perf_counter();history=[];times=[]
    state=deepcopy(state)
    summary={'status':'failed','accepted':False,'updates':0,'label_type':label_type,'history':history,'batch_size':len(partition.record_ids),'checkpoint_new_process_verified':False}
    c={**config,'model':asdict(model.config),'normalization':asdict(normalization)}
    prov={**provenance,'label_type':label_type,'normalization':asdict(normalization),'input_arrays':{k:array_identity(v) for k,v in input_arrays(partition).items()},'record_ids':list(partition.record_ids),'control_computation_hash':canonical_hash(c)}
    write_json(root/'config.json',c);write_json(root/'provenance.json',prov);np.savez(root/'inputs.npz',**input_arrays(partition))
    try:
        if min(c['max_updates'],c['validation_interval'],c['wall_seconds'],c['learning_rate'])<=0 or not 0<c['gate_fraction']<1: raise ValueError('Invalid control settings')
        b=make_batch(partition.positions,partition.atomic_numbers,partition.energy_residual,partition.forces)
        predict=jax.jit(lambda p:energy_and_forces(p,b,model=model))
        def evaluate():
            e,f=jax.device_get(predict(state['params']));assert_finite((e,f))
            metrics=mse(e,np.asarray(f).reshape(partition.forces.shape),partition.energy_residual,partition.forces)
            row={'update':int(state['update']),**metrics}
            if history and history[-1]['update']==row['update']: history[-1]=row
            else: history.append(row)
            write_json(root/'history.json',history)
            return metrics
        initial=evaluate();summary['initial_mse']=dict(initial)
        summary['baselines']={'zero_force_mse_eV2_per_A2':float(np.mean(partition.forces**2)),
                              'zero_energy_residual_mse_eV2':float(np.mean(partition.energy_residual**2)),
                              'mean_energy_mse_eV2':float(np.var(partition.energy_residual))}
        step=build_step(model,normalization,c['learning_rate']);t=time.perf_counter()
        compiled=step.lower(state['params'],state['optimizer'],b).compile();summary['compilation_seconds']=time.perf_counter()-t
        t=time.perf_counter();warm=compiled(state['params'],state['optimizer'],b);jax.block_until_ready(warm);assert_finite(warm)
        summary['warmup_seconds']=time.perf_counter()-t
        stop='max_updates'
        while state['update']<c['max_updates']:
            if time.perf_counter()-start+10>=c['wall_seconds']: stop='wall_budget';break
            t=time.perf_counter();params,opt,metrics=compiled(state['params'],state['optimizer'],b)
            jax.block_until_ready((params,opt,metrics));assert_finite((params,opt,metrics));times.append(time.perf_counter()-t)
            state={**state,'params':params,'optimizer':opt,'update':int(state['update'])+1}
            summary['updates']=int(state['update'])
            if state['update']%c['validation_interval']==0:
                current=evaluate();save_checkpoint(root/'last.msgpack',state,prov)
                print(json.dumps({'control':label_type,'update':state['update'],**current}),flush=True)
                if all(current[k]<c['gate_fraction']*initial[k] for k in current): stop='learning_acceptance';break
        final=evaluate();save_checkpoint(root/'last.msgpack',state,prov)
        e,f=jax.device_get(predict(state['params']));np.savez(root/'reference.npz',energy=e,forces=f)
        subprocess.run([sys.executable,'-m','force_consistency.diagnostics','--probe',str(root.resolve())],check=True,capture_output=True)
        accepted=all(final[k]<c['gate_fraction']*initial[k] for k in final)
        summary.update(status='passed' if accepted else 'incomplete',accepted=accepted,final_mse=final,
                       fraction_remaining={k:final[k]/initial[k] if initial[k]>0 else None for k in final},stop_reason=stop,
                       checkpoint_new_process_verified=True)
    except Exception as exc:
        summary['failure_reason']=f'{type(exc).__name__}: {exc}'
        summary['stop_reason']='failure'
        if isinstance(exc,subprocess.CalledProcessError): summary['probe_stderr']=exc.stderr.decode()[-2000:]
    summary.update(elapsed_seconds=time.perf_counter()-start,steady_update_seconds_total=sum(times),
                   steady_update_seconds_mean=float(np.mean(times)) if times else None,provenance=prov)
    write_json(root/'summary.json',summary)
    return summary


def run_after_teacher(teacher,run_real,root):
    if teacher.get('accepted') and teacher.get('status')=='passed': return run_real()
    root=Path(root);root.mkdir(parents=True,exist_ok=False)
    result={'status':'blocked','accepted':False,'updates':0,'stop_reason':'teacher_gate_failed','executed':False}
    write_json(root/'summary.json',result);return result


def verify_source_settings(config,source_config,source_provenance,manifest,normalization,state):
    for key in ('seed','learning_rate','expected_split_hash','expected_learning_subset_hash'):
        if source_config.get(key)!=config.get(key): raise ValueError(f'Historical config mismatch: {key}')
    if ModelConfig(**source_config['model'])!=ModelConfig(**config['model']) or source_config.get('mode')!='learning_check' or source_config.get('batch_size')!=4:
        raise ValueError('Historical model or learning mode mismatch')
    if state['update']!=2000: raise ValueError('Historical update mismatch')
    if source_provenance['dataset_identity']!=manifest['identity'] or source_provenance['split_hash']!=manifest['split_hash'] or source_provenance['normalization']!=asdict(normalization):
        raise ValueError('Historical preparation mismatch')
    if json.loads(state['normalization_json'])!=asdict(normalization): raise ValueError('Historical state normalization mismatch')


def diagnose(config):
    root=Path(config['run_directory'])
    if root.exists() and any(root.iterdir()): raise ValueError('Run directory is not empty')
    root.mkdir(parents=True,exist_ok=True);write_json(root/'config.json',config)
    summary={'status':'failed','execution_valid':False,'full_pilot':'pending',
             'teacher':{'status':'blocked','executed':False},'real':{'status':'blocked','executed':False}}
    active_phase='A'
    try:
        source=Path(config['source_run'])
        required={'last.msgpack','learning-subset.json','config.json','provenance.json'}
        if set(config['source_hashes'])!=required: raise ValueError('Incomplete historical source hashes')
        for name,digest in config['source_hashes'].items():
            if sha256(source/name)!=digest: raise ValueError(f'Historical source identity mismatch: {name}')
        device=require_backend(config['backend'])
        if device.platform!='cpu': raise ValueError('Diagnostic protocol requires CPU')
        partition,manifest,norm=load_training_partition(config['data_directory'],expected_split_hash=config['expected_split_hash'])
        if manifest['identity']!=config['expected_prepared_identity']: raise ValueError('Prepared identity mismatch')
        p=verified_subset(partition,manifest,json.loads((source/'learning-subset.json').read_text()),config['expected_learning_subset_hash'])
        if len(p.record_ids)!=16 or config['batch_size']!=16: raise ValueError('Diagnostic requires fixed full batch16')
        with jax.default_device(device):
            model=EnergyModel(ModelConfig(**config['model']));b=make_batch(p.positions,p.atomic_numbers,p.energy_residual,p.forces)
            state=initialize_state(model,b,norm,config['seed'],config['learning_rate'])
            small=make_batch(p.positions[:4],p.atomic_numbers[:4]);four=initialize_state(model,small,norm,config['seed'],config['learning_rate'])
            if any(not np.array_equal(a,c) for a,c in zip(jax.tree.leaves(state['params']),jax.tree.leaves(four['params']))): raise ValueError('Batch shape changes initialization')
            historic_prov=json.loads((source/'provenance.json').read_text())
            historic=load_checkpoint(source/'last.msgpack',historic_prov,template=state)
            verify_source_settings(config,json.loads((source/'config.json').read_text()),historic_prov,manifest,norm,historic)
            prov={**code_provenance(Path(__file__).resolve().parents[2]),'dataset_identity':manifest['identity'],'split_hash':manifest['split_hash'],
                  'subset_hash':config['expected_learning_subset_hash'],'historical_source':{'hashes':config['source_hashes'],'provenance':historic_prov},
                  'backend':device.platform,'device_kind':device.device_kind,'precision':'float64','config_hash':canonical_hash(config)}
            write_json(root/'provenance.json',prov);write_json(root/'subset.json',{'record_ids':list(p.record_ids),'input_arrays':{k:array_identity(v) for k,v in input_arrays(p).items()}})
            append_evidence(config,'A','Frozen readout trial preregistered',{'config_hash':canonical_hash(config),'run_directory':str(root),'rcond':config['lstsq_rcond'],'controls':'fresh and historical backbones, joint and force-only'})
            solves={}
            for name,params in [('fresh',state['params']),('historical_final',historic['params'])]:
                x,g=readout_design(params,p,model.config)
                actual=verify_design(x,g,params,p,model)
                nonzero=np.r_[np.asarray(jax.random.normal(jax.random.PRNGKey(config['teacher_seed']),(model.config.features,),dtype=jnp.float64))*config['teacher_kernel_std'],.07]
                witness=verify_design(x,g,with_head(params,nonzero),p,model)
                solves[name]={'actual_head_reconstruction':actual,'nonzero_head_reconstruction':witness,'solutions':[]}
                for mode in ('force_only','joint'):
                    solution=solve_readout(x,g,p.energy_residual,p.forces,norm,mode=mode,rcond=config['lstsq_rcond'])
                    solution['model_reconstruction']=verify_design(x,g,with_head(params,solution['coefficients']),p,model)
                    solves[name]['solutions'].append(solution)
            write_json(root/'readout.json',solves);summary['readout']=solves
            append_evidence(config,'A','Frozen readout trial completed',{'artifact':str(root/'readout.json'),'sha256':sha256(root/'readout.json'),'solutions':{name:[{k:s[k] for k in ('mode','rank','force_mse_eV2_per_A2','energy_mse_eV2','coefficient_l2')} for s in v['solutions']] for name,v in solves.items()}})
            active_phase='B'
            append_evidence(config,'B','Representable teacher trial preregistered',{'config_hash':canonical_hash(config),'seed':config['teacher_seed'],'kernel_std':config['teacher_kernel_std'],'max_updates':config['max_updates'],'gate_fraction':config['gate_fraction'],'batch_size':16})
            synthetic,teacher=build_teacher(state['params'],p,model,config['teacher_seed'],config['teacher_kernel_std'])
            teacher_norm=replace(norm,energy_offset_eV=0.)
            teacher_state=initialize_state(model,b,teacher_norm,config['seed'],config['learning_rate'])
            save_checkpoint(root/'teacher-parameters.msgpack',{**teacher_state,'params':teacher}, {**prov,'purpose':'untrained synthetic teacher'})
            summary['teacher']=run_control(config,synthetic,model,teacher_norm,teacher_state,root/'teacher',prov,'synthetic_teacher')
            append_evidence(config,'B','Representable teacher trial outcome',{k:summary['teacher'].get(k) for k in ('status','accepted','updates','initial_mse','final_mse','fraction_remaining','stop_reason','failure_reason')})
            active_phase='C'
            if summary['teacher']['status']=='passed':
                append_evidence(config,'C','Real full-batch trial preregistered',{'config_hash':canonical_hash(config),'batch_size':16,'max_updates':config['max_updates'],'gate_fraction':config['gate_fraction'],'subset_hash':config['expected_learning_subset_hash']})
            summary['real']=run_after_teacher(summary['teacher'],lambda:run_control(config,p,model,norm,state,root/'real',prov,'real_DFT'),root/'real')
            append_evidence(config,'C','Real full-batch trial outcome',{k:summary['real'].get(k) for k in ('status','executed','accepted','updates','initial_mse','final_mse','fraction_remaining','stop_reason','failure_reason')})
            summary['execution_valid']=summary['teacher']['status']!='failed' and summary['real']['status']!='failed'
            summary['status']='completed' if summary['execution_valid'] else 'failed'
    except Exception as exc:
        summary['failure_reason']=f'{type(exc).__name__}: {exc}'
        append_evidence(config,active_phase,'Diagnostic failed before completion',{'failure_reason':summary['failure_reason'],'run_directory':str(root)})
    write_json(root/'summary.json',summary)
    return summary


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--probe',required=True)
    control_probe(parser.parse_args().probe)
