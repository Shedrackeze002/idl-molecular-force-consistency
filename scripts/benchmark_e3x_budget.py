"""Six bounded, isolated shadow-update probes; never a scientific fit."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
import numpy as np
from e3x_phase_common import (ROOT,ARMS,BALANCED,SPLIT,source_pin,assert_imports,dataset_pin,
    require_equal_pin,fixed_config,fresh_directory,read_json,save_json,digest,child_environment)


def cpu_context():
    import jax
    from force_consistency.runs import require_backend
    return jax.default_device(require_backend('cpu'))


def measure_update(operation,synchronize,*,clock=time.perf_counter):
    start=clock();result,kernel=operation();synchronize(result)
    return {'full_seconds':clock()-start,'kernel_seconds':kernel,'value':result}


def ordinary_update(state,bundle,compiled,losses,*,start,wall_seconds,validation_seconds=0.,kernel_history=None):
    import jax
    from force_consistency.train import sample_training_batch
    from force_consistency.runs import assert_finite
    # Same ordinary host work as train(); reserve check precedes sampling.
    if time.perf_counter()-start+validation_seconds+10>=wall_seconds:raise TimeoutError('Ordinary update wall budget')
    rng=np.random.Generator(np.random.PCG64());rng.bit_generator.state=json.loads(state['sampler_rng_json'])
    batch,new_rng=sample_training_batch(bundle,rng.bit_generator.state,4)
    t=time.perf_counter();params,opt,metrics=compiled(state['params'],state['optimizer'],batch)
    jax.block_until_ready((params,opt,metrics));kernel=time.perf_counter()-t
    if kernel_history is not None:kernel_history.append(kernel)
    assert_finite((params,opt,metrics))
    state={**state,'params':params,'optimizer':opt,'update':state['update']+1,'sampler_rng_json':json.dumps(new_rng,sort_keys=True)}
    losses.append({'update':int(state['update']),**{k:float(v) for k,v in metrics.items()}})
    _=state['update']%500  # ordinary non-validation branch in the production loop
    return (state,batch['record_indices']),kernel


def setup(data,width,arm):
    import jax
    from force_consistency.data.prepare import load_training_bundle
    from force_consistency.train import RunConfig,model_and_predictor,initialize_state,sample_training_batch,new_sampler_state
    assert_imports();bundle=load_training_bundle(data,expected_split_hash=SPLIT)
    if len(bundle.train.record_ids)!=4891 or len(bundle.validation.record_ids)!=974:raise ValueError('Wrong full dataset')
    recipe='A' if width==32 else 'C'
    cfg=RunConfig(**fixed_config(recipe,data,'unused',50000,180,arm=arm,strength=1. if arm=='consistency' else 0.))
    cfg=RunConfig(**{**asdict(cfg),'loss_weights':dict(BALANCED)})
    model,predictor=model_and_predictor(cfg);batch,_=sample_training_batch(bundle,new_sampler_state(0),4)
    state=initialize_state(model,batch,bundle.normalization,0,cfg.learning_rate)
    return cfg,bundle,model,predictor,batch,state


def timing_worker(data,output,width,arm):
    with cpu_context():return _timing_worker(data,output,width,arm)

def _timing_worker(data,output,width,arm):
    import jax
    from force_consistency.train import build_step,initialization_identity,initialize_state
    from force_consistency.metrics import partition_batch,validate
    started=time.perf_counter();out=fresh_directory(output)
    cfg,bundle,model,predictor,batch,initial=setup(data,width,arm)
    initial_id=initialization_identity(initial);seed1=initialization_identity(initialize_state(model,batch,bundle.normalization,1,cfg.learning_rate))
    # Scalar objective/LR values do not enter model init; verify alternate LR explicitly.
    alternate=initialization_identity(initialize_state(model,batch,bundle.normalization,0,.123))
    if alternate!=initial_id:raise ValueError('Learning rate affected initial state')
    init_seconds=time.perf_counter()-started
    step=build_step(model,bundle.normalization,cfg.learning_rate,arm=arm,consistency_strength=cfg.consistency_strength,loss_weights=cfg.loss_weights)
    t=time.perf_counter();compiled=step.lower(initial['params'],initial['optimizer'],batch).compile()
    predict=jax.jit(lambda p,b:predictor(p,b,model=model))
    sizes={4}
    for part in (bundle.validation,bundle.validation_minima_diagnostics):
        if len(part.record_ids)%4:sizes.add(len(part.record_ids)%4)
    for n in sorted(sizes):predict.lower(initial['params'],partition_batch(bundle.train,np.arange(n))).compile()
    compile_seconds=time.perf_counter()-t
    vals=[]
    for _ in range(3):
        t=time.perf_counter();v=validate(initial['params'],bundle,model,4,predict=predict);vals.append(time.perf_counter()-t)
        if v['record_count']!=974:raise ValueError('Incomplete primary validation')
    state=initial;losses=[];kernel_history=[];warm=time.perf_counter()
    def update():return ordinary_update(state,bundle,compiled,losses,start=started,wall_seconds=180,validation_seconds=max(vals),kernel_history=kernel_history)
    for _ in range(3):
        (state,_),_=update()
    warm_seconds=time.perf_counter()-warm
    durations=[];kernels=[];sampled=[]
    for _ in range(100):
        result=measure_update(update,lambda x:None)
        state,indices=result['value'];sampled.append([bundle.train.record_ids[int(i)] for i in np.asarray(indices)])
        durations.append(result['full_seconds']);kernels.append(result['kernel_seconds'])
    stats=jax.devices('cpu')[0].memory_stats()
    record={'id':f'{width}-{arm}','width':width,'arm':arm,'status':'complete','full_update_seconds':durations,
            'kernel_update_seconds':kernels,'validation_seconds':vals,'initialization_seconds':init_seconds,
            'compilation_seconds':compile_seconds,'warmup_seconds':warm_seconds,'initial_identity':initial_id,'seed1_identity':seed1,
            'config':asdict(cfg),'sampled_ids':sampled,'discarded_exposures':{'warmup':12,'timed':400},
            'validation_count':974,'remainder_shapes':sorted(sizes),'source_pin':source_pin(),'actual_backend':'cpu','actual_device_kind':jax.devices('cpu')[0].device_kind,
            'full_update_block_means':np.asarray(durations).reshape(5,20).mean(axis=1).tolist(),'kernel_block_means':np.asarray(kernels).reshape(5,20).mean(axis=1).tolist(),'load_average':os.getloadavg(),
            'peak_memory_bytes':stats.get('peak_bytes_in_use') if stats else None,'memory_method':'JAX CPU device.memory_stats; None means unavailable',
            'worker_elapsed_seconds':time.perf_counter()-started}
    save_json(out/'result.json',record)


def calibrate_checkpoint(path,state,provenance,control):
    """Instrument the unchanged production checkpoint API, including all fixed work."""
    from force_consistency import runs
    actual_atomic=runs.atomic_write;timing={};started=time.perf_counter()
    def observed_atomic(destination,data):
        timing['serialization_seconds']=time.perf_counter()-started;timing['bytes']=len(data)
        t=time.perf_counter();actual_atomic(destination,data);timing['write_seconds']=time.perf_counter()-t
    runs.atomic_write=observed_atomic
    try:runs.save_checkpoint(path,state,provenance,run_control=control)
    finally:runs.atomic_write=actual_atomic
    return timing


def io_worker(data,output):
    with cpu_context():return _io_worker(data,output)

def _io_worker(data,output):
    import jax
    from flax import serialization
    from force_consistency.metrics import validate
    from force_consistency.runs import save_checkpoint
    start=time.perf_counter();out=fresh_directory(output)
    cfg,bundle,model,predictor,batch,state=setup(data,64,'consistency')
    predict=jax.jit(lambda p,b:predictor(p,b,model=model))
    base=validate(state['params'],bundle,model,4,predict=predict)
    def worst(value):
        if isinstance(value,float):return -1.2345678901234567e+123
        if isinstance(value,list):return [worst(x) for x in value]
        if isinstance(value,dict):return {k:worst(v) for k,v in value.items()}
        return value
    history=[{**worst(base),'update':i*500} for i in range(401)]
    keys=('energy_mse_eV2','force_mse_eV2_per_A2','normalized_energy_mse','normalized_force_mse','weighted_energy_loss','weighted_force_loss','consistency_mse_eV2_per_A2','normalized_consistency_mse','weighted_consistency_loss','scaled_loss')
    row={'update':200000,**{k:-1.2345678901234567e+123 for k in keys}}
    losses=[row]*200000;best_payload=serialization.to_bytes(state)
    control={'best_score':.01,'best_update':200000,'best_state':best_payload,'validation_history':history}
    provenance=source_pin()
    samples={};durations={}
    for name,operation in [('validation',lambda:(json.dumps(history,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()),('losses',lambda:(json.dumps(losses,sort_keys=True,indent=2,allow_nan=False)+'\n').encode())]:
        t=time.perf_counter();b=operation();ser=time.perf_counter()-t
        t=time.perf_counter()
        with (out/(name+'.fixture')).open('xb') as stream:stream.write(b);stream.flush();os.fsync(stream.fileno())
        write=time.perf_counter()-t
        samples[name]=len(b);durations[name]={'serialization_seconds':ser,'write_seconds':write}
    full=calibrate_checkpoint(out/'checkpoint.fixture',state,provenance,control)
    empty=calibrate_checkpoint(out/'checkpoint-empty.fixture',state,provenance,{**control,'validation_history':[]})
    one=calibrate_checkpoint(out/'checkpoint-one.fixture',state,provenance,{**control,'validation_history':history[:1]})
    samples['checkpoint']=full['bytes'];durations['checkpoint']={k:full[k] for k in ('serialization_seconds','write_seconds')}
    empty_cp=empty['bytes'];one_cp=one['bytes'];event_cp=max(one_cp-empty_cp,(samples['checkpoint']-empty_cp)/401)

    one_val=len((json.dumps(history[:1],sort_keys=True,indent=2)+'\n').encode())
    empty_val=len((json.dumps([],sort_keys=True,indent=2)+'\n').encode())
    empty_loss=empty_val;one_loss=len((json.dumps([row],sort_keys=True,indent=2)+'\n').encode())
    part=bundle.validation
    # Raw NPZ prediction/reference arrays and Unicode IDs/groups, two summary events,
    # config/provenance/references, plus 2x compression/header/temp margin and64KiB.
    array_bytes=2*(part.forces.nbytes+part.energy_residual.nbytes)
    id_bytes=4*len(part.record_ids)*(max(map(len,part.record_ids))+max(len(g) for g in bundle.manifest['splits']['groups']['aspirin']['validation']))
    fixed_components={'prediction_reference_arrays':array_bytes,'unicode_ids_groups':id_bytes,
                      'summary_and_provenance':2*one_val+3*len(json.dumps(provenance).encode()),
                      'config_and_reload_references':65536}
    fixed_bound=2*sum(fixed_components.values())+65536
    invoke([sys.executable,__file__,'--worker','fixture-reload','--data',str(data),'--output',str(out)],out/'reload.txt',max(1,180-(time.perf_counter()-start)),isolate=False)
    reload_seconds=read_json(out/'reload-result.json')['elapsed_seconds']
    model_record={'validation_base_bytes':empty_val,'validation_event_bytes':max(one_val-empty_val,(samples['validation']-empty_val)/401),
        'loss_base_bytes':empty_loss,'loss_row_bytes':max(one_loss-empty_loss,(samples['losses']-empty_loss)/200000),
        'checkpoint_base_bytes':empty_cp,'checkpoint_event_bytes':event_cp,'fixed_output_bytes':fixed_bound,'fixed_output_bound_components':fixed_components,
        'seconds_per_byte':max(sum(durations[n].values())/samples[n] for n in samples),
        'fixed_event_seconds':max(empty['serialization_seconds']+empty['write_seconds'],one['serialization_seconds']+one['write_seconds']),
        'production_checkpoint_fixed_calibrations':{'empty':empty,'one':one},
        'reload_and_score_seconds':max(60.,3*reload_seconds),'measured_fixture_reload_validation_seconds':reload_seconds,'reload_forecast_rule':'max(60 seconds, three cold fixture reload/full-validation times); representative dual-head calibration, not measured per-arm finalization',
        'scope':'synthetic width64 dual-head/Adam, 401 validation events/200000 longest-schema loss rows; fsync; forecast only',
        'fixture_sizes':samples,'fixture_durations':durations,'source_pin':source_pin()}
    save_json(out/'result.json',model_record)


def fixture_reload(data,output):
    with cpu_context():return _fixture_reload(data,output)

def _fixture_reload(data,output):
    """Read the one synthetic fixture in a new process; no fitting or model selection."""
    import jax
    from force_consistency.runs import load_checkpoint
    from force_consistency.metrics import validate
    started=time.perf_counter();out=Path(output)
    cfg,bundle,model,predictor,batch,template=setup(data,64,'consistency')
    state=load_checkpoint(out/'checkpoint.fixture',source_pin(),template=template)
    predict=jax.jit(lambda p,b:predictor(p,b,model=model))
    v=validate(state['params'],bundle,model,4,predict=predict)
    if v['record_count']!=974:raise ValueError('Incomplete fixture reload validation')
    save_json(out/'reload-result.json',{'elapsed_seconds':time.perf_counter()-started,'synthetic_fixture_only':True,'record_count':974})


def invoke(command,log,timeout,*,isolate=True):
    if timeout<=0:raise TimeoutError('No process deadline remains')
    started=time.perf_counter();deadline=started+timeout
    with Path(log).open('xb') as stream:
        proc=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=isolate,cwd=ROOT,env=child_environment())
        def send(sig):
            try:
                if isolate:os.killpg(proc.pid,sig)
                else:proc.send_signal(sig)
            except ProcessLookupError:pass
        try:
            code=proc.wait(timeout=max(.001,timeout-min(2.,timeout*.25)))
            if code:raise RuntimeError(f'Worker exit {code}; see {log}')
        finally:
            if proc.poll() is None:
                send(signal.SIGTERM)
                try:proc.wait(timeout=max(.001,min(.5,(deadline-time.perf_counter())/2)))
                except subprocess.TimeoutExpired:
                    send(signal.SIGKILL);proc.wait(timeout=max(.001,deadline-time.perf_counter()))
                save_json(str(log)+'.termination.json',{'term_then_kill_if_needed':True,'reaped_returncode':proc.poll(),'elapsed_seconds':time.perf_counter()-started,'budget_seconds':timeout})


def run_probes(data,output,approval):
    gate=read_json(approval)
    if gate['status']!='accepted_for_probes':raise ValueError('Combined implementation review required')
    assert_imports();require_equal_pin(source_pin(),gate['source_pin']);require_equal_pin(dataset_pin(data),gate['dataset_pin'])
    root=fresh_directory(output);save_json(root/'APPROVAL.json',gate);started=time.perf_counter();result={'status':'incomplete','cells':[]}
    try:
        for width in (32,64):
            for arm in ARMS:
                remaining=1800-(time.perf_counter()-started)
                if remaining<=0:raise TimeoutError('Total probe budget')
                dest=root/f'{width}-{arm}'
                invoke([sys.executable,__file__,'--worker',f'{width}-{arm}','--data',str(data),'--output',str(dest)],root/f'{width}-{arm}.txt',min(180,remaining))
                cell=read_json(dest/'result.json');require_equal_pin(cell['source_pin'],gate['source_pin']);result['cells'].append(cell)
        for width in (32,64):
            pair=[c for c in result['cells'] if c['width']==width and c['arm'] in ('direct','consistency')]
            for key in ('initial_identity','seed1_identity'):
                if pair[0][key]!=pair[1][key]:raise ValueError('Unpaired direct/consistency initialization')
        invoke([sys.executable,__file__,'--worker','io','--data',str(data),'--output',str(root/'io')],root/'io.txt',min(180,1800-(time.perf_counter()-started)))
        result['io_model']=read_json(root/'io/result.json');result['status']='complete'
        require_equal_pin(source_pin(),gate['source_pin']);require_equal_pin(dataset_pin(data),gate['dataset_pin'])
    except Exception as exc:result['failure_reason']=f'{type(exc).__name__}: {exc}'
    result['elapsed_seconds']=time.perf_counter()-started;result['source_pin']=gate['source_pin'];result['dataset_pin']=gate['dataset_pin']
    save_json(root/'result.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);p.add_argument('--worker');p.add_argument('--approval');a=p.parse_args()
    if a.worker=='io':io_worker(a.data,a.output)
    elif a.worker=='fixture-reload':fixture_reload(a.data,a.output)
    elif a.worker:
        width,arm=a.worker.split('-');timing_worker(a.data,a.output,int(width),arm)
    else:
        if not a.approval:raise ValueError('Review approval required')
        x=run_probes(a.data,a.output,a.approval)
        if x['status']!='complete':raise SystemExit(x['failure_reason'])

if __name__=='__main__':main()
